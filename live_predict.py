import orjson
import requests
import urllib3
from sseclient import SSEClient
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl
import joblib
import os
import warnings
import time
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
import csv
from collections import deque

# --- Suppress Warnings ---
warnings.filterwarnings('ignore')
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ============================================================================
# 0. CONFIGURATION
# ============================================================================
class Config:
    # --- API and Paths ---
    API_KEY = ""
    STREAM_URL = ""
    FLAG_URL = ""
    MODEL_PATH = r"E:\PythonStuff\checkpoints\NEW-best-fraud-model-epoch=09-val_auc=0.9887.ckpt"
    PREPROCESSOR_DIR = "preprocessors"
    OUTPUT_CSV_FILE = "prediction_log.csv"

    # --- Performance Tuning ---
    NUM_PREDICTION_WORKERS = 12  # Can be lower now due to batching efficiency
    NUM_FLAGGING_WORKERS = 128
    NUM_HISTORY_SHARDS = 64  # Reduces lock contention
    MAX_FLAGS_PER_SECOND = 50

    # --- Batching Logic ---
    BATCH_SIZE = 256  # Number of transactions to process at once
    BATCH_TIMEOUT_SECONDS = 0.5  # Max time to wait before processing a batch

    # --- Model & Business Logic ---
    DECISION_THRESHOLD = 0.95
    HISTORY_MAX_AGE_SECONDS = 20


# ============================================================================
# 1. RateLimiter, MODEL AND PREDICTOR CLASS DEFINITIONS (Largely Unchanged)
# ============================================================================

class RateLimiter:
    """A thread-safe token bucket rate limiter."""

    def __init__(self, max_calls, period=1.0):
        self.max_calls, self.period = max_calls, period
        self.lock = threading.Lock()
        self.tokens = self.max_calls
        self.last_refill_time = time.monotonic()

    def _refill_tokens(self):
        now = time.monotonic()
        elapsed = now - self.last_refill_time
        if elapsed > self.period:
            new_tokens = elapsed * (self.max_calls / self.period)
            self.tokens = min(self.max_calls, self.tokens + new_tokens)
            self.last_refill_time = now

    def acquire(self):
        while True:
            with self.lock:
                self._refill_tokens()
                if self.tokens >= 1:
                    self.tokens -= 1
                    return True
            time.sleep(0.01)

    def __enter__(self):
        self.acquire();
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class FraudTransformer(pl.LightningModule):
    def __init__(self, num_numerical_features, categorical_embedding_dims, model_dim=64, nhead=4, num_layers=2, lr=1e-4,
                 pos_weight=None):
        super().__init__()
        self.save_hyperparameters()
        self.embedding_layers = nn.ModuleList(
            [nn.Embedding(vocab_size, emb_dim) for vocab_size, emb_dim in self.hparams.categorical_embedding_dims])
        total_embedding_dim = sum([emb_dim for _, emb_dim in self.hparams.categorical_embedding_dims])
        total_feature_dim = total_embedding_dim + self.hparams.num_numerical_features
        self.input_projection = nn.Linear(total_feature_dim, self.hparams.model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hparams.model_dim, nhead=self.hparams.nhead,
                                                   batch_first=True, dim_feedforward=self.hparams.model_dim * 4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.hparams.num_layers)
        self.classifier = nn.Linear(self.hparams.model_dim, 2)

    def forward(self, numerical_seq, categorical_seq, src_key_padding_mask):
        embedded_cats = [self.embedding_layers[i](categorical_seq[..., i]) for i in range(len(self.embedding_layers))]
        all_embeddings = torch.cat(embedded_cats, dim=-1)
        x = torch.cat([numerical_seq, all_embeddings], dim=-1)
        x = self.input_projection(x)
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        return self.classifier(x)


class TransactionPredictor:
    def __init__(self, model_path, preprocessor_dir):
        # ... (init is the same)
        self.model_path, self.preprocessor_dir = model_path, preprocessor_dir
        try:
            self.scalers = joblib.load(os.path.join(self.preprocessor_dir, 'scalers.pkl'))
            self.encoders = joblib.load(os.path.join(self.preprocessor_dir, 'encoders.pkl'))
            self.num_cols = joblib.load(os.path.join(self.preprocessor_dir, 'num_cols.pkl'))
            self.cat_cols = joblib.load(os.path.join(self.preprocessor_dir, 'cat_cols.pkl'))
        except FileNotFoundError as e:
            raise FileNotFoundError(f"FATAL: Preprocessor files not found: {e}")
        self.model = FraudTransformer.load_from_checkpoint(self.model_path, map_location=torch.device('cpu'))
        self.model.eval()
        print("Model and preprocessors loaded successfully.")

    def _preprocess_batch(self, batch_df):
        # Preprocessing is now done on a batch DataFrame, which is more efficient
        data = batch_df.copy()
        data['trans_datetime'] = pd.to_datetime(data['trans_date'] + ' ' + data['trans_time'], errors='coerce')
        data['dob'] = pd.to_datetime(data['dob'], errors='coerce')
        data['age'] = (data['trans_datetime'] - data['dob']).dt.days / 365.25
        data['hour'], data['day_of_week'] = data['trans_datetime'].dt.hour, data['trans_datetime'].dt.dayofweek
        data['hour_sin'], data['hour_cos'] = np.sin(2 * np.pi * data['hour'] / 24), np.cos(
            2 * np.pi * data['hour'] / 24)
        data['day_of_week_sin'], data['day_of_week_cos'] = np.sin(2 * np.pi * data['day_of_week'] / 7), np.cos(
            2 * np.pi * data['day_of_week'] / 7)
        data = data.sort_values(by=['cc_num', 'unix_time'])
        data['time_since_last_transaction'] = data.groupby('cc_num')['unix_time'].diff().fillna(0)
        for col in self.num_cols:
            if col in data.columns: data[col] = self.scalers[col].transform(data[[col]])
        for col in self.cat_cols:
            if col in data.columns:
                data[col] = data[col].astype(str)  # Ensure consistent type
                known_classes = list(self.encoders[col].classes_)
                # Use pd.Categorical for efficient encoding
                data[col] = pd.Categorical(data[col], categories=known_classes).codes + 1
        return data

    def predict_batch(self, user_sequences):
        # <--- CHANGE 2: This method now processes a batch of sequences
        if not user_sequences:
            return pd.DataFrame()

        # Create a single large DataFrame from all sequences for batch preprocessing
        all_transactions = [tx for seq in user_sequences.values() for tx in seq]
        df_raw = pd.DataFrame(all_transactions)
        original_data = df_raw[['trans_num', 'cc_num']].copy()

        df_processed = self._preprocess_batch(df_raw)

        # Group processed data to reconstruct sequences
        grouped = df_processed.groupby('cc_num')
        all_num_seqs, all_cat_seqs, seq_indices, cc_order = [], [], [], []

        for cc_num, group in grouped:
            cc_order.append(cc_num)
            all_num_seqs.append(torch.tensor(group[self.num_cols].values, dtype=torch.float32))
            all_cat_seqs.append(torch.tensor(group[self.cat_cols].values, dtype=torch.long))
            seq_indices.extend(group.index.tolist())

        if not all_num_seqs: return pd.DataFrame()

        # Pytorch operations are now performed on the entire batch
        num_data_padded = pad_sequence(all_num_seqs, batch_first=True, padding_value=0.0)
        cat_data_padded = pad_sequence(all_cat_seqs, batch_first=True, padding_value=0)
        seq_lengths = torch.tensor([len(s) for s in all_num_seqs])
        padding_mask = torch.arange(num_data_padded.size(1))[None, :] >= seq_lengths[:, None]

        with torch.no_grad():
            logits = self.model(num_data_padded, cat_data_padded, padding_mask)
            probabilities = torch.softmax(logits, dim=-1)
            fraud_probs = probabilities[:, :, 1]

        results_df = pd.DataFrame({
            'fraud_probability': fraud_probs[
                torch.arange(len(seq_lengths)).unsqueeze(1), torch.arange(num_data_padded.size(1))].flatten()[
                :len(seq_indices)],
            'original_index': seq_indices
        }).set_index('original_index')

        return original_data.join(results_df)


# ============================================================================
# 2. NEW High-Performance Data Handling
# ============================================================================

class ShardedHistoryManager:
    # <--- CHANGE 3: Manages history with sharded locks to prevent contention
    def __init__(self, num_shards, max_age_seconds):
        self.num_shards = num_shards
        self.max_age_seconds = max_age_seconds
        self._shards = [{} for _ in range(num_shards)]
        self._locks = [threading.Lock() for _ in range(num_shards)]

    def _get_shard_index(self, cc_num):
        return hash(cc_num) % self.num_shards

    def add_and_get_history(self, transaction):
        cc_num = transaction['cc_num']
        shard_index = self._get_shard_index(cc_num)
        with self._locks[shard_index]:
            history = self._shards[shard_index].get(cc_num, deque())

            # Prune old transactions from the left of the deque
            cutoff_time = transaction['unix_time'] - self.max_age_seconds
            while history and history[0]['unix_time'] < cutoff_time:
                history.popleft()

            history.append(transaction)
            self._shards[shard_index][cc_num] = history
            return list(history)


# ============================================================================
# 3. LIVE PREDICTION MAIN SCRIPT
# ============================================================================

# --- Global Shared Resources ---
predictor = None
history_manager = ShardedHistoryManager(Config.NUM_HISTORY_SHARDS, Config.HISTORY_MAX_AGE_SECONDS)
raw_transaction_queue = queue.Queue(maxsize=10000)
batch_queue = queue.Queue(maxsize=100)  # Queues of batches
rate_limiter = RateLimiter(max_calls=Config.MAX_FLAGS_PER_SECOND, period=1)
flagging_pool = ThreadPoolExecutor(max_workers=Config.NUM_FLAGGING_WORKERS, thread_name_prefix='FlagWorker')
csv_lock = threading.Lock()
shutdown_event = threading.Event()


def flag_transaction_callback(future):
    try:
        result = future.result()
        log_entry = future.context
        log_entry['flag_success'] = result.get('success', False)
        log_entry['flag_reason'] = result.get('reason', 'N/A')
        log_to_csv(log_entry)
    except Exception as e:
        print(f"Error in flagging callback: {e}")


def flag_transaction(trans_num, flag_value):
    with rate_limiter:
        payload = {"trans_num": trans_num, "flag_value": flag_value}
        try:
            response = requests.post(Config.FLAG_URL, headers={"X-API-Key": Config.API_KEY}, json=payload, verify=False,
                                     timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"success": False, "reason": f"API request failed: {e}"}


def log_to_csv(data_dict):
    field_names = ['timestamp', 'trans_num', 'cc_num', 'amount', 'category', 'dob', 'fraud_probability',
                   'predicted_label', 'flag_success', 'flag_reason']
    file_exists = os.path.isfile(Config.OUTPUT_CSV_FILE)
    with csv_lock:
        with open(Config.OUTPUT_CSV_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=field_names, extrasaction='ignore')
            if not file_exists or f.tell() == 0:
                writer.writeheader()
            writer.writerow(data_dict)


def validate_transaction(transaction):
    """Performs cheap, initial validation before adding to a batch."""
    if not transaction.get('trans_num') or not transaction.get('cc_num') or not transaction.get(
            'trans_date') or not transaction.get('trans_time') or not transaction.get('dob'):
        return None, "Skipping transaction with missing core fields."

    try:
        # Pre-calculate unix_time if missing, it's critical
        if 'unix_time' not in transaction:
            trans_datetime = pd.to_datetime(transaction['trans_date'] + ' ' + transaction['trans_time'])
            transaction['unix_time'] = int(trans_datetime.timestamp())
        else:  # ensure it's an int
            transaction['unix_time'] = int(transaction['unix_time'])
        return transaction, None
    except (TypeError, KeyError, ValueError):
        return None, f"Skipping transaction {transaction.get('trans_num')}: date/time fields are malformed."


def batch_collector():
    # <--- CHANGE 4: This thread creates batches of transactions
    """Pulls transactions from the raw queue and pushes them as batches."""
    print("Batch collector started.")
    batch = []
    last_batch_time = time.time()
    while not shutdown_event.is_set():
        try:
            # Wait for a transaction, but with a timeout to send partial batches
            timeout = max(0, Config.BATCH_TIMEOUT_SECONDS - (time.time() - last_batch_time))
            transaction = raw_transaction_queue.get(timeout=timeout)

            transaction, error = validate_transaction(transaction)
            if error:
                # print(error) # Optional: for debugging
                continue

            batch.append(transaction)
            if len(batch) >= Config.BATCH_SIZE:
                batch_queue.put(batch)
                batch = []
                last_batch_time = time.time()
        except queue.Empty:
            # Timeout reached, send whatever we have
            if batch:
                batch_queue.put(batch)
                batch = []
            last_batch_time = time.time()
    print("Batch collector shutting down.")


def prediction_worker():
    # <--- CHANGE 5: Worker now processes an entire batch at once
    """Pulls a batch of transactions, gets predictions, and flags them."""
    print("Prediction worker started.")
    while not shutdown_event.is_set() or not batch_queue.empty():
        try:
            batch = batch_queue.get(timeout=1)

            user_sequences_for_prediction = {}
            for tx in batch:
                # Update history and get the full sequence for this user
                history = history_manager.add_and_get_history(tx)
                user_sequences_for_prediction[tx['cc_num']] = history

            # Get predictions for the entire batch of sequences
            predictions = predictor.predict_batch(user_sequences_for_prediction)

            if predictions.empty:
                continue

            # Create a map for quick lookup
            tx_map = {tx['trans_num']: tx for tx in batch}

            # Find the new predictions and flag them
            for _, row in predictions.iterrows():
                trans_num = row['trans_num']
                if trans_num in tx_map:
                    fraud_probability = row['fraud_probability']
                    is_fraud = 1 if fraud_probability >= Config.DECISION_THRESHOLD else 0

                    log_entry = {
                        **tx_map[trans_num],  # includes original transaction fields
                        'timestamp': int(time.time()),
                        'fraud_probability': f"{fraud_probability:.4f}",
                        'predicted_label': is_fraud,
                        'amount': tx_map[trans_num].get('amt')
                    }

                    future = flagging_pool.submit(flag_transaction, trans_num, is_fraud)
                    future.context = log_entry
                    future.add_done_callback(flag_transaction_callback)

            batch_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in prediction worker: {e}")
    print("Prediction worker shutting down.")


def stream_listener():
    print("Connecting to transaction stream...")
    try:
        response = requests.get(Config.STREAM_URL, headers={"X-API-Key": Config.API_KEY}, stream=True, verify=False,
                                timeout=(10, 60))
        if response.status_code != 200:
            print(f"FATAL: Connection failed. Status: {response.status_code}, Msg: {response.text}")
            return
        print("Connection successful. Listening for events...")
        client = SSEClient(response)
        for event in client.events():
            if shutdown_event.is_set():
                break
            if event.data:
                try:
                    raw_transaction_queue.put(orjson.loads(event.data))
                except orjson.JSONDecodeError:
                    print("Warning: Malformed JSON from stream.")
    except requests.exceptions.RequestException as e:
        if not shutdown_event.is_set():
            print(f"NETWORK ERROR in stream listener: {e}")
    finally:
        print("Stream listener has stopped.")


if __name__ == '__main__':
    try:
        predictor = TransactionPredictor(model_path=Config.MODEL_PATH, preprocessor_dir=Config.PREPROCESSOR_DIR)
    except Exception as e:
        print(f"Could not initialize predictor: {e}")
        exit(1)

    print("Starting background services...")
    print(f"Using {Config.NUM_HISTORY_SHARDS} history shards to reduce lock contention.")
    print(f"Batching up to {Config.BATCH_SIZE} transactions or waiting {Config.BATCH_TIMEOUT_SECONDS}s.")

    # Start the batch collector
    batch_thread = threading.Thread(target=batch_collector, daemon=True)
    batch_thread.start()

    # Start the prediction workers
    prediction_threads = []
    for i in range(Config.NUM_PREDICTION_WORKERS):
        t = threading.Thread(target=prediction_worker, daemon=True)
        t.start()
        prediction_threads.append(t)

    # Main loop to listen to the stream
    while True:
        try:
            stream_listener()
            if shutdown_event.is_set():  # Graceful exit from loop
                break
            backoff_time = 15
            print(f"Connection lost. Waiting {backoff_time} seconds before reconnecting...")
            time.sleep(backoff_time)
        except KeyboardInterrupt:
            print("\nCtrl+C received. Shutting down gracefully...")
            shutdown_event.set()
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred in the main loop: {e}")
            shutdown_event.set()
            break

    print("\nWaiting for prediction queue to empty...")
    batch_queue.join()
    print("All predictions complete. Waiting for final API flags to be sent...")
    flagging_pool.shutdown(wait=True)
    print("\nAll tasks processed. Application has shut down.")