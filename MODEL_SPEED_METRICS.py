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
from datetime import datetime, timedelta

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


# ============================================================================
# 1. MODEL AND PREDICTOR CLASS DEFINITIONS
# (These must be identical to the ones used for training)
# ============================================================================

class FraudTransformer(pl.LightningModule):
    # This class definition must match the one used for training
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
        logits = self.classifier(x)
        return logits


class TransactionPredictor:
    # This class is a copy of the one in your live script
    def __init__(self, model_path, preprocessor_dir):
        self.model_path = model_path
        self.preprocessor_dir = preprocessor_dir
        try:
            self.scalers = joblib.load(os.path.join(self.preprocessor_dir, 'scalers.pkl'))
            self.encoders = joblib.load(os.path.join(self.preprocessor_dir, 'encoders.pkl'))
            self.num_cols = joblib.load(os.path.join(self.preprocessor_dir, 'num_cols.pkl'))
            self.cat_cols = joblib.load(os.path.join(self.preprocessor_dir, 'cat_cols.pkl'))
        except FileNotFoundError as e:
            raise FileNotFoundError(f"FATAL: Preprocessor files not found: {e}")
        self.model = FraudTransformer.load_from_checkpoint(self.model_path, map_location=torch.device('cpu'))
        self.model.eval()

    def _preprocess_data(self, df):
        data = df.copy()
        data['trans_datetime'] = pd.to_datetime(data['trans_date'] + ' ' + data['trans_time'])
        data['dob'] = pd.to_datetime(data['dob'])
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
                encoded_col = pd.Series(0, index=data.index, dtype=int, name=col)
                known_classes = list(self.encoders[col].classes_)
                known_mask = data[col].isin(known_classes)
                if known_mask.any():
                    encoded_col[known_mask] = self.encoders[col].transform(data[col][known_mask]) + 1
                data[col] = encoded_col
        return data

    def predict(self, df_raw):
        original_data = df_raw[['trans_num', 'cc_num']].copy()
        df_processed = self._preprocess_data(df_raw)
        grouped = df_processed.groupby('cc_num')
        all_num_seqs, all_cat_seqs, seq_indices = [], [], []
        for _, group in grouped:
            all_num_seqs.append(torch.tensor(group[self.num_cols].values, dtype=torch.float32))
            all_cat_seqs.append(torch.tensor(group[self.cat_cols].values, dtype=torch.long))
            seq_indices.extend(group.index.tolist())
        if not all_num_seqs: return pd.DataFrame()
        num_data_padded = pad_sequence(all_num_seqs, batch_first=True, padding_value=0.0)
        cat_data_padded = pad_sequence(all_cat_seqs, batch_first=True, padding_value=0)
        with torch.no_grad():
            seq_lengths = torch.tensor([len(s) for s in all_num_seqs])
            padding_mask = torch.arange(num_data_padded.size(1))[None, :] >= seq_lengths[:, None]
            logits = self.model(num_data_padded, cat_data_padded, padding_mask)
            probabilities = torch.softmax(logits, dim=-1)
            fraud_probs = probabilities[:, :, 1]
        results = []
        current_idx = 0
        for i, length in enumerate(seq_lengths):
            user_probs = fraud_probs[i, :length]
            for j in range(length):
                results.append({'original_index': seq_indices[current_idx], 'fraud_probability': user_probs[j].item()})
                current_idx += 1
        return original_data.join(pd.DataFrame(results).set_index('original_index'))


# ============================================================================
# 2. HELPER FUNCTION TO GENERATE REALISTIC DUMMY DATA
# ============================================================================
def generate_dummy_data(num_transactions, num_users, cat_cols, num_cols):
    """Generates a DataFrame of transactions that mimics the real data structure."""
    print(f"Generating {num_transactions} dummy transactions for {num_users} users...")
    data = []
    start_time = datetime.now()

    # Get known categories from a loaded encoder to make data more realistic
    # NOTE: This assumes 'category' is a categorical column.
    try:
        encoders = joblib.load(os.path.join(PREPROCESSOR_DIR, 'encoders.pkl'))
        known_cats = encoders['category'].classes_
        known_merchants = encoders['merchant'].classes_
        known_jobs = encoders['job'].classes_
    except Exception:
        known_cats = ['gas_transport', 'food_dining', 'shopping_pos']
        known_merchants = ['Walmart', 'Amazon', 'Shell']
        known_jobs = ['Software Engineer', 'Doctor', 'Mechanic']

    for i in range(num_transactions):
        user_id = 1000 + (i % num_users)
        trans_time = start_time + timedelta(seconds=i * 2)

        tx = {
            'trans_num': f'bench_{i}',
            'cc_num': user_id,
            'trans_date': trans_time.strftime('%Y-%m-%d'),
            'trans_time': trans_time.strftime('%H:%M:%S'),
            'unix_time': int(trans_time.timestamp()),
            'dob': (start_time - timedelta(days=np.random.randint(365 * 20, 365 * 65))).strftime('%Y-%m-%d'),

            # Categorical
            'category': np.random.choice(known_cats),
            'merchant': np.random.choice(known_merchants),
            'gender': np.random.choice(['M', 'F']),
            'state': np.random.choice(['NY', 'CA', 'TX', 'FL', 'IL']),
            'job': np.random.choice(known_jobs),

            # Numerical
            'amt': round(np.random.uniform(5.0, 1500.0), 2),
            'lat': round(np.random.uniform(25.0, 49.0), 4),
            'long': round(np.random.uniform(-125.0, -70.0), 4),
            'city_pop': np.random.randint(50000, 9000000),
            'merch_lat': round(np.random.uniform(25.0, 49.0), 4),
            'merch_long': round(np.random.uniform(-125.0, -70.0), 4),
        }
        data.append(tx)

    return pd.DataFrame(data)


# ============================================================================
# 3. BENCHMARKING SCRIPT
# ============================================================================

if __name__ == '__main__':

    # --- CONFIGURE YOUR BENCHMARK HERE ---
    MODEL_PATH = r"E:\PythonStuff\checkpoints\best-fraud-model-epoch=09-val_auc=0.9851.ckpt"
    PREPROCESSOR_DIR = "preprocessors"

    # Parameters for the throughput test
    THROUGHPUT_TRANSACTIONS = 5000  # Number of transactions to process in the main test
    THROUGHPUT_USERS = 500  # Number of unique users for these transactions

    print("=" * 60)
    print("           MODEL PERFORMANCE BENCHMARK")
    print("=" * 60)

    # --- 1. Initialization Time ---
    # Measures the one-time cost of loading the model and preprocessors.
    print("\n[1] Measuring Initialization Time...")
    start_time = time.monotonic()
    try:
        predictor = TransactionPredictor(model_path=MODEL_PATH, preprocessor_dir=PREPROCESSOR_DIR)
    except Exception as e:
        print(f"Failed to initialize predictor: {e}")
        exit()
    init_duration = time.monotonic() - start_time
    print(f"    -> Model and preprocessors loaded in: {init_duration:.4f} seconds")

    # --- Generate Dummy Data ---
    # We use the loaded preprocessors to get the column lists
    dummy_df = generate_dummy_data(THROUGHPUT_TRANSACTIONS, THROUGHPUT_USERS, predictor.cat_cols, predictor.num_cols)

    # --- 2. Single Sequence Latency ---
    # Measures the time for one end-to-end prediction on a small sequence.
    print("\n[2] Measuring Single Sequence Prediction Latency...")
    # Get all transactions for one user to simulate a single sequence prediction
    single_user_df = dummy_df[dummy_df['cc_num'] == 1000]

    # Warm-up run (the first run can be slower due to caching)
    _ = predictor.predict(single_user_df)

    start_time = time.monotonic()
    _ = predictor.predict(single_user_df)
    latency_duration = time.monotonic() - start_time
    print(
        f"    -> End-to-end latency for one sequence of {len(single_user_df)} transactions: {latency_duration * 1000:.2f} ms")

    # --- 3. Throughput Test ---
    # Measures how many transactions the model can process per second in a batch.
    # This is the most important metric for a high-volume system.
    print("\n[3] Measuring Throughput...")

    # Warm-up run
    _ = predictor.predict(dummy_df.head(10))

    start_time = time.monotonic()
    _ = predictor.predict(dummy_df)  # Process the entire DataFrame
    throughput_duration = time.monotonic() - start_time

    throughput = THROUGHPUT_TRANSACTIONS / throughput_duration

    print(f"    -> Processed {THROUGHPUT_TRANSACTIONS} transactions in {throughput_duration:.4f} seconds.")
    print(f"    -> MODEL THROUGHPUT: {throughput:.2f} transactions per second")
    print("=" * 60)

    print(
        "\nBenchmark complete. The 'Throughput' value represents your model's maximum processing speed on a single CPU core.")