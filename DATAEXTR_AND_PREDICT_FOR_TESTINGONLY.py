import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl
import joblib
import os
import warnings
from sklearn.metrics import confusion_matrix, classification_report

# --- Suppress Warnings ---
warnings.filterwarnings('ignore')


# ============================================================================
# 1. CONFIGURATION
# ============================================================================
class Config:
    # IMPORTANT: Update these paths to matches the new environment
    MODEL_PATH = r"NEW-best-fraud-model-epoch=09-val_auc=0.9850.ckpt"
    PREPROCESSOR_DIR = "preprocessors"
    DECISION_THRESHOLD = 0.76


# ============================================================================
# 2. MODEL DEFINITION (Unchanged)
# ============================================================================
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


# ============================================================================
# 3. PIPELINE CLASS
# ============================================================================
class FraudDetectionPipeline:
    def __init__(self, model_path, preprocessor_dir):
        self.model_path = model_path
        self.preprocessor_dir = preprocessor_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        try:
            self.scalers = joblib.load(os.path.join(self.preprocessor_dir, 'scalers.pkl'))
            self.encoders = joblib.load(os.path.join(self.preprocessor_dir, 'encoders.pkl'))
            self.num_cols = joblib.load(os.path.join(self.preprocessor_dir, 'num_cols.pkl'))
            self.cat_cols = joblib.load(os.path.join(self.preprocessor_dir, 'cat_cols.pkl'))
            self.model = FraudTransformer.load_from_checkpoint(self.model_path, map_location=self.device)
            self.model.eval()
            print("✅ Model and preprocessors loaded successfully.")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"FATAL: Could not find preprocessor or model files: {e}")

    def _validate_input_columns(self, df: pd.DataFrame):
        required_cols = set(self.num_cols + self.cat_cols + ['trans_date', 'trans_time', 'dob', 'cc_num', 'unix_time'])
        engineered_cols = {'age', 'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
                           'time_since_last_transaction'}
        # We only need to check for columns that are NOT engineered by this script
        cols_to_check = [col for col in required_cols if col not in engineered_cols]

        missing_cols = [col for col in cols_to_check if col not in df.columns]
        if missing_cols:
            raise ValueError(
                "The input data is missing required columns that the model was trained on.\n"
                f"Missing columns: {missing_cols}\n"
                "Please ensure your input CSV contains ALL features used during training."
            )

    def extract_features(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        print("Starting feature extraction...")
        self._validate_input_columns(df_raw)
        data = df_raw.copy()

        # Time-based features
        data['trans_datetime'] = pd.to_datetime(data['trans_date'] + ' ' + data['trans_time'], errors='coerce')
        data['dob'] = pd.to_datetime(data['dob'], errors='coerce')
        data['age'] = (data['trans_datetime'] - data['dob']).dt.days / 365.25
        data['hour'] = data['trans_datetime'].dt.hour
        data['day_of_week'] = data['trans_datetime'].dt.dayofweek

        # Cyclical encoding
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        data['day_of_week_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['day_of_week_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)

        # Sequential features
        data = data.sort_values(by=['cc_num', 'unix_time'])
        data['time_since_last_transaction'] = data.groupby('cc_num')['unix_time'].diff().fillna(0)

        # Apply pre-trained scalers
        for col in self.num_cols:
            if col in data.columns:
                data[col] = self.scalers[col].transform(data[[col]])

        # Apply pre-trained encoders
        for col in self.cat_cols:
            if col in data.columns:
                data[col] = data[col].astype(str)
                known_classes = list(self.encoders[col].classes_)
                # Map unseen categories to 0 (assuming 0 is reserved for unknown/padding)
                data[col] = pd.Categorical(data[col], categories=known_classes).codes + 1

        print("✅ Feature extraction complete.")
        return data

    def predict_and_evaluate(self, df_processed: pd.DataFrame, true_labels_col: str):
        print("Starting prediction and evaluation...")
        if true_labels_col not in df_processed.columns:
            raise ValueError(f"True labels column '{true_labels_col}' not found.")

        # Prepare sequences
        grouped = df_processed.groupby('cc_num')
        all_num_seqs, all_cat_seqs, original_indices = [], [], []

        # We use strict column selection here based on what the model expects
        for _, group in grouped:
            all_num_seqs.append(torch.tensor(group[self.num_cols].values, dtype=torch.float32))
            all_cat_seqs.append(torch.tensor(group[self.cat_cols].values, dtype=torch.long))
            original_indices.extend(group.index.tolist())

        if not all_num_seqs: return pd.DataFrame(), "No data", np.array([])

        # Pad and move to device
        num_data_padded = pad_sequence(all_num_seqs, batch_first=True, padding_value=0.0).to(self.device)
        cat_data_padded = pad_sequence(all_cat_seqs, batch_first=True, padding_value=0).to(self.device)
        seq_lengths = torch.tensor([len(s) for s in all_num_seqs], device=self.device)
        padding_mask = torch.arange(num_data_padded.size(1), device=self.device)[None, :] >= seq_lengths[:, None]

        # Predict
        with torch.no_grad():
            logits = self.model(num_data_padded, cat_data_padded, padding_mask)
            fraud_probs_padded = torch.softmax(logits, dim=-1)[:, :, 1]

        # Unpad results
        fraud_probs_flat = []
        for i, length in enumerate(seq_lengths):
            fraud_probs_flat.extend(fraud_probs_padded[i, :length].cpu().numpy())

        # Format output
        results_df = pd.DataFrame({'fraud_probability': fraud_probs_flat}, index=original_indices)
        df_results = df_processed.join(results_df)
        df_results['predicted_label'] = (df_results['fraud_probability'] >= Config.DECISION_THRESHOLD).astype(int)

        # Evaluate
        report = classification_report(df_results[true_labels_col], df_results['predicted_label'])
        conf_matrix = confusion_matrix(df_results[true_labels_col], df_results['predicted_label'])

        print("✅ Prediction and evaluation complete.")
        return df_results, report, conf_matrix


# ============================================================================
# 4. EXAMPLE USAGE
# ============================================================================
if __name__ == '__main__':
    from io import StringIO

    # --- YOUR DATA!!!! ---
    # --- PLEASE ADAPT TO YOUR NEEDS ---
    csv_data = """
    ####
"""
    try:
        pipeline = FraudDetectionPipeline(Config.MODEL_PATH, Config.PREPROCESSOR_DIR)

        # Simulating loading data from a file/server
        raw_df = pd.read_csv(StringIO(csv_data))
        print(f"Loaded {len(raw_df)} transactions.")

        # 1. EXTRACT
        processed_df = pipeline.extract_features(raw_df)

        # 2. PREDICT & EVALUATE
        final_df, report, cm = pipeline.predict_and_evaluate(processed_df, 'is_fraud')

        print("\n--- Results Preview ---")
        print(final_df[['trans_num', 'amt', 'is_fraud', 'fraud_probability', 'predicted_label']].head())
        print("\n--- Classification Report ---\n", report)
        print("\n--- Confusion Matrix ---\n", cm)

    except Exception as e:
        print(f"\nERROR: {e}")