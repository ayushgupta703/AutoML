# app/preprocess_pipeline.py
import os
import joblib
import pandas as pd
from pipeline import (
    load_dataset,
    handle_missing_values,
    encode_categorical_columns,
    scale_features,
    TARGET_FILE,
    ENCODERS_PATH,
    TARGET_ENCODER_PATH,
    SCALER_PATH,
    FEATURE_ORDER_PATH
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
RAW_PATH = os.path.join(DATA_DIR, "raw.csv")
PREPROCESSED_PATH = os.path.join(DATA_DIR, "preprocessed_dataset.csv")


def run_preprocessing(selected_target=None):
    """
    Preprocess raw.csv and save:
      - data/preprocessed_dataset.csv
      - app/label_encoders.pkl      (feature encoders)
      - app/target_encoder.pkl      (encoder for target if categorical)
      - app/scaler.pkl
      - app/feature_columns.pkl     (list of feature column names used for training)
      - app/target_column.txt
    Returns dict with logs and preprocessed_path.
    """
    logs = []
    try:
        if not os.path.exists(RAW_PATH):
            logs.append(f"❌ Raw file not found at {RAW_PATH}. Upload raw dataset first.")
            return {"success": False, "logs": logs}

        df = load_dataset(RAW_PATH)
        if df is None:
            logs.append("❌ Failed to load raw dataset.")
            return {"success": False, "logs": logs}

        logs.append(f"📥 Loaded raw dataset: shape={df.shape}")
        logs.append(f"📄 Preview: {df.head(3).to_dict(orient='records')}")

        # Determine target
        if selected_target and selected_target in df.columns:
            target = selected_target
            logs.append(f"🎯 Using target provided by UI: {target}")
        else:
            target = df.columns[-1]
            logs.append(f"ℹ No valid target provided; defaulting to last column: {target}")

        # Save target to file
        os.makedirs(os.path.dirname(TARGET_FILE), exist_ok=True)
        with open(TARGET_FILE, "w", encoding="utf-8") as f:
            f.write(target)
        logs.append(f"💾 Saved target column to: {TARGET_FILE}")

        # Handle missing values
        df = handle_missing_values(df)
        logs.append("🧹 Missing values handled.")

        # Save original target column to preserve mapping (in case it's object)
        # We will encode it below if needed
        # Encode categorical columns (features + possibly target)
        df_encoded, encoders = encode_categorical_columns(df)
        logs.append(f"🔤 Encoded categorical columns: {list(encoders.keys())}")

        # If target was encoded, separate its encoder
        target_encoder = None
        if target in encoders:
            target_encoder = encoders.pop(target)
            joblib.dump(target_encoder, TARGET_ENCODER_PATH)
            logs.append(f"🔒 Saved target encoder to: {TARGET_ENCODER_PATH}")

        # Determine feature columns (order) — exclude target
        feature_cols = [c for c in df_encoded.columns if c != target]
        joblib.dump(feature_cols, FEATURE_ORDER_PATH)
        logs.append(f"📋 Saved feature column order: {FEATURE_ORDER_PATH}")

        # Scale features only (function will save scaler)
        df_scaled, scaler = scale_features(df_encoded, target)
        logs.append("📏 Scaled feature columns (target not scaled).")

        # Save feature encoders (remaining encoders) and scaler (scale_features already saved scaler)
        joblib.dump(encoders, ENCODERS_PATH)
        logs.append(f"💾 Saved feature encoders to: {ENCODERS_PATH}")
        # scaler already saved in pipeline.scale_features via joblib.dump(SCALER_PATH) in pipeline.py
        if os.path.exists(SCALER_PATH):
            logs.append(f"💾 Scaler saved to: {SCALER_PATH}")
        else:
            # Save scaler as fallback
            joblib.dump(scaler, SCALER_PATH)
            logs.append(f"💾 Scaler saved to: {SCALER_PATH}")

        # Save preprocessed CSV (df_scaled includes target as original/encoded values)
        os.makedirs(os.path.dirname(PREPROCESSED_PATH), exist_ok=True)
        df_scaled.to_csv(PREPROCESSED_PATH, index=False)
        logs.append(f"💾 Preprocessed dataset saved to: {PREPROCESSED_PATH}")

        logs.append("✅ Preprocessing completed successfully.")
        return {"success": True, "logs": logs, "preprocessed_path": PREPROCESSED_PATH}

    except Exception as e:
        logs.append(f"❌ Exception in preprocessing: {e}")
        return {"success": False, "logs": logs, "error": str(e)}
