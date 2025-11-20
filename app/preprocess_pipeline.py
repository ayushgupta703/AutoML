# app/preprocess_pipeline.py
import os
import joblib
import pandas as pd
from pipeline import load_dataset, get_dataset_info, choose_target_column, handle_missing_values, encode_categorical_columns, scale_features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))           # .../AutoML/app
DATA_DIR = os.path.join(BASE_DIR, "..", "data")                 # .../AutoML/data
RAW_PATH = os.path.join(DATA_DIR, "raw.csv")                    # expected raw upload path
PREPROCESSED_PATH = os.path.join(DATA_DIR, "preprocessed_dataset.csv")
LABEL_ENCODERS_PATH = os.path.join(BASE_DIR, "label_encoders.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")


def run_preprocessing(target: str = None):
    """
    Loads raw.csv, runs preprocessing functions and saves preprocessed_dataset.csv.
    If `target` is provided, use it as the target column; otherwise existing logic (choose_target_column) will run.
    Returns: dict { success: bool, logs: [str], preprocessed_path:, target: }
    """
    logs = []
    try:
        logs.append("🔁 Preprocessing started.")

        # 1) Load dataset (use raw.csv uploaded from UI)
        if not os.path.exists(RAW_PATH):
            logs.append(f"❌ Raw file not found at {RAW_PATH}. Upload raw dataset first.")
            return {"success": False, "logs": logs}

        df = load_dataset(RAW_PATH)
        if df is None:
            logs.append("❌ Failed to load dataset (see console).")
            return {"success": False, "logs": logs}
        logs.append(f"✅ Loaded dataset: shape={df.shape}")

        # 2) Preview
        try:
            preview = df.head(5).to_dict(orient="records")
            logs.append(f"Preview (first 5 rows): {preview}")
        except Exception:
            logs.append("⚠ Unable to create preview for dataset.")

        # 3) Missing values before
        missing_total = int(df.isnull().sum().sum())
        logs.append(f"🧾 Missing values (total) before processing: {missing_total}")

        # 4) Handle missing values
        df = handle_missing_values(df)
        logs.append("🧹 Missing value handling complete (if any).")

        # 5) Handle target column selection:
        if target and target in df.columns:
            chosen_target = target
            logs.append(f"🎯 Using target column (from UI): {chosen_target}")
        else:
            # fallback to previous behavior (choose_target_column uses input if interactive)
            # we use the helper choose_target_column but avoid input(): if it expects input() it will default to last column
            try:
                chosen_target = target if (target in df.columns) else df.columns[-1]
                logs.append(f"ℹ No valid target provided in UI. Defaulting to last column: {chosen_target}")
            except Exception:
                chosen_target = df.columns[-1]
                logs.append(f"ℹ Defaulting to last column as target: {chosen_target}")

        # 6) Encode categorical columns
        df_encoded, label_encoders = encode_categorical_columns(df)
        if label_encoders:
            joblib.dump(label_encoders, LABEL_ENCODERS_PATH)
            logs.append(f"🔡 Saved label encoders to: {LABEL_ENCODERS_PATH}")
        else:
            logs.append("ℹ No categorical encoders were created (no object dtype columns).")

        # 7) Scale features (requires target)
        df_scaled, scaler = scale_features(df_encoded, chosen_target)
        # Ensure the scaler is saved in the app/ folder
        try:
            joblib.dump(scaler, SCALER_PATH)
            logs.append(f"📏 Scaler saved to: {SCALER_PATH}")
        except Exception as e:
            logs.append(f"⚠ Failed to save scaler: {e}")

        # 8) Save preprocessed dataset
        os.makedirs(os.path.dirname(PREPROCESSED_PATH), exist_ok=True)
        df_scaled.to_csv(PREPROCESSED_PATH, index=False)
        logs.append(f"💾 Preprocessed dataset saved to: {PREPROCESSED_PATH}")

        logs.append("✅ Preprocessing finished successfully.")
        return {"success": True, "logs": logs, "preprocessed_path": PREPROCESSED_PATH, "target": chosen_target}

    except Exception as e:
        logs.append(f"❌ Exception during preprocessing: {e}")
        return {"success": False, "logs": logs, "error": str(e)}
