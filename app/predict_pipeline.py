# app/predict_pipeline.py
import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

NEW_DATA_PATH = os.path.join(DATA_DIR, "new_data.csv")
PRED_OUTPUT_PATH = os.path.join(DATA_DIR, "predicted_output.csv")

BEST_MODEL_PATH = os.path.join(BASE_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "label_encoders.pkl")
TARGET_ENCODER_PATH = os.path.join(BASE_DIR, "target_encoder.pkl")
FEATURE_ORDER_PATH = os.path.join(BASE_DIR, "feature_columns.pkl")
TARGET_FILE = os.path.join(BASE_DIR, "target_column.txt")


def run_prediction():
    logs = []
    try:
        if not os.path.exists(NEW_DATA_PATH):
            logs.append(f"❌ New data not found at {NEW_DATA_PATH}. Upload new data first.")
            return {"success": False, "logs": logs}

        df_new = pd.read_csv(NEW_DATA_PATH)
        logs.append(f"📥 Loaded new data: shape={df_new.shape}")

        if not os.path.exists(BEST_MODEL_PATH):
            logs.append(f"❌ Model not found at {BEST_MODEL_PATH}. Train first.")
            return {"success": False, "logs": logs}

        model = joblib.load(BEST_MODEL_PATH)
        logs.append(f"📦 Loaded model: {BEST_MODEL_PATH}")

        # Load encoders (features)
        encoders = {}
        if os.path.exists(ENCODERS_PATH):
            encoders = joblib.load(ENCODERS_PATH)
            logs.append(f"🔤 Loaded feature encoders: {list(encoders.keys())}")

        # Load target encoder
        target_encoder = None
        if os.path.exists(TARGET_ENCODER_PATH):
            target_encoder = joblib.load(TARGET_ENCODER_PATH)
            logs.append("🎯 Loaded target encoder.")

        # Load scaler
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            logs.append("📏 Loaded scaler.")
        else:
            scaler = None
            logs.append("⚠ Scaler not found; proceeding without scaling (likely to fail).")

        # Load target column name
        if os.path.exists(TARGET_FILE):
            with open(TARGET_FILE, "r", encoding="utf-8") as f:
                target_col = f.read().strip()
            logs.append(f"🎯 Target column: {target_col}")
        else:
            target_col = None
            logs.append("⚠ target_column.txt missing; prediction may be misaligned.")

        # Load feature order
        feature_order = None
        if os.path.exists(FEATURE_ORDER_PATH):
            feature_order = joblib.load(FEATURE_ORDER_PATH)
            logs.append(f"📋 Loaded training feature order ({len(feature_order)} cols).")
        else:
            # fallback to current columns except target if present
            feature_order = [c for c in df_new.columns if c != target_col]
            logs.append("ℹ Feature order file missing; using new_data columns as order.")

        # Ensure all needed feature columns exist in new data
        missing_cols = [c for c in feature_order if c not in df_new.columns]
        if missing_cols:
            logs.append(f"⚠ Missing columns in new data: {missing_cols}")
            # add missing columns with zeros (or NaN->filled later)
            for c in missing_cols:
                df_new[c] = 0
            logs.append("ℹ Added missing columns with default 0 values.")

        # Reorder dataframe to match training feature order
        X_new = df_new[feature_order].copy()
        logs.append("🔀 Reordered new data to training feature order.")

        # Apply encoders to categorical columns where encoder exists
        for col, encoder in encoders.items():
            if col in X_new.columns:
                try:
                    # Convert to string first for safety, then transform unseen to mode mapping
                    X_new[col] = X_new[col].astype(str)
                    # handle unseen values: map unseen to most frequent class (encoder.classes_[0])
                    X_new[col] = X_new[col].apply(
                        lambda v: encoder.transform([v])[0] if v in encoder.classes_ else encoder.transform([encoder.classes_[0]])[0]
                    )
                    logs.append(f"🔡 Encoded column: {col}")
                except Exception as e:
                    logs.append(f"⚠ Failed to encode column {col}: {e}")

        # Apply scaler if present
        if scaler is not None:
            try:
                X_scaled = scaler.transform(X_new)
                X_in = pd.DataFrame(X_scaled, columns=feature_order)
                logs.append("📏 Scaler applied to new data.")
            except Exception as e:
                logs.append(f"⚠ Scaler transform failed: {e}. Using raw values.")
                X_in = X_new
        else:
            X_in = X_new

        # Predict
        try:
            preds = model.predict(X_in)
            logs.append("✅ Prediction completed.")
        except Exception as e:
            logs.append(f"❌ Model prediction failed: {e}")
            return {"success": False, "logs": logs, "error": str(e)}

        # Decode predictions if target encoder present
        if target_encoder is not None:
            try:
                decoded = target_encoder.inverse_transform(preds.astype(int))
                logs.append("🔍 Decoded predictions using target encoder.")
            except Exception:
                decoded = preds
                logs.append("⚠ Failed to decode predictions; returning raw preds.")
        else:
            decoded = preds

        # Save output: add predicted column name
        out_df = df_new.copy()
        pred_col_name = f"Predicted_{target_col}" if target_col else "Predicted"
        out_df[pred_col_name] = decoded
        os.makedirs(os.path.dirname(PRED_OUTPUT_PATH), exist_ok=True)
        out_df.to_csv(PRED_OUTPUT_PATH, index=False)
        logs.append(f"💾 Saved predictions to: {PRED_OUTPUT_PATH}")

        return {"success": True, "logs": logs, "predicted_path": PRED_OUTPUT_PATH}

    except Exception as e:
        logs.append(f"❌ Unexpected error in prediction pipeline: {e}")
        return {"success": False, "logs": logs, "error": str(e)}
