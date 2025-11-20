# app/predict_pipeline.py
import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
NEW_DATA_PATH = os.path.join(DATA_DIR, "new_data.csv")
PREDICTED_PATH = os.path.join(DATA_DIR, "predicted_output.csv")
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
LABEL_ENCODERS_PATH = os.path.join(BASE_DIR, "label_encoders.pkl")

DEFAULT_SPECIES_MAP = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}


def run_prediction():
    """
    Loads new_data.csv, applies scaler/model and saves predicted_output.csv.
    Returns: dict { success, logs, predicted_path }
    """
    logs = []
    try:
        logs.append("🔮 Prediction started.")

        if not os.path.exists(NEW_DATA_PATH):
            logs.append(f"❌ New data not found at {NEW_DATA_PATH}. Upload new data first.")
            return {"success": False, "logs": logs}

        df = pd.read_csv(NEW_DATA_PATH)
        logs.append(f"✅ Loaded new data: shape={df.shape}")

        # Load model
        if not os.path.exists(MODEL_PATH):
            logs.append(f"❌ Model not found at {MODEL_PATH}. Train model first.")
            return {"success": False, "logs": logs}
        model = joblib.load(MODEL_PATH)
        logs.append(f"📦 Loaded model from: {MODEL_PATH}")

        # Load scaler if present, else predict raw
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            try:
                X_scaled = scaler.transform(df)
                X = pd.DataFrame(X_scaled, columns=df.columns)
                logs.append("📏 Applied scaler to new data.")
            except Exception as e:
                logs.append(f"⚠ Failed to apply scaler: {e} - predicting on raw data.")
                X = df
        else:
            logs.append("ℹ No scaler found. Predicting on raw data.")
            X = df

        # Run prediction
        try:
            preds = model.predict(X)
            logs.append("✅ Prediction completed.")
        except Exception as e:
            logs.append(f"❌ Error during model prediction: {e}")
            return {"success": False, "logs": logs, "error": str(e)}

        # Decode labels if encoders exist
        decoded = []
        if os.path.exists(LABEL_ENCODERS_PATH):
            try:
                encs = joblib.load(LABEL_ENCODERS_PATH)
                first_le = list(encs.values())[0]
                decoded = first_le.inverse_transform(preds.astype(int))
                logs.append("🔡 Decoded labels using saved label encoder.")
            except Exception:
                decoded = [str(p) for p in preds]
                logs.append("⚠ Failed to decode with saved encoders - using raw predictions.")
        else:
            try:
                decoded = [DEFAULT_SPECIES_MAP.get(int(p), str(p)) for p in preds]
                logs.append("ℹ Decoded labels using default Iris mapping.")
            except Exception:
                decoded = [str(p) for p in preds]
                logs.append("ℹ Using raw prediction values (no decoding).")

        # Save results
        df["Predicted"] = decoded
        os.makedirs(os.path.dirname(PREDICTED_PATH), exist_ok=True)
        df.to_csv(PREDICTED_PATH, index=False)
        logs.append(f"💾 Predictions saved to: {PREDICTED_PATH}")

        logs.append("✅ Prediction finished successfully.")
        return {"success": True, "logs": logs, "predicted_path": PREDICTED_PATH}

    except Exception as e:
        logs.append(f"❌ Exception during prediction: {e}")
        return {"success": False, "logs": logs, "error": str(e)}
