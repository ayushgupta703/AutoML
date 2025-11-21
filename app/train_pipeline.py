# app/train_pipeline.py
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from pipeline import TARGET_FILE, FEATURE_ORDER_PATH

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
PREPROCESSED_PATH = os.path.join(DATA_DIR, "preprocessed_dataset.csv")

BEST_MODEL_PATH = os.path.join(BASE_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "label_encoders.pkl")
TARGET_ENCODER_PATH = os.path.join(BASE_DIR, "target_encoder.pkl")
FEATURE_ORDER_PATH_APP = FEATURE_ORDER_PATH  # same path

STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
ACCURACY_PLOT = os.path.join(STATIC_DIR, "accuracy_plot.png")


def run_training():
    logs = []
    try:
        logs.append("🏁 Training started.")

        if not os.path.exists(PREPROCESSED_PATH):
            logs.append(f"❌ Preprocessed dataset not found at {PREPROCESSED_PATH}. Run preprocessing first.")
            return {"success": False, "logs": logs}

        df = pd.read_csv(PREPROCESSED_PATH)
        logs.append(f"📥 Loaded preprocessed data: shape={df.shape}")

        # Load target (saved during preprocessing)
        if os.path.exists(TARGET_FILE):
            with open(TARGET_FILE, "r", encoding="utf-8") as f:
                target = f.read().strip()
            logs.append(f"🎯 Target loaded from file: {target}")
        else:
            # fallback
            target = df.columns[-1]
            logs.append(f"⚠ No target file found; defaulting to last column: {target}")

        # Load feature order (if exists) and ensure X uses that order
        feature_cols = None
        if os.path.exists(FEATURE_ORDER_PATH_APP):
            feature_cols = joblib.load(FEATURE_ORDER_PATH_APP)
            logs.append(f"📋 Loaded feature order ({len(feature_cols)} columns).")
        else:
            feature_cols = [c for c in df.columns if c != target]
            logs.append("ℹ Feature order file not found; using df column order.")

        # Ensure all feature cols exist in df (drop any missing)
        feature_cols = [c for c in feature_cols if c in df.columns]
        X = df[feature_cols]
        y = df[target]

        logs.append(f"🔀 Train/test split preparing. Features: {X.shape}, Target: {y.shape}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Models to evaluate
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "SVM": SVC()
        }

        results = {}
        trained_models = {}

        for name, model in models.items():
            logs.append(f"⚙ Training {name} ...")
            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                acc = float(accuracy_score(y_test, preds))
                results[name] = acc
                trained_models[name] = model
                logs.append(f"➡ {name} accuracy: {acc:.4f}")
            except Exception as e:
                logs.append(f"❌ Error training {name}: {e}")

        if not results:
            logs.append("❌ No models trained successfully.")
            return {"success": False, "logs": logs}

        # Save accuracy plot
        try:
            plt.figure(figsize=(7, 4))
            plt.bar(list(results.keys()), list(results.values()), color="#6CB4EE")
            plt.ylim(0, 1.05)
            plt.ylabel("Accuracy")
            plt.title("Model Accuracy Comparison")
            plt.xticks(rotation=20)
            plt.tight_layout()
            plt.savefig(ACCURACY_PLOT, dpi=150)
            plt.close()
            logs.append(f"🖼 Accuracy plot saved to {ACCURACY_PLOT}")
        except Exception as e:
            logs.append(f"⚠ Could not save accuracy plot: {e}")

        # Select and save best model
        best_name = max(results, key=results.get)
        best_model = trained_models[best_name]
        joblib.dump(best_model, BEST_MODEL_PATH)
        logs.append(f"💾 Best model ({best_name}) saved to: {BEST_MODEL_PATH}")

        # Confirm presence of scaler and encoders
        if os.path.exists(SCALER_PATH):
            logs.append(f"📏 Scaler present at {SCALER_PATH}")
        else:
            logs.append("⚠ Scaler missing - predictions may fail on new data.")

        if os.path.exists(ENCODERS_PATH):
            logs.append(f"🔤 Feature encoders present at {ENCODERS_PATH}")
        else:
            logs.append("ℹ No feature encoders found.")

        if os.path.exists(TARGET_ENCODER_PATH):
            logs.append(f"🎯 Target encoder present at {TARGET_ENCODER_PATH}")
        else:
            logs.append("ℹ No target encoder found (target might be numeric).")

        logs.append("✅ Training finished.")
        return {"success": True, "logs": logs, "results": results, "best_model": best_name, "plot": ACCURACY_PLOT}

    except Exception as e:
        logs.append(f"❌ Unexpected error during training: {e}")
        return {"success": False, "logs": logs, "error": str(e)}
