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
from pipeline import choose_target_column

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
PREPROCESSED_PATH = os.path.join(DATA_DIR, "preprocessed_dataset.csv")
BEST_MODEL_FILENAME_PREFIX = "best_model_"
BEST_MODEL_PATH = os.path.join(BASE_DIR, "best_model.pkl")  # will save generic name for easy loading
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
ACCURACY_PLOT = os.path.join(STATIC_DIR, "accuracy_plot.png")


def run_training():
    """
    Load preprocessed dataset, train multiple models, save best model and produce accuracy plot.
    Returns: dict { success, logs, results, best_model, plot }
    """
    logs = []
    try:
        logs.append("🏁 Training started.")

        if not os.path.exists(PREPROCESSED_PATH):
            logs.append(f"❌ Preprocessed data not found at {PREPROCESSED_PATH}. Run preprocessing first.")
            return {"success": False, "logs": logs}

        df = pd.read_csv(PREPROCESSED_PATH)
        logs.append(f"✅ Loaded preprocessed data: shape={df.shape}")

        # Determine target column (use last column by default)
        target = df.columns[-1]
        X = df.drop(columns=[target])
        y = df[target]

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logs.append(f"🔀 Train/Test split created. Train: {X_train.shape}, Test: {X_test.shape}")

        # Candidate models
        models = {
            "Logistic Regression": LogisticRegression(max_iter=500),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "SVM": SVC()
        }

        results = {}
        trained_models = {}

        for name, model in models.items():
            logs.append(f"⚙ Training {name} ...")
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = float(accuracy_score(y_test, preds))
            results[name] = acc
            trained_models[name] = model
            logs.append(f"➡ {name} accuracy: {acc:.4f}")

        # Save accuracy plot
        try:
            plt.figure(figsize=(6, 3))
            names = list(results.keys())
            vals = list(results.values())
            plt.bar(names, vals, color="#6CB4EE")
            plt.ylim(0, 1.05)
            plt.ylabel("Accuracy")
            plt.title("Model Accuracy Comparison")
            plt.tight_layout()
            plt.savefig(ACCURACY_PLOT, dpi=150)
            plt.close()
            logs.append(f"🖼 Accuracy plot saved to: {ACCURACY_PLOT}")
        except Exception as e:
            logs.append(f"⚠ Could not save accuracy plot: {e}")

        # Select best model
        best_name = max(results, key=results.get)
        best_model = trained_models[best_name]

        # Save best model to app/ as a stable filename the UI expects
        joblib.dump(best_model, BEST_MODEL_PATH)
        logs.append(f"💾 Best model ({best_name}) saved to: {BEST_MODEL_PATH}")

        # If scaler already exists in app/, keep it (created in preprocessing)
        if os.path.exists(SCALER_PATH):
            logs.append(f"🔁 Scaler present at: {SCALER_PATH}")
        else:
            logs.append("ℹ Scaler not found in app/ (ensure preprocessing saved it).")

        logs.append("✅ Training finished successfully.")
        return {"success": True, "logs": logs, "results": results, "best_model": best_name, "plot": ACCURACY_PLOT}

    except Exception as e:
        logs.append(f"❌ Exception during training: {e}")
        return {"success": False, "logs": logs, "error": str(e)}
