import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# classification models & metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# regression models & metrics
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
PREPROCESSED_PATH = os.path.join(DATA_DIR, "preprocessed_dataset.csv")

BEST_MODEL_PATH = os.path.join(BASE_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "label_encoders.pkl")
TARGET_ENCODER_PATH = os.path.join(BASE_DIR, "target_encoder.pkl")
FEATURE_ORDER_PATH = os.path.join(BASE_DIR, "feature_columns.pkl")
TARGET_FILE = os.path.join(BASE_DIR, "target_column.txt")
TASK_FILE = os.path.join(BASE_DIR, "task_type.txt")

STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
ACCURACY_PLOT = os.path.join(STATIC_DIR, "accuracy_plot.png")


def run_training(models_to_train=None):
    logs = []
    try:
        logs.append("🏁 Training started.")

        if not os.path.exists(PREPROCESSED_PATH):
            logs.append(f"❌ Preprocessed dataset not found at {PREPROCESSED_PATH}. Run preprocessing first.")
            return {"success": False, "logs": logs}

        df = pd.read_csv(PREPROCESSED_PATH)
        logs.append(f"📥 Loaded preprocessed data: shape={df.shape}")

        # Load target
        if os.path.exists(TARGET_FILE):
            with open(TARGET_FILE, "r", encoding="utf-8") as f:
                target = f.read().strip()
            logs.append(f"🎯 Target loaded: {target}")
        else:
            target = df.columns[-1]
            logs.append(f"⚠ No target file found; defaulting to last column: {target}")

        # Load task type
        task = "classification"
        if os.path.exists(TASK_FILE):
            with open(TASK_FILE, "r", encoding="utf-8") as f:
                task = f.read().strip().lower()
            logs.append(f"🛠 Task type: {task}")
        else:
            logs.append("⚠ task_type.txt missing; defaulting to classification.")

        # Load feature order or infer
        if os.path.exists(FEATURE_ORDER_PATH):
            feature_cols = joblib.load(FEATURE_ORDER_PATH)
            logs.append(f"📋 Loaded feature order ({len(feature_cols)} cols).")
        else:
            feature_cols = [c for c in df.columns if c != target]
            logs.append("ℹ Feature order file not found; using df column order.")

        # Ensure columns exist
        feature_cols = [c for c in feature_cols if c in df.columns]
        X = df[feature_cols]
        y = df[target]

        logs.append(f"🔀 Prepared features {X.shape} and target {y.shape}")

        # split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logs.append(f"🔀 Train/test split done: train={X_train.shape}, test={X_test.shape}")

        results = {}
        trained_models = {}

        if task == "classification":
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "SVM": SVC(),
            }

            if models_to_train:
                want = {str(m).strip() for m in models_to_train if str(m).strip()}
                models = {name: m for name, m in models.items() if name in want}
                logs.append(f"🤖 Model filter applied (classification): {list(models.keys())}")
            for name, model in models.items():
                logs.append(f"⚙ Training (classification) {name} ...")
                try:
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    acc = float(accuracy_score(y_test, preds))
                    results[name] = acc
                    trained_models[name] = model
                    logs.append(f"➡ {name} accuracy: {acc:.4f}")
                except Exception as e:
                    logs.append(f"❌ Error training {name}: {e}")

            # Save accuracy plot
            try:
                plt.figure(figsize=(7,4))
                plt.bar(list(results.keys()), list(results.values()), color="#6CB4EE")
                plt.ylim(0,1.05)
                plt.ylabel("Accuracy")
                plt.title("Model Accuracy Comparison")
                plt.xticks(rotation=20)
                plt.tight_layout()
                plt.savefig(ACCURACY_PLOT, dpi=150)
                plt.close()
                logs.append(f"🖼 Accuracy plot saved to {ACCURACY_PLOT}")
            except Exception as e:
                logs.append(f"⚠ Could not save accuracy plot: {e}")

        else:  # regression
            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest Regressor": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "SVR": SVR(),
            }

            if models_to_train:
                want = {str(m).strip() for m in models_to_train if str(m).strip()}
                models = {name: m for name, m in models.items() if name in want}
                logs.append(f"🤖 Model filter applied (regression): {list(models.keys())}")
            for name, model in models.items():
                logs.append(f"⚙ Training (regression) {name} ...")
                try:
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    rmse = math.sqrt(mean_squared_error(y_test, preds))
                    mae = mean_absolute_error(y_test, preds)
                    r2 = r2_score(y_test, preds)
                    # we will use RMSE as the score (lower is better) so store negative RMSE to pick max later
                    results[name] = -rmse
                    trained_models[name] = model
                    logs.append(f"➡ {name} RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
                except Exception as e:
                    logs.append(f"❌ Error training {name}: {e}")

            # Save simple metric plot (RMSE)
            try:
                names = list(results.keys())
                rmse_vals = [-v for v in results.values()]  # convert back to rmse
                plt.figure(figsize=(7,4))
                plt.bar(names, rmse_vals, color="#FFB56B")
                plt.ylabel("RMSE")
                plt.title("Model RMSE Comparison")
                plt.xticks(rotation=20)
                plt.tight_layout()
                plt.savefig(ACCURACY_PLOT, dpi=150)
                plt.close()
                logs.append(f"🖼 RMSE plot saved to {ACCURACY_PLOT}")
            except Exception as e:
                logs.append(f"⚠ Could not save regression plot: {e}")

        if not results:
            logs.append("❌ No models trained successfully.")
            return {"success": False, "logs": logs}

        # choose best
        best_name = max(results, key=results.get)
        best_model = trained_models[best_name]
        joblib.dump(best_model, BEST_MODEL_PATH)
        logs.append(f"💾 Best model ({best_name}) saved to: {BEST_MODEL_PATH}")

        logs.append("✅ Training finished.")
        return {"success": True, "logs": logs, "results": results, "best_model": best_name, "plot": ACCURACY_PLOT}

    except Exception as e:
        logs.append(f"❌ Unexpected error during training: {e}")
        return {"success": False, "logs": logs, "error": str(e)}
