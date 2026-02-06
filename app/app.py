import os
from flask import Flask, render_template, request, send_file, flash, session
import ai_agent
import preprocess_pipeline
import train_pipeline
import predict_pipeline

app = Flask(__name__)
app.secret_key = "super_secret_major_project_key"

# ----------------------------------------------------
# SESSION SAFE CONVERTER
# ----------------------------------------------------
def _session_safe(obj):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {str(k): _session_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_session_safe(v) for v in obj]
    if isinstance(obj, (str, int, bool)):
        return obj
    if hasattr(obj, "item"):
        return obj.item()
    try:
        return float(obj)
    except:
        return str(obj)

# ----------------------------------------------------
# PATHS
# ----------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

RAW_PATH = os.path.join(DATA_DIR, "raw.csv")
PREPROC_PATH = os.path.join(DATA_DIR, "preprocessed_dataset.csv")
NEW_DATA_PATH = os.path.join(DATA_DIR, "new_data.csv")
PRED_OUTPUT_PATH = os.path.join(DATA_DIR, "predicted_output.csv")

MODEL_PATH = os.path.join(BASE_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
TASK_FILE = os.path.join(BASE_DIR, "task_type.txt")

# ----------------------------------------------------
# SINGLE PAGE ROUTE
# ----------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":

        action = request.form.get("action")

        # ---------------------------
        # UPLOAD RAW DATA
        # ---------------------------
        if action == "upload_raw":
            file = request.files.get("file")

            if file and file.filename.endswith(".csv"):
                os.makedirs(DATA_DIR, exist_ok=True)
                file.save(RAW_PATH)
                flash("Dataset uploaded successfully.", "success")
            else:
                flash("Invalid CSV file.", "danger")

        # ---------------------------
        # RUN AUTOML (AI + PREPROCESS + TRAIN)
        # ---------------------------
        elif action == "run_automl":

            if not os.path.exists(RAW_PATH):
                flash("Upload dataset first.", "danger")
            else:
                try:
                    decision = ai_agent.analyze_dataset(RAW_PATH)

                    decision = {
                        "target_column": decision.get("target_column"),
                        "problem_type": decision.get("problem_type", "classification"),
                        "models": decision.get("models", []),
                        "reasoning": decision.get("reasoning", "")
                    }

                    with open(TASK_FILE, "w") as f:
                        f.write(decision["problem_type"])

                    pre_res = preprocess_pipeline.run_preprocessing(
                        selected_target=decision["target_column"]
                    )

                    if pre_res.get("success"):
                        train_res = train_pipeline.run_training(
                            models_to_train=decision["models"]
                        )
                    else:
                        train_res = {"success": False, "logs": ["Preprocessing failed."]}

                    combined = {
                        "decision": decision,
                        "preprocess": pre_res,
                        "train": train_res
                    }

                    session["last_activity"] = _session_safe(combined)
                    session["last_activity_type"] = "automl"

                    flash("AutoML completed successfully.", "success")

                except Exception as e:
                    flash(f"AutoML failed: {e}", "danger")

        # ---------------------------
        # UPLOAD NEW DATA + PREDICT
        # ---------------------------
        elif action == "predict":

            file = request.files.get("predict_file")

            if file and file.filename.endswith(".csv"):
                file.save(NEW_DATA_PATH)
                res = predict_pipeline.run_prediction()

                session["last_activity"] = _session_safe(res)
                session["last_activity_type"] = "predict"

                flash("Prediction completed.", "success")
            else:
                flash("Invalid prediction CSV.", "danger")

    # STATUS FLAGS
    status = {
        "raw": os.path.exists(RAW_PATH),
        "preprocessed": os.path.exists(PREPROC_PATH),
        "model": os.path.exists(MODEL_PATH),
        "predicted": os.path.exists(PRED_OUTPUT_PATH)
    }

    last = session.get("last_activity")
    last_type = session.get("last_activity_type")

    return render_template("index.html", status=status, last=last, last_type=last_type)

# ----------------------------------------------------
# DOWNLOAD FILES
# ----------------------------------------------------
@app.route("/download/<ftype>")
def download(ftype):

    mapping = {
        "raw": RAW_PATH,
        "preprocessed": PREPROC_PATH,
        "predicted": PRED_OUTPUT_PATH
    }

    path = mapping.get(ftype)

    if path and os.path.exists(path):
        return send_file(path, as_attachment=True)

    flash("File not found.", "danger")
    return render_template("index.html")

# ----------------------------------------------------
if __name__ == "__main__":
    app.run(debug=False)