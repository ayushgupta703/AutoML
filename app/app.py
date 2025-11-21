# app/app.py  (replace existing)
import os
from flask import Flask, render_template, request, redirect, url_for, send_file, flash, session

import preprocess_pipeline
import train_pipeline
import predict_pipeline

app = Flask(__name__)
app.secret_key = "super_secret_major_project_key"

# ----- DIRECTORIES -----
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
# HOME PAGE
# ----------------------------------------------------
@app.route("/")
def index():
    status = {
        "raw": os.path.exists(RAW_PATH),
        "preprocessed": os.path.exists(PREPROC_PATH),
        "model": os.path.exists(MODEL_PATH),
        "scaler": os.path.exists(SCALER_PATH),
        "new_data": os.path.exists(NEW_DATA_PATH),
        "predicted": os.path.exists(PRED_OUTPUT_PATH)
    }

    # Retrieve last action logs (if any) from session
    last = session.get("last_activity")  # dictionary returned by wrappers
    last_type = session.get("last_activity_type")  # 'preprocess' or 'train' or 'predict'
    return render_template("index.html", status=status, last=last, last_type=last_type)


# ----------------------------------------------------
# UPLOAD RAW DATASET (with Task Type)
# ----------------------------------------------------
@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files.get("file")
        task_type = request.form.get("task_type", "classification").strip().lower()

        if not file:
            flash("Please select a CSV file.", "danger")
            return redirect(url_for("upload"))
        if not file.filename.endswith(".csv"):
            flash("Only CSV files allowed.", "danger")
            return redirect(url_for("upload"))

        # Save raw CSV
        os.makedirs(os.path.join(BASE_DIR, "..", "data"), exist_ok=True)
        raw_path = os.path.join(BASE_DIR, "..", "data", "raw.csv")
        file.save(raw_path)

        # Save task type
        with open(TASK_FILE, "w", encoding="utf-8") as f:
            f.write(task_type)

        flash(f"Raw dataset uploaded and task set to '{task_type}'.", "success")
        # go to preprocess page where user can select target too
        return redirect(url_for("preprocess"))

    return render_template("upload.html")



# ----------------------------------------------------
# PREPROCESSING (now supports target selection)
# ----------------------------------------------------
@app.route("/preprocess", methods=["GET", "POST"])
def preprocess():
    import pandas as pd

    # GET → show dropdown with column names
    if request.method == "GET":
        columns = []
        if os.path.exists(RAW_PATH):
            try:
                df = pd.read_csv(RAW_PATH, nrows=0)
                columns = list(df.columns)
            except:
                pass

        return render_template("preprocess.html", columns=columns, res=None)

    # POST → run preprocessing and show results on same page
    selected_target = request.form.get("target_column") or None

    res = preprocess_pipeline.run_preprocessing(selected_target=selected_target)

    # after preprocessing, raw still exists, so load columns again
    columns = []
    if os.path.exists(RAW_PATH):
        try:
            df = pd.read_csv(RAW_PATH, nrows=0)
            columns = list(df.columns)
        except:
            pass

    return render_template("preprocess.html", columns=columns, res=res)




# ----------------------------------------------------
# TRAINING (redirect to home with logs)
# ----------------------------------------------------
@app.route("/train", methods=["GET", "POST"])
def train():
    if request.method == "GET":
        return render_template("train.html", res=None)

    res = train_pipeline.run_training()
    return render_template("train.html", res=res)



# ----------------------------------------------------
# UPLOAD NEW DATA FOR PREDICTION
# ----------------------------------------------------
@app.route("/predict-upload", methods=["GET", "POST"])
def predict_upload():
    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            flash("Please upload a CSV file first.", "danger")
            return redirect(url_for("predict_upload"))
        if not file.filename.endswith(".csv"):
            flash("Only CSV files allowed.", "danger")
            return redirect(url_for("predict_upload"))
        file.save(NEW_DATA_PATH)
        flash("New data uploaded successfully!", "success")
        return redirect(url_for("predict"))
    return render_template("predict_upload.html")


# ----------------------------------------------------
# PREDICT (no redirect)
# ----------------------------------------------------
@app.route("/predict")
def predict():
    res = predict_pipeline.run_prediction()

    # we keep predict page as-is (user asked predict outcome page to remain)
    # also store last_activity if you want:
    session["last_activity"] = res
    session["last_activity_type"] = "predict"

    return render_template("predict.html", res=res)


# ----------------------------------------------------
# DOWNLOAD FILES
# ----------------------------------------------------
@app.route("/download/<ftype>")
def download(ftype):
    if ftype == "raw":
        path = RAW_PATH
    elif ftype == "preprocessed":
        path = PREPROC_PATH
    elif ftype == "predicted":
        path = PRED_OUTPUT_PATH
    else:
        flash("Invalid file.", "danger")
        return redirect(url_for("index"))

    if not os.path.exists(path):
        flash("File does not exist!", "danger")
        return redirect(url_for("index"))
    return send_file(path, as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
