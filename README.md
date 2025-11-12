# 🤖 AutoML Using AI Agents

An intelligent system that automates the process of building Machine Learning pipelines — from data preprocessing to model training, evaluation, and prediction — using rule-based AI decision agents.

---

## 🧩 Project Overview

### 🎯 Objective

To automate the ML workflow by enabling an AI agent to:

- Analyze datasets
- Handle preprocessing automatically
- Select suitable ML models
- Train, evaluate, and save the best-performing model

---

## 🧱 System Design (Architecture)

flowchart TD
    A[User / Dataset Upload] --> B[Data Preprocessing Module]
    B[Data Preprocessing Module] --> C[AI Agent Decision Module]
    C[AI Agent Decision Module] --> D[AutoML Training Pipeline]
    D[AutoML Training Pipeline] --> E[Model Evaluation & Selection]
    E[Model Evaluation & Selection] --> F[Model Saving (.pkl Files)]
    F[Model Saving (.pkl Files)] --> G[Prediction Module]
    G[Prediction Module] --> H[Streamlit Web UI (Future Integration)]


### 🧠 Modules

1. **Data Preprocessing** – handles missing values, encoding, scaling.
2. **AI Agent (Rule-based)** – auto-analyzes dataset & suggests models.
3. **Model Training** – trains multiple models, evaluates, and saves the best one.
4. **Prediction** – predicts on new unseen data using saved model & scaler.
5. **Streamlit UI (Upcoming)** – will provide an interactive interface.

---

## ⚙️ Technologies Used

- **Language:** Python 3.11
- **Libraries:** Scikit-learn, Pandas, NumPy, Seaborn, Matplotlib, Joblib
- **Automation:** Rule-based agent logic
- **Future Scope:** LLM-based AI agents, Streamlit UI

---

## 📂 Folder Structure

````bash
AutoML-using-AI-Agents/
│
├── app/
│ ├── pipeline.py
│ ├── train_pipeline.py
│ ├── predict_pipeline.py
│ ├── test_pipeline.py
│ └── logs/
│
├── data/
│ ├── Iris.csv
│ ├── preprocessed_dataset.csv
│ ├── new_data.csv
│ └── predicted_output.csv
│
├── best_model_Logistic_Regression.pkl
├── scaler.pkl
├── requirements.txt
├── README.md
└── .gitignore

````

---

## 🚀 How to Run Locally
```bash
# Clone the repository
git clone https://github.com/your-username/AutoML.git
cd AutoML

# Create a virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run training pipeline
python app/train_pipeline.py

# Run prediction
python app/predict_pipeline.py

````

📅 Current Progress
| Module                      | Status         |
| --------------------------- | -------------- |
| Data Preprocessing          | ✅ Completed    |
| Model Training & Evaluation | ✅ Completed    |
| Prediction Module           | ✅ Completed    |
| AI Agent Integration        | ⚙️ In Progress |
| Streamlit UI                | 🕓 Upcoming    |