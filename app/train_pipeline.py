import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib as jb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from pipeline import choose_target_column


# -------------------------
# Load the cleaned dataset
# -------------------------
df = pd.read_csv("../data/preprocessed_dataset.csv")   # This file was saved in Week 2

# -------------------------
# Separate features and target
# -------------------------
target_column = choose_target_column(df)  # change this if your target column is different
X = df.drop(target_column, axis=1)
y = df[target_column]

# -------------------------
# Train-Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)

# -------------------------
# Train Multiple Model
# -------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC()
}

results = {}
trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    trained_models[name] = model
    print(f"✅ {name} Accuracy: {acc:.4f}")


# -------------------------
#Find Best Model
# -------------------------
best_model_name = max(results, key=results.get)
best_model = trained_models[best_model_name]
print(f"\n🏆 Best Model: {best_model_name} with Accuracy: {results[best_model_name]:.4f}")


# -------------------------
# Step 3a: Plot accuracy comparison
# -------------------------
plt.figure(figsize=(6, 4))
sns.barplot(x=list(results.keys()), y=list(results.values()), palette="viridis")
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xticks(rotation=20)
plt.ylim(0, 1.1)
plt.show()

# -------------------------
# Step 3b: Confusion matrix for best model
# -------------------------
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=set(y), yticklabels=set(y))
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# -------------------------
# Step 4: Save Best Model & Scaler
# -------------------------
# Save the best trained model
model_filename = f"best_model_{best_model_name.replace(' ', '_')}.pkl"
jb.dump(best_model, model_filename)
print(f"💾 Best model saved as: {model_filename}")

# note: make sure you also saved the scaler in preprocessing step
try:
    scaler = jb.load("scaler.pkl")  # load previously saved scaler
    jb.dump(scaler, "saved_scaler.pkl")
    print("💾 Scaler saved as: saved_scaler.pkl")
except:
    print("⚠️ Scaler not found. Did you save it in preprocessing step?")
