import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Load the trained model and scaler
# -------------------------------
model_path = "best_model_Logistic_Regression.pkl"   # change if your file name is different
scaler_path = "scaler.pkl"

print("📦 Loading saved model and scaler...")
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

print("✅ Model and scaler loaded successfully!\n")

# -------------------------------
# Load new unseen data
# -------------------------------
new_data = pd.read_csv("new_data.csv")
print("📂 New data loaded successfully!")
print(new_data.head())

# -------------------------------
# Scale the new data (using same scaler)
# -------------------------------
scaled_data = scaler.transform(new_data)

# -------------------------------
# Make predictions
# -------------------------------
predictions = model.predict(scaled_data)
print("\n🔮 Predictions on new data:")
print(predictions)

# -------------------------------
# Save predictions to CSV
# -------------------------------
new_data["Predicted_Species"] = predictions
new_data.to_csv("predicted_output.csv", index=False)
print("\n✅ Predictions saved to 'predicted_output.csv'!")
