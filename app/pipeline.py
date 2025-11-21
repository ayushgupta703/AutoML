# app/pipeline.py
import os
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "label_encoders.pkl")
TARGET_ENCODER_PATH = os.path.join(BASE_DIR, "target_encoder.pkl")
FEATURE_ORDER_PATH = os.path.join(BASE_DIR, "feature_columns.pkl")
TARGET_FILE = os.path.join(BASE_DIR, "target_column.txt")


def load_dataset(file_path: str):
    """Load CSV into pandas DataFrame, return df or None and print/log on error."""
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded dataset {file_path} with shape {df.shape}")
        return df
    except Exception as e:
        print(f"Failed to load dataset {file_path}: {e}")
        return None


def get_dataset_preview(df, n=5):
    try:
        return df.head(n).to_dict(orient="records")
    except Exception:
        return []


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values:
      - numeric -> mean
      - categorical (object) -> mode
    """
    missing_count = df.isnull().sum().sum()
    if missing_count == 0:
        return df

    for col in df.columns:
        if df[col].dtype == "object":
            if df[col].isnull().any():
                df[col].fillna(df[col].mode().iloc[0], inplace=True)
        else:
            if df[col].isnull().any():
                df[col].fillna(df[col].mean(), inplace=True)
    return df


def encode_categorical_columns(df: pd.DataFrame):
    """
    Encode categorical columns using LabelEncoder.
    Returns: (df_encoded, encoders_dict)
    Encoders dict maps column_name -> LabelEncoder
    """
    encoders = {}
    df_copy = df.copy()
    for col in df_copy.columns:
        if df_copy[col].dtype == "object":
            le = LabelEncoder()
            df_copy[col] = df_copy[col].astype(str)
            df_copy[col] = le.fit_transform(df_copy[col])
            encoders[col] = le
    return df_copy, encoders


def scale_features(df: pd.DataFrame, target_col: str):
    """
    Scale feature columns (exclude target_col). Saves scaler to SCALER_PATH.
    Returns: (df_scaled, scaler)
    """
    scaler = StandardScaler()
    feature_cols = [c for c in df.columns if c != target_col]
    df_scaled = df.copy()
    if feature_cols:
        scaled = scaler.fit_transform(df[feature_cols])
        df_scaled[feature_cols] = scaled
    joblib.dump(scaler, SCALER_PATH)
    return df_scaled, scaler
