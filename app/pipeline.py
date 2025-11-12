import pandas as pd
import joblib as jb
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Loads a dataset from a CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Dataset loaded successfully! Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None


def get_dataset_info(df: pd.DataFrame):
    """
    Prints dataset info: columns, types, missing values.
    """
    print("\n📊 Dataset Info:")
    print(df.info())
    print("\n🔍 First 5 rows:")
    print(df.head())
    print("\n❓ Missing values:")
    print(df.isnull().sum())


def choose_target_column(df: pd.DataFrame) -> str:
    """
    Asks user to choose a target column.
    If the input is invalid, defaults to the last column.
    """
    print("\nAvailable columns:", list(df.columns))
    user_input = input("👉 Enter the target column name (or press Enter to use last column): ").strip()

    if user_input in df.columns:
        print(f"✅ Target column set to: {user_input}")
        return user_input
    else:
        suggested = df.columns[-1]
        print(f"⚠️ Invalid input. Using last column by default: {suggested}")
        return suggested


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles missing values ONLY if they exist.
    - Numeric columns → fill with mean
    - Categorical columns → fill with mode
    """
    missing_count = df.isnull().sum().sum()
    if missing_count == 0:
        print("✅ No missing values found. Skipping cleaning.")
        return df

    print(f"⚠️ Found {missing_count} missing values. Handling them now...")
    for col in df.columns:
        if df[col].dtype == "object":
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].mean(), inplace=True)

    print("🧹 Missing values handled.")
    return df



def encode_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes categorical columns into numeric values.
    """
    label_encoders = {}
    for col in df.columns:
        if df[col].dtype == "object":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
            print(f"🔡 Encoded column: {col}")
    return df, label_encoders


def scale_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    scaler = StandardScaler()
    features = df.drop(columns=[target])
    scaled_features = scaler.fit_transform(features)

    df_scaled = pd.DataFrame(scaled_features, columns=features.columns)
    df_scaled[target] = df[target].values  

    # Save scaler for later use
    jb.dump(scaler, "scaler.pkl")
    print("💾 Scaler saved as: scaler.pkl")

    print("📏 Features scaled successfully.")
    return df_scaled, scaler
