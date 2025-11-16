from pipeline import load_dataset, get_dataset_info, choose_target_column, handle_missing_values, encode_categorical_columns, scale_features
# Load dataset
df = load_dataset("../data/Iris.csv")

if df is not None:
    get_dataset_info(df)

    # Select target
    target = choose_target_column(df)

    # Handle missing values
    df = handle_missing_values(df)

    # Encode categorical columns
    df, encoders = encode_categorical_columns(df)

    # Scale features
    df_scaled, scaler = scale_features(df, target)

    print("\n✅ Final Preprocessed Dataset Preview:")
    print(df_scaled.head(150))

    df_scaled.to_csv("../data/preprocessed_dataset.csv", index=False)
    print("✅ Preprocessed dataset saved as 'preprocessed_dataset.csv'")

    