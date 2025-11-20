import pandas as pd
import numpy as np
import joblib

# --- 1. Load Artifacts and Feature Order ---
try:
    # Load the best model, scaler, and target encoder
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    le = joblib.load('label_encoder.pkl')

    # Load the feature column order (CRITICAL for One-Hot Encoding alignment)
    feature_columns = pd.read_csv('X_full_columns.csv', header=None)[0].tolist()

    print("✅ Model, Scaler, Encoder, and Feature Order Loaded.")
except FileNotFoundError as e:
    print(f"Error loading required files: {e}. Please ensure clean_code.py and model_building.py were run successfully.")
    exit()

# ----------------------------------------------------
# --- 2. Define Custom Input (RAW, UNSCALED DATA) ---
# ----------------------------------------------------

# Row 1: High Risk (Sleep Apnea / Original Input)
# Row 2: Insomnia Target (High Stress, Low Quality Sleep)
# Row 3: No Disorder Target (Low Stress, High Quality Sleep)
custom_data = {
    'Gender': ['Male', 'Female', 'Male'],
    'Age': [28, 35, 40],
    'Occupation': ['Sales Representative', 'Accountant', 'Engineer'],
    'Sleep Duration': [5.9, 5.5, 8.2],
    'Quality of Sleep': [4, 3, 9],
    'Physical Activity Level': [30, 40, 75],
    'Stress Level': [8, 9, 3], # Target Insomnia (9), Target No Disorder (3)
    'BMI Category': ['Obese', 'Normal', 'Normal'],
    'Systolic': [140, 120, 115],
    'Diastolic': [90, 80, 75],
    'Heart Rate': [85, 75, 60],
    'Daily Steps': [3000, 4500, 10000],
}

new_df = pd.DataFrame(custom_data)

# --- 3. Preprocessing Steps (Must exactly match clean_code.py) ---

def preprocess_new_data(df, feature_columns):

    # a) Fix BMI Category (Unify 'Normal Weight' to 'Normal')
    df['BMI Category'] = df['BMI Category'].replace('Normal Weight', 'Normal')

    # b) One-Hot Encode Categorical Features
    categorical_cols = ['Gender', 'Occupation', 'BMI Category']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # c) Reindex/Align Columns: CRITICAL STEP for consistent feature ordering
    # Add any missing one-hot columns (with 0s) and drop extra columns
    missing_cols = set(feature_columns) - set(df.columns)
    for col in missing_cols:
        df[col] = 0
    df = df[feature_columns]

    return df

X_processed = preprocess_new_data(new_df.copy(), feature_columns)

# --- 4. Scale Features ---
# Apply the saved StandardScaler
X_scaled = scaler.transform(X_processed)
X_scaled_df = pd.DataFrame(X_scaled, columns=X_processed.columns)

# --- 5. Make Prediction ---
prediction_encoded = model.predict(X_scaled_df)
prediction_label = le.inverse_transform(prediction_encoded)

# --- 6. Display Result ---
print("\n--- Custom Inputs & Predictions ---")
print("Input Profiles:")
print(new_df.to_markdown(index=False))

print(f"\nModel Used: **{model.__class__.__name__}**")

results = pd.DataFrame({
    'Input Row': new_df.index + 1,
    'Predicted Disorder': prediction_label
})
print("\nPredictions:")
print(results.to_markdown(index=False))
