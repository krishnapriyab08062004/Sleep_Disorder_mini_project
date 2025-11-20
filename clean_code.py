import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

# NOTE: You must install the 'imbalanced-learn' package for SMOTE: pip install imbalanced-learn

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("cleaned_sleep_data.csv")
print("✅ Dataset Loaded")

# Drop Person ID as it's a unique identifier, not a feature
df = df.drop('Person ID', axis=1)

# --- 1. Data Cleaning & Transformation ---


# 2. Fix BMI Category (Unify 'Normal Weight' and 'Normal')
df['BMI Category'] = df['BMI Category'].replace('Normal Weight', 'Normal')

# -------------------------------
# Handle Missing Values (Check only, as data is generally clean)
# -------------------------------
if df.isnull().sum().any():
    print("Warning: Missing values found. Simple imputation applied.")
    # Simple Imputation (optional, only if needed)
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].median())

# -------------------------------
# Handle Outliers (IQR Capping)
# -------------------------------

def cap_outliers_iqr(df, column):
    """Caps outliers using the 1.5 * IQR rule."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
    df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    return df

# Identify numerical columns for capping
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()

for col in numerical_cols:
    df = cap_outliers_iqr(df.copy(), col)


# -------------------------------
# Encode Categorical Variables
# -------------------------------

# 1. Label encode Sleep Disorder (Target Variable)
le = LabelEncoder()
df['Sleep Disorder'] = le.fit_transform(df['Sleep Disorder'])
joblib.dump(le, "label_encoder.pkl") # Save encoder

# 2. One-Hot Encode remaining categorical features
categorical_cols = df.select_dtypes(include='object').columns.tolist()
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)


# -------------------------------
# Feature Scaling (Standardization/Normalization)
# -------------------------------

scaler = StandardScaler()
X = df.drop('Sleep Disorder', axis=1)
y = df['Sleep Disorder']
# Fit and transform features, save scaler
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
joblib.dump(scaler, "scaler.pkl")
# Save feature column order for model deployment
X.columns.to_series().to_csv('X_full_columns.csv', index=False, header=False)

# -------------------------------
# Split Dataset
# -------------------------------
X_train_raw, X_test, y_train_raw, y_test = train_test_split(
    X_scaled_df, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# Handle Class Imbalance with SMOTE (on training data only)
# -------------------------------

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_raw, y_train_raw)


# -------------------------------
# Save preprocessed data
# -------------------------------
X_train_smote.to_csv("X_train_clean.csv", index=False)
X_test.to_csv("X_test_clean.csv", index=False)
y_train_smote.to_csv("y_train_clean.csv", index=False, header=True)
y_test.to_csv("y_test_clean.csv", index=False, header=True)
print("  X_train_smote shape:", X_train_smote.shape)
print("  y_train_smote shape:", y_train_smote.shape)
print("\n✅ Preprocessing Complete")
print(f"Train shape after SMOTE: {X_train_smote.shape}, Test shape: {X_test.shape}")
print(f"Class distribution after SMOTE:\n{y_train_smote.value_counts()}")
print("\nProcessed data and encoders saved to files: X_train_clean.csv, X_test_clean.csv, y_train_clean.csv, y_test_clean.csv, scaler.pkl, label_encoder.pkl, X_full_columns.csv")