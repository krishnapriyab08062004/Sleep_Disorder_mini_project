
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import learning_curve, StratifiedKFold
import joblib
import warnings

# -----------------------------
# Suppress warnings
# -----------------------------
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# -----------------------------
# 1️⃣ Load Data
# -----------------------------
try:
    X_train = pd.read_csv('X_train_clean.csv')
    y_train = pd.read_csv('y_train_clean.csv')['Sleep Disorder'].values
    X_test = pd.read_csv('X_test_clean.csv')
    y_test = pd.read_csv('y_test_clean.csv')['Sleep Disorder'].values
    print("✅ Training and Test Data Loaded")
except FileNotFoundError:
    print("Error: Required files not found. Please run clean_code.py first.")
    exit()

# -----------------------------
# 2️⃣ Train Models
# -----------------------------
print("\nTraining Support Vector Machine (SVC)...")
svm_model = SVC(kernel='rbf', random_state=42, probability=True)
svm_model.fit(X_train, y_train)

print("Training Gradient Boosting Classifier (GBM)...")
gbm_model = GradientBoostingClassifier(
    n_estimators=150, learning_rate=0.1, max_depth=3, subsample=0.8, random_state=42
)
gbm_model.fit(X_train, y_train)

# -----------------------------
# 3️⃣ Evaluate Models on Test Data
# -----------------------------
models = {'SVM': svm_model, 'Gradient Boosting': gbm_model}
results = []

print("\n--- Model Evaluation on Test Data ---")
for name, model in models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    results.append({'Model': name, 'Accuracy': accuracy, 'F1-Score': f1})
    print(f"{name} - Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")

results_df = pd.DataFrame(results)

# -----------------------------
# 4️⃣ Select Best Model
# -----------------------------
best_model_name = results_df.loc[results_df['F1-Score'].idxmax(), 'Model']
best_model = models[best_model_name]

# Save the best model
joblib.dump(best_model, 'best_model.pkl')
print(f"\n✅ Best model selected and saved: {best_model_name} -> 'best_model.pkl'")

# -----------------------------
# 5️⃣ Training and Testing Accuracy
# -----------------------------
train_pred = best_model.predict(X_train)
test_pred = best_model.predict(X_test)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)

print("\n--- Accuracy Summary ---")
print(f"Training Accuracy : {train_acc*100:.2f}%")
print(f"Testing Accuracy  : {test_acc*100:.2f}%")

# -----------------------------
# 6️⃣ Actual vs Predicted Table
# -----------------------------
results_table = pd.DataFrame({
    'Actual': y_test,
    'Predicted': test_pred
})
print("\n--- Actual vs Predicted ---")
print(results_table.to_markdown(index=False))

# -----------------------------
# 7️⃣ Classification Report & Confusion Matrix
# -----------------------------
print("\n--- Classification Report ---")
print(classification_report(y_test, test_pred))

cm = confusion_matrix(y_test, test_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -----------------------------
# 8️⃣ Learning Curve (Corrected)
# -----------------------------
from sklearn.model_selection import StratifiedKFold, learning_curve

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

train_sizes, train_scores, test_scores = learning_curve(
    best_model,
    X_train,
    y_train,
    cv=cv,
    train_sizes=np.linspace(0.2, 1.0, 5),
    scoring='accuracy',
    n_jobs=-1,
    shuffle=True,           # ensures random distribution
    random_state=42
)

# Mean and std
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(8,6))
plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Accuracy (CV)')
plt.plot(train_sizes, test_mean, 'o-', color='green', label='Validation Accuracy (CV)')
plt.axhline(y=train_acc, color='blue', linestyle='--', label='Full Train Accuracy')  # optional
plt.axhline(y=test_acc, color='green', linestyle='--', label='Full Test Accuracy')   # optional
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='green')
plt.title(f"Learning Curve: {best_model_name}")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.grid(True)
plt.show()
