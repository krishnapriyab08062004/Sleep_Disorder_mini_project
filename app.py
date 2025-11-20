
# from flask import Flask, render_template, request
# import pandas as pd
# import numpy as np
# import joblib

# app = Flask(__name__)

# # -----------------------------------------------------
# # Load model, scaler, encoders, and expected columns
# # -----------------------------------------------------
# model = joblib.load('best_model.pkl')
# scaler = joblib.load('scaler.pkl')
# label_encoder = joblib.load('label_encoder.pkl')
# feature_columns = pd.read_csv('X_full_columns.csv', header=None)[0].tolist()

# NUMERIC_COLUMNS = [
#     'Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level',
#     'Stress Level', 'Heart Rate', 'Daily Steps', 'Systolic', 'Diastolic'
# ]
# CATEGORICAL_COLUMNS = ['Gender', 'Occupation', 'BMI Category']

# # Inspirational quotes
# QUOTES = {
#     'Insomnia': "Rest is not idleness... – John Lubbock",
#     'Sleep Apnea': "Sleep is the golden chain... – Thomas Dekker",
#     'None': "A good laugh and a long sleep are the best cures. – Irish Proverb",
#     'Normal': "Sleep is the best meditation. – Dalai Lama"
# }

# @app.route('/')
# def home():
#     return render_template('intro.html')

# @app.route('/classify')
# def classify():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.form.to_dict()

#         # Create base dataframe
#         input_df = pd.DataFrame(0, index=[0], columns=feature_columns)

#         # Map field names (BP fields)
#         if 'Systolic BP' in data:
#             data['Systolic'] = data.pop('Systolic BP')
#         if 'Diastolic BP' in data:
#             data['Diastolic'] = data.pop('Diastolic BP')

#         # Fill numeric fields
#         for col in NUMERIC_COLUMNS:
#             if col in data:
#                 input_df[col] = float(data[col])

#         # Fill categorical (one-hot)
#         for cat_col in CATEGORICAL_COLUMNS:
#             value = data.get(cat_col)
#             if value:
#                 if cat_col == 'BMI Category' and value == 'Normal Weight':
#                     value = 'Normal'
#                 dummy_col = f"{cat_col}_{value}"
#                 if dummy_col in input_df.columns:
#                     input_df[dummy_col] = 1

#         # Align column order
#         input_df = input_df[feature_columns]

#         # Scale input
#         input_scaled = scaler.transform(input_df)

#         # Predict
#         pred_encoded = model.predict(input_scaled)[0]
#         prediction = label_encoder.inverse_transform([pred_encoded])[0]

#         # Quote
#         quote = QUOTES.get(prediction, QUOTES['Normal'])

#         return render_template('result.html', prediction=prediction, quote=quote)

#     except Exception as e:
#         print("\n--- ERROR ---")
#         print(str(e))
#         print("-------------")
#         return render_template('result.html', prediction='Error', quote=f"An error occurred: {str(e)}")

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import numpy as np
import joblib
from functools import wraps

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'  # Change this to a random string

# -----------------------------------------------------
# Load model, scaler, encoders, and expected columns
# -----------------------------------------------------
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')
feature_columns = pd.read_csv('X_full_columns.csv', header=None)[0].tolist()

# ------------------------------
# Compute Feature Importance
# ------------------------------
feature_importance = {}

# 1) Tree-based models (has .feature_importances_)
if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
    feature_importance = dict(zip(feature_columns, importances))

# 2) Linear models (has .coef_)
elif hasattr(model, 'coef_'):
    importances = np.abs(model.coef_)[0]
    feature_importance = dict(zip(feature_columns, importances))

# 3) Fallback to zero importance if model has none
else:
    feature_importance = {col: 0 for col in feature_columns}


NUMERIC_COLUMNS = [
    'Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level',
    'Stress Level', 'Heart Rate', 'Daily Steps', 'Systolic', 'Diastolic'
]
CATEGORICAL_COLUMNS = ['Gender', 'Occupation', 'BMI Category']

# Inspirational quotes
QUOTES = {
    'Insomnia': "Rest is not idleness... – John Lubbock",
    'Sleep Apnea': "Sleep is the golden chain... – Thomas Dekker",
    'None': "A good laugh and a long sleep are the best cures. – Irish Proverb",
    'Normal': "Sleep is the best meditation. – Dalai Lama"
}

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def home():
    # If user is logged in, redirect to intro page
    if 'username' in session:
        return redirect(url_for('intro'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        if username:
            session['username'] = username
            return redirect(url_for('intro'))
        else:
            return render_template('login.html', error='Please enter your name')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/intro')
@login_required
def intro():
    return render_template('intro.html', username=session['username'])

@app.route('/classify')
@login_required
def classify():
    return render_template('index.html', username=session['username'])
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm = request.form.get('confirm_password')

        if password != confirm:
            return render_template('register.html', error="Passwords do not match")

        # In your case, no database → simply proceed to login
        session['username'] = username
        return redirect(url_for('intro'))

    return render_template('register.html')

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        data = request.form.to_dict()

        # Create base dataframe
        input_df = pd.DataFrame(0, index=[0], columns=feature_columns)

        # Map field names (BP fields)
        if 'Systolic BP' in data:
            data['Systolic'] = data.pop('Systolic BP')
        if 'Diastolic BP' in data:
            data['Diastolic'] = data.pop('Diastolic BP')
 
        # Fill numeric fields
        for col in NUMERIC_COLUMNS:
            if col in data:
                input_df[col] = float(data[col])

        # Fill categorical (one-hot)
        for cat_col in CATEGORICAL_COLUMNS:
            value = data.get(cat_col)
            if value:
                if cat_col == 'BMI Category' and value == 'Normal Weight':
                    value = 'Normal'
                dummy_col = f"{cat_col}_{value}"
                if dummy_col in input_df.columns:
                    input_df[dummy_col] = 1

        # Align column order
        input_df = input_df[feature_columns]

        # Scale input
        input_scaled = scaler.transform(input_df)

        # Predict
        pred_encoded = model.predict(input_scaled)[0]
        prediction = label_encoder.inverse_transform([pred_encoded])[0]

        # Quote
        quote = QUOTES.get(prediction, QUOTES['Normal'])

        return render_template('result.html', prediction=prediction, quote=quote, username=session['username'],feature_importance=feature_importance)

    except Exception as e:
        print("\n--- ERROR ---")
        print(str(e))
        print("-------------")
        return render_template('result.html', prediction='Error', quote=f"An error occurred: {str(e)}", username=session['username'])

if __name__ == '__main__':
    app.run(debug=True)