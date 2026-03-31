# predictor/views.py

import os
import joblib
import pandas as pd
from django.shortcuts import render

# -------------------------------
# 1️⃣ Define paths and load ML model
# -------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "ml_models", "credit_model.pkl")
scaler_path = os.path.join(BASE_DIR, "ml_models", "scaler.pkl")

# Load model and scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# -------------------------------
# 2️⃣ Prediction Function
# -------------------------------
def predict_credit(data):
    """
    Takes a dictionary input:
    {
        "Income": float,
        "Debt": float,
        "Payment_History": str ("Good"/"Late"/"Very Late"),
        "Employment_Years": float
    }
    Returns: "Creditworthy ✅" or "Not Creditworthy ❌"
    """
    df = pd.DataFrame([data])

    # Encode Payment_History
    df["Payment_History"] = df["Payment_History"].map({
        "Good": 2,
        "Late": 1,
        "Very Late": 0
    })

    # Feature engineering: Debt-to-Income Ratio
    df["DTI"] = df["Debt"] / (df["Income"] + 1)

    # Ensure same columns as training
    feature_columns = ["Income", "Debt", "Payment_History", "Employment_Years", "DTI"]
    df = df.reindex(columns=feature_columns, fill_value=0)

    # Predict using Random Forest
    prediction = model.predict(df)

    return "Creditworthy ✅" if prediction[0] == 1 else "Not Creditworthy ❌"

# -------------------------------
# 3️⃣ Home View
# -------------------------------
def home(request):
    """
    Renders the home page with the prediction form
    """
    return render(request, "form.html")

# -------------------------------
# 4️⃣ Prediction View
# -------------------------------
def predict_view(request):
    """
    Handles POST request from the form,
    calls the prediction function,
    and returns the result to the template
    """
    result = None  # default

    if request.method == "POST":
        try:
            data = {
                "Income": float(request.POST.get("income", 0)),
                "Debt": float(request.POST.get("debt", 0)),
                "Payment_History": request.POST.get("payment_history", "Good"),
                "Employment_Years": float(request.POST.get("employment_years", 0))
            }

            result = predict_credit(data)
        except Exception as e:
            result = f"Error: {str(e)}"

    return render(request, "form.html", {"result": result})