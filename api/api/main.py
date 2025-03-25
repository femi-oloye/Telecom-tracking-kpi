from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

app = FastAPI()

# Load trained model
model = joblib.load("../ml_model/churn_model.pkl")

@app.post("/predict/")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]  # Get the first prediction

    # Ensure the output is JSON serializable
    if isinstance(prediction, np.generic):  
        prediction = prediction.item()  # Convert NumPy type to Python type

    return {"churn_prediction": str(prediction)}  # Convert output to string
