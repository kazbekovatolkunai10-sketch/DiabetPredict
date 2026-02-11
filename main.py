from fastapi import APIRouter
from pydantic import BaseModel
from joblib import load
import joblib
import os


scaler = joblib.load('scaler.pkl')
model = joblib.load('log_model.pkl')


diabetes_predict = APIRouter(prefix='/predict', tags=['Diabetes Predict'])

class DiabetesPredictSchema(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int


@diabetes_predict.post('/')
async def diabetes_predicted(diabetes: DiabetesPredictSchema):
    diabetes_dict = diabetes.dict()


    features = list(diabetes_dict.values())

    scaled_data = scaler.transform([features])
    diabetes = model.predict_proba(scaled_data)[0]
    probability = float(diabetes[1])
    if probability > 0.5:
        diabetes_label = "Yes"
    else:
        diabetes_label = "No"

    return {
        "diabetes": diabetes_label,
        "probability": round(probability, 2)
    }
