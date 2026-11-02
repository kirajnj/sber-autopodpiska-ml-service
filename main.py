import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

pipeline = joblib.load('pipeline.pkl')


class VisitData(BaseModel):
    visit_number: int
    screen_width: int
    screen_height: int
    visit_hour: int
    is_weekend: int
    utm_source: str
    utm_medium: str
    utm_campaign: str
    utm_adcontent: str
    device_category: str
    device_brand: str
    device_browser: str
    geo_country: str
    geo_city: str


class PredictionResponse(BaseModel):
    prediction: int
    probability: float


@app.post('/predict', response_model=PredictionResponse)
def predict(data: VisitData):
    df = pd.DataFrame([data.dict()])
    prob = pipeline.predict_proba(df)[0][1]
    pred = int(prob >= 0.5)
    return PredictionResponse(prediction=pred, probability=round(prob, 4))


@app.get('/status')
def status():
    return "I'm OK"


@app.get('/version')
def version():
    return "Random forest for SberAuto "