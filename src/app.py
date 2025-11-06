from fastapi import FastAPI
from pydantic import BaseModel
import joblib, pandas as pd

app = FastAPI(title='Churn Light MLOps Demo')

class Features(BaseModel):
    __root__: dict

MODEL = None

@app.on_event("startup")
def load_model():
    global MODEL
    MODEL = joblib.load("model/churn_model.pkl")

@app.post("/predict")
def predict(payload: Features):
    data = payload.__root__
    X = pd.DataFrame([data])
    pred = MODEL.predict(X)[0]
    return {"churn_prediction": int(pred)}
