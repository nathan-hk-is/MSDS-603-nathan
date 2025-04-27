from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import pandas as pd
import mlflow
from metaflow import metadata
metadata('local')
from metaflow import FlowSpec, step, Flow
from typing import List, Dict, Any

app = FastAPI(
    title='Cat predictor',
    description='Make predictions related to cats.',
    version='0.1',
)

# Defining path operation for root endpoint
@app.get('/')
def main():
    return {'message': 'This is a model for classifying cats'}

class PredictRequest(BaseModel):
    rows: List[Dict[str, Any]]

@app.on_event('startup')
def load_artifacts():
    global model_pipeline
    run = Flow('TrainFlow').latest_run
    model_pipeline = run['end'].task.data.model


# Defining path operation for /predict endpoint
@app.post('/predict')
def predict(request: PredictRequest):
    df = pd.DataFrame(request.rows)
    preds = model_pipeline.predict(df)
    return {'Predictions': preds.tolist()}
