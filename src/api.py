from fastapi import FastAPI
import pandas as pd
import uvicorn
from pycaret.classification import load_model, predict_model

app = FastAPI()

@app.post("/new_ml_model")
def new_ml_model(file: str):
    file = 'src/data/saved_models/model'
    globals()["model"] = load_model(file)
    return {"Ended": True}


