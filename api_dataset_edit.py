# -*- coding: utf-8 -*-

import pandas as pd
from pycaret.classification import load_model, predict_model
from fastapi import FastAPI
import uvicorn
from pydantic import create_model

# Create the app
app = FastAPI()

# Load trained Pipeline
model = load_model("api_dataset_edit")

# Create input/output pydantic models
input_model = create_model("api_dataset_edit_input", **{'type_of_failure': 3.0, 'time_repair': 0.31511762738227844, 'cost': 0.44999998807907104, 'criticality': 0.4129999876022339, 'humid': 21.0, 'temp': 148.0})
output_model = create_model("api_dataset_edit_output", **{"prediction":0, "score":0.8})


# Define predict function
@app.post("/predict", response_model=output_model)
def predict(data: input_model):
    data = pd.DataFrame([data.dict()])
    predictions = predict_model(model, data=data)
    return {"prediction": predictions["prediction_label"].iloc[0], "score": predictions["prediction_score"].iloc[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
