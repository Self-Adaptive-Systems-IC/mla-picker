# -*- coding: utf-8 -*-

import pandas as pd
from pycaret.classification import load_model, predict_model
from fastapi import FastAPI
import uvicorn
from pydantic import create_model

# Create the app
app = FastAPI()

# Load trained Pipeline
model = load_model("src/api/api_2023-06-13")

# Create input/output pydantic models
input_model = create_model("src/api/api_2023-06-13_input", **{'type_of_failure': 6.0, 'time_repair': 0.2708224356174469, 'cost': 0.6079999804496765, 'criticality': 0.5809999704360962, 'humid': 5.0, 'temp': 78.0})
output_model = create_model("src/api/api_2023-06-13_output", prediction=0)


# Define predict function
@app.post("/predict", response_model=output_model)
def predict(data: input_model):
    data = pd.DataFrame([data.dict()])
    predictions = predict_model(model, data=data)
    return {"prediction": predictions["prediction_label"].iloc[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)