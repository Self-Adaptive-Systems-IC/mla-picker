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
input_model = create_model("api_dataset_edit_input", **{'type_of_failure': 5.0, 'time_repair': 0.19300000369548798, 'cost': 0.4480000138282776, 'criticality': 0.26100000739097595, 'humid': 79.0, 'temp': 62.0})
output_model = create_model("api_dataset_edit_output", **{"prediction":0, "score":0.8})


# Define predict function
@app.post("/predict", response_model=output_model)
def predict(data: input_model):
    data = pd.DataFrame([data.dict()])
    predictions = predict_model(model, data=data)
    return {"prediction": predictions["prediction_label"].iloc[0], "score": predictions["prediction_score"].iloc[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
