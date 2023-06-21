from os import name
from fastapi import FastAPI
import pandas as pd
import uvicorn
from pycaret.classification import load_model, predict_model

app = FastAPI()

model = load_model("src/data/saved_models/model")
print(model)
model_dict = model.__dict__

print("---------------------------------------------")
print(type(model_dict))
print(model_dict["_feature_names_in"])
print("---------------------------------------------")
# print(model.__class__)
# print(model)
# print(model)
# print(model.__str__())
# print(type(model))
@app.post("/new_ml_model")
def new_ml_model(file: str):
    file = 'src/data/saved_models/model'
    globals()["model"] = load_model(file)
    return {"Ended": True}



if __name__ == "__main__":
    uvicorn.run("api:app", port=5000, reload=True, reload_excludes="*.pkl")
