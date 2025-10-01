import os

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ml.data import apply_label, process_data
from ml.model import inference, load_model


# DO NOT MODIFY
class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(
        ..., example="Married-civ-spouse", alias="marital-status"
    )
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")


# --- Load saved artifacts (model, encoder, lb) ---
# Using paths relative to this file's directory
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR = os.path.join(_PROJECT_DIR, "model")

_MODEL_PATH = os.path.join(_MODEL_DIR, "model.pkl")
_ENCODER_PATH = os.path.join(_MODEL_DIR, "encoder.pkl")
_LB_PATH = os.path.join(_MODEL_DIR, "lb.pkl")

# Your load_model likely returns (model, encoder, lb) when given all three paths.
# If your implementation differs, adjust accordingly.
model, encoder, lb = load_model(_MODEL_PATH, _ENCODER_PATH, _LB_PATH)

app = FastAPI(title="Income Prediction API", version="1.0.0")


@app.get("/")
async def get_root():
    """Say hello!"""
    return {"message": "Hello from the API!"}


@app.post("/data/")
def post_inference(data: Data):
    try:
        data_dict = data.dict()
        data_df = {k.replace("_", "-"): [v] for k, v in data_dict.items()}
        data_df = pd.DataFrame.from_dict(data_df)

        cat_features = [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ]

        X, _, _, _ = process_data(
            data_df,
            categorical_features=cat_features,
            label=None,
            training=False,
            encoder=encoder,
            lb=lb,  # keep if your process_data expects it
        )

        pred = inference(model, X)
        return {"result": apply_label(pred)}  # <= array, not int
    except Exception as e:
        # Log & surface the real reason
        import sys
        import traceback

        traceback.print_exc(file=sys.stderr)
        raise HTTPException(status_code=500, detail=str(e))
