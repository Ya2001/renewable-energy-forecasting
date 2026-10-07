"""FastAPI service: estimate hourly wind or solar production from weather values.

Feed it forecast weather (for example from a numerical weather prediction) to get a
production forecast. The model maps weather in the same hour to production for the
French grid; it does not forecast the weather itself.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
FEATURES = ["ALLSKY_SFC_SW_DWN", "ALLSKY_SFC_SW_DNI", "ALLSKY_SFC_SW_DIFF", "T2M", "WS10M"]

app = FastAPI(
    title="Renewable production model",
    version="1.0.0",
    description="Hourly wind and solar production (MWh, French grid) from weather inputs.",
)


class Weather(BaseModel):
    ALLSKY_SFC_SW_DWN: float = Field(..., ge=0, le=1500, description="Shortwave downward irradiance, Wh/m^2")
    ALLSKY_SFC_SW_DNI: float = Field(..., ge=0, le=1500, description="Direct normal irradiance, Wh/m^2")
    ALLSKY_SFC_SW_DIFF: float = Field(..., ge=0, le=1000, description="Diffuse irradiance, Wh/m^2")
    T2M: float = Field(..., ge=-40, le=55, description="Temperature at 2 m, degrees C")
    WS10M: float = Field(..., ge=0, le=60, description="Wind speed at 10 m, m/s")


class PredictRequest(BaseModel):
    source: Literal["wind", "solar"]
    weather: Weather


class PredictResponse(BaseModel):
    source: str
    predicted_mwh: float
    model: str


@lru_cache(maxsize=2)
def load_model(source: str):
    path = ARTIFACTS / f"{source}_model.joblib"
    if not path.exists():
        raise FileNotFoundError(path)
    return joblib.load(path)


def model_info() -> dict:
    path = ARTIFACTS / "model_info.json"
    return json.loads(path.read_text()) if path.exists() else {}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/model-info")
def info() -> dict:
    """Which model serves each source and how it scored on held-out data."""
    return model_info()


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    try:
        model = load_model(request.source)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="model artifact missing; run scripts.train_final")
    frame = pd.DataFrame([request.weather.model_dump()])[FEATURES]
    value = float(model.predict(frame)[0])
    name = model_info().get(request.source, {}).get("model", "unknown")
    return PredictResponse(source=request.source, predicted_mwh=round(max(value, 0.0), 1), model=name)
