from pathlib import Path
from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).parent / "model.joblib"
model = joblib.load(MODEL_PATH)
FEATURES = list(model.feature_names_in_)

app = FastAPI(title="Predicción de falla cardíaca")


class Paciente(BaseModel):
    Age: int = Field(ge=1, le=120)
    Sex: Literal["M", "F"]
    ChestPainType: Literal["ATA", "NAP", "ASY", "TA"]
    RestingBP: Optional[float] = Field(default=None, ge=0, le=300)
    Cholesterol: Optional[float] = Field(default=None, ge=0, le=700)
    FastingBS: Literal[0, 1]
    RestingECG: Literal["Normal", "ST", "LVH"]
    MaxHR: int = Field(ge=30, le=250)
    ExerciseAngina: Literal["Y", "N"]
    Oldpeak: float = Field(ge=-5, le=10)
    ST_Slope: Literal["Up", "Flat", "Down"]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(paciente: Paciente):
    datos = paciente.model_dump()
    for col in ["RestingBP", "Cholesterol"]:
        if datos[col] in (None, 0):
            datos[col] = np.nan
    df = pd.DataFrame([datos])[FEATURES]
    proba = float(model.predict_proba(df)[0, 1])
    return {"probabilidad_enfermedad": round(proba, 4),
            "prediccion": int(proba >= 0.5)}
