"""FastAPI service for the trained mushroom classifier."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app import ml
from app.schemas import ModelInfo, Mushroom, MushroomBatch, MushroomPrediction

MODEL_NAME = "mushroom"
MODEL: dict | None = None
HOW_TO_TRAIN = "python train_mushroom.py"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL
    MODEL = ml.load(MODEL_NAME)
    yield
    MODEL = None


app = FastAPI(
    title="Mushroom Classification API",
    description=(
        "Educational classifier for the supplied categorical mushroom dataset. "
        "Never use this API to decide whether a real mushroom is safe to eat."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def loaded_model() -> dict:
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail=f"The model is not trained yet: run `{HOW_TO_TRAIN}`.",
        )
    return MODEL


@app.get("/", include_in_schema=False)
def home() -> dict:
    return {
        "message": "Mushroom Classification API",
        "docs": "/docs",
        "health": "/health",
        "warning": "Never use this model to decide whether a real mushroom is safe to eat.",
    }


@app.get("/health", tags=["service"])
def health() -> dict:
    return {"status": "ok", "models": {MODEL_NAME: MODEL is not None}}


@app.get("/models", response_model=list[ModelInfo], tags=["service"])
def models() -> list[dict]:
    if MODEL is None:
        return [{"name": MODEL_NAME, "loaded": False}]
    return [
        {
            "name": MODEL_NAME,
            "loaded": True,
            "algorithm": MODEL["algorithm"],
            "trained_at": MODEL["trained_at"],
            "metrics": MODEL["metrics"],
        }
    ]


@app.post("/mushroom/predict", response_model=MushroomPrediction, tags=["mushroom"])
def predict_mushroom(mushroom: Mushroom) -> dict:
    row = mushroom.model_dump(by_alias=True)
    return ml.predict_mushrooms(loaded_model(), [row])[0]


@app.post(
    "/mushroom/predict/batch",
    response_model=list[MushroomPrediction],
    tags=["mushroom"],
)
def predict_mushroom_batch(batch: MushroomBatch) -> list[dict]:
    rows = [mushroom.model_dump(by_alias=True) for mushroom in batch.mushrooms]
    return ml.predict_mushrooms(loaded_model(), rows)

