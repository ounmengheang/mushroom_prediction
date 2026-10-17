"""Load the trained models and make predictions with them."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import joblib
import pandas as pd
import sklearn

# Where train_mushroom.py saves the model (override for tests).
MODELS_DIR = Path(os.environ.get("MODELS_DIR", Path(__file__).parent.parent / "models"))


def load(name: str) -> dict | None:
    """The saved bundle (model + metadata), or None if that model was not trained yet."""
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        return None
    bundle = joblib.load(path)  # only load model files you trust: joblib can run code
    if bundle.get("sklearn_version") != sklearn.__version__:
        warnings.warn(
            f"{path.name} was trained with scikit-learn {bundle.get('sklearn_version')} but "
            f"{sklearn.__version__} is installed: retrain it if predictions look wrong.",
            stacklevel=2,
        )
    return bundle


def predict_mushroom(bundle: dict, mushrooms: list[dict]) -> list[dict]:
    frame = pd.DataFrame(mushrooms, columns=bundle["features"])
    model = bundle["model"]
    labels = list(model.classes_)
    out = []
    for row in model.predict_proba(frame):
        probabilities = {label: round(float(p), 4) for label, p in zip(labels, row, strict=True)}
        label = max(probabilities, key=probabilities.get)
        class_name = "poisonous" if label == "p" else "edible"
        out.append(
            {"label": label, "class_name": class_name, "confidence": probabilities[label],
             "probabilities": probabilities}
        )
    return out
