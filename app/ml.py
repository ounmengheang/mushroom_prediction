"""Load the trained model and make mushroom predictions."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import joblib
import pandas as pd
import sklearn

MODELS_DIR = Path(os.environ.get("MODELS_DIR", Path(__file__).parent.parent / "models"))


def load(name: str = "mushroom") -> dict | None:
    """Return a trusted saved model bundle, or None when it has not been trained."""
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        return None

    bundle = joblib.load(path)
    if bundle.get("sklearn_version") != sklearn.__version__:
        warnings.warn(
            f"{path.name} was trained with scikit-learn {bundle.get('sklearn_version')} but "
            f"{sklearn.__version__} is installed; retrain it if predictions look wrong.",
            stacklevel=2,
        )
    return bundle


def predict_mushrooms(bundle: dict, mushrooms: list[dict]) -> list[dict]:
    """Predict one or more mushrooms using poisonous-class probability."""
    frame = pd.DataFrame(mushrooms, columns=bundle["features"])
    probabilities = bundle["model"].predict_proba(frame)[:, 1]
    threshold = float(bundle["threshold"])

    predictions = []
    for probability in probabilities:
        poisonous_probability = float(probability)
        poisonous = poisonous_probability >= threshold
        predictions.append(
            {
                "classification": "poisonous" if poisonous else "edible",
                "poisonous_probability": round(poisonous_probability, 4),
                "confidence": round(
                    poisonous_probability if poisonous else 1 - poisonous_probability, 4
                ),
                "threshold": threshold,
            }
        )
    return predictions
