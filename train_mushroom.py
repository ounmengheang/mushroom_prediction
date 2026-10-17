"""Train a classifier for the mushroom dataset and save it for the API."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = Path(__file__).parent / "mushrooms.csv"
TARGET = "class"


def train(
    models_dir: str | Path | None = None,
    data_path: str | Path = DATA_PATH,
    verbose: bool = True,
) -> dict:
    """Train the mushroom classifier and return the saved model bundle."""
    data = pd.read_csv(data_path)
    if TARGET not in data.columns:
        raise ValueError(f"Dataset must contain the {TARGET!r} target column")

    features = [column for column in data.columns if column != TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        data[features], data[TARGET], test_size=0.2, random_state=42, stratify=data[TARGET]
    )
    preprocessor = ColumnTransformer(
        [("categorical", OneHotEncoder(handle_unknown="ignore"), features)],
        remainder="drop",
    )
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    pipeline.fit(x_train, y_train)

    probabilities = pipeline.predict_proba(x_test)
    predictions = pipeline.classes_[probabilities.argmax(axis=1)]
    poisonous_index = list(pipeline.classes_).index("p")
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro")),
        "roc_auc": float(roc_auc_score(y_test, probabilities[:, poisonous_index])),
    }
    bundle = {
        "model": pipeline,
        "features": features,
        "target": TARGET,
        "labels": list(pipeline.classes_),
        "positive_label": "p",
        "metrics": metrics,
        "algorithm": "logistic regression with one-hot encoding",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
    }

    destination = Path(models_dir) if models_dir else Path(__file__).parent / "models"
    destination.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, destination / "mushroom.joblib")
    if verbose:
        print(f"Saved mushroom model to {destination / 'mushroom.joblib'}")
        print(metrics)
    return bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    args = parser.parse_args()
    train(models_dir=args.models_dir, data_path=args.data_path)