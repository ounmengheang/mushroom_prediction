"""Train, evaluate, and save the mushroom classifier."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

HERE = Path(__file__).parent
DATA = HERE / "mushrooms.csv"
MODELS = HERE / "models"
TARGET = "class"
FEATURES = [
    "cap-shape",
    "cap-surface",
    "cap-color",
    "bruises",
    "odor",
    "gill-attachment",
    "gill-spacing",
    "gill-size",
    "gill-color",
    "stalk-shape",
    "stalk-root",
    "stalk-surface-above-ring",
    "stalk-surface-below-ring",
    "stalk-color-above-ring",
    "stalk-color-below-ring",
    "veil-color",
    "ring-number",
    "ring-type",
    "spore-print-color",
    "population",
    "habitat",
]


def step(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 60 - len(title)))


def build_model():
    return make_pipeline(
        OneHotEncoder(handle_unknown="ignore"),
        RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
    )


def choose_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    """Prefer high poisonous recall, then choose the best precision among qualifying cutoffs."""
    candidates = []
    for threshold in np.arange(0.1, 0.91, 0.01):
        predictions = probabilities >= threshold
        recall = recall_score(y_true, predictions)
        precision = precision_score(y_true, predictions, zero_division=0)
        if recall >= 0.99:
            candidates.append((precision, threshold))
    return round(float(max(candidates)[1] if candidates else 0.5), 2)


def train(data_path: Path = DATA, models_dir: Path = MODELS, verbose: bool = True) -> dict:
    say = print if verbose else (lambda *args, **kwargs: None)
    show = step if verbose else (lambda title: None)

    show("1. Load and validate the data")
    frame = pd.read_csv(data_path, dtype=str)
    required = set(FEATURES) | {TARGET}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    if not set(frame[TARGET].unique()) <= {"e", "p"}:
        raise ValueError("The class column must contain only 'e' and 'p'.")
    say(f"{len(frame):,} rows, {len(FEATURES)} useful categorical features")
    say("veil-type is excluded because it has only one value")

    X = frame[FEATURES]
    y = frame[TARGET].map({"e": 0, "p": 1})
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    show("2. Choose a safety-oriented threshold on training data")
    model = build_model()
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    training_probabilities = cross_val_predict(
        model, X_train, y_train, cv=folds, method="predict_proba", n_jobs=-1
    )[:, 1]
    threshold = choose_threshold(y_train, training_probabilities)
    say(f"threshold: {threshold:.2f} (selected for at least 99% cross-validated poisonous recall)")

    show("3. Fit and evaluate on the held-out test set")
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = probabilities >= threshold
    metrics = {
        "accuracy": round(accuracy_score(y_test, predictions), 4),
        "precision": round(precision_score(y_test, predictions), 4),
        "recall": round(recall_score(y_test, predictions), 4),
        "f1": round(f1_score(y_test, predictions), 4),
        "roc_auc": round(roc_auc_score(y_test, probabilities), 4),
    }
    for name, value in metrics.items():
        say(f"{name:<10} {value:.4f}")

    show("4. Save the complete model bundle")
    bundle = {
        "model": model,
        "algorithm": "one-hot encoder + random forest",
        "features": FEATURES,
        "labels": {0: "edible", 1: "poisonous"},
        "threshold": threshold,
        "metrics": metrics,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
    }
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / "mushroom.joblib"
    joblib.dump(bundle, path)
    say(f"saved to {path}")
    return bundle


if __name__ == "__main__":
    train()
