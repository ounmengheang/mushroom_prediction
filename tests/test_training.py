"""The training script produces a useful mushroom classifier."""

import joblib


def test_mushroom_model(models_dir):
    bundle = joblib.load(models_dir / "mushroom.joblib")
    assert bundle["metrics"]["roc_auc"] > 0.95
    assert bundle["metrics"]["accuracy"] > 0.95
    assert bundle["target"] == "class"
    assert bundle["labels"] == ["e", "p"]
    assert len(bundle["features"]) == 22
