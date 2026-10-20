import joblib


def test_mushroom_model(models_dir):
    bundle = joblib.load(models_dir / "mushroom.joblib")
    assert len(bundle["features"]) == 21
    assert "veil-type" not in bundle["features"]
    assert bundle["metrics"]["accuracy"] > 0.95
    assert bundle["metrics"]["recall"] > 0.95
    assert 0.1 <= bundle["threshold"] <= 0.9
