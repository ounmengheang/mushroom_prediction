from __future__ import annotations

import pandas as pd
import pytest

import train_mushroom
from app import ml


@pytest.fixture(scope="session")
def models_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("models")
    train_mushroom.train(models_dir=path, verbose=False)
    return path


@pytest.fixture(scope="session")
def example_rows():
    frame = pd.read_csv(train_mushroom.DATA, dtype=str)
    edible = frame.loc[frame["class"] == "e", train_mushroom.FEATURES].iloc[0].to_dict()
    poisonous = frame.loc[frame["class"] == "p", train_mushroom.FEATURES].iloc[0].to_dict()
    return edible, poisonous


@pytest.fixture
def client(models_dir, monkeypatch):
    from fastapi.testclient import TestClient

    from app.main import app

    monkeypatch.setattr(ml, "MODELS_DIR", models_dir)
    with TestClient(app) as test_client:
        yield test_client
