"""Train the mushroom model once into a temporary folder for the test session."""

import pytest

import train_mushroom
from app import ml


@pytest.fixture(scope="session")
def models_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("models")
    train_mushroom.train(models_dir=path, verbose=False)
    return path


@pytest.fixture
def client(models_dir, monkeypatch):
    from fastapi.testclient import TestClient

    from app.main import app

    monkeypatch.setattr(ml, "MODELS_DIR", models_dir)
    with TestClient(app) as test_client:  # `with` runs the lifespan: the models are loaded
        yield test_client
