"""The API accepts mushrooms.csv-shaped records and returns a class prediction."""

import pandas as pd


def test_mushroom_prediction(client):
    row = pd.read_csv("mushrooms.csv").drop(columns=["class"]).iloc[0].to_dict()
    response = client.post("/mushroom/predict", json=row)

    assert response.status_code == 200
    body = response.json()
    assert body["label"] in {"e", "p"}
    assert body["class_name"] in {"edible", "poisonous"}
    assert set(body["probabilities"]) == {"e", "p"}


def test_mushroom_prediction_rejects_unknown_columns(client):
    row = pd.read_csv("mushrooms.csv").drop(columns=["class"]).iloc[0].to_dict()
    row["not-a-mushroom-feature"] = "x"

    response = client.post("/mushroom/predict", json=row)

    assert response.status_code == 422