def test_health_and_model_info(client):
    assert client.get("/health").json() == {"status": "ok", "models": {"mushroom": True}}
    model = client.get("/models").json()[0]
    assert model["loaded"] is True
    assert model["metrics"]["recall"] > 0.95


def test_known_edible_and_poisonous_rows(client, example_rows):
    edible, poisonous = example_rows
    edible_result = client.post("/mushroom/predict", json=edible)
    poisonous_result = client.post("/mushroom/predict", json=poisonous)
    assert edible_result.status_code == 200
    assert poisonous_result.status_code == 200
    assert edible_result.json()["classification"] == "edible"
    assert poisonous_result.json()["classification"] == "poisonous"


def test_batch_prediction_preserves_order(client, example_rows):
    edible, poisonous = example_rows
    response = client.post(
        "/mushroom/predict/batch", json={"mushrooms": [edible, poisonous]}
    )
    assert response.status_code == 200
    assert [item["classification"] for item in response.json()] == ["edible", "poisonous"]


def test_invalid_category_and_missing_field_return_422(client, example_rows):
    edible, _ = example_rows
    invalid = edible | {"odor": "invalid"}
    assert client.post("/mushroom/predict", json=invalid).status_code == 422

    missing = edible.copy()
    missing.pop("odor")
    assert client.post("/mushroom/predict", json=missing).status_code == 422


def test_extra_field_returns_422(client, example_rows):
    edible, _ = example_rows
    assert client.post("/mushroom/predict", json=edible | {"class": "e"}).status_code == 422
