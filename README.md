# Mushroom Classification API

A small project training a categorical mushroom classifier, saves the complete preprocessing/model bundle with joblib, and serve predictions through FastAPI.

> **Safety:** This is an educational model trained on a limited dataset. Never use it to decide
> whether a real mushroom is safe to touch or eat.

## Setup and run

```bash
cd "/Users/vatheka/Desktop/dichi/Techprenuer-2.0/AI Deployment/mushroom-classification-project"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_mushroom.py
fastapi dev app/main.py
```

Open <http://127.0.0.1:8000/docs> for the interactive API documentation.

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Service and model status |
| `GET` | `/models` | Training metadata and held-out metrics |
| `POST` | `/mushroom/predict` | Classify one mushroom |
| `POST` | `/mushroom/predict/batch` | Classify up to 1,000 mushrooms |

The API accepts the letter codes used by `mushrooms.csv`. Ready-to-run examples are in
`requests.http`.

## Tests

```bash
pytest
```

Tests train into a temporary directory, so they do not overwrite the deployed model.
