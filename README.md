# Mushroom Classification API

Train a binary classifier that predicts whether a mushroom is **edible (`e`)** or
**poisonous (`p`)**, then serve it as a FastAPI application.

The project follows the classic **train offline, serve online** split:

```
TRAINING (offline)                          SERVING (online)
──────────────────                          ─────────────────
mushrooms.csv                               FastAPI starts
     ↓                                           ↓
one-hot encode categories                   lifespan loads models/mushroom.joblib
     ↓                                           ↓
logistic regression                         model kept in memory
     ↓                                           ↓
joblib.dump → models/mushroom.joblib        POST /mushroom/predict → JSON
```

## Dataset

`mushrooms.csv` contains **8,124 rows** and **23 columns**:

- `class` — the target: `e` (edible, 4,208 rows) or `p` (poisonous, 3,916 rows)
- 22 categorical features: `cap-shape`, `cap-surface`, `cap-color`, `bruises`, `odor`,
  `gill-attachment`, `gill-spacing`, `gill-size`, `gill-color`, `stalk-shape`, `stalk-root`,
  `stalk-surface-above-ring`, `stalk-surface-below-ring`, `stalk-color-above-ring`,
  `stalk-color-below-ring`, `veil-type`, `veil-color`, `ring-number`, `ring-type`,
  `spore-print-color`, `population`, `habitat`

Data notes:

- There are no empty cells, but `stalk-root` contains **2,480 `?` values**. They are treated as
  just another category by the encoder.
- `veil-type` has only one value (`p`) in the whole file, so it carries no information. It is kept
  for schema completeness.

## Project layout

```
mushroom_prediction/
├── mushrooms.csv              # dataset
├── train_mushroom.py          # training script, saves the model bundle
├── models/                    # generated artifacts (git-ignored)
│   └── mushroom.joblib
├── app/
│   ├── main.py                # FastAPI app, lifespan, routes
│   ├── ml.py                  # model loading and prediction logic
│   └── schemas.py             # Pydantic request/response models
├── notebooks/
│   └── 1_mushroom_model.ipynb # quick exploration / training demo
└── tests/
    ├── conftest.py            # trains a throwaway model, injects a TestClient
    ├── test_api.py            # API contract tests
    └── test_training.py       # model quality tests
```

## Requirements

Python 3.10+ and the following packages:

```powershell
pip install fastapi uvicorn pandas scikit-learn joblib pytest httpx
```

## Train

```powershell
python train_mushroom.py
```

This reads `mushrooms.csv`, splits it 80/20 with `stratify` on `class`, fits a
`OneHotEncoder` + `LogisticRegression` pipeline, evaluates it on the held-out set, and writes
`models/mushroom.joblib`.

The saved file is a **bundle dict**, not just the estimator:

| Key | Meaning |
|---|---|
| `model` | the fitted scikit-learn `Pipeline` |
| `features` | the 22 feature names, in training order |
| `target` | `"class"` |
| `labels` | `["e", "p"]` |
| `positive_label` | `"p"` |
| `metrics` | accuracy, macro F1, ROC-AUC on the test split |
| `algorithm` | human-readable description |
| `trained_at` | UTC timestamp |
| `sklearn_version` | version used at training time |

Optional flags: `--models-dir PATH` and `--data-path PATH`.

## Run the API

```powershell
fastapi dev app/main.py
```

or, equivalently:

```powershell
uvicorn app.main:app --reload
```

Interactive documentation is available at `http://127.0.0.1:8000/docs` (and `/redoc`).

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | small landing message with links |
| `GET` | `/health` | liveness check and whether the model is loaded |
| `GET` | `/models` | algorithm, training timestamp and metrics for loaded models |
| `POST` | `/mushroom/predict` | classify one mushroom |

`POST /mushroom/predict` expects a JSON object using the **CSV column names**:

```json
{
  "cap-shape": "x",
  "cap-surface": "s",
  "cap-color": "n",
  "bruises": "t",
  "odor": "p",
  "gill-attachment": "f",
  "gill-spacing": "c",
  "gill-size": "n",
  "gill-color": "k",
  "stalk-shape": "e",
  "stalk-root": "e",
  "stalk-surface-above-ring": "s",
  "stalk-surface-below-ring": "s",
  "stalk-color-above-ring": "w",
  "stalk-color-below-ring": "w",
  "veil-type": "p",
  "veil-color": "w",
  "ring-number": "o",
  "ring-type": "p",
  "spore-print-color": "k",
  "population": "s",
  "habitat": "u"
}
```

Example response:

```json
{
  "label": "p",
  "class_name": "poisonous",
  "confidence": 0.9859,
  "probabilities": { "e": 0.0141, "p": 0.9859 }
}
```

### Error behavior

- **422** — a required field is missing, an unknown field is supplied (`extra="forbid"`), or a
  value is empty. Validation happens before the model runs.
- **503** — the model file is missing. The response tells you to run `python train_mushroom.py`.

### Model path

`app/ml.py` loads from `models/mushroom.joblib` by default. Set the `MODELS_DIR` environment
variable to load from a different directory (the tests use this).

If the installed scikit-learn version differs from `sklearn_version` in the bundle, loading
succeeds but emits a warning suggesting a retrain.

## Test

```powershell
pytest
```

The suite trains its own model into a temporary directory, so it never depends on or overwrites
the checked-in artifact. It covers:

- model quality (`accuracy` and `roc_auc` above 0.95, 22 features, correct labels)
- a real CSV row producing a valid prediction
- unknown columns being rejected with 422

## Design notes

- **Load once, not per request.** The model is loaded in FastAPI's `lifespan` startup hook and
  reused for every request. Reading the file on each call would be unnecessary I/O.
- **Plain `def` routes.** scikit-learn inference is CPU-bound, so sync handlers are appropriate;
  FastAPI runs them in a threadpool instead of blocking the event loop.
- **Save metadata with the model.** The bundle carries feature order, labels and metrics, which a
  bare estimator cannot provide.
- **Validate at the boundary.** Pydantic schemas reject malformed input before it reaches pandas
  or the model.
