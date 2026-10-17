# Mushroom Model

This project trains a binary classifier from `mushrooms.csv`. The `class` column is the target:
`e` means edible and `p` means poisonous. The remaining 22 columns are categorical mushroom
features and are one-hot encoded before logistic regression.

## Train

```powershell
python train_mushroom.py
```

The model is saved to `models/mushroom.joblib`.

## Run the API

```powershell
fastapi dev app/main.py
```

Send a record using the same feature names as the CSV to `POST /mushroom/predict`. Interactive
documentation is available at `/docs`.