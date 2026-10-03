# Data

This project uses the **Cardiovascular Disease dataset** (70,000 patient records) published on Kaggle:
https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset

The CSV is not included in this repo. To retrain the models:

1. Download `cardio_train.csv` from the link above
2. Put it in this folder: `data/cardio_train.csv`
3. From the project root, run `python -m bp_model.train`

The trained models in `models/` are already included, so you only need the data to retrain. The web app and tests work without it.

## Columns

| Column | Meaning |
|---|---|
| `age` | Age in days (converted to years during cleaning) |
| `gender` | 1 = female, 2 = male |
| `height` | cm |
| `weight` | kg |
| `ap_hi` / `ap_lo` | Systolic / diastolic blood pressure, mmHg |
| `cholesterol`, `gluc` | 1 = normal, 2 = above normal, 3 = well above normal |
| `smoke`, `alco`, `active` | 0 / 1 |
| `cardio` | Cardiovascular disease diagnosis (not used as a feature) |
