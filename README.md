# Sri Lanka Food Price Prediction: Random Forest vs XGBoost vs Stacking

**Problem (regression):** predict the price (LKR per unit) of a food item at a Sri Lankan market.
**Dataset:** WFP Food Prices for Sri Lanka (Humanitarian Data Exchange), 32,495 monthly market prices, 2004-2025.
**Models:** Random Forest, XGBoost, and Stacking (RF + XGBoost, Ridge meta-model), all with hand-designed hyperparameters (notebook, section 3).
**Preprocessing:** leakage and redundant column removal, null detection and imputation, one-hot encoding, scaling, log target transform, time-based split (train 2004-2024, test 2025).
**Inference:** one 2025 row is held out before training; `models/model.pkl` is the exported best model.

| Model | RMSE (LKR) | MAE (LKR) | R2 | MAPE % |
|---|---|---|---|---|
| Random Forest | 192.96 | 129.38 | 0.911 | 27.5 |
| XGBoost | 186.79 | 125.55 | 0.917 | 26.9 |
| Stacking | 187.61 | 125.82 | 0.916 | 27.0 |

## Files
- `food_price_analysis.ipynb`: full analysis (EDA, preprocessing, parameters, comparison, strengths/limitations, interpretation, inference)
- `pipeline.py`: shared preprocessing and model definitions
- `train.py`: trains the models and writes `models/*.pkl` and `models/report.json`
- `app.py` + `static/index.html`: FastAPI backend and dashboard

## Notes for GitHub
`models/all_models.pkl` (about 227 MB, holds all three models for the dashboard) is git-ignored because GitHub rejects files over 100 MB. It is recreated by `python train.py`. The small `models/model.pkl` (best model) is committed.

## Run
```
pip install -r requirements.txt
python train.py
uvicorn app:app --reload      # http://127.0.0.1:8000
```
