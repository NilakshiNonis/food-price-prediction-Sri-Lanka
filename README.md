# Sri Lanka Food Price Prediction: Random Forest vs XGBoost vs Stacking

A class activity for ML Lecture 7 (ensemble methods). The project predicts the price of a food item at a Sri Lankan market and compares three ensemble models. It includes a FastAPI backend and a web dashboard for live predictions.

## Problem and data
- **Task (regression):** predict the price in LKR per unit of a food item from its commodity, market, price type, year and month.
- **Dataset:** WFP Food Prices for Sri Lanka, from the Humanitarian Data Exchange (HDX). It has 32,496 monthly market price records from 2004 to 2025, covering 42 commodities and 44 markets.
- **Note on coverage:** about 84% of the records are from 2023-2025, so recent prices are known much better than older ones.

## Method
| Step | What was done |
|---|---|
| Cleaning | Removed `usdprice` (it would leak the target), constant columns, ID columns and the redundant `admin2` column |
| Null handling | 194 rows with missing region became "Unknown"; missing latitude/longitude were filled with the median |
| Encoding and scaling | One-hot encoding for categories, standard scaling for numbers |
| Target transform | Models learn `log1p(price)` because prices range from about 11 to 3,700 LKR; predictions are converted back to LKR |
| Split | **Time-based:** train on 2004-2024, test on 2025. One row from the newest month is held out for single-sample inference |
| Models | Random Forest, XGBoost, and Stacking (RF + XGBoost with a Ridge meta-model), all with hand-designed hyperparameters (see the notebook, section 3) |

## Results (2025 test set)
| Model | RMSE (LKR) | MAE (LKR) | R² | MAPE |
|---|---|---|---|---|
| Random Forest | 192.85 | 129.33 | 0.9116 | 27.5% |
| **XGBoost** | **186.23** | **125.29** | **0.9175** | **26.8%** |
| Stacking (RF + XGBoost) | 187.22 | 125.64 | 0.9167 | 27.0% |

Results come from my own training run on my machine. Re-running `python train.py` can give slightly different numbers depending on library versions.

## Key findings
- **XGBoost was the most accurate**, with Stacking very close and Random Forest slightly behind. The differences between the models are small.
- **The evaluation split matters more than the model.** A random split gives a misleadingly high R² of about 0.98, because the model sees neighbouring months. The honest time-based split gives about 0.92.
- **Error depends on the food type.** Stable items such as cereals are predicted very well (about 6% average error), but vegetables and fruit have the highest percentage error (about 40%) because their prices change with season and weather.
- **The models mostly learn what the item is.** Category, commodity and unit dominate feature importance, while market, year and month add little.

## Limitations
- Tree models cannot predict prices outside the range they saw in training, which is a problem when prices keep rising.
- The model has no information about recent prices, so it cannot follow short-term swings. Adding lag features (last month's price for the same item and market) is the most promising improvement.
- Most of the data is from 2023-2025, so older periods are covered less well.
- The data comes from an international agency (WFP) about Sri Lankan markets, and was not collected by a Sri Lankan organisation.

## Project files
| File | Purpose |
|---|---|
| `food_price_analysis.ipynb` | Full analysis: EDA, preprocessing, parameter design, comparison, strengths and limitations, interpretation, inference |
| `pipeline.py` | Shared data cleaning and model definitions |
| `train.py` | Trains the three models and saves `models/model.pkl` (best model), `models/all_models.pkl` and `models/report.json` |
| `app.py` | FastAPI backend (`/api/predict`, `/api/metrics`, `/api/info`) |
| `static/index.html` | Web dashboard |
| `data/wfp_food_prices_lka.csv` | The dataset |

## How to run
1. Install Python 3.10 or newer.
2. Open a terminal in the project folder and install the libraries:
   ```
   pip install -r requirements.txt
   ```
3. Train the models (takes a few minutes):
   ```
   python train.py
   ```
4. Start the app:
   ```
   python -m uvicorn app:app --reload
   ```
5. Open http://127.0.0.1:8000 in a browser. Choose a commodity and market and click **Predict price**, or click **Load held-out sample** to see the inference demo, which compares the predictions with the real price.

To read the analysis, open `food_price_analysis.ipynb`. It already contains all outputs.

## Notes
- `models/all_models.pkl` (about 227 MB) is excluded from this repository because GitHub rejects files over 100 MB. `python train.py` recreates it, and the dashboard needs it. The small `models/model.pkl` (best model) is included.
- Single-sample inference: one 2025 row (dried fish in Badulla, actual price LKR 2,640) was held out before training. The best model predicted about LKR 2,106, roughly 20% below the actual price.
