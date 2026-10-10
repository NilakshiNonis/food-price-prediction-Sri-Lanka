"""Shared data loading, preprocessing and model definitions (used by the notebook, train.py and app.py)."""
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

DATA = "data/wfp_food_prices_lka.csv"
TARGET = "price"                       # retail/wholesale price in LKR per unit
CAT = ["commodity", "category", "unit", "pricetype", "market", "admin1"]
NUM = ["latitude", "longitude", "year", "month"]
SPLIT_DATE = "2025-01-01"              # time-based split: train 2004-2024, test 2025 (most rows are from 2023+)

def load_clean(path=DATA):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.drop_duplicates()
    # constant / redundant / leaky columns: priceflag & currency never vary; ids duplicate names;
    # usdprice is just price / exchange rate (target leakage); admin2 is redundant with market+lat/lon
    df = df.drop(columns=["priceflag", "currency", "market_id", "commodity_id", "usdprice", "admin2"])
    df["year"], df["month"] = df["date"].dt.year, df["date"].dt.month
    df["admin1"] = df["admin1"].fillna("Unknown")          # imputation of missing region (194 rows)
    return df.sort_values("date").reset_index(drop=True)   # lat/lon NaNs are median-imputed in the pipeline

def preprocessor():
    num = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat = OneHotEncoder(handle_unknown="ignore")
    return ColumnTransformer([("num", num, NUM), ("cat", cat, CAT)])

def rf():  # hand-designed: deep-ish trees to capture commodity x market x time, leaf>=3 to damp noise
    return RandomForestRegressor(n_estimators=300, max_depth=20, min_samples_leaf=3, max_features=0.5, n_jobs=-1, random_state=42)

def xgb():  # smaller learning rate + more trees, row/column subsampling and L2 for regularisation
    return XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.7,
                        min_child_weight=3, reg_lambda=2.0, n_jobs=-1, random_state=42)

def stack():
    return StackingRegressor([("rf", rf()), ("xgb", xgb())], final_estimator=RidgeCV(alphas=[0.1, 1, 10]), cv=KFold(3, shuffle=True, random_state=42), n_jobs=1)

def build(estimator):
    """Full pipeline: preprocess -> model, with log1p target transform (prices span 11 to 3,700 LKR)."""
    return TransformedTargetRegressor(Pipeline([("pre", preprocessor()), ("model", estimator)]), func=np.log1p, inverse_func=np.expm1)

def split(df):
    train, test = df[df.date < SPLIT_DATE], df[df.date >= SPLIT_DATE]
    cols = CAT + NUM
    return train[cols], test[cols], train[TARGET], test[TARGET]
