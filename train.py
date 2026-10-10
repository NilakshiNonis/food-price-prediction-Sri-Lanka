"""Trains Random Forest, XGBoost and Stacking; compares them; exports .pkl files and a report for the dashboard."""
import json, pickle, numpy as np, pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pipeline as P

def nn(o):  # NaN -> None so the JSON report is valid
    if isinstance(o, dict): return {k: nn(v) for k, v in o.items()}
    return None if isinstance(o, float) and o != o else o


df = P.load_clean()
# single-sample inference: hold out one row from the newest month BEFORE any splitting
hold = df[df.date == df.date.max()].sample(1, random_state=7)
df = df.drop(hold.index)
Xtr, Xte, ytr, yte = P.split(df)
print("train", Xtr.shape, "test", Xte.shape)

models, results, imp = {}, {}, {}
for name, est in {"Random Forest": P.rf(), "XGBoost": P.xgb(), "Stacking (RF+XGB)": P.stack()}.items():
    m = P.build(est).fit(Xtr, ytr)
    p = m.predict(Xte)
    results[name] = {"rmse": round(float(np.sqrt(mean_squared_error(yte, p))), 2), "mae": round(float(mean_absolute_error(yte, p)), 2),
                     "r2": round(float(r2_score(yte, p)), 4), "mape_pct": round(float(np.mean(np.abs(yte - p) / yte) * 100), 2)}
    models[name] = m
    print(name, results[name])
    if name != "Stacking (RF+XGB)":
        s = Xte.sample(3000, random_state=1)
        pi = permutation_importance(m, s, yte.loc[s.index], n_repeats=3, random_state=0, scoring="neg_mean_absolute_error")
        imp[name] = sorted(({"feature": c, "importance": round(float(v), 2)} for c, v in zip(s.columns, pi.importances_mean)), key=lambda d: -d["importance"])

winner = min(results, key=lambda n: results[n]["rmse"])
pickle.dump(models[winner], open("models/model.pkl", "wb"))        # best model
pickle.dump(models, open("models/all_models.pkl", "wb"))           # all three, used by the dashboard

lookup = df.groupby("commodity").agg(category=("category", lambda s: s.mode()[0]), unit=("unit", lambda s: s.mode()[0])).to_dict("index")
mk = df.groupby("market").agg(admin1=("admin1", lambda s: s.mode()[0]), latitude=("latitude", "median"), longitude=("longitude", "median")).to_dict("index")
hrow = hold.iloc[0]
json.dump(nn({"results": results, "winner": winner, "importance": imp, "commodities": lookup, "markets": mk,
           "pricetypes": sorted(df.pricetype.unique()), "years": [int(df.year.min()), int(df.year.max())],
           "holdout": {"features": {c: (hrow[c].item() if hasattr(hrow[c], "item") else hrow[c]) for c in P.CAT + P.NUM}, "actual": float(hrow[P.TARGET])},
           "train_rows": len(Xtr), "test_rows": len(Xte)}), open("models/report.json", "w"), indent=1, default=str)
print("winner:", winner, "| holdout actual:", hrow[P.TARGET])
