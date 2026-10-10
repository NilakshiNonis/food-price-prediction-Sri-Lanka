"""FastAPI backend. Run: uvicorn app:app --reload   (open http://127.0.0.1:8000)"""
import json, pickle, pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pipeline as P

models = pickle.load(open("models/all_models.pkl", "rb"))   # unpickled with the same pipeline.py on the path
report = json.load(open("models/report.json"))
app = FastAPI(title="Sri Lanka Food Price Predictor")

class Req(BaseModel):
    commodity: str
    market: str
    pricetype: str = "Retail"
    year: int
    month: int

@app.get("/api/info")
def info():
    keys = ("commodities", "markets", "pricetypes", "years", "winner", "holdout", "train_rows", "test_rows")
    return {k: report[k] for k in keys}

@app.get("/api/metrics")
def metrics():
    return {"results": report["results"], "importance": report["importance"]}

@app.post("/api/predict")
def predict(r: Req):
    c, m = report["commodities"].get(r.commodity), report["markets"].get(r.market)
    if not c or not m: raise HTTPException(422, "Unknown commodity or market")
    if not 1 <= r.month <= 12: raise HTTPException(422, "Month must be 1-12")
    row = pd.DataFrame([{"commodity": r.commodity, "category": c["category"], "unit": c["unit"], "pricetype": r.pricetype,
                         "market": r.market, "admin1": m["admin1"], "latitude": m["latitude"] if m["latitude"] is not None else float("nan"), "longitude": m["longitude"] if m["longitude"] is not None else float("nan"),
                         "year": r.year, "month": r.month}])[P.CAT + P.NUM]
    return {"unit": c["unit"], "winner": report["winner"],
            "predictions": {n: round(float(mod.predict(row)[0]), 2) for n, mod in models.items()}}

app.mount("/static", StaticFiles(directory="static"), name="static")
@app.get("/")
def index(): return FileResponse("static/index.html")
