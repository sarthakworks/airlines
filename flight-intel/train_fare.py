"""Train a flight-fare predictor and build a composite 'best airline for a route'
ranking that blends predicted price with airline on-time reliability (DGCA OTP).

Data:
  data/raw/fares/Airlines-Flights-Data-Analysis__airlines_flights_data.csv  (300k rows)
  data/processed/airline_otp.csv  (from eda_dgca.py)

Run:  python train_fare.py
"""
from __future__ import annotations

import json
import os
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

HERE = os.path.dirname(os.path.abspath(__file__))
FARE_CSV = os.path.join(HERE, "data", "raw", "fares",
                        "Airlines-Flights-Data-Analysis__airlines_flights_data.csv")
OTP_CSV = os.path.join(HERE, "data", "processed", "airline_otp.csv")
MODEL_DIR = os.path.join(HERE, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

CAT_FEATURES = ["airline", "source_city", "destination_city",
                "departure_time", "arrival_time", "stops", "class"]
NUM_FEATURES = ["duration", "days_left"]
TARGET = "price"

# fare-dataset airline name -> DGCA OTP airline name
OTP_NAME = {"spicejet": "Spicejet", "airasia": "Air Asia", "vistara": "Vistara",
            "gofirst": "GoAir", "indigo": "Indigo", "airindia": "Air India"}
DEFUNCT = {"GO_FIRST"}          # Go First ceased ops 2023
MERGED_NOTE = {"Vistara": "merged into Air India (2024-25)"}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z]", "", str(s).lower())


def load_data() -> pd.DataFrame:
    df = pd.read_csv(FARE_CSV)
    df = df.drop(columns=[c for c in ["index", "Unnamed: 0"] if c in df.columns])
    return df


def train():
    df = load_data()
    print(f"Fare rows: {len(df):,}")
    print("Airlines:", sorted(df.airline.unique()))
    print("Cities  :", sorted(df.source_city.unique()))

    X, y = df[CAT_FEATURES + NUM_FEATURES], df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES)],
        remainder="passthrough",
    )
    model = Pipeline([("pre", pre),
                      ("gb", HistGradientBoostingRegressor(
                          max_iter=400, learning_rate=0.08, max_depth=8,
                          random_state=42))])
    model.fit(Xtr, ytr)

    pred = model.predict(Xte)
    mae = mean_absolute_error(yte, pred)
    r2 = r2_score(yte, pred)
    mape = float(np.mean(np.abs((yte - pred) / yte)) * 100)
    print(f"\nFARE MODEL  MAE=₹{mae:,.0f}  MAPE={mape:.1f}%  R²={r2:.3f}")

    joblib.dump(model, os.path.join(MODEL_DIR, "fare_model.joblib"))

    # Reference row per (route, airline) for realistic enumeration at inference.
    # Prefer NON-STOP (what travellers usually want) when the route has any; else
    # fall back to the most common stop count.
    def _ref_row(g):
        ns = g[g["stops"] == "zero"]
        use = ns if len(ns) else g
        return pd.Series({"duration": round(use["duration"].median(), 2),
                          "stops": "zero" if len(ns) else use["stops"].mode().iat[0],
                          "n": len(g)})

    ref = (df.groupby(["source_city", "destination_city", "airline"])
             .apply(_ref_row, include_groups=False).reset_index())
    ref.to_csv(os.path.join(MODEL_DIR, "route_airline_ref.csv"), index=False)
    typ_arr = (df.groupby("departure_time")["arrival_time"]
                 .agg(lambda s: s.mode().iat[0]).to_dict())

    meta = {"mae": mae, "mape": mape, "r2": r2,
            "cat_features": CAT_FEATURES, "num_features": NUM_FEATURES,
            "typical_arrival": typ_arr,
            "cities": sorted(df.source_city.unique().tolist()),
            "airlines": sorted(df.airline.unique().tolist()),
            "departure_times": sorted(df.departure_time.unique().tolist())}
    with open(os.path.join(MODEL_DIR, "fare_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return model, ref, typ_arr


def _otp_means() -> dict:
    if not os.path.exists(OTP_CSV):
        return {}
    otp = pd.read_csv(OTP_CSV)
    return otp.groupby("airline")["otp"].mean().to_dict()


def rank_route(source: str, dest: str, days_left: int = 15,
               travel_class: str = "Economy", departure_time: str = "Morning",
               price_weight: float = 0.5, reliability_weight: float = 0.5,
               exclude_defunct: bool = True):
    """Rank airlines on a route by a blend of predicted price + reliability."""
    model = joblib.load(os.path.join(MODEL_DIR, "fare_model.joblib"))
    ref = pd.read_csv(os.path.join(MODEL_DIR, "route_airline_ref.csv"))
    meta = json.load(open(os.path.join(MODEL_DIR, "fare_meta.json")))
    otp_means = _otp_means()

    cand = ref[(ref.source_city == source) & (ref.destination_city == dest)].copy()
    if exclude_defunct:
        cand = cand[~cand.airline.isin(DEFUNCT)]
    if cand.empty:
        return pd.DataFrame()

    arr = meta["typical_arrival"].get(departure_time, "Night")
    rows = []
    for _, r in cand.iterrows():
        feat = pd.DataFrame([{
            "airline": r.airline, "source_city": source, "destination_city": dest,
            "departure_time": departure_time, "arrival_time": arr,
            "stops": r.stops, "class": travel_class,
            "duration": r.duration, "days_left": days_left}])
        price = float(model.predict(feat)[0])
        otp = otp_means.get(OTP_NAME.get(_norm(r.airline), ""), np.nan)
        rows.append({"airline": r.airline, "pred_price": round(price),
                     "otp": round(otp, 1) if otp == otp else None,
                     "note": MERGED_NOTE.get(r.airline, "")})

    out = pd.DataFrame(rows)
    # Normalise for a composite score (cheaper + more punctual = better).
    p = out["pred_price"]
    out["price_score"] = (p.max() - p) / (p.max() - p.min()) if p.max() > p.min() else 1.0
    o = out["otp"].fillna(out["otp"].mean() if out["otp"].notna().any() else 75)
    out["rel_score"] = (o - o.min()) / (o.max() - o.min()) if o.max() > o.min() else 1.0
    out["score"] = (price_weight * out["price_score"] +
                    reliability_weight * out["rel_score"]).round(3)
    return out.sort_values("score", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    train()
    print("\n" + "=" * 60)
    for src, dst in [("Delhi", "Mumbai"), ("Bangalore", "Kolkata")]:
        print(f"\n### Best airlines {src} -> {dst}  (Economy, book 15 days out, morning)")
        r = rank_route(src, dst)
        if r.empty:
            print("  (no data for this route)")
        else:
            print(r[["airline", "pred_price", "otp", "score", "note"]].to_string(index=False))
