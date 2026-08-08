"""Inference helpers for the flight-intel dashboard.

Loads the trained fare model + DGCA tidy tables once and exposes route-level
queries: airline ranking, best time-of-day, booking-window curve, class gap,
plus airline league tables (reliability, complaints) and route demand.
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
PROC = os.path.join(HERE, "data", "processed")
FARE_CSV = os.path.join(HERE, "data", "raw", "fares",
                        "Airlines-Flights-Data-Analysis__airlines_flights_data.csv")

OTP_NAME = {"spicejet": "Spicejet", "airasia": "Air Asia", "vistara": "Vistara",
            "gofirst": "GoAir", "indigo": "Indigo", "airindia": "Air India"}
DEFUNCT = {"GO_FIRST"}
MERGED_NOTE = {"Vistara": "merged into Air India (2024-25)"}
CITY_TO_DGCA = {"Delhi": "DELHI", "Mumbai": "MUMBAI", "Bangalore": "BENGALURU",
                "Kolkata": "KOLKATA", "Hyderabad": "HYDERABAD", "Chennai": "CHENNAI"}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z]", "", str(s).lower())


@lru_cache(maxsize=1)
def artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "fare_model.joblib"))
    ref = pd.read_csv(os.path.join(MODEL_DIR, "route_airline_ref.csv"))
    meta = json.load(open(os.path.join(MODEL_DIR, "fare_meta.json")))
    otp_path = os.path.join(PROC, "airline_otp.csv")
    otp_means = (pd.read_csv(otp_path).groupby("airline")["otp"].mean().to_dict()
                 if os.path.exists(otp_path) else {})
    return model, ref, meta, otp_means


def cities() -> list[str]:
    return artifacts()[2]["cities"]


def metrics() -> dict:
    m = artifacts()[2]
    return {"mae": m["mae"], "mape": m["mape"], "r2": m["r2"]}


def _candidates(source: str, dest: str, exclude_defunct=True) -> pd.DataFrame:
    _, ref, _, _ = artifacts()
    cand = ref[(ref.source_city == source) & (ref.destination_city == dest)].copy()
    if exclude_defunct:
        cand = cand[~cand.airline.isin(DEFUNCT)]
    return cand


def _avg_price(source, dest, days_left=15, dep_time="Morning", travel_class="Economy"):
    model, _, meta, _ = artifacts()
    cand = _candidates(source, dest)
    if cand.empty:
        return None
    arr = meta["typical_arrival"].get(dep_time, "Night")
    feats = [{
        "airline": r.airline, "source_city": source, "destination_city": dest,
        "departure_time": dep_time, "arrival_time": arr, "stops": r.stops,
        "class": travel_class, "duration": r.duration, "days_left": days_left}
        for _, r in cand.iterrows()]
    return float(model.predict(pd.DataFrame(feats)).mean())


def rank_route(source, dest, days_left=15, travel_class="Economy",
               departure_time="Morning", price_weight=0.5, reliability_weight=0.5):
    model, _, meta, otp_means = artifacts()
    cand = _candidates(source, dest)
    if cand.empty:
        return pd.DataFrame()
    arr = meta["typical_arrival"].get(departure_time, "Night")
    rows = []
    for _, r in cand.iterrows():
        feat = pd.DataFrame([{
            "airline": r.airline, "source_city": source, "destination_city": dest,
            "departure_time": departure_time, "arrival_time": arr, "stops": r.stops,
            "class": travel_class, "duration": r.duration, "days_left": days_left}])
        price = float(model.predict(feat)[0])
        otp = otp_means.get(OTP_NAME.get(_norm(r.airline), ""), np.nan)
        rows.append({"airline": r.airline, "pred_price": round(price),
                     "otp": round(otp, 1) if otp == otp else None,
                     "note": MERGED_NOTE.get(r.airline, "")})
    out = pd.DataFrame(rows)
    p = out["pred_price"]
    out["price_score"] = (p.max() - p) / (p.max() - p.min()) if p.max() > p.min() else 1.0
    o = out["otp"].fillna(out["otp"].mean() if out["otp"].notna().any() else 75)
    out["rel_score"] = ((o - 60) / 30).clip(0, 1)  # 60%→0, 90%→1 (calibrated, not min-max)
    tot = max(price_weight + reliability_weight, 1e-9)
    out["score"] = ((price_weight * out["price_score"] +
                     reliability_weight * out["rel_score"]) / tot).round(3)
    return out.sort_values("score", ascending=False).reset_index(drop=True)


def best_time(source, dest, days_left=15, travel_class="Economy") -> pd.DataFrame:
    _, _, meta, _ = artifacts()
    data = [(t, _avg_price(source, dest, days_left=days_left, dep_time=t,
                           travel_class=travel_class)) for t in meta["departure_times"]]
    data = [(t, p) for t, p in data if p is not None]
    return pd.DataFrame(data, columns=["time", "avg_price"]).sort_values("avg_price")


def booking_curve(source, dest, travel_class="Economy", dep_time="Morning") -> pd.DataFrame:
    days = [1, 3, 5, 7, 10, 15, 20, 25, 30, 40, 45]
    data = [(d, _avg_price(source, dest, days_left=d, dep_time=dep_time,
                           travel_class=travel_class)) for d in days]
    data = [(d, p) for d, p in data if p is not None]
    return pd.DataFrame(data, columns=["days_left", "avg_price"])


def class_compare(source, dest) -> dict:
    return {"Economy": _avg_price(source, dest, travel_class="Economy"),
            "Business": _avg_price(source, dest, travel_class="Business")}


def reliability_table() -> pd.DataFrame:
    p = os.path.join(PROC, "airline_otp.csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    df = pd.read_csv(p)
    out = df.groupby("airline")["otp"].mean().round(1).reset_index()
    return out.sort_values("otp", ascending=False).reset_index(drop=True)


def complaints_table() -> pd.DataFrame:
    p = os.path.join(PROC, "airline_grievances.csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    df = pd.read_csv(p)
    out = df.groupby("airline")["grievances"].sum().astype(int).reset_index()
    return out.sort_values("grievances", ascending=False).reset_index(drop=True)


@lru_cache(maxsize=1)
def _fare_df():
    pq = FARE_CSV.replace(".csv", ".parquet")  # prefer parquet if present (smaller, for deploy)
    df = pd.read_parquet(pq) if os.path.exists(pq) else pd.read_csv(FARE_CSV)
    # Some IndiGo flight IDs were Excel-mangled into sci-notation ("6.00E-218" -> "6E-218").
    df["flight"] = df["flight"].astype(str).str.replace(r"^6\.0*E-(\d+)$", r"6E-\1", regex=True)
    return df


def rank_flights(source, dest, days_left=15, travel_class="Economy",
                 departure_time=None, price_weight=0.5, reliability_weight=0.5,
                 top_n=15, exclude_defunct=True) -> pd.DataFrame:
    """Rank individual FLIGHT NUMBERS on a route by a price + reliability composite.

    Uses observed fares from the 300k dataset near the requested booking window.
    Note: the dataset has time-of-day buckets (not exact clock times) and no
    day-of-week / calendar date.
    """
    df = _fare_df()
    sub = df[(df.source_city == source) & (df.destination_city == dest)
             & (df["class"] == travel_class)].copy()
    if departure_time:
        sub = sub[sub.departure_time == departure_time]
    if exclude_defunct:
        sub = sub[~sub.airline.isin(DEFUNCT)]
    if sub.empty:
        return pd.DataFrame()
    # Prices near the requested booking window (fall back to all if none in range).
    win = sub[(sub.days_left >= days_left - 3) & (sub.days_left <= days_left + 3)]
    if win.empty:
        win = sub
    agg = win.groupby("flight").agg(
        airline=("airline", "first"),
        departure_time=("departure_time", lambda s: s.mode().iat[0]),
        arrival_time=("arrival_time", lambda s: s.mode().iat[0]),
        stops=("stops", lambda s: s.mode().iat[0]),
        duration=("duration", "median"),
        price=("price", "median")).reset_index()

    _, _, _, otp_means = artifacts()
    agg["otp"] = agg.airline.map(lambda a: otp_means.get(OTP_NAME.get(_norm(a), ""), np.nan))
    p = agg["price"]
    agg["price_score"] = (p.max() - p) / (p.max() - p.min()) if p.max() > p.min() else 1.0
    o = agg["otp"].fillna(agg["otp"].mean() if agg["otp"].notna().any() else 75)
    agg["rel_score"] = ((o - 60) / 30).clip(0, 1)  # 60%→0, 90%→1 (calibrated, not min-max)
    tot = max(price_weight + reliability_weight, 1e-9)
    agg["score"] = ((price_weight * agg.price_score + reliability_weight * agg.rel_score) / tot).round(3)
    agg["price"] = agg["price"].round().astype(int)
    agg["otp"] = agg["otp"].round(1)
    agg["duration"] = agg["duration"].round(2)
    return agg.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)


def route_demand(source, dest):
    """Return (total_pax, monthly_series_df) for a route from DGCA city-pair data."""
    p = os.path.join(PROC, "route_city_monthly.csv")
    if not os.path.exists(p):
        return None, pd.DataFrame()
    c1, c2 = CITY_TO_DGCA.get(source), CITY_TO_DGCA.get(dest)
    if not c1 or not c2:
        return None, pd.DataFrame()
    key = " <-> ".join(sorted([c1, c2]))
    df = pd.read_csv(p)
    sub = df[df["route"] == key]
    if sub.empty:
        return None, pd.DataFrame()
    total = float(sub["total_pax"].sum())
    monthly = (sub.groupby(["Year", "Month"])["total_pax"].sum().reset_index())
    monthly["date"] = pd.to_datetime(dict(year=monthly.Year, month=monthly.Month, day=1))
    return total, monthly.sort_values("date")[["date", "total_pax"]].set_index("date")
