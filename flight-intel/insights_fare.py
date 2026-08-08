"""Best-time and best-booking-window insights from the trained fare model.

Answers: what time of day is cheapest, how far ahead to book, and Economy vs
Business gap — per route. Run: python insights_fare.py [Source] [Dest]
"""
from __future__ import annotations

import json
import os
import sys

import joblib
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
DEFUNCT = {"GO_FIRST"}

model = joblib.load(os.path.join(MODEL_DIR, "fare_model.joblib"))
ref = pd.read_csv(os.path.join(MODEL_DIR, "route_airline_ref.csv"))
meta = json.load(open(os.path.join(MODEL_DIR, "fare_meta.json")))
TYP_ARR = meta["typical_arrival"]


def _avg_price(source, dest, days_left=15, dep_time="Morning", travel_class="Economy"):
    cand = ref[(ref.source_city == source) & (ref.destination_city == dest)
               & (~ref.airline.isin(DEFUNCT))]
    if cand.empty:
        return None
    feats = [{
        "airline": r.airline, "source_city": source, "destination_city": dest,
        "departure_time": dep_time, "arrival_time": TYP_ARR.get(dep_time, "Night"),
        "stops": r.stops, "class": travel_class,
        "duration": r.duration, "days_left": days_left} for _, r in cand.iterrows()]
    return float(model.predict(pd.DataFrame(feats)).mean())


def report(source: str, dest: str):
    print(f"\n================  {source} -> {dest}  ================")

    print("\nBest time of day (avg Economy fare, 15 days out):")
    times = [(t, _avg_price(source, dest, dep_time=t)) for t in meta["departure_times"]]
    times = [(t, p) for t, p in times if p is not None]
    for t, p in sorted(times, key=lambda x: x[1]):
        print(f"  {t:14s} ₹{p:6,.0f}")

    print("\nBest booking window (avg Economy fare, morning):")
    for d in [1, 3, 5, 7, 10, 15, 20, 30, 45]:
        p = _avg_price(source, dest, days_left=d)
        if p is not None:
            print(f"  {d:2d} days out  ₹{p:6,.0f}")

    eco = _avg_price(source, dest, travel_class="Economy")
    biz = _avg_price(source, dest, travel_class="Business")
    if eco and biz:
        print(f"\nEconomy ₹{eco:,.0f}  vs  Business ₹{biz:,.0f}  "
              f"({biz/eco:.1f}x)")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "Delhi"
    dst = sys.argv[2] if len(sys.argv) > 2 else "Mumbai"
    report(src, dst)
