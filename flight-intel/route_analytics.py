"""Route / sector 'problem' analytics.

DGCA's free data gives on-time performance per AIRLINE (not per route) and no
per-flight delays. So route reliability here is a transparent PROXY: the
on-time record of the airlines that operate each route, weighted by how much
each serves it (row share in the 300k fare dataset). Limited to the 6 metros
that the fare dataset covers.

Also: complaint categories (system-wide) and busiest/least-busy routes (demand,
covering all 1,933 city-pairs).
"""
from __future__ import annotations

import os

import pandas as pd

from predict import DEFUNCT, MODEL_DIR, OTP_NAME, PROC, _norm

RAW_DAILY = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "data", "raw", "dgca", "daily.csv")


def _otp_means() -> dict:
    p = os.path.join(PROC, "airline_otp.csv")
    if not os.path.exists(p):
        return {}
    return pd.read_csv(p).groupby("airline")["otp"].mean().to_dict()


def route_reliability(exclude_defunct: bool = True) -> pd.DataFrame:
    """Airline-mix-weighted on-time proxy per route (worst first)."""
    ref = pd.read_csv(os.path.join(MODEL_DIR, "route_airline_ref.csv"))
    if exclude_defunct:
        ref = ref[~ref.airline.isin(DEFUNCT)]
    otp = _otp_means()
    ref = ref.copy()
    ref["otp"] = ref.airline.map(lambda a: otp.get(OTP_NAME.get(_norm(a), ""), float("nan")))
    ref = ref.dropna(subset=["otp"])
    ref["route"] = ref.apply(
        lambda r: " ↔ ".join(sorted([r.source_city, r.destination_city])), axis=1)
    ref["wtd"] = ref.otp * ref.n
    g = ref.groupby("route").agg(num=("wtd", "sum"), den=("n", "sum")).reset_index()
    g["reliability_otp"] = (g.num / g.den).round(1)
    airlines = (ref.groupby("route")["airline"]
                .agg(lambda s: ", ".join(sorted(s.unique()))).rename("airlines"))
    g = g.merge(airlines, on="route")
    return g[["route", "reliability_otp", "airlines"]].sort_values("reliability_otp").reset_index(drop=True)


def complaint_categories() -> pd.DataFrame:
    """System-wide grievance totals by category (raw magnitude)."""
    df = pd.read_csv(RAW_DAILY)
    cats = {"Baggage": "Grievances (Baggage)", "Check-in": "Grievances (Check-In)",
            "Meals": "Grievances (Meals)", "Refunds": "Grievances (Refunds)",
            "Security": "Grievances (Security)", "Security Check": "Grievances (Security Check)",
            "Customs": "Grievances (Customs)", "Immigration": "Grievances (Immigration)",
            "Others": "Grievances (Others)"}
    rows = [(name, int(pd.to_numeric(df[col], errors="coerce").sum()))
            for name, col in cats.items() if col in df.columns]
    return (pd.DataFrame(rows, columns=["category", "total"])
            .sort_values("total", ascending=False).reset_index(drop=True))


def busiest_routes(n: int = 10, ascending: bool = False) -> pd.DataFrame:
    """Top (or bottom) routes by total passengers across all 1,933 city-pairs."""
    p = os.path.join(PROC, "route_city_monthly.csv")
    df = pd.read_csv(p)
    out = (df.groupby("route")["total_pax"].sum().reset_index()
           .sort_values("total_pax", ascending=ascending).head(n).reset_index(drop=True))
    out["million_pax"] = (out["total_pax"] / 1e6).round(2)
    return out[["route", "million_pax"]]


if __name__ == "__main__":
    print("=== WORST routes by reliability proxy (6 metros) ===")
    print(route_reliability().head(6).to_string(index=False))
    print("\n=== Complaint categories ===")
    print(complaint_categories().to_string(index=False))
    print("\n=== Busiest routes ===")
    print(busiest_routes().to_string(index=False))
