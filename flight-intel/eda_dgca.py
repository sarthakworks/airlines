"""First-pass cleaning + insights on the DGCA mirror data.

Produces tidy CSVs in data/processed/ and prints headline numbers for:
  - airline on-time performance (reliability)
  - airline grievance volume (complaints)
  - busiest domestic city-pairs (route/sector demand)
"""
from __future__ import annotations

import os
import re

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw", "dgca")
OUT = os.path.join(HERE, "data", "processed")
os.makedirs(OUT, exist_ok=True)

INDIAN_CARRIERS = [
    "Air India", "IndiGo", "SpiceJet", "Vistara", "GoAir", "Akasa Air",
    "Alliance Air", "Air Asia", "Air India Express", "Aix Connect", "Star Air",
]


def clean_otp():
    df = pd.read_csv(os.path.join(RAW, "daily.csv"))
    otp_cols = [c for c in df.columns if c.startswith("On Time Performance (")]
    tidy = df[["Date"] + otp_cols].melt(id_vars="Date", var_name="airline", value_name="otp")
    tidy["airline"] = tidy["airline"].str.replace(r"On Time Performance \((.*)\)", r"\1", regex=True)
    # OTP is stored as strings like "99.8%" — strip the percent sign before parsing.
    tidy["otp"] = pd.to_numeric(tidy["otp"].astype(str).str.rstrip("%"), errors="coerce")
    tidy = tidy.dropna(subset=["otp"])
    tidy.to_csv(os.path.join(OUT, "airline_otp.csv"), index=False)

    print("\n=== ON-TIME PERFORMANCE (reliability) ===")
    print(f"rows={len(tidy)}  dates={tidy['Date'].min()} .. {tidy['Date'].max()}  "
          f"({tidy['Date'].nunique()} report days)")
    rank = tidy.groupby("airline")["otp"].agg(["mean", "count"]).sort_values("mean", ascending=False)
    rank["mean"] = rank["mean"].round(1)
    print("Mean OTP % by airline (higher = more reliable):")
    print(rank.to_string())


def clean_grievances():
    df = pd.read_csv(os.path.join(RAW, "daily.csv"))
    cols = {c: c[len("Grievances ("):-1] for c in df.columns if c.startswith("Grievances (")}
    airline_cols = {c: name for c, name in cols.items() if name in INDIAN_CARRIERS}
    sub = df[["Date"] + list(airline_cols)].copy()
    sub = sub.rename(columns=airline_cols)
    tidy = sub.melt(id_vars="Date", var_name="airline", value_name="grievances")
    tidy["grievances"] = pd.to_numeric(tidy["grievances"], errors="coerce")
    tidy = tidy.dropna(subset=["grievances"])
    tidy.to_csv(os.path.join(OUT, "airline_grievances.csv"), index=False)

    print("\n=== GRIEVANCES (complaints, raw volume — needs per-pax normalisation) ===")
    print(f"rows={len(tidy)}  dates={tidy['Date'].min()} .. {tidy['Date'].max()}")
    tot = tidy.groupby("airline")["grievances"].sum().sort_values(ascending=False)
    print(tot.to_string())


def clean_routes():
    df = pd.read_csv(os.path.join(RAW, "domestic_city.csv"))
    df["total_pax"] = df["PaxToCity2"].fillna(0) + df["PaxFromCity2"].fillna(0)
    # direction-agnostic route key
    df["route"] = df.apply(lambda r: " <-> ".join(sorted([str(r["City1"]), str(r["City2"])])), axis=1)
    df.to_csv(os.path.join(OUT, "route_city_monthly.csv"), index=False)

    print("\n=== ROUTES / SECTORS (domestic city-pair demand) ===")
    print(f"rows={len(df)}  years={int(df['Year'].min())}..{int(df['Year'].max())}  "
          f"distinct city-pairs={df['route'].nunique()}")
    top = df.groupby("route")["total_pax"].sum().sort_values(ascending=False).head(12)
    print("Busiest domestic city-pairs (total passengers, all-time in data):")
    print((top / 1e6).round(2).astype(str).add(" M").to_string())


if __name__ == "__main__":
    clean_otp()
    clean_grievances()
    clean_routes()
    print(f"\nTidy CSVs written to {OUT}")
