"""Unified 'real schedules' source over the user-provided CSVs in data/raw/random.

These add what the fare model lacked: exact clock times, real dates / day-of-week,
and flight numbers with actual prices. They are historical ("old data": 2018-2024),
so treat times/fares as indicative, not live.

Normalised schema:
  flight_no, airline, origin, destination, dep_time, arr_time,
  day_of_week, date, price, cls, stops, source_ds
"""
from __future__ import annotations

import os
import re
from functools import lru_cache

import pandas as pd

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw", "random")

_CITY_ALIASES = {
    "bombay": "Mumbai", "bengaluru": "Bangalore", "calcutta": "Kolkata",
    "madras": "Chennai", "new delhi": "Delhi", "trivandrum": "Thiruvananthapuram",
    "pondicherry": "Puducherry",
}
COLS = ["flight_no", "airline", "origin", "destination", "dep_time", "arr_time",
        "day_of_week", "date", "price", "cls", "stops", "source_ds"]


_AIRLINE_ALIASES = {
    "indigo": "IndiGo", "goair": "Go First", "go air": "Go First", "go first": "Go First",
    "airasia india": "AirAsia", "air asia": "AirAsia", "spicejet": "SpiceJet",
    "air india": "Air India", "vistara": "Vistara", "akasa air": "Akasa Air",
    "alliance air": "Alliance Air", "star air": "Star Air", "trujet": "TruJet",
    "jet airways": "Jet Airways", "jetlite": "JetLite",
}


def canon_city(name) -> str:
    s = re.sub(r"\s+", " ", str(name).strip()).title()
    return _CITY_ALIASES.get(s.lower(), s)


def canon_airline(name) -> str:
    s = re.sub(r"\s+", " ", str(name).strip())
    return _AIRLINE_ALIASES.get(s.lower(), s)


def _hhmm(val):
    """Extract a HH:MM string (handles trailing tokens like '14:15 26 Feb')."""
    m = re.search(r"(\d{1,2}:\d{2})", str(val))
    return m.group(1) if m else None


def _to_minutes(t) -> int:
    m = re.match(r"(\d{1,2}):(\d{2})", str(t))
    return int(m.group(1)) * 60 + int(m.group(2)) if m else 9999


def _price(val):
    s = re.sub(r"[^\d.]", "", str(val))
    try:
        return int(float(s)) if s else None
    except ValueError:
        return None


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=COLS)


def _load_goibibo() -> pd.DataFrame:
    pq = os.path.join(RAW, "goibibo_flights_data.parquet")
    csv = os.path.join(RAW, "goibibo_flights_data.csv")
    if os.path.exists(pq):
        df = pd.read_parquet(pq)
    elif os.path.exists(csv):
        df = pd.read_csv(csv, usecols=lambda c: not str(c).startswith("Unnamed"))
    else:
        return _empty()
    date = pd.to_datetime(df["flight date"], dayfirst=True, errors="coerce")
    out = pd.DataFrame({
        "flight_no": df["flight_num"], "airline": df["airline"],
        "origin": df["from"].map(canon_city), "destination": df["to"].map(canon_city),
        "dep_time": df["dep_time"].map(_hhmm), "arr_time": df["arr_time"].map(_hhmm),
        "day_of_week": date.dt.day_name(), "date": date.dt.date.astype("string"),
        "price": df["price"].map(_price), "cls": df["class"].astype(str).str.title(),
        "stops": df["stops"], "source_ds": "goibibo(2023)"})
    return out


def _load_flights() -> pd.DataFrame:
    p = os.path.join(RAW, "flights.csv")
    if not os.path.exists(p):
        return _empty()
    df = pd.read_csv(p)
    date = pd.to_datetime(df["date_of_journey"], errors="coerce")
    return pd.DataFrame({
        "flight_no": None, "airline": df["airline"],
        "origin": df["Source"].map(canon_city), "destination": df["destination"].map(canon_city),
        "dep_time": df["dep_time"].map(_hhmm), "arr_time": df["Arrival_time"].map(_hhmm),
        "day_of_week": date.dt.day_name(), "date": date.dt.date.astype("string"),
        "price": df["Price"].map(_price), "cls": "Economy",
        "stops": df["Total_stops"], "source_ds": "fares(2021-24)"})


def _load_schedule() -> pd.DataFrame:
    p = os.path.join(RAW, "Flight_Schedule.csv")
    if not os.path.exists(p):
        return _empty()
    df = pd.read_csv(p)
    return pd.DataFrame({
        "flight_no": df["flightNumber"].astype("string"), "airline": df["airline"],
        "origin": df["origin"].map(canon_city), "destination": df["destination"].map(canon_city),
        "dep_time": df["scheduledDepartureTime"].map(_hhmm),
        "arr_time": df["scheduledArrivalTime"].map(_hhmm),
        "day_of_week": df["dayOfWeek"].astype(str), "date": None,
        "price": None, "cls": "Economy", "stops": None, "source_ds": "schedule(2018-19)"})


@lru_cache(maxsize=1)
def combined() -> pd.DataFrame:
    frames = []
    for loader in (_load_goibibo, _load_flights, _load_schedule):
        try:
            frames.append(loader())
        except Exception:  # noqa: BLE001 - skip a bad file rather than fail everything
            pass
    df = pd.concat(frames, ignore_index=True) if frames else _empty()
    df = df.dropna(subset=["origin", "destination", "dep_time"])
    # Drop obvious test/junk rows and canonicalise airline names.
    df = df[~df["airline"].astype(str).str.contains("test", case=False, na=False)]
    df["airline"] = df["airline"].map(canon_airline)
    return df


@lru_cache(maxsize=1)
def available_cities() -> list[str]:
    df = combined()
    return sorted(set(df["origin"]) | set(df["destination"]))


def list_scheduled_flights(origin: str, destination: str, day_of_week: str | None = None,
                           travel_class: str | None = None, limit: int = 20) -> pd.DataFrame:
    """Real scheduled flights on a route with exact times and (where available) price.

    Deduplicates to representative flights (median price across observed dates).
    """
    o, d = canon_city(origin), canon_city(destination)
    df = combined()
    sub = df[(df.origin == o) & (df.destination == d)].copy()
    if day_of_week:
        dow = day_of_week.strip().title()
        sub = sub[sub.day_of_week.astype(str).str.contains(dow, case=False, na=False)]
    if travel_class:
        sub = sub[sub.cls.astype(str).str.contains(travel_class, case=False, na=False)]
    if sub.empty:
        return sub

    grp = (sub.groupby(["flight_no", "airline", "dep_time", "arr_time", "source_ds"], dropna=False)
           .agg(price=("price", "median"),
                stops=("stops", lambda s: s.dropna().iloc[0] if s.notna().any() else None),
                cls=("cls", lambda s: s.dropna().iloc[0] if s.notna().any() else None),
                observations=("dep_time", "size")).reset_index())
    grp["_min"] = grp["dep_time"].map(_to_minutes)
    grp = grp.sort_values("_min").drop(columns="_min")
    grp["price"] = grp["price"].round().astype("Int64")
    return grp.head(limit).reset_index(drop=True)


def route_summary(origin: str, destination: str) -> dict:
    o, d = canon_city(origin), canon_city(destination)
    sub = combined()
    sub = sub[(sub.origin == o) & (sub.destination == d)]
    if sub.empty:
        return {}
    price = sub["price"].dropna()
    return {"origin": o, "destination": d, "records": int(len(sub)),
            "airlines": sorted(sub["airline"].dropna().unique().tolist()),
            "price_min": int(price.min()) if len(price) else None,
            "price_median": int(price.median()) if len(price) else None,
            "sources": sorted(sub["source_ds"].unique().tolist())}


if __name__ == "__main__":
    df = combined()
    print(f"Combined schedule rows: {len(df):,}")
    print(f"Cities: {len(available_cities())}  e.g. {available_cities()[:15]}")
    print("\n=== Delhi -> Mumbai (representative flights) ===")
    print(list_scheduled_flights("Delhi", "Mumbai", limit=10).to_string(index=False))
    print("\n=== Delhi -> Hyderabad on Monday ===")
    print(list_scheduled_flights("Delhi", "Hyderabad", day_of_week="Monday", limit=8).to_string(index=False))
    print("\nsummary:", route_summary("Delhi", "Mumbai"))
