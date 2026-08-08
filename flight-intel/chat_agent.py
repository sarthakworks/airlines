"""Natural-language agent over the flight-intel models.

A LangGraph ReAct agent whose tools wrap predict.py, so a user can ask e.g.
"cheapest reliable morning Delhi-Mumbai booked 3 weeks out" and get a grounded
answer. Reuses the free Groq key from ../project/.env.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

import pandas as pd

import predict as P
import route_analytics as RA
import schedules as SCH

# Reuse the M4 project's free Groq key (single source), allow a local override.
load_dotenv(Path(__file__).resolve().parent.parent / "project" / ".env")
load_dotenv(Path(__file__).resolve().parent / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

CITY_ALIASES = {
    "delhi": "Delhi", "new delhi": "Delhi", "del": "Delhi",
    "mumbai": "Mumbai", "bombay": "Mumbai", "bom": "Mumbai",
    "bangalore": "Bangalore", "bengaluru": "Bangalore", "blr": "Bangalore",
    "kolkata": "Kolkata", "calcutta": "Kolkata", "ccu": "Kolkata",
    "hyderabad": "Hyderabad", "hyd": "Hyderabad",
    "chennai": "Chennai", "madras": "Chennai", "maa": "Chennai",
}
TIME_ALIASES = {
    "early morning": "Early_Morning", "early_morning": "Early_Morning",
    "dawn": "Early_Morning", "morning": "Morning", "afternoon": "Afternoon",
    "evening": "Evening", "night": "Night", "late night": "Late_Night",
    "late_night": "Late_Night", "red eye": "Late_Night", "red-eye": "Late_Night",
}


def _city(name: str) -> str | None:
    return CITY_ALIASES.get(re.sub(r"[^a-z ]", "", str(name).lower()).strip())


def _time(name: str) -> str:
    return TIME_ALIASES.get(str(name).lower().replace("-", " ").strip(), "Morning")


def _weights(priority: str) -> tuple[float, float]:
    p = str(priority).lower()
    if any(w in p for w in ("cheap", "price", "fare", "budget")):
        return 1.0, 0.0
    if any(w in p for w in ("reliab", "on-time", "on time", "punctual", "delay")):
        return 0.0, 1.0
    return 0.5, 0.5


_CITIES_MSG = "I only have data for Delhi, Mumbai, Bangalore, Kolkata, Hyderabad, Chennai."


# --------------------------------------------------------------------- tools
@tool
def rank_airlines(source: str, destination: str, days_left: int = 15,
                  travel_class: str = "Economy", departure_time: str = "Morning",
                  priority: str = "balanced") -> str:
    """Rank airlines on a route by a blend of predicted fare and on-time reliability.
    priority is 'cheapest', 'balanced', or 'most_reliable'. Use this for "which
    airline is best/cheapest/most reliable for <route>"."""
    s, d = _city(source), _city(destination)
    if not s or not d:
        return _CITIES_MSG
    pw, rw = _weights(priority)
    df = P.rank_route(s, d, days_left=days_left, travel_class=travel_class,
                      departure_time=_time(departure_time), price_weight=pw,
                      reliability_weight=rw)
    if df.empty:
        return f"No fare data for {s} to {d}."
    head = f"Ranked airlines {s}->{d} ({travel_class}, {days_left} days ahead, {_time(departure_time)}, priority={priority}):"
    lines = [f"{i+1}. {r.airline}: ~Rs.{r.pred_price:,} , on-time {r.otp}% (score {r.score})"
             + (f" [{r.note}]" if r.note else "") for i, r in df.iterrows()]
    return head + "\n" + "\n".join(lines)


@tool
def cheapest_time_of_day(source: str, destination: str, days_left: int = 15,
                         travel_class: str = "Economy") -> str:
    """Show the cheapest departure time-of-day for a route (Economy by default)."""
    s, d = _city(source), _city(destination)
    if not s or not d:
        return _CITIES_MSG
    df = P.best_time(s, d, days_left=days_left, travel_class=travel_class)
    if df.empty:
        return f"No data for {s} to {d}."
    return f"Avg {travel_class} fare by departure time {s}->{d}:\n" + "\n".join(
        f"{r.time}: ~Rs.{r.avg_price:,.0f}" for _, r in df.iterrows())


@tool
def best_booking_window(source: str, destination: str, travel_class: str = "Economy",
                        departure_time: str = "Morning") -> str:
    """Show how fare changes with how many days ahead you book, for a route."""
    s, d = _city(source), _city(destination)
    if not s or not d:
        return _CITIES_MSG
    df = P.booking_curve(s, d, travel_class=travel_class, dep_time=_time(departure_time))
    if df.empty:
        return f"No data for {s} to {d}."
    cheapest = df.loc[df.avg_price.idxmin()]
    body = "\n".join(f"{int(r.days_left)} days ahead: ~Rs.{r.avg_price:,.0f}" for _, r in df.iterrows())
    return (f"Fare vs booking window {s}->{d} ({travel_class}):\n{body}\n"
            f"Cheapest ~{int(cheapest.days_left)} days ahead (~Rs.{cheapest.avg_price:,.0f}).")


AIRLINE_STATUS = {
    "GoAir": "defunct since 2023", "Vistara": "merged into Air India 2024-25",
    "Air Asia": "merged into Air India Express",
}


@tool
def airline_reliability_ranking() -> str:
    """Overall airline on-time performance ranking (mean OTP %, 2021-2026, all routes).
    Note which carriers are defunct/merged so they aren't recommended for future travel."""
    df = P.reliability_table()
    if df.empty:
        return "No reliability data available."
    rows = []
    for i, r in df.iterrows():
        note = AIRLINE_STATUS.get(r.airline, "")
        rows.append(f"{i+1}. {r.airline}: {r.otp}%" + (f" [{note}]" if note else ""))
    return ("Airline on-time performance (mean %, higher = better; "
            "prefer currently-operating carriers for future travel):\n" + "\n".join(rows))


@tool
def airline_complaints_ranking() -> str:
    """Airline grievance/complaint volumes (2021-2026, raw totals, all routes)."""
    df = P.complaints_table()
    if df.empty:
        return "No complaints data available."
    return ("Airline grievance volume (raw totals; larger carriers naturally get more):\n"
            + "\n".join(f"{i+1}. {r.airline}: {int(r.grievances):,}" for i, r in df.iterrows()))


@tool
def route_demand(source: str, destination: str) -> str:
    """Total passenger demand for a route (DGCA city-pair traffic)."""
    s, d = _city(source), _city(destination)
    if not s or not d:
        return _CITIES_MSG
    total, _ = P.route_demand(s, d)
    if not total:
        return f"No demand data for {s} to {d}."
    return f"{s} <-> {d}: ~{total/1e6:,.1f} million passengers (DGCA, all-time in dataset)."


@tool
def route_reliability_ranking(worst_first: bool = True) -> str:
    """Rank routes by an on-time RELIABILITY PROXY = the on-time record of the airlines
    operating each route, weighted by how much they serve it. Use for "worst/best route"
    questions. Covers the 6 metros only; metro routes share similar airline mixes so the
    differences are small."""
    df = RA.route_reliability()
    if df.empty:
        return "No route reliability data available."
    df = df if worst_first else df.sort_values("reliability_otp", ascending=False)
    label = "least" if worst_first else "most"
    rows = [f"{r.route}: {r.reliability_otp}% (airlines: {r.airlines})"
            for _, r in df.head(6).iterrows()]
    return (f"Routes by on-time reliability proxy ({label} reliable first; "
            f"based on operating airlines' OTP, 6 metros):\n" + "\n".join(rows))


@tool
def busiest_routes(n: int = 8, least_busy: bool = False) -> str:
    """Busiest (or least busy) domestic routes by total passengers, across all 1,933
    city-pairs. Set least_busy=True for the quietest routes."""
    df = RA.busiest_routes(n=n, ascending=least_busy)
    if df.empty:
        return "No route demand data available."
    label = "Least busy" if least_busy else "Busiest"
    return (f"{label} domestic routes (million passengers, all-time in DGCA data):\n"
            + "\n".join(f"{r.route}: {r.million_pax}M" for _, r in df.iterrows()))


@tool
def complaint_categories() -> str:
    """What KINDS of complaints are most common system-wide (baggage, refunds, check-in,
    meals, security, etc.). Use for 'most common complaint/issue' questions."""
    df = RA.complaint_categories()
    if df.empty:
        return "No complaint-category data available."
    return ("Complaint categories (raw totals, 2021-26; 'Others' is a catch-all):\n"
            + "\n".join(f"{r.category}: {int(r.total):,}" for _, r in df.iterrows()))


@tool
def list_flights_on_route(source: str, destination: str, days_left: int = 15,
                          travel_class: str = "Economy", departure_time: str = "",
                          top_n: int = 10) -> str:
    """List actual FLIGHT NUMBERS on a route with time-of-day, stops, estimated fare,
    on-time %, and a composite score (price + reliability), ranked best first. Use this
    when the user wants specific flights/flight numbers. NOTE: the dataset has time-of-day
    buckets (not exact clock times) and no day-of-week/date."""
    s, d = _city(source), _city(destination)
    if not s or not d:
        return _CITIES_MSG
    dt = _time(departure_time) if departure_time else None
    df = P.rank_flights(s, d, days_left=days_left, travel_class=travel_class,
                        departure_time=dt, top_n=top_n)
    if df.empty:
        return f"No flight-level data for {s} to {d}."
    lines = [f"{i+1}. {r.flight} ({r.airline}) — {r.departure_time}, {r.stops} stop(s), "
             f"~Rs.{int(r.price):,}, on-time {r.otp}% (score {r.score})"
             for i, r in df.iterrows()]
    return (f"Flights {s}->{d} ({travel_class}, ~{days_left} days before travel), "
            f"ranked by price + reliability:\n" + "\n".join(lines)
            + "\nNote: time-of-day buckets only — no exact clock time or day-of-week in this dataset.")


@tool
def scheduled_flights_on_route(source: str, destination: str, day_of_week: str = "",
                               travel_class: str = "", limit: int = 12) -> str:
    """Real flight schedules with EXACT departure/arrival clock times, flight numbers, and
    (where available) actual prices — from added historical datasets (2018-2024), covering
    ~84 cities INCLUDING non-metros. Use this whenever the user wants exact times, specific
    days ('on Monday'), a schedule, or 'flight number with time and cost'. day_of_week is
    optional (e.g. Monday)."""
    df = SCH.list_scheduled_flights(source, destination, day_of_week=day_of_week or None,
                                    travel_class=travel_class or None, limit=limit)
    o, d = SCH.canon_city(source), SCH.canon_city(destination)
    if df.empty:
        cities = SCH.available_cities()
        return (f"No scheduled-flight records for {o} to {d}. Covered cities include: "
                + ", ".join(cities[:40]) + " …")
    lines = []
    for _, r in df.iterrows():
        fno = r.flight_no if pd.notna(r.flight_no) else "(no flight #)"
        price = f"~Rs.{int(r.price):,}" if pd.notna(r.price) else "price n/a"
        arr = r.arr_time if pd.notna(r.arr_time) else "?"
        lines.append(f"{r.dep_time}->{arr}  {fno} {r.airline}  {price}  "
                     f"[{r.stops if pd.notna(r.stops) else '?'}, {r.source_ds}]")
    return (f"Scheduled flights {o}->{d}"
            + (f" on {day_of_week.title()}" if day_of_week else "")
            + " (historical 2018-2024 data, sorted by departure time):\n" + "\n".join(lines)
            + "\nNote: historical schedules/fares — times and prices are indicative, not live.")


TOOLS = [rank_airlines, cheapest_time_of_day, best_booking_window,
         airline_reliability_ranking, airline_complaints_ranking, route_demand,
         route_reliability_ranking, busiest_routes, complaint_categories,
         list_flights_on_route, scheduled_flights_on_route]

SYSTEM = (
    "You are an India domestic-flight advisor. Answer ONLY using the tools provided; "
    "never invent fares, on-time numbers, or airlines.\n"
    "TWO DATA SCOPES: (1) the FARE-MODEL + reliability tools cover 6 metros (Delhi, Mumbai, "
    "Bangalore, Kolkata, Hyderabad, Chennai) with time-of-day buckets (Early_Morning/Morning/"
    "Afternoon/Evening/Night/Late_Night) — use for predicted fares, best airline/time, booking "
    "window, reliability, complaints. (2) the SCHEDULE tool (scheduled_flights_on_route) covers "
    "~84 cities including non-metros with EXACT clock times, flight numbers, day-of-week and "
    "actual historical (2018-2024) prices.\n"
    "Interpret the user: convert '3 weeks out'->days_left=21, 'a month ahead'->30, "
    "'tomorrow/last minute'->1; map 'red-eye/late night'->Late_Night, 'morning'->Morning; "
    "'cheapest'->priority=cheapest, 'most reliable/on-time'->priority=most_reliable, else balanced.\n"
    "Pick the right tool(s), then give a concise, friendly answer with the concrete numbers "
    "returned. Note that fares are relative estimates from a 2022 dataset, not live prices, "
    "when quoting them. If a city/route isn't covered, say which cities are available.\n"
    "For 'worst/best route' use route_reliability_ranking — it's a PROXY from the operating "
    "airlines' on-time record (metros only, small differences); say so briefly. For "
    "busiest/quietest routes use busiest_routes; for 'most common complaints/issues' use "
    "complaint_categories. For EXACT clock times, specific days ('on Monday'), a schedule, or "
    "'flight number with time and cost' — prefer scheduled_flights_on_route (historical "
    "2018-2024, ~84 cities); say the times/prices are historical, not live. Use "
    "list_flights_on_route for the 6-metro modelled fares with time-of-day buckets only."
)


@lru_cache(maxsize=1)
def _get_agent():
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set — add it to ../project/.env")
    from langgraph.prebuilt import create_react_agent
    llm = ChatOpenAI(model=GROQ_MODEL, temperature=0, api_key=GROQ_API_KEY,
                     base_url=GROQ_BASE_URL, timeout=60, max_retries=2)
    return create_react_agent(llm, TOOLS)


def answer(query: str) -> str:
    """Answer a natural-language flight question. Returns the assistant's text."""
    try:
        agent = _get_agent()
        res = agent.invoke({"messages": [SystemMessage(content=SYSTEM),
                                         HumanMessage(content=query)]})
        txt = res["messages"][-1].content.strip()
        return txt or "Sorry, I couldn't work that out — try rephrasing."
    except Exception as exc:  # noqa: BLE001
        return f"Sorry, I hit an error answering that: {exc}"


if __name__ == "__main__":
    for q in [
        "What's the cheapest reliable morning flight from Delhi to Mumbai booked 3 weeks out?",
        "Best time of day to fly Bangalore to Delhi?",
        "How far ahead should I book Chennai to Kolkata?",
        "Which airline is most on-time?",
    ]:
        print("\nQ:", q)
        print("A:", answer(q))
