"""LLM chain that turns a natural-language question into a read-only SQL query
for the `flights` table.

The dataset stores airports as IATA codes (DEL, BOM, BLR...) while users type
city names (Delhi, Mumbai, Bengaluru...), so the prompt carries an explicit
city->code map plus the full schema.
"""
from __future__ import annotations

import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.llm import get_llm

SCHEMA = """
Table: flights
Columns:
  id BIGINT, flight_no TEXT, airline_code TEXT, airline_name TEXT,
  origin TEXT (IATA code), destination TEXT (IATA code),
  departure_date DATE, departure_time TIME, arrival_date DATE, arrival_time TIME,
  status TEXT ('On Time' | 'Delayed' | 'Cancelled'),
  delay_minutes INTEGER, delay_reason TEXT, terminal TEXT, gate TEXT,
  aircraft_type TEXT, seats_total INTEGER, seats_booked INTEGER, fare_inr INTEGER
Note: seats available = seats_total - seats_booked.
"""

CITY_CODES = """
Delhi=DEL, New Delhi=DEL, Mumbai=BOM, Bombay=BOM, Bengaluru=BLR, Bangalore=BLR,
Chennai=MAA, Hyderabad=HYD, Kolkata=CCU, Pune=PNQ, Nagpur=NAG, Goa=GOI,
Ahmedabad=AMD, Kochi=COK, Cochin=COK, Jaipur=JAI, Lucknow=LKO, Varanasi=VNS,
Chandigarh=IXC, Guwahati=GAU, Patna=PAT, Bhubaneswar=BBI, Indore=IDR,
Thiruvananthapuram=TRV, Coimbatore=CJB, Visakhapatnam=VTZ, Srinagar=SXR
"""

_SQL_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a PostgreSQL expert for an airline support system. Convert the user's "
     "question into ONE valid, read-only PostgreSQL SELECT query over the `flights` "
     "table.\n\n"
     f"{SCHEMA}\n"
     "City name -> IATA airport code mapping (map any city the user mentions):\n"
     f"{CITY_CODES}\n\n"
     "RULES:\n"
     "1. Output ONLY the SQL. No markdown fences, no commentary, no trailing semicolon.\n"
     "2. SELECT statements only. Never INSERT/UPDATE/DELETE/DROP/ALTER/etc.\n"
     "3. Query only the flights table and only the columns listed above.\n"
     "4. origin/destination are IATA codes — translate city names using the map. "
     "Compare codes in UPPERCASE (e.g. origin = 'DEL').\n"
     "5. flight_no matching should be case-insensitive: use UPPER(flight_no) = 'XXXX'.\n"
     "6. Dates are 'YYYY-MM-DD'. If the user gives a date like '11 Nov 2026', use "
     "departure_date = '2026-11-11'.\n"
     "7. 'available seats' -> (seats_total - seats_booked). 'under 7000' -> fare_inr < 7000.\n"
     "8. For broad searches (routes, lists) add ORDER BY departure_time and LIMIT 50. "
     "For a specific flight, select the relevant columns.\n"
     "9. Select only the columns needed to answer, plus flight_no for context."),
    ("human", "Question: {query}\nSQL:"),
])


def _clean_sql(raw: str) -> str:
    """Strip markdown fences / labels and keep a single statement."""
    text = (raw or "").strip()
    # Remove ```sql ... ``` fences if present.
    text = re.sub(r"^```(?:sql)?", "", text, flags=re.I).strip()
    text = re.sub(r"```$", "", text).strip()
    # Drop a leading "sql:" / "SQL" label.
    text = re.sub(r"^sql\s*:?\s*", "", text, flags=re.I).strip()
    # Keep only up to the first semicolon (single statement).
    if ";" in text:
        text = text.split(";", 1)[0]
    return text.strip()


_chain = None


def generate_sql(query: str) -> str:
    """Generate a read-only SQL SELECT for the given natural-language query."""
    global _chain
    if _chain is None:
        _chain = _SQL_PROMPT | get_llm(temperature=0) | StrOutputParser()
    return _clean_sql(_chain.invoke({"query": query}))
