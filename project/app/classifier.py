"""Input classification chain.

Routes each user query into one of three lanes:
    need_sql        -> flight data lives in PostgreSQL (status, fares, seats...)
    non_sql         -> airline policy / FAQ, answered by the RAG agent
    out_of_context  -> unrelated to airline support -> polite fallback
"""
from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.llm import get_llm

VALID_LABELS = {"need_sql", "non_sql", "out_of_context"}

_CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are the router for an airline customer-support assistant. Classify the "
     "user's query into EXACTLY ONE of these labels and output ONLY the label:\n\n"
     "need_sql        -> needs live flight-record data from the flights database: "
     "flight status, delays, cancellations, departure/arrival times, gate, terminal, "
     "aircraft, seat availability, fare/price, or searching flights by route/date.\n"
     "non_sql         -> airline policy or FAQ that does NOT need the flight table: "
     "baggage rules, check-in/boarding times, refunds, cancellations policy, pets, "
     "special assistance, documents, name changes, general 'how do I...' airline questions.\n"
     "out_of_context  -> anything unrelated to this airline's support (trivia, general "
     "knowledge, coding, math, other companies, etc.).\n\n"
     "Output only one token: need_sql, non_sql, or out_of_context."),
    ("human", "Examples:\n"
     "Q: What is the status of flight 6E477 on 10 Nov 2026? -> need_sql\n"
     "Q: Show flights from Delhi to Goa under 7000 -> need_sql\n"
     "Q: How many seats are available on flight AI101? -> need_sql\n"
     "Q: How much free baggage is allowed on domestic flights? -> non_sql\n"
     "Q: Can I travel with my pet? -> non_sql\n"
     "Q: What is the airline's cancellation policy? -> non_sql\n"
     "Q: What is the capital of France? -> out_of_context\n"
     "Q: Explain generative AI. -> out_of_context\n\n"
     "Now classify this query:\nQ: {query} ->"),
])


def _normalise(raw: str) -> str:
    text = (raw or "").strip().lower()
    # Direct hit first, then substring fallback (model may add punctuation/words).
    for label in ("out_of_context", "need_sql", "non_sql"):
        if label in text.replace(" ", "_"):
            return label
    if "sql" in text and "non" not in text:
        return "need_sql"
    if "policy" in text or "faq" in text or "non" in text:
        return "non_sql"
    return "out_of_context"


# The runnable chain: prompt -> llm -> string. Kept module-level so it is built once.
def _build_chain():
    return _CLASSIFIER_PROMPT | get_llm(temperature=0) | StrOutputParser()


_chain = None


def classify(query: str) -> str:
    """Return one of: need_sql | non_sql | out_of_context."""
    global _chain
    if _chain is None:
        _chain = _build_chain()
    raw = _chain.invoke({"query": query})
    return _normalise(raw)
