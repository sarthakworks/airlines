"""End-to-end orchestration — the single entry point the API and UI call.

    input guardrail -> classify -> route (SQL agent | RAG | fallback) -> output guardrail

Every stage degrades gracefully: a missing LLM key or a backend error returns a
clear, safe message instead of a stack trace.
"""
from __future__ import annotations

from typing import Any

from app import config, guardrails
from app.classifier import classify
from app.llm import LLMConfigError, get_llm

_LLM_SETUP_MSG = (
    "The language model isn't configured yet. Add a free GROQ_API_KEY to your .env "
    "(https://console.groq.com/keys) or set LLM_PROVIDER=ollama for a local model."
)


def _fallback_answer(query: str) -> str:
    """Final LLM response for out-of-context queries (polite redirect)."""
    try:
        return get_llm(temperature=0).invoke(
            "You are an airline customer-support assistant. The user's question is "
            "outside airline support. Politely, in 1-2 sentences, say it's outside "
            "what you can help with and invite an airline-related question. Do not "
            "answer the off-topic question.\n\nUser: " + query
        ).content.strip()
    except LLMConfigError:
        raise
    except Exception:  # noqa: BLE001
        return ("I'm an airline customer-support assistant, so I can only help with "
                "flights, fares, baggage, check-in, refunds, and similar topics.")


def _result(route: str, answer: str, *, sql: str | None = None,
            sources: list[str] | None = None, blocked: bool = False,
            guardrail: dict | None = None) -> dict[str, Any]:
    return {"route": route, "answer": answer, "sql": sql, "sources": sources,
            "blocked": blocked, "guardrail": guardrail}


def answer_query(query: str) -> dict[str, Any]:
    """Run one query through the full pipeline and return a structured result."""
    query = (query or "").strip()

    # 1) Input guardrail (rule-based, deterministic).
    ig = guardrails.check_input(query)
    if not ig.allowed:
        return _result("blocked_input", ig.message, blocked=True,
                       guardrail={"stage": "input", "category": ig.category, "reason": ig.reason})

    # 1b) Optional LLM safety moderation (fails open if model unavailable).
    if config.GUARDRAIL_LLM:
        lm = guardrails.llm_moderate(query)
        if not lm.allowed:
            return _result("blocked_input", lm.message, blocked=True,
                           guardrail={"stage": "input_llm", "category": lm.category, "reason": lm.reason})

    # 2) Classify + 3) route.
    try:
        route = classify(query)

        if route == "need_sql":
            from app.sql_agent import run_flight_query
            res = run_flight_query(query)
            answer, sql, sources = res["answer"], res["sql"], None
        elif route == "non_sql":
            from app.rag import rag_answer
            res = rag_answer(query)
            answer, sql, sources = res["answer"], None, res["sources"]
        else:  # out_of_context
            answer, sql, sources = _fallback_answer(query), None, None

    except LLMConfigError:
        return _result("error", _LLM_SETUP_MSG)
    except Exception as exc:  # noqa: BLE001
        return _result("error",
                       "Sorry, something went wrong while handling your request. "
                       "Please try again.",
                       guardrail={"stage": "exception", "category": "error", "reason": str(exc)})

    # 4) Output guardrail.
    og = guardrails.check_output(answer)
    if not og.allowed:
        return _result("blocked_output", og.message, sql=sql, blocked=True,
                       guardrail={"stage": "output", "category": og.category, "reason": og.reason})

    return _result(route, answer, sql=sql, sources=sources)


def warmup() -> None:
    """Best-effort pre-build of the retriever so the first query is fast."""
    try:
        from app.rag import get_retriever
        get_retriever()
    except Exception:  # noqa: BLE001
        pass
