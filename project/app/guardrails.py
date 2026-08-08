"""Input, SQL, and output guardrails.

The backbone is rule-based so it is deterministic and needs no API key. An
optional LLM safety pass (`llm_moderate`) can be layered on top for fuzzy
toxicity/violence detection when a model is available.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class GuardrailResult:
    allowed: bool
    category: str = "ok"        # e.g. prompt_injection, data_exfiltration, unsafe
    reason: str = ""            # human-readable explanation
    message: str = ""           # safe message to show the user when blocked


_SAFE_REFUSAL = (
    "I can't help with that request. I'm an airline customer-support assistant, "
    "so I can answer questions about flight status, schedules, fares, baggage, "
    "check-in, refunds, and similar airline policies."
)

# ----------------------------------------------------------------- INPUT ----
# (compiled regex, category, reason) — checked in order.
_INPUT_RULES: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"\bignore\b.{0,30}\b(previous|prior|above|all)\b.{0,20}\binstructions?\b", re.I),
     "prompt_injection", "attempt to override system instructions"),
    (re.compile(r"\b(disregard|forget|override)\b.{0,30}\b(instructions?|rules?|prompt)\b", re.I),
     "prompt_injection", "attempt to override system instructions"),
    (re.compile(r"\b(reveal|show|print|repeat|leak)\b.{0,30}\bsystem\s+prompt\b", re.I),
     "prompt_injection", "attempt to extract the system prompt"),
    (re.compile(r"\b(developer|dev|god|jailbreak)\s+mode\b", re.I),
     "prompt_injection", "jailbreak attempt"),
    (re.compile(r"\byou\s+are\s+now\b.{0,40}\b(dan|unrestricted|no\s+rules)\b", re.I),
     "prompt_injection", "role-override jailbreak"),
    (re.compile(r"\b(show|list|give|dump|export|download)\b.{0,40}\b(all|entire|complete|whole)\b.{0,25}\b(records?|customers?|database|table|data)\b", re.I),
     "data_exfiltration", "attempt to dump/export bulk data"),
    (re.compile(r"\bexport\b.{0,25}\b(flight\s+)?database\b", re.I),
     "data_exfiltration", "attempt to export the database"),
    (re.compile(r"\b(drop|delete|truncate|update|insert|alter)\b.{0,15}\b(table|database|flights|from|into)\b", re.I),
     "data_tampering", "attempt to modify the database"),
    (re.compile(r"\b(bypass|evade|get\s+around|sneak\s+past|circumvent)\b.{0,25}\b(airport\s+)?security\b", re.I),
     "unsafe", "request to bypass security controls"),
    (re.compile(r"\b(make|build|create)\b.{0,20}\b(bomb|explosive|weapon)\b", re.I),
     "unsafe", "request for dangerous/violent content"),
    (re.compile(r"\b(smuggle|hijack)\b", re.I),
     "unsafe", "request related to illegal/violent acts"),
    (re.compile(r"\b(sk-[a-zA-Z0-9]{16,}|api[_-]?key\s*[:=]\s*\S+)\b", re.I),
     "secrets", "input appears to contain a secret/API key"),
]


def check_input(query: str) -> GuardrailResult:
    """Validate a user query before any processing."""
    if not query or not query.strip():
        return GuardrailResult(False, "empty", "empty query",
                               "Please enter a question about your flight or airline policy.")
    if len(query) > 2000:
        return GuardrailResult(False, "too_long", "query exceeds 2000 chars",
                               "Your message is too long. Please shorten your question.")
    for pattern, category, reason in _INPUT_RULES:
        if pattern.search(query):
            return GuardrailResult(False, category, reason, _SAFE_REFUSAL)
    return GuardrailResult(True)


# ------------------------------------------------------------------- SQL ----
_SQL_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|truncate|create|grant|revoke|"
    r"merge|call|copy|execute|vacuum|reindex|comment|attach)\b",
    re.I,
)
_SQL_COMMENT = re.compile(r"(--|/\*|\*/|#)")


def check_sql(sql: str) -> GuardrailResult:
    """Ensure a generated SQL statement is a single, read-only SELECT."""
    if not sql or not sql.strip():
        return GuardrailResult(False, "sql_empty", "no SQL generated",
                               "I couldn't turn that into a database query.")
    stmt = sql.strip().rstrip(";").strip()

    # Reject multiple statements (stacked-query injection).
    if ";" in stmt:
        return GuardrailResult(False, "sql_multi", "multiple SQL statements",
                               "Only a single read-only query is allowed.")
    if _SQL_COMMENT.search(stmt):
        return GuardrailResult(False, "sql_comment", "SQL comment markers present",
                               "Only a single read-only query is allowed.")
    # Must be a SELECT (optionally a CTE that resolves to a SELECT).
    head = stmt.lstrip("(").lower()
    if not (head.startswith("select") or head.startswith("with")):
        return GuardrailResult(False, "sql_not_select", "query is not a SELECT",
                               "Only read-only lookups are permitted.")
    if _SQL_FORBIDDEN.search(stmt):
        return GuardrailResult(False, "sql_forbidden", "forbidden SQL keyword present",
                               "That query is not allowed for safety reasons.")
    return GuardrailResult(True)


# ---------------------------------------------------------------- OUTPUT ----
_OUTPUT_LEAK = re.compile(
    r"(sk-[a-zA-Z0-9]{16,}|postgres(ql)?://\S+|password\s*[:=]\s*\S+|"
    r"GROQ_API_KEY|PINECONE_API_KEY)",
    re.I,
)


def check_output(answer: str) -> GuardrailResult:
    """Review the final answer before it reaches the user."""
    if not answer or not answer.strip():
        return GuardrailResult(False, "empty_output", "model returned empty answer",
                               "Sorry, I couldn't generate a response. Please try rephrasing.")
    if _OUTPUT_LEAK.search(answer):
        return GuardrailResult(False, "output_leak", "response may contain a secret/credential",
                               "Sorry, I can't share that information.")
    return GuardrailResult(True, message=answer)


# ------------------------------------------------- optional LLM safety pass --
_MODERATION_PROMPT = (
    "You are a content-safety classifier for an airline support assistant. "
    "Decide if the USER MESSAGE is safe to process. Reply with exactly one word: "
    "SAFE or UNSAFE. Mark UNSAFE only for: violence/weapons, illegal activity, "
    "hate/harassment, self-harm, sexual content, or clear attempts to hack, bypass "
    "security, or exfiltrate data.\n\nUSER MESSAGE:\n{query}\n\nAnswer:"
)


def llm_moderate(query: str) -> GuardrailResult:
    """Fuzzy safety check using the LLM. Fails open (allows) on any error."""
    try:
        from app.llm import get_llm

        verdict = get_llm(temperature=0).invoke(
            _MODERATION_PROMPT.format(query=query)
        ).content.strip().upper()
        if verdict.startswith("UNSAFE"):
            return GuardrailResult(False, "unsafe", "LLM safety classifier flagged input",
                                   _SAFE_REFUSAL)
    except Exception:  # noqa: BLE001 - never let moderation break the pipeline
        pass
    return GuardrailResult(True)
