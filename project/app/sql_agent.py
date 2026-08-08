"""AI agent that answers flight-data questions by executing SQL as a tool.

Primary path: a LangGraph ReAct agent with an `execute_flight_sql` tool.
Fallback path: if the model can't tool-call, we run the pre-generated SQL
directly and summarise the rows. Either way the SQL passes the guardrail and
runs in a read-only DB session.
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from app.database import execute_sql_query, result_to_text
from app.guardrails import check_sql
from app.llm import get_llm
from app.sql_agent_prompt import AGENT_SYSTEM
from app.sql_chain import generate_sql


@tool
def execute_flight_sql(sql: str) -> str:
    """Run ONE read-only PostgreSQL SELECT against the flights table and return
    the matching rows as text. The input must be a single SELECT statement."""
    gr = check_sql(sql)
    if not gr.allowed:
        return (f"BLOCKED by guardrail ({gr.category}): {gr.reason}. "
                "Rewrite it as a single read-only SELECT on the flights table.")
    return result_to_text(execute_sql_query(sql))


_agent = None


def _get_agent():
    global _agent
    if _agent is None:
        from langgraph.prebuilt import create_react_agent
        _agent = create_react_agent(get_llm(temperature=0), [execute_flight_sql])
    return _agent


def _direct_answer(query: str, sql: str) -> str:
    """Deterministic fallback: guardrail -> execute -> summarise."""
    gr = check_sql(sql)
    if not gr.allowed:
        return ("I couldn't build a safe database query for that. "
                "Please ask about a specific flight, route, date, fare, or status.")
    rows_text = result_to_text(execute_sql_query(sql))
    summary = get_llm(temperature=0).invoke(
        f"You are an airline support agent. Using ONLY the data below, answer the "
        f"customer's question concisely and clearly. If the data says no rows were "
        f"found, tell them no matching flights were found.\n\n"
        f"Question: {query}\n\nDatabase result:\n{rows_text}\n\nAnswer:"
    ).content
    return summary.strip()


def run_flight_query(query: str) -> dict:
    """Answer a flight-data question. Returns {answer, sql, rows_text}."""
    sql = generate_sql(query)
    try:
        agent = _get_agent()
        messages = [
            SystemMessage(content=AGENT_SYSTEM),
            HumanMessage(content=(
                f"Customer question: {query}\n\n"
                f"Suggested SQL (verify, then call the tool): {sql}"
            )),
        ]
        result = agent.invoke({"messages": messages})
        answer = result["messages"][-1].content.strip()
        if not answer:  # some models return empty after tool calls
            raise ValueError("empty agent answer")
        return {"answer": answer, "sql": sql, "rows_text": None}
    except Exception:  # noqa: BLE001 - fall back to deterministic path
        return {"answer": _direct_answer(query, sql), "sql": sql, "rows_text": None}
