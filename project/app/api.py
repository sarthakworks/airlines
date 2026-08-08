"""FastAPI backend exposing the airline support pipeline as a REST API."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app import config
from app.database import table_status
from app.llm import llm_available
from app.pipeline import answer_query, warmup


class ChatRequest(BaseModel):
    query: str = Field(..., description="User's airline-support question")


class ChatResponse(BaseModel):
    route: str
    answer: str
    sql: str | None = None
    sources: list[str] | None = None
    blocked: bool = False
    guardrail: dict | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    warmup()  # pre-build the retriever/vector store
    yield


app = FastAPI(
    title="AI-Powered Airline Customer Support System",
    description="LangChain + LangGraph pipeline: classifier -> SQL agent / RAG -> guardrails.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {"service": "airline-support", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    llm_ok, llm_info = llm_available()
    db = table_status()
    return {
        "status": "ok",
        "llm": {"ready": llm_ok, "info": llm_info, "provider": config.LLM_PROVIDER},
        "database": db,
        "vector_store": config.VECTOR_STORE,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    return answer_query(req.query)
