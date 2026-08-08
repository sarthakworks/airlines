"""Central configuration, loaded from environment variables (.env).

Every setting has a sensible default so the app can run locally with an
essentially empty .env (the one exception is the LLM key — see llm.py).
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root = the directory that contains this `app/` package.
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env from the project root (no-op if the file is absent).
load_dotenv(BASE_DIR / ".env")


def _get(name: str, default: str = "") -> str:
    return os.getenv(name, default)


# --------------------------------------------------------------------- paths
DATA_DIR = BASE_DIR / "data"
CSV_PATH = DATA_DIR / "Flights_Schedule_Data_v1.csv"
PDF_PATH = DATA_DIR / "Knowledge_Base_for_Airline_Info_and_FAQs.pdf"

# ----------------------------------------------------------------------- LLM
LLM_PROVIDER = _get("LLM_PROVIDER", "groq").lower()

GROQ_API_KEY = _get("GROQ_API_KEY")
GROQ_MODEL = _get("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_BASE_URL = _get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

OLLAMA_MODEL = _get("OLLAMA_MODEL", "llama3.1")
OLLAMA_BASE_URL = _get("OLLAMA_BASE_URL", "http://localhost:11434/v1")

LLM_TEMPERATURE = float(_get("LLM_TEMPERATURE", "0") or 0)

# ---------------------------------------------------------------- embeddings
EMBED_MODEL = _get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DIM = int(_get("EMBED_DIM", "384") or 384)

# -------------------------------------------------------------- vector store
VECTOR_STORE = _get("VECTOR_STORE", "chroma").lower()
CHROMA_DIR = str((BASE_DIR / _get("CHROMA_DIR", "./chroma_db").lstrip("./")).resolve())
CHROMA_COLLECTION = "airline_faq"

PINECONE_API_KEY = _get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = _get("PINECONE_INDEX_NAME", "airline-faq-index")
PINECONE_CLOUD = _get("PINECONE_CLOUD", "aws")
PINECONE_REGION = _get("PINECONE_REGION", "us-east-1")

# ------------------------------------------------------------------ database
DB_PARAMS = {
    "host": _get("DB_HOST", "localhost"),
    "port": _get("DB_PORT", "5432"),
    "user": _get("DB_USER", "airline"),
    "password": _get("DB_PASSWORD", "airline_pw"),
    "dbname": _get("DB_NAME", "airline"),
}

# ----------------------------------------------------------------------- api
API_BASE_URL = _get("API_BASE_URL", "http://localhost:8000")
API_PORT = int(_get("API_PORT", "8000") or 8000)
STREAMLIT_PORT = int(_get("STREAMLIT_PORT", "8501") or 8501)

# ------------------------------------------------------------------- retrieval
RETRIEVER_K = int(_get("RETRIEVER_K", "4") or 4)
CHUNK_SIZE = int(_get("CHUNK_SIZE", "800") or 800)
CHUNK_OVERLAP = int(_get("CHUNK_OVERLAP", "120") or 120)

# ------------------------------------------------------------------ guardrails
# Run the optional LLM-based safety moderation on top of the rule-based checks.
# Costs one extra LLM call per query; set to "false" to conserve free-tier quota.
GUARDRAIL_LLM = _get("GUARDRAIL_LLM", "true").lower() in ("1", "true", "yes", "on")
