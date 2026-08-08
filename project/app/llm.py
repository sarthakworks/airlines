"""LLM factory.

Both supported providers speak the OpenAI-compatible protocol, so a single
`ChatOpenAI` wrapper covers them:

* groq   -> Groq free tier   (needs a free GROQ_API_KEY, no credit card)
* ollama -> local Ollama     (no account at all; `ollama serve` on localhost)
"""
from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI

from app import config


class LLMConfigError(RuntimeError):
    """Raised when the selected provider is not configured correctly."""


@lru_cache(maxsize=None)
def get_llm(temperature: float | None = None) -> ChatOpenAI:
    """Return a cached ChatOpenAI client for the configured provider."""
    temp = config.LLM_TEMPERATURE if temperature is None else temperature

    if config.LLM_PROVIDER == "groq":
        if not config.GROQ_API_KEY:
            raise LLMConfigError(
                "LLM_PROVIDER=groq but GROQ_API_KEY is empty. Create a free key "
                "at https://console.groq.com/keys and put it in your .env, or set "
                "LLM_PROVIDER=ollama to run a fully-local model instead."
            )
        return ChatOpenAI(
            model=config.GROQ_MODEL,
            temperature=temp,
            api_key=config.GROQ_API_KEY,
            base_url=config.GROQ_BASE_URL,
            timeout=60,
            max_retries=2,
        )

    if config.LLM_PROVIDER == "ollama":
        return ChatOpenAI(
            model=config.OLLAMA_MODEL,
            temperature=temp,
            api_key="ollama",  # Ollama ignores the key but the client requires one
            base_url=config.OLLAMA_BASE_URL,
            timeout=120,
            max_retries=1,
        )

    raise LLMConfigError(
        f"Unknown LLM_PROVIDER={config.LLM_PROVIDER!r}. Use 'groq' or 'ollama'."
    )


def llm_available() -> tuple[bool, str]:
    """Cheap readiness check used by /health and the UI (no network call)."""
    if config.LLM_PROVIDER == "groq":
        if not config.GROQ_API_KEY:
            return False, "GROQ_API_KEY missing"
        return True, f"groq:{config.GROQ_MODEL}"
    if config.LLM_PROVIDER == "ollama":
        return True, f"ollama:{config.OLLAMA_MODEL}"
    return False, f"bad provider {config.LLM_PROVIDER}"
