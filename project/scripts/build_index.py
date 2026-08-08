"""Build the RAG vector index from the FAQ PDF.

Usage:  python scripts/build_index.py [--force]
Uses local fastembed embeddings (no API key). Writes to Chroma by default, or
Pinecone if VECTOR_STORE=pinecone.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402
from app.rag import build_index  # noqa: E402


def main() -> None:
    force = "--force" in sys.argv
    if not config.PDF_PATH.exists():
        print(f"PDF not found at {config.PDF_PATH}. Run scripts/download_data.py first.")
        sys.exit(1)
    print(f"Building '{config.VECTOR_STORE}' index with embeddings '{config.EMBED_MODEL}' …")
    n = build_index(force=force)
    print(f"✓ indexed {n} chunks")
    if config.VECTOR_STORE == "chroma":
        print(f"  persisted to {config.CHROMA_DIR}")


if __name__ == "__main__":
    main()
