"""Retrieval-Augmented Generation over the Airline Info & FAQ knowledge base.

* Embeddings : fastembed (local ONNX, no API key)
* Vector store: Chroma (local, default) or Pinecone (if VECTOR_STORE=pinecone)
* Chain      : retrieve -> augment prompt -> LLM -> grounded answer
"""
from __future__ import annotations

import os
from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from app import config
from app.llm import get_llm


# --------------------------------------------------------------- embeddings
@lru_cache(maxsize=1)
def get_embeddings():
    try:
        from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
    except ImportError:  # older/newer layout
        from langchain_community.embeddings import FastEmbedEmbeddings  # type: ignore
    return FastEmbedEmbeddings(model_name=config.EMBED_MODEL)


# ------------------------------------------------------- load + split PDF ---
def load_and_split() -> list[Document]:
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    docs = PyMuPDFLoader(str(config.PDF_PATH)).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


# ------------------------------------------------------------ vector store --
def _chroma_exists() -> bool:
    return os.path.isfile(os.path.join(config.CHROMA_DIR, "chroma.sqlite3"))


def build_index(force: bool = False) -> int:
    """Build (or rebuild) the vector index from the FAQ PDF. Returns chunk count."""
    chunks = load_and_split()
    embeddings = get_embeddings()

    if config.VECTOR_STORE == "pinecone":
        _build_pinecone(chunks, embeddings)
    else:
        from langchain_chroma import Chroma
        if force and _chroma_exists():
            import shutil
            shutil.rmtree(config.CHROMA_DIR, ignore_errors=True)
        os.makedirs(config.CHROMA_DIR, exist_ok=True)
        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=config.CHROMA_COLLECTION,
            persist_directory=config.CHROMA_DIR,
        )
    return len(chunks)


def _build_pinecone(chunks, embeddings):
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone, ServerlessSpec

    if not config.PINECONE_API_KEY:
        raise RuntimeError("VECTOR_STORE=pinecone but PINECONE_API_KEY is empty.")
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    existing = [i["name"] for i in pc.list_indexes()]
    if config.PINECONE_INDEX_NAME not in existing:
        pc.create_index(
            name=config.PINECONE_INDEX_NAME,
            dimension=config.EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
        )
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=config.PINECONE_INDEX_NAME,
    )


@lru_cache(maxsize=1)
def get_vectorstore():
    """Load the existing vector store (building Chroma on the fly if missing)."""
    embeddings = get_embeddings()
    if config.VECTOR_STORE == "pinecone":
        from langchain_pinecone import PineconeVectorStore
        return PineconeVectorStore(
            index_name=config.PINECONE_INDEX_NAME, embedding=embeddings
        )
    from langchain_chroma import Chroma
    if not _chroma_exists():
        build_index()
        get_vectorstore.cache_clear()  # rebuild cache after first build
    return Chroma(
        collection_name=config.CHROMA_COLLECTION,
        embedding_function=embeddings,
        persist_directory=config.CHROMA_DIR,
    )


@lru_cache(maxsize=1)
def get_retriever():
    return get_vectorstore().as_retriever(search_kwargs={"k": config.RETRIEVER_K})


# ---------------------------------------------------------------- RAG chain -
_RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful airline customer-support assistant. Answer the customer's "
     "question using ONLY the context below, which comes from the airline's official "
     "Info & FAQ knowledge base. If the answer isn't in the context, say you don't have "
     "that information and suggest contacting airline support. Be concise and friendly.\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])


def format_docs(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(d.page_content for d in docs)


_rag_chain = None


def _build_rag_chain():
    retriever = get_retriever()
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | _RAG_PROMPT
        | get_llm(temperature=0)
        | StrOutputParser()
    )


def rag_answer(query: str) -> dict:
    """Answer a policy/FAQ question from the knowledge base. Returns {answer, sources}."""
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = _build_rag_chain()
    answer = _rag_chain.invoke(query).strip()
    # Also surface the retrieved snippets for transparency in the UI.
    try:
        docs = get_retriever().invoke(query)
        sources = [d.page_content[:200] for d in docs]
    except Exception:  # noqa: BLE001
        sources = []
    return {"answer": answer, "sources": sources}
