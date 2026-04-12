from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_community.vectorstores import Chroma

from config.settings import CHROMA_DB_PATH, embeddings

_COLLECTION_NAME = "langchain"


def get_vectorstore() -> Chroma:
    """
    Load or create a Chroma vector store at the configured persistence path.

    Returns:
        Chroma: The initialized Chroma vector store instance.
    """
    db_path = Path(CHROMA_DB_PATH)
    db_path.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(db_path),
    )


def search_documents(query: str, k: int = 3) -> list[dict[str, Any]]:
    """
    Search the vector store for the top-k most similar document chunks.

    Args:
        query: The user query string.
        k: Number of top results to return.

    Returns:
        list[dict[str, Any]]: A list of search results in the format:
            {
                "content": str,
                "source": str,
                "page": int
            }

        Returns an empty list if the vector store has no indexed documents
        or if the query is empty.
    """
    if not query or not query.strip():
        return []

    vectorstore = get_vectorstore()

    if not is_vectorstore_ready():
        return []

    documents = vectorstore.similarity_search(query=query, k=max(1, k))

    results: list[dict[str, Any]] = []
    for doc in documents:
        metadata = doc.metadata or {}
        page_value = metadata.get("page", metadata.get("page_number", 0))

        try:
            page = int(page_value)
        except (TypeError, ValueError):
            page = 0

        results.append(
            {
                "content": doc.page_content,
                "source": str(metadata.get("source", "")),
                "page": page,
            }
        )

    return results


def is_vectorstore_ready() -> bool:
    """
    Check whether the vector store contains any indexed documents.

    Returns:
        bool: True if the vector store has documents, otherwise False.
    """
    try:
        vectorstore = get_vectorstore()
        collection = vectorstore._collection
        count = collection.count()
        return count > 0
    except Exception:
        return False