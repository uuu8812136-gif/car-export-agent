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


def hybrid_search(query: str, k: int = 5) -> list[dict]:
    """
    Hybrid RAG: 向量检索 + BM25 关键词检索，结果合并去重排序。
    2026最佳实践：两路检索覆盖语义和关键词，召回率提升30%+。
    """
    from rank_bm25 import BM25Okapi

    # 向量检索
    vector_results = search_documents(query, k=k)

    # BM25 关键词检索（从已有向量库的文档集中构建）
    try:
        vs = get_vectorstore()
        collection = vs._collection
        all_docs = collection.get(include=["documents", "metadatas"])
        corpus = all_docs.get("documents") or []
        metadatas = all_docs.get("metadatas") or []

        if corpus:
            tokenized = [doc.lower().split() for doc in corpus]
            bm25 = BM25Okapi(tokenized)
            scores = bm25.get_scores(query.lower().split())
            top_k_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

            bm25_results = [
                {
                    "content": corpus[i],
                    "source": str((metadatas[i] or {}).get("source", "")),
                    "page": int((metadatas[i] or {}).get("page", 0)),
                    "score": float(scores[i]),
                    "method": "bm25",
                }
                for i in top_k_idx if scores[i] > 0
            ]
        else:
            bm25_results = []
    except Exception:
        bm25_results = []

    # 标记向量检索来源
    for r in vector_results:
        r.setdefault("method", "vector")

    # 合并去重（按 content 去重，保留第一次出现）
    seen = set()
    merged = []
    for r in vector_results + bm25_results:
        key = r["content"][:100]
        if key not in seen:
            seen.add(key)
            merged.append(r)

    return merged[:k]


def hyde_search(query: str, k: int = 5, llm=None) -> list[dict]:
    """
    HyDE (Hypothetical Document Embeddings): 先生成假设性答案，用答案做检索。
    适合模糊或不完整的查询，召回率提升20-40%。
    """
    if llm is None:
        from config.settings import llm as _llm
        llm = _llm

    # 生成假设性文档
    try:
        from langchain_core.messages import HumanMessage
        hyde_prompt = (
            f"Write a detailed answer about the following question regarding car export:\n{query}\n\n"
            "Answer as if you are an automotive export specialist. Be specific about models, prices, and specs."
        )
        hypothetical_doc = llm.invoke([HumanMessage(content=hyde_prompt)])
        enhanced_query = getattr(hypothetical_doc, "content", str(hypothetical_doc))
    except Exception:
        enhanced_query = query  # fallback to original query

    # 用生成的假设答案做向量检索
    return search_documents(enhanced_query, k=k)