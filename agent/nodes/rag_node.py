from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from agent.state import AgentState
from config.prompts import RAG_ANSWER_PROMPT
from config.settings import llm
from rag.vectorstore import get_vectorstore


def _get_last_user_message(messages: Sequence[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            content = message.content
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                parts: List[str] = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text = item.get("text", "")
                        if text:
                            parts.append(str(text))
                return "\n".join(parts).strip()
            return str(content).strip()
    return ""


def _format_source(metadata: Optional[Dict[str, Any]], index: int) -> str:
    if not metadata:
        return f"chunk_{index}"

    source = (
        metadata.get("source")
        or metadata.get("file_name")
        or metadata.get("filename")
        or metadata.get("document_name")
        or metadata.get("title")
        or metadata.get("manual")
        or f"chunk_{index}"
    )

    page = metadata.get("page")
    section = metadata.get("section")
    chunk_id = metadata.get("chunk_id")

    parts = [str(source)]
    if page is not None:
        parts.append(f"page={page}")
    if section:
        parts.append(f"section={section}")
    if chunk_id is not None:
        parts.append(f"chunk={chunk_id}")

    return " | ".join(parts)


def _build_retrieved_context(documents: Sequence[Document]) -> tuple[str, List[str]]:
    context_blocks: List[str] = []
    sources: List[str] = []

    for idx, doc in enumerate(documents, start=1):
        source_label = _format_source(doc.metadata, idx)
        sources.append(source_label)
        page_content = doc.page_content.strip()
        block = f"[Source {idx}: {source_label}]\n{page_content}"
        context_blocks.append(block)

    return "\n\n".join(context_blocks), sources


def _invoke_llm(question: str, retrieved_context: str) -> str:
    prompt = f"{RAG_ANSWER_PROMPT}\n\nUser question: {question}\n\nRetrieved context:\n{retrieved_context}"
    response = llm.invoke(prompt)

    if isinstance(response, AIMessage):
        content = response.content
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text", "")
                    if text:
                        parts.append(str(text))
            return "\n".join(parts).strip()
        return str(content).strip()

    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    return str(content).strip()


def retrieve_and_answer(state: AgentState) -> dict:
    messages = state.get("messages", [])
    agent_steps = list(state.get("agent_steps", []))
    user_query = _get_last_user_message(messages)

    vectorstore = get_vectorstore()
    if vectorstore is None:
        draft_answer = "Product manual not loaded yet"
        agent_steps.append("RAG skipped: vectorstore not initialized")
        return {
            "draft_answer": draft_answer,
            "retrieved_context": "",
            "agent_steps": agent_steps,
        }

    documents = vectorstore.similarity_search(user_query, k=3) if user_query else []
    retrieved_context, sources = _build_retrieved_context(documents)

    if documents:
        draft_answer = _invoke_llm(user_query, retrieved_context)
        unique_sources = sorted(set(sources))
        agent_steps.append(
            f"RAG retrieved {len(documents)} chunks from {', '.join(unique_sources)}"
        )
    else:
        draft_answer = _invoke_llm(
            user_query,
            "No relevant product manual content was retrieved from the vector store.",
        )
        agent_steps.append("RAG retrieved 0 chunks from no sources")

    source_tag = (
        "\n\n---\n"
        "📚 *Source: **Product Knowledge Base (RAG)** — "
        "Answer is based on retrieved document context, not AI parametric memory.*"
    )
    draft_answer = draft_answer + source_tag

    return {
        "draft_answer": draft_answer,
        "retrieved_context": retrieved_context,
        "agent_steps": agent_steps,
    }