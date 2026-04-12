from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from agent.state import AgentState
from config.settings import llm


class GradeDocument(BaseModel):
    binary_score: str = Field(description="yes or no")


# Use json_mode for more reliable structured output
try:
    grader_llm = llm.with_structured_output(GradeDocument, method="json_mode")
except Exception:
    grader_llm = llm.with_structured_output(GradeDocument)


def _extract_user_question(state: AgentState) -> str:
    messages = state.get("messages", [])
    if not messages:
        return ""

    last_message = messages[-1]

    content = getattr(last_message, "content", "")
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part).strip()

    return str(content).strip()


def grade_documents(state: AgentState) -> dict[str, Any]:
    try:
        question = _extract_user_question(state)
        retrieved_context = state.get("retrieved_context", "") or ""
        agent_steps = list(state.get("agent_steps", []))

        if not retrieved_context.strip():
            agent_steps.append(
                "Document grader: No retrieved context available to grade"
            )
            return {
                "retrieved_context": "",
                "agent_steps": agent_steps,
            }

        chunks = [chunk.strip() for chunk in retrieved_context.split("\n\n") if chunk.strip()]
        total_chunks = len(chunks)

        if total_chunks == 0:
            agent_steps.append(
                "Document grader: No retrieved context available to grade"
            )
            return {
                "retrieved_context": "",
                "agent_steps": agent_steps,
            }

        kept_chunks: list[str] = []

        for chunk in chunks:
            prompt = (
                "You are a CRAG-style document relevance grader.\n"
                "Determine whether the retrieved document chunk is relevant to answering "
                "the user's question.\n\n"
                "Respond with a binary score:\n"
                "- yes: if the chunk contains information that helps answer the question\n"
                "- no: if the chunk is irrelevant, off-topic, or not useful\n\n"
                f"User question:\n{question}\n\n"
                f"Document chunk:\n{chunk}"
            )

            try:
                result = grader_llm.invoke([HumanMessage(content=prompt)])
                score = result.binary_score.strip().lower() if hasattr(result, "binary_score") else "yes"
            except Exception:
                # Fallback: call plain llm and parse text
                plain = llm.invoke([HumanMessage(content=prompt + "\n\nReply with only: yes or no")])
                score = str(getattr(plain, "content", "yes")).strip().lower()[:3]
            if score == "yes":
                kept_chunks.append(chunk)

        if not kept_chunks:
            agent_steps.append(
                "Document grader: No relevant chunks found — knowledge base cannot answer this query"
            )
            return {
                "retrieved_context": "",
                "agent_steps": agent_steps,
            }

        filtered_context = "\n\n".join(kept_chunks)
        agent_steps.append(
            f"Document grader: {len(kept_chunks)}/{total_chunks} chunks passed relevance check"
        )

        return {
            "retrieved_context": filtered_context,
            "agent_steps": agent_steps,
        }

    except Exception as exc:
        agent_steps = list(state.get("agent_steps", []))
        agent_steps.append(f"Document grader error: {exc}")
        return {
            "retrieved_context": state.get("retrieved_context", "") or "",
            "agent_steps": agent_steps,
        }