from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from agent.state import AgentState
from config.settings import llm


class GroundednessCheck(BaseModel):
    grounded: str = Field(
        description="yes if answer is supported by context/price data, no if hallucinated"
    )
    reason: str = Field(description="brief reason")


class AnswerQualityCheck(BaseModel):
    answers_question: str = Field(
        description="yes if the response actually addresses what was asked, no otherwise"
    )
    reason: str = Field(description="brief reason")


groundedness_llm = llm.with_structured_output(GroundednessCheck)
quality_llm = llm.with_structured_output(AnswerQualityCheck)


def _normalize_binary(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized.startswith("y"):
        return "yes"
    if normalized.startswith("n"):
        return "no"
    return normalized


def _get_last_user_question(state: AgentState) -> str:
    messages = state.get("messages", []) or []
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            content = getattr(message, "content", "")
            return str(content or "")
        message_type = getattr(message, "type", "")
        if message_type == "human":
            content = getattr(message, "content", "")
            return str(content or "")
    return ""


def check_hallucination(state: AgentState) -> dict[str, Any]:
    current_retry = int(state.get("reflection_count", 0) or 0)
    current_steps = list(state.get("agent_steps", []) or [])
    draft_answer = str(state.get("draft_answer", "") or "")
    retrieved_context = str(state.get("retrieved_context", "") or "")
    price_result_str = str(state.get("price_result", {}) or {})
    question = _get_last_user_question(state)

    if current_retry >= 2:
        current_steps.append(
            "Hallucination grader: max retries reached — passing through"
        )
        return {
            "needs_retry": False,
            "hallucination_status": "reviewed",
            "reflection_count": current_retry,
            "agent_steps": current_steps,
        }

    combined_context = (
        f"Retrieved Context:\n{retrieved_context}\n\n"
        f"Price Data:\n{price_result_str}"
    )

    try:
        grounded_result = groundedness_llm.invoke(
            [
                HumanMessage(
                    content=(
                        "Does this answer rely only on the provided context/price data? "
                        "Answer yes or no.\n\n"
                        f"Context:\n{combined_context}\n\n"
                        f"Draft Answer:\n{draft_answer}"
                    )
                )
            ]
        )

        quality_result = quality_llm.invoke(
            [
                HumanMessage(
                    content=(
                        "Does this answer actually address the user question? "
                        "Answer yes or no.\n\n"
                        f"Question:\n{question}\n\n"
                        f"Draft Answer:\n{draft_answer}"
                    )
                )
            ]
        )

        grounded = _normalize_binary(grounded_result.grounded)
        answers_question = _normalize_binary(quality_result.answers_question)

        if grounded == "yes" and answers_question == "yes":
            current_steps.append(
                "Hallucination grader: VERIFIED — grounded and answers question"
            )
            return {
                "needs_retry": False,
                "hallucination_status": "verified",
                "reflection_count": current_retry,
                "agent_steps": current_steps,
            }

        updated_retry = current_retry + 1
        current_steps.append(
            "Hallucination grader: FLAGGED — "
            f"{grounded_result.reason} | {quality_result.reason}"
        )
        return {
            "needs_retry": True,
            "hallucination_status": "flagged",
            "reflection_count": updated_retry,
            "agent_steps": current_steps,
        }

    except Exception as exc:
        current_steps.append(f"Hallucination grader: error — {exc}")
        return {
            "needs_retry": False,
            "hallucination_status": "reviewed",
            "reflection_count": current_retry,
            "agent_steps": current_steps,
        }