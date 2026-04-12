from __future__ import annotations

import json
from typing import Any, Dict

from langchain_core.messages import HumanMessage

from config.prompts import REFLECTION_PROMPT
from agent.state import AgentState
from config.settings import llm


def reflect_on_answer(state: AgentState) -> Dict[str, Any]:
    """
    Reflect on the current draft answer and decide whether the agent should retry.

    Logic:
    1. If reflection_count >= 2: set needs_retry=False and return immediately.
    2. Build an evaluation prompt using the original question, draft answer, and retrieved context.
    3. Call the LLM with REFLECTION_PROMPT.
    4. Parse JSON response: {"score": int, "reason": str, "needs_retry": bool}
    5. Update reflection_score with the score from the response.
    6. If score < 6: set needs_retry=True and increment reflection_count.
    7. Else: set needs_retry=False.
    8. Append a reflection summary to agent_steps.
    9. If JSON parsing fails, default to score=7.
    """
    current_reflection_count: int = int(state.get("reflection_count", 0) or 0)
    current_steps = list(state.get("agent_steps", []) or [])

    if current_reflection_count >= 2:
        current_steps.append("Reflection skipped: max retries reached")
        return {
            "needs_retry": False,
            "reflection_count": current_reflection_count,
            "agent_steps": current_steps,
        }

    question: str = str(state.get("question", "") or "")
    draft_answer: str = str(state.get("draft_answer", "") or state.get("answer", "") or "")
    retrieved_context_raw = state.get("retrieved_context", "") or state.get("context", "")

    if isinstance(retrieved_context_raw, list):
        retrieved_context = "\n\n".join(str(item) for item in retrieved_context_raw)
    else:
        retrieved_context = str(retrieved_context_raw)

    evaluation_input = (
        f"Original Question:\n{question}\n\n"
        f"Draft Answer:\n{draft_answer}\n\n"
        f"Retrieved Context:\n{retrieved_context}\n\n"
        'Return JSON only in this format: {"score": int, "reason": str, "needs_retry": bool}'
    )

    # llm imported from config.settings

    score: int = 7
    reason: str = "Reflection parsing failed; defaulted to acceptable score."
    parsed_needs_retry: bool = False

    try:
        messages = [
            HumanMessage(
                content=f"{REFLECTION_PROMPT}\n\n{evaluation_input}"
            )
        ]
        response = llm.invoke(messages)

        if hasattr(response, "content"):
            raw_content = response.content
        else:
            raw_content = str(response)

        if isinstance(raw_content, list):
            raw_text = "".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in raw_content
            )
        else:
            raw_text = str(raw_content).strip()

        parsed = json.loads(raw_text)
        score = int(parsed.get("score", 7))
        reason = str(parsed.get("reason", "No reason provided."))
        parsed_needs_retry = bool(parsed.get("needs_retry", score < 6))
    except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
        score = 7
        reason = "Reflection parsing failed; defaulted to acceptable score."
        parsed_needs_retry = False

    score = max(0, min(score, 10))

    updated_reflection_count = current_reflection_count
    if score < 6:
        needs_retry = True
        updated_reflection_count += 1
    else:
        needs_retry = False

    if parsed_needs_retry and score < 6:
        needs_retry = True

    current_steps.append(f"Reflection score: {score}/10 - {reason}")

    return {
        "reflection_score": score,
        "needs_retry": needs_retry,
        "reflection_count": updated_reflection_count,
        "agent_steps": current_steps,
    }