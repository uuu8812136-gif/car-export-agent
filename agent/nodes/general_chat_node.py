from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from config.prompts import GENERAL_CHAT_PROMPT
from agent.state import AgentState
from config.settings import llm


def _normalize_conversation_history(history: Any) -> List[BaseMessage]:
    """
    Normalize conversation history into LangChain message objects.

    Supports:
    - List[BaseMessage]
    - List[dict] with keys like {"role": "...", "content": "..."}
    - Plain strings
    """
    if not history:
        return []

    normalized: List[BaseMessage] = []

    if isinstance(history, list):
        for item in history:
            if isinstance(item, BaseMessage):
                normalized.append(item)
            elif isinstance(item, dict):
                role = str(item.get("role", "")).lower()
                content = str(item.get("content", "")).strip()
                if not content:
                    continue

                if role in {"system"}:
                    normalized.append(SystemMessage(content=content))
                elif role in {"assistant", "ai"}:
                    normalized.append(AIMessage(content=content))
                else:
                    normalized.append(HumanMessage(content=content))
            elif isinstance(item, str):
                content = item.strip()
                if content:
                    normalized.append(HumanMessage(content=content))

    return normalized


def general_chat(state: AgentState) -> Dict[str, Any]:
    """
    Generate a friendly, professional general chat response in a car export sales persona.

    Logic:
    - Call llm with GENERAL_CHAT_PROMPT + conversation history
    - Set draft_answer = response
    - Update agent_steps with "General chat response generated"
    """
    # llm imported from config.settings

    conversation_history = _normalize_conversation_history(
        state.get("conversation_history") or state.get("messages") or []
    )

    system_prompt = (
        f"{GENERAL_CHAT_PROMPT}\n\n"
        "You are a friendly, professional Chinese car export sales consultant. "
        "Maintain a helpful sales persona focused on Chinese vehicle exports, including brands "
        "such as BYD, Chery, MG, Geely, and SAIC. When relevant, discuss pricing in USD and "
        "use appropriate Incoterms such as FOB and CIF."
    )

    messages: List[BaseMessage] = [SystemMessage(content=system_prompt), *conversation_history]

    response = llm.invoke(messages)
    draft_answer = response.content if hasattr(response, "content") else str(response)
    draft_answer = (
        draft_answer
        + "\n\n---\n"
        "💬 *General guidance — no price database lookup performed. "
        "For specific pricing, please ask about a vehicle model and destination.*"
    )

    current_steps = list(state.get("agent_steps", []))
    current_steps.append("General chat response generated")

    return {
        "draft_answer": draft_answer,
        "agent_steps": current_steps,
    }