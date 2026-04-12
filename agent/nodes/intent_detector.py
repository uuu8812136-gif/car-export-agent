from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from agent.state import AgentState
from config.prompts import INTENT_DETECTION_PROMPT
from config.settings import llm


_VALID_INTENTS = {
    "general_chat",
    "price_query",
    "product_info",
    "contract_request",
}


def _extract_last_user_message(messages: List[Any]) -> str:
    """
    Extract the latest user message content from a LangGraph/LangChain message list.
    Falls back to the last message content if no HumanMessage is found.
    """
    if not messages:
        return ""

    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            content = getattr(message, "content", "")
            return content if isinstance(content, str) else str(content)

    last_message = messages[-1]
    content = getattr(last_message, "content", "")
    return content if isinstance(content, str) else str(content)


def _normalize_intent(raw_intent: str) -> str:
    """
    Normalize LLM output into one of the supported intent labels.
    Defaults to general_chat for any unexpected output.
    """
    normalized = raw_intent.strip().lower()
    return normalized if normalized in _VALID_INTENTS else "general_chat"


def detect_intent(state: AgentState) -> Dict[str, Any]:
    """
    Detect user intent from the latest user message using the configured LLM.

    Returns:
        dict: {
            "intent": str,
            "agent_steps": list[str]
        }
    """
    messages = state.get("messages", [])
    existing_steps = list(state.get("agent_steps", []))

    last_user_message = _extract_last_user_message(messages)

    prompt_messages = [
        SystemMessage(content=INTENT_DETECTION_PROMPT),
        HumanMessage(content=last_user_message),
    ]

    response = llm.invoke(prompt_messages)

    if hasattr(response, "content"):
        raw_intent = response.content
    else:
        raw_intent = str(response)

    if not isinstance(raw_intent, str):
        raw_intent = str(raw_intent)

    intent = _normalize_intent(raw_intent)

    updated_steps = existing_steps + [f"Intent detected: {intent}"]

    return {
        "intent": intent,
        "agent_steps": updated_steps,
    }