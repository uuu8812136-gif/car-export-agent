from __future__ import annotations

from typing import Annotated, Any

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: str
    retrieved_context: str
    price_result: dict[str, Any]
    draft_answer: str
    reflection_score: int
    reflection_count: int
    needs_retry: bool
    contract_data: dict[str, Any]
    contract_path: str
    agent_steps: list[str]
    hallucination_status: str
    # V2 fields
    price_confidence_score: float
    needs_human_review: bool
    reflection_log: list[dict[str, Any]]
    reflection_strictness: str          # "strict" | "normal" | "lenient"
    human_intervention_requested: bool
    intervention_edit: str
    user_role: str                       # "sales" | "admin"
    session_id: str                      # for checkpointer routing


def get_default_state() -> AgentState:
    return {
        "messages": [],
        "intent": "",
        "retrieved_context": "",
        "price_result": {},
        "draft_answer": "",
        "reflection_score": 0,
        "reflection_count": 0,
        "needs_retry": False,
        "contract_data": {},
        "contract_path": "",
        "agent_steps": [],
        "hallucination_status": "",
        # V2 fields
        "price_confidence_score": 0.0,
        "needs_human_review": False,
        "reflection_log": [],
        "reflection_strictness": "normal",
        "human_intervention_requested": False,
        "intervention_edit": "",
        "user_role": "sales",
        "session_id": "",
    }