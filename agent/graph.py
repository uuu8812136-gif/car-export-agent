from __future__ import annotations

from typing import Any

import uuid

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agent.nodes.contract_node import generate_contract
from agent.nodes.doc_grader import grade_documents
from agent.nodes.general_chat_node import general_chat
from agent.nodes.human_intervention import check_human_intervention
from agent.nodes.reflection_pipeline import run_reflection_pipeline
from agent.nodes.intent_detector import detect_intent
from agent.nodes.price_node import query_price
from agent.nodes.rag_node import retrieve_and_answer
from agent.state import AgentState

_checkpointer = MemorySaver()


def respond(state: AgentState) -> dict[str, Any]:
    messages = list(state.get("messages", []))
    draft_answer = state.get("draft_answer", "")

    if draft_answer:
        messages.append(AIMessage(content=draft_answer))

    return {
        "messages": messages,
    }


def intent_router(state: AgentState) -> str:
    intent = state.get("intent", "")

    if intent == "price_query":
        return "price_node"
    if intent == "product_info":
        return "rag_node"
    if intent == "contract_request":
        return "contract_node"
    return "general_chat_node"


def post_rag_router(state: AgentState) -> str:
    return "doc_grader"


def hallucination_retry_router(state: AgentState) -> str:
    if state.get("needs_retry", False):
        return intent_router(state)
    # Route to human intervention if confidence is low or explicitly requested
    if state.get("needs_human_review", False) or state.get("human_intervention_requested", False):
        return "human_intervention"
    return "respond"


graph = StateGraph(AgentState)

graph.add_node("intent_detector", detect_intent)
graph.add_node("price_node", query_price)
graph.add_node("rag_node", retrieve_and_answer)
graph.add_node("contract_node", generate_contract)
graph.add_node("general_chat_node", general_chat)
graph.add_node("doc_grader", grade_documents)
graph.add_node("reflection_pipeline", run_reflection_pipeline)
graph.add_node("human_intervention", check_human_intervention)
graph.add_node("respond", respond)

graph.add_edge(START, "intent_detector")

graph.add_conditional_edges(
    "intent_detector",
    intent_router,
    {
        "price_node": "price_node",
        "rag_node": "rag_node",
        "contract_node": "contract_node",
        "general_chat_node": "general_chat_node",
    },
)

graph.add_edge("price_node", "reflection_pipeline")
graph.add_edge("contract_node", "reflection_pipeline")
graph.add_edge("general_chat_node", "reflection_pipeline")

graph.add_conditional_edges(
    "rag_node",
    post_rag_router,
    {
        "doc_grader": "doc_grader",
    },
)

graph.add_edge("doc_grader", "reflection_pipeline")

graph.add_conditional_edges(
    "reflection_pipeline",
    hallucination_retry_router,
    {
        "price_node": "price_node",
        "rag_node": "rag_node",
        "contract_node": "contract_node",
        "general_chat_node": "general_chat_node",
        "human_intervention": "human_intervention",
        "respond": "respond",
    },
)

graph.add_edge("human_intervention", "respond")
graph.add_edge("respond", END)

app = graph.compile(checkpointer=_checkpointer)


def run_agent(
    user_message: str,
    chat_history: list,
    session_id: str = "",
    user_role: str = "sales",
    reflection_strictness: str = "normal",
    human_intervention_requested: bool = False,
) -> tuple[str, list[str], dict]:
    try:
        normalized_history: list[BaseMessage] = []

        for item in chat_history:
            if isinstance(item, BaseMessage):
                normalized_history.append(item)
            elif isinstance(item, dict):
                role = str(item.get("role", "")).lower()
                content = str(item.get("content", ""))
                if role == "assistant":
                    normalized_history.append(AIMessage(content=content))
                else:
                    normalized_history.append(HumanMessage(content=content))
            elif isinstance(item, str):
                normalized_history.append(HumanMessage(content=item))
            else:
                normalized_history.append(HumanMessage(content=str(item)))

        messages: list[BaseMessage] = [
            *normalized_history,
            HumanMessage(content=user_message),
        ]

        sid = session_id or str(uuid.uuid4())

        initial_state: AgentState = {
            "messages": messages,
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
            "price_confidence_score": 0.0,
            "needs_human_review": False,
            "reflection_log": [],
            "reflection_strictness": reflection_strictness,
            "human_intervention_requested": human_intervention_requested,
            "intervention_edit": "",
            "user_role": user_role,
            "session_id": sid,
        }

        config = {"configurable": {"thread_id": sid}}
        result: AgentState = app.invoke(initial_state, config=config)

        response_text = ""
        for message in reversed(result.get("messages", [])):
            if isinstance(message, AIMessage):
                content = message.content
                if isinstance(content, str):
                    response_text = content
                else:
                    response_text = str(content)
                break

        # Fallback: if graph was interrupted (HITL) before respond node ran,
        # use draft_answer directly so the caller still gets the pending response
        if not response_text:
            response_text = result.get("draft_answer", "")

        agent_steps_raw = result.get("agent_steps", [])
        agent_steps: list[str] = [str(step) for step in agent_steps_raw]

        contract_info: dict[str, Any] = {
            "contract_path": result.get("contract_path", ""),
            "contract_data": result.get("contract_data", {}),
            "hallucination_status": result.get("hallucination_status", ""),
            "price_confidence_score": result.get("price_confidence_score", 0.0),
            "reflection_log": result.get("reflection_log", []),
            "session_id": sid,
        }

        return response_text, agent_steps, contract_info

    except Exception as exc:
        error_message = f"Agent execution failed: {exc}"
        return error_message, ["run_agent_error"], {"contract_path": "", "contract_data": {}, "hallucination_status": "error"}