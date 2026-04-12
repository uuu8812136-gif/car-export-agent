"""
agent/nodes/human_intervention.py
Human-in-the-loop intervention node using LangGraph interrupt().

Flow:
  1. If needs_human_review OR human_intervention_requested: pause via interrupt()
  2. On resume: use intervention_edit (if provided) or original draft_answer
  3. Log intervention to intervention_log.json
  4. Optionally sync edited content to ChromaDB knowledge base
"""
from __future__ import annotations

import uuid
from typing import Any

from langgraph.types import interrupt

from agent.state import AgentState
from agent.utils.intervention_log import log_intervention


def check_human_intervention(state: AgentState) -> dict[str, Any]:
    """
    Gate node: decide whether to trigger human interrupt.
    Routes to interrupt only when confidence is low OR explicitly requested.
    """
    needs_review = bool(state.get("needs_human_review", False))
    requested = bool(state.get("human_intervention_requested", False))
    agent_steps = list(state.get("agent_steps", []))

    if not needs_review and not requested:
        return {"agent_steps": agent_steps}

    # Trigger LangGraph interrupt — execution pauses here
    # The UI will present the draft_answer for editing
    # On resume, state["intervention_edit"] contains the edited text (or "")
    trigger_reason = "low confidence" if needs_review else "manual request"
    agent_steps.append(f"⏸️ Human intervention triggered ({trigger_reason})")

    edited = interrupt({
        "type": "human_intervention",
        "reason": trigger_reason,
        "draft_answer": state.get("draft_answer", ""),
        "confidence_score": state.get("price_confidence_score", 0.0),
    })

    # After resume: edited is either the new text or None/""
    edited_text = edited if isinstance(edited, str) and edited.strip() else ""
    final_answer = edited_text if edited_text else state.get("draft_answer", "")

    # Log the intervention
    entry = log_intervention(
        session_id=state.get("session_id", "unknown"),
        user_role=state.get("user_role", "sales"),
        original_response=state.get("draft_answer", ""),
        edited_response=final_answer,
        reason=trigger_reason,
    )

    # Sync to ChromaDB if content was actually edited
    if edited_text and edited_text != state.get("draft_answer", ""):
        _sync_to_knowledge_base(final_answer, state)
        agent_steps.append("✅ Edited content synced to knowledge base")
    else:
        agent_steps.append("✅ Human review completed (no edits)")

    return {
        "draft_answer": final_answer,
        "human_intervention_requested": False,
        "needs_human_review": False,
        "agent_steps": agent_steps,
    }


def _sync_to_knowledge_base(content: str, state: AgentState) -> None:
    """
    Write edited content back to ChromaDB as a new knowledge document.
    This allows sales edits to improve future RAG retrieval.
    """
    try:
        from langchain_core.documents import Document
        from rag.vectorstore import get_vectorstore

        doc = Document(
            page_content=content,
            metadata={
                "source": "human_intervention",
                "session_id": state.get("session_id", ""),
                "user_role": state.get("user_role", "sales"),
                "intent": state.get("intent", ""),
            },
        )
        vs = get_vectorstore()
        vs.add_documents([doc])
    except Exception:
        # Non-critical: KB sync failure should not block the response
        pass
