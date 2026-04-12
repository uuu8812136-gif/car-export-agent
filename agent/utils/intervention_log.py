"""
agent/utils/intervention_log.py
Logs all human interventions to JSON file for audit trail.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from config.settings import PROJECT_ROOT

_LOG_FILE: Path = PROJECT_ROOT / "data" / "intervention_log.json"


def log_intervention(
    session_id: str,
    user_role: str,
    original_response: str,
    edited_response: str,
    chat_id: str = "",
    reason: str = "",
) -> dict[str, Any]:
    _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    entry: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "session_id": session_id,
        "user_role": user_role,
        "chat_id": chat_id,
        "original_response": original_response,
        "edited_response": edited_response,
        "reason": reason,
        "synced_to_kb": False,
    }

    existing: list[dict[str, Any]] = []
    if _LOG_FILE.exists():
        try:
            existing = json.loads(_LOG_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = []

    existing.append(entry)
    _LOG_FILE.write_text(
        json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return entry


def load_interventions(user_role: str = "admin", session_id: str = "") -> list[dict[str, Any]]:
    """Admin sees all; sales sees only their own session."""
    if not _LOG_FILE.exists():
        return []
    try:
        all_logs: list[dict[str, Any]] = json.loads(_LOG_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

    if user_role == "admin":
        return all_logs
    return [e for e in all_logs if e.get("session_id") == session_id]


def update_intervention_status(
    timestamp: str, session_id: str, approved: bool, sync_to_kb: bool = False
) -> bool:
    """Update approval status of an intervention log entry.

    Args:
        timestamp: The entry's timestamp (ISO string, first 19 chars)
        session_id: The entry's session_id
        approved: True to approve, False to reject
        sync_to_kb: If True, also sync the edited_response to ChromaDB

    Returns:
        True if entry was found and updated, False otherwise.
    """
    if not _LOG_FILE.exists():
        return False
    try:
        all_logs: list[dict[str, Any]] = json.loads(_LOG_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False

    updated = False
    for entry in all_logs:
        if (
            entry.get("timestamp", "")[:19] == timestamp[:19]
            and entry.get("session_id") == session_id
        ):
            entry["approved"] = approved
            entry["reviewed_at"] = datetime.now().isoformat(timespec="seconds")
            if sync_to_kb and entry.get("edited_response"):
                _do_sync_to_kb(entry)
                entry["synced_to_kb"] = True
            updated = True
            break

    if updated:
        _LOG_FILE.write_text(
            json.dumps(all_logs, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return updated


def _do_sync_to_kb(entry: dict[str, Any]) -> None:
    """Sync an intervention entry's edited_response to ChromaDB."""
    try:
        from langchain_core.documents import Document
        from rag.vectorstore import get_vectorstore

        doc = Document(
            page_content=entry.get("edited_response", ""),
            metadata={
                "source": "admin_approved_intervention",
                "session_id": entry.get("session_id", ""),
                "user_role": entry.get("user_role", ""),
                "timestamp": entry.get("timestamp", ""),
            },
        )
        vs = get_vectorstore()
        vs.add_documents([doc])
    except Exception:
        pass  # Non-critical
