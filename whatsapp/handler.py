from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from config.settings import PROJECT_ROOT

logger = logging.getLogger(__name__)

WHATSAPP_HISTORY_FILE: Path = PROJECT_ROOT / "data" / "whatsapp_history.json"

# In-memory conversation history per sender (cleared on restart)
_chat_histories: dict[str, list[dict[str, str]]] = {}


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def load_history() -> list[dict[str, Any]]:
    """Return all stored WhatsApp messages (inbound + outbound)."""
    if WHATSAPP_HISTORY_FILE.exists():
        try:
            return json.loads(WHATSAPP_HISTORY_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
    return []


def _append_message(
    chat_id: str,
    sender_name: str,
    direction: str,
    text: str,
) -> None:
    """Persist a single message to disk."""
    history = load_history()
    history.append(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "chat_id": chat_id,
            "sender_name": sender_name,
            "direction": direction,  # "inbound" | "outbound"
            "text": text,
        }
    )
    WHATSAPP_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    WHATSAPP_HISTORY_FILE.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Payload parsing
# ---------------------------------------------------------------------------

def _extract_text_message(
    payload: dict[str, Any],
) -> tuple[str, str, str] | None:
    """
    Parse a Green API webhook payload.

    Returns:
        (chat_id, sender_name, message_text) or None if not a text message.
    """
    if payload.get("typeWebhook") != "incomingMessageReceived":
        return None

    msg_data = payload.get("messageData", {})
    if msg_data.get("typeMessage") != "textMessage":
        return None

    text = msg_data.get("textMessageData", {}).get("textMessage", "").strip()
    if not text:
        return None

    sender_data = payload.get("senderData", {})
    chat_id = sender_data.get("chatId", "unknown@c.us")
    sender_name = sender_data.get("senderName") or chat_id.split("@")[0]

    return chat_id, sender_name, text


# ---------------------------------------------------------------------------
# Main handler
# ---------------------------------------------------------------------------

def handle_incoming(payload: dict[str, Any]) -> dict[str, Any]:
    """
    End-to-end handler for an incoming WhatsApp webhook event.

    1. Parse the Green API payload.
    2. Fetch per-sender conversation history.
    3. Run the LangGraph agent.
    4. Persist both messages.
    5. Send the reply via Green API.
    6. Return a processing summary.
    """
    # Import here to avoid circular imports at module load time
    from agent.graph import run_agent  # noqa: PLC0415
    from whatsapp.sender import send_message  # noqa: PLC0415

    extracted = _extract_text_message(payload)
    if extracted is None:
        logger.debug("Skipped non-text webhook: %s", payload.get("typeWebhook"))
        return {"status": "skipped", "reason": "not a supported text message"}

    chat_id, sender_name, user_text = extracted
    logger.info("Inbound [%s / %s]: %s", sender_name, chat_id, user_text[:120])

    # Persist inbound
    _append_message(chat_id, sender_name, "inbound", user_text)

    # Build LangChain-compatible message history (list of dicts)
    chat_history = _chat_histories.get(chat_id, [])

    # Run agent
    try:
        response_text, agent_steps, contract_info = run_agent(user_text, chat_history)
    except Exception as exc:
        logger.error("Agent error for %s: %s", chat_id, exc, exc_info=True)
        response_text = (
            "Sorry, I encountered an internal error. "
            "Please try again or contact our sales team directly."
        )
        agent_steps = []
        contract_info = {}

    # Update in-memory history (keep last 20 turns to avoid context overflow)
    history = _chat_histories.setdefault(chat_id, [])
    history += [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": response_text},
    ]
    _chat_histories[chat_id] = history[-20:]

    # Persist outbound
    _append_message(chat_id, "Agent", "outbound", response_text)

    # Send reply (best-effort — errors are logged, not raised)
    reply_sent = send_message(chat_id, response_text)

    return {
        "status": "processed",
        "chat_id": chat_id,
        "sender_name": sender_name,
        "user_text": user_text,
        "response": response_text,
        "reply_sent": reply_sent,
        "contract_info": contract_info,
    }
