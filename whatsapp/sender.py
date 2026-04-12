from __future__ import annotations

import logging

import httpx

from config.settings import GREEN_API_BASE_URL, GREEN_API_INSTANCE_ID, GREEN_API_TOKEN

logger = logging.getLogger(__name__)


def send_message(chat_id: str, text: str) -> bool:
    """
    Send a WhatsApp text message via Green API.

    Args:
        chat_id: WhatsApp chat ID, e.g. "8613800138000@c.us"
        text:    Message body to send.

    Returns:
        True on success, False on any error.
    """
    if not GREEN_API_INSTANCE_ID or not GREEN_API_TOKEN:
        logger.warning(
            "GREEN_API_INSTANCE_ID / GREEN_API_TOKEN not configured. "
            "Message logged but NOT sent."
        )
        return False

    url = (
        f"{GREEN_API_BASE_URL}/waInstance{GREEN_API_INSTANCE_ID}"
        f"/sendMessage/{GREEN_API_TOKEN}"
    )

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, json={"chatId": chat_id, "message": text})
            resp.raise_for_status()
            logger.info("Sent to %s → idMessage=%s", chat_id, resp.json().get("idMessage"))
            return True
    except httpx.HTTPStatusError as exc:
        logger.error("HTTP %s sending to %s: %s", exc.response.status_code, chat_id, exc)
    except httpx.RequestError as exc:
        logger.error("Network error sending to %s: %s", chat_id, exc)
    except Exception as exc:  # pragma: no cover
        logger.error("Unexpected error sending to %s: %s", chat_id, exc)
    return False
