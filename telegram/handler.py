"""
telegram/handler.py
Telegram Bot handler — polls for messages and routes through the AI Agent.
Works without a public webhook URL, suitable for local demo.
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from config.settings import PROJECT_ROOT

_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
_API = f"https://api.telegram.org/bot{_TOKEN}"
_HISTORY_FILE: Path = PROJECT_ROOT / "data" / "telegram_history.json"
_OFFSET_FILE: Path = PROJECT_ROOT / "data" / "telegram_offset.txt"

# Per-chat conversation history: {chat_id: [{"role": ..., "content": ...}]}
_chat_histories: dict[str, list[dict]] = {}

# 线程锁：防止多个轮询线程同时运行
_polling_lock = threading.Lock()

# 已处理的 update_id 集合：去重安全网
_processed_update_ids: set[int] = set()


# ── Telegram API helpers ────────────────────────────────────────────

def _get_updates(offset: int = 0, timeout: int = 20) -> list[dict]:
    try:
        r = httpx.get(
            f"{_API}/getUpdates",
            params={"offset": offset, "timeout": timeout, "allowed_updates": ["message"]},
            timeout=timeout + 5,
        )
        data = r.json()
        return data.get("result", []) if data.get("ok") else []
    except Exception:
        return []


def _send_message(chat_id: int | str, text: str) -> None:
    """Send a message, auto-splitting if > 4096 chars (Telegram limit)."""
    chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
    for chunk in chunks:
        try:
            httpx.post(
                f"{_API}/sendMessage",
                json={"chat_id": chat_id, "text": chunk, "parse_mode": "Markdown"},
                timeout=15,
            )
        except Exception:
            pass


def _send_typing(chat_id: int | str) -> None:
    try:
        httpx.post(
            f"{_API}/sendChatAction",
            json={"chat_id": chat_id, "action": "typing"},
            timeout=5,
        )
    except Exception:
        pass


# ── History persistence ─────────────────────────────────────────────

def _load_history_file() -> list[dict]:
    if not _HISTORY_FILE.exists():
        return []
    try:
        return json.loads(_HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def _append_to_history_file(entry: dict) -> None:
    _HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    records = _load_history_file()
    records.append(entry)
    _HISTORY_FILE.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_telegram_history() -> list[dict]:
    return _load_history_file()


# ── Offset persistence ─────────────────────────────────────────────

def _load_offset() -> int:
    """从文件读取上次处理到的偏移量，文件不存在或损坏返回0。"""
    try:
        if _OFFSET_FILE.exists():
            text = _OFFSET_FILE.read_text(encoding="utf-8").strip()
            if text:
                return int(text)
    except (ValueError, OSError):
        pass
    return 0


def _save_offset(offset: int) -> None:
    """把当前偏移量写入文件，下次启动可以接着读。"""
    try:
        _OFFSET_FILE.parent.mkdir(parents=True, exist_ok=True)
        _OFFSET_FILE.write_text(str(offset), encoding="utf-8")
    except OSError:
        pass


# ── Message processing ──────────────────────────────────────────────

def process_message(update: dict) -> None:
    """Process one Telegram update: run agent and send reply."""
    msg = update.get("message", {})
    if not msg:
        return

    chat_id = str(msg["chat"]["id"])
    text = msg.get("text", "").strip()
    user = msg.get("from", {})
    username = user.get("username") or user.get("first_name", "unknown")

    if not text or text.startswith("/"):
        if text == "/start":
            _send_message(
                chat_id,
                "👋 *Welcome to Auto Export AI Agent!*\n\n"
                "I can help you with:\n"
                "• 💰 Car prices (FOB/CIF)\n"
                "• 🚗 Vehicle specifications\n"
                "• 📋 Export quotation contracts\n\n"
                "Try: _BYD Seal CIF price to Lagos?_",
            )
        return

    _send_typing(chat_id)

    # Retrieve per-chat history
    history = _chat_histories.get(chat_id, [])

    # Run through the Agent
    try:
        from agent.graph import run_agent
        result = run_agent(
            text,
            history,
            session_id=f"tg-{chat_id}",
            user_role="sales",
        )
        if isinstance(result, tuple):
            answer = str(result[0]) if result[0] else "Sorry, I couldn't process that."
        elif isinstance(result, dict):
            answer = result.get("response") or result.get("draft_answer") or "No response."
        else:
            answer = str(result)
    except Exception as e:
        answer = f"⚠️ Agent error: {str(e)[:200]}"

    # Update in-memory history
    history.append({"role": "user", "content": text})
    history.append({"role": "assistant", "content": answer})
    _chat_histories[chat_id] = history[-20:]  # keep last 20 turns

    # Send reply
    _send_message(chat_id, answer)

    # Persist to file for Streamlit monitoring
    _append_to_history_file({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "chat_id": chat_id,
        "username": username,
        "user_message": text,
        "agent_reply": answer,
    })


# ── Polling loop ────────────────────────────────────────────────────

def run_polling() -> None:
    """Long-polling loop. Run this as a background thread or separate process."""
    # 第二层保险：线程锁，防止多个轮询同时跑
    if not _polling_lock.acquire(blocking=False):
        print("[Telegram] 轮询已在运行，跳过重复启动。")
        return

    try:
        print(f"[Telegram] Bot started — @{os.getenv('TELEGRAM_BOT_USERNAME', 'bot')}")
        print(f"[Telegram] Send messages at: t.me/{os.getenv('TELEGRAM_BOT_USERNAME', 'bot')}")

        # 第一层保险：从文件恢复上次的偏移量
        offset = _load_offset()

        while True:
            updates = _get_updates(offset=offset)
            for update in updates:
                uid = update["update_id"]
                offset = uid + 1
                _save_offset(offset)

                # 第三层保险：跳过已处理的消息ID
                if uid in _processed_update_ids:
                    continue
                _processed_update_ids.add(uid)
                if len(_processed_update_ids) > 10000:
                    _processed_update_ids.clear()

                try:
                    process_message(update)
                except Exception as e:
                    print(f"[Telegram] Error processing update: {e}")
            time.sleep(0.5)
    finally:
        _polling_lock.release()
