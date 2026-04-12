"""
generate_whatsapp.py - Use GPT to generate WhatsApp integration code
Run: python generate_whatsapp.py
"""
import time
from pathlib import Path
from openai import OpenAI

OPENAI_API_KEY = "sk-b9ede3798a406b316e96984687f3040d2ac80a723b3cd3681a4d9d776c283336"
BASE_URL = "https://hk.ticketpro.cc/v1"
MODEL = "gpt-5.4"
MODEL_FALLBACK = "gpt-4.1"
PROJECT_ROOT = Path("H:/car-export-agent")

client = OpenAI(api_key=OPENAI_API_KEY, base_url=BASE_URL)

SYS = (
    "You are an expert Python developer specializing in FastAPI and Streamlit. "
    "Generate complete, production-ready Python code with NO placeholders, NO TODOs, "
    "NO incomplete sections. Use type hints. All functions fully implemented."
)

CTX = (
    "EXISTING PROJECT CONTEXT:\n"
    "- agent/graph.py: run_agent(user_message:str, chat_history:list) -> tuple[str, list, dict]\n"
    "  Returns (response_text, agent_steps_list, contract_info_dict)\n"
    "- whatsapp/handler.py exports:\n"
    "    handle_incoming(payload: dict) -> dict  (processes Green API webhook, calls run_agent)\n"
    "    load_history() -> list[dict]  (reads data/whatsapp_history.json)\n"
    "  Each history item has keys: timestamp, chat_id, sender_name, direction (inbound/outbound), text\n"
    "- config/settings.py exports: GREEN_API_INSTANCE_ID, GREEN_API_TOKEN, GREEN_API_BASE_URL (all str)\n"
)

SERVER_PROMPT = (
    CTX
    + "\nGenerate server.py - a FastAPI application serving as the WhatsApp gateway.\n\n"
    + "REQUIRED at the very top of the file:\n"
    + "  import sys\n"
    + "  from pathlib import Path\n"
    + "  sys.path.insert(0, str(Path(__file__).resolve().parent))\n\n"
    + "Then these imports:\n"
    + "  import logging\n"
    + "  from typing import Any\n"
    + "  import uvicorn\n"
    + "  from fastapi import FastAPI, HTTPException, Request\n"
    + "  from fastapi.responses import JSONResponse\n"
    + "  from whatsapp.handler import handle_incoming, load_history\n"
    + "  from config.settings import GREEN_API_INSTANCE_ID, GREEN_API_TOKEN\n\n"
    + "  logging.basicConfig(level=logging.INFO)\n"
    + "  logger = logging.getLogger(__name__)\n\n"
    + "  app = FastAPI(title='Car Export Agent - WhatsApp Gateway', version='1.0.0')\n\n"
    + "Implement these 4 endpoints:\n"
    + "1. GET /health -> returns dict: status=ok, green_api_configured=bool\n"
    + "2. POST /webhook -> parse request JSON body, call handle_incoming(payload), return JSONResponse(result)\n"
    + "   Log the typeWebhook field. Handle JSON parse errors with HTTP 400.\n"
    + "3. POST /simulate -> body JSON with fields: phone (str), name (str), message (str)\n"
    + "   Build a fake Green API webhook payload and call handle_incoming.\n"
    + "   The fake payload structure:\n"
    + "     typeWebhook: incomingMessageReceived\n"
    + "     senderData: chatId=phone+@c.us, sender=phone+@c.us, senderName=name\n"
    + "     messageData: typeMessage=textMessage, textMessageData.textMessage=message\n"
    + "   Return JSONResponse(result). Raise HTTPException(status_code=422) if message is empty.\n"
    + "4. GET /messages -> return JSONResponse with key messages containing load_history() result\n\n"
    + "At the bottom:\n"
    + "  if __name__ == '__main__':\n"
    + "      uvicorn.run('server:app', host='0.0.0.0', port=8000, reload=False)\n"
)

APP_PROMPT = (
    CTX
    + "\nGenerate app.py - Streamlit application with TWO TABS: Chat and WhatsApp Inquiries.\n\n"
    + "Required imports:\n"
    + "  import streamlit as st\n"
    + "  from pathlib import Path\n"
    + "  from typing import Any\n"
    + "  from agent.graph import run_agent\n"
    + "  from whatsapp.handler import handle_incoming, load_history\n\n"
    + "Page config: page_title=Car Export AI Agent, page_icon=car emoji, layout=wide\n\n"
    + "Session state keys: messages (list of role/content dicts), agent_steps (list of str), last_contract_path (str or None)\n\n"
    + "Implement these functions:\n\n"
    + "1. init_session_state() - initialize all session state keys to defaults\n\n"
    + "2. normalize_agent_result(result: Any) -> tuple[str, list[str], str | None]\n"
    + "   Handle all return types from run_agent: str, tuple, or dict\n"
    + "   Return (response_text, steps_list, contract_path_or_None)\n\n"
    + "3. get_step_icon(step: str) -> str\n"
    + "   Return emoji based on keyword in step string:\n"
    + "   'intent' in step -> magnifying glass emoji\n"
    + "   'price' or 'pricing' -> money bag emoji\n"
    + "   'rag' or 'retrieval' or 'knowledge' -> books emoji\n"
    + "   'contract' or 'quotation' -> memo emoji\n"
    + "   'reflect' -> mirror emoji\n"
    + "   else -> gear emoji\n\n"
    + "4. render_sidebar()\n"
    + "   Inside st.sidebar:\n"
    + "   - Header: Agent Reasoning Steps\n"
    + "   - For each step in session_state.agent_steps: st.expander with icon from get_step_icon and step text\n"
    + "   - If last_contract_path is set and file exists: st.download_button to download the .md file\n"
    + "   - Caption with tech stack info at bottom\n\n"
    + "5. render_chat_tab()\n"
    + "   - Show all messages from st.session_state.messages using st.chat_message\n"
    + "   - st.chat_input for user input with placeholder text\n"
    + "   - On input: append user msg, spinner 'Thinking...', call run_agent with message and history\n"
    + "   - normalize result, update session state, call st.rerun()\n"
    + "   - Use try/except to handle errors gracefully\n\n"
    + "6. render_whatsapp_tab()\n"
    + "   col_left, col_right = st.columns([3, 2])\n\n"
    + "   col_left - Conversation History:\n"
    + "   - Subheader and refresh button\n"
    + "   - messages = load_history()\n"
    + "   - If empty: st.info with setup instructions\n"
    + "   - Else: group messages by chat_id into a dict\n"
    + "   - For each group: st.expander with phone number as title (strip @c.us)\n"
    + "   - Inside each expander: for each message render a styled HTML chat bubble\n"
    + "     Use st.markdown with unsafe_allow_html=True\n"
    + "     Inbound (buyer): white background, display flex, justify-content flex-start, border-radius: 0px 18px 18px 18px\n"
    + "     Outbound (agent): #DCF8C6 green background, display flex, justify-content flex-end, border-radius: 18px 0px 18px 18px\n"
    + "     Show: sender_name, message text, time HH:MM extracted from timestamp string\n\n"
    + "   col_right - Simulate:\n"
    + "   - Subheader: Simulate WhatsApp Message\n"
    + "   - st.form with clear_on_submit=True containing:\n"
    + "       phone text_input (default: 8613800138000, help text about format)\n"
    + "       name text_input (default: Demo Buyer)\n"
    + "       message text_area with placeholder\n"
    + "       submit button\n"
    + "   - On submit: build Green API fake payload dict, call handle_incoming(payload)\n"
    + "     Show st.success, response with st.markdown, contract info if present\n"
    + "     Then call st.rerun()\n"
    + "   - Below form: st.markdown with numbered Green API setup instructions and link to green-api.com\n\n"
    + "7. main()\n"
    + "   - init_session_state()\n"
    + "   - render_sidebar()\n"
    + "   - st.title with car emoji and title text\n"
    + "   - st.caption with Chinese subtitle and Powered by LangGraph + RAG + GPT\n"
    + "   - tab_chat, tab_wa = st.tabs with appropriate emoji labels\n"
    + "   - with tab_chat: render_chat_tab()\n"
    + "   - with tab_wa: render_whatsapp_tab()\n\n"
    + "if __name__ == '__main__': main()\n"
)

FILES = [
    {"path": "server.py", "prompt": SERVER_PROMPT},
    {"path": "app.py", "prompt": APP_PROMPT},
]


def call_gpt(prompt: str, model: str = MODEL) -> str:
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYS},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=4096,
        )
        return r.choices[0].message.content
    except Exception as e:
        if model != MODEL_FALLBACK:
            print(f"  Retry with {MODEL_FALLBACK}: {e}")
            return call_gpt(prompt, MODEL_FALLBACK)
        raise


def strip_fences(raw: str) -> str:
    s = raw.strip()
    if not s.startswith("```"):
        return raw
    lines = s.splitlines()
    end = -1 if lines[-1].strip() == "```" else len(lines)
    return "\n".join(lines[1:end])


def generate_file(info: dict) -> bool:
    path = PROJECT_ROOT / info["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[GEN] {info['path']} ...")
    try:
        raw = call_gpt(info["prompt"])
        content = strip_fences(raw)
        path.write_text(content, encoding="utf-8")
        print(f"  [OK] saved {info['path']} ({len(content)} chars)")
        return True
    except Exception as e:
        print(f"  [ERR] {e}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("WhatsApp Feature Generator — GPT")
    print("=" * 50)
    ok = sum(generate_file(f) for f in FILES)
    print(f"\nDone: {ok}/{len(FILES)}")
