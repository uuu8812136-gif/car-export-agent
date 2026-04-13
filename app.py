import html
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st


@st.cache_resource(show_spinner=False)
def _load_agent():
    """Import and compile the LangGraph agent once, cache across reruns."""
    from agent.graph import run_agent
    return run_agent


@st.cache_resource(show_spinner=False)
def _load_whatsapp():
    from whatsapp.handler import handle_incoming, load_history
    return handle_incoming, load_history

st.set_page_config(page_title="汽车出口AI助手", page_icon="🚗", layout="wide", initial_sidebar_state="expanded")


# ---------------------------------------------------------------------------
# CSS Injection  (uses design tokens from ui/theme.py)
# ---------------------------------------------------------------------------
def inject_css() -> None:
    from ui.theme import css_variables, Colors, Radius, Shadow, FONT_STACK

    st.markdown(
        f"""
        <style>
        {css_variables()}

        /* ── Global ────────────────────────────────────── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        body, .stApp {{
            font-family: var(--font-family);
            background: linear-gradient(135deg, {Colors.bg} 0%, #EEF2FF 50%, {Colors.bg} 100%);
        }}

        #MainMenu, footer {{ visibility: hidden; }}
        header[data-testid="stHeader"] {{
            height: 0 !important; min-height: 0 !important;
            padding: 0 !important; visibility: hidden !important;
        }}
        .stApp > div:first-child {{ margin-top: 0 !important; }}
        .block-container {{ padding-top: 1rem !important; padding-bottom: 1rem !important; }}

        /* ── Sidebar ───────────────────────────────────── */
        [data-testid="stSidebar"] {{
            background: rgba(255,255,255,0.92);
            backdrop-filter: blur(24px);
            -webkit-backdrop-filter: blur(24px);
            border-right: 1px solid var(--color-border-light);
        }}

        /* ── Cards ─────────────────────────────────────── */
        .pro-card {{
            background: var(--color-card);
            border-radius: var(--radius-lg);
            padding: var(--space-xl);
            margin: var(--space-sm) 0;
            box-shadow: var(--shadow-sm);
            border: 1px solid var(--color-border-light);
            transition: box-shadow 0.2s ease, transform 0.2s ease;
        }}
        .pro-card:hover {{
            box-shadow: var(--shadow-md);
            transform: translateY(-1px);
        }}

        /* ── Badges ────────────────────────────────────── */
        .badge-verified {{
            display: inline-flex; align-items: center; gap: 6px;
            background: #ECFDF5; color: {Colors.success};
            border-radius: 20px; padding: 5px 14px;
            font-size: 13px; font-weight: 600;
        }}
        .badge-flagged {{
            display: inline-flex; align-items: center; gap: 6px;
            background: #FEF2F2; color: {Colors.danger};
            border-radius: 20px; padding: 5px 14px;
            font-size: 13px; font-weight: 600;
        }}
        .badge-reviewed {{
            display: inline-flex; align-items: center; gap: 6px;
            background: #FFFBEB; color: {Colors.warning};
            border-radius: 20px; padding: 5px 14px;
            font-size: 13px; font-weight: 600;
        }}
        .badge-idle {{
            display: inline-flex; align-items: center; gap: 6px;
            background: {Colors.border_light}; color: {Colors.text_secondary};
            border-radius: 20px; padding: 5px 14px;
            font-size: 13px; font-weight: 600;
        }}

        /* ── WhatsApp Bubbles ──────────────────────────── */
        .wa-inbound {{
            background: var(--color-card);
            border-radius: 4px 16px 16px 16px;
            padding: 10px 14px; max-width: 80%;
            box-shadow: var(--shadow-sm);
            font-size: 14px;
            border: 1px solid var(--color-border-light);
            word-wrap: break-word; overflow-wrap: break-word;
        }}
        .wa-outbound {{
            background: #DCF8C6;
            border-radius: 16px 4px 16px 16px;
            padding: 10px 14px; max-width: 80%;
            margin-left: auto;
            box-shadow: var(--shadow-sm);
            font-size: 14px;
            word-wrap: break-word; overflow-wrap: break-word;
        }}

        /* ── Buttons ───────────────────────────────────── */
        .stButton>button {{
            border-radius: var(--radius-sm);
            font-weight: 500;
            font-size: 14px;
            transition: all 0.2s ease;
            border: 1px solid var(--color-border);
        }}
        .stButton>button:hover {{
            transform: translateY(-1px);
            box-shadow: var(--shadow-sm);
        }}

        /* ── Tabs ──────────────────────────────────────── */
        .stTabs [data-baseweb="tab"] {{
            border-radius: var(--radius-sm);
            padding: 8px 16px;
            font-weight: 500;
        }}
        .stTabs [aria-selected="true"] {{
            background: var(--color-card);
            box-shadow: var(--shadow-sm);
        }}

        /* ── Inputs ────────────────────────────────────── */
        .stTextInput>div>div>input,
        .stTextArea>div>div>textarea {{
            border-radius: var(--radius-sm);
            border: 1.5px solid var(--color-border);
            font-size: 15px;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }}
        .stTextInput>div>div>input:focus,
        .stTextArea>div>div>textarea:focus {{
            border-color: var(--color-accent) !important;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.12) !important;
        }}

        /* ── Thinking Dots Animation ───────────────────── */
        @keyframes thinking-bounce {{
            0%, 80%, 100% {{ transform: translateY(0); opacity: 0.4; }}
            40% {{ transform: translateY(-6px); opacity: 1; }}
        }}
        .thinking-dots {{
            display: inline-flex; align-items: center; gap: 4px;
            padding: 8px 0;
        }}
        .thinking-dots span {{
            width: 8px; height: 8px; border-radius: 50%;
            background: var(--color-accent);
            animation: thinking-bounce 1.4s infinite ease-in-out both;
        }}
        .thinking-dots span:nth-child(2) {{ animation-delay: 0.16s; }}
        .thinking-dots span:nth-child(3) {{ animation-delay: 0.32s; }}

        /* ── Login Glassmorphism Card ──────────────────── */
        .glass-card {{
            background: rgba(255, 255, 255, 0.75);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-radius: var(--radius-xl);
            border: 1px solid rgba(255, 255, 255, 0.5);
            box-shadow: 0 8px 32px rgba(15, 23, 42, 0.10);
            padding: var(--space-xxl);
        }}

        /* ── Login Background ──────────────────────────── */
        .login-bg {{
            background: linear-gradient(135deg, {Colors.primary} 0%, #1E3A5F 40%, #3B82F6 100%);
            border-radius: var(--radius-xl);
            padding: var(--space-xxxl) var(--space-xxl);
            margin: -1rem -0.5rem 0 -0.5rem;
            min-height: auto;
            display: flex; flex-direction: column;
            align-items: center; justify-content: center;
        }}

        /* ── Typing Animation ──────────────────────────── */
        @keyframes typing {{
            from {{ width: 0; }}
            to {{ width: 100%; }}
        }}
        @keyframes blink-caret {{
            from, to {{ border-color: transparent; }}
            50% {{ border-color: rgba(255,255,255,0.7); }}
        }}
        .typing-text {{
            overflow: hidden;
            white-space: nowrap;
            border-right: 2px solid rgba(255,255,255,0.7);
            animation: typing 2.5s steps(30, end), blink-caret 0.75s step-end infinite;
            display: inline-block;
            max-width: 100%;
        }}

        /* ── Tech Pill Badges ──────────────────────────── */
        .tech-pill {{
            display: inline-block;
            background: rgba(255,255,255,0.12);
            color: rgba(255,255,255,0.8);
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 20px;
            padding: 4px 12px;
            font-size: 11px;
            font-weight: 500;
            letter-spacing: 0.5px;
        }}

        /* ── Intervention Panel ────────────────────────── */
        .intervention-panel {{
            background: #FFFBEB;
            border-radius: var(--radius-md);
            padding: var(--space-lg) var(--space-xl);
            border: 1px solid {Colors.warning};
            margin-bottom: var(--space-lg);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent_steps" not in st.session_state:
        st.session_state.agent_steps = []
    if "last_contract_path" not in st.session_state:
        st.session_state.last_contract_path = None
    if "hallucination_status" not in st.session_state:
        st.session_state.hallucination_status = ""
    # V2 fields
    if "user_role" not in st.session_state:
        st.session_state.user_role = "sales"
    if "user_name" not in st.session_state:
        st.session_state.user_name = ""
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "session_id" not in st.session_state:
        import uuid
        st.session_state.session_id = str(uuid.uuid4())
    if "reflection_strictness" not in st.session_state:
        st.session_state.reflection_strictness = "normal"
    if "reflection_log" not in st.session_state:
        st.session_state.reflection_log = []
    if "price_confidence" not in st.session_state:
        st.session_state.price_confidence = 0.0
    if "intervention_mode" not in st.session_state:
        st.session_state.intervention_mode = False


def normalize_agent_result(result: Any) -> tuple[str, list[str], str | None]:
    text: str = ""
    steps: list[str] = []
    contract_path: str | None = None

    if isinstance(result, str):
        text = result
    elif isinstance(result, tuple):
        if len(result) >= 1:
            text = str(result[0]) if result[0] is not None else ""
        if len(result) >= 2 and isinstance(result[1], list):
            steps = [str(step) for step in result[1]]
        elif len(result) >= 2 and result[1] is not None:
            steps = [str(result[1])]
        if len(result) >= 3 and isinstance(result[2], dict):
            contract_info = result[2]
            candidate_keys = (
                "contract_path",
                "path",
                "file_path",
                "contract_file",
                "quotation_path",
            )
            for key in candidate_keys:
                value = contract_info.get(key)
                if value:
                    contract_path = str(value)
                    break
    elif isinstance(result, dict):
        text = str(
            result.get("response")
            or result.get("response_text")
            or result.get("text")
            or result.get("message")
            or ""
        )
        raw_steps = result.get("steps") or result.get("agent_steps") or []
        if isinstance(raw_steps, list):
            steps = [str(step) for step in raw_steps]
        elif raw_steps:
            steps = [str(raw_steps)]

        contract_info = result.get("contract_info") or {}
        if isinstance(contract_info, dict):
            candidate_keys = (
                "contract_path",
                "path",
                "file_path",
                "contract_file",
                "quotation_path",
            )
            for key in candidate_keys:
                value = contract_info.get(key)
                if value:
                    contract_path = str(value)
                    break
        if not contract_path:
            direct_path = result.get("contract_path")
            if direct_path:
                contract_path = str(direct_path)
    else:
        text = str(result)

    return text, steps, contract_path


def get_step_icon(step: str) -> str:
    step_lower = step.lower()
    if "intent" in step_lower:
        return "🔍"
    if any(w in step_lower for w in ["price", "pricing", "csv", "lookup", "database"]):
        return "📊"
    if any(w in step_lower for w in ["rag", "retrieval", "doc grader", "document", "knowledge"]):
        return "📚"
    if any(w in step_lower for w in ["contract", "quotation"]):
        return "📝"
    if any(w in step_lower for w in ["hallucination", "verified", "flagged", "grader"]):
        return "🛡️"
    if "reflect" in step_lower:
        return "🔄"
    if "error" in step_lower:
        return "⚠️"
    return "⚙️"


def get_hallucination_badge_html(status: str) -> str:
    if status == "verified":
        return '<span class="badge-verified">VERIFIED</span>'
    if status == "flagged":
        return '<span class="badge-flagged">FLAGGED</span>'
    if status == "reviewed":
        return '<span class="badge-reviewed">REVIEWED</span>'
    return '<span class="badge-idle">Idle</span>'


def run_agent_and_update(user_message: str, history: list) -> None:
    try:
        run_agent = _load_agent()
        raw = run_agent(
            user_message,
            history,
            session_id=st.session_state.session_id,
            user_role=st.session_state.user_role,
            reflection_strictness=st.session_state.reflection_strictness,
        )
        text, steps, contract = normalize_agent_result(raw)
        st.session_state.messages.append({"role": "assistant", "content": text})
        st.session_state.agent_steps = steps
        st.session_state.last_contract_path = contract
        if isinstance(raw, tuple) and len(raw) >= 3:
            info = raw[2] if raw[2] else {}
            if isinstance(info, dict):
                st.session_state.hallucination_status = info.get("hallucination_status", "reviewed")
                st.session_state.price_confidence = info.get("price_confidence_score", 0.0)
                st.session_state.reflection_log = info.get("reflection_log", [])
            else:
                st.session_state.hallucination_status = "reviewed"
        else:
            st.session_state.hallucination_status = "reviewed"
    except Exception as exc:
        st.session_state.messages.append({"role": "assistant", "content": f"处理出错，请重试。({exc})"})
        st.session_state.hallucination_status = ""


def _format_time(ts: str) -> str:
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(ts.strip(), fmt).strftime("%H:%M")
        except ValueError:
            continue

    try:
        if ts.strip().isdigit():
            return datetime.fromtimestamp(int(ts.strip())).strftime("%H:%M")
    except (ValueError, OSError):
        pass

    return ts[:5] if len(ts) >= 5 else ts


# ---------------------------------------------------------------------------
# System Health Check
# ---------------------------------------------------------------------------
def _check_system_health() -> tuple[bool, bool, bool]:
    """Check CSV, LLM connectivity, ChromaDB status."""
    csv_ok = Path("data/prices.csv").exists()
    try:
        from config.settings import OPENAI_API_KEY, OPENAI_BASE_URL
        llm_ok = bool(OPENAI_API_KEY and OPENAI_BASE_URL)
    except Exception:
        llm_ok = False
    try:
        from rag.vectorstore import is_vectorstore_ready
        chroma_ok = is_vectorstore_ready()
    except Exception:
        chroma_ok = False
    return csv_ok, llm_ok, chroma_ok


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar() -> None:
    from ui.theme import Colors, Radius
    from ui.components import render_system_health, render_progress_steps, render_confidence_bar

    with st.sidebar:
        # ── Brand ─────────────────────────────────────────
        st.markdown(
            f'<div style="padding: 4px 0 16px;">'
            f'<div style="display: flex; align-items: center; gap: 10px;">'
            f'<div style="font-size: 28px; line-height: 1;">🚗</div>'
            f'<div>'
            f'<div style="font-weight: 700; font-size: 16px; color: {Colors.text}; letter-spacing: -0.3px;">'
            f'Car Export AI</div>'
            f'<div style="font-size: 11px; color: {Colors.text_tertiary}; letter-spacing: 0.3px;">'
            f'Defense-in-Depth Sales Agent</div>'
            f'</div></div></div>',
            unsafe_allow_html=True,
        )

        # ── User Info ─────────────────────────────────────
        st.markdown(
            f'<div style="height: 1px; background: {Colors.border_light}; margin: 4px 0 12px;"></div>',
            unsafe_allow_html=True,
        )
        role_icon = "👑" if st.session_state.user_role == "admin" else "👤"
        role_label = "Admin" if st.session_state.user_role == "admin" else "Sales"
        st.markdown(
            f'<div style="display: flex; align-items: center; gap: 10px; '
            f'background: {Colors.bg}; border-radius: {Radius.sm}px; '
            f'padding: 10px 14px;">'
            f'<span style="font-size: 18px;">{role_icon}</span>'
            f'<div>'
            f'<div style="font-size: 14px; font-weight: 600; color: {Colors.text};">'
            f'{st.session_state.user_name}</div>'
            f'<div style="font-size: 11px; color: {Colors.text_tertiary};">{role_label}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        if st.button("退出登录", use_container_width=True, key="logout_btn"):
            import uuid as _uuid
            keys_to_reset = [
                "logged_in", "user_name", "user_role", "messages",
                "agent_steps", "reflection_log", "last_contract_path",
                "hallucination_status", "price_confidence",
                "intervention_mode", "pending_response",
            ]
            for _k in keys_to_reset:
                st.session_state.pop(_k, None)
            st.session_state.session_id = str(_uuid.uuid4())
            st.rerun()

        # ── System Health ─────────────────────────────────
        st.markdown(
            f'<div style="height: 1px; background: {Colors.border_light}; margin: 12px 0;"></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="font-size: 11px; font-weight: 600; color: {Colors.text_secondary}; '
            f'text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 8px;">'
            f'System Status</div>',
            unsafe_allow_html=True,
        )
        csv_ok, llm_ok, chroma_ok = _check_system_health()
        st.markdown(render_system_health(csv_ok, llm_ok, chroma_ok), unsafe_allow_html=True)

        # ── Reflection Strictness ─────────────────────────
        st.markdown(
            f'<div style="height: 1px; background: {Colors.border_light}; margin: 12px 0;"></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="font-size: 11px; font-weight: 600; color: {Colors.text_secondary}; '
            f'text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 8px;">'
            f'Reflection Strictness</div>',
            unsafe_allow_html=True,
        )
        strictness_map = {"宽松 (Lenient)": "lenient", "标准 (Normal)": "normal", "严格 (Strict)": "strict"}
        sel = st.select_slider(
            "",
            options=list(strictness_map.keys()),
            value="标准 (Normal)",
            key="strictness_slider",
            label_visibility="collapsed",
        )
        st.session_state.reflection_strictness = strictness_map[sel]
        desc = {"lenient": "Fast mode -- skip extra checks", "normal": "Standard 3-step verification", "strict": "Maximum compliance + upsell check"}
        st.caption(desc.get(st.session_state.reflection_strictness, ""))

        # ── Hallucination Guard ───────────────────────────
        st.markdown(
            f'<div style="height: 1px; background: {Colors.border_light}; margin: 12px 0;"></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="font-size: 11px; font-weight: 600; color: {Colors.text_secondary}; '
            f'text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 8px;">'
            f'Guard Status</div>',
            unsafe_allow_html=True,
        )
        st.markdown(get_hallucination_badge_html(st.session_state.hallucination_status), unsafe_allow_html=True)

        conf = st.session_state.price_confidence
        if conf > 0:
            st.markdown(render_confidence_bar(conf), unsafe_allow_html=True)

        # ── Reflection Progress ───────────────────────────
        if st.session_state.reflection_log:
            last_log = st.session_state.reflection_log[-1] if st.session_state.reflection_log else {}
            s1_ok = last_log.get("step1_fact_check", {}).get("passed", False) if last_log else False
            s2_ok = last_log.get("step2_compliance", {}).get("passed", False) if last_log else False
            s3_ok = bool(last_log.get("step3_upsell")) if last_log else False

            pipeline_steps = [
                {"label": "FactCheck", "status": "done" if s1_ok else ("active" if not s1_ok and (s2_ok or s3_ok) else "active")},
                {"label": "Compliance", "status": "done" if s2_ok else ("active" if s1_ok else "pending")},
                {"label": "Upsell", "status": "done" if s3_ok else ("active" if s2_ok else "pending")},
            ]
            st.markdown(
                f'<div style="height: 1px; background: {Colors.border_light}; margin: 12px 0;"></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="font-size: 11px; font-weight: 600; color: {Colors.text_secondary}; '
                f'text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 4px;">'
                f'Reflection Pipeline</div>',
                unsafe_allow_html=True,
            )
            st.markdown(render_progress_steps(pipeline_steps), unsafe_allow_html=True)

        # ── Architecture Diagram (collapsed) ──────────────
        st.markdown(
            f'<div style="height: 1px; background: {Colors.border_light}; margin: 12px 0;"></div>',
            unsafe_allow_html=True,
        )
        with st.expander("Architecture", expanded=False):
            st.markdown(
                f'<div style="background: {Colors.bg}; border-radius: {Radius.sm}px; '
                f'padding: 10px 12px; font-size: 11px; color: {Colors.text}; '
                f'line-height: 2; font-family: \'JetBrains Mono\', monospace;">'
                f'Query &rarr; Intent Classification<br>'
                f'&nbsp;&darr; [Price] RapidFuzz + Cache<br>'
                f'&nbsp;&darr; [RAG] Doc Grader (CRAG)<br>'
                f'&nbsp;&darr; 3-Step Reflection<br>'
                f'&nbsp;&nbsp;&nbsp;S1 Price Accuracy<br>'
                f'&nbsp;&nbsp;&nbsp;S2 Compliance<br>'
                f'&nbsp;&nbsp;&nbsp;S3 Upsell Check<br>'
                f'&nbsp;&darr; [Low Conf] HITL<br>'
                f'&nbsp;&#10003; Output / &#8635; Retry(&le;2)'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── Reasoning Steps ───────────────────────────────
        st.markdown(
            f'<div style="height: 1px; background: {Colors.border_light}; margin: 12px 0;"></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="font-size: 11px; font-weight: 600; color: {Colors.text_secondary}; '
            f'text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 8px;">'
            f'Reasoning Steps</div>',
            unsafe_allow_html=True,
        )
        if st.session_state.agent_steps:
            for idx, step in enumerate(st.session_state.agent_steps, start=1):
                icon = get_step_icon(step)
                with st.expander(f"{icon} Step {idx}", expanded=False):
                    st.caption(step)
        else:
            st.caption("Send a message to see reasoning trace...")

        # ── Reflection Log ────────────────────────────────
        if st.session_state.reflection_log:
            st.markdown(
                f'<div style="height: 1px; background: {Colors.border_light}; margin: 12px 0;"></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="font-size: 11px; font-weight: 600; color: {Colors.text_secondary}; '
                f'text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 8px;">'
                f'Reflection Log</div>',
                unsafe_allow_html=True,
            )
            for i, log in enumerate(st.session_state.reflection_log, 1):
                with st.expander(f"Round {i}", expanded=False):
                    s1 = log.get("step1_fact_check", {})
                    s2 = log.get("step2_compliance", {})
                    s3 = log.get("step3_upsell", {})
                    st.markdown(f"**Step1 FactCheck**: {'PASS' if s1.get('passed') else 'FAIL'}")
                    if not s1.get("passed"):
                        st.caption(f"Error: {s1.get('error_type')} | Fix: {s1.get('correction_plan')}")
                    st.markdown(f"**Step2 Compliance**: {'PASS' if s2.get('passed') else 'FAIL'}")
                    if not s2.get("passed"):
                        st.caption(f"Error: {s2.get('error_type')} | Fix: {s2.get('correction_plan')}")
                    if s3:
                        st.markdown(f"**Step3 Upsell**: {'Recommend ' + s3.get('recommended_model','') if s3.get('should_upsell') else 'No recommendation'}")
                    st.caption(f"Overall: {'Pass' if log.get('overall_passed') else 'Fail'} | Strictness: {log.get('strictness_level','')}")

        # ── Contract Download ─────────────────────────────
        cp = st.session_state.last_contract_path
        if cp:
            p = Path(cp)
            if p.exists() and p.is_file():
                st.markdown(
                    f'<div style="height: 1px; background: {Colors.border_light}; margin: 12px 0;"></div>',
                    unsafe_allow_html=True,
                )
                with p.open("rb") as f:
                    st.download_button("Download Contract", f.read(), p.name, "text/markdown", use_container_width=True)

        # ── Footer ────────────────────────────────────────
        st.markdown(
            f'<div style="height: 1px; background: {Colors.border_light}; margin: 12px 0;"></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="font-size: 10px; color: {Colors.text_tertiary}; text-align: center; '
            f'letter-spacing: 0.3px; line-height: 1.8;">'
            f'LangGraph &middot; RapidFuzz &middot; CRAG<br>'
            f'3-Step Reflection &middot; HITL &middot; GPT</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Chat Tab  (using st.chat_message for native Streamlit experience)
# ---------------------------------------------------------------------------
def render_chat_tab() -> None:
    from ui.theme import Colors, Radius
    from ui.components import render_thinking_animation

    # Header
    st.markdown(
        f'<div style="margin-bottom: 16px;">'
        f'<h2 style="font-size: 20px; font-weight: 700; color: {Colors.text}; margin: 0 0 4px;">'
        f'Inquiry Simulation</h2>'
        f'<p style="color: {Colors.text_secondary}; font-size: 13px; margin: 0;">'
        f'Simulate WhatsApp customer inquiries &middot; Auto pricing &middot; Contract generation &middot; '
        f'3-step hallucination guard</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Example queries (only when no messages)
    if not st.session_state.messages:
        st.markdown(
            f'<p style="font-weight: 600; color: {Colors.text}; font-size: 13px; margin-bottom: 8px;">'
            f'Try a sample inquiry:</p>',
            unsafe_allow_html=True,
        )
        examples = [
            "Hi, what's the CIF price of BYD Seal to Lagos port?",
            "I need a 7-seat SUV under $20,000, what do you have?",
            "Contract: 2x Chery Tiggo 8 Pro, buyer ABC Trading, Dubai",
        ]
        cols = st.columns(3)
        for col, query in zip(cols, examples):
            if col.button(query, use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": query})
                with st.spinner("Thinking..."):
                    run_agent_and_update(query, [])
                st.rerun()
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Chat history using native st.chat_message
    if st.session_state.messages:
        for msg in st.session_state.messages:
            role = msg.get("role", "assistant")
            content = str(msg.get("content", ""))
            with st.chat_message(role):
                st.markdown(content)

    # ── HITL Intervention Panel ───────────────────────────
    if st.session_state.intervention_mode and st.session_state.messages:
        last_ai = next(
            (m["content"] for m in reversed(st.session_state.messages) if m.get("role") == "assistant"),
            "",
        )
        st.markdown(
            '<div class="intervention-panel">'
            f'<div style="font-weight: 600; color: {Colors.warning}; font-size: 14px; margin-bottom: 8px;">'
            f'Human-in-the-Loop -- Edit and confirm before sending</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        edited = st.text_area("Edit response", value=last_ai, height=150, key="intervention_edit_box")
        reason = st.text_input("Reason (optional)", placeholder="e.g., Price needs manual confirmation", key="intervention_reason")
        col_confirm, col_cancel = st.columns(2)
        if col_confirm.button("Confirm & Send", use_container_width=True, key="intervention_confirm", type="primary"):
            from agent.utils.intervention_log import log_intervention
            log_intervention(
                session_id=st.session_state.session_id,
                user_role=st.session_state.user_role,
                original_response=last_ai,
                edited_response=edited,
                reason=reason,
            )
            for i in range(len(st.session_state.messages) - 1, -1, -1):
                if st.session_state.messages[i].get("role") == "assistant":
                    st.session_state.messages[i]["content"] = edited
                    break
            st.session_state.intervention_mode = False
            st.success("Edit saved and logged to audit trail")
            st.rerun()
        if col_cancel.button("Cancel", use_container_width=True, key="intervention_cancel"):
            st.session_state.intervention_mode = False
            st.rerun()

    # ── Input Bar ─────────────────────────────────────────
    col_input, col_send, col_hitl = st.columns([5, 1, 1])
    with col_input:
        user_input = st.text_input(
            "",
            placeholder="Type your inquiry, e.g.: BYD Seal CIF price to Lagos?",
            label_visibility="collapsed",
            key="chat_input_box",
        )
    with col_send:
        submitted = st.button("Send", use_container_width=True, key="chat_send_btn", type="primary")
    with col_hitl:
        if st.button("HITL", use_container_width=True, key="hitl_btn", help="Pause AI, manually edit last response"):
            st.session_state.intervention_mode = True
            st.rerun()

    if submitted and user_input.strip():
        clean_input = user_input.strip()
        st.session_state.messages.append({"role": "user", "content": clean_input})
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
        with st.spinner("Thinking..."):
            run_agent_and_update(clean_input, history)
        st.rerun()


# ---------------------------------------------------------------------------
# WhatsApp Tab
# ---------------------------------------------------------------------------
def render_whatsapp_tab() -> None:
    from ui.theme import Colors, Radius

    st.markdown(
        f'<div style="margin-bottom: 16px;">'
        f'<h2 style="font-size: 20px; font-weight: 700; color: {Colors.text}; margin: 0 0 4px;">'
        f'WhatsApp Inquiry Manager</h2>'
        f'<p style="color: {Colors.text_secondary}; font-size: 13px; margin: 0;">'
        f'Receive and simulate real WhatsApp business inquiries</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        header_col, refresh_col = st.columns([5, 1])
        with header_col:
            st.markdown(
                f'<h4 style="font-weight: 600; color: {Colors.text}; font-size: 15px; margin: 0;">'
                f'Conversations</h4>',
                unsafe_allow_html=True,
            )
        with refresh_col:
            if st.button("Refresh", help="Reload conversations", use_container_width=True, key="wa_refresh"):
                st.rerun()

        _, load_history = _load_whatsapp()
        messages = load_history()

        if not messages:
            st.markdown(
                f'<div class="pro-card" style="text-align: center; padding: 40px 20px;">'
                f'<div style="font-size: 40px; margin-bottom: 12px; opacity: 0.5;">📭</div>'
                f'<div style="font-weight: 600; color: {Colors.text}; font-size: 15px; margin-bottom: 6px;">'
                f'No inquiries yet</div>'
                f'<div style="color: {Colors.text_secondary}; font-size: 13px; line-height: 1.6;">'
                f'Use the simulator on the right to send test messages<br>'
                f'or configure Green API for real WhatsApp inquiries'
                f'</div></div>',
                unsafe_allow_html=True,
            )
        else:
            grouped: dict[str, list[Any]] = defaultdict(list)
            for msg in messages:
                grouped[str(msg.get("chat_id", "unknown"))].append(msg)

            sorted_groups = sorted(
                grouped.items(),
                key=lambda x: str(x[1][-1].get("timestamp", "")) if x[1] else "",
                reverse=True,
            )

            for chat_id, chat_msgs in sorted_groups:
                phone = chat_id.replace("@c.us", "")
                sorted_msgs = sorted(chat_msgs, key=lambda m: str(m.get("timestamp", "")))
                preview_text = str(sorted_msgs[-1].get("text", "")) if sorted_msgs else ""
                last_preview = preview_text[:35] + ("..." if len(preview_text) > 35 else "")

                with st.expander(f"{phone}  --  {last_preview}", expanded=False):
                    for msg in sorted_msgs:
                        direction = str(msg.get("direction", "inbound")).lower()
                        sender = html.escape(str(msg.get("sender_name", "Unknown")))
                        text = html.escape(str(msg.get("text", ""))).replace("\n", "<br>")
                        t = _format_time(str(msg.get("timestamp", "")))

                        if direction == "inbound":
                            bubble = (
                                '<div style="display:flex;justify-content:flex-start;margin:6px 0">'
                                '<div class="wa-inbound">'
                                f'<div style="font-size:11px;color:{Colors.text_tertiary};margin-bottom:4px;font-weight:500">{sender} &middot; {t}</div>'
                                f'<div style="color:{Colors.text}">{text}</div>'
                                '</div></div>'
                            )
                        else:
                            bubble = (
                                '<div style="display:flex;justify-content:flex-end;margin:6px 0">'
                                '<div class="wa-outbound">'
                                f'<div style="font-size:11px;color:#5E8E60;margin-bottom:4px;font-weight:500">Agent &middot; {t}</div>'
                                f'<div style="color:{Colors.text}">{text}</div>'
                                '</div></div>'
                            )
                        st.markdown(bubble, unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="pro-card">', unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size: 15px; font-weight: 600; color: {Colors.text}; margin-bottom: 4px;">'
            f'Simulate WhatsApp Message</div>'
            f'<div style="font-size: 12px; color: {Colors.text_secondary}; margin-bottom: 12px;">'
            f'Test locally before connecting real WhatsApp</div>',
            unsafe_allow_html=True,
        )

        phone = st.text_input("Phone Number", value="8613800138000", help="International format, no + prefix", key="wa_phone")
        name = st.text_input("Buyer Name", value="Demo Buyer", key="wa_name")
        message = st.text_area(
            "Message",
            placeholder="Hello, I need pricing for 5x BYD Seal to Mombasa, Kenya. Please quote CIF.",
            height=110,
            key="wa_message",
        )
        wa_submitted = st.button("Send Simulation", use_container_width=True, key="wa_send", type="primary")

        if wa_submitted:
            if not phone.strip() or not message.strip():
                st.error("Phone and message cannot be empty")
            else:
                fake_payload = {
                    "typeWebhook": "incomingMessageReceived",
                    "instanceData": {"idInstance": "demo"},
                    "timestamp": int(datetime.now().timestamp()),
                    "senderData": {
                        "chatId": f"{phone.strip()}@c.us",
                        "sender": f"{phone.strip()}@c.us",
                        "senderName": name.strip() or "Demo Buyer",
                    },
                    "messageData": {
                        "typeMessage": "textMessage",
                        "textMessageData": {"textMessage": message.strip()},
                    },
                }
                try:
                    with st.spinner("Processing..."):
                        handle_incoming, _ = _load_whatsapp()
                        result = handle_incoming(fake_payload)
                    st.success("Message processed successfully")
                    resp = (
                        result.get("response")
                        or result.get("response_text")
                        or result.get("text")
                        or "(no response)"
                    )
                    st.markdown(f"**Agent Reply:**\n\n{resp}")
                    if result.get("contract_info"):
                        st.json(result["contract_info"])
                    st.rerun()
                except Exception as exc:
                    st.error(f"Processing failed: {exc}")

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        with st.expander("Green API Setup Guide", expanded=False):
            st.markdown(
                "1. Register at [green-api.com](https://green-api.com/) (free tier)\n"
                "2. Connect WhatsApp and obtain Instance ID + Token\n"
                "3. Set webhook URL: `http://your-server:8000/webhook`\n"
                "4. Add credentials to `.env`\n"
                "5. Run `python server.py`"
            )


# ---------------------------------------------------------------------------
# Login Page
# ---------------------------------------------------------------------------
def _render_login_page() -> None:
    from ui.theme import Colors, Radius

    # Full-viewport gradient background
    st.markdown(
        f'<div class="login-bg">'
        # Title block
        f'<div style="text-align: center; margin-bottom: 32px;">'
        f'<div style="font-size: 48px; margin-bottom: 12px;">🚗</div>'
        f'<h1 style="color: white; font-size: 28px; font-weight: 800; margin: 0 0 8px; '
        f'letter-spacing: -0.5px;">Car Export AI Sales Assistant</h1>'
        f'<div class="typing-text" style="color: rgba(255,255,255,0.8); font-size: 14px; '
        f'margin: 0 auto; display: inline-block;">'
        f'WhatsApp Inquiry &middot; Knowledge QA &middot; Contract Generation</div>'
        f'</div>'
        # Feature badges row
        f'<div style="display: flex; gap: 12px; margin-bottom: 32px; flex-wrap: wrap; justify-content: center;">'
        f'<div style="background: rgba(255,255,255,0.1); backdrop-filter: blur(8px); '
        f'border: 1px solid rgba(255,255,255,0.15); border-radius: {Radius.md}px; '
        f'padding: 14px 16px; text-align: center; min-width: 100px;">'
        f'<div style="font-size: 22px; margin-bottom: 4px;">📱</div>'
        f'<div style="font-size: 11px; font-weight: 600; color: rgba(255,255,255,0.9); line-height: 1.3;">'
        f'WhatsApp<br>Integration</div></div>'
        f'<div style="background: rgba(255,255,255,0.1); backdrop-filter: blur(8px); '
        f'border: 1px solid rgba(255,255,255,0.15); border-radius: {Radius.md}px; '
        f'padding: 14px 16px; text-align: center; min-width: 100px;">'
        f'<div style="font-size: 22px; margin-bottom: 4px;">🗄️</div>'
        f'<div style="font-size: 11px; font-weight: 600; color: rgba(255,255,255,0.9); line-height: 1.3;">'
        f'Vehicle DB<br>Q&A</div></div>'
        f'<div style="background: rgba(255,255,255,0.1); backdrop-filter: blur(8px); '
        f'border: 1px solid rgba(255,255,255,0.15); border-radius: {Radius.md}px; '
        f'padding: 14px 16px; text-align: center; min-width: 100px;">'
        f'<div style="font-size: 22px; margin-bottom: 4px;">📋</div>'
        f'<div style="font-size: 11px; font-weight: 600; color: rgba(255,255,255,0.9); line-height: 1.3;">'
        f'Auto Contract<br>Generation</div></div>'
        f'<div style="background: rgba(255,255,255,0.1); backdrop-filter: blur(8px); '
        f'border: 1px solid rgba(255,255,255,0.15); border-radius: {Radius.md}px; '
        f'padding: 14px 16px; text-align: center; min-width: 100px;">'
        f'<div style="font-size: 22px; margin-bottom: 4px;">🛡️</div>'
        f'<div style="font-size: 11px; font-weight: 600; color: rgba(255,255,255,0.9); line-height: 1.3;">'
        f'3-Step Reflection<br>Anti-Hallucination</div></div>'
        f'</div>'
        # Tech pills
        f'<div style="display: flex; gap: 8px; flex-wrap: wrap; justify-content: center; margin-bottom: 24px;">'
        f'<span class="tech-pill">LangGraph</span>'
        f'<span class="tech-pill">CRAG</span>'
        f'<span class="tech-pill">HITL</span>'
        f'<span class="tech-pill">Hybrid RAG</span>'
        f'<span class="tech-pill">RapidFuzz</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Login form (outside the gradient div, rendered by Streamlit natively)
    _, col, _ = st.columns([1.2, 1, 1.2])
    with col:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown(
                f'<div style="text-align: center; margin-bottom: 12px;">'
                f'<div style="font-size: 17px; font-weight: 700; color: {Colors.text};">Sign In</div>'
                f'<div style="font-size: 12px; color: {Colors.text_secondary}; margin-top: 2px;">'
                f'Enter your name to continue</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            name_input = st.text_input("Name", placeholder="Your name", key="login_name_main",
                                       label_visibility="collapsed")
            role_sel = st.selectbox(
                "Role",
                ["Sales", "Admin"],
                key="login_role_main",
            )
            if st.button("Sign In", use_container_width=True, type="primary", key="login_btn_main"):
                if name_input.strip():
                    st.session_state.user_name = name_input.strip()
                    st.session_state.user_role = "admin" if "Admin" in role_sel else "sales"
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("Please enter your name")

        # Scenario description
        st.markdown(
            f'<div style="background: {Colors.accent_light}; border-radius: {Radius.sm}px; '
            f'padding: 14px 16px; margin-top: 12px;">'
            f'<div style="font-size: 12px; font-weight: 600; color: {Colors.accent}; margin-bottom: 6px;">'
            f'Typical Use Cases</div>'
            f'<div style="font-size: 12px; color: {Colors.text}; line-height: 1.8;">'
            f'&bull; Customer asks "BYD Seal export to Lagos, how much?" &rarr; Auto-reply with FOB/CIF quote<br>'
            f'&bull; Customer asks "7-seat SUV under $20k?" &rarr; Filter and recommend from vehicle DB<br>'
            f'&bull; Customer confirms model &rarr; Auto-generate quotation contract<br>'
            f'&bull; Low confidence &rarr; Pause and hand off to human sales rep'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Telegram Tab
# ---------------------------------------------------------------------------
def _render_telegram_tab() -> None:
    """Telegram Bot real-time conversation monitor."""
    import os as _os
    from telegram.handler import load_telegram_history
    from ui.theme import Colors, Radius

    st.markdown(
        f'<div style="margin-bottom: 16px;">'
        f'<h2 style="font-size: 20px; font-weight: 700; color: {Colors.text}; margin: 0 0 4px;">'
        f'Telegram Bot Monitor</h2>'
        f'<p style="color: {Colors.text_secondary}; font-size: 13px; margin: 0;">'
        f'Real-time customer inquiries via Telegram, auto-replied by AI Agent</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    bot_name = _os.getenv("TELEGRAM_BOT_USERNAME", "ksnzizjwns_bot")
    token = _os.getenv("TELEGRAM_BOT_TOKEN", "")

    if token:
        st.markdown(
            f'<div style="display: flex; align-items: center; gap: 10px; '
            f'background: #ECFDF5; border-radius: {Radius.sm}px; '
            f'padding: 12px 16px; margin-bottom: 16px; '
            f'border: 1px solid rgba(16, 185, 129, 0.2);">'
            f'<div style="width: 8px; height: 8px; border-radius: 50%; background: {Colors.success};"></div>'
            f'<span style="font-size: 13px; color: {Colors.text};">'
            f'Bot running &mdash; '
            f'<a href="https://t.me/{bot_name}" target="_blank" '
            f'style="color: {Colors.accent}; font-weight: 600; text-decoration: none;">'
            f't.me/{bot_name}</a></span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.warning("TELEGRAM_BOT_TOKEN not configured")
        return

    col_r, col_spacer = st.columns([1, 5])
    with col_r:
        if st.button("Refresh", use_container_width=True, key="tg_refresh"):
            st.rerun()

    records = load_telegram_history()
    if not records:
        st.markdown(
            f'<div class="pro-card" style="text-align: center; padding: 32px 20px;">'
            f'<div style="font-size: 36px; margin-bottom: 10px; opacity: 0.4;">💬</div>'
            f'<div style="font-size: 14px; color: {Colors.text_secondary};">'
            f'No conversations yet. Send a message on Telegram to get started.</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f'<div style="font-size: 13px; color: {Colors.text_secondary}; margin-bottom: 12px;">'
        f'{len(records)} conversation{"s" if len(records) != 1 else ""} total</div>',
        unsafe_allow_html=True,
    )

    recent = records[-20:]
    for idx, rec in enumerate(reversed(recent)):
        ts = rec.get("timestamp", "")[:16]
        username = rec.get("username", "unknown")
        user_msg = rec.get("user_message", "")
        agent_reply = rec.get("agent_reply", "")
        is_latest = (idx == 0)

        with st.expander(f"[{ts}] @{username}: {user_msg[:50]}", expanded=is_latest):
            col_u, col_a = st.columns(2)
            with col_u:
                st.markdown(
                    f'<div style="font-size: 12px; font-weight: 600; color: {Colors.text_secondary}; '
                    f'margin-bottom: 6px;">Customer</div>',
                    unsafe_allow_html=True,
                )
                st.text_area("", value=user_msg, height=100, disabled=True,
                             key=f"tg_u_{idx}")
            with col_a:
                st.markdown(
                    f'<div style="font-size: 12px; font-weight: 600; color: {Colors.text_secondary}; '
                    f'margin-bottom: 6px;">Agent Reply</div>',
                    unsafe_allow_html=True,
                )
                st.text_area("", value=agent_reply, height=100, disabled=True,
                             key=f"tg_a_{idx}")
            if is_latest:
                st.markdown(
                    f'<div style="font-size: 11px; color: {Colors.accent}; font-weight: 500; '
                    f'margin-top: 4px;">Latest conversation</div>',
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# Admin Tab
# ---------------------------------------------------------------------------
def _render_admin_tab() -> None:
    """Admin-only: review all human intervention logs with approve/reject + batch KB sync."""
    from agent.utils.intervention_log import load_interventions, update_intervention_status
    from ui.theme import Colors, Radius
    from ui.components import render_metric_card

    st.markdown(
        f'<div style="margin-bottom: 16px;">'
        f'<h2 style="font-size: 20px; font-weight: 700; color: {Colors.text}; margin: 0 0 4px;">'
        f'Intervention Review</h2>'
        f'<p style="color: {Colors.text_secondary}; font-size: 13px; margin: 0;">'
        f'Review all HITL intervention records, approve/reject and sync to knowledge base</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    col_r, col_f = st.columns([3, 1])
    with col_f:
        if st.button("Refresh", use_container_width=True, key="admin_refresh"):
            st.rerun()

    logs = load_interventions(user_role="admin")
    if not logs:
        st.markdown(
            f'<div class="pro-card" style="text-align: center; padding: 32px 20px;">'
            f'<div style="font-size: 36px; margin-bottom: 10px; opacity: 0.4;">📋</div>'
            f'<div style="font-size: 14px; color: {Colors.text_secondary};">'
            f'No intervention records yet</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        return

    # Summary stats
    total = len(logs)
    approved_count = sum(1 for e in logs if e.get("approved") is True)
    rejected_count = sum(1 for e in logs if e.get("approved") is False)
    pending_count = total - approved_count - rejected_count
    synced_count = sum(1 for e in logs if e.get("synced_to_kb"))

    # Styled metric cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(render_metric_card("Total Records", total, "📋", Colors.accent), unsafe_allow_html=True)
    with c2:
        st.markdown(render_metric_card("Approved", approved_count, "✅", Colors.success), unsafe_allow_html=True)
    with c3:
        st.markdown(render_metric_card("Rejected", rejected_count, "✗", Colors.danger), unsafe_allow_html=True)
    with c4:
        st.markdown(render_metric_card("Synced to KB", synced_count, "📥", Colors.accent), unsafe_allow_html=True)

    # Plotly pie chart — distribution of approved / rejected / pending
    try:
        import plotly.graph_objects as go

        labels = ["Approved", "Rejected", "Pending"]
        values = [approved_count, rejected_count, pending_count]
        colors_pie = [Colors.success, Colors.danger, Colors.warning]

        # Only show chart if there is data
        if any(v > 0 for v in values):
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker=dict(colors=colors_pie, line=dict(color="white", width=2)),
                textinfo="label+value",
                textfont=dict(size=12),
                hoverinfo="label+percent+value",
            )])
            fig.update_layout(
                showlegend=False,
                margin=dict(t=20, b=20, l=20, r=20),
                height=220,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                annotations=[dict(
                    text=f"<b>{total}</b><br><span style='font-size:11px;color:{Colors.text_secondary}'>total</span>",
                    x=0.5, y=0.5, font_size=20, showarrow=False,
                    font=dict(color=Colors.text),
                )],
            )
            st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        pass  # Plotly not installed, skip chart silently

    st.markdown(
        f'<div style="height: 1px; background: {Colors.border_light}; margin: 16px 0;"></div>',
        unsafe_allow_html=True,
    )

    # Batch sync button
    pending_sync = [e for e in logs if e.get("approved") is True and not e.get("synced_to_kb")]
    if pending_sync:
        if st.button(
            f"Batch sync {len(pending_sync)} approved records to KB",
            type="primary",
            use_container_width=True,
            key="admin_batch_sync",
        ):
            synced = 0
            for entry in pending_sync:
                ok = update_intervention_status(
                    timestamp=entry.get("timestamp", ""),
                    session_id=entry.get("session_id", ""),
                    approved=True,
                    sync_to_kb=True,
                )
                if ok:
                    synced += 1
            st.success(f"Synced {synced}/{len(pending_sync)} records to ChromaDB")
            st.rerun()

    st.markdown(
        f'<div style="font-size: 13px; color: {Colors.text_secondary}; margin-bottom: 12px;">'
        f'{total} records total ({pending_count} pending review)</div>',
        unsafe_allow_html=True,
    )

    for entry in reversed(logs):
        ts = entry.get("timestamp", "")[:16]
        sid = entry.get("session_id", "")[-8:]
        role = entry.get("user_role", "")
        approved = entry.get("approved")
        synced = entry.get("synced_to_kb", False)

        # Status badge
        if approved is True:
            status_badge = "Approved" + (" + Synced" if synced else "")
            badge_color = Colors.success
        elif approved is False:
            status_badge = "Rejected"
            badge_color = Colors.danger
        else:
            status_badge = "Pending"
            badge_color = Colors.warning

        with st.expander(f"[{ts}] {role} ...{sid}  |  {status_badge}", expanded=(approved is None)):
            col_orig, col_edit = st.columns(2)
            with col_orig:
                st.markdown(
                    f'<div style="font-size: 12px; font-weight: 600; color: {Colors.text_secondary}; '
                    f'margin-bottom: 6px;">Original Response</div>',
                    unsafe_allow_html=True,
                )
                st.text_area("", value=entry.get("original_response", ""), height=120, disabled=True, key=f"orig_{sid}_{ts}")
            with col_edit:
                st.markdown(
                    f'<div style="font-size: 12px; font-weight: 600; color: {Colors.text_secondary}; '
                    f'margin-bottom: 6px;">Edited Response</div>',
                    unsafe_allow_html=True,
                )
                st.text_area("", value=entry.get("edited_response", ""), height=120, disabled=True, key=f"edit_{sid}_{ts}")
            if entry.get("reason"):
                st.caption(f"Reason: {entry['reason']}")
            if entry.get("reviewed_at"):
                st.caption(f"Reviewed: {entry['reviewed_at']}")

            # Approve / Reject buttons (only if not yet reviewed)
            if approved is None:
                col_a, col_r2, col_s = st.columns(3)
                with col_a:
                    if st.button("Approve", key=f"approve_{sid}_{ts}", use_container_width=True):
                        update_intervention_status(
                            timestamp=entry.get("timestamp", ""),
                            session_id=entry.get("session_id", ""),
                            approved=True,
                        )
                        st.rerun()
                with col_r2:
                    if st.button("Reject", key=f"reject_{sid}_{ts}", use_container_width=True):
                        update_intervention_status(
                            timestamp=entry.get("timestamp", ""),
                            session_id=entry.get("session_id", ""),
                            approved=False,
                        )
                        st.rerun()
                with col_s:
                    if st.button("Approve + Sync", key=f"sync_{sid}_{ts}", use_container_width=True, type="primary"):
                        update_intervention_status(
                            timestamp=entry.get("timestamp", ""),
                            session_id=entry.get("session_id", ""),
                            approved=True,
                            sync_to_kb=True,
                        )
                        st.success("Synced to knowledge base")
                        st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    inject_css()
    init_session_state()

    # ── Login gate ────────────────────────────────────────
    if not st.session_state.logged_in:
        _render_login_page()
        return

    render_sidebar()

    # Telegram bot 改为独立进程运行 (scripts/run_telegram.py)
    # 不再在 Streamlit 里启动轮询，避免多进程重复回复

    tabs = ["💬  Inquiry Chat", "📱  WhatsApp", "🤖  Telegram"]
    if st.session_state.get("user_role") == "admin":
        tabs.append("🔍  Admin Review")

    tab_results = st.tabs(tabs)
    with tab_results[0]:
        render_chat_tab()
    with tab_results[1]:
        render_whatsapp_tab()
    with tab_results[2]:
        _render_telegram_tab()
    if len(tab_results) > 3:
        with tab_results[3]:
            _render_admin_tab()


if __name__ == "__main__":
    main()
