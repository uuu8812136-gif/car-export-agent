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


def inject_css() -> None:
    st.markdown(
        """
        <style>
        body, .stApp {
            font-family: -apple-system, BlinkMacSystemFont, SF Pro Display, Helvetica Neue, sans-serif;
            background: #F2F2F7;
        }

        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        /* 完全隐藏顶部工具栏，消除占位空间 */
        header[data-testid="stHeader"] {
            height: 0 !important;
            min-height: 0 !important;
            padding: 0 !important;
            visibility: hidden !important;
        }
        /* 消除 header 留下的 padding */
        .stApp > div:first-child {
            margin-top: 0 !important;
        }
        .block-container {
            padding-top: 1rem !important;
        }

        [data-testid="stSidebar"] {
            background: rgba(255,255,255,0.95);
            backdrop-filter: blur(20px);
            border-right: 1px solid rgba(0,0,0,0.06);
        }

        .ios-card {
            background: white;
            border-radius: 16px;
            padding: 20px;
            margin: 8px 0;
            box-shadow: 0 2px 12px rgba(0,0,0,0.06);
            border: 1px solid rgba(0,0,0,0.04);
        }

        .badge-verified {
            display: inline-block;
            background: #34C759;
            color: white;
            border-radius: 20px;
            padding: 6px 16px;
            font-size: 13px;
            font-weight: 600;
        }

        .badge-flagged {
            display: inline-block;
            background: #FF3B30;
            color: white;
            border-radius: 20px;
            padding: 6px 16px;
            font-size: 13px;
            font-weight: 600;
        }

        .badge-reviewed {
            display: inline-block;
            background: #FF9500;
            color: white;
            border-radius: 20px;
            padding: 6px 16px;
            font-size: 13px;
            font-weight: 600;
        }

        .badge-idle {
            display: inline-block;
            background: #8E8E93;
            color: white;
            border-radius: 20px;
            padding: 6px 16px;
            font-size: 13px;
            font-weight: 600;
        }

        .user-bubble {
            background: #007AFF;
            color: white;
            border-radius: 18px 18px 4px 18px;
            padding: 12px 16px;
            max-width: 75%;
            display: inline-block;
            font-size: 15px;
            line-height: 1.5;
            box-shadow: 0 2px 8px rgba(0,122,255,0.25);
            word-wrap: break-word;
            overflow-wrap: break-word;
        }

        .assistant-bubble {
            background: white;
            color: #1C1C1E;
            border-radius: 18px 18px 18px 4px;
            padding: 12px 16px;
            max-width: 80%;
            display: inline-block;
            font-size: 15px;
            line-height: 1.5;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            border: 1px solid rgba(0,0,0,0.06);
            word-wrap: break-word;
            overflow-wrap: break-word;
        }

        .wa-inbound {
            background: white;
            border-radius: 0px 16px 16px 16px;
            padding: 10px 14px;
            max-width: 80%;
            box-shadow: 0 1px 4px rgba(0,0,0,0.08);
            font-size: 14px;
            border: 1px solid rgba(0,0,0,0.05);
            word-wrap: break-word;
            overflow-wrap: break-word;
        }

        .wa-outbound {
            background: #DCF8C6;
            border-radius: 16px 0px 16px 16px;
            padding: 10px 14px;
            max-width: 80%;
            margin-left: auto;
            box-shadow: 0 1px 4px rgba(0,0,0,0.08);
            font-size: 14px;
            word-wrap: break-word;
            overflow-wrap: break-word;
        }

        .stButton>button {
            border-radius: 12px;
            font-weight: 500;
            transition: all 0.2s ease;
            font-size: 14px;
        }

        .stButton>button:hover {
            transform: scale(1.02);
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 10px;
            padding: 8px 16px;
            font-weight: 500;
        }

        .stTabs [aria-selected="true"] {
            background: white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        .stTextInput>div>div>input,
        .stTextArea>div>div>textarea {
            border-radius: 12px;
            border: 1.5px solid #E5E5EA;
            font-size: 15px;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }

        .stTextInput>div>div>input:focus,
        .stTextArea>div>div>textarea:focus {
            border-color: #007AFF !important;
            box-shadow: 0 0 0 3px rgba(0,122,255,0.1) !important;
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 1.5rem;
        }
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
        return '<span class="badge-verified">🟢 VERIFIED</span>'
    if status == "flagged":
        return '<span class="badge-flagged">🔴 FLAGGED</span>'
    if status == "reviewed":
        return '<span class="badge-reviewed">🟡 REVIEWED</span>'
    return '<span class="badge-idle">⬜ 待机中</span>'


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


def render_sidebar() -> None:
    with st.sidebar:
        # Brand
        st.markdown(
            '<div style="padding:8px 0 12px">'
            '<div style="font-size:32px">🚗</div>'
            '<div style="font-weight:700;font-size:17px;color:#1C1C1E">汽车出口 AI V2</div>'
            '<div style="font-size:11px;color:#8E8E93">Defense-in-Depth · 人工介入 · CRAG</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        # ── 用户信息 ─────────────────────────────────────
        st.markdown("---")
        role_icon = "👑" if st.session_state.user_role == "admin" else "👤"
        st.markdown(
            f'<div style="background:#F2F2F7;border-radius:10px;padding:10px 14px;font-size:13px;">'
            f'<span style="font-size:16px">{role_icon}</span> '
            f'<b style="color:#1C1C1E">{st.session_state.user_name}</b>'
            f'<span style="color:#8E8E93;margin-left:6px">({st.session_state.user_role})</span>'
            f'</div>',
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

        # ── 反思严格度控制 ───────────────────────────────
        st.markdown("---")
        st.markdown("**⚙️ 反思严格度**")
        strictness_map = {"宽松 (Lenient)": "lenient", "标准 (Normal)": "normal", "严格 (Strict)": "strict"}
        sel = st.select_slider(
            "",
            options=list(strictness_map.keys()),
            value="标准 (Normal)",
            key="strictness_slider",
            label_visibility="collapsed",
        )
        st.session_state.reflection_strictness = strictness_map[sel]
        desc = {"lenient": "⚡ 跳过追加推荐检查，响应更快", "normal": "✅ 标准三步检查", "strict": "🔒 最严格合规+追加推荐"}
        st.caption(desc.get(st.session_state.reflection_strictness, ""))

        # ── 幻觉防护状态 ─────────────────────────────────
        st.markdown("---")
        st.markdown("**🛡️ 幻觉防护状态**")
        st.markdown(get_hallucination_badge_html(st.session_state.hallucination_status), unsafe_allow_html=True)
        conf = st.session_state.price_confidence
        if conf > 0:
            color = "#34C759" if conf >= 0.85 else "#FF9500" if conf >= 0.70 else "#FF3B30"
            st.markdown(
                f'<div style="margin-top:6px;font-size:12px;color:{color};font-weight:600">'
                f'匹配置信度: {conf*100:.0f}%</div>',
                unsafe_allow_html=True,
            )

        # ── Pipeline 架构图 ─────────────────────────────
        st.markdown("---")
        st.markdown("**🏗️ 防幻觉架构 (2026)**")
        st.markdown(
            '<div style="background:#F2F2F7;border-radius:10px;padding:10px 12px;'
            'font-size:11px;color:#1C1C1E;line-height:2;font-family:monospace">'
            '询盘 → 意图分类<br>'
            '&nbsp;↓ [价格] RapidFuzz+Cache<br>'
            '&nbsp;↓ [RAG] Doc Grader (CRAG)<br>'
            '&nbsp;↓ 三步反思工作流<br>'
            '&nbsp;&nbsp;&nbsp;Step1 价格准确性<br>'
            '&nbsp;&nbsp;&nbsp;Step2 话术合规<br>'
            '&nbsp;&nbsp;&nbsp;Step3 追加推荐<br>'
            '&nbsp;↓ [低置信] 人工介入<br>'
            '&nbsp;✅ 输出 / 🔄 重试(≤2)'
            '</div>',
            unsafe_allow_html=True,
        )

        # ── 推理步骤 ─────────────────────────────────────
        st.markdown("---")
        st.markdown("**🧠 推理步骤**")
        if st.session_state.agent_steps:
            for idx, step in enumerate(st.session_state.agent_steps, start=1):
                icon = get_step_icon(step)
                with st.expander(f"{icon} 步骤 {idx}", expanded=False):
                    st.caption(step)
        else:
            st.caption("💭 发送消息后查看推理过程...")

        # ── 反思日志 ─────────────────────────────────────
        if st.session_state.reflection_log:
            st.markdown("---")
            st.markdown("**📋 反思日志**")
            for i, log in enumerate(st.session_state.reflection_log, 1):
                with st.expander(f"第 {i} 轮反思", expanded=False):
                    s1 = log.get("step1_fact_check", {})
                    s2 = log.get("step2_compliance", {})
                    s3 = log.get("step3_upsell", {})
                    st.markdown(f"**Step1 FactCheck**: {'✅ PASS' if s1.get('passed') else '❌ FAIL'}")
                    if not s1.get("passed"):
                        st.caption(f"错误: {s1.get('error_type')} | 修正: {s1.get('correction_plan')}")
                    st.markdown(f"**Step2 Compliance**: {'✅ PASS' if s2.get('passed') else '❌ FAIL'}")
                    if not s2.get("passed"):
                        st.caption(f"错误: {s2.get('error_type')} | 修正: {s2.get('correction_plan')}")
                    if s3:
                        st.markdown(f"**Step3 Upsell**: {'推荐 ' + s3.get('recommended_model','') if s3.get('should_upsell') else '不推荐'}")
                    st.caption(f"整体: {'通过' if log.get('overall_passed') else '失败'} | 严格度: {log.get('strictness_level','')}")

        # ── 合同下载 ─────────────────────────────────────
        cp = st.session_state.last_contract_path
        if cp:
            p = Path(cp)
            if p.exists() and p.is_file():
                st.markdown("---")
                with p.open("rb") as f:
                    st.download_button("📄 下载报价合同", f.read(), p.name, "text/markdown", use_container_width=True)

        st.markdown("---")
        st.caption("LangGraph · RapidFuzz · CRAG · 三步反思 · HITL · GPT")


def render_chat_tab() -> None:
    st.markdown(
        '<h2 style="font-size:22px;font-weight:700;color:#1C1C1E;margin-bottom:2px">💬 WhatsApp 询盘模拟对话</h2>'
        '<p style="color:#8E8E93;font-size:13px;margin-bottom:14px">'
        '模拟真实 WhatsApp 客户问询 · 车辆知识库问答 · 自动报价 · 合同生成 · 三步防幻觉检查'
        '</p>',
        unsafe_allow_html=True,
    )

    if not st.session_state.messages:
        st.markdown('<p style="font-weight:600;color:#1C1C1E;font-size:14px">💡 模拟客户发来的询盘：</p>', unsafe_allow_html=True)
        examples = [
            "Hi, what's the CIF price of BYD Seal to Lagos port?",
            "I need a 7-seat SUV under $20,000, what do you have?",
            "Contract: 2x Chery Tiggo 8 Pro, buyer ABC Trading, Dubai",
        ]
        cols = st.columns(3)
        for col, query in zip(cols, examples):
            if col.button(query, use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": query})
                with st.spinner("🤔 思考中..."):
                    run_agent_and_update(query, [])
                st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state.messages:
        chat_html = '<div style="max-height:420px;overflow-y:auto;padding:4px 0;margin-bottom:12px">'
        for msg in st.session_state.messages:
            role = msg.get("role", "assistant")
            content = html.escape(str(msg.get("content", ""))).replace("\n", "<br>")
            if role == "user":
                chat_html += (
                    '<div style="display:flex;justify-content:flex-end;margin-bottom:10px">'
                    f'<div class="user-bubble">{content}</div>'
                    '</div>'
                )
            else:
                chat_html += (
                    '<div style="display:flex;justify-content:flex-start;margin-bottom:10px">'
                    f'<div class="assistant-bubble">{content}</div>'
                    '</div>'
                )
        chat_html += "</div>"
        st.markdown(chat_html, unsafe_allow_html=True)

    # ── 人工介入面板 ──────────────────────────────────
    if st.session_state.intervention_mode and st.session_state.messages:
        last_ai = next(
            (m["content"] for m in reversed(st.session_state.messages) if m.get("role") == "assistant"),
            "",
        )
        st.markdown(
            '<div style="background:#FFF3CD;border-radius:12px;padding:12px 16px;'
            'border:1px solid #FF9500;margin-bottom:12px">'
            '<b style="color:#FF9500">⏸️ 人工介入模式 — 编辑后点击确认发送</b></div>',
            unsafe_allow_html=True,
        )
        edited = st.text_area("编辑回复内容", value=last_ai, height=150, key="intervention_edit_box")
        reason = st.text_input("介入原因（可选）", placeholder="例如：价格需人工确认", key="intervention_reason")
        col_confirm, col_cancel = st.columns(2)
        if col_confirm.button("✅ 确认并发送", use_container_width=True, key="intervention_confirm"):
            from agent.utils.intervention_log import log_intervention
            log_intervention(
                session_id=st.session_state.session_id,
                user_role=st.session_state.user_role,
                original_response=last_ai,
                edited_response=edited,
                reason=reason,
            )
            # Replace last AI message with edited version
            for i in range(len(st.session_state.messages) - 1, -1, -1):
                if st.session_state.messages[i].get("role") == "assistant":
                    st.session_state.messages[i]["content"] = edited
                    break
            st.session_state.intervention_mode = False
            st.success("✅ 修改已保存并记录到审计日志")
            st.rerun()
        if col_cancel.button("❌ 取消介入", use_container_width=True, key="intervention_cancel"):
            st.session_state.intervention_mode = False
            st.rerun()

    # ── 输入栏 ───────────────────────────────────────
    col_input, col_send, col_hitl = st.columns([5, 1, 1])
    with col_input:
        user_input = st.text_input(
            "",
            placeholder="输入询盘内容，例如：BYD Seal出口到拉各斯的价格？",
            label_visibility="collapsed",
            key="chat_input_box",
        )
    with col_send:
        submitted = st.button("发送 ↑", use_container_width=True, key="chat_send_btn")
    with col_hitl:
        if st.button("⏸️ 介入", use_container_width=True, key="hitl_btn", help="暂停AI，手动编辑最后一条回复"):
            st.session_state.intervention_mode = True
            st.rerun()

    if submitted and user_input.strip():
        clean_input = user_input.strip()
        st.session_state.messages.append({"role": "user", "content": clean_input})
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
        with st.spinner("🤔 思考中..."):
            run_agent_and_update(clean_input, history)
        st.rerun()


def render_whatsapp_tab() -> None:
    st.markdown(
        '<h2 style="font-size:24px;font-weight:700;color:#1C1C1E;margin-bottom:4px">📱 WhatsApp 询盘管理</h2>'
        '<p style="color:#8E8E93;font-size:14px;margin-bottom:16px">接收和模拟真实WhatsApp商业询盘</p>',
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        header_col, refresh_col = st.columns([5, 1])
        with header_col:
            st.markdown('<h4 style="font-weight:600;color:#1C1C1E;font-size:17px;margin:0">对话记录</h4>', unsafe_allow_html=True)
        with refresh_col:
            if st.button("🔄", help="刷新对话记录", use_container_width=True):
                st.rerun()

        _, load_history = _load_whatsapp()
        messages = load_history()

        if not messages:
            empty_html = (
                '<div class="ios-card" style="text-align:center;padding:40px 20px">'
                '<div style="font-size:48px;margin-bottom:12px">📭</div>'
                '<div style="font-weight:600;color:#1C1C1E;font-size:16px;margin-bottom:8px">暂无询盘记录</div>'
                '<div style="color:#8E8E93;font-size:14px;line-height:1.6">'
                '使用右侧模拟器发送测试消息<br>或配置 Green API 接收真实 WhatsApp 询盘'
                '</div></div>'
            )
            st.markdown(empty_html, unsafe_allow_html=True)
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

                with st.expander(f"📱 {phone}  ·  {last_preview}", expanded=False):
                    for msg in sorted_msgs:
                        direction = str(msg.get("direction", "inbound")).lower()
                        sender = html.escape(str(msg.get("sender_name", "Unknown")))
                        text = html.escape(str(msg.get("text", ""))).replace("\n", "<br>")
                        t = _format_time(str(msg.get("timestamp", "")))

                        if direction == "inbound":
                            bubble = (
                                '<div style="display:flex;justify-content:flex-start;margin:6px 0">'
                                '<div class="wa-inbound">'
                                f'<div style="font-size:11px;color:#8E8E93;margin-bottom:4px;font-weight:500">{sender} · {t}</div>'
                                f'<div style="color:#1C1C1E">{text}</div>'
                                '</div></div>'
                            )
                        else:
                            bubble = (
                                '<div style="display:flex;justify-content:flex-end;margin:6px 0">'
                                '<div class="wa-outbound">'
                                f'<div style="font-size:11px;color:#5E8E60;margin-bottom:4px;font-weight:500">Agent · {t}</div>'
                                f'<div style="color:#1C1C1E">{text}</div>'
                                '</div></div>'
                            )
                        st.markdown(bubble, unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="ios-card">', unsafe_allow_html=True)
        st.markdown("**📨 模拟 WhatsApp 消息**")
        st.caption("在接入真实WhatsApp前，用此功能本地测试")

        phone = st.text_input("手机号码", value="8613800138000", help="国际格式，不含+号", key="wa_phone")
        name = st.text_input("买家姓名", value="Demo Buyer", key="wa_name")
        message = st.text_area(
            "消息内容",
            placeholder="Hello, I need pricing for 5x BYD Seal to Mombasa, Kenya. Please quote CIF.",
            height=110,
            key="wa_message",
        )
        wa_submitted = st.button("🚀 发送模拟消息", use_container_width=True, key="wa_send")

        if wa_submitted:
            if not phone.strip() or not message.strip():
                st.error("手机号和消息不能为空")
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
                    with st.spinner("处理中..."):
                        handle_incoming, _ = _load_whatsapp()
                        result = handle_incoming(fake_payload)
                    st.success("✅ 消息处理成功")
                    resp = (
                        result.get("response")
                        or result.get("response_text")
                        or result.get("text")
                        or "（无回复内容）"
                    )
                    st.markdown(f"**Agent 回复：**\n\n{resp}")
                    if result.get("contract_info"):
                        st.json(result["contract_info"])
                    st.rerun()
                except Exception as exc:
                    st.error(f"处理失败：{exc}")

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown('<div class="ios-card">', unsafe_allow_html=True)
        st.markdown("**⚙️ Green API 接入说明**")
        st.markdown(
            "1. 注册 [green-api.com](https://green-api.com/)（免费）\n"
            "2. 连接WhatsApp → 获取 Instance ID + Token\n"
            "3. 设置Webhook: `http://your-server:8000/webhook`\n"
            "4. 在 `.env` 填入配置\n"
            "5. 启动 `python server.py`"
        )
        st.markdown("</div>", unsafe_allow_html=True)


def _render_login_page() -> None:
    """Full-page centered login — no gap, everything in one column."""
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        # ── Brand header ──
        st.markdown(
            '<div style="text-align:center;padding:32px 0 20px;">'
            '<div style="font-size:52px;line-height:1;">🚗</div>'
            '<div style="font-size:22px;font-weight:700;color:#1C1C1E;margin-top:10px;">汽车出口 AI 销售助手</div>'
            '<div style="font-size:13px;color:#8E8E93;margin-top:4px;">'
            'WhatsApp 询盘自动回复 · 车辆知识库问答 · 报价合同生成'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        # ── Feature row ──
        f1, f2, f3, f4 = st.columns(4)
        for col_f, icon, label in [
            (f1, "📱", "WhatsApp\n接入"),
            (f2, "🗄️", "车辆数据库\n问答"),
            (f3, "📋", "报价单\n自动生成"),
            (f4, "🛡️", "三步反思\n防幻觉"),
        ]:
            col_f.markdown(
                f'<div style="background:white;border-radius:10px;padding:10px 6px;'
                f'text-align:center;box-shadow:0 1px 6px rgba(0,0,0,0.07);margin-bottom:4px;">'
                f'<div style="font-size:20px;">{icon}</div>'
                f'<div style="font-size:10px;font-weight:600;color:#1C1C1E;'
                f'margin-top:4px;white-space:pre-line;line-height:1.3;">{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # ── Login card ──
        with st.container(border=True):
            st.markdown(
                '<p style="font-size:16px;font-weight:600;color:#1C1C1E;margin:0 0 4px;">登录系统</p>',
                unsafe_allow_html=True,
            )
            name_input = st.text_input("姓名", placeholder="请输入您的姓名", key="login_name_main",
                                       label_visibility="collapsed")
            st.caption("姓名")
            role_sel = st.selectbox(
                "角色",
                ["👤 销售 (Sales)", "👑 管理员 (Admin)"],
                key="login_role_main",
            )
            if st.button("登 录", use_container_width=True, type="primary", key="login_btn_main"):
                if name_input.strip():
                    st.session_state.user_name = name_input.strip()
                    st.session_state.user_role = "admin" if "Admin" in role_sel else "sales"
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("请输入姓名")
            st.caption("LangGraph · RapidFuzz · ChromaDB · HITL")

        # ── Scenario description ──
        st.markdown(
            '<div style="background:#EFF6FF;border-radius:10px;padding:14px 16px;margin-top:12px;">'
            '<div style="font-size:12px;font-weight:600;color:#1D4ED8;margin-bottom:6px;">💡 典型使用场景</div>'
            '<div style="font-size:12px;color:#374151;line-height:1.8;">'
            '• WhatsApp 客户问「BYD Seal 出口到拉各斯多少钱？」→ 自动回复带 FOB/CIF 报价<br>'
            '• 客户问「你们有没有 7 座 SUV，预算 2 万美元以内？」→ 从车辆库筛选推荐<br>'
            '• 客户确认车型 → 自动生成报价合同 PDF<br>'
            '• 置信度低时暂停，转给人工销售接手'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )


def main() -> None:
    inject_css()
    init_session_state()

    # ── 登录拦截：未登录先显示登录页 ──────────────────────
    if not st.session_state.logged_in:
        _render_login_page()
        return

    render_sidebar()

    # 启动 Telegram bot（每次 session 只启动一次）
    if "telegram_bot_started" not in st.session_state:
        import os as _os
        if _os.getenv("TELEGRAM_BOT_TOKEN"):
            import threading
            from telegram.handler import run_polling
            t = threading.Thread(target=run_polling, daemon=True)
            t.start()
        st.session_state.telegram_bot_started = True

    tabs = ["💬  在线询盘", "📱  WhatsApp", "🤖  Telegram"]
    if st.session_state.get("user_role") == "admin":
        tabs.append("🔍  介入审查")

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


def _render_telegram_tab() -> None:
    """Telegram Bot 实时对话监控面板。"""
    import os as _os
    from telegram.handler import load_telegram_history

    st.markdown(
        '<h2 style="font-size:22px;font-weight:700;color:#1C1C1E;margin-bottom:4px">🤖 Telegram Bot 实时对话</h2>'
        '<p style="color:#8E8E93;font-size:13px;margin-bottom:16px">真实客户通过 Telegram 发来的询盘，由 AI Agent 自动回复</p>',
        unsafe_allow_html=True,
    )

    bot_name = _os.getenv("TELEGRAM_BOT_USERNAME", "ksnzizjwns_bot")
    token = _os.getenv("TELEGRAM_BOT_TOKEN", "")

    if token:
        st.success(f"✅ Bot 运行中 — [t.me/{bot_name}](https://t.me/{bot_name})  ← 点击发消息测试")
    else:
        st.warning("⚠️ TELEGRAM_BOT_TOKEN 未配置")
        return

    col_r, _ = st.columns([1, 4])
    with col_r:
        if st.button("🔄 刷新", use_container_width=True):
            st.rerun()

    records = load_telegram_history()
    if not records:
        st.info("暂无对话记录，去 Telegram 发条消息试试 👆")
        return

    st.markdown(f"**共 {len(records)} 条对话**")

    recent = records[-20:]
    for idx, rec in enumerate(reversed(recent)):  # 用 idx 保证 key 唯一
        ts = rec.get("timestamp", "")[:16]
        username = rec.get("username", "unknown")
        user_msg = rec.get("user_message", "")
        agent_reply = rec.get("agent_reply", "")

        with st.expander(f"[{ts}] @{username}: {user_msg[:50]}", expanded=(idx == 0)):
            col_u, col_a = st.columns(2)
            with col_u:
                st.markdown("**👤 客户消息**")
                st.text_area("", value=user_msg, height=100, disabled=True,
                             key=f"tg_u_{idx}")
            with col_a:
                st.markdown("**🤖 Agent 回复**")
                st.text_area("", value=agent_reply, height=100, disabled=True,
                             key=f"tg_a_{idx}")


def _render_admin_tab() -> None:
    """Admin-only: review all human intervention logs with approve/reject + batch KB sync."""
    from agent.utils.intervention_log import load_interventions, update_intervention_status

    st.markdown(
        '<h2 style="font-size:22px;font-weight:700;color:#1C1C1E;margin-bottom:4px">🔍 介入审查（管理员）</h2>'
        '<p style="color:#8E8E93;font-size:13px;margin-bottom:16px">查看所有人工介入记录，支持批量审核并同步到知识库</p>',
        unsafe_allow_html=True,
    )

    col_r, col_f = st.columns([3, 1])
    with col_f:
        if st.button("🔄 刷新", use_container_width=True):
            st.rerun()

    logs = load_interventions(user_role="admin")
    if not logs:
        st.info("暂无介入记录")
        return

    # Summary stats
    total = len(logs)
    approved_count = sum(1 for e in logs if e.get("approved") is True)
    rejected_count = sum(1 for e in logs if e.get("approved") is False)
    pending_count = total - approved_count - rejected_count
    synced_count = sum(1 for e in logs if e.get("synced_to_kb"))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📋 总计", total)
    c2.metric("✅ 已批准", approved_count)
    c3.metric("❌ 已拒绝", rejected_count)
    c4.metric("📥 同步到KB", synced_count)

    st.markdown("---")

    # Batch sync button: sync all approved but not yet synced entries
    pending_sync = [e for e in logs if e.get("approved") is True and not e.get("synced_to_kb")]
    if pending_sync:
        if st.button(f"📤 批量同步 {len(pending_sync)} 条已批准记录到知识库", type="primary", use_container_width=True):
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
            st.success(f"✅ 已同步 {synced}/{len(pending_sync)} 条记录到 ChromaDB 知识库")
            st.rerun()

    st.markdown(f"**共 {total} 条记录**（待审核 {pending_count} 条）")

    for entry in reversed(logs):
        ts = entry.get("timestamp", "")[:16]
        sid = entry.get("session_id", "")[-8:]
        role = entry.get("user_role", "")
        approved = entry.get("approved")
        synced = entry.get("synced_to_kb", False)

        # Status badge
        if approved is True:
            status_badge = "✅ 已批准" + (" · 📥 已同步KB" if synced else "")
        elif approved is False:
            status_badge = "❌ 已拒绝"
        else:
            status_badge = "⏳ 待审核"

        with st.expander(f"[{ts}] {role} · session ...{sid}  {status_badge}", expanded=(approved is None)):
            col_orig, col_edit = st.columns(2)
            with col_orig:
                st.markdown("**原始回复**")
                st.text_area("", value=entry.get("original_response", ""), height=120, disabled=True, key=f"orig_{sid}_{ts}")
            with col_edit:
                st.markdown("**编辑后回复**")
                st.text_area("", value=entry.get("edited_response", ""), height=120, disabled=True, key=f"edit_{sid}_{ts}")
            if entry.get("reason"):
                st.caption(f"介入原因：{entry['reason']}")
            if entry.get("reviewed_at"):
                st.caption(f"审核时间：{entry['reviewed_at']}")

            # Approve / Reject buttons (only if not yet reviewed)
            if approved is None:
                col_a, col_r2, col_s = st.columns(3)
                with col_a:
                    if st.button("✅ 批准", key=f"approve_{sid}_{ts}", use_container_width=True):
                        update_intervention_status(
                            timestamp=entry.get("timestamp", ""),
                            session_id=entry.get("session_id", ""),
                            approved=True,
                        )
                        st.rerun()
                with col_r2:
                    if st.button("❌ 拒绝", key=f"reject_{sid}_{ts}", use_container_width=True):
                        update_intervention_status(
                            timestamp=entry.get("timestamp", ""),
                            session_id=entry.get("session_id", ""),
                            approved=False,
                        )
                        st.rerun()
                with col_s:
                    if st.button("📤 批准并同步KB", key=f"sync_{sid}_{ts}", use_container_width=True):
                        update_intervention_status(
                            timestamp=entry.get("timestamp", ""),
                            session_id=entry.get("session_id", ""),
                            approved=True,
                            sync_to_kb=True,
                        )
                        st.success("✅ 已同步到知识库")
                        st.rerun()


if __name__ == "__main__":
    main()