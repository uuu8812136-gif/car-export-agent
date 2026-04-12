"""
generate_hallucination_guards.py
Use GPT to generate 2026-style anti-hallucination nodes for car export agent.
Run: python generate_hallucination_guards.py
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
    "You are an expert Python developer specializing in LangChain, LangGraph, and production AI systems. "
    "Generate complete, production-ready Python code with NO placeholders, NO TODOs. "
    "All imports must be real and correct. Use type hints. All functions fully implemented."
)

CTX = (
    "EXISTING PROJECT CONTEXT:\n"
    "- AgentState is a TypedDict in agent/state.py with these keys:\n"
    "  messages(list), intent(str), retrieved_context(str), price_result(dict),\n"
    "  draft_answer(str), reflection_score(int), reflection_count(int), needs_retry(bool),\n"
    "  contract_data(dict), contract_path(str), agent_steps(list), hallucination_status(str)\n"
    "- hallucination_status is a NEW field, default empty string, set by hallucination_grader\n"
    "- config/settings.py exports: llm (ChatOpenAI instance with base_url=https://hk.ticketpro.cc/v1)\n"
    "- All nodes receive state: AgentState and return dict with updated state fields\n"
    "- LangChain version: 0.3.x, LangGraph: 0.2.x\n"
    "- Use: from langchain_core.pydantic_v1 import BaseModel, Field  (NOT from pydantic)\n"
)

DOC_GRADER_PROMPT = (
    CTX
    + "\nGenerate agent/nodes/doc_grader.py — CRAG-style document relevance grader.\n\n"
    + "PURPOSE: Filter irrelevant RAG chunks BEFORE passing to the generator.\n"
    + "This prevents hallucinations caused by irrelevant context being fed to LLM.\n\n"
    + "IMPORTS needed:\n"
    + "  from __future__ import annotations\n"
    + "  from typing import Any\n"
    + "  from langchain_core.pydantic_v1 import BaseModel, Field\n"
    + "  from langchain_core.messages import HumanMessage\n"
    + "  from agent.state import AgentState\n"
    + "  from config.settings import llm\n\n"
    + "Define Pydantic model:\n"
    + "  class GradeDocument(BaseModel):\n"
    + "      binary_score: str = Field(description='yes or no')\n\n"
    + "Create grader_llm = llm.with_structured_output(GradeDocument)\n\n"
    + "def grade_documents(state: AgentState) -> dict[str, Any]:\n"
    + "  Extract user question from last message in state['messages']\n"
    + "  Get retrieved_context string from state\n"
    + "  Split retrieved_context by double newline to get individual chunks\n"
    + "  For each chunk: call grader_llm with a prompt asking if this chunk is relevant to the question\n"
    + "  Keep only chunks where binary_score == 'yes'\n"
    + "  If no chunks pass: set retrieved_context to empty string and append to agent_steps:\n"
    + "    'Document grader: No relevant chunks found — knowledge base cannot answer this query'\n"
    + "  If chunks pass: join them back, append to agent_steps:\n"
    + "    f'Document grader: {kept}/{total} chunks passed relevance check'\n"
    + "  Return dict with: retrieved_context (filtered), agent_steps (updated)\n"
    + "  Wrap in try/except — if error, return state unchanged with error note in steps\n"
)

HALLUCINATION_GRADER_PROMPT = (
    CTX
    + "\nGenerate agent/nodes/hallucination_grader.py — LLM-as-Judge binary hallucination checker.\n\n"
    + "PURPOSE: After generation, verify the draft_answer is grounded AND answers the question.\n"
    + "This replaces the old Self-Reflection (0-10 score) with 2026-standard binary verification.\n\n"
    + "IMPORTS needed:\n"
    + "  from __future__ import annotations\n"
    + "  from typing import Any\n"
    + "  from langchain_core.pydantic_v1 import BaseModel, Field\n"
    + "  from langchain_core.messages import HumanMessage\n"
    + "  from agent.state import AgentState\n"
    + "  from config.settings import llm\n\n"
    + "Define TWO Pydantic models:\n"
    + "  class GroundednessCheck(BaseModel):\n"
    + "      grounded: str = Field(description='yes if answer is supported by context/price data, no if hallucinated')\n"
    + "      reason: str = Field(description='brief reason')\n\n"
    + "  class AnswerQualityCheck(BaseModel):\n"
    + "      answers_question: str = Field(description='yes if the response actually addresses what was asked, no otherwise')\n"
    + "      reason: str = Field(description='brief reason')\n\n"
    + "Create:\n"
    + "  groundedness_llm = llm.with_structured_output(GroundednessCheck)\n"
    + "  quality_llm = llm.with_structured_output(AnswerQualityCheck)\n\n"
    + "def check_hallucination(state: AgentState) -> dict[str, Any]:\n"
    + "  current_retry = int(state.get('reflection_count', 0) or 0)\n"
    + "  current_steps = list(state.get('agent_steps', []) or [])\n"
    + "  draft_answer = str(state.get('draft_answer', '') or '')\n"
    + "  retrieved_context = str(state.get('retrieved_context', '') or '')\n"
    + "  price_result_str = str(state.get('price_result', {}) or {})\n"
    + "  question = last user message from state['messages']\n\n"
    + "  If reflection_count >= 2:\n"
    + "    append 'Hallucination grader: max retries reached — passing through' to steps\n"
    + "    return needs_retry=False, hallucination_status='reviewed', reflection_count, agent_steps\n\n"
    + "  Build context string: retrieved_context + price_result_str combined\n\n"
    + "  Call groundedness_llm with prompt:\n"
    + "    'Does this answer rely only on the provided context/price data? Answer yes or no.'\n"
    + "    Include: context, draft_answer\n\n"
    + "  Call quality_llm with prompt:\n"
    + "    'Does this answer actually address the user question? Answer yes or no.'\n"
    + "    Include: question, draft_answer\n\n"
    + "  Both must be 'yes' to pass. If either is 'no':\n"
    + "    needs_retry = True, increment reflection_count\n"
    + "    hallucination_status = 'flagged'\n"
    + "    append 'Hallucination grader: FLAGGED — {grounded.reason} | {quality.reason}' to steps\n\n"
    + "  If both pass:\n"
    + "    needs_retry = False\n"
    + "    hallucination_status = 'verified'\n"
    + "    append 'Hallucination grader: VERIFIED — grounded and answers question' to steps\n\n"
    + "  Wrap in try/except — on error default needs_retry=False, hallucination_status='reviewed'\n"
    + "  Return dict with: needs_retry, hallucination_status, reflection_count, agent_steps\n"
)

GRAPH_PROMPT = (
    CTX
    + "\nGenerate agent/graph.py — LangGraph state machine with 2026 Defense-in-Depth anti-hallucination.\n\n"
    + "IMPORTS:\n"
    + "  from __future__ import annotations\n"
    + "  from typing import Any\n"
    + "  from langchain_core.messages import AIMessage, HumanMessage, BaseMessage\n"
    + "  from langgraph.graph import END, START, StateGraph\n"
    + "  from agent.state import AgentState\n"
    + "  Import all node functions\n\n"
    + "NODE FUNCTIONS to import:\n"
    + "  detect_intent from agent.nodes.intent_detector\n"
    + "  query_price from agent.nodes.price_node\n"
    + "  retrieve_and_answer from agent.nodes.rag_node\n"
    + "  generate_contract from agent.nodes.contract_node\n"
    + "  general_chat from agent.nodes.general_chat_node\n"
    + "  grade_documents from agent.nodes.doc_grader\n"
    + "  check_hallucination from agent.nodes.hallucination_grader\n\n"
    + "GRAPH STRUCTURE:\n"
    + "  def respond(state) -> dict: append draft_answer as AIMessage to messages, return updated state\n\n"
    + "  def intent_router(state) -> str:\n"
    + "    route based on state['intent']:\n"
    + "      price_query -> 'price_node'\n"
    + "      product_info -> 'rag_node'\n"
    + "      contract_request -> 'contract_node'\n"
    + "      default -> 'general_chat_node'\n\n"
    + "  def post_rag_router(state) -> str:\n"
    + "    After RAG: always route to 'doc_grader' (CRAG filtering step)\n"
    + "    Return 'doc_grader'\n\n"
    + "  def hallucination_retry_router(state) -> str:\n"
    + "    If state['needs_retry'] is True: return based on original intent (use intent_router logic)\n"
    + "    Else: return 'respond'\n\n"
    + "  Build StateGraph(AgentState):\n"
    + "    Nodes: intent_detector, price_node, rag_node, contract_node, general_chat_node,\n"
    + "           doc_grader, hallucination_grader, respond\n\n"
    + "    Edges:\n"
    + "    START -> intent_detector\n"
    + "    intent_detector -> conditional(intent_router)\n"
    + "    price_node -> hallucination_grader  (price uses DB, no doc grading needed)\n"
    + "    contract_node -> hallucination_grader\n"
    + "    general_chat_node -> hallucination_grader\n"
    + "    rag_node -> doc_grader  (RAG MUST go through document grader first)\n"
    + "    doc_grader -> hallucination_grader\n"
    + "    hallucination_grader -> conditional(hallucination_retry_router)\n"
    + "    respond -> END\n\n"
    + "  app = graph.compile()\n\n"
    + "  def run_agent(user_message: str, chat_history: list) -> tuple[str, list[str], dict]:\n"
    + "    Build initial_state with messages containing chat_history + new HumanMessage\n"
    + "    Initialize all state fields to defaults (intent='', retrieved_context='', etc.)\n"
    + "    Include hallucination_status='' in initial state\n"
    + "    result = app.invoke(initial_state)\n"
    + "    Extract response_text from last AIMessage in result['messages']\n"
    + "    Extract agent_steps from result['agent_steps']\n"
    + "    Build contract_info dict from result['contract_path'] and result['contract_data']\n"
    + "    Also include hallucination_status in contract_info dict\n"
    + "    Return (response_text, agent_steps, contract_info)\n"
    + "    Wrap entire function in try/except\n"
)

FILES = [
    {"path": "agent/nodes/doc_grader.py", "prompt": DOC_GRADER_PROMPT},
    {"path": "agent/nodes/hallucination_grader.py", "prompt": HALLUCINATION_GRADER_PROMPT},
    {"path": "agent/graph.py", "prompt": GRAPH_PROMPT},
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
    print("=" * 55)
    print("Anti-Hallucination Guards Generator (2026 stack)")
    print("=" * 55)
    ok = sum(generate_file(f) for f in FILES)
    print(f"\nDone: {ok}/{len(FILES)}")
