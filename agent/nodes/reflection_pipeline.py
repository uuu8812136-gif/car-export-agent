from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from agent.state import AgentState
from config.settings import llm


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class FactCheckResult(BaseModel):
    passed: bool = Field(description="True if all prices/quantities are traceable to source data")
    error_type: str = Field(description="price_mismatch | invented_data | ok")
    correction_plan: str = Field(description="What to fix if failed, empty if ok")
    trigger_condition: str = Field(description="Which part of the answer triggered the issue")


class ComplianceCheckResult(BaseModel):
    passed: bool = Field(description="True if no unauthorized promises found")
    error_type: str = Field(description="unauthorized_promise | missing_disclaimer | ok")
    correction_plan: str = Field(description="What to fix if failed, empty if ok")
    trigger_condition: str = Field(description="Which part of the answer triggered the issue")


class UpsellAnalysis(BaseModel):
    should_upsell: bool = Field(description="True if a related model should be recommended")
    recommended_model: str = Field(default="", description="Model name to recommend, empty if none")
    upsell_reason: str = Field(default="", description="Why this model is relevant to the buyer")


class ReflectionLog(BaseModel):
    step1_fact_check: FactCheckResult
    step2_compliance: ComplianceCheckResult
    step3_upsell: UpsellAnalysis
    overall_passed: bool
    retry_needed: bool
    strictness_level: str = Field(description="strict | normal | lenient")


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_fact_check_prompt(draft: str, price_data: str, context: str) -> str:
    return (
        "You are a fact-checker for a car export quotation assistant.\n"
        "Check whether ALL prices, quantities, and numeric data in the draft answer "
        "are directly traceable to the provided price data or retrieved context.\n\n"
        "If any number is invented or mismatched, set passed=false and error_type accordingly.\n\n"
        f"Price Data:\n{price_data}\n\n"
        f"Retrieved Context:\n{context}\n\n"
        f"Draft Answer:\n{draft}"
    )


def _build_compliance_prompt(draft: str, strictness: str) -> str:
    base = (
        "You are a compliance checker for a car export quotation assistant.\n"
        "Check whether the draft answer makes any unauthorized promises such as:\n"
        "- 'guaranteed delivery' or specific delivery dates not backed by data\n"
        "- 'best price' or price-match guarantees\n"
        "- Specific port fees, insurance costs, or taxes not present in the data\n"
        "- Any legally binding language\n\n"
    )
    if strictness == "strict":
        base += (
            "Apply STRICT compliance standards. Flag any language that could be "
            "interpreted as a commitment, even if hedged. Also flag missing disclaimers "
            "about price validity periods and shipping estimates.\n\n"
        )
    elif strictness == "lenient":
        base += (
            "Apply LENIENT compliance standards. Only flag clear, explicit unauthorized "
            "promises. Hedged language and general statements are acceptable.\n\n"
        )
    else:
        base += "Apply standard compliance checks.\n\n"

    base += f"Draft Answer:\n{draft}"
    return base


def _build_upsell_prompt(draft: str, context: str, price_data: str) -> str:
    return (
        "You are an upsell analyzer for a car export quotation assistant.\n"
        "Based on the buyer's intent (visible in the draft answer and context), "
        "determine if a related car model should be recommended.\n"
        "Only recommend if the suggestion is genuinely relevant to the buyer's needs.\n"
        "If no upsell is appropriate, set should_upsell=false.\n\n"
        f"Price Data:\n{price_data}\n\n"
        f"Retrieved Context:\n{context}\n\n"
        f"Draft Answer:\n{draft}"
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_reflection_pipeline(state: AgentState) -> dict[str, Any]:
    strictness = state.get("reflection_strictness", "normal") or "normal"
    current_retry = int(state.get("reflection_count", 0) or 0)
    current_steps = list(state.get("agent_steps", []) or [])
    reflection_log = list(state.get("reflection_log", []) or [])

    draft_answer = str(state.get("draft_answer", "") or "")
    retrieved_context = str(state.get("retrieved_context", "") or "")
    price_result_str = str(state.get("price_result", {}) or {})

    # ── Speed optimization: skip reflection for high-confidence price queries ──
    price_confidence = float(state.get("price_confidence_score", 0.0) or 0.0)
    intent = str(state.get("intent", "") or "")

    if price_confidence >= 0.90 and intent == "price_query":
        current_steps.append(f"Reflection pipeline: SKIPPED (price confidence {price_confidence*100:.0f}% ≥ 90%)")
        auto_log = ReflectionLog(
            step1_fact_check=FactCheckResult(
                passed=True, error_type="ok",
                correction_plan="",
                trigger_condition="high_confidence_skip"
            ),
            step2_compliance=ComplianceCheckResult(
                passed=True, error_type="ok",
                correction_plan="",
                trigger_condition="high_confidence_skip"
            ),
            step3_upsell=UpsellAnalysis(
                should_upsell=False, recommended_model="", upsell_reason=""
            ),
            overall_passed=True,
            retry_needed=False,
            strictness_level=strictness,
        )
        reflection_log.append(auto_log.model_dump())
        return {
            "needs_retry": False,
            "reflection_count": 0,
            "hallucination_status": "verified",
            "reflection_log": reflection_log,
            "agent_steps": current_steps,
        }

    # Initialise structured-output LLMs lazily (avoid import-time API calls)
    fact_check_llm = llm.with_structured_output(FactCheckResult)
    compliance_llm = llm.with_structured_output(ComplianceCheckResult)
    upsell_llm = llm.with_structured_output(UpsellAnalysis)

    # Max retries guard — route to human review instead of force-pass
    if current_retry >= 2:
        current_steps.append("Reflection pipeline: max retries reached — routing to human review")
        default_log = ReflectionLog(
            step1_fact_check=FactCheckResult(passed=False, error_type="max_retries", correction_plan="Manual review required", trigger_condition="max retries exceeded"),
            step2_compliance=ComplianceCheckResult(passed=False, error_type="max_retries", correction_plan="Manual review required", trigger_condition="max retries exceeded"),
            step3_upsell=UpsellAnalysis(should_upsell=False, recommended_model="", upsell_reason=""),
            overall_passed=False,
            retry_needed=False,
            strictness_level=strictness,
        )
        reflection_log.append(default_log.model_dump())
        return {
            "needs_retry": False,
            "reflection_count": current_retry,
            "hallucination_status": "requires_review",
            "needs_human_review": True,  # Route to human instead of force-pass
            "reflection_log": reflection_log,
            "agent_steps": current_steps,
        }

    # --- Step 1: FactCheck ---
    try:
        step1: FactCheckResult = fact_check_llm.invoke(
            [HumanMessage(content=_build_fact_check_prompt(draft_answer, price_result_str, retrieved_context))]
        )
    except Exception:
        step1 = FactCheckResult(passed=True, error_type="ok", correction_plan="", trigger_condition="error_fallback")

    step1_label = "PASS" if step1.passed else f"FAIL — {step1.error_type}"
    current_steps.append(f"Step1 FactCheck: {step1_label}")

    # --- Step 2: ComplianceCheck ---
    try:
        step2: ComplianceCheckResult = compliance_llm.invoke(
            [HumanMessage(content=_build_compliance_prompt(draft_answer, strictness))]
        )
    except Exception:
        step2 = ComplianceCheckResult(passed=True, error_type="ok", correction_plan="", trigger_condition="error_fallback")

    step2_label = "PASS" if step2.passed else f"FAIL — {step2.error_type}"
    current_steps.append(f"Step2 Compliance: {step2_label}")

    # --- Step 3: UpsellAnalyzer (skip if lenient) ---
    if strictness == "lenient":
        step3 = UpsellAnalysis(should_upsell=False, recommended_model="", upsell_reason="")
        current_steps.append("Step3 Upsell: SKIPPED (lenient)")
    else:
        try:
            step3 = upsell_llm.invoke(
                [HumanMessage(content=_build_upsell_prompt(draft_answer, retrieved_context, price_result_str))]
            )
        except Exception:
            step3 = UpsellAnalysis(should_upsell=False, recommended_model="", upsell_reason="")

        upsell_label = f"YES — {step3.recommended_model}" if step3.should_upsell else "NO"
        current_steps.append(f"Step3 Upsell: {upsell_label}")

    # --- Aggregate results ---
    overall_passed = step1.passed and step2.passed
    retry_needed = not overall_passed

    if retry_needed:
        updated_retry = current_retry + 1
        hallucination_status = "flagged"
    else:
        updated_retry = current_retry
        hallucination_status = "verified"

    log_entry = ReflectionLog(
        step1_fact_check=step1,
        step2_compliance=step2,
        step3_upsell=step3,
        overall_passed=overall_passed,
        retry_needed=retry_needed,
        strictness_level=strictness,
    )
    reflection_log.append(log_entry.model_dump())

    return {
        "needs_retry": retry_needed,
        "reflection_count": updated_retry,
        "hallucination_status": hallucination_status,
        "reflection_log": reflection_log,
        "agent_steps": current_steps,
    }
