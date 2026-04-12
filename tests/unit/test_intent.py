"""Unit tests for intent detection prompt rules — no API calls."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def test_intent_detection_prompt_exists():
    from config.prompts import INTENT_DETECTION_PROMPT
    assert "price_query" in INTENT_DETECTION_PROMPT
    assert "product_info" in INTENT_DETECTION_PROMPT
    assert "contract_request" in INTENT_DETECTION_PROMPT
    assert "under" in INTENT_DETECTION_PROMPT.lower()  # 确认有预算筛选规则


def test_intent_detection_prompt_has_examples():
    from config.prompts import INTENT_DETECTION_PROMPT
    assert "20000" in INTENT_DETECTION_PROMPT or "budget" in INTENT_DETECTION_PROMPT.lower()


def test_reflection_log_schema():
    """Verify Pydantic models have required fields."""
    from agent.nodes.reflection_pipeline import FactCheckResult, ComplianceCheckResult
    fields = FactCheckResult.model_fields
    assert "error_type" in fields
    assert "correction_plan" in fields
    assert "trigger_condition" in fields


def test_agent_state_has_required_fields():
    from agent.state import get_default_state
    state = get_default_state()
    required = ["messages", "intent", "draft_answer", "reflection_log",
                "price_confidence_score", "needs_human_review", "session_id"]
    for field in required:
        assert field in state, f"Missing state field: {field}"
