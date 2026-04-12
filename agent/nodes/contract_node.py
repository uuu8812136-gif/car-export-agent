from __future__ import annotations

import csv
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from config.settings import CONTRACTS_OUTPUT_DIR, CONTRACTS_TEMPLATE_PATH, PRICES_CSV_PATH, llm
from config.prompts import CONTRACT_EXTRACT_PROMPT
from agent.state import AgentState


def _messages_to_text(messages: List[BaseMessage]) -> str:
    lines: List[str] = []
    for msg in messages:
        role = "user"
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        elif isinstance(msg, SystemMessage):
            role = "system"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


def _extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    raise ValueError("Failed to parse JSON object from LLM response.")


def _normalize_key(value: str) -> str:
    return value.strip().lower().replace("-", " ").replace("_", " ")


def _safe_int(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if value is None:
        raise ValueError("Quantity is missing.")
    cleaned = str(value).strip().replace(",", "")
    if not cleaned:
        raise ValueError("Quantity is empty.")
    return int(float(cleaned))


def _lookup_price(car_model: str) -> float:
    csv_path = Path(PRICES_CSV_PATH)
    if not csv_path.exists():
        raise FileNotFoundError(f"Price file not found: {csv_path}")

    target = _normalize_key(car_model)

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("prices.csv is missing headers.")

        normalized_headers = {_normalize_key(h): h for h in reader.fieldnames}
        model_col = None
        price_col = None

        for candidate in ["car model", "model"]:
            if candidate in normalized_headers:
                model_col = normalized_headers[candidate]
                break

        for candidate in ["unit price", "price", "usd price", "fob price", "cif price"]:
            if candidate in normalized_headers:
                price_col = normalized_headers[candidate]
                break

        if model_col is None or price_col is None:
            raise ValueError("prices.csv must contain model and price columns.")

        for row in reader:
            row_model = str(row.get(model_col, "")).strip()
            if _normalize_key(row_model) == target:
                raw_price = str(row.get(price_col, "")).strip().replace(",", "").replace("$", "")
                if not raw_price:
                    raise ValueError(f"Price is empty for model: {car_model}")
                return float(raw_price)

    raise ValueError(f"Price not found for car model: {car_model}")


def _generate_quote_number() -> str:
    date_str = datetime.now().strftime("%Y%m%d")
    suffix = random.randint(100, 999)
    return f"QT-{date_str}-{suffix}"


def _read_template() -> str:
    template_path = Path(CONTRACTS_TEMPLATE_PATH)
    if not template_path.exists():
        raise FileNotFoundError(f"Contract template not found: {template_path}")
    return template_path.read_text(encoding="utf-8")


def _ensure_output_dir() -> Path:
    output_dir = Path(CONTRACTS_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _append_step(state: AgentState, step: str) -> List[str]:
    existing = list(state.get("agent_steps", []))
    existing.append(step)
    return existing


def generate_contract(state: AgentState) -> dict:
    messages: List[BaseMessage] = list(state.get("messages", []))
    if not messages:
        raise ValueError("No conversation messages found in state.")

    # llm imported from config.settings
    conversation_text = _messages_to_text(messages)

    extraction_prompt = (
        f"{CONTRACT_EXTRACT_PROMPT}\n\n"
        "Extract the following fields and return ONLY valid JSON:\n"
        "buyer_company, buyer_country, car_model, car_brand, quantity, destination_port\n\n"
        f"Conversation:\n{conversation_text}"
    )

    llm_response = llm.invoke(extraction_prompt)
    response_text = llm_response.content if hasattr(llm_response, "content") else str(llm_response)
    extracted = _extract_json_object(response_text)

    required_fields = [
        "buyer_company",
        "buyer_country",
        "car_model",
        "car_brand",
        "quantity",
        "destination_port",
    ]
    missing = [field for field in required_fields if extracted.get(field) in (None, "", [])]
    if missing:
        # Graceful degradation: ask for missing info instead of raising
        field_labels = {
            "buyer_company": "buyer company name",
            "buyer_country": "buyer country",
            "car_model": "vehicle model",
            "car_brand": "vehicle brand",
            "quantity": "quantity",
            "destination_port": "destination port",
        }
        missing_friendly = ", ".join(field_labels.get(f, f) for f in missing)
        agent_steps = list(state.get("agent_steps", []))
        agent_steps.append(f"Contract: missing fields — {missing_friendly}")
        draft_answer = (
            "To generate your quotation, I need a few more details:\n\n"
            + "\n".join(f"- **{field_labels.get(f, f).title()}**" for f in missing)
            + "\n\nPlease provide these details and I'll prepare your quotation immediately.\n\n"
            "---\n📝 *Contract generation requires complete buyer and vehicle information.*"
        )
        return {"draft_answer": draft_answer, "agent_steps": agent_steps}

    buyer_company = str(extracted["buyer_company"]).strip()
    buyer_country = str(extracted["buyer_country"]).strip()
    car_model = str(extracted["car_model"]).strip()
    car_brand = str(extracted["car_brand"]).strip()
    quantity = _safe_int(extracted["quantity"])
    destination_port = str(extracted["destination_port"]).strip()

    unit_price = _lookup_price(car_model)
    total_amount = unit_price * quantity
    quote_number = _generate_quote_number()

    template = _read_template()
    filled_contract = template.format(
        quote_number=quote_number,
        date=datetime.now().strftime("%Y-%m-%d"),
        buyer_company=buyer_company,
        buyer_country=buyer_country,
        car_model=car_model,
        car_brand=car_brand,
        quantity=quantity,
        destination_port=destination_port,
        unit_price=f"{unit_price:,.2f}",
        total_amount=f"{total_amount:,.2f}",
        currency="USD",
    )

    output_dir = _ensure_output_dir()
    contract_path = output_dir / f"{quote_number}.md"
    contract_path.write_text(filled_contract, encoding="utf-8")

    draft_answer = (
        f"Quotation **{quote_number}** has been generated successfully.\n\n"
        f"- **Buyer:** {buyer_company} ({buyer_country})\n"
        f"- **Vehicle:** {car_brand} {car_model} × {quantity} units\n"
        f"- **Total:** USD {total_amount:,.0f}\n"
        f"- **Destination:** {destination_port}\n\n"
        "You can download the full quotation document from the sidebar.\n\n"
        "---\n"
        "📝 *Contract generated using verified price database. "
        "All prices cross-referenced with CSV data — no AI-estimated figures.*"
    )
    agent_steps = _append_step(
        state,
        f"Generated contract {quote_number} for {buyer_company}, model {car_model}, quantity {quantity}, total USD {total_amount:,.2f}",
    )

    return {
        "quote_number": quote_number,
        "contract_path": str(contract_path),
        "draft_answer": draft_answer,
        "contract_data": {
            "buyer_company": buyer_company,
            "buyer_country": buyer_country,
            "car_model": car_model,
            "car_brand": car_brand,
            "quantity": quantity,
            "destination_port": destination_port,
            "unit_price": unit_price,
            "total_amount": total_amount,
            "currency": "USD",
        },
        "agent_steps": agent_steps,
    }