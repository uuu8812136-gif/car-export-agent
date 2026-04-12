"""Enhanced price node using RapidFuzz fuzzy matching + SQLite cache."""
from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz import process as fuzz_process

from agent.state import AgentState
from agent.utils.price_cache import PriceCache
from config.settings import PRICES_CSV_PATH, llm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", _safe_str(text).lower()).strip()


import re as _re


def _parse_price_range(text: str) -> tuple[float | None, float | None]:
    """Extract price range from natural language query (USD)."""
    text_lower = _safe_str(text).lower()
    # Pattern: "between X and Y", "X to Y", "X-Y"
    range_pat = _re.compile(
        r"(?:between\s+)?(\d[\d,]*)\s*(?:to|-|and)\s*(\d[\d,]*)\s*(?:usd|dollars?|\$)?",
        _re.IGNORECASE,
    )
    m = range_pat.search(text_lower)
    if m:
        lo = float(m.group(1).replace(",", ""))
        hi = float(m.group(2).replace(",", ""))
        return (min(lo, hi), max(lo, hi))

    # Pattern: "under/below/less than X"
    under_pat = _re.compile(
        r"(?:under|below|less\s+than)\s+(\d[\d,]*)\s*(?:usd|dollars?|\$)?",
        _re.IGNORECASE,
    )
    m = under_pat.search(text_lower)
    if m:
        return (None, float(m.group(1).replace(",", "")))

    # Pattern: "over/above/more than X"
    over_pat = _re.compile(
        r"(?:over|above|more\s+than|at\s+least)\s+(\d[\d,]*)\s*(?:usd|dollars?|\$)?",
        _re.IGNORECASE,
    )
    m = over_pat.search(text_lower)
    if m:
        return (float(m.group(1).replace(",", "")), None)

    return (None, None)


def _filter_by_price_range(
    df: pd.DataFrame, min_usd: float | None, max_usd: float | None
) -> pd.DataFrame:
    """Filter DataFrame by FOB price range."""
    if min_usd is None and max_usd is None:
        return df
    try:
        fob = pd.to_numeric(df["fob_price_usd"], errors="coerce")
        mask = pd.Series([True] * len(df), index=df.index)
        if min_usd is not None:
            mask &= fob >= min_usd
        if max_usd is not None:
            mask &= fob <= max_usd
        filtered = df[mask]
        return filtered if len(filtered) > 0 else df  # fallback to full list
    except Exception:
        return df


def _get_last_user_message(state: AgentState) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, dict):
            if _safe_str(message.get("role", "")).lower() == "user":
                return _safe_str(message.get("content", ""))
        else:
            if _safe_str(getattr(message, "type", "")).lower() == "human":
                return _safe_str(getattr(message, "content", ""))
            if _safe_str(getattr(message, "role", "")).lower() == "user":
                return _safe_str(getattr(message, "content", ""))
    return ""


def _load_prices() -> pd.DataFrame:
    df = pd.read_csv(PRICES_CSV_PATH)
    df.columns = [str(c).strip() for c in df.columns]
    return df.fillna("")


# ---------------------------------------------------------------------------
# Fuzzy matching
# ---------------------------------------------------------------------------

def _fuzzy_match(query: str, df: pd.DataFrame) -> tuple[pd.Series | None, float]:
    """Return (best_row, score 0-100) using RapidFuzz WRatio."""
    norm_query = _normalize(query)
    if not norm_query:
        return None, 0.0

    # Match against model_name
    model_col = "model_name"
    choices = {idx: _normalize(str(v)) for idx, v in df[model_col].items()}
    best = fuzz_process.extractOne(
        norm_query, choices, scorer=fuzz.WRatio, score_cutoff=0,
    )

    best_idx: int | None = None
    best_score: float = 0.0

    if best is not None:
        best_idx = best[2]
        best_score = best[1]

    # Also try brand matching and combine
    if "brand" in df.columns:
        brand_choices = {idx: _normalize(str(v)) for idx, v in df["brand"].items()}
        brand_best = fuzz_process.extractOne(
            norm_query, brand_choices, scorer=fuzz.WRatio, score_cutoff=0,
        )
        if brand_best is not None and brand_best[1] > best_score:
            best_idx = brand_best[2]
            best_score = brand_best[1]

    if best_idx is not None:
        return df.iloc[best_idx], best_score
    return None, 0.0


# ---------------------------------------------------------------------------
# Source tags
# ---------------------------------------------------------------------------

_SOURCE_TAG = (
    "\n\n---\n"
    "📊 *Source: Verified CSV Database | Product ID: {pid} | "
    "Updated: {date} | Confidence: {score:.0f}%*"
)

_NOT_FOUND_TAG = (
    "\n\n---\n"
    "🚫 *No match found. Confidence: {score:.0f}% (threshold: 70%). "
    "Routing to human agent.*"
)

_WARNING = "⚠️ Match confidence: {score:.0f}% — please verify model name"


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------

def query_price(state: AgentState) -> dict[str, Any]:
    user_message = _get_last_user_message(state)
    agent_steps = list(state.get("agent_steps", []))
    cache = PriceCache()

    # Check cache first
    cache_key = _normalize(user_message)
    cached = cache.get(cache_key)
    if cached is not None:
        agent_steps.append("Price lookup: served from cache")
        return {
            "price_result": cached["price_result"],
            "draft_answer": cached["draft_answer"],
            "price_confidence_score": cached["score"],
            "needs_human_review": cached.get("needs_human_review", False),
            "agent_steps": agent_steps,
        }

    df = _load_prices()

    # --- Price range filtering ---
    min_usd, max_usd = _parse_price_range(user_message)
    if min_usd is not None or max_usd is not None:
        df_filtered = _filter_by_price_range(df, min_usd, max_usd)
        agent_steps.append(
            f"Price range filter: USD {min_usd or 'any'}–{max_usd or 'any'} → {len(df_filtered)}/{len(df)} models"
        )
    else:
        df_filtered = df
    row, score = _fuzzy_match(user_message, df_filtered)
    confidence = score / 100.0  # 0-1 scale for state

    # --- Secondary retrieval: if initial score 70-84%, try alternative matching ---
    if row is not None and 70 <= score < 85:
        # Try matching on full name (brand + model)
        alt_query = _normalize(user_message)
        full_names = {
            idx: _normalize(f"{r.get('brand','')} {r.get('model_name','')} {r.get('variant','')}")
            for idx, r in df_filtered.iterrows()
        }
        alt_best = fuzz_process.extractOne(alt_query, full_names, scorer=fuzz.token_sort_ratio, score_cutoff=0)
        if alt_best is not None and alt_best[1] > score:
            row = df_filtered.iloc[alt_best[2]]
            score = alt_best[1]
            agent_steps.append(f"Secondary retrieval: improved score to {score:.0f}%")

    # --- score >= 85: auto-answer ---
    if row is not None and score >= 85:
        pid = row.get('product_id', f'PROD-{int(getattr(row, "name", 0)):03d}')
        price_result = {
            "match_found": True,
            "count": 1,
            "cars": [row.to_dict()],
        }
        answer_lines = [
            f"**{row.get('brand', '')} {row.get('model_name', '')}**",
            f"- Variant: {row.get('variant', 'N/A')}",
            f"- FOB Price: USD {row.get('fob_price_usd', 'N/A')}",
            f"- CIF Price: USD {row.get('cif_price_usd', 'N/A')}",
            f"- Year: {row.get('year', 'N/A')}",
        ]
        draft_answer = "\n".join(answer_lines) + _SOURCE_TAG.format(
            pid=pid, date=row.get("update_time", row.get("year", "N/A")), score=score,
        )
        agent_steps.append(f"Price lookup: matched '{row.get('model_name', '')}' (confidence {score:.0f}%)")

        cache.set(cache_key, {
            "price_result": price_result,
            "draft_answer": draft_answer,
            "score": confidence,
            "needs_human_review": False,
        })
        return {
            "price_result": price_result,
            "draft_answer": draft_answer,
            "price_confidence_score": confidence,
            "needs_human_review": False,
            "agent_steps": agent_steps,
        }

    # --- score 70-84: answer with warning ---
    if row is not None and score >= 70:
        pid = row.get('product_id', f'PROD-{int(getattr(row, "name", 0)):03d}')
        price_result = {
            "match_found": True,
            "count": 1,
            "cars": [row.to_dict()],
        }
        answer_lines = [
            f"**{row.get('brand', '')} {row.get('model_name', '')}**",
            f"- Variant: {row.get('variant', 'N/A')}",
            f"- FOB Price: USD {row.get('fob_price_usd', 'N/A')}",
            f"- CIF Price: USD {row.get('cif_price_usd', 'N/A')}",
            f"- Year: {row.get('year', 'N/A')}",
            "",
            _WARNING.format(score=score),
        ]
        draft_answer = "\n".join(answer_lines) + _SOURCE_TAG.format(
            pid=pid, date=row.get("update_time", row.get("year", "N/A")), score=score,
        )
        agent_steps.append(f"Price lookup: fuzzy match '{row.get('model_name', '')}' (confidence {score:.0f}% — warning)")

        cache.set(cache_key, {
            "price_result": price_result,
            "draft_answer": draft_answer,
            "score": confidence,
            "needs_human_review": False,
        })
        return {
            "price_result": price_result,
            "draft_answer": draft_answer,
            "price_confidence_score": confidence,
            "needs_human_review": False,
            "agent_steps": agent_steps,
        }

    # --- score < 70: hard reject ---
    model_list = "\n".join(
        f"- **{r.get('brand', '')} {r.get('model_name', '')}** "
        f"(FOB: USD {r.get('fob_price_usd', 'N/A')})"
        for r in df.to_dict(orient="records")
    )
    draft_answer = (
        "I'm sorry, but the model you inquired about is **not found** "
        "in our verified price catalog.\n\n"
        "**Models currently available for export:**\n"
        + model_list
        + _NOT_FOUND_TAG.format(score=score)
    )
    price_result = {
        "match_found": False,
        "count": len(df),
        "cars": df.to_dict(orient="records"),
    }
    agent_steps.append(f"Price lookup: no match (confidence {score:.0f}%) — routing to human")

    return {
        "price_result": price_result,
        "draft_answer": draft_answer,
        "price_confidence_score": confidence,
        "needs_human_review": True,
        "agent_steps": agent_steps,
    }
