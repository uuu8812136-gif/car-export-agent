"""
Reusable HTML component generators for the Car Export AI Sales Assistant.

Every function returns an HTML string intended for use with
``st.markdown(html, unsafe_allow_html=True)``.
"""

from __future__ import annotations

from ui.theme import Colors, Radius, Shadow, Spacing


# ---------------------------------------------------------------------------
# Metric Card
# ---------------------------------------------------------------------------
def render_metric_card(label: str, value: str | int, icon: str, color: str | None = None) -> str:
    """Compact stat card with icon, value, and label."""
    c = color or Colors.accent
    return (
        f'<div style="'
        f'background: {Colors.card}; border-radius: {Radius.md}px; '
        f'padding: {Spacing.lg}px {Spacing.xl}px; '
        f'border: 1px solid {Colors.border_light}; '
        f'box-shadow: {Shadow.sm}; '
        f'display: flex; align-items: center; gap: {Spacing.md}px;'
        f'">'
        f'<div style="'
        f'width: 40px; height: 40px; border-radius: {Radius.sm}px; '
        f'background: {c}14; display: flex; align-items: center; '
        f'justify-content: center; font-size: 18px; flex-shrink: 0;'
        f'">{icon}</div>'
        f'<div>'
        f'<div style="font-size: 22px; font-weight: 700; color: {Colors.text}; line-height: 1.2;">{value}</div>'
        f'<div style="font-size: 12px; color: {Colors.text_secondary}; margin-top: 2px;">{label}</div>'
        f'</div>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Status Badge
# ---------------------------------------------------------------------------
_BADGE_VARIANTS: dict[str, tuple[str, str]] = {
    "verified":  (Colors.success, "#ECFDF5"),
    "flagged":   (Colors.danger,  "#FEF2F2"),
    "reviewed":  (Colors.warning, "#FFFBEB"),
    "idle":      (Colors.text_secondary, Colors.border_light),
}


def render_status_badge(text: str, variant: str = "idle") -> str:
    """Colored pill badge."""
    fg, bg = _BADGE_VARIANTS.get(variant, _BADGE_VARIANTS["idle"])
    return (
        f'<span style="'
        f'display: inline-flex; align-items: center; gap: 6px; '
        f'background: {bg}; color: {fg}; '
        f'border-radius: 20px; padding: 5px 14px; '
        f'font-size: 13px; font-weight: 600; line-height: 1;'
        f'">{text}</span>'
    )


# ---------------------------------------------------------------------------
# Progress Steps  (pipeline visualization)
# ---------------------------------------------------------------------------
def render_progress_steps(steps: list[dict[str, str]]) -> str:
    """
    Horizontal pipeline steps.

    *steps* is a list of ``{"label": "...", "status": "done|active|pending"}``.
    """
    items_html = ""
    for i, step in enumerate(steps):
        label = step.get("label", f"Step {i+1}")
        status = step.get("status", "pending")

        if status == "done":
            dot_bg = Colors.success
            dot_icon = "&#10003;"  # checkmark
            label_color = Colors.text
        elif status == "active":
            dot_bg = Colors.accent
            dot_icon = "&#9679;"  # filled circle
            label_color = Colors.accent
        else:
            dot_bg = Colors.border
            dot_icon = ""
            label_color = Colors.text_tertiary

        connector = ""
        if i < len(steps) - 1:
            next_done = steps[i + 1].get("status") == "done"
            line_color = Colors.success if (status == "done" and next_done) else Colors.border
            connector = (
                f'<div style="flex: 1; height: 2px; background: {line_color}; '
                f'margin: 0 4px; align-self: center;"></div>'
            )

        items_html += (
            f'<div style="display: flex; flex-direction: column; align-items: center; min-width: 56px;">'
            f'<div style="'
            f'width: 24px; height: 24px; border-radius: 50%; '
            f'background: {dot_bg}; color: white; '
            f'display: flex; align-items: center; justify-content: center; '
            f'font-size: 12px; font-weight: 700;'
            f'">{dot_icon}</div>'
            f'<div style="font-size: 11px; color: {label_color}; margin-top: 4px; text-align: center; '
            f'white-space: nowrap;">{label}</div>'
            f'</div>'
            f'{connector}'
        )

    return (
        f'<div style="display: flex; align-items: flex-start; gap: 0; '
        f'padding: {Spacing.sm}px 0;">'
        f'{items_html}'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# System Health Traffic Light
# ---------------------------------------------------------------------------
def render_system_health(csv_ok: bool, llm_ok: bool, chroma_ok: bool) -> str:
    """Three-indicator health strip."""

    def _dot(ok: bool, label: str) -> str:
        color = Colors.success if ok else Colors.danger
        icon = "&#10003;" if ok else "&#10007;"
        return (
            f'<div style="display: flex; align-items: center; gap: 6px;">'
            f'<div style="'
            f'width: 8px; height: 8px; border-radius: 50%; background: {color};'
            f'"></div>'
            f'<span style="font-size: 12px; color: {Colors.text_secondary};">{label}</span>'
            f'<span style="font-size: 12px; color: {color}; font-weight: 600;">{icon}</span>'
            f'</div>'
        )

    return (
        f'<div style="'
        f'display: flex; flex-direction: column; gap: 6px; '
        f'background: {Colors.bg}; border-radius: {Radius.sm}px; '
        f'padding: {Spacing.md}px {Spacing.lg}px; '
        f'border: 1px solid {Colors.border_light};'
        f'">'
        f'{_dot(csv_ok, "CSV Data")}'
        f'{_dot(llm_ok, "LLM API")}'
        f'{_dot(chroma_ok, "ChromaDB")}'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Price Card
# ---------------------------------------------------------------------------
def render_price_card(
    brand: str,
    model: str,
    fob: str,
    cif: str,
    confidence: float,
) -> str:
    """Styled price result card with confidence bar."""
    pct = max(0, min(100, int(confidence * 100)))
    if confidence >= 0.85:
        bar_color = Colors.success
    elif confidence >= 0.70:
        bar_color = Colors.warning
    else:
        bar_color = Colors.danger

    return (
        f'<div style="'
        f'background: {Colors.card}; border-radius: {Radius.md}px; '
        f'padding: {Spacing.xl}px; '
        f'border: 1px solid {Colors.border_light}; box-shadow: {Shadow.sm};'
        f'">'
        f'<div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 12px;">'
        f'<div>'
        f'<span style="font-size: 16px; font-weight: 700; color: {Colors.text};">{brand}</span> '
        f'<span style="font-size: 14px; color: {Colors.text_secondary};">{model}</span>'
        f'</div>'
        f'<span style="font-size: 12px; font-weight: 600; color: {bar_color};">{pct}%</span>'
        f'</div>'
        f'<div style="display: flex; gap: 24px; margin-bottom: 12px;">'
        f'<div><span style="font-size: 11px; color: {Colors.text_tertiary};">FOB</span>'
        f'<div style="font-size: 18px; font-weight: 700; color: {Colors.text};">{fob}</div></div>'
        f'<div><span style="font-size: 11px; color: {Colors.text_tertiary};">CIF</span>'
        f'<div style="font-size: 18px; font-weight: 700; color: {Colors.text};">{cif}</div></div>'
        f'</div>'
        f'<div style="background: {Colors.border_light}; border-radius: 4px; height: 4px; overflow: hidden;">'
        f'<div style="width: {pct}%; height: 100%; background: {bar_color}; '
        f'border-radius: 4px; transition: width 0.4s ease;"></div>'
        f'</div>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Thinking Animation
# ---------------------------------------------------------------------------
def render_thinking_animation() -> str:
    """Three-dot bouncing CSS animation for loading states."""
    return (
        '<div class="thinking-dots">'
        '<span></span><span></span><span></span>'
        '</div>'
    )


# ---------------------------------------------------------------------------
# Confidence Progress Bar (for sidebar)
# ---------------------------------------------------------------------------
def render_confidence_bar(value: float) -> str:
    """Colored horizontal progress bar with percentage label."""
    pct = max(0, min(100, int(value * 100)))
    if value >= 0.85:
        bar_color = Colors.success
    elif value >= 0.70:
        bar_color = Colors.warning
    else:
        bar_color = Colors.danger

    return (
        f'<div style="margin: 6px 0;">'
        f'<div style="display: flex; justify-content: space-between; margin-bottom: 4px;">'
        f'<span style="font-size: 12px; color: {Colors.text_secondary};">Confidence</span>'
        f'<span style="font-size: 12px; font-weight: 700; color: {bar_color};">{pct}%</span>'
        f'</div>'
        f'<div style="background: {Colors.border_light}; border-radius: 4px; height: 6px; overflow: hidden;">'
        f'<div style="width: {pct}%; height: 100%; background: {bar_color}; '
        f'border-radius: 4px; transition: width 0.4s ease;"></div>'
        f'</div>'
        f'</div>'
    )
