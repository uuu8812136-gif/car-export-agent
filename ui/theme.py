"""
Design Token System for the Car Export AI Sales Assistant.

Provides a single source of truth for colors, spacing, typography,
shadows, and border-radius values used across all UI components.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Color Tokens
# ---------------------------------------------------------------------------
class Colors:
    primary = "#0F172A"        # dark navy
    accent = "#3B82F6"         # blue
    accent_light = "#DBEAFE"   # light blue tint
    success = "#10B981"        # green
    warning = "#F59E0B"        # amber
    danger = "#EF4444"         # red
    bg = "#F8FAFC"             # subtle off-white
    card = "#FFFFFF"
    text = "#0F172A"
    text_secondary = "#64748B" # muted grey
    text_tertiary = "#94A3B8"  # very muted
    border = "#E2E8F0"         # light border
    border_light = "#F1F5F9"   # very light border


# ---------------------------------------------------------------------------
# Spacing Scale (px)
# ---------------------------------------------------------------------------
class Spacing:
    xs = 4
    sm = 8
    md = 12
    lg = 16
    xl = 24
    xxl = 32
    xxxl = 48


# ---------------------------------------------------------------------------
# Typography
# ---------------------------------------------------------------------------
FONT_STACK = "'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"

class FontSize:
    xs = "11px"
    sm = "13px"
    md = "14px"
    lg = "16px"
    xl = "20px"
    xxl = "24px"
    hero = "32px"


# ---------------------------------------------------------------------------
# Shadows
# ---------------------------------------------------------------------------
class Shadow:
    sm = "0 1px 2px rgba(15, 23, 42, 0.04), 0 1px 3px rgba(15, 23, 42, 0.06)"
    md = "0 4px 6px rgba(15, 23, 42, 0.04), 0 2px 4px rgba(15, 23, 42, 0.06)"
    lg = "0 10px 15px rgba(15, 23, 42, 0.06), 0 4px 6px rgba(15, 23, 42, 0.04)"


# ---------------------------------------------------------------------------
# Border Radius
# ---------------------------------------------------------------------------
class Radius:
    sm = 8
    md = 12
    lg = 16
    xl = 20


# ---------------------------------------------------------------------------
# CSS Custom Properties Generator
# ---------------------------------------------------------------------------
def css_variables() -> str:
    """Return a :root block with all design tokens as CSS custom properties."""
    return f"""
    :root {{
        --color-primary: {Colors.primary};
        --color-accent: {Colors.accent};
        --color-accent-light: {Colors.accent_light};
        --color-success: {Colors.success};
        --color-warning: {Colors.warning};
        --color-danger: {Colors.danger};
        --color-bg: {Colors.bg};
        --color-card: {Colors.card};
        --color-text: {Colors.text};
        --color-text-secondary: {Colors.text_secondary};
        --color-text-tertiary: {Colors.text_tertiary};
        --color-border: {Colors.border};
        --color-border-light: {Colors.border_light};

        --space-xs: {Spacing.xs}px;
        --space-sm: {Spacing.sm}px;
        --space-md: {Spacing.md}px;
        --space-lg: {Spacing.lg}px;
        --space-xl: {Spacing.xl}px;
        --space-xxl: {Spacing.xxl}px;
        --space-xxxl: {Spacing.xxxl}px;

        --font-family: {FONT_STACK};
        --font-xs: {FontSize.xs};
        --font-sm: {FontSize.sm};
        --font-md: {FontSize.md};
        --font-lg: {FontSize.lg};
        --font-xl: {FontSize.xl};
        --font-xxl: {FontSize.xxl};
        --font-hero: {FontSize.hero};

        --shadow-sm: {Shadow.sm};
        --shadow-md: {Shadow.md};
        --shadow-lg: {Shadow.lg};

        --radius-sm: {Radius.sm}px;
        --radius-md: {Radius.md}px;
        --radius-lg: {Radius.lg}px;
        --radius-xl: {Radius.xl}px;
    }}
    """
