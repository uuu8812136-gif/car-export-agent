"""
generate_modern_ui.py - GPT generates iOS-style modern Streamlit UI
Run: venv/Scripts/python generate_modern_ui.py
"""
from pathlib import Path
from openai import OpenAI

OPENAI_API_KEY = "sk-b9ede3798a406b316e96984687f3040d2ac80a723b3cd3681a4d9d776c283336"
BASE_URL = "https://hk.ticketpro.cc/v1"
MODEL = "gpt-5.4"
MODEL_FALLBACK = "gpt-4.1"
PROJECT_ROOT = Path("H:/car-export-agent")

client = OpenAI(api_key=OPENAI_API_KEY, base_url=BASE_URL)

SYS = (
    "You are an expert Streamlit developer and iOS UI designer. "
    "Generate complete, production-ready Python code with NO placeholders, NO TODOs. "
    "Use type hints. All functions fully implemented."
)

PROMPT_FILE = PROJECT_ROOT / "_ui_prompt.txt"
APP_PROMPT = PROMPT_FILE.read_text(encoding="utf-8")


def call_gpt(prompt: str, model: str = MODEL) -> str:
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYS}, {"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=4096,
        )
        return r.choices[0].message.content
    except Exception as e:
        if model != MODEL_FALLBACK:
            print(f"  Retry: {e}")
            return call_gpt(prompt, MODEL_FALLBACK)
        raise


def strip_fences(raw: str) -> str:
    s = raw.strip()
    if not s.startswith("```"):
        return raw
    lines = s.splitlines()
    end = -1 if lines[-1].strip() == "```" else len(lines)
    return "\n".join(lines[1:end])


if __name__ == "__main__":
    print("Generating iOS-style modern UI via GPT...")
    raw = call_gpt(APP_PROMPT)
    content = strip_fences(raw)
    out = PROJECT_ROOT / "app.py"
    out.write_text(content, encoding="utf-8")
    print(f"Done: app.py ({len(content)} chars)")
