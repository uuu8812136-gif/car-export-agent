"""
overnight_test.py — 无 LLM 调用的本地测试，可跑整夜
测试内容：
  1. 单元测试（pytest，无API调用）
  2. CSV 数据完整性
  3. 价格区间解析逻辑
  4. 关键模块可导入性
  5. Telegram bot API 连通性（仅 getMe，不调 LLM）
  6. 生成测试报告 scripts/test_report.json
"""
import sys, os, json, time, subprocess, traceback
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent

# bash 里 /h/ 等价于 H:/ ，但 subprocess 是 Windows 进程需要真实路径
# H:/car-export-agent 正斜杠写法 Windows 也认识
WIN_ROOT = Path("H:/car-export-agent") if Path("H:/car-export-agent").exists() else ROOT

sys.path.insert(0, str(WIN_ROOT))
sys.path.insert(0, str(WIN_ROOT / ".pip-packages"))

os.environ.update({
    "OPENAI_API_KEY": "sk-b9ede3798a406b316e96984687f3040d2ac80a723b3cd3681a4d9d776c283336",
    "OPENAI_BASE_URL": "https://hk.ticketpro.cc/v1",
    "LANGCHAIN_TRACING_V2": "false",
    "TELEGRAM_BOT_TOKEN": "8525472639:AAEpXShI-IWTlMP5XIeSYE_U3LhqJH3VNck",
})

REPORT_FILE = ROOT / "scripts" / "test_report.json"
results = []

def record(name, passed, detail=""):
    ts = datetime.now().strftime("%H:%M:%S")
    status = "PASS" if passed else "FAIL"
    results.append({"time": ts, "test": name, "status": status, "detail": detail})
    print(f"[{ts}] {status:4} | {name}" + (f" — {detail}" if detail else ""))

def save_report(round_n, total_pass, total_fail):
    REPORT_FILE.parent.mkdir(exist_ok=True)
    report = {
        "generated": datetime.now().isoformat(),
        "rounds_completed": round_n,
        "total_pass": total_pass,
        "total_fail": total_fail,
        "results": results[-100:],  # 最近100条
    }
    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

# ── 测试函数 ─────────────────────────────────────────────────────────

def test_csv_integrity():
    import pandas as pd
    df = pd.read_csv(WIN_ROOT / "data" / "prices.csv")
    required = ["model_name", "brand", "fob_price_usd", "cif_price_usd", "product_id", "update_time"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return False, f"Missing columns: {missing}"
    fob = pd.to_numeric(df["fob_price_usd"], errors="coerce")
    cif = pd.to_numeric(df["cif_price_usd"], errors="coerce")
    if not (cif >= fob).all():
        return False, "CIF < FOB detected"
    return True, f"{len(df)} rows OK"

def test_price_range_parse():
    from agent.nodes.price_node import _parse_price_range
    cases = [
        ("under 20000 USD", (None, 20000.0)),
        ("between 15000 and 25000", (15000.0, 25000.0)),
        ("over 10000", (10000.0, None)),
        ("BYD Seal price", (None, None)),
        ("5-seat SUV", (None, None)),  # 不应崩溃
    ]
    for query, expected in cases:
        result = _parse_price_range(query)
        if result != expected:
            return False, f"'{query}' → {result}, expected {expected}"
    return True, f"{len(cases)} cases OK"

def test_price_filter():
    import pandas as pd
    from agent.nodes.price_node import _filter_by_price_range
    df = pd.DataFrame({"model_name": ["A","B","C"], "fob_price_usd": [10000, 18000, 30000]})
    r = _filter_by_price_range(df, None, 20000)
    if len(r) != 2:
        return False, f"Expected 2 rows under 20k, got {len(r)}"
    r2 = _filter_by_price_range(df, 100000, 200000)
    if len(r2) != 3:  # fallback to full list
        return False, f"Empty filter should fallback, got {len(r2)}"
    return True, "filter logic OK"

def test_imports():
    mods = [
        "agent.state",
        "agent.nodes.price_node",
        "agent.nodes.reflection_pipeline",
        "agent.nodes.human_intervention",
        "config.prompts",
        "telegram.handler",
    ]
    for mod in mods:
        try:
            __import__(mod)
        except Exception as e:
            return False, f"{mod}: {e}"
    return True, f"{len(mods)} modules OK"

def test_pydantic_schema():
    from agent.nodes.reflection_pipeline import FactCheckResult, ComplianceCheckResult, ReflectionLog
    for cls in [FactCheckResult, ComplianceCheckResult]:
        fields = cls.model_fields
        for f in ["error_type", "correction_plan", "trigger_condition"]:
            if f not in fields:
                return False, f"{cls.__name__} missing field {f}"
    return True, "Pydantic schemas OK"

def test_agent_state():
    from agent.state import get_default_state
    state = get_default_state()
    required = ["messages", "intent", "draft_answer", "reflection_log",
                "price_confidence_score", "needs_human_review", "session_id"]
    missing = [f for f in required if f not in state]
    if missing:
        return False, f"Missing state fields: {missing}"
    return True, "AgentState OK"

def test_telegram_api():
    import httpx
    try:
        r = httpx.get(
            "https://api.telegram.org/bot8525472639:AAEpXShI-IWTlMP5XIeSYE_U3LhqJH3VNck/getMe",
            timeout=10
        )
        data = r.json()
        if data.get("ok") and data["result"]["username"] == "ksnzizjwns_bot":
            return True, "Bot online"
        return False, f"Unexpected response: {data}"
    except Exception as e:
        return False, str(e)

def test_pytest():
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/unit/", "-q", "--tb=short", "--no-header"],
        capture_output=True, text=True, cwd=str(WIN_ROOT),
        env={**os.environ, "PYTHONPATH": f"{WIN_ROOT}{os.pathsep}{WIN_ROOT / '.pip-packages'}"},
        timeout=60
    )
    passed = "failed" not in r.stdout and r.returncode == 0
    lines = r.stdout.strip().split("\n")
    summary = lines[-1] if lines else "no output"
    return passed, summary

def test_git_status():
    r = subprocess.run(["git", "status", "--short"], capture_output=True, text=True, cwd=str(WIN_ROOT))
    dirty = r.stdout.strip()
    return True, f"{'clean' if not dirty else dirty[:60]}"

# ── 主循环 ───────────────────────────────────────────────────────────

TESTS = [
    ("CSV integrity",     test_csv_integrity),
    ("Price range parse", test_price_range_parse),
    ("Price filter",      test_price_filter),
    ("Module imports",    test_imports),
    ("Pydantic schema",   test_pydantic_schema),
    ("AgentState fields", test_agent_state),
    ("Telegram API",      test_telegram_api),
    ("pytest suite",      test_pytest),
    ("Git status",        test_git_status),
]

END_TIME = time.time() + 5 * 3600  # 5 小时
round_n = 0
total_pass = 0
total_fail = 0

print(f"Starting 5-hour test run. End time: {datetime.fromtimestamp(END_TIME).strftime('%H:%M:%S')}")
print("=" * 60)

while time.time() < END_TIME:
    round_n += 1
    print(f"\n{'='*60}")
    print(f"Round {round_n} — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    for name, fn in TESTS:
        try:
            passed, detail = fn()
        except Exception as e:
            passed, detail = False, traceback.format_exc().splitlines()[-1]
        record(name, passed, detail)
        if passed:
            total_pass += 1
        else:
            total_fail += 1

    save_report(round_n, total_pass, total_fail)
    print(f"\nRound {round_n} complete — Pass: {total_pass}, Fail: {total_fail}")

    # 每轮等 30 分钟（不消耗资源）
    sleep_secs = min(1800, END_TIME - time.time())
    if sleep_secs > 0:
        print(f"Sleeping {sleep_secs/60:.0f} min until next round...")
        time.sleep(sleep_secs)

print("\n" + "="*60)
print(f"5-hour test complete. Total Pass: {total_pass}, Fail: {total_fail}")
save_report(round_n, total_pass, total_fail)
