"""
RAGAS 评估脚本 — 汽车出口销售 AI Agent
用法：python eval/run_ragas.py

前提：
1. 设置好 .env 中的 OPENAI_API_KEY
2. 可选：设置 LANGSMITH_API_KEY 同时记录到 LangSmith

输出：
- 控制台打印 faithfulness、answer_relevancy、context_recall 分数
- 结果保存到 eval/results/ragas_report.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

# 确保项目根目录在 Python 路径中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os
os.environ.setdefault("PYTHONPATH", str(PROJECT_ROOT))

# 加载 .pip-packages（如果存在）
pip_pkg = PROJECT_ROOT / ".pip-packages"
if pip_pkg.exists():
    sys.path.insert(0, str(pip_pkg))


def load_test_dataset() -> list[dict]:
    ds_path = Path(__file__).parent / "test_dataset.json"
    with open(ds_path, encoding="utf-8") as f:
        return json.load(f)


def collect_answers(dataset: list[dict]) -> list[dict]:
    """对每条测试问题跑一遍 Agent，收集答案和检索上下文。"""
    print("正在收集 Agent 回答（共 {} 条）...".format(len(dataset)))
    from agent.graph import run_agent

    results = []
    for i, item in enumerate(dataset, 1):
        q = item["question"]
        print(f"  [{i}/{len(dataset)}] {q[:60]}...")
        try:
            raw = run_agent(q, [], session_id="eval-session", user_role="sales")
            # 提取答案文本
            if isinstance(raw, tuple):
                answer = str(raw[0]) if raw[0] else ""
            elif isinstance(raw, dict):
                answer = raw.get("response", raw.get("draft_answer", ""))
            else:
                answer = str(raw)
        except Exception as e:
            answer = f"[ERROR] {e}"

        results.append({
            "question": q,
            "answer": answer,
            "contexts": item.get("contexts", []),
            "ground_truth": item.get("ground_truth", ""),
        })
    return results


def run_evaluation(results: list[dict]) -> dict:
    """用 RAGAS 对收集到的答案打分。"""
    print("\n正在运行 RAGAS 评估...")
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_recall
        from datasets import Dataset

        ds = Dataset.from_list(results)
        scores = evaluate(
            ds,
            metrics=[faithfulness, answer_relevancy, context_recall],
        )
        return scores
    except ImportError:
        print("提示：ragas 未安装，跳过评估。运行：pip install ragas datasets")
        return {}


def save_results(results: list[dict], scores: dict) -> None:
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {k: float(v) for k, v in scores.items()} if scores else {},
        "details": results,
    }
    out_path = out_dir / "ragas_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存：{out_path}")


def main() -> None:
    print("=" * 60)
    print("汽车出口 AI Agent — RAGAS 评估")
    print("=" * 60)

    dataset = load_test_dataset()
    results = collect_answers(dataset)
    scores = run_evaluation(results)

    if scores:
        print("\n📊 评估结果：")
        for metric, value in scores.items():
            bar = "█" * int(float(value) * 20)
            print(f"  {metric:<25} {float(value):.3f}  {bar}")

    save_results(results, scores)
    print("\n✅ 评估完成")


if __name__ == "__main__":
    main()
