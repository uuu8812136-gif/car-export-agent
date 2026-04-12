#!/bin/bash
echo ""
echo " ============================================"
echo "  汽车出口 AI 销售助手 - 启动中..."
echo " ============================================"
echo ""

cd "$(dirname "$0")"

export PYTHONPATH="$(pwd)/.pip-packages"
export OPENAI_API_KEY="sk-b9ede3798a406b316e96984687f3040d2ac80a723b3cd3681a4d9d776c283336"
export OPENAI_BASE_URL="https://hk.ticketpro.cc/v1"
export LANGSMITH_API_KEY="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"
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_PROJECT="car-export-agent"

echo " 依赖路径: $PYTHONPATH"
echo " 启动 Streamlit..."
echo ""

python3 -m streamlit run app.py
