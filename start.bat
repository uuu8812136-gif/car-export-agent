@echo off
chcp 65001 >/dev/null
echo.
echo  ============================================
echo   汽车出口 AI 销售助手 - 启动中...
echo  ============================================
echo.

cd /d %~dp0

set PYTHONPATH=%~dp0.pip-packages
set OPENAI_API_KEY=sk-b9ede3798a406b316e96984687f3040d2ac80a723b3cd3681a4d9d776c283336
set OPENAI_BASE_URL=https://hk.ticketpro.cc/v1
set LANGSMITH_API_KEY=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
set LANGCHAIN_TRACING_V2=true
set LANGCHAIN_PROJECT=car-export-agent

echo  依赖路径: %PYTHONPATH%
echo  启动 Streamlit...
echo.

python -m streamlit run app.py

pause
