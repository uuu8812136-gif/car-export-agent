@echo off
chcp 65001 >nul
echo.
echo  ============================================
echo   汽车出口 AI 销售助手 - 启动中...
echo  ============================================
echo.

cd /d %~dp0

set PYTHONPATH=%~dp0.pip-packages
set OPENAI_API_KEY=sk-b9ede3798a406b316e96984687f3040d2ac80a723b3cd3681a4d9d776c283336
set OPENAI_BASE_URL=https://hk.ticketpro.cc/v1
set LANGCHAIN_TRACING_V2=false
set TELEGRAM_BOT_TOKEN=8525472639:AAEpXShI-IWTlMP5XIeSYE_U3LhqJH3VNck
set TELEGRAM_BOT_USERNAME=ksnzizjwns_bot

echo  [1/2] 启动 Telegram Bot (@ksnzizjwns_bot)...
start "Telegram Bot" python telegram_bot.py

echo  [2/2] 启动 Streamlit 演示界面...
echo.
python -m streamlit run app.py

pause
