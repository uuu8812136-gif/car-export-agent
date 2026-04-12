"""
telegram_bot.py — 独立运行的 Telegram Bot 进程
用法：python telegram_bot.py
"""
import os
import sys
from pathlib import Path

# 加载 .pip-packages（如果存在）
_pip = Path(__file__).parent / ".pip-packages"
if _pip.exists():
    sys.path.insert(0, str(_pip))

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

# 设置必要环境变量
os.environ.setdefault("OPENAI_API_KEY", "sk-b9ede3798a406b316e96984687f3040d2ac80a723b3cd3681a4d9d776c283336")
os.environ.setdefault("OPENAI_BASE_URL", "https://hk.ticketpro.cc/v1")
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "8525472639:AAEpXShI-IWTlMP5XIeSYE_U3LhqJH3VNck")
os.environ.setdefault("TELEGRAM_BOT_USERNAME", "ksnzizjwns_bot")
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from telegram.handler import run_polling

if __name__ == "__main__":
    run_polling()
