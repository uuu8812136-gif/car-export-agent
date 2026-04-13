"""独立运行 Telegram Bot — 不依赖 Streamlit，不需要浏览器打开。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from telegram.handler import run_polling

if __name__ == "__main__":
    run_polling()
