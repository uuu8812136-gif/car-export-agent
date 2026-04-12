from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# 强制覆盖系统环境变量，确保使用项目指定的代理
import os as _os
_PROXY_KEY = "sk-b9ede3798a406b316e96984687f3040d2ac80a723b3cd3681a4d9d776c283336"
_PROXY_URL = "https://hk.ticketpro.cc/v1"
_os.environ["OPENAI_API_KEY"] = _PROXY_KEY
_os.environ["OPENAI_BASE_URL"] = _PROXY_URL
_os.environ["OPENAI_API_BASE"] = _PROXY_URL

# LangSmith 可观测性（设置环境变量后自动启用）
import os as _lsm_os
if _lsm_os.getenv("LANGSMITH_API_KEY"):
    _lsm_os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    _lsm_os.environ.setdefault("LANGCHAIN_PROJECT", "car-export-agent")

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

CHROMA_DB_PATH: Path = PROJECT_ROOT / "rag" / "chroma_db"
PRICES_CSV_PATH: Path = PROJECT_ROOT / "data" / "prices.csv"
CONTRACTS_OUTPUT_DIR: Path = PROJECT_ROOT / "contracts" / "output"
CONTRACTS_TEMPLATE_PATH: Path = PROJECT_ROOT / "contracts" / "templates" / "quote_template.md"

CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
PRICES_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
CONTRACTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CONTRACTS_TEMPLATE_PATH.parent.mkdir(parents=True, exist_ok=True)

# LLM 配置（懒加载：模块导入时不实例化，等首次调用时才创建）
OPENAI_API_KEY: str = os.getenv(
    "OPENAI_API_KEY",
    "sk-b9ede3798a406b316e96984687f3040d2ac80a723b3cd3681a4d9d776c283336",
)
OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://hk.ticketpro.cc/v1")

_llm_instance = None
_embeddings_instance = None


def _get_llm():
    global _llm_instance
    if _llm_instance is None:
        from langchain_openai import ChatOpenAI
        _llm_instance = ChatOpenAI(
            model="gpt-5.4",
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            temperature=0,
        )
    return _llm_instance


def _get_embeddings():
    global _embeddings_instance
    if _embeddings_instance is None:
        from langchain_openai import OpenAIEmbeddings
        _embeddings_instance = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            dimensions=1024,  # 压缩维度，节省存储
        )
    return _embeddings_instance


class _LazyProxy:
    """Transparent proxy that defers heavy object creation to first attribute access."""
    def __init__(self, factory):
        object.__setattr__(self, "_factory", factory)
        object.__setattr__(self, "_obj", None)

    def _resolve(self):
        obj = object.__getattribute__(self, "_obj")
        if obj is None:
            factory = object.__getattribute__(self, "_factory")
            obj = factory()
            object.__setattr__(self, "_obj", obj)
        return obj

    def __getattr__(self, name):
        return getattr(self._resolve(), name)

    def __call__(self, *args, **kwargs):
        return self._resolve()(*args, **kwargs)


llm = _LazyProxy(_get_llm)
embeddings = _LazyProxy(_get_embeddings)

# ---------------------------------------------------------------------------
# WhatsApp — Green API
# ---------------------------------------------------------------------------
# Register free at https://green-api.com and set these in your .env:
#   GREEN_API_INSTANCE_ID=your_instance_id
#   GREEN_API_TOKEN=your_api_token
GREEN_API_INSTANCE_ID: str = os.getenv("GREEN_API_INSTANCE_ID", "")
GREEN_API_TOKEN: str = os.getenv("GREEN_API_TOKEN", "")
GREEN_API_BASE_URL: str = "https://api.green-api.com"

__all__ = [
    "llm",
    "embeddings",
    "PROJECT_ROOT",
    "CHROMA_DB_PATH",
    "PRICES_CSV_PATH",
    "CONTRACTS_OUTPUT_DIR",
    "CONTRACTS_TEMPLATE_PATH",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "GREEN_API_INSTANCE_ID",
    "GREEN_API_TOKEN",
    "GREEN_API_BASE_URL",
]
