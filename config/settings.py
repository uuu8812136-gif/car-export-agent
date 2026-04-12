from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction as _ChromaEF


class _ChromaEmbeddings(Embeddings):
    """LangChain-compatible wrapper around ChromaDB's built-in ONNX embeddings.
    Avoids loading sentence-transformers / torch — much lighter on memory."""

    def __init__(self) -> None:
        self._ef = _ChromaEF()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(v) for v in vec] for vec in self._ef(texts)]

    def embed_query(self, text: str) -> list[float]:
        return [float(v) for v in self._ef([text])[0]]

load_dotenv()

# 强制覆盖系统环境变量，确保使用项目指定的代理
import os as _os
_PROXY_KEY = "sk-b9ede3798a406b316e96984687f3040d2ac80a723b3cd3681a4d9d776c283336"
_PROXY_URL = "https://hk.ticketpro.cc/v1"
_os.environ["OPENAI_API_KEY"] = _PROXY_KEY
_os.environ["OPENAI_BASE_URL"] = _PROXY_URL
_os.environ["OPENAI_API_BASE"] = _PROXY_URL

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

CHROMA_DB_PATH: Path = PROJECT_ROOT / "rag" / "chroma_db"
PRICES_CSV_PATH: Path = PROJECT_ROOT / "data" / "prices.csv"
CONTRACTS_OUTPUT_DIR: Path = PROJECT_ROOT / "contracts" / "output"
CONTRACTS_TEMPLATE_PATH: Path = PROJECT_ROOT / "contracts" / "templates" / "quote_template.md"

CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
PRICES_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
CONTRACTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CONTRACTS_TEMPLATE_PATH.parent.mkdir(parents=True, exist_ok=True)

# LLM — GPT-5.4 via proxy
OPENAI_API_KEY: str = os.getenv(
    "OPENAI_API_KEY",
    "sk-b9ede3798a406b316e96984687f3040d2ac80a723b3cd3681a4d9d776c283336",
)
OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://hk.ticketpro.cc/v1")

llm: ChatOpenAI = ChatOpenAI(
    model="gpt-5.4",
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    temperature=0,
)

embeddings: _ChromaEmbeddings = _ChromaEmbeddings()

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
