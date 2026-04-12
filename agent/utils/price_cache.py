"""SQLite-based TTL cache for price lookups."""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

from config.settings import PROJECT_ROOT

_DB_PATH: Path = PROJECT_ROOT / "data" / "price_cache.db"


@contextmanager
def _connect() -> Generator[sqlite3.Connection, None, None]:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH), timeout=5)
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS cache "
            "(key TEXT PRIMARY KEY, value TEXT, expires_at REAL)"
        )
        yield conn
        conn.commit()
    finally:
        conn.close()


class PriceCache:
    """Thread-safe SQLite cache with TTL support."""

    @staticmethod
    def get(key: str) -> dict[str, Any] | None:
        with _connect() as conn:
            row = conn.execute(
                "SELECT value, expires_at FROM cache WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                return None
            value, expires_at = row
            if time.time() > expires_at:
                conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                return None
            return json.loads(value)

    @staticmethod
    def set(key: str, value: dict[str, Any], ttl_seconds: int = 86400) -> None:
        with _connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, value, expires_at) VALUES (?, ?, ?)",
                (key, json.dumps(value, ensure_ascii=False), time.time() + ttl_seconds),
            )

    @staticmethod
    def clear_expired() -> int:
        with _connect() as conn:
            cur = conn.execute("DELETE FROM cache WHERE expires_at < ?", (time.time(),))
            return cur.rowcount

    @staticmethod
    def clear_all() -> None:
        with _connect() as conn:
            conn.execute("DELETE FROM cache")

    @staticmethod
    def stats() -> dict[str, int]:
        with _connect() as conn:
            total   = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            expired = conn.execute("SELECT COUNT(*) FROM cache WHERE expires_at < ?", (time.time(),)).fetchone()[0]
        return {"total": total, "expired": expired, "active": total - expired}
