"""Unit tests for telegram/handler.py — no API calls required."""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


class TestOffsetPersistence:
    """测试偏移量的保存和读取。"""

    def test_load_offset_missing_file(self, tmp_path):
        """文件不存在时应该返回 0。"""
        from telegram import handler

        original = handler._OFFSET_FILE
        handler._OFFSET_FILE = tmp_path / "nonexistent.txt"
        try:
            assert handler._load_offset() == 0
        finally:
            handler._OFFSET_FILE = original

    def test_save_load_roundtrip(self, tmp_path):
        """保存偏移量再读出来，值应该一致。"""
        from telegram import handler

        offset_file = tmp_path / "telegram_offset.txt"
        original = handler._OFFSET_FILE
        handler._OFFSET_FILE = offset_file
        try:
            handler._save_offset(42)
            assert handler._load_offset() == 42
            handler._save_offset(999)
            assert handler._load_offset() == 999
        finally:
            handler._OFFSET_FILE = original

    def test_load_offset_corrupt_file(self, tmp_path):
        """文件内容损坏时应该返回 0，不能崩溃。"""
        from telegram import handler

        offset_file = tmp_path / "telegram_offset.txt"
        offset_file.write_text("not_a_number", encoding="utf-8")
        original = handler._OFFSET_FILE
        handler._OFFSET_FILE = offset_file
        try:
            assert handler._load_offset() == 0
        finally:
            handler._OFFSET_FILE = original


class TestDeduplication:
    """测试消息去重机制。"""

    def test_dedup_set_detects_duplicate(self):
        """重复的 update_id 应该能被检测到。"""
        from telegram import handler

        handler._processed_update_ids.clear()
        handler._processed_update_ids.add(12345)
        assert 12345 in handler._processed_update_ids
        assert 99999 not in handler._processed_update_ids
        handler._processed_update_ids.clear()


class TestPollingLock:
    """测试线程锁能防止重复启动。"""

    def test_lock_prevents_double_acquire(self):
        """第二次抢锁应该失败。"""
        from telegram.handler import _polling_lock

        got_first = _polling_lock.acquire(blocking=False)
        assert got_first is True
        got_second = _polling_lock.acquire(blocking=False)
        assert got_second is False
        _polling_lock.release()


class TestHistoryAppend:
    """测试历史记录文件写入。"""

    def test_append_creates_file(self, tmp_path):
        """文件不存在时应该能自动创建。"""
        from telegram import handler

        history_file = tmp_path / "test_history.json"
        original = handler._HISTORY_FILE
        handler._HISTORY_FILE = history_file
        try:
            handler._append_to_history_file({"test": "entry"})
            data = json.loads(history_file.read_text(encoding="utf-8"))
            assert len(data) == 1
            assert data[0]["test"] == "entry"
        finally:
            handler._HISTORY_FILE = original
