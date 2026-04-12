"""Unit tests for price_node.py — no API calls required."""
import sys
from pathlib import Path

# 添加项目根到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
import pandas as pd


# ── 测试 _parse_price_range ─────────────────────────────────────────
class TestParsePriceRange:
    def setup_method(self):
        from agent.nodes.price_node import _parse_price_range
        self.parse = _parse_price_range

    def test_range_between(self):
        lo, hi = self.parse("I need a car between 15000 and 20000 USD")
        assert lo == 15000.0
        assert hi == 20000.0

    def test_range_dash(self):
        lo, hi = self.parse("budget 15000-20000")
        assert lo == 15000.0
        assert hi == 20000.0

    def test_under(self):
        lo, hi = self.parse("under 20000 USD")
        assert lo is None
        assert hi == 20000.0

    def test_over(self):
        lo, hi = self.parse("over 15000 dollars")
        assert lo == 15000.0
        assert hi is None

    def test_no_range(self):
        lo, hi = self.parse("BYD Seal price to Lagos")
        assert lo is None
        assert hi is None

    def test_invalid_string_no_crash(self):
        """Should not crash on queries like '5-seat SUV'."""
        lo, hi = self.parse("I need a 5-seat SUV")
        # Either returns None,None or valid floats — must not raise
        assert True


# ── 测试 _filter_by_price_range ────────────────────────────────────
class TestFilterByPriceRange:
    def setup_method(self):
        from agent.nodes.price_node import _filter_by_price_range
        self.filter = _filter_by_price_range
        self.df = pd.DataFrame({
            "model_name": ["Car A", "Car B", "Car C"],
            "fob_price_usd": [14000, 18000, 25000],
        })

    def test_filter_max(self):
        result = self.filter(self.df, None, 20000)
        assert len(result) == 2
        assert 25000 not in result["fob_price_usd"].values

    def test_filter_min(self):
        result = self.filter(self.df, 15000, None)
        assert len(result) == 2
        assert 14000 not in result["fob_price_usd"].values

    def test_filter_range(self):
        result = self.filter(self.df, 15000, 20000)
        assert len(result) == 1
        assert result.iloc[0]["fob_price_usd"] == 18000

    def test_empty_result_fallback(self):
        """When filter returns empty, should fallback to full list."""
        result = self.filter(self.df, 100000, 200000)
        assert len(result) == len(self.df)

    def test_no_filter(self):
        result = self.filter(self.df, None, None)
        assert len(result) == len(self.df)


# ── 测试 CSV 数据完整性 ─────────────────────────────────────────────
class TestPricesCSV:
    def setup_method(self):
        self.csv_path = Path(__file__).resolve().parent.parent.parent / "data" / "prices.csv"
        self.df = pd.read_csv(self.csv_path)

    def test_required_columns_exist(self):
        required = ["model_name", "brand", "fob_price_usd", "cif_price_usd", "product_id", "update_time"]
        for col in required:
            assert col in self.df.columns, f"Missing column: {col}"

    def test_no_empty_product_ids(self):
        assert self.df["product_id"].notna().all()
        assert (self.df["product_id"] != "").all()

    def test_prices_are_numeric(self):
        assert pd.to_numeric(self.df["fob_price_usd"], errors="coerce").notna().all()
        assert pd.to_numeric(self.df["cif_price_usd"], errors="coerce").notna().all()

    def test_cif_greater_than_fob(self):
        fob = pd.to_numeric(self.df["fob_price_usd"])
        cif = pd.to_numeric(self.df["cif_price_usd"])
        assert (cif >= fob).all(), "CIF should always be >= FOB"

    def test_at_least_10_models(self):
        assert len(self.df) >= 10
