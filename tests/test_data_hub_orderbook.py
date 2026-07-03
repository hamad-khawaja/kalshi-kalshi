"""Tests for DataHub's Kalshi orderbook WebSocket message parsing.

Covers the real Kalshi API message shapes (yes_dollars_fp/no_dollars_fp for
snapshots, price_dollars/delta_fp for deltas) rather than the legacy
non-fp field names the parser previously (incorrectly) expected.
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from src.data.data_hub import DataHub

TICKER = "KXBTC15M-26FEB121415-B50000"


@pytest.fixture
def data_hub() -> DataHub:
    return DataHub(
        kalshi_rest=MagicMock(),
        kalshi_ws=MagicMock(),
        feeds={},
        scanners={},
    )


def _snapshot_msg(seq: int = 1) -> dict:
    return {
        "type": "orderbook_snapshot",
        "seq": seq,
        "msg": {
            "market_ticker": TICKER,
            "yes_dollars_fp": [["0.52", "100.00"], ["0.50", "200.00"]],
            "no_dollars_fp": [["0.48", "150.00"]],
        },
    }


def _delta_msg(seq: int, side: str, price: str, delta_fp: str) -> dict:
    return {
        "type": "orderbook_delta",
        "seq": seq,
        "msg": {
            "market_ticker": TICKER,
            "side": side,
            "price_dollars": price,
            "delta_fp": delta_fp,
        },
    }


def test_snapshot_parses_fp_fields_into_levels(data_hub: DataHub) -> None:
    data_hub._on_orderbook_update(TICKER, _snapshot_msg())

    ob = data_hub._orderbook_cache[TICKER]
    assert [(lvl.price_dollars, lvl.quantity) for lvl in ob.yes_levels] == [
        (Decimal("0.52"), 100),
        (Decimal("0.50"), 200),
    ]
    assert [(lvl.price_dollars, lvl.quantity) for lvl in ob.no_levels] == [
        (Decimal("0.48"), 150),
    ]


def test_delta_increases_existing_level(data_hub: DataHub) -> None:
    data_hub._on_orderbook_update(TICKER, _snapshot_msg(seq=1))
    data_hub._on_orderbook_update(
        TICKER, _delta_msg(seq=2, side="yes", price="0.52", delta_fp="25.00")
    )

    ob = data_hub._orderbook_cache[TICKER]
    yes_at_52 = next(lvl for lvl in ob.yes_levels if lvl.price_dollars == Decimal("0.52"))
    assert yes_at_52.quantity == 125


def test_delta_removes_level_when_quantity_hits_zero(data_hub: DataHub) -> None:
    data_hub._on_orderbook_update(TICKER, _snapshot_msg(seq=1))
    data_hub._on_orderbook_update(
        TICKER, _delta_msg(seq=2, side="no", price="0.48", delta_fp="-150.00")
    )

    ob = data_hub._orderbook_cache[TICKER]
    assert ob.no_levels == []


def test_delta_adds_new_price_level_sorted_descending(data_hub: DataHub) -> None:
    data_hub._on_orderbook_update(TICKER, _snapshot_msg(seq=1))
    data_hub._on_orderbook_update(
        TICKER, _delta_msg(seq=2, side="yes", price="0.55", delta_fp="10.00")
    )

    ob = data_hub._orderbook_cache[TICKER]
    prices = [lvl.price_dollars for lvl in ob.yes_levels]
    assert prices == [Decimal("0.55"), Decimal("0.52"), Decimal("0.50")]


def test_delta_before_snapshot_is_ignored(data_hub: DataHub) -> None:
    data_hub._on_orderbook_update(
        TICKER, _delta_msg(seq=1, side="yes", price="0.52", delta_fp="10.00")
    )

    assert TICKER not in data_hub._orderbook_cache


def test_stale_seq_is_deduped(data_hub: DataHub) -> None:
    data_hub._on_orderbook_update(TICKER, _snapshot_msg(seq=5))
    data_hub._on_orderbook_update(
        TICKER, _delta_msg(seq=3, side="yes", price="0.52", delta_fp="999.00")
    )

    ob = data_hub._orderbook_cache[TICKER]
    yes_at_52 = next(lvl for lvl in ob.yes_levels if lvl.price_dollars == Decimal("0.52"))
    assert yes_at_52.quantity == 100  # unchanged — stale delta rejected


def test_book_has_real_levels_after_snapshot(data_hub: DataHub) -> None:
    """Regression test: this used to always be empty due to the field-name bug,
    which manifested as permanent `thin_book_no_levels` rejections."""
    data_hub._on_orderbook_update(TICKER, _snapshot_msg())

    ob = data_hub._orderbook_cache[TICKER]
    assert ob.best_yes_bid == Decimal("0.52")
    assert ob.best_no_bid == Decimal("0.48")
    assert ob.implied_yes_prob is not None
