import pytest
from app.engine.microstructure import PriceLevel, BinaryMarketBook
from app.engine.depth import (
    walk_book_depth,
    evaluate_depth_profile,
    depth_adjusted_gross_edge
)

def test_walk_book_single_level():
    levels = [PriceLevel(price_cents=45, quantity=100), PriceLevel(price_cents=47, quantity=200)]
    # Target size 50 <= top-of-book (100)
    sim = walk_book_depth(levels, 50, side="ask")

    assert sim.target_qty == 50
    assert sim.filled_qty == 50
    assert sim.is_fully_filled is True
    assert sim.total_cost_cents == 50 * 45
    assert sim.vwap_cents == 45.0
    assert sim.slippage_cents == 0.0
    assert sim.marginal_price_cents == 45
    assert sim.levels_swept == 1

def test_walk_book_multi_level_slippage():
    # Asks:
    # Level 1: 50 @ 40c
    # Level 2: 50 @ 42c
    # Level 3: 100 @ 45c
    levels = [
        PriceLevel(price_cents=40, quantity=50),
        PriceLevel(price_cents=42, quantity=50),
        PriceLevel(price_cents=45, quantity=100),
    ]

    sim = walk_book_depth(levels, 150, side="ask")
    assert sim.target_qty == 150
    assert sim.filled_qty == 150
    assert sim.is_fully_filled is True

    # Expected: 50 * 40 (2000) + 50 * 42 (2100) + 50 * 45 (2250) = 6350 cents
    assert sim.total_cost_cents == 6350
    assert pytest.approx(sim.vwap_cents, rel=1e-4) == 6350 / 150  # ~42.3333c
    assert pytest.approx(sim.slippage_cents, rel=1e-4) == (6350 / 150) - 40.0  # ~2.3333c
    assert sim.marginal_price_cents == 45
    assert sim.levels_swept == 3

def test_walk_book_insufficient_depth():
    levels = [PriceLevel(price_cents=45, quantity=50)]
    sim = walk_book_depth(levels, 100, side="ask")

    assert sim.target_qty == 100
    assert sim.filled_qty == 50
    assert sim.is_fully_filled is False
    assert sim.total_cost_cents == 50 * 45
    assert sim.vwap_cents == 45.0
    assert sim.levels_swept == 1

def test_walk_bid_side_selling():
    # Bids descending: 50 @ 38c, 50 @ 36c
    levels = [
        PriceLevel(price_cents=38, quantity=50),
        PriceLevel(price_cents=36, quantity=50),
    ]

    sim = walk_book_depth(levels, 100, side="bid")
    assert sim.is_fully_filled is True
    # 50 * 38 + 50 * 36 = 3700 cents / 100 = 37c
    assert sim.vwap_cents == 37.0
    # Selling slippage = best_bid (38) - vwap (37) = 1.0c
    assert sim.slippage_cents == 1.0
    assert sim.marginal_price_cents == 36

def test_evaluate_depth_profile():
    book = BinaryMarketBook(
        ticker="TEST-DEPTH",
        bids=[PriceLevel(price_cents=44, quantity=100), PriceLevel(price_cents=42, quantity=200)],
        asks=[PriceLevel(price_cents=48, quantity=80), PriceLevel(price_cents=50, quantity=150)]
    )

    profile = evaluate_depth_profile(book, target_sizes=[50, 100, 200, 500])
    assert profile.ticker == "TEST-DEPTH"
    assert profile.total_bid_depth == 300
    assert profile.total_ask_depth == 230

    sim_50 = profile.ask_simulations[50]
    assert sim_50.is_fully_filled is True
    assert sim_50.vwap_cents == 48.0
    assert sim_50.slippage_cents == 0.0

    sim_100 = profile.ask_simulations[100]
    assert sim_100.is_fully_filled is True
    # 80 @ 48 + 20 @ 50 = 3840 + 1000 = 4840 / 100 = 48.4c
    assert pytest.approx(sim_100.vwap_cents, rel=1e-4) == 48.4
    assert pytest.approx(sim_100.slippage_cents, rel=1e-4) == 0.4

    sim_500 = profile.ask_simulations[500]
    assert sim_500.is_fully_filled is False
    assert sim_500.filled_qty == 230

def test_depth_adjusted_gross_edge_decay():
    # Book with escalating asks:
    # 50 @ 40c ($0.40)
    # 50 @ 46c ($0.46)
    # 100 @ 55c ($0.55)
    book = BinaryMarketBook(
        ticker="DECAY-TEST",
        bids=[PriceLevel(price_cents=35, quantity=100)],
        asks=[
            PriceLevel(price_cents=40, quantity=50),
            PriceLevel(price_cents=46, quantity=50),
            PriceLevel(price_cents=55, quantity=100),
        ]
    )

    fair_prob = 0.50  # 50c fair value

    # At size 50: VWAP is 40c ($0.40). Edge = 0.50 - 0.40 = +$0.10 (+10c edge)
    edge_50 = depth_adjusted_gross_edge(fair_prob, book, size=50)
    assert pytest.approx(edge_50, rel=1e-4) == 0.10

    # At size 100: VWAP is (50*40 + 50*46)/100 = 43c ($0.43). Edge = 0.50 - 0.43 = +$0.07 (+7c edge)
    edge_100 = depth_adjusted_gross_edge(fair_prob, book, size=100)
    assert pytest.approx(edge_100, rel=1e-4) == 0.07

    # At size 200: VWAP is (2000 + 2300 + 5500)/200 = 9800/200 = 49c ($0.49). Edge = 0.50 - 0.49 = +$0.01 (+1c edge)
    edge_200 = depth_adjusted_gross_edge(fair_prob, book, size=200)
    assert pytest.approx(edge_200, rel=1e-4) == 0.01

    # At size 500: Depth exhausted -> None
    edge_500 = depth_adjusted_gross_edge(fair_prob, book, size=500)
    assert edge_500 is None
