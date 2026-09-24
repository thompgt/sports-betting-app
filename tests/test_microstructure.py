import pytest
from app.engine.microstructure import (
    reflect_no_bid_to_yes_ask,
    reflect_yes_bid_to_no_ask,
    PriceLevel,
    BinaryMarketBook,
    evaluate_multi_outcome_parity,
    MIN_PRICE_CENTS,
    MAX_PRICE_CENTS
)

def test_no_bid_to_yes_ask_reflection():
    # A bid for NO at 51c is an offer to sell YES at 49c
    assert reflect_no_bid_to_yes_ask(51) == 49
    assert reflect_no_bid_to_yes_ask(1) == 99
    assert reflect_no_bid_to_yes_ask(99) == 1

    # Bounds: 0 and 100 or out-of-range should be rejected
    assert reflect_no_bid_to_yes_ask(0) is None
    assert reflect_no_bid_to_yes_ask(100) is None
    assert reflect_no_bid_to_yes_ask(-5) is None
    assert reflect_no_bid_to_yes_ask(105) is None

def test_yes_bid_to_no_ask_reflection():
    assert reflect_yes_bid_to_no_ask(45) == 55
    assert reflect_yes_bid_to_no_ask(0) is None
    assert reflect_yes_bid_to_no_ask(100) is None

def test_binary_market_book_construction():
    # Raw YES bids: 45c (100 qty), 44c (250 qty)
    # Raw NO bids: 51c (80 qty) -> YES ask 49c (80 qty), 50c (120 qty) -> YES ask 50c (120 qty)
    yes_bids = [(45, 100), (44, 250)]
    no_bids = [(51, 80), (50, 120)]

    book = BinaryMarketBook.from_raw_stacks("TEST-TICKER", yes_bids, no_bids)

    assert book.best_bid == 45
    assert book.best_bid_qty == 100
    assert book.best_ask == 49
    assert book.best_ask_qty == 80
    assert book.spread == 4
    assert not book.is_crossed
    assert not book.is_one_sided
    assert book.is_tradable
    assert book.mid_price == 47.0

def test_crossed_book_rejection():
    # YES bid at 52c, NO bid at 50c -> YES ask 50c
    # Bid (52c) > Ask (50c) -> CROSSED!
    yes_bids = [(52, 100)]
    no_bids = [(50, 100)]

    book = BinaryMarketBook.from_raw_stacks("CROSSED-TICKER", yes_bids, no_bids)
    assert book.is_crossed
    assert not book.is_tradable
    assert book.spread == -2
    assert book.mid_price is None

def test_one_sided_book_handling():
    # Only bids, no asks
    book_bids_only = BinaryMarketBook.from_raw_stacks("BIDS-ONLY", [(45, 100)], [])
    assert book_bids_only.is_one_sided
    assert not book_bids_only.is_tradable
    assert book_bids_only.best_bid == 45
    assert book_bids_only.best_ask is None
    assert book_bids_only.mid_price is None

    # Only asks (from NO bids)
    book_asks_only = BinaryMarketBook.from_raw_stacks("ASKS-ONLY", [], [(51, 100)])
    assert book_asks_only.is_one_sided
    assert not book_asks_only.is_tradable
    assert book_asks_only.best_bid is None
    assert book_asks_only.best_ask == 49
    assert book_asks_only.mid_price is None

def test_micro_price_and_order_flow_imbalance():
    # Symmetric depth: 100 bid @ 45c, 100 ask @ 49c
    book_sym = BinaryMarketBook.from_raw_stacks("SYM", [(45, 100)], [(51, 100)])
    assert book_sym.mid_price == 47.0
    assert book_sym.micro_price == 47.0
    assert book_sym.order_flow_imbalance == 0.0

    # Heavy bid queue: 900 bid @ 45c, 100 ask @ 49c
    # Micro-price should skew towards 49c: (900 * 49 + 100 * 45) / 1000 = (44100 + 4500) / 1000 = 48.6c
    book_heavy_bid = BinaryMarketBook.from_raw_stacks("HEAVY-BID", [(45, 900)], [(51, 100)])
    assert book_heavy_bid.mid_price == 47.0
    assert pytest.approx(book_heavy_bid.micro_price, rel=1e-3) == 48.6
    assert pytest.approx(book_heavy_bid.order_flow_imbalance, rel=1e-3) == 0.8  # (900 - 100) / 1000

    # Heavy ask queue: 100 bid @ 45c, 900 ask @ 49c
    # Micro-price should skew towards 45c: (100 * 49 + 900 * 45) / 1000 = (4900 + 40500) / 1000 = 45.4c
    book_heavy_ask = BinaryMarketBook.from_raw_stacks("HEAVY-ASK", [(45, 100)], [(51, 900)])
    assert pytest.approx(book_heavy_ask.micro_price, rel=1e-3) == 45.4
    assert pytest.approx(book_heavy_ask.order_flow_imbalance, rel=1e-3) == -0.8

def test_multi_outcome_basket_parity():
    # 3-outcome event: Candidate A, Candidate B, Candidate C
    # Underpriced Dutch book scenario:
    # A: ask 30c (from NO bid 70c), bid 28c
    # B: ask 32c (from NO bid 68c), bid 30c
    # C: ask 35c (from NO bid 65c), bid 33c
    # Sum of asks = 30 + 32 + 35 = 97c (< 100c) -> 3c guaranteed mathematical edge!
    book_a = BinaryMarketBook.from_raw_stacks("CAND-A", [(28, 100)], [(70, 100)])
    book_b = BinaryMarketBook.from_raw_stacks("CAND-B", [(30, 100)], [(68, 100)])
    book_c = BinaryMarketBook.from_raw_stacks("CAND-C", [(33, 100)], [(65, 100)])

    report = evaluate_multi_outcome_parity(
        "EVENT-ELEC",
        {"CAND-A": book_a, "CAND-B": book_b, "CAND-C": book_c}
    )
    assert report is not None
    assert report.ask_sum_cents == 97
    assert report.bid_sum_cents == 91
    assert report.is_dutch_book_underpriced is True
    assert report.underpricing_edge_cents == 3
    assert report.is_overpriced is False
    assert report.overpricing_edge_cents == 0

    # Fair probabilities sum to 1.0
    total_fair = sum(report.fair_probabilities.values())
    assert pytest.approx(total_fair, rel=1e-6) == 1.0
