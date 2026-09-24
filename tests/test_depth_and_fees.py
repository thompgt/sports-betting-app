import pytest
from unittest.mock import MagicMock
from app.engine.fees import (
    KalshiFeeModel,
    evaluate_single_contract_mispricing,
    evaluate_basket_dutch_book_net
)
from app.engine.microstructure import PriceLevel, BinaryMarketBook
from app.engine.depth import walk_book_depth
from app.services.kalshi_detection_service import KalshiMispricingDetector
from app.storage.database import DatabaseManager
from app.storage.models import KalshiMispricing
from app.ingestion.kalshi_client import ParsedOrderbook, OrderbookLevel

def test_kalshi_taker_fee_schedule():
    fee_model = KalshiFeeModel(taker_rate=0.07)

    # 100 contracts at 50c ($0.50):
    # raw = 0.07 * 100 * 0.50 * 0.50 * 100 = 175 cents = $1.75
    fee_cents = fee_model.taker_fee_cents(contracts=100, price_dollars=0.50)
    assert fee_cents == 175
    assert fee_model.taker_fee_dollars(100, 0.50) == 1.75
    assert fee_model.fee_per_contract_cents(100, 0.50) == 1.75

    # 100 contracts at 10c ($0.10):
    # raw = 0.07 * 100 * 0.10 * 0.90 * 100 = 63 cents = $0.63
    assert fee_model.taker_fee_cents(100, 0.10) == 63
    assert fee_model.fee_per_contract_cents(100, 0.10) == 0.63

    # Symmetry: 100 contracts at 90c ($0.90):
    assert fee_model.taker_fee_cents(100, 0.90) == 63

    # Ceil ceiling behavior: fractional cent rounds up to integer cent
    # 1 contract at 50c: 0.07 * 1 * 0.25 * 100 = 1.75 cents -> ceil = 2 cents
    assert fee_model.taker_fee_cents(1, 0.50) == 2

def test_phantom_edge_rejection():
    # Price = 50c, Fair = 51c (Gross edge = +1.0c)
    # Taker fee at 50c = 1.75c
    # Net edge = 1.0c - 1.75c = -0.75c -> NOT profitable!
    mispricing = evaluate_single_contract_mispricing(
        ticker="PHANTOM-TICKER",
        fair_prob=0.51,
        vwap_cents=50.0,
        contracts=100
    )

    assert mispricing.gross_edge_cents == 1.0
    assert mispricing.fee_per_contract_cents == 1.75
    assert pytest.approx(mispricing.net_edge_cents, rel=1e-4) == -0.75
    assert mispricing.is_mathematically_profitable is False
    assert "No mathematical mispricing" in mispricing.rationale

def test_genuine_mathematical_mispricing():
    # Price = 50c, Fair = 55c (Gross edge = +5.0c)
    # Taker fee at 50c = 1.75c
    # Net edge = 5.0c - 1.75c = +3.25c -> PROFITABLE!
    mispricing = evaluate_single_contract_mispricing(
        ticker="GENUINE-TICKER",
        fair_prob=0.55,
        vwap_cents=50.0,
        contracts=100
    )

    assert pytest.approx(mispricing.gross_edge_cents, rel=1e-4) == 5.0
    assert mispricing.fee_per_contract_cents == 1.75
    assert pytest.approx(mispricing.net_edge_cents, rel=1e-4) == 3.25
    assert mispricing.is_mathematically_profitable is True
    assert pytest.approx(mispricing.net_ev_dollars, rel=1e-4) == 3.25
    assert "Mathematical mispricing detected" in mispricing.rationale

def test_depth_slippage_erodes_edge_across_tiers():
    # Orderbook with escalating asks:
    # 50 @ 46c
    # 50 @ 49c
    # 100 @ 55c
    levels = [
        PriceLevel(price_cents=46, quantity=50),
        PriceLevel(price_cents=49, quantity=50),
        PriceLevel(price_cents=55, quantity=100),
    ]

    fair_prob = 0.52  # 52c fair price

    # At tier 50: VWAP is 46c. Gross edge = 52 - 46 = +6.0c
    # Fee at 46c for 50 contracts:
    # 0.07 * 50 * 0.46 * 0.54 * 100 = 86.94 cents -> ceil = 87 cents (1.74c/contract)
    # Net edge = 6.0 - 1.74 = +4.26c -> PROFITABLE
    sim_50 = walk_book_depth(levels, 50, side="ask")
    m_50 = evaluate_single_contract_mispricing("TIER-TEST", fair_prob, sim_50.vwap_cents, 50)
    assert m_50.is_mathematically_profitable is True

    # At tier 100: VWAP is (50*46 + 50*49)/100 = 47.5c. Gross edge = 52 - 47.5 = +4.5c
    # Fee at 47.5c for 100 contracts:
    # 0.07 * 100 * 0.475 * 0.525 * 100 = 174.56 cents -> ceil = 175 cents (1.75c/contract)
    # Net edge = 4.5 - 1.75 = +2.75c -> STILL PROFITABLE
    sim_100 = walk_book_depth(levels, 100, side="ask")
    m_100 = evaluate_single_contract_mispricing("TIER-TEST", fair_prob, sim_100.vwap_cents, 100)
    assert m_100.is_mathematically_profitable is True

    # At tier 200: VWAP is (50*46 + 50*49 + 100*55)/200 = (2300 + 2450 + 5500)/200 = 51.25c
    # Gross edge = 52 - 51.25 = +0.75c
    # Fee at 51.25c for 200 contracts:
    # 0.07 * 200 * 0.5125 * 0.4875 * 100 = 349.77 cents -> ceil = 350 cents (1.75c/contract)
    # Net edge = 0.75 - 1.75 = -1.00c -> UNPROFITABLE!
    sim_200 = walk_book_depth(levels, 200, side="ask")
    m_200 = evaluate_single_contract_mispricing("TIER-TEST", fair_prob, sim_200.vwap_cents, 200)
    assert m_200.is_mathematically_profitable is False

def test_basket_dutch_book_net_of_fees():
    # 3 outcomes with total VWAPs = 92c
    # Guaranteed payout = 100c (Gross edge = 8c)
    outcome_vwaps = {
        "OUT-1": 30.0,
        "OUT-2": 30.0,
        "OUT-3": 32.0,
    }
    m = evaluate_basket_dutch_book_net("EVENT-BASKET", outcome_vwaps, contracts=100)
    assert m.gross_edge_cents == 8.0
    # Fees per contract:
    # OUT-1 (30c): 0.07 * 100 * 0.3 * 0.7 * 100 = 147c
    # OUT-2 (30c): 147c
    # OUT-3 (32c): 0.07 * 100 * 0.32 * 0.68 * 100 = 152.32 -> 153c
    # Total fees = 147 + 147 + 153 = 447c ($4.47) -> 4.47c per basket
    # Net edge = 8.0c - 4.47c = +3.53c per basket -> PROFITABLE!
    assert m.fee_cents_total == 447
    assert pytest.approx(m.net_edge_cents, rel=1e-4) == 3.53
    assert m.is_mathematically_profitable is True

def test_detector_records_mispricing_without_trading():
    db = DatabaseManager("sqlite:///:memory:")
    mock_client = MagicMock()
    mock_client.get_orderbook.return_value = ParsedOrderbook(
        ticker="TEST-DETECTOR",
        timestamp=1700000000.0,
        bids=[OrderbookLevel(price_cents=45, quantity=200)],
        asks=[OrderbookLevel(price_cents=48, quantity=200)],
    )

    detector = KalshiMispricingDetector(
        client=mock_client,
        db_manager=db,
        depth_tiers=[50, 100]
    )

    # Fair prob 55c > VWAP 48c -> gross edge 7c > fee 1.75c -> net mispricing
    detected = detector.scan_market("TEST-DETECTOR", fair_prob=0.55)
    assert len(detected) == 2

    # Verify audit persistence into database
    session = db.get_session()
    records = session.query(KalshiMispricing).filter_by(ticker="TEST-DETECTOR").all()
    assert len(records) == 2
    assert records[0].net_edge_cents > 0
    session.close()

    # Verify NO trade execution methods exist on detector
    assert not hasattr(detector, "place_order")
    assert not hasattr(detector, "execute_trade")
    assert not hasattr(detector, "submit_order")
