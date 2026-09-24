import pytest
from unittest.mock import MagicMock
from app.storage.database import DatabaseManager
from app.storage.models import (
    Base,
    KalshiMarket,
    KalshiOrderbookSnapshot,
    KalshiOrderbookLevel,
    KalshiMispricing
)
from app.ingestion.kalshi_ingestor import KalshiIngestor
from app.ingestion.kalshi_client import KalshiRestClient

@pytest.fixture
def test_db():
    # Use SQLite in-memory database to test models and ingestion
    db = DatabaseManager("sqlite:///:memory:")
    return db

def test_database_schema_creation(test_db):
    session = test_db.get_session()
    # Confirm tables are accessible
    assert session.query(KalshiMarket).count() == 0
    assert session.query(KalshiOrderbookSnapshot).count() == 0
    assert session.query(KalshiOrderbookLevel).count() == 0
    assert session.query(KalshiMispricing).count() == 0
    session.close()

def test_ingest_markets_upsert(test_db):
    mock_client = MagicMock(spec=KalshiRestClient)
    mock_client.get_markets.return_value = {
        "markets": [
            {
                "ticker": "KXTEST-01",
                "event_ticker": "KXEVENT-01",
                "title": "Will Candidate A win?",
                "status": "open",
                "open_time": "2026-01-01T00:00:00Z",
                "close_time": "2026-12-31T23:59:59Z",
                "tick_size": 1,
                "last_price": 50,
                "volume": 1000
            },
            {
                "ticker": "KXTEST-02",
                "event_ticker": "KXEVENT-01",
                "title": "Will Candidate B win?",
                "status": "open",
                "tick_size": 1,
                "volume": 500
            }
        ]
    }

    ingestor = KalshiIngestor(mock_client, test_db)
    tickers = ingestor.ingest_markets()
    assert tickers == ["KXTEST-01", "KXTEST-02"]

    session = test_db.get_session()
    assert session.query(KalshiMarket).count() == 2
    m1 = session.query(KalshiMarket).filter_by(ticker="KXTEST-01").first()
    assert m1.event_ticker == "KXEVENT-01"
    assert m1.volume == 1000
    session.close()

    # Test idempotency and updates: update volume
    mock_client.get_markets.return_value["markets"][0]["volume"] = 1500
    mock_client.get_markets.return_value["markets"][0]["last_price"] = 55
    ingestor.ingest_markets()

    session2 = test_db.get_session()
    assert session2.query(KalshiMarket).count() == 2
    m1_updated = session2.query(KalshiMarket).filter_by(ticker="KXTEST-01").first()
    assert m1_updated.volume == 1500
    assert m1_updated.last_price_cents == 55
    session2.close()

def test_ingest_orderbook_snapshot_and_levels(test_db):
    mock_client = MagicMock(spec=KalshiRestClient)
    mock_client.get_raw_orderbook.return_value = {
        "orderbook_fp": {
            "yes_dollars": [["0.4500", "100.00"], ["0.4400", "200.00"]],
            "no_dollars": [["0.5100", "80.00"], ["0.5000", "120.00"]]
        }
    }
    # Use real parsing logic on the mock client
    mock_client.parse_orderbook_response = KalshiRestClient.parse_orderbook_response

    ingestor = KalshiIngestor(mock_client, test_db)
    snap_id = ingestor.ingest_orderbook("KXTEST-01")
    assert snap_id is not None

    session = test_db.get_session()
    snap = session.query(KalshiOrderbookSnapshot).filter_by(id=snap_id).first()
    assert snap.ticker == "KXTEST-01"
    assert snap.best_bid_cents == 45
    assert snap.best_ask_cents == 49
    assert snap.spread_cents == 4
    assert snap.mid_price_cents == 47.0
    assert snap.is_crossed is False
    assert snap.bid_depth_total == 300  # 100 + 200
    assert snap.ask_depth_total == 200  # 80 + 120

    levels = session.query(KalshiOrderbookLevel).filter_by(snapshot_id=snap_id).all()
    assert len(levels) == 4
    bids = [lvl for lvl in levels if lvl.side == "bid"]
    asks = [lvl for lvl in levels if lvl.side == "ask"]
    assert len(bids) == 2
    assert len(asks) == 2

    assert bids[0].price_cents == 45
    assert bids[0].quantity == 100
    assert asks[0].price_cents == 49
    assert asks[0].quantity == 80
    session.close()

def test_record_mispricing_model(test_db):
    session = test_db.get_session()
    mispricing = KalshiMispricing(
        ticker="KXTEST-MISPRICE",
        mispricing_type="single_contract_depth",
        fair_prob=0.55,
        depth_evaluated=100,
        effective_price_cents=49.0,
        fee_cents=1.72,
        gross_edge_cents=6.0,
        net_edge_cents=4.28,
        ev_pct=8.73,
        details_json='{"vwap": 49.0, "reason": "Ask below fair value"}'
    )
    session.add(mispricing)
    session.commit()

    saved = session.query(KalshiMispricing).filter_by(ticker="KXTEST-MISPRICE").first()
    assert saved is not None
    assert saved.gross_edge_cents == 6.0
    assert saved.net_edge_cents == 4.28
    assert saved.depth_evaluated == 100
    session.close()
