import pytest
from datetime import datetime, timedelta
from app.storage.database import DatabaseManager
from app.storage.models import DetectedEdge
from app.storage.auditor import EdgeAuditor

@pytest.fixture
def db_manager():
    return DatabaseManager("sqlite:///:memory:")

def test_edge_insertion(db_manager):
    session = db_manager.get_session()
    edge = DetectedEdge(
        canonical_game_id="game_123",
        sport="NHL",
        market_type="h2h",
        bookmaker_name="DraftKings",
        odds_offered=2.1,
        fair_odds=2.0,
        calculated_ev=0.05
    )
    session.add(edge)
    session.commit()
    
    saved_edge = session.query(DetectedEdge).first()
    assert saved_edge.canonical_game_id == "game_123"
    assert saved_edge.calculated_ev == 0.05

def test_edge_auditor_close_out(db_manager):
    session = db_manager.get_session()
    edge = DetectedEdge(
        canonical_game_id="game_abc",
        sport="NBA",
        market_type="h2h",
        bookmaker_name="FanDuel",
        odds_offered=1.9,
        fair_odds=1.85,
        calculated_ev=0.02,
        is_active=True
    )
    session.add(edge)
    session.commit()
    
    auditor = EdgeAuditor(session)
    auditor.close_out_market("game_abc", 1.88)
    
    updated_edge = session.query(DetectedEdge).filter_by(canonical_game_id="game_abc").first()
    assert updated_edge.is_active is False
