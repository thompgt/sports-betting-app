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
    auditor.close_out_market("game_abc", "h2h", "FanDuel", 1.88)

    updated_edge = session.query(DetectedEdge).filter_by(canonical_game_id="game_abc").first()
    assert updated_edge.is_active is False
    assert updated_edge.closing_line == 1.88
    assert updated_edge.clv_pct is not None
    # took 1.9, market closed at 1.88 -> we got the longer price -> positive CLV
    assert updated_edge.clv_pct > 0
    assert updated_edge.clv_pct == pytest.approx(1 / 1.88 - 1 / 1.9)

def test_edge_auditor_negative_clv(db_manager):
    session = db_manager.get_session()
    session.add(DetectedEdge(
        canonical_game_id="game_neg", sport="NBA", market_type="h2h",
        bookmaker_name="FanDuel", outcome_name="Boston Celtics",
        odds_offered=1.9, fair_odds=1.85, calculated_ev=0.02, is_active=True,
    ))
    session.commit()

    EdgeAuditor(session).close_out_market("game_neg", "h2h", "FanDuel", 2.05)
    edge = session.query(DetectedEdge).filter_by(canonical_game_id="game_neg").one()
    # market drifted out to 2.05 after we flagged 1.9 -> we lost to the close
    assert edge.clv_pct < 0

def test_edge_auditor_scopes_closeout_to_outcome(db_manager):
    """CLV is only meaningful when the closing price is for the same side."""
    session = db_manager.get_session()
    for outcome, price in [("Home", 1.9), ("Away", 2.1)]:
        session.add(DetectedEdge(
            canonical_game_id="game_two_sided", sport="NHL", market_type="h2h",
            bookmaker_name="BetMGM", outcome_name=outcome,
            odds_offered=price, fair_odds=2.0, calculated_ev=0.03, is_active=True,
        ))
    session.commit()

    EdgeAuditor(session).close_out_market("game_two_sided", "h2h", "BetMGM", 1.95, outcome_name="Home")

    home = session.query(DetectedEdge).filter_by(outcome_name="Home").one()
    away = session.query(DetectedEdge).filter_by(outcome_name="Away").one()
    assert home.is_active is False and home.closing_line == 1.95
    assert away.is_active is True and away.closing_line is None
