from sqlalchemy.orm import Session
from app.storage.models import DetectedEdge
from app.engine.math_utils import decimal_to_implied_prob

class EdgeAuditor:
    def __init__(self, session: Session):
        self.session = session

    def close_out_market(self, game_id: str, market_type: str, bookmaker_name: str, closing_odds: float):
        """
        Updates historical edges for a specific game/market/bookmaker with closing
        line information and computes Closing Line Value (CLV). Used for CLV tracking.

        A positive clv_pct means the odds we bet implied a lower probability than the
        closing line settled at - i.e. we beat the closing line.
        """
        edges = self.session.query(DetectedEdge).filter(
            DetectedEdge.canonical_game_id == game_id,
            DetectedEdge.market_type == market_type,
            DetectedEdge.bookmaker_name == bookmaker_name,
            DetectedEdge.is_active == True
        ).all()

        for edge in edges:
            edge.is_active = False
            edge.closing_line = closing_odds
            edge.clv_pct = decimal_to_implied_prob(edge.odds_offered) - decimal_to_implied_prob(closing_odds)

        self.session.commit()
