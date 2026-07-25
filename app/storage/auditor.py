from typing import Optional
from sqlalchemy.orm import Session
from app.storage.models import DetectedEdge
from app.engine.math_utils import decimal_to_implied_prob

class EdgeAuditor:
    def __init__(self, session: Session):
        self.session = session

    def close_out_market(self, game_id: str, market_type: str, bookmaker_name: str,
                         closing_odds: float, outcome_name: Optional[str] = None):
        """
        Updates historical edges for a specific game/market/bookmaker with closing
        line information and computes Closing Line Value (CLV). Used for CLV tracking.

        `outcome_name` scopes the closeout to a single side of the market; CLV is only
        meaningful when the closing price refers to the same outcome the edge was on.
        Passing None closes out every active edge on the market (legacy behaviour).

        clv_pct = implied(closing) - implied(offered). A positive value means the
        price we flagged was longer than the price the market closed at - i.e. we
        beat the closing line.
        """
        query = self.session.query(DetectedEdge).filter(
            DetectedEdge.canonical_game_id == game_id,
            DetectedEdge.market_type == market_type,
            DetectedEdge.bookmaker_name == bookmaker_name,
            DetectedEdge.is_active == True
        )
        if outcome_name is not None:
            query = query.filter(DetectedEdge.outcome_name == outcome_name)
        edges = query.all()

        for edge in edges:
            edge.is_active = False
            edge.closing_line = closing_odds
            edge.clv_pct = decimal_to_implied_prob(closing_odds) - decimal_to_implied_prob(edge.odds_offered)

        self.session.commit()
