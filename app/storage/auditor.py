from sqlalchemy.orm import Session
from app.storage.models import DetectedEdge

class EdgeAuditor:
    def __init__(self, session: Session):
        self.session = session

    def close_out_market(self, game_id: str, final_closing_odds: float):
        """
        Updates historical edges for a game with closing line information.
        Used for CLV (Closing Line Value) tracking.
        """
        edges = self.session.query(DetectedEdge).filter(
            DetectedEdge.canonical_game_id == game_id,
            DetectedEdge.is_active == True
        ).all()
        
        for edge in edges:
            # Mark as inactive since the market is closed
            edge.is_active = False
            
        self.session.commit()
