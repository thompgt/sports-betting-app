import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Tuple

from app.ingestion.mock_provider import MockOddsClient
from app.engine.resolver import EntityResolver, TeamCanonical, GameCanonical
from app.engine.math_utils import (
    american_to_decimal, 
    decimal_to_implied_prob, 
    strip_vig_power_method, 
    calculate_ev
)
from app.storage.database import DatabaseManager
from app.storage.models import DetectedEdge

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EdgeCache:
    def __init__(self, ttl_minutes: int = 15, ev_spike_threshold: float = 0.02):
        self.cache: Dict[Tuple[str, str, str], Tuple[datetime, float]] = {}
        self.ttl = timedelta(minutes=ttl_minutes)
        self.ev_spike_threshold = ev_spike_threshold

    def should_record(self, game_id: str, market: str, bookmaker: str, current_ev: float) -> bool:
        key = (game_id, market, bookmaker)
        now = datetime.utcnow()
        
        if key not in self.cache:
            self.cache[key] = (now, current_ev)
            return True
            
        last_time, last_ev = self.cache[key]
        
        # Check TTL
        if now - last_time > self.ttl:
            self.cache[key] = (now, current_ev)
            return True
            
        # Check EV Spike
        if abs(current_ev - last_ev) > self.ev_spike_threshold:
            self.cache[key] = (now, current_ev)
            return True
            
        return False

async def main():
    # 1. Setup Resolver with some canonical data matching our mock
    teams = [
        TeamCanonical(id="c2e4b962-2b1b-41c6-a992-47b9ec9e76c6", name="New York Rangers", aliases=["NY Rangers"]),
        TeamCanonical(id="d2222222-2222-2222-2222-222222222222", name="Los Angeles Lakers", aliases=["Lakers", "LA Lakers"]),
        TeamCanonical(id="d3333333-3333-3333-3333-333333333333", name="NY Knicks", aliases=["Knicks"]),
        TeamCanonical(id="d4444444-4444-4444-4444-444444444444", name="Boston Celtics", aliases=["Celtics"]),
    ]
    games = [
        GameCanonical(
            id="e1111111-1111-1111-1111-111111111111", 
            home_team_id="c2e4b962-2b1b-41c6-a992-47b9ec9e76c6", 
            away_team_id="d2222222-2222-2222-2222-222222222222", 
            start_time=datetime(2026, 5, 27, 20, 0, 0)
        ),
        GameCanonical(
            id="e2222222-2222-2222-2222-222222222222", 
            home_team_id="d3333333-3333-3333-3333-333333333333", 
            away_team_id="d4444444-4444-4444-4444-444444444444", 
            start_time=datetime(2026, 5, 27, 22, 0, 0)
        )
    ]
    resolver = EntityResolver(teams, games)
    
    # 2. Setup Database and Cache
    db_manager = DatabaseManager("sqlite:///./mock_simulation.db")
    edge_cache = EdgeCache()
    
    # 3. Setup Mock Client
    client = MockOddsClient("tests/fixtures/sample_odds_payload.json")
    
    logger.info("Starting End-to-End Mock Simulation...")
    
    async for event in client.stream_live_odds(polling_interval=0.5):
        logger.info(f"Processing Event: {event.home_team} vs {event.away_team}")
        
        # Resolve Game
        game_id = resolver.resolve_game(event.home_team, event.away_team, event.commence_time)
        if not game_id:
            logger.warning(f"Could not resolve game: {event.home_team} vs {event.away_team}")
            continue
            
        # For each market, calculate fair odds (using multi-way devigging across all books)
        # Note: In a real app, you'd aggregate all bookmaker lines for a market first.
        for market_key in ["h2h"]:
            all_outcomes_probs = []
            bookie_lines = []
            
            for bookmaker in event.bookmakers:
                market = next((m for m in bookmaker.markets if m.key == market_key), None)
                if not market: continue
                
                # Convert to decimal and implied prob
                try:
                    outcomes = {o.name: american_to_decimal(o.price) for o in market.outcomes}
                    implied_probs = [decimal_to_implied_prob(p) for p in outcomes.values()]
                    
                    # Store lines for later EV check
                    bookie_lines.append((bookmaker.title, outcomes))
                    
                    # Add to aggregate for "market fair price" estimation
                    all_outcomes_probs.append(implied_probs)
                except Exception as e:
                    logger.error(f"Error processing odds for {bookmaker.title}: {e}")

            if not all_outcomes_probs: continue
            
            # Simple average of implied probabilities across books as a proxy for "market consensus"
            # In a pro system, you'd use a more complex weighting.
            avg_probs = [sum(p[i] for p in all_outcomes_probs) / len(all_outcomes_probs) for i in range(len(all_outcomes_probs[0]))]
            
            try:
                fair_probs = strip_vig_power_method(avg_probs)
                fair_decimals = [1/p for p in fair_probs]
                
                # Check each bookmaker for EV+ opportunities
                for bookie_name, outcomes in bookie_lines:
                    for i, (outcome_name, price) in enumerate(outcomes.items()):
                        ev = calculate_ev(price, fair_probs[i])
                        
                        if ev > 0.02: # Lower threshold to 2% for simulation visibility
                            logger.info(f"!!! EDGE DETECTED !!! {bookie_name} | {outcome_name} | Price: {price:.2f} | Fair: {fair_decimals[i]:.2f} | EV: {ev:.2%}")
                            
                            if edge_cache.should_record(str(game_id), market_key, bookie_name, ev):
                                session = db_manager.get_session()
                                edge_record = DetectedEdge(
                                    canonical_game_id=str(game_id),
                                    sport=event.sport_key,
                                    market_type=market_key,
                                    bookmaker_name=bookie_name,
                                    odds_offered=price,
                                    fair_odds=fair_decimals[i],
                                    calculated_ev=ev
                                )
                                session.add(edge_record)
                                session.commit()
                                logger.info(f"Saved edge to database.")
                            else:
                                logger.info("Edge already in cache and stable, skipping DB write.")
            except Exception as e:
                logger.error(f"Math Error: {e}")

    logger.info("Simulation Complete.")

if __name__ == "__main__":
    asyncio.run(main())
