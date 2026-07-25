import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Tuple, Set

from app.core.config import Settings
from app.ingestion.base import OddsProvider
from app.engine.resolver import EntityResolver
from app.engine.math_utils import (
    american_to_decimal,
    decimal_to_implied_prob,
    strip_vig_power_method,
    calculate_ev
)
from app.storage.database import DatabaseManager
from app.storage.models import DetectedEdge
from app.storage.auditor import EdgeAuditor

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

        if now - last_time > self.ttl:
            self.cache[key] = (now, current_ev)
            return True

        if abs(current_ev - last_ev) > self.ev_spike_threshold:
            self.cache[key] = (now, current_ev)
            return True

        return False

class EdgeDetectionService:
    """
    Continuously polls an OddsProvider on an interval, resolves entities,
    devigs the market, detects +EV edges, persists them, and closes out
    markets for games that have started (recording CLV). Analogous to a
    trading bot's poll/decide loop, minus order execution.
    """

    def __init__(self, provider: OddsProvider, resolver: EntityResolver,
                 db_manager: DatabaseManager, settings: Settings):
        self.provider = provider
        self.resolver = resolver
        self.db_manager = db_manager
        self.settings = settings
        self.edge_cache = EdgeCache(settings.edge_cache_ttl_minutes, settings.edge_cache_spike_threshold)
        # (game_id, market, bookmaker, outcome) -> most recently seen decimal price.
        # Keyed per outcome so the closing line compared against an edge is the price
        # of the same side of the market.
        self._last_odds_seen: Dict[Tuple[str, str, str, str], float] = {}
        self._closed_games: Set[str] = set()

    async def run_forever(self) -> None:
        backoff = 1.0
        while True:
            try:
                await self.poll_once()
                backoff = 1.0
            except Exception:
                logger.exception("Poll cycle failed; backing off %.0fs", backoff)
                await asyncio.sleep(min(backoff, self.settings.max_backoff_seconds))
                backoff = min(backoff * 2, self.settings.max_backoff_seconds)
                continue
            await asyncio.sleep(self.settings.poll_interval_seconds)

    async def poll_once(self) -> None:
        events = await self.provider.get_odds()
        logger.info("Polled %d events", len(events))
        session = self.db_manager.get_session()
        try:
            for event in events:
                self._process_event(event, session)
            self._run_clv_closeouts(session)
        finally:
            session.close()

    def _process_event(self, event, session) -> None:
        logger.info("Processing Event: %s vs %s", event.home_team, event.away_team)

        game_id = self.resolver.resolve_game(event.home_team, event.away_team, event.commence_time)
        if not game_id:
            logger.warning("Could not resolve game: %s vs %s", event.home_team, event.away_team)
            return

        for market_key in ["h2h"]:
            all_outcomes_probs = []
            bookie_lines = []

            for bookmaker in event.bookmakers:
                market = next((m for m in bookmaker.markets if m.key == market_key), None)
                if not market:
                    continue

                try:
                    outcomes = {o.name: american_to_decimal(o.price) for o in market.outcomes}
                    implied_probs = [decimal_to_implied_prob(p) for p in outcomes.values()]

                    bookie_lines.append((bookmaker.title, outcomes))
                    all_outcomes_probs.append(implied_probs)

                    for outcome_name, price in outcomes.items():
                        self._last_odds_seen[(str(game_id), market_key, bookmaker.title, outcome_name)] = price
                except Exception as e:
                    logger.error("Error processing odds for %s: %s", bookmaker.title, e)

            if not all_outcomes_probs:
                continue

            avg_probs = [sum(p[i] for p in all_outcomes_probs) / len(all_outcomes_probs) for i in range(len(all_outcomes_probs[0]))]

            try:
                fair_probs = strip_vig_power_method(avg_probs)
                fair_decimals = [1 / p for p in fair_probs]

                for bookie_name, outcomes in bookie_lines:
                    for i, (outcome_name, price) in enumerate(outcomes.items()):
                        ev = calculate_ev(price, fair_probs[i])

                        if ev > self.settings.ev_threshold:
                            logger.info(
                                "!!! EDGE DETECTED !!! %s | %s | Price: %.2f | Fair: %.2f | EV: %.2f%%",
                                bookie_name, outcome_name, price, fair_decimals[i], ev * 100
                            )

                            if self.edge_cache.should_record(str(game_id), market_key, bookie_name, ev):
                                edge_record = DetectedEdge(
                                    canonical_game_id=str(game_id),
                                    sport=event.sport_key,
                                    market_type=market_key,
                                    bookmaker_name=bookie_name,
                                    outcome_name=outcome_name,
                                    odds_offered=price,
                                    fair_odds=fair_decimals[i],
                                    calculated_ev=ev
                                )
                                session.add(edge_record)
                                session.commit()
                                logger.info("Saved edge to database.")
                            else:
                                logger.info("Edge already in cache and stable, skipping DB write.")
            except Exception as e:
                logger.error("Math Error: %s", e)

    def _run_clv_closeouts(self, session) -> None:
        auditor = EdgeAuditor(session)
        now = datetime.utcnow()
        for game in self.resolver.games:
            game_id = str(game.id)
            if game_id in self._closed_games or game.start_time > now:
                continue

            for (seen_game_id, market, bookie, outcome), price in self._last_odds_seen.items():
                if seen_game_id == game_id:
                    auditor.close_out_market(game_id, market, bookie, price, outcome_name=outcome)

            self._closed_games.add(game_id)
