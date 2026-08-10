import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Set

from app.core.config import Settings
from app.ingestion.base import OddsProvider
from app.engine.resolver import EntityResolver
from app.engine.math_utils import (
    american_to_decimal,
    decimal_to_implied_prob,
    strip_vig_power_method,
    pool_log_odds,
    calculate_ev
)
from app.storage.database import DatabaseManager
from app.storage.models import DetectedEdge
from app.storage.auditor import EdgeAuditor

logger = logging.getLogger(__name__)

def utc_now() -> datetime:
    """
    Timezone-aware UTC now. datetime.utcnow() is deprecated and, worse, returns a
    naive datetime that compares fine against other naive datetimes right up
    until a tz-aware one arrives from a feed - at which point the comparison
    raises TypeError.
    """
    return datetime.now(timezone.utc)

def as_utc(value: datetime) -> datetime:
    """
    Coerces a datetime to tz-aware UTC. Naive values are assumed to already be
    UTC, which is what every naive timestamp in this codebase means; aware
    values are converted. This keeps a feed or database that yields either kind
    from blowing up a comparison.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)

class EdgeCache:
    def __init__(self, ttl_minutes: int = 15, ev_spike_threshold: float = 0.02):
        self.cache: Dict[Tuple[str, str, str], Tuple[datetime, float]] = {}
        self.ttl = timedelta(minutes=ttl_minutes)
        self.ev_spike_threshold = ev_spike_threshold

    def should_record(self, game_id: str, market: str, bookmaker: str, current_ev: float) -> bool:
        key = (game_id, market, bookmaker)
        now = utc_now()

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

    def purge_expired(self) -> int:
        """
        Drops entries past their TTL. An expired entry can never suppress a write
        again - should_record() re-records on any entry older than the TTL - so it
        is pure memory growth in a process meant to run unattended for weeks.
        Returns the number of entries evicted.
        """
        cutoff = utc_now() - self.ttl
        stale = [k for k, (seen_at, _) in self.cache.items() if seen_at < cutoff]
        for k in stale:
            del self.cache[k]
        return len(stale)

    def forget_game(self, game_id: str) -> None:
        """Drops every entry for a game that will never be quoted again."""
        for k in [k for k in self.cache if k[0] == game_id]:
            del self.cache[k]

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
        # Games closed out in THIS process. Not the durable record - that is the
        # is_active flag in the database, which is what survives a restart.
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

            # Everything below is bounded-memory housekeeping. Without it the
            # three in-memory structures below only ever grow, and this service
            # is documented as running unattended.
            evicted = self.edge_cache.purge_expired()
            if evicted:
                logger.debug("Evicted %d expired edge-cache entries", evicted)
        finally:
            session.close()

    def _process_event(self, event, session) -> None:
        logger.info("Processing Event: %s vs %s", event.home_team, event.away_team)

        game_id = self.resolver.resolve_game(event.home_team, event.away_team, event.commence_time)
        if not game_id:
            logger.warning("Could not resolve game: %s vs %s", event.home_team, event.away_team)
            return

        for market_key in ["h2h"]:
            # (bookmaker title, {outcome name -> decimal price}, {outcome name -> fair prob})
            bookie_lines: List[Tuple[str, Dict[str, float], Dict[str, float]]] = []

            for bookmaker in event.bookmakers:
                market = next((m for m in bookmaker.markets if m.key == market_key), None)
                if not market:
                    continue

                try:
                    outcomes = {o.name: american_to_decimal(o.price) for o in market.outcomes}
                    if len(outcomes) != len(market.outcomes):
                        raise ValueError("Duplicate outcome names in market")

                    names = list(outcomes.keys())
                    implied_probs = [decimal_to_implied_prob(outcomes[n]) for n in names]

                    # Devig each book against its OWN overround before it joins the
                    # consensus. Averaging vigged probabilities and devigging the
                    # average strips a margin no book actually quoted, so a book with
                    # a fat margin drags every other book's fair price with it.
                    fair_by_name = dict(zip(names, strip_vig_power_method(implied_probs)))

                    bookie_lines.append((bookmaker.title, outcomes, fair_by_name))

                    for outcome_name, price in outcomes.items():
                        self._last_odds_seen[(str(game_id), market_key, bookmaker.title, outcome_name)] = price
                except Exception as e:
                    logger.error("Error processing odds for %s: %s", bookmaker.title, e)

            if not bookie_lines:
                continue

            # Outcomes are matched by NAME, never by position. Feeds list outcomes in
            # whatever order they please; indexing positionally pairs one book's home
            # fair probability with another book's away price and manufactures a large
            # phantom edge. Books quoting a different outcome set entirely (e.g. a
            # three-way line with a draw against two-way moneylines) are not
            # comparable and are dropped rather than mis-indexed.
            consensus_names = self._consensus_outcome_set(bookie_lines)
            if not consensus_names:
                continue

            comparable = [
                line for line in bookie_lines if set(line[2].keys()) == set(consensus_names)
            ]
            dropped = [line[0] for line in bookie_lines if line not in comparable]
            if dropped:
                logger.warning(
                    "Excluding %s from the %s consensus: outcome set differs from %s",
                    ", ".join(dropped), market_key, sorted(consensus_names)
                )

            try:
                # Pool the already-fair lines in log-odds space, mirroring
                # consensus() in crates/edge-core/src/consensus.rs.
                pooled = pool_log_odds(
                    [[fair_by_name[n] for n in consensus_names] for _, _, fair_by_name in comparable]
                )
                fair_probs = dict(zip(consensus_names, pooled))
                fair_decimals = {n: 1 / p for n, p in fair_probs.items()}

                for bookie_name, outcomes, _ in comparable:
                    for outcome_name, price in outcomes.items():
                        ev = calculate_ev(price, fair_probs[outcome_name])

                        if ev > self.settings.ev_threshold:
                            logger.info(
                                "!!! EDGE DETECTED !!! %s | %s | Price: %.2f | Fair: %.2f | EV: %.2f%%",
                                bookie_name, outcome_name, price, fair_decimals[outcome_name], ev * 100
                            )

                            if self.edge_cache.should_record(str(game_id), market_key, bookie_name, ev):
                                edge_record = DetectedEdge(
                                    canonical_game_id=str(game_id),
                                    sport=event.sport_key,
                                    market_type=market_key,
                                    bookmaker_name=bookie_name,
                                    outcome_name=outcome_name,
                                    odds_offered=price,
                                    fair_odds=fair_decimals[outcome_name],
                                    calculated_ev=ev
                                )
                                session.add(edge_record)
                                session.commit()
                                logger.info("Saved edge to database.")
                            else:
                                logger.info("Edge already in cache and stable, skipping DB write.")
            except Exception as e:
                logger.error("Math Error: %s", e)

    @staticmethod
    def _consensus_outcome_set(bookie_lines) -> List[str]:
        """
        Picks the outcome set the consensus is built on: the one the most books agree
        on, ties broken by whichever appeared first. Returns the names in that book's
        listing order, which is only used to keep the pooled vectors aligned - every
        lookup downstream is by name.
        """
        counts: Dict[frozenset, int] = {}
        order: Dict[frozenset, List[str]] = {}
        for _, _, fair_by_name in bookie_lines:
            key = frozenset(fair_by_name.keys())
            counts[key] = counts.get(key, 0) + 1
            order.setdefault(key, list(fair_by_name.keys()))

        if not counts:
            return []
        best = max(counts, key=lambda k: counts[k])
        return order[best]

    def _run_clv_closeouts(self, session) -> None:
        auditor = EdgeAuditor(session)
        now = utc_now()
        for game in self.resolver.games:
            game_id = str(game.id)
            # as_utc() on both sides: a single tz-aware kickoff time used to raise
            # TypeError here, and the caller's blanket `except` swallowed it -
            # silently aborting every CLV closeout for the rest of the run.
            if game_id in self._closed_games or as_utc(game.start_time) > now:
                continue

            # A restart empties _closed_games, which used to mean every started
            # game was closed out again - overwriting closing lines and CLV that
            # were already recorded. The is_active flag is the durable record of
            # whether a game still has anything to close, so consult it.
            if self._has_active_edges(session, game_id):
                for (seen_game_id, market, bookie, outcome), price in self._last_odds_seen.items():
                    if seen_game_id == game_id:
                        auditor.close_out_market(game_id, market, bookie, price, outcome_name=outcome)

            self._closed_games.add(game_id)
            self._forget_game(game_id)

    @staticmethod
    def _has_active_edges(session, game_id: str) -> bool:
        return session.query(DetectedEdge).filter(
            DetectedEdge.canonical_game_id == game_id,
            DetectedEdge.is_active == True
        ).first() is not None

    def _forget_game(self, game_id: str) -> None:
        """
        Releases the per-game working state once a game is closed out. Its prices
        are never quoted again, so retaining them only grows the process. The
        closing lines themselves are already durable in the database.
        """
        for key in [k for k in self._last_odds_seen if k[0] == game_id]:
            del self._last_odds_seen[key]
        self.edge_cache.forget_game(game_id)
