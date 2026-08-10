"""
Regression tests for the consensus/edge-detection path in EdgeDetectionService.

These cover two arithmetic bugs that are invisible in the happy path and produce
plausible-looking but wrong fair prices:

  1. Pooling vigged probabilities across books and devigging the average, rather
     than devigging each book and pooling the fair prices.
  2. Pairing fair probabilities with prices positionally, so a book that lists
     its outcomes in a different order gets its home fair price judged against
     its away offer.
"""

import math
from datetime import datetime, timedelta, timezone
from typing import List, Tuple
from uuid import uuid4

import pytest

from app.core.config import Settings
from app.engine.math_utils import (
    american_to_decimal,
    decimal_to_implied_prob,
    pool_log_odds,
    strip_vig_power_method,
)
from app.models.schemas.odds import Bookmaker, Market, OddsEvent, Outcome
from app.services.detection_service import EdgeDetectionService


class _StubResolver:
    """Resolves every event to one fixed game id; games list is empty so no closeouts."""

    def __init__(self):
        self.game_id = uuid4()
        self.games = []

    def resolve_game(self, home_team, away_team, commence_time):
        return self.game_id


class _StubSession:
    """Collects the DetectedEdge rows the service tries to persist."""

    def __init__(self):
        self.added = []

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        pass

    def close(self):
        pass


def _service() -> Tuple[EdgeDetectionService, _StubResolver]:
    resolver = _StubResolver()
    settings = Settings(ev_threshold=-1.0)  # record every outcome, so we can read fair prices back
    svc = EdgeDetectionService(
        provider=None, resolver=resolver, db_manager=None, settings=settings
    )
    return svc, resolver


def _event(books: List[Tuple[str, List[Tuple[str, int]]]]) -> OddsEvent:
    """books = [(title, [(outcome_name, american_price), ...]), ...]"""
    return OddsEvent(
        id="evt-1",
        sport_key="basketball_nba",
        commence_time=datetime.now(timezone.utc) + timedelta(hours=6),
        home_team="Boston Celtics",
        away_team="Miami Heat",
        bookmakers=[
            Bookmaker(
                key=title.lower(),
                title=title,
                last_update=datetime.now(timezone.utc),
                markets=[
                    Market(
                        key="h2h",
                        outcomes=[Outcome(name=n, price=p) for n, p in outcomes],
                    )
                ],
            )
            for title, outcomes in books
        ],
    )


def _fair_by_outcome(session: _StubSession, bookmaker: str) -> dict:
    return {
        e.outcome_name: e.fair_odds for e in session.added if e.bookmaker_name == bookmaker
    }


def test_pool_log_odds_is_the_logit_mean_and_sums_to_one():
    pooled = pool_log_odds([[0.5, 0.5], [0.8, 0.2]])
    assert math.isclose(sum(pooled), 1.0, abs_tol=1e-12)
    # Logit mean of 0.5 and 0.8 is sigmoid((0 + log 4)/2) = 2/3 before renormalising;
    # the complement pools to 1/3, so the pair already sums to 1 and survives intact.
    assert math.isclose(pooled[0], 2.0 / 3.0, rel_tol=1e-9)
    # A logit pool is not an arithmetic mean.
    assert not math.isclose(pooled[0], 0.65, rel_tol=1e-3)

    with pytest.raises(ValueError):
        pool_log_odds([])
    with pytest.raises(ValueError):
        pool_log_odds([[0.5, 0.5], [0.4, 0.3, 0.3]])


def test_books_are_devigged_before_they_are_pooled():
    """
    Two books that disagree, quoting very different margins. Pooling first mixes
    the fat-margin book's overround into the thin-margin book's price; the
    consensus must instead be the pool of each book's own fair line.
    """
    # Thin book: -105 / -105, a near-fair coin flip.
    # Fat book:  -400 / +225, a heavily-juiced favourite.
    books = [
        ("ThinBook", [("Boston Celtics", -105), ("Miami Heat", -105)]),
        ("FatBook", [("Boston Celtics", -400), ("Miami Heat", 225)]),
    ]
    svc, _ = _service()
    session = _StubSession()
    svc._process_event(_event(books), session)

    fair = _fair_by_outcome(session, "ThinBook")
    got = 1.0 / fair["Boston Celtics"]

    # Correct: devig each book, then pool the fair probabilities in log-odds.
    per_book_fair = [
        strip_vig_power_method(
            [decimal_to_implied_prob(american_to_decimal(p)) for _, p in outcomes]
        )
        for _, outcomes in books
    ]
    expected = pool_log_odds(per_book_fair)[0]

    # Wrong (the old behaviour): average the vigged probabilities, then devig once.
    vigged = [
        [decimal_to_implied_prob(american_to_decimal(p)) for _, p in outcomes]
        for _, outcomes in books
    ]
    avg = [sum(v[i] for v in vigged) / len(vigged) for i in range(2)]
    naive = strip_vig_power_method(avg)[0]

    # The two genuinely differ, so this test has teeth.
    assert not math.isclose(expected, naive, rel_tol=1e-4)
    assert math.isclose(got, expected, rel_tol=1e-9)


def test_fair_probabilities_are_keyed_by_outcome_not_position():
    """
    Both books quote the same market; the second lists the away team first.
    Positional indexing pairs the home fair price with the away offer and
    manufactures a large phantom edge.
    """
    books = [
        ("HomeFirst", [("Boston Celtics", -300), ("Miami Heat", 250)]),
        ("AwayFirst", [("Miami Heat", 250), ("Boston Celtics", -300)]),
    ]
    svc, _ = _service()
    session = _StubSession()
    svc._process_event(_event(books), session)

    a = _fair_by_outcome(session, "HomeFirst")
    b = _fair_by_outcome(session, "AwayFirst")

    # Identical prices must get identical fair values regardless of listing order.
    assert math.isclose(a["Boston Celtics"], b["Boston Celtics"], rel_tol=1e-9)
    assert math.isclose(a["Miami Heat"], b["Miami Heat"], rel_tol=1e-9)
    # And the favourite must be priced as the favourite for both.
    assert a["Boston Celtics"] < a["Miami Heat"]
    assert b["Boston Celtics"] < b["Miami Heat"]

    # Every recorded EV is consistent with the fair price it was judged against.
    for e in session.added:
        assert math.isclose(
            e.calculated_ev,
            (1.0 / e.fair_odds) * (e.odds_offered - 1) - (1 - 1.0 / e.fair_odds),
            rel_tol=1e-9,
        )


def test_books_quoting_a_different_outcome_set_are_excluded():
    """A book offering a draw does not belong in a two-way consensus."""
    two_way = [("Boston Celtics", -140), ("Miami Heat", 120)]
    books = [
        ("TwoWayA", two_way),
        ("TwoWayB", two_way),
        ("ThreeWay", [("Boston Celtics", 150), ("Draw", 250), ("Miami Heat", 200)]),
    ]
    svc, _ = _service()
    session = _StubSession()
    svc._process_event(_event(books), session)

    # The odd book out is not priced at all, and never contributes to the pool.
    assert "ThreeWay" not in {e.bookmaker_name for e in session.added}

    # The consensus is exactly the two-way books' own fair line - the three-way
    # book's probabilities (which include a draw) never enter the pool.
    expected = strip_vig_power_method(
        [decimal_to_implied_prob(american_to_decimal(p)) for _, p in two_way]
    )
    fair = _fair_by_outcome(session, "TwoWayA")
    assert math.isclose(fair["Boston Celtics"], 1.0 / expected[0], rel_tol=1e-9)
