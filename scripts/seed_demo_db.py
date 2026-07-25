"""
Populate a demo SQLite database with SIMULATED market data.

IMPORTANT: every number this script produces is synthetic. There is no real
bookmaker feed anywhere in this repo -- `MockOddsClient` reads JSON fixtures
off disk. This script generates a plausible-looking multi-book market, replays
it through the *real* pipeline (EntityResolver -> devig -> EV -> EdgeCache ->
SQLite -> EdgeAuditor), and leaves behind a database the Streamlit dashboard
can render for documentation screenshots.

Usage:
    $env:PYTHONPATH = "."; python scripts/seed_demo_db.py
"""

from __future__ import annotations

import asyncio
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import text

from app.core.config import Settings
from app.engine.resolver import EntityResolver
from app.ingestion.mock_provider import MockOddsClient
from app.services.detection_service import EdgeDetectionService
from app.storage.database import DatabaseManager
from app.storage.repository import seed_if_empty, load_resolver

REPO_ROOT = Path(__file__).resolve().parents[1]
WORK_DIR = REPO_ROOT / ".demo_tmp"
DB_PATH = REPO_ROOT / "lineedge_demo.db"

SEED = 20260527
CYCLES = 12
CYCLE_MINUTES = 30

BOOKMAKERS = [
    # (key, title, price bias) -- bias shifts a book's view of the home side.
    # A negative bias means the book quotes the home side at a lower implied
    # probability (i.e. longer odds), which is what can create detectable +EV.
    ("pinnacle", "Pinnacle", 0.000),
    ("draftkings", "DraftKings", 0.004),
    ("fanduel", "FanDuel", -0.006),
    ("betmgm", "BetMGM", 0.014),
    ("caesars", "Caesars", -0.012),
    ("betrivers", "BetRivers", -0.010),
]

# Book-level overround (vig). Sharp, low-margin books run tighter.
BOOK_VIG = {
    "Pinnacle": 1.014,
    "DraftKings": 1.042,
    "FanDuel": 1.045,
    "BetMGM": 1.050,
    "Caesars": 1.020,
    "BetRivers": 1.018,
}

# Per-poll pricing noise (std-dev, in probability units). Wider dispersion
# between books is what an edge detector is looking for in the first place.
PRICE_NOISE_SD = 0.020


def _uuid(n: int) -> str:
    return f"{n:08d}-0000-4000-8000-{n:012d}"


def build_universe(now: datetime):
    """Canonical teams/games plus the market parameters used to simulate prices."""
    teams = [
        ("New York Rangers", ["NY Rangers", "Rangers"]),
        ("Los Angeles Kings", ["LA Kings", "Kings"]),
        ("New York Knicks", ["NY Knicks", "Knicks"]),
        ("Boston Celtics", ["Celtics", "BOS Celtics"]),
        ("Kansas City Chiefs", ["KC Chiefs", "Chiefs"]),
        ("Buffalo Bills", ["Bills", "BUF Bills"]),
    ]
    team_records = [
        {"id": _uuid(i + 1), "name": name, "aliases": aliases}
        for i, (name, aliases) in enumerate(teams)
    ]
    by_name = {t["name"]: t["id"] for t in team_records}

    games = [
        {
            "id": _uuid(101),
            "home": "New York Rangers",
            "away": "Los Angeles Kings",
            # the fixture deliberately uses aliases to exercise the resolver
            "home_feed_name": "NY Rangers",
            "away_feed_name": "LA Kings",
            "sport": "icehockey_nhl",
            "start_time": now + timedelta(hours=4),
            "true_home_prob": 0.56,
        },
        {
            "id": _uuid(102),
            "home": "New York Knicks",
            "away": "Boston Celtics",
            "home_feed_name": "Knicks",
            "away_feed_name": "Boston Celtics",
            "sport": "basketball_nba",
            "start_time": now + timedelta(hours=7),
            "true_home_prob": 0.41,
        },
        {
            "id": _uuid(103),
            "home": "Kansas City Chiefs",
            "away": "Buffalo Bills",
            "home_feed_name": "KC Chiefs",
            "away_feed_name": "Bills",
            "sport": "americanfootball_nfl",
            # tips off shortly; the closing snapshot below backdates it so the
            # EdgeAuditor closes this market out and records CLV
            "start_time": now + timedelta(minutes=20),
            "true_home_prob": 0.52,
        },
    ]

    seed_payload = {
        "teams": team_records,
        "games": [
            {
                "id": g["id"],
                "home_team_id": by_name[g["home"]],
                "away_team_id": by_name[g["away"]],
                "sport": g["sport"],
                "start_time": g["start_time"].replace(microsecond=0).isoformat(),
            }
            for g in games
        ],
    }
    return seed_payload, games


def prob_to_american(prob: float) -> int:
    """Convert an implied probability to the nearest American price."""
    prob = min(max(prob, 0.02), 0.98)
    decimal = 1.0 / prob
    if decimal >= 2.0:
        return int(round((decimal - 1.0) * 100))
    return int(round(-100.0 / (decimal - 1.0)))


def build_cycle_fixture(games, cycle: int, sim_time: datetime, rng: random.Random):
    """One polling snapshot: every book quotes both sides of every h2h market."""
    events = []
    for idx, game in enumerate(games):
        # the "true" price drifts slowly over the session (a random walk)
        drift = rng.gauss(0, 0.006) * cycle
        true_home = min(max(game["true_home_prob"] + drift, 0.15), 0.85)

        bookmakers = []
        for key, title, bias in BOOKMAKERS:
            noise = rng.gauss(0, PRICE_NOISE_SD)
            book_home = min(max(true_home + bias + noise, 0.10), 0.90)
            book_away = 1.0 - book_home

            vig = BOOK_VIG[title]
            quoted_home = book_home * vig
            quoted_away = book_away * vig

            bookmakers.append({
                "key": key,
                "title": title,
                "last_update": (sim_time - timedelta(seconds=rng.randint(5, 90))).isoformat(),
                "markets": [{
                    "key": "h2h",
                    "outcomes": [
                        {"name": game["home_feed_name"], "price": prob_to_american(quoted_home)},
                        {"name": game["away_feed_name"], "price": prob_to_american(quoted_away)},
                    ],
                }],
            })

        events.append({
            "id": f"sim_{idx:03d}_{cycle:03d}",
            "sport_key": game["sport"],
            "commence_time": game["start_time"].replace(microsecond=0).isoformat(),
            "home_team": game["home_feed_name"],
            "away_team": game["away_feed_name"],
            "bookmakers": bookmakers,
        })
    return events


async def main() -> None:
    rng = random.Random(SEED)
    now = datetime.utcnow().replace(microsecond=0)

    WORK_DIR.mkdir(exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()

    seed_payload, games = build_universe(now)
    seed_path = WORK_DIR / "canonical_entities.json"
    seed_path.write_text(json.dumps(seed_payload, indent=2), encoding="utf-8")

    settings = Settings(
        db_url=f"sqlite:///{DB_PATH.as_posix()}",
        canonical_seed_path=str(seed_path),
        mock_fixture_path=str(WORK_DIR / "cycle.json"),
        # a fresh service per cycle already resets the cache; keep the real defaults
        ev_threshold=0.02,
    )

    db_manager = DatabaseManager(settings.db_url)
    session = db_manager.get_session()
    seed_if_empty(session, str(seed_path))
    resolver = load_resolver(session)
    session.close()

    start = now - timedelta(minutes=CYCLE_MINUTES * (CYCLES - 1))

    for cycle in range(CYCLES):
        sim_time = start + timedelta(minutes=CYCLE_MINUTES * cycle)
        fixture = build_cycle_fixture(games, cycle, sim_time, rng)
        Path(settings.mock_fixture_path).write_text(json.dumps(fixture, indent=2), encoding="utf-8")

        provider = MockOddsClient(settings.mock_fixture_path)
        # A fresh service per cycle gives each simulated poll a clean EdgeCache,
        # which is what a long-lived process would look like across a 15m TTL.
        service = EdgeDetectionService(provider, resolver, db_manager, settings)

        with db_manager.engine.connect() as conn:
            before = conn.execute(text("SELECT COALESCE(MAX(id), 0) FROM detected_edges")).scalar_one()

        await service.poll_once()

        # Backdate this cycle's rows to the simulated poll time so the dashboard's
        # time-series view spans the whole simulated session.
        with db_manager.engine.begin() as conn:
            conn.execute(
                text("UPDATE detected_edges SET timestamp_detected = :ts WHERE id > :before"),
                {"ts": sim_time.isoformat(sep=" "), "before": before},
            )

    # --- Closing snapshot -------------------------------------------------
    # One more poll, with the NFL game backdated so its start time has passed.
    # `ev_threshold` is raised out of reach so this cycle opens no new positions:
    # it exists purely to record the market's closing prices, which is what
    # EdgeDetectionService._run_clv_closeouts() hands to the EdgeAuditor.
    closing_games = [
        g.model_copy(update={"start_time": g.start_time - timedelta(hours=1)})
        if str(g.id) == _uuid(103) else g
        for g in resolver.games
    ]
    closing_resolver = EntityResolver(resolver.teams, closing_games)
    closing_settings = settings.model_copy(update={"ev_threshold": 10.0})

    fixture = build_cycle_fixture(games, CYCLES, now, rng)
    Path(settings.mock_fixture_path).write_text(json.dumps(fixture, indent=2), encoding="utf-8")
    closing_service = EdgeDetectionService(
        MockOddsClient(settings.mock_fixture_path), closing_resolver, db_manager, closing_settings
    )
    await closing_service.poll_once()

    with db_manager.engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM detected_edges")).scalar_one()
        active = conn.execute(text("SELECT COUNT(*) FROM detected_edges WHERE is_active = 1")).scalar_one()
        closed = conn.execute(text("SELECT COUNT(*) FROM detected_edges WHERE closing_line IS NOT NULL")).scalar_one()

    print(f"Demo DB written to {DB_PATH}")
    print(f"  edges recorded : {total}")
    print(f"  active (+EV)   : {active}")
    print(f"  CLV closed-out : {closed}")
    if active == 0 or closed == 0:
        raise SystemExit("Demo DB is not representative (need both active and closed-out edges).")


if __name__ == "__main__":
    asyncio.run(main())
