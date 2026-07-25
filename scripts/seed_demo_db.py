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
import sys
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import Settings
from app.engine.resolver import EntityResolver
from app.ingestion.mock_provider import MockOddsClient
from app.services.detection_service import EdgeDetectionService
from app.storage.database import DatabaseManager
from app.storage.repository import seed_if_empty, load_resolver
from scripts.market_sim import build_cycle_fixture, build_universe, entity_uuid

REPO_ROOT = Path(__file__).resolve().parents[1]
WORK_DIR = REPO_ROOT / ".demo_tmp"
DB_PATH = REPO_ROOT / "lineedge_demo.db"

SEED = 20260527
CYCLES = 12
CYCLE_MINUTES = 30

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
        if str(g.id) == entity_uuid(103) else g
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
