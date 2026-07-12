import asyncio
import argparse

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.storage.database import DatabaseManager
from app.storage.repository import seed_if_empty, load_resolver
from app.ingestion.mock_provider import MockOddsClient
from app.services.detection_service import EdgeDetectionService

def build_service() -> EdgeDetectionService:
    settings = get_settings()
    setup_logging(settings)

    db_manager = DatabaseManager(settings.db_url)
    session = db_manager.get_session()
    seed_if_empty(session, settings.canonical_seed_path)
    resolver = load_resolver(session)
    session.close()

    provider = MockOddsClient(settings.mock_fixture_path)  # swap point for a real provider later

    return EdgeDetectionService(provider, resolver, db_manager, settings)

async def main(once: bool = False) -> None:
    service = build_service()
    if once:
        await service.poll_once()
    else:
        await service.run_forever()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run a single poll cycle and exit (for smoke-testing/CI).")
    args = parser.parse_args()
    asyncio.run(main(once=args.once))
