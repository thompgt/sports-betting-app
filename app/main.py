import asyncio
import argparse
import logging
import os
import sys

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.storage.database import DatabaseManager
from app.storage.repository import seed_if_empty, load_resolver
from app.ingestion.mock_provider import MockOddsClient
from app.services.detection_service import EdgeDetectionService

from app.ingestion.kalshi_client import KalshiRestClient
from app.ingestion.kalshi_ingestor import KalshiIngestor
from app.services.kalshi_detection_service import KalshiMispricingDetector

logger = logging.getLogger(__name__)

def build_service() -> EdgeDetectionService:
    settings = get_settings()
    setup_logging(settings)

    db_manager = DatabaseManager(settings.db_url)
    session = db_manager.get_session()
    seed_if_empty(session, settings.canonical_seed_path)
    resolver = load_resolver(session)
    session.close()

    provider = MockOddsClient(settings.mock_fixture_path)

    return EdgeDetectionService(provider, resolver, db_manager, settings)

def run_kalshi_mispricing_detection(db_manager: DatabaseManager, limit: int = 10) -> None:
    """
    Scans active Kalshi markets for mathematical mispricings, factoring in
    order book depth and transaction fee approximations.
    STRICTLY DOES NOT TRADE.
    """
    logger.info("Starting Kalshi mathematical mispricing detection (NO TRADING)...")
    client = KalshiRestClient()
    detector = KalshiMispricingDetector(client=client, db_manager=db_manager)

    markets_resp = client.get_markets(status="open", limit=limit)
    markets = markets_resp.get("markets", [])
    logger.info("Scanning %d open Kalshi markets for mathematical mispricings...", len(markets))

    total_detected = 0
    for m in markets:
        ticker = m.get("ticker")
        if not ticker:
            continue
        mispricings = detector.scan_market(ticker)
        if mispricings:
            total_detected += len(mispricings)

    logger.info("Detection scan complete. Total mathematical mispricings identified: %d", total_detected)

async def main() -> None:
    parser = argparse.ArgumentParser(description="Kalshi Mispricing Detection & Ingestion Engine")
    parser.add_argument("--once", action="store_true", help="Run a single poll cycle and exit.")
    parser.add_argument("--verify-kalshi", action="store_true", help="Verify live Kalshi REST endpoint connectivity.")
    parser.add_argument("--ingest-kalshi", action="store_true", help="Ingest Kalshi markets and orderbooks into database.")
    parser.add_argument("--detect-mispricings", action="store_true", help="Detect mathematical mispricings across Kalshi books.")
    parser.add_argument("--limit", type=int, default=10, help="Market count limit for Kalshi operations.")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings)
    db_url = os.getenv("LINEEDGE_DB_URL", settings.db_url)
    db_manager = DatabaseManager(db_url)

    if args.verify_kalshi:
        from scripts.verify_kalshi_rest import verify_kalshi_endpoints
        success = verify_kalshi_endpoints()
        if not success:
            exit(1)
        return

    if args.ingest_kalshi:
        client = KalshiRestClient()
        ingestor = KalshiIngestor(client, db_manager)
        tickers = ingestor.ingest_markets(status="open", limit=args.limit)
        ingestor.ingest_batch_orderbooks(tickers[:args.limit])
        logger.info("Kalshi ingestion complete.")
        return

    if args.detect_mispricings:
        run_kalshi_mispricing_detection(db_manager, limit=args.limit)
        return

    # Default legacy service loop
    service = build_service()
    if args.once:
        await service.poll_once()
    else:
        await service.run_forever()

if __name__ == "__main__":
    asyncio.run(main())

