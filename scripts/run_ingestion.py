#!/usr/bin/env python3
"""
Kalshi Data Ingestion Runner.

Runs continuous or single-cycle ingestion of Kalshi markets and orderbooks into MySQL (or configured DB).
"""

import sys
import os
import time
import argparse
import logging

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.ingestion.kalshi_client import KalshiRestClient
from app.ingestion.kalshi_ingestor import KalshiIngestor
from app.storage.database import DatabaseManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("KalshiIngestionRunner")

def run_cycle(ingestor: KalshiIngestor, limit: int = 10, depth: int = 25) -> None:
    logger.info("Starting Kalshi data ingestion cycle...")
    tickers = ingestor.ingest_markets(status="open", limit=limit)
    logger.info("Ingested %d markets metadata.", len(tickers))

    # Take first batch of tickers to snapshot orderbooks
    selected = tickers[:limit]
    logger.info("Ingesting orderbooks for %d active tickers...", len(selected))
    ingested_books = ingestor.ingest_batch_orderbooks(selected, depth=depth)
    logger.info("Cycle complete. Successfully ingested %d orderbook snapshots.", ingested_books)

def main():
    parser = argparse.ArgumentParser(description="Kalshi Market Data Ingestor")
    parser.add_argument("--once", action="store_true", help="Run a single ingestion cycle and exit.")
    parser.add_argument("--interval", type=int, default=30, help="Poll interval in seconds between cycles.")
    parser.add_argument("--limit", type=int, default=10, help="Max number of markets to ingest per cycle.")
    parser.add_argument("--depth", type=int, default=25, help="Order book depth levels to fetch.")
    parser.add_argument("--db-url", type=str, default=None, help="Database URL (defaults to LINEEDGE_DB_URL or sqlite).")
    args = parser.parse_args()

    db_url = args.db_url or os.getenv("LINEEDGE_DB_URL", "sqlite:///./lineedge.db")
    logger.info("Connecting to database: %s", db_url.split("@")[-1] if "@" in db_url else db_url)

    db_manager = DatabaseManager(db_url)
    client = KalshiRestClient()
    ingestor = KalshiIngestor(client, db_manager)

    if args.once:
        run_cycle(ingestor, limit=args.limit, depth=args.depth)
    else:
        logger.info("Running continuous ingestion every %d seconds. Press Ctrl+C to stop.", args.interval)
        while True:
            try:
                run_cycle(ingestor, limit=args.limit, depth=args.depth)
            except Exception as e:
                logger.exception("Error in ingestion cycle: %s", e)
            time.sleep(args.interval)

if __name__ == "__main__":
    main()
