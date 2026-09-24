"""
Kalshi Data Ingestor.

Handles ingesting Kalshi markets, orderbooks, and depth levels into MySQL (or configured SQLAlchemy DB).
Ensures transaction safety, idempotency, and foreign key integrity between snapshots and depth levels.
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional, Tuple, Dict, Any

from sqlalchemy.orm import Session
from app.ingestion.kalshi_client import KalshiRestClient, ParsedOrderbook
from app.engine.microstructure import BinaryMarketBook, PriceLevel
from app.storage.database import DatabaseManager
from app.storage.models import KalshiMarket, KalshiOrderbookSnapshot, KalshiOrderbookLevel

logger = logging.getLogger(__name__)

def parse_iso_datetime(dt_str: Optional[str]) -> Optional[datetime]:
    if not dt_str:
        return None
    try:
        # Replace trailing Z with UTC
        cleaned = dt_str.replace("Z", "+00:00")
        return datetime.fromisoformat(cleaned)
    except Exception:
        return None

class KalshiIngestor:
    def __init__(self, client: KalshiRestClient, db_manager: DatabaseManager):
        self.client = client
        self.db_manager = db_manager

    def ingest_markets(
        self,
        status: str = "open",
        series_ticker: Optional[str] = None,
        event_ticker: Optional[str] = None,
        limit: int = 100,
    ) -> List[str]:
        """
        Fetches open markets from Kalshi and upserts them into `kalshi_markets`.
        Returns list of ingested market tickers.
        """
        payload = self.client.get_markets(
            status=status,
            series_ticker=series_ticker,
            event_ticker=event_ticker,
            limit=limit,
        )
        raw_markets = payload.get("markets", [])
        logger.info("Fetched %d markets from Kalshi REST API", len(raw_markets))

        tickers_ingested: List[str] = []
        session = self.db_manager.get_session()
        try:
            for m in raw_markets:
                ticker = m.get("ticker")
                if not ticker:
                    continue

                event_tick = m.get("event_ticker") or ""
                title = m.get("title") or ticker
                market_status = m.get("status") or status
                open_time = parse_iso_datetime(m.get("open_time"))
                close_time = parse_iso_datetime(m.get("close_time"))
                tick_size = int(m.get("tick_size") or 1)
                last_price = m.get("last_price")
                last_price_cents = int(last_price) if last_price is not None else None
                vol = int(m.get("volume") or 0)

                existing = session.query(KalshiMarket).filter_by(ticker=ticker).first()
                if existing:
                    existing.event_ticker = event_tick
                    existing.title = title
                    existing.status = market_status
                    existing.open_time = open_time
                    existing.close_time = close_time
                    existing.tick_size_cents = tick_size
                    existing.last_price_cents = last_price_cents
                    existing.volume = vol
                    existing.updated_at = datetime.now(timezone.utc)
                else:
                    new_market = KalshiMarket(
                        ticker=ticker,
                        event_ticker=event_tick,
                        title=title,
                        status=market_status,
                        open_time=open_time,
                        close_time=close_time,
                        tick_size_cents=tick_size,
                        last_price_cents=last_price_cents,
                        volume=vol,
                        updated_at=datetime.now(timezone.utc),
                    )
                    session.add(new_market)

                tickers_ingested.append(ticker)

            session.commit()
            logger.info("Successfully upserted %d markets into database", len(tickers_ingested))
            return tickers_ingested
        except Exception:
            session.rollback()
            logger.exception("Error ingesting markets into database")
            raise
        finally:
            session.close()

    def ingest_orderbook(
        self,
        ticker: str,
        depth: Optional[int] = 25,
    ) -> Optional[int]:
        """
        Fetches the orderbook for a market, parses it into microstructure model,
        and saves snapshot and all depth levels into `kalshi_orderbook_snapshots` and
        `kalshi_orderbook_levels`.
        Returns the snapshot_id.
        """
        try:
            raw_payload = self.client.get_raw_orderbook(ticker, depth=depth)
            parsed_ob = self.client.parse_orderbook_response(ticker, raw_payload)
        except Exception as e:
            logger.warning("Failed to fetch orderbook for %s: %e", ticker, e)
            return None

        # Build microstructure book
        mb = BinaryMarketBook(
            ticker=ticker,
            bids=[PriceLevel(price_cents=b.price_cents, quantity=b.quantity) for b in parsed_ob.bids],
            asks=[PriceLevel(price_cents=a.price_cents, quantity=a.quantity) for a in parsed_ob.asks],
        )

        total_bid_qty = sum(b.quantity for b in mb.bids)
        total_ask_qty = sum(a.quantity for a in mb.asks)

        session = self.db_manager.get_session()
        try:
            snapshot = KalshiOrderbookSnapshot(
                ticker=ticker,
                timestamp=datetime.fromtimestamp(parsed_ob.timestamp, tz=timezone.utc),
                best_bid_cents=mb.best_bid,
                best_ask_cents=mb.best_ask,
                spread_cents=mb.spread,
                mid_price_cents=mb.mid_price,
                micro_price_cents=mb.micro_price,
                is_crossed=mb.is_crossed,
                bid_depth_total=total_bid_qty,
                ask_depth_total=total_ask_qty,
            )
            session.add(snapshot)
            session.flush()  # get snapshot.id

            snapshot_id = snapshot.id

            levels: List[KalshiOrderbookLevel] = []
            for idx, b in enumerate(mb.bids):
                levels.append(
                    KalshiOrderbookLevel(
                        snapshot_id=snapshot_id,
                        ticker=ticker,
                        side="bid",
                        level_index=idx,
                        price_cents=b.price_cents,
                        quantity=b.quantity,
                    )
                )

            for idx, a in enumerate(mb.asks):
                levels.append(
                    KalshiOrderbookLevel(
                        snapshot_id=snapshot_id,
                        ticker=ticker,
                        side="ask",
                        level_index=idx,
                        price_cents=a.price_cents,
                        quantity=a.quantity,
                    )
                )

            if levels:
                session.add_all(levels)

            session.commit()
            logger.debug(
                "Ingested orderbook snapshot #%d for %s (%d bids, %d asks)",
                snapshot_id, ticker, len(mb.bids), len(mb.asks)
            )
            return snapshot_id
        except Exception:
            session.rollback()
            logger.exception("Error saving orderbook snapshot for %s", ticker)
            raise
        finally:
            session.close()

    def ingest_batch_orderbooks(self, tickers: List[str], depth: Optional[int] = 25) -> int:
        count = 0
        for ticker in tickers:
            snap_id = self.ingest_orderbook(ticker, depth=depth)
            if snap_id is not None:
                count += 1
        return count
