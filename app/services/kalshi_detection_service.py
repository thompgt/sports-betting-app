"""
Kalshi Mathematical Mispricing Detection Service.

Continuously scans Kalshi markets and order books, factors in order book depth and
transaction fee schedules, and determines if true mathematical mispricing exists.

STRICT GUARANTEE: Does NOT place trades or execute orders. Only evaluates mispricing mathematically.
"""

import logging
import json
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from app.ingestion.kalshi_client import KalshiRestClient
from app.engine.microstructure import BinaryMarketBook, PriceLevel
from app.engine.depth import walk_book_depth, evaluate_depth_profile
from app.engine.fees import (
    KalshiFeeModel,
    MathematicalMispricing,
    evaluate_single_contract_mispricing,
    evaluate_basket_dutch_book_net
)
from app.storage.database import DatabaseManager
from app.storage.models import KalshiMispricing

logger = logging.getLogger(__name__)

class KalshiMispricingDetector:
    """
    Detector service that mathematical models mispricings net of fees and depth slippage.
    """

    def __init__(
        self,
        client: KalshiRestClient,
        db_manager: DatabaseManager,
        fee_model: Optional[KalshiFeeModel] = None,
        depth_tiers: Optional[List[int]] = None,
    ):
        self.client = client
        self.db_manager = db_manager
        self.fee_model = fee_model or KalshiFeeModel()
        self.depth_tiers = depth_tiers or [10, 50, 100, 250]

    def scan_market(
        self,
        ticker: str,
        fair_prob: Optional[float] = None,
    ) -> List[MathematicalMispricing]:
        """
        Scans a single Kalshi market orderbook across multiple depth tiers.
        If fair_prob is None, estimates fair probability via micro-price.
        Returns list of detected mathematical mispricings (if any).
        """
        try:
            parsed_ob = self.client.get_orderbook(ticker, depth=25)
        except Exception as e:
            logger.warning("Could not fetch orderbook for %s: %s", ticker, e)
            return []

        book = BinaryMarketBook(
            ticker=ticker,
            bids=[PriceLevel(price_cents=b.price_cents, quantity=b.quantity) for b in parsed_ob.bids],
            asks=[PriceLevel(price_cents=a.price_cents, quantity=a.quantity) for a in parsed_ob.asks],
        )

        if not book.is_tradable:
            logger.debug("Market %s is not tradable (crossed or empty). Skipping.", ticker)
            return []

        # If external fair probability not provided, use volume-weighted micro-price as conservative fair value
        if fair_prob is None:
            if book.micro_price is None:
                return []
            fair_prob = book.micro_price / 100.0

        detected: List[MathematicalMispricing] = []

        for size in self.depth_tiers:
            fill_sim = walk_book_depth(book.asks, size, side="ask")
            if not fill_sim.is_fully_filled:
                logger.debug("Size %d exceeds book depth for %s (filled %d)", size, ticker, fill_sim.filled_qty)
                continue

            mispricing = evaluate_single_contract_mispricing(
                ticker=ticker,
                fair_prob=fair_prob,
                vwap_cents=fill_sim.vwap_cents,
                contracts=size,
                fee_model=self.fee_model,
            )

            if mispricing.is_mathematically_profitable:
                logger.info("!!! MATHEMATICAL MISPRICING FOUND !!! %s", mispricing.rationale)
                self._record_mispricing(mispricing, details={
                    "best_ask": book.best_ask,
                    "vwap_cents": fill_sim.vwap_cents,
                    "slippage_cents": fill_sim.slippage_cents,
                    "marginal_price": fill_sim.marginal_price_cents,
                    "micro_price": book.micro_price
                })
                detected.append(mispricing)

        return detected

    def scan_event_basket_arbitrage(
        self,
        event_ticker: str,
        outcome_tickers: List[str],
        basket_size: int = 50
    ) -> Optional[MathematicalMispricing]:
        """
        Evaluates mutually exclusive outcomes for a basket Dutch book mispricing net of fees.
        """
        outcome_vwaps: Dict[str, float] = {}

        for ticker in outcome_tickers:
            try:
                parsed_ob = self.client.get_orderbook(ticker, depth=25)
                book = BinaryMarketBook(
                    ticker=ticker,
                    bids=[PriceLevel(price_cents=b.price_cents, quantity=b.quantity) for b in parsed_ob.bids],
                    asks=[PriceLevel(price_cents=a.price_cents, quantity=a.quantity) for a in parsed_ob.asks],
                )
                if not book.is_tradable:
                    return None

                sim = walk_book_depth(book.asks, basket_size, side="ask")
                if not sim.is_fully_filled:
                    return None
                outcome_vwaps[ticker] = sim.vwap_cents
            except Exception:
                return None

        mispricing = evaluate_basket_dutch_book_net(
            event_ticker=event_ticker,
            outcome_vwaps=outcome_vwaps,
            contracts=basket_size,
            fee_model=self.fee_model
        )

        if mispricing.is_mathematically_profitable:
            logger.info("!!! BASKET DUTCH BOOK MISPRICING !!! %s", mispricing.rationale)
            self._record_mispricing(mispricing, details={"outcome_vwaps": outcome_vwaps})
            return mispricing

        return None

    def _record_mispricing(self, mispricing: MathematicalMispricing, details: Dict[str, Any]) -> None:
        """
        Persists detected mathematical mispricing to MySQL / SQLite database for auditing.
        Does NOT execute any orders.
        """
        session = self.db_manager.get_session()
        try:
            record = KalshiMispricing(
                ticker=mispricing.ticker,
                timestamp=datetime.now(timezone.utc),
                mispricing_type=mispricing.mispricing_type,
                fair_prob=mispricing.fair_prob,
                depth_evaluated=mispricing.depth_evaluated,
                effective_price_cents=mispricing.effective_price_cents,
                fee_cents=mispricing.fee_cents_total,
                gross_edge_cents=mispricing.gross_edge_cents,
                net_edge_cents=mispricing.net_edge_cents,
                ev_pct=mispricing.ev_pct,
                details_json=json.dumps(details),
            )
            session.add(record)
            session.commit()
            logger.info("Saved mathematical mispricing audit record to database for %s", mispricing.ticker)
        except Exception:
            session.rollback()
            logger.exception("Failed to record mispricing in database")
        finally:
            session.close()
