import time
import math
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import urllib.request
import urllib.error
import json
import asyncio

logger = logging.getLogger(__name__)

DEFAULT_KALSHI_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
DEFAULT_USER_AGENT = "LineEdge-KalshiClient/1.0"

@dataclass
class OrderbookLevel:
    price_cents: int
    quantity: int

    @property
    def price_dollars(self) -> float:
        return self.price_cents / 100.0

@dataclass
class ParsedOrderbook:
    ticker: str
    timestamp: float
    bids: List[OrderbookLevel] = field(default_factory=list)  # YES bids, sorted desc by price
    asks: List[OrderbookLevel] = field(default_factory=list)  # YES asks (reflected from NO bids), sorted asc by price

    @property
    def best_bid(self) -> Optional[int]:
        return self.bids[0].price_cents if self.bids else None

    @property
    def best_ask(self) -> Optional[int]:
        return self.asks[0].price_cents if self.asks else None

    @property
    def spread(self) -> Optional[int]:
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None

    @property
    def is_crossed(self) -> bool:
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_bid >= self.best_ask
        return False

    @property
    def is_one_sided(self) -> bool:
        return (not self.bids and bool(self.asks)) or (bool(self.bids) and not self.asks)

    @property
    def is_empty(self) -> bool:
        return not self.bids and not self.asks

    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2.0
        return None

class KalshiClientError(Exception):
    """Base exception for Kalshi REST client errors."""
    pass

class KalshiHttpError(KalshiClientError):
    def __init__(self, status_code: int, message: str):
        super().__init__(f"HTTP {status_code}: {message}")
        self.status_code = status_code
        self.message = message

class KalshiRestClient:
    """
    REST client for Kalshi's public market data endpoints.
    
    Supports:
    - /markets with query filters (status, series_ticker, cursor, limit)
    - /events with query filters
    - /markets/{ticker}/orderbook with dual support for `orderbook_fp` and legacy `orderbook`.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_KALSHI_BASE_URL,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: float = 10.0,
        rate_limit_per_sec: float = 9.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.user_agent = user_agent
        self.timeout = timeout
        self.min_interval = 1.0 / rate_limit_per_sec if rate_limit_per_sec > 0 else 0.0
        self._last_request_time = 0.0

    def _rate_limit(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request_time = time.monotonic()

    def _request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._rate_limit()
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        if params:
            query_parts = []
            for k, v in params.items():
                if v is not None:
                    query_parts.append(f"{k}={urllib.parse.quote(str(v))}")
            if query_parts:
                url = f"{url}?{'&'.join(query_parts)}"

        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": self.user_agent,
                "Accept": "application/json",
            }
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                status_code = response.status
                body = response.read().decode("utf-8")
                if status_code != 200:
                    raise KalshiHttpError(status_code, body)
                return json.loads(body)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8") if e.fp else ""
            raise KalshiHttpError(e.code, body) from e
        except urllib.error.URLError as e:
            raise KalshiClientError(f"Network error accessing {url}: {e.reason}") from e
        except json.JSONDecodeError as e:
            raise KalshiClientError(f"Failed to parse JSON response from {url}: {e}") from e

    def get_events(self, status: str = "open", cursor: Optional[str] = None, limit: int = 100) -> Dict[str, Any]:
        params = {"status": status, "limit": limit}
        if cursor:
            params["cursor"] = cursor
        return self._request("events", params)

    def get_markets(
        self,
        status: str = "open",
        series_ticker: Optional[str] = None,
        event_ticker: Optional[str] = None,
        cursor: Optional[str] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        params = {"status": status, "limit": limit}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if event_ticker:
            params["event_ticker"] = event_ticker
        if cursor:
            params["cursor"] = cursor
        return self._request("markets", params)

    def get_market(self, ticker: str) -> Dict[str, Any]:
        return self._request(f"markets/{ticker}")

    def get_raw_orderbook(self, ticker: str, depth: Optional[int] = None) -> Dict[str, Any]:
        params = {}
        if depth:
            params["depth"] = depth
        return self._request(f"markets/{ticker}/orderbook", params)

    @staticmethod
    def parse_orderbook_response(ticker: str, payload: Dict[str, Any], ts: Optional[float] = None) -> ParsedOrderbook:
        """
        Parses Kalshi's orderbook response.
        Handles:
        1. Modern `orderbook_fp`:
           {'orderbook_fp': {'yes_dollars': [['0.4500', '100.00']], 'no_dollars': [['0.5100', '80.00']]}}
        2. Legacy `orderbook`:
           {'orderbook': {'yes': [[45, 100]], 'no': [[51, 80]]}}
        
        Reflects NO bids into YES asks:
        A NO bid at price p_no cents is a YES ask at (100 - p_no) cents.
        """
        if ts is None:
            ts = time.time()

        raw_yes: List[Tuple[int, int]] = []
        raw_no: List[Tuple[int, int]] = []

        if "orderbook_fp" in payload and payload["orderbook_fp"]:
            fp = payload["orderbook_fp"]
            yes_list = fp.get("yes_dollars") or []
            no_list = fp.get("no_dollars") or []

            for item in yes_list:
                if len(item) >= 2:
                    try:
                        p_cents = int(round(float(item[0]) * 100.0))
                        q = int(round(float(item[1])))
                        if 1 <= p_cents < 100 and q > 0:
                            raw_yes.append((p_cents, q))
                    except (ValueError, TypeError):
                        continue

            for item in no_list:
                if len(item) >= 2:
                    try:
                        p_cents = int(round(float(item[0]) * 100.0))
                        q = int(round(float(item[1])))
                        if 1 <= p_cents < 100 and q > 0:
                            raw_no.append((p_cents, q))
                    except (ValueError, TypeError):
                        continue

        elif "orderbook" in payload and payload["orderbook"]:
            ob = payload["orderbook"]
            yes_list = ob.get("yes") or []
            no_list = ob.get("no") or []

            for item in yes_list:
                if len(item) >= 2:
                    try:
                        p_cents = int(item[0])
                        q = int(item[1])
                        if 1 <= p_cents < 100 and q > 0:
                            raw_yes.append((p_cents, q))
                    except (ValueError, TypeError):
                        continue

            for item in no_list:
                if len(item) >= 2:
                    try:
                        p_cents = int(item[0])
                        q = int(item[1])
                        if 1 <= p_cents < 100 and q > 0:
                            raw_no.append((p_cents, q))
                    except (ValueError, TypeError):
                        continue

        # Bids: YES bids descending by price
        bids = [OrderbookLevel(price_cents=p, quantity=q) for p, q in raw_yes]
        bids.sort(key=lambda x: x.price_cents, reverse=True)

        # Asks: Reflected from NO bids: YES ask = 100 - NO bid, ascending by price
        asks: List[OrderbookLevel] = []
        for no_p, no_q in raw_no:
            yes_ask_cents = 100 - no_p
            if 1 <= yes_ask_cents < 100:
                asks.append(OrderbookLevel(price_cents=yes_ask_cents, quantity=no_q))
        asks.sort(key=lambda x: x.price_cents, reverse=False)

        return ParsedOrderbook(ticker=ticker, timestamp=ts, bids=bids, asks=asks)

    def get_orderbook(self, ticker: str, depth: Optional[int] = None) -> ParsedOrderbook:
        payload = self.get_raw_orderbook(ticker, depth=depth)
        return self.parse_orderbook_response(ticker, payload)

    # Async wrapper methods for integration with async pipelines
    async def get_events_async(self, status: str = "open", cursor: Optional[str] = None, limit: int = 100) -> Dict[str, Any]:
        return await asyncio.to_thread(self.get_events, status=status, cursor=cursor, limit=limit)

    async def get_markets_async(
        self,
        status: str = "open",
        series_ticker: Optional[str] = None,
        event_ticker: Optional[str] = None,
        cursor: Optional[str] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        return await asyncio.to_thread(
            self.get_markets,
            status=status,
            series_ticker=series_ticker,
            event_ticker=event_ticker,
            cursor=cursor,
            limit=limit,
        )

    async def get_orderbook_async(self, ticker: str, depth: Optional[int] = None) -> ParsedOrderbook:
        return await asyncio.to_thread(self.get_orderbook, ticker=ticker, depth=depth)
