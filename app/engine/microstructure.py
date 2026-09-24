"""
Kalshi Market Microstructure Models.

Captures the unique microstructure of Kalshi binary prediction markets:
1. Dual bid-stack reflection: YES Ask = 100 - NO Bid, NO Ask = 100 - YES Bid.
2. Central limit order book hygiene: crossed book detection, one-sidedness, spread.
3. Micro-price and Order Flow Imbalance (OFI) based on touch volume.
4. Multi-outcome event parity: Dutch book underpricing (sum(Asks) < 1.00) and overpricing (sum(Bids) > 1.00).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import math

# Valid binary price bounds in cents (1 to 99)
MIN_PRICE_CENTS = 1
MAX_PRICE_CENTS = 99
SETTLEMENT_PAYOUT_CENTS = 100

def reflect_no_bid_to_yes_ask(no_bid_cents: int) -> Optional[int]:
    """
    Reflects a NO bid into a YES ask:
    A bid to buy NO at p cents is economically identical to an offer
    to sell YES at (100 - p) cents.
    
    A NO bid of 0c or 100c cannot be a valid active resting order
    and is rejected.
    """
    yes_ask = SETTLEMENT_PAYOUT_CENTS - no_bid_cents
    if MIN_PRICE_CENTS <= yes_ask <= MAX_PRICE_CENTS:
        return yes_ask
    return None

def reflect_yes_bid_to_no_ask(yes_bid_cents: int) -> Optional[int]:
    """
    Reflects a YES bid into a NO ask:
    A bid to buy YES at p cents is an offer to sell NO at (100 - p) cents.
    """
    no_ask = SETTLEMENT_PAYOUT_CENTS - yes_bid_cents
    if MIN_PRICE_CENTS <= no_ask <= MAX_PRICE_CENTS:
        return no_ask
    return None

@dataclass(frozen=True)
class PriceLevel:
    price_cents: int
    quantity: int

    def __post_init__(self):
        if self.quantity < 0:
            raise ValueError(f"Quantity must be non-negative, got {self.quantity}")

    @property
    def price_dollars(self) -> float:
        return self.price_cents / 100.0

@dataclass
class BinaryMarketBook:
    """
    Represents the full two-sided limit order book for a single binary contract (YES).
    """
    ticker: str
    bids: List[PriceLevel] = field(default_factory=list)  # Sorted descending by price
    asks: List[PriceLevel] = field(default_factory=list)  # Sorted ascending by price

    def __post_init__(self):
        # Ensure proper sorting
        self.bids = sorted(
            [b for b in self.bids if b.quantity > 0 and MIN_PRICE_CENTS <= b.price_cents <= MAX_PRICE_CENTS],
            key=lambda x: x.price_cents,
            reverse=True
        )
        self.asks = sorted(
            [a for a in self.asks if a.quantity > 0 and MIN_PRICE_CENTS <= a.price_cents <= MAX_PRICE_CENTS],
            key=lambda x: x.price_cents,
            reverse=False
        )

    @classmethod
    def from_raw_stacks(
        cls,
        ticker: str,
        yes_bids: List[Tuple[int, int]],
        no_bids: List[Tuple[int, int]]
    ) -> "BinaryMarketBook":
        """
        Constructs a two-sided YES book from raw YES bid and NO bid stacks.
        """
        bids = [
            PriceLevel(price_cents=p, quantity=q)
            for p, q in yes_bids
            if MIN_PRICE_CENTS <= p <= MAX_PRICE_CENTS and q > 0
        ]

        asks = []
        for no_p, no_q in no_bids:
            if no_q > 0:
                reflected_yes_ask = reflect_no_bid_to_yes_ask(no_p)
                if reflected_yes_ask is not None:
                    asks.append(PriceLevel(price_cents=reflected_yes_ask, quantity=no_q))

        return cls(ticker=ticker, bids=bids, asks=asks)

    @property
    def best_bid(self) -> Optional[int]:
        return self.bids[0].price_cents if self.bids else None

    @property
    def best_ask(self) -> Optional[int]:
        return self.asks[0].price_cents if self.asks else None

    @property
    def best_bid_qty(self) -> int:
        return self.bids[0].quantity if self.bids else 0

    @property
    def best_ask_qty(self) -> int:
        return self.asks[0].quantity if self.asks else 0

    @property
    def spread(self) -> Optional[int]:
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None

    @property
    def is_crossed(self) -> bool:
        """A book is crossed if best_bid >= best_ask."""
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_bid >= self.best_ask
        return False

    @property
    def is_one_sided(self) -> bool:
        """True if exactly one side of the book has quotes."""
        return (len(self.bids) > 0 and len(self.asks) == 0) or (len(self.bids) == 0 and len(self.asks) > 0)

    @property
    def is_empty(self) -> bool:
        return len(self.bids) == 0 and len(self.asks) == 0

    @property
    def is_tradable(self) -> bool:
        """A book is tradable only if both sides exist and it is not crossed."""
        return (
            self.best_bid is not None
            and self.best_ask is not None
            and not self.is_crossed
        )

    @property
    def mid_price(self) -> Optional[float]:
        """Simple mid-price in cents."""
        if self.is_tradable:
            return (self.best_bid + self.best_ask) / 2.0
        return None

    @property
    def micro_price(self) -> Optional[float]:
        """
        Volume-weighted mid-price (micro-price) in cents:
        P_micro = (Q_bid * P_ask + Q_ask * P_bid) / (Q_bid + Q_ask)
        
        When bid volume > ask volume, micro-price shifts closer to P_ask,
        reflecting upward buy pressure.
        """
        if not self.is_tradable:
            return None

        q_b = self.best_bid_qty
        q_a = self.best_ask_qty
        total_q = q_b + q_a
        if total_q == 0:
            return self.mid_price

        p_b = float(self.best_bid)
        p_a = float(self.best_ask)
        return (q_b * p_a + q_a * p_b) / total_q

    @property
    def order_flow_imbalance(self) -> Optional[float]:
        """
        Order flow imbalance (OFI) ratio between -1.0 (pure ask pressure) and +1.0 (pure bid pressure):
        OFI = (Q_bid - Q_ask) / (Q_bid + Q_ask)
        """
        if not self.is_tradable:
            return None
        total_q = self.best_bid_qty + self.best_ask_qty
        if total_q == 0:
            return 0.0
        return (self.best_bid_qty - self.best_ask_qty) / total_q


@dataclass
class MultiOutcomeParityReport:
    """
    Evaluates mathematical consistency across mutually exclusive and exhaustive outcomes
    belonging to a single event.
    """
    event_ticker: str
    outcome_tickers: List[str]
    ask_sum_cents: int
    bid_sum_cents: int
    is_dutch_book_underpriced: bool  # ask_sum < 100c: guaranteed arbitrage buying basket
    is_overpriced: bool              # bid_sum > 100c: guaranteed arbitrage selling basket
    underpricing_edge_cents: int     # 100 - ask_sum if positive
    overpricing_edge_cents: int      # bid_sum - 100 if positive
    fair_probabilities: Dict[str, float]

def evaluate_multi_outcome_parity(
    event_ticker: str,
    outcome_books: Dict[str, BinaryMarketBook]
) -> Optional[MultiOutcomeParityReport]:
    """
    Checks mathematical parity for an event with mutually exclusive outcomes.
    
    If the sum of best ask prices across all outcomes is < 100c, a Dutch book
    underpricing anomaly exists.
    If the sum of best bid prices across all outcomes is > 100c, an overpricing
    anomaly exists.
    """
    if len(outcome_books) < 2:
        return None

    # Verify all books are healthy and two-sided
    for ticker, book in outcome_books.items():
        if not book.is_tradable:
            return None

    tickers = list(outcome_books.keys())
    ask_sum = sum(outcome_books[t].best_ask for t in tickers)
    bid_sum = sum(outcome_books[t].best_bid for t in tickers)

    is_dutch_book = ask_sum < SETTLEMENT_PAYOUT_CENTS
    is_overpriced = bid_sum > SETTLEMENT_PAYOUT_CENTS
    underpriced_edge = max(0, SETTLEMENT_PAYOUT_CENTS - ask_sum)
    overpriced_edge = max(0, bid_sum - SETTLEMENT_PAYOUT_CENTS)

    # Compute fair probabilities via mid-price normalization
    mid_prices = {t: outcome_books[t].mid_price for t in tickers}
    total_mid = sum(mid_prices.values())
    if total_mid > 0:
        fair_probs = {t: mid_prices[t] / total_mid for t in tickers}
    else:
        fair_probs = {t: 1.0 / len(tickers) for t in tickers}

    return MultiOutcomeParityReport(
        event_ticker=event_ticker,
        outcome_tickers=tickers,
        ask_sum_cents=ask_sum,
        bid_sum_cents=bid_sum,
        is_dutch_book_underpriced=is_dutch_book,
        is_overpriced=is_overpriced,
        underpricing_edge_cents=underpriced_edge,
        overpricing_edge_cents=overpriced_edge,
        fair_probabilities=fair_probs,
    )
