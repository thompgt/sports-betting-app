"""
Order Book Depth and VWAP Execution Pricing Models.

Calculates cumulative book depth, order book walking, volume-weighted average price (VWAP),
liquidity slippage, and size-dependent edge decay for Kalshi prediction markets.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from app.engine.microstructure import PriceLevel, BinaryMarketBook

@dataclass(frozen=True)
class DepthLevelSummary:
    level_index: int
    price_cents: int
    quantity: int
    cumulative_qty: int
    cumulative_cost_cents: int

@dataclass
class FillSimulation:
    target_qty: int
    filled_qty: int
    is_fully_filled: bool
    total_cost_cents: int
    vwap_cents: float
    slippage_cents: float   # Difference from top-of-book (touch) price
    marginal_price_cents: int
    levels_swept: int

    @property
    def vwap_dollars(self) -> float:
        return self.vwap_cents / 100.0

@dataclass
class DepthProfile:
    ticker: str
    best_bid_cents: Optional[int]
    best_ask_cents: Optional[int]
    total_bid_depth: int
    total_ask_depth: int
    ask_simulations: Dict[int, FillSimulation]  # target_qty -> FillSimulation (for buying YES)
    bid_simulations: Dict[int, FillSimulation]  # target_qty -> FillSimulation (for selling YES)

def walk_book_depth(
    levels: List[PriceLevel],
    target_qty: int,
    side: str = "ask"
) -> FillSimulation:
    """
    Simulates walking the order book depth to execute target_qty contracts.
    
    side='ask': Walking asks to BUY contracts (levels sorted ascending by price).
    side='bid': Walking bids to SELL contracts (levels sorted descending by price).
    
    Returns FillSimulation with exact VWAP, cumulative cost, and slippage.
    """
    if target_qty <= 0:
        raise ValueError(f"Target quantity must be positive, got {target_qty}")
    if not levels:
        return FillSimulation(
            target_qty=target_qty,
            filled_qty=0,
            is_fully_filled=False,
            total_cost_cents=0,
            vwap_cents=0.0,
            slippage_cents=0.0,
            marginal_price_cents=0,
            levels_swept=0,
        )

    best_price = levels[0].price_cents
    remaining = target_qty
    total_cost = 0
    levels_swept = 0
    marginal_price = best_price

    for lvl in levels:
        if remaining <= 0:
            break

        take_qty = min(remaining, lvl.quantity)
        total_cost += take_qty * lvl.price_cents
        remaining -= take_qty
        levels_swept += 1
        marginal_price = lvl.price_cents

    filled_qty = target_qty - remaining
    is_fully_filled = (remaining == 0)

    if filled_qty > 0:
        vwap = total_cost / float(filled_qty)
        if side == "ask":
            # For asks (buying), slippage = vwap - best_ask (positive indicates paid more)
            slippage = vwap - float(best_price)
        else:
            # For bids (selling), slippage = best_bid - vwap (positive indicates sold for less)
            slippage = float(best_price) - vwap
    else:
        vwap = float(best_price)
        slippage = 0.0

    return FillSimulation(
        target_qty=target_qty,
        filled_qty=filled_qty,
        is_fully_filled=is_fully_filled,
        total_cost_cents=total_cost,
        vwap_cents=vwap,
        slippage_cents=slippage,
        marginal_price_cents=marginal_price,
        levels_swept=levels_swept,
    )

def evaluate_depth_profile(
    book: BinaryMarketBook,
    target_sizes: Optional[List[int]] = None
) -> DepthProfile:
    """
    Evaluates order book liquidity and depth slippage across standard quantity tiers.
    Default tiers: 10, 50, 100, 250, 500 contracts.
    """
    if target_sizes is None:
        target_sizes = [10, 50, 100, 250, 500]

    total_bid_depth = sum(b.quantity for b in book.bids)
    total_ask_depth = sum(a.quantity for a in book.asks)

    ask_sims: Dict[int, FillSimulation] = {}
    for size in target_sizes:
        ask_sims[size] = walk_book_depth(book.asks, size, side="ask")

    bid_sims: Dict[int, FillSimulation] = {}
    for size in target_sizes:
        bid_sims[size] = walk_book_depth(book.bids, size, side="bid")

    return DepthProfile(
        ticker=book.ticker,
        best_bid_cents=book.best_bid,
        best_ask_cents=book.best_ask,
        total_bid_depth=total_bid_depth,
        total_ask_depth=total_ask_depth,
        ask_simulations=ask_sims,
        bid_simulations=bid_sims,
    )

def depth_adjusted_gross_edge(
    fair_prob: float,
    book: BinaryMarketBook,
    size: int
) -> Optional[float]:
    """
    Computes depth-adjusted gross edge per contract in dollars:
    Gross Edge = Fair Price - VWAP_ask(size) in dollars.
    
    Returns None if book has no asks or insufficient liquidity to fill size.
    """
    sim = walk_book_depth(book.asks, size, side="ask")
    if not sim.is_fully_filled:
        return None

    vwap_dollars = sim.vwap_dollars
    return fair_prob - vwap_dollars
