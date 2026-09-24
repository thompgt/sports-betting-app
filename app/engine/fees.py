"""
Kalshi Transaction Fee Approximations and Mathematical Mispricing Logic.

Implements the CFTC-regulated Kalshi fee schedule and evaluates whether
a mathematical mispricing exists net of transaction fees and order book depth slippage.

RULE: DO NOT TRADE. Only determine whether mispricing exists mathematically.
"""

import math
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

# Published Kalshi Standard Fee Schedule parameters:
# fee = ceil(rate * C * P * (1 - P)) in cents
DEFAULT_TAKER_RATE = 0.07  # 7%
DEFAULT_MAKER_RATE = 0.00  # 0%

@dataclass(frozen=True)
class KalshiFeeModel:
    taker_rate: float = DEFAULT_TAKER_RATE
    maker_rate: float = DEFAULT_MAKER_RATE

    def taker_fee_cents(self, contracts: int, price_dollars: float) -> int:
        """
        Calculates total taker fee in cents for an order of `contracts` at `price_dollars`.
        Fee formula: ceil(rate * contracts * P * (1 - P) * 100) / 100 dollars = ceil(...) cents.
        
        Examples:
        - At 50c ($0.50) for 100 contracts: ceil(0.07 * 100 * 0.5 * 0.5 * 100) = ceil(175.0) = 175 cents ($1.75).
        - Fee per contract at 50c is 1.75 cents.
        - At 10c ($0.10) for 100 contracts: ceil(0.07 * 100 * 0.1 * 0.9 * 100) = ceil(63.0) = 63 cents ($0.63).
        """
        if contracts <= 0 or price_dollars <= 0.0 or price_dollars >= 1.0:
            return 0
        raw_cents = self.taker_rate * float(contracts) * price_dollars * (1.0 - price_dollars) * 100.0
        return int(math.ceil(raw_cents - 1e-9))

    def taker_fee_dollars(self, contracts: int, price_dollars: float) -> float:
        return self.taker_fee_cents(contracts, price_dollars) / 100.0

    def fee_per_contract_cents(self, contracts: int, price_dollars: float) -> float:
        if contracts <= 0:
            return 0.0
        return self.taker_fee_cents(contracts, price_dollars) / float(contracts)

@dataclass
class MathematicalMispricing:
    ticker: str
    mispricing_type: str  # 'single_contract_depth', 'basket_dutch_book'
    fair_prob: float
    depth_evaluated: int
    effective_price_cents: float
    fee_cents_total: int
    fee_per_contract_cents: float
    gross_edge_cents: float
    net_edge_cents: float
    net_ev_dollars: float
    ev_pct: float
    is_mathematically_profitable: bool
    rationale: str

def evaluate_single_contract_mispricing(
    ticker: str,
    fair_prob: float,
    vwap_cents: float,
    contracts: int,
    fee_model: Optional[KalshiFeeModel] = None
) -> MathematicalMispricing:
    """
    Evaluates whether buying YES contracts at VWAP price constitutes a mathematical mispricing
    after deducting Kalshi's taker transaction fee.
    
    A mispricing ONLY exists if Net Edge > 0.
    """
    if fee_model is None:
        fee_model = KalshiFeeModel()

    price_dollars = vwap_cents / 100.0
    fair_cents = fair_prob * 100.0
    gross_edge_cents = fair_cents - vwap_cents

    fee_cents_total = fee_model.taker_fee_cents(contracts, price_dollars)
    fee_per_contract_cents = fee_cents_total / float(contracts) if contracts > 0 else 0.0

    net_edge_cents = gross_edge_cents - fee_per_contract_cents
    net_ev_dollars = (net_edge_cents * contracts) / 100.0

    total_cost_dollars = (vwap_cents * contracts + fee_cents_total) / 100.0
    ev_pct = (net_ev_dollars / total_cost_dollars * 100.0) if total_cost_dollars > 0 else 0.0

    is_profitable = net_edge_cents > 0.0

    if is_profitable:
        rationale = (
            f"Mathematical mispricing detected: Fair {fair_cents:.2f}c beats "
            f"VWAP {vwap_cents:.2f}c + fee {fee_per_contract_cents:.2f}c "
            f"(Net edge: +{net_edge_cents:.2f}c/contract, EV: +{ev_pct:.2f}%)"
        )
    else:
        rationale = (
            f"No mathematical mispricing: Gross edge {gross_edge_cents:+.2f}c is wiped out "
            f"by fee {fee_per_contract_cents:.2f}c (Net edge: {net_edge_cents:+.2f}c/contract)"
        )

    return MathematicalMispricing(
        ticker=ticker,
        mispricing_type="single_contract_depth",
        fair_prob=fair_prob,
        depth_evaluated=contracts,
        effective_price_cents=vwap_cents,
        fee_cents_total=fee_cents_total,
        fee_per_contract_cents=fee_per_contract_cents,
        gross_edge_cents=gross_edge_cents,
        net_edge_cents=net_edge_cents,
        net_ev_dollars=net_ev_dollars,
        ev_pct=ev_pct,
        is_mathematically_profitable=is_profitable,
        rationale=rationale,
    )

def evaluate_basket_dutch_book_net(
    event_ticker: str,
    outcome_vwaps: Dict[str, float],
    contracts: int,
    fee_model: Optional[KalshiFeeModel] = None
) -> MathematicalMispricing:
    """
    Evaluates whether buying a basket of mutually exclusive outcomes is mathematically mispriced
    (Dutch book arbitrage) net of fees across all legs.
    """
    if fee_model is None:
        fee_model = KalshiFeeModel()

    total_ask_vwap_cents = sum(outcome_vwaps.values())
    total_fees_cents = 0

    for ticker, vwap_cents in outcome_vwaps.items():
        total_fees_cents += fee_model.taker_fee_cents(contracts, vwap_cents / 100.0)

    fee_per_basket_cents = total_fees_cents / float(contracts) if contracts > 0 else 0.0

    # Payout is exactly 100c ($1.00) per basket
    payout_cents = 100.0
    gross_edge_cents = payout_cents - total_ask_vwap_cents
    net_edge_cents = gross_edge_cents - fee_per_basket_cents

    net_ev_dollars = (net_edge_cents * contracts) / 100.0
    total_cost_dollars = (total_ask_vwap_cents * contracts + total_fees_cents) / 100.0
    ev_pct = (net_ev_dollars / total_cost_dollars * 100.0) if total_cost_dollars > 0 else 0.0

    is_profitable = net_edge_cents > 0.0
    rationale = (
        f"Basket Dutch Book: Basket VWAP {total_ask_vwap_cents:.2f}c + fees {fee_per_basket_cents:.2f}c "
        f"vs Guaranteed Payout {payout_cents:.0f}c (Net edge: {net_edge_cents:+.2f}c, EV: {ev_pct:+.2f}%)"
    )

    return MathematicalMispricing(
        ticker=event_ticker,
        mispricing_type="basket_dutch_book",
        fair_prob=1.0,
        depth_evaluated=contracts,
        effective_price_cents=total_ask_vwap_cents,
        fee_cents_total=total_fees_cents,
        fee_per_contract_cents=fee_per_basket_cents,
        gross_edge_cents=gross_edge_cents,
        net_edge_cents=net_edge_cents,
        net_ev_dollars=net_ev_dollars,
        ev_pct=ev_pct,
        is_mathematically_profitable=is_profitable,
        rationale=rationale,
    )
