//! Orders, their lifecycle, and the events the matching engine emits.

use edge_core::types::{ClientOrderId, MarketId, OrderId, Price, Qty, Side, StrategyId, Ts};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum OrderType {
    #[default]
    Limit,
    /// Executes against whatever is resting, at any price. On a binary contract
    /// the worst case is bounded ($1 or $0), which is why a market order is
    /// merely expensive here rather than unbounded as it is in equities — but
    /// it is still the fastest way to give away an edge, so strategies should
    /// prefer marketable limits.
    Market,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TimeInForce {
    /// Rest any unfilled remainder on the book.
    #[default]
    Gtc,
    /// Fill what is available immediately, cancel the rest.
    Ioc,
    /// Fill entirely and immediately, or do nothing at all.
    Fok,
    /// Never take liquidity. Rejected outright if it would cross. This is the
    /// only safe order type for a market maker: on venues where the maker fee
    /// is zero and the taker fee is 1.75c, a quote that accidentally crosses
    /// does not merely lose the spread, it inverts the strategy's economics.
    PostOnly,
}

impl TimeInForce {
    #[inline]
    pub const fn rests(self) -> bool {
        matches!(self, TimeInForce::Gtc | TimeInForce::PostOnly)
    }
}

/// What to do when an order would trade against another order from the same
/// owner. Wash trading is prohibited on regulated venues, and beyond the
/// compliance question it is pure fee leakage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SelfTradePrevention {
    /// Cancel the resting order and let the incoming one continue. The default
    /// because it expresses "my new opinion supersedes my old one".
    #[default]
    CancelResting,
    /// Cancel the incoming order's remainder, leave the book alone.
    CancelIncoming,
    /// Reduce both by the overlap without producing a trade.
    DecrementBoth,
    /// Permit the trade. Only for venues that allow it and for backtests
    /// reproducing a historical tape.
    Allow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Order {
    pub id: OrderId,
    pub client_id: ClientOrderId,
    pub market: MarketId,
    /// Owner. Also the unit of self-trade prevention: two strategies in one
    /// process are separate owners and may legitimately trade with each other
    /// only if the venue permits it.
    pub strategy: StrategyId,
    pub side: Side,
    /// Limit price. Ignored for [`OrderType::Market`].
    pub price: Price,
    /// Size as submitted.
    pub qty: Qty,
    /// Size still working.
    pub remaining: Qty,
    pub order_type: OrderType,
    pub tif: TimeInForce,
    pub stp: SelfTradePrevention,
    /// Submission time, supplied by the caller rather than read from a clock,
    /// so a replay reproduces the original run exactly.
    pub ts: Ts,
    /// Engine arrival sequence. Total order over every order the engine has
    /// seen, and the ultimate tiebreak for price-time priority.
    pub seq: u64,
}

impl Order {
    pub fn limit(
        id: OrderId,
        market: MarketId,
        strategy: StrategyId,
        side: Side,
        price: Price,
        qty: Qty,
    ) -> Self {
        Order {
            id,
            client_id: ClientOrderId(0),
            market,
            strategy,
            side,
            price,
            qty,
            remaining: qty,
            order_type: OrderType::Limit,
            tif: TimeInForce::Gtc,
            stp: SelfTradePrevention::default(),
            ts: Ts::ZERO,
            seq: 0,
        }
    }

    pub fn with_tif(mut self, tif: TimeInForce) -> Self {
        self.tif = tif;
        self
    }

    pub fn with_type(mut self, t: OrderType) -> Self {
        self.order_type = t;
        self
    }

    pub fn with_stp(mut self, stp: SelfTradePrevention) -> Self {
        self.stp = stp;
        self
    }

    pub fn with_ts(mut self, ts: Ts) -> Self {
        self.ts = ts;
        self
    }

    pub fn with_client_id(mut self, id: ClientOrderId) -> Self {
        self.client_id = id;
        self
    }

    #[inline]
    pub fn filled(&self) -> Qty {
        self.qty - self.remaining
    }

    #[inline]
    pub fn is_done(&self) -> bool {
        self.remaining.get() <= 0
    }

    /// The effective limit for matching: a market order is a limit at the
    /// extreme of the price range, which on a binary contract is a real,
    /// bounded number rather than an infinity.
    #[inline]
    pub fn effective_limit(&self) -> Price {
        match self.order_type {
            OrderType::Limit => self.price,
            OrderType::Market => match self.side {
                Side::Buy => Price::ONE,
                Side::Sell => Price::ZERO,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RejectReason {
    /// Price is not on the venue's tick grid, or outside `(0, 1)`.
    InvalidPrice,
    InvalidQty,
    /// A post-only order that would have taken liquidity.
    WouldCross,
    /// A fill-or-kill that could not be filled in full.
    InsufficientLiquidity,
    /// The market is not open.
    MarketNotTradable,
    DuplicateOrderId,
    UnknownOrder,
    /// The order would have traded against its own owner.
    SelfTrade,
}

/// A trade. Emitted once per maker order consumed, so one aggressive order that
/// sweeps three levels produces three fills.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fill {
    pub seq: u64,
    pub market: MarketId,
    /// Execution price, always the **maker's** limit — the resting order set the
    /// terms, and the aggressor accepted them. Charging the taker their own
    /// limit price instead is the classic way a naive backtest invents profit.
    pub price: Price,
    pub qty: Qty,
    pub taker_order: OrderId,
    pub maker_order: OrderId,
    pub taker_strategy: StrategyId,
    pub maker_strategy: StrategyId,
    /// Side of the aggressor, which is the trade's signed direction for
    /// order-flow-imbalance features.
    pub taker_side: Side,
    pub ts: Ts,
}

/// Everything the engine reports back about an action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum BookEvent {
    /// The order was validated and is now working (possibly after partial fills).
    Accepted {
        seq: u64,
        order: OrderId,
        market: MarketId,
        resting_qty: Qty,
    },
    Rejected {
        seq: u64,
        order: OrderId,
        market: MarketId,
        reason: RejectReason,
    },
    Filled(Fill),
    Cancelled {
        seq: u64,
        order: OrderId,
        market: MarketId,
        remaining: Qty,
    },
    /// The unfilled remainder of an IOC, or the whole of an unfilled FOK.
    Expired {
        seq: u64,
        order: OrderId,
        market: MarketId,
        remaining: Qty,
    },
}

impl BookEvent {
    pub fn order_id(&self) -> OrderId {
        match self {
            BookEvent::Accepted { order, .. }
            | BookEvent::Rejected { order, .. }
            | BookEvent::Cancelled { order, .. }
            | BookEvent::Expired { order, .. } => *order,
            BookEvent::Filled(f) => f.taker_order,
        }
    }

    pub fn seq(&self) -> u64 {
        match self {
            BookEvent::Accepted { seq, .. }
            | BookEvent::Rejected { seq, .. }
            | BookEvent::Cancelled { seq, .. }
            | BookEvent::Expired { seq, .. } => *seq,
            BookEvent::Filled(f) => f.seq,
        }
    }

    pub fn as_fill(&self) -> Option<&Fill> {
        match self {
            BookEvent::Filled(f) => Some(f),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use edge_core::types::{MarketId, OrderId, StrategyId};

    fn o(side: Side, price: i64, qty: i64) -> Order {
        Order::limit(
            OrderId(1),
            MarketId(0),
            StrategyId(1),
            side,
            Price::from_cents(price),
            Qty(qty),
        )
    }

    #[test]
    fn a_market_order_limits_at_the_contract_bound() {
        let buy = o(Side::Buy, 50, 10).with_type(OrderType::Market);
        assert_eq!(buy.effective_limit(), Price::ONE);
        let sell = o(Side::Sell, 50, 10).with_type(OrderType::Market);
        assert_eq!(sell.effective_limit(), Price::ZERO);
    }

    #[test]
    fn only_resting_tifs_rest() {
        assert!(TimeInForce::Gtc.rests());
        assert!(TimeInForce::PostOnly.rests());
        assert!(!TimeInForce::Ioc.rests());
        assert!(!TimeInForce::Fok.rests());
    }

    #[test]
    fn fill_progress_tracks_remaining() {
        let mut order = o(Side::Buy, 50, 100);
        assert_eq!(order.filled(), Qty(0));
        assert!(!order.is_done());
        order.remaining = Qty(40);
        assert_eq!(order.filled(), Qty(60));
        order.remaining = Qty(0);
        assert!(order.is_done());
    }

    #[test]
    fn events_expose_their_order_and_sequence() {
        let e = BookEvent::Cancelled {
            seq: 7,
            order: OrderId(3),
            market: MarketId(0),
            remaining: Qty(5),
        };
        assert_eq!(e.order_id(), OrderId(3));
        assert_eq!(e.seq(), 7);
        assert!(e.as_fill().is_none());
    }
}
