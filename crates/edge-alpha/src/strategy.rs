//! The strategy contract.
//!
//! A strategy here is a **pure function of a market snapshot into intents**. It
//! does not submit orders, does not know about the venue, does not know its own
//! bankroll beyond what it is handed, and cannot consult a clock. Everything it
//! needs arrives in [`MarketView`]; everything it wants leaves as [`Action`].
//!
//! That restriction buys three things that matter more than the convenience it
//! costs:
//!
//! - **Determinism.** The same view produces the same intents, so a live session
//!   can be replayed from its journal and reproduce every decision exactly.
//! - **Testability.** A strategy is tested by constructing a book and reading
//!   the intents back, with no engine, no venue and no time involved.
//! - **Enforceable risk.** Nothing a strategy emits reaches a venue without
//!   passing the risk engine. A strategy that asks for a thousand contracts and
//!   is given thirty cannot route around the difference, because it never had
//!   the ability to send an order in the first place.
//!
//! Sizing is likewise not the strategy's final say. Strategies express *how much
//! they want* and *how confident they are*; the risk layer decides what that
//! translates to in contracts.

use edge_book::book::OrderBook;
use edge_book::order::{Fill, TimeInForce};
use edge_core::market::{MarketSpec, MarketStatus};
use edge_core::types::{EventId, Leg, MarketId, OrderId, Price, Prob, Qty, Side, StrategyId, Ts};
use std::borrow::Cow;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::features::Features;
use crate::predictor::Prediction;

/// Why a strategy did something, carried on the intent itself.
///
/// A `Cow` rather than a `&'static str` for one reason: intents are journalled
/// and replayed, and a reason that cannot be read back from disk is a reason
/// that vanishes exactly when a post-mortem needs it. Strategies pass string
/// literals and pay nothing; the journal deserialises into owned strings.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Reason(pub Cow<'static, str>);

impl From<&'static str> for Reason {
    fn from(s: &'static str) -> Self {
        Reason(Cow::Borrowed(s))
    }
}

impl From<String> for Reason {
    fn from(s: String) -> Self {
        Reason(Cow::Owned(s))
    }
}

impl std::ops::Deref for Reason {
    type Target = str;
    fn deref(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Reason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// A strategy's own resting order, as it sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RestingOrder {
    pub id: OrderId,
    pub side: Side,
    pub price: Price,
    pub remaining: Qty,
    pub ts: Ts,
}

impl RestingOrder {
    pub fn age_secs(&self, now: Ts) -> f64 {
        (now.0 - self.ts.0).max(0) as f64 / 1e9
    }
}

/// Everything a strategy is allowed to see about one market at one instant.
///
/// Borrowed rather than owned so the engine can build one per market per tick
/// without allocating. The lifetime is the tick.
#[derive(Debug, Clone, Copy)]
pub struct MarketView<'a> {
    pub spec: &'a MarketSpec,
    pub book: &'a OrderBook,
    /// Microstructure features, absent on a one-sided or cold book.
    pub features: Option<&'a Features>,
    /// The model's forecast, absent when no model covers this market.
    pub prediction: Option<&'a Prediction>,
    /// Independent fair value from cross-venue consensus, where one exists.
    /// Kept separate from `prediction` deliberately: consensus and model are
    /// different kinds of evidence and a strategy may want only one of them.
    pub consensus: Option<Prob>,
    /// Signed net position in contracts. Positive is long YES.
    pub position: Qty,
    /// Average cost of the open position, in dollars per contract.
    pub avg_cost: f64,
    /// Capital the strategy may size against.
    pub bankroll: f64,
    /// The strategy's own working orders in this market.
    pub resting: &'a [RestingOrder],
    pub now: Ts,
}

impl MarketView<'_> {
    /// The best fair value available, preferring the model when it has earned
    /// weight and falling back through consensus to the market's own mid.
    ///
    /// A strategy that just wants "the number" should call this rather than
    /// reimplementing the precedence, which is easy to get subtly wrong in a way
    /// that silently trades on an untrained model.
    pub fn fair(&self) -> Option<Prob> {
        if let Some(p) = self.prediction
            && !p.is_market_echo()
        {
            return Some(p.fair);
        }
        if let Some(c) = self.consensus {
            return Some(c);
        }
        self.book.mid().map(|m| Prob::clamped(m.dollars()))
    }

    /// Fair value from evidence that is **independent of this market's own
    /// price** — a model that has earned weight, or cross-venue consensus.
    ///
    /// Distinct from [`Self::fair`], which falls back to the mid. That fallback
    /// is right for a maker deciding where to quote and catastrophic for a taker
    /// deciding whether to cross: comparing the mid against the touch always
    /// shows half a spread of "edge", so a taker using it crosses continuously
    /// and pays the spread for the privilege. Anything that takes liquidity must
    /// use this and refuse to trade when it returns `None`.
    pub fn independent_fair(&self) -> Option<Prob> {
        if let Some(p) = self.prediction
            && !p.is_market_echo()
        {
            return Some(p.fair);
        }
        self.consensus
    }

    /// Whether this market can be traded at all right now. Checked by every
    /// strategy before anything else, because quoting a halted or closed market
    /// is a pure source of rejections.
    pub fn is_tradable(&self) -> bool {
        self.spec.status == MarketStatus::Open && self.book.best_bid().is_some() && self.book.best_ask().is_some()
    }

    /// Seconds until resolution, or infinity where none is scheduled.
    pub fn time_left(&self) -> f64 {
        self.spec.seconds_to_close(self.now).unwrap_or(f64::INFINITY)
    }

    pub fn resting_on(&self, side: Side) -> impl Iterator<Item = &RestingOrder> {
        self.resting.iter().filter(move |o| o.side == side)
    }
}

/// Every market resolving on one event, seen together.
///
/// A separate view because arbitrage is not a per-market question. The YES leg
/// on one venue and the NO leg on another are one position, and a strategy that
/// can only see them one at a time cannot tell the difference between an
/// arbitrage and two unrelated bets. Strategies that do not care about this
/// simply do not implement [`Strategy::on_event`].
#[derive(Debug, Clone, Copy)]
pub struct EventView<'a> {
    pub event: EventId,
    /// One entry per market on the event, across every venue.
    pub markets: &'a [MarketView<'a>],
    pub bankroll: f64,
    pub now: Ts,
}

impl<'a> EventView<'a> {
    /// Markets carrying the given leg, in the order supplied.
    pub fn leg(&self, leg: Leg) -> impl Iterator<Item = &MarketView<'a>> {
        self.markets.iter().filter(move |m| m.spec.leg == leg)
    }

    /// Whether every market on the event is currently tradable. Arbitrage is
    /// worthless if only one side of it can be executed.
    pub fn all_tradable(&self) -> bool {
        !self.markets.is_empty() && self.markets.iter().all(|m| m.is_tradable())
    }
}

/// A strategy's request. Never an instruction — the risk engine may resize or
/// refuse any of these.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OrderIntent {
    pub market: MarketId,
    pub side: Side,
    pub price: Price,
    /// Contracts wanted, before risk sizing.
    pub qty: Qty,
    pub tif: TimeInForce,
    /// Why, in a form that survives into the journal. Post-mortems on a losing
    /// session are impossible without this and cheap with it.
    pub reason: Reason,
}

impl OrderIntent {
    pub fn new(market: MarketId, side: Side, price: Price, qty: Qty, reason: impl Into<Reason>) -> Self {
        OrderIntent {
            market,
            side,
            price,
            qty,
            tif: TimeInForce::Gtc,
            reason: reason.into(),
        }
    }

    pub fn with_tif(mut self, tif: TimeInForce) -> Self {
        self.tif = tif;
        self
    }

    /// A quote: post-only, so it either makes liquidity or is rejected. A market
    /// maker that accidentally crosses pays the taker fee and inherits the
    /// adverse selection it was being paid to avoid.
    pub fn quote(market: MarketId, side: Side, price: Price, qty: Qty, reason: impl Into<Reason>) -> Self {
        Self::new(market, side, price, qty, reason).with_tif(TimeInForce::PostOnly)
    }

    /// A taking order: immediate-or-cancel, so an edge that has already
    /// disappeared does not leave a resting order behind at a stale price.
    pub fn take(market: MarketId, side: Side, price: Price, qty: Qty, reason: impl Into<Reason>) -> Self {
        Self::new(market, side, price, qty, reason).with_tif(TimeInForce::Ioc)
    }

    /// Worst-case capital committed if this fills completely, before fees.
    pub fn notional(&self) -> f64 {
        let per = match self.side {
            Side::Buy => self.price.dollars(),
            Side::Sell => self.price.complement().dollars(),
        };
        per * self.qty.get() as f64
    }
}

/// What a strategy wants done.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum Action {
    Place(OrderIntent),
    Cancel { order_id: OrderId, reason: Reason },
}

impl Action {
    pub fn intent(&self) -> Option<&OrderIntent> {
        match self {
            Action::Place(i) => Some(i),
            Action::Cancel { .. } => None,
        }
    }

    pub fn reason(&self) -> &str {
        match self {
            Action::Place(i) => &i.reason,
            Action::Cancel { reason, .. } => reason,
        }
    }
}

/// Running per-strategy accounting, for attribution and for the dashboard.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct StrategyStats {
    pub intents: u64,
    pub cancels: u64,
    pub fills: u64,
    pub contracts: i64,
    /// Notional traded, in dollars.
    pub volume: f64,
    /// Fills where this strategy was the resting side. The maker share is the
    /// single clearest indicator of whether a quoting strategy is actually
    /// quoting or has degenerated into crossing the spread.
    pub maker_fills: u64,
}

impl StrategyStats {
    pub fn maker_share(&self) -> f64 {
        if self.fills == 0 {
            0.0
        } else {
            self.maker_fills as f64 / self.fills as f64
        }
    }
}

/// The contract every trading strategy implements.
pub trait Strategy: Send {
    fn id(&self) -> StrategyId;

    fn name(&self) -> &'static str;

    /// React to a market snapshot by appending intents to `out`.
    ///
    /// Appending rather than returning a `Vec` so the engine can reuse one
    /// buffer across thousands of markets per tick. An empty result — the
    /// overwhelmingly common case — costs nothing.
    fn on_market(&mut self, view: &MarketView<'_>, out: &mut Vec<Action>);

    /// React to every market on one event at once.
    ///
    /// Defaulted to nothing because only the cross-market strategies need it,
    /// and a per-market strategy given an event view would be tempted to
    /// double-act on markets it has already seen.
    fn on_event(&mut self, _view: &EventView<'_>, _out: &mut Vec<Action>) {}

    /// A fill involving this strategy. `is_maker` says which side of it the
    /// strategy was on.
    fn on_fill(&mut self, _fill: &Fill, _is_maker: bool) {}

    /// A market this strategy holds resolved. `outcome` is whether it settled YES.
    fn on_settle(&mut self, _market: MarketId, _outcome: bool) {}

    /// Called when trading is halted, so a strategy can drop internal state that
    /// would be wrong on resumption.
    fn on_halt(&mut self) {}

    fn stats(&self) -> StrategyStats {
        StrategyStats::default()
    }
}

/// Bookkeeping shared by every strategy, so the trait's stats method is not
/// reimplemented — usually slightly differently — five times over.
#[derive(Debug, Clone, Copy, Default)]
pub struct StatsRecorder(StrategyStats);

impl StatsRecorder {
    pub fn stats(&self) -> StrategyStats {
        self.0
    }

    /// Record actions and hand them straight back, so a strategy can wrap its
    /// emit path without restructuring it.
    pub fn record(&mut self, actions: &[Action]) {
        for a in actions {
            match a {
                Action::Place(_) => self.0.intents += 1,
                Action::Cancel { .. } => self.0.cancels += 1,
            }
        }
    }

    pub fn record_fill(&mut self, fill: &Fill, is_maker: bool) {
        self.0.fills += 1;
        self.0.maker_fills += u64::from(is_maker);
        self.0.contracts += fill.qty.get();
        self.0.volume += fill.price.dollars() * fill.qty.get() as f64;
    }
}

#[cfg(test)]
#[allow(dead_code)] // helpers exist for every strategy module, not just the current ones
pub(crate) mod harness {
    //! Shared scaffolding for strategy tests: build a book, take a view, read
    //! the intents back. No engine, no clock, no venue.
    use super::*;
    use edge_book::order::Order;
    use edge_core::market::MarketSpec;
    use edge_core::types::{EventId, MarketId, OrderId, StrategyId, VenueId};

    pub const M: MarketId = MarketId(0);

    pub struct Sim {
        pub book: OrderBook,
        pub spec: MarketSpec,
        pub resting: Vec<RestingOrder>,
        pub position: Qty,
        pub avg_cost: f64,
        pub bankroll: f64,
        pub now: Ts,
        seq: u64,
        next: u64,
    }

    impl Default for Sim {
        fn default() -> Self {
            Self::new()
        }
    }

    impl Sim {
        pub fn new() -> Self {
            Self::for_market(M, EventId(0), VenueId(1))
        }

        /// A market with a chosen identity, for tests that need several books
        /// on one event.
        pub fn for_market(market: MarketId, event: EventId, venue: VenueId) -> Self {
            Sim {
                book: OrderBook::new(market, 10_000),
                spec: MarketSpec::new(market, event, venue, "TEST"),
                resting: Vec::new(),
                position: Qty::ZERO,
                avg_cost: 0.0,
                bankroll: 10_000.0,
                now: Ts(0),
                seq: 0,
                next: 1_000,
            }
        }

        /// Rest a counterparty order in the book.
        pub fn rest(&mut self, side: Side, cents: i64, qty: i64) -> OrderId {
            self.next += 1;
            let id = OrderId(self.next);
            let mut out = Vec::new();
            let market = self.book.market();
            self.book.submit(
                Order::limit(id, market, StrategyId(99), side, Price::from_cents(cents), Qty(qty)),
                &mut self.seq,
                &mut out,
            );
            id
        }

        /// A two-sided book with the given inside prices.
        pub fn quoted(bid: i64, ask: i64) -> Self {
            let mut s = Self::new();
            s.rest(Side::Buy, bid, 500);
            s.rest(Side::Sell, ask, 500);
            s
        }

        pub fn view<'a>(&'a self, prediction: Option<&'a Prediction>, consensus: Option<Prob>) -> MarketView<'a> {
            MarketView {
                spec: &self.spec,
                book: &self.book,
                features: None,
                prediction,
                consensus,
                position: self.position,
                avg_cost: self.avg_cost,
                bankroll: self.bankroll,
                resting: &self.resting,
                now: self.now,
            }
        }

        pub fn plain(&self) -> MarketView<'_> {
            self.view(None, None)
        }

        /// Run a strategy once and collect what it wanted.
        pub fn step(&self, s: &mut dyn Strategy, view: &MarketView<'_>) -> Vec<Action> {
            let mut out = Vec::new();
            s.on_market(view, &mut out);
            out
        }
    }

    /// Build a feature vector from named values, leaving the rest at zero.
    /// Tests should say `[("z_score", 2.5)]` rather than counting array slots.
    pub fn feats(mid: f64, pairs: &[(&str, f64)]) -> Features {
        let mut v = [0.0; crate::features::N_FEATURES];
        v[0] = mid;
        v[crate::features::N_FEATURES - 1] = 1.0;
        for (name, value) in pairs {
            let i = crate::features::FEATURE_NAMES
                .iter()
                .position(|n| n == name)
                .unwrap_or_else(|| panic!("no such feature: {name}"));
            v[i] = *value;
        }
        Features::from_values(v, mid)
    }

    pub fn places(actions: &[Action]) -> Vec<OrderIntent> {
        actions.iter().filter_map(|a| a.intent().cloned()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::harness::*;
    use super::*;
    use edge_core::types::MarketId;

    #[test]
    fn fair_value_prefers_the_model_only_once_it_has_earned_weight() {
        let sim = Sim::quoted(40, 42);

        let echo = Prediction {
            z: [0.0; crate::features::N_FEATURES],
            market_logit: 0.0,
            score: 0.0,
            market: Prob::new(0.41).unwrap(),
            model: Prob::new(0.60).unwrap(),
            fair: Prob::new(0.41).unwrap(),
            weight: 0.0,
        };
        // An unweighted model must not displace consensus, however confident it
        // looks: `model` says 60c and is ignored.
        let v = sim.view(Some(&echo), Some(Prob::new(0.55).unwrap()));
        assert_eq!(v.fair().unwrap().get(), 0.55);

        let earned = Prediction {
            weight: 0.4,
            fair: Prob::new(0.48).unwrap(),
            ..echo
        };
        let v = sim.view(Some(&earned), Some(Prob::new(0.55).unwrap()));
        assert_eq!(v.fair().unwrap().get(), 0.48);
    }

    #[test]
    fn fair_value_falls_back_to_the_mid_with_no_evidence_at_all() {
        let sim = Sim::quoted(40, 42);
        assert!((sim.plain().fair().unwrap().get() - 0.41).abs() < 1e-9);
    }

    #[test]
    fn a_one_sided_or_halted_market_is_not_tradable() {
        let mut sim = Sim::new();
        sim.rest(Side::Buy, 40, 100);
        assert!(!sim.plain().is_tradable(), "one-sided book");

        let mut sim = Sim::quoted(40, 42);
        assert!(sim.plain().is_tradable());
        sim.spec.status = MarketStatus::Halted;
        assert!(!sim.plain().is_tradable());
    }

    #[test]
    fn time_left_is_infinite_without_a_scheduled_close() {
        let mut sim = Sim::quoted(40, 42);
        assert_eq!(sim.plain().time_left(), f64::INFINITY);

        sim.spec.closes_at = Some(Ts(3_600_000_000_000));
        assert!((sim.plain().time_left() - 3_600.0).abs() < 1e-6);
    }

    #[test]
    fn quotes_post_and_takes_cross() {
        let q = OrderIntent::quote(M, Side::Buy, Price::from_cents(40), Qty(10), "q");
        assert_eq!(q.tif, TimeInForce::PostOnly);

        let t = OrderIntent::take(M, Side::Buy, Price::from_cents(42), Qty(10), "t");
        assert_eq!(t.tif, TimeInForce::Ioc);
    }

    #[test]
    fn notional_prices_a_sell_as_the_complement() {
        // Selling YES at 40c commits 60c of capital, not 40c. Sizing against the
        // quoted price instead is how a short book quietly runs at 1.5x its
        // intended risk.
        let buy = OrderIntent::new(M, Side::Buy, Price::from_cents(40), Qty(100), "b");
        assert!((buy.notional() - 40.0).abs() < 1e-9);

        let sell = OrderIntent::new(M, Side::Sell, Price::from_cents(40), Qty(100), "s");
        assert!((sell.notional() - 60.0).abs() < 1e-9);
    }

    #[test]
    fn resting_orders_filter_by_side_and_report_their_age() {
        let mut sim = Sim::quoted(40, 42);
        sim.now = Ts(5_000_000_000);
        sim.resting = vec![
            RestingOrder {
                id: OrderId(1),
                side: Side::Buy,
                price: Price::from_cents(39),
                remaining: Qty(10),
                ts: Ts(0),
            },
            RestingOrder {
                id: OrderId(2),
                side: Side::Sell,
                price: Price::from_cents(43),
                remaining: Qty(10),
                ts: Ts(4_000_000_000),
            },
        ];
        let v = sim.plain();
        assert_eq!(v.resting_on(Side::Buy).count(), 1);
        assert!((v.resting_on(Side::Buy).next().unwrap().age_secs(v.now) - 5.0).abs() < 1e-9);
        assert!((v.resting_on(Side::Sell).next().unwrap().age_secs(v.now) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn the_recorder_separates_maker_and_taker_fills() {
        use edge_book::order::Fill;
        let mut r = StatsRecorder::default();
        r.record(&[
            Action::Place(OrderIntent::new(M, Side::Buy, Price::from_cents(40), Qty(5), "x")),
            Action::Cancel {
                order_id: OrderId(1),
                reason: "y".into(),
            },
        ]);
        assert_eq!(r.stats().intents, 1);
        assert_eq!(r.stats().cancels, 1);

        let fill = Fill {
            seq: 1,
            market: MarketId(0),
            price: Price::from_cents(40),
            qty: Qty(10),
            taker_order: OrderId(2),
            maker_order: OrderId(1),
            taker_strategy: StrategyId(2),
            maker_strategy: StrategyId(1),
            taker_side: Side::Buy,
            ts: Ts(0),
        };
        r.record_fill(&fill, true);
        r.record_fill(&fill, false);
        assert_eq!(r.stats().fills, 2);
        assert_eq!(r.stats().maker_share(), 0.5);
        assert_eq!(r.stats().contracts, 20);
        assert!((r.stats().volume - 8.0).abs() < 1e-9);
    }

    #[test]
    fn actions_serialise_with_their_reason_intact() {
        let a = Action::Place(OrderIntent::quote(M, Side::Buy, Price::from_cents(40), Qty(5), "skew"));
        let s = serde_json::to_string(&a).unwrap();
        assert!(s.contains("skew"), "{s}");
        let back: Action = serde_json::from_str(&s).unwrap();
        assert_eq!(a, back);
    }
}
