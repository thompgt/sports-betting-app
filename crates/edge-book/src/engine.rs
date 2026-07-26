//! The matching engine: many books, one deterministic event stream.
//!
//! The engine owns the global sequence counter, which is what makes the whole
//! system replayable. Every event across every market carries a strictly
//! increasing sequence number assigned here, so a journal of engine output can
//! be replayed to reconstruct the exact state that produced it — the property
//! that lets a backtest be evidence about the code that will actually run.
//!
//! Nothing in this module reads a clock. Timestamps arrive with commands.

use std::collections::{HashMap, HashSet};

use edge_core::types::{MarketId, OrderId, Qty, StrategyId, Ts};

use crate::book::OrderBook;
use crate::latency::{LatencyHistogram, LatencySnapshot};
use crate::order::{BookEvent, Order, RejectReason};

/// An instruction to the engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Command {
    Submit(Order),
    Cancel(OrderId),
    /// Shrink a resting order, keeping its queue position.
    Reduce(OrderId, Qty),
    /// Pull every order belonging to one strategy. The first thing a risk
    /// breach does.
    CancelAllForStrategy(StrategyId),
    /// Pull everything in one market — a venue halt, or a stale-data guard.
    ClearMarket(MarketId),
    /// Pull everything, everywhere. The kill switch.
    CancelAll,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EngineStats {
    pub commands: u64,
    pub orders_accepted: u64,
    pub orders_rejected: u64,
    pub fills: u64,
    pub volume: i64,
    pub cancels: u64,
}

#[derive(Debug)]
pub struct MatchingEngine {
    books: HashMap<MarketId, OrderBook>,
    /// Which market each live order is in, so a cancel needs only an id.
    order_market: HashMap<OrderId, MarketId>,
    /// Candidate order ids per strategy. A superset of what is live — entries
    /// for filled or cancelled orders are pruned lazily on use, which keeps the
    /// submit path free of bookkeeping it would otherwise pay on every fill.
    by_strategy: HashMap<StrategyId, HashSet<OrderId>>,
    seq: u64,
    next_order_id: u64,
    stats: EngineStats,
    latency: LatencyHistogram,
    events: Vec<BookEvent>,
}

impl Default for MatchingEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl MatchingEngine {
    pub fn new() -> Self {
        MatchingEngine {
            books: HashMap::new(),
            order_market: HashMap::new(),
            by_strategy: HashMap::new(),
            seq: 0,
            next_order_id: 0,
            stats: EngineStats::default(),
            latency: LatencyHistogram::new(),
            events: Vec::with_capacity(64),
        }
    }

    /// Start tracking a market. Idempotent, so a catalogue refresh is safe.
    pub fn add_market(&mut self, market: MarketId, tick_size: i64) {
        self.books
            .entry(market)
            .or_insert_with(|| OrderBook::new(market, tick_size));
    }

    pub fn book(&self, market: MarketId) -> Option<&OrderBook> {
        self.books.get(&market)
    }

    pub fn book_mut(&mut self, market: MarketId) -> Option<&mut OrderBook> {
        self.books.get_mut(&market)
    }

    pub fn markets(&self) -> impl Iterator<Item = MarketId> + '_ {
        self.books.keys().copied()
    }

    #[inline]
    pub fn sequence(&self) -> u64 {
        self.seq
    }

    pub fn stats(&self) -> EngineStats {
        self.stats
    }

    pub fn latency(&self) -> LatencySnapshot {
        self.latency.snapshot()
    }

    /// Allocate an engine-unique order id.
    pub fn next_order_id(&mut self) -> OrderId {
        self.next_order_id += 1;
        OrderId(self.next_order_id)
    }

    /// Apply one command and return the events it produced.
    ///
    /// `elapsed_nanos` is how long the caller measured this command taking, fed
    /// back for latency accounting. Passing it in rather than reading a clock
    /// here keeps the engine deterministic; a backtest passes zero.
    pub fn apply(&mut self, cmd: Command, ts: Ts, elapsed_nanos: u64) -> &[BookEvent] {
        self.events.clear();
        self.stats.commands += 1;

        match cmd {
            Command::Submit(mut order) => {
                let Some(book) = self.books.get_mut(&order.market) else {
                    self.seq += 1;
                    self.events.push(BookEvent::Rejected {
                        seq: self.seq,
                        order: order.id,
                        market: order.market,
                        reason: RejectReason::MarketNotTradable,
                    });
                    self.stats.orders_rejected += 1;
                    self.latency.record(elapsed_nanos);
                    return &self.events;
                };
                order.seq = self.seq + 1;
                if order.ts == Ts::ZERO {
                    order.ts = ts;
                }
                let (id, market, strategy) = (order.id, order.market, order.strategy);
                book.submit(order, &mut self.seq, &mut self.events);

                // Only orders that actually rested need tracking.
                if book.order(id).is_some() {
                    self.order_market.insert(id, market);
                    self.by_strategy.entry(strategy).or_default().insert(id);
                }
            }

            Command::Cancel(id) => {
                self.cancel_one(id);
            }

            Command::Reduce(id, qty) => {
                let market = self.order_market.get(&id).copied();
                let result = match market.and_then(|m| self.books.get_mut(&m)) {
                    Some(book) => book.reduce(id, qty),
                    None => Err(RejectReason::UnknownOrder),
                };
                self.seq += 1;
                match result {
                    Ok(remaining) if remaining.get() == 0 => {
                        self.order_market.remove(&id);
                        self.events.push(BookEvent::Cancelled {
                            seq: self.seq,
                            order: id,
                            market: market.unwrap_or(MarketId(0)),
                            remaining: Qty::ZERO,
                        });
                        self.stats.cancels += 1;
                    }
                    Ok(remaining) => self.events.push(BookEvent::Accepted {
                        seq: self.seq,
                        order: id,
                        market: market.unwrap_or(MarketId(0)),
                        resting_qty: remaining,
                    }),
                    Err(reason) => self.events.push(BookEvent::Rejected {
                        seq: self.seq,
                        order: id,
                        market: market.unwrap_or(MarketId(0)),
                        reason,
                    }),
                }
            }

            Command::CancelAllForStrategy(strategy) => {
                let ids: Vec<OrderId> = self
                    .by_strategy
                    .get(&strategy)
                    .map(|s| s.iter().copied().collect())
                    .unwrap_or_default();
                for id in ids {
                    self.cancel_one(id);
                }
                self.by_strategy.remove(&strategy);
            }

            Command::ClearMarket(market) => {
                if let Some(book) = self.books.get_mut(&market) {
                    let before = self.events.len();
                    book.clear(&mut self.seq, ts, &mut self.events);
                    for e in &self.events[before..] {
                        self.order_market.remove(&e.order_id());
                    }
                    self.stats.cancels += (self.events.len() - before) as u64;
                }
            }

            Command::CancelAll => {
                let markets: Vec<MarketId> = self.books.keys().copied().collect();
                for m in markets {
                    if let Some(book) = self.books.get_mut(&m) {
                        book.clear(&mut self.seq, ts, &mut self.events);
                    }
                }
                self.order_market.clear();
                self.by_strategy.clear();
                self.stats.cancels += self.events.len() as u64;
            }
        }

        for e in &self.events {
            match e {
                BookEvent::Accepted { .. } => self.stats.orders_accepted += 1,
                BookEvent::Rejected { .. } => self.stats.orders_rejected += 1,
                BookEvent::Filled(f) => {
                    self.stats.fills += 1;
                    self.stats.volume += f.qty.get();
                }
                BookEvent::Cancelled { .. } | BookEvent::Expired { .. } => {}
            }
        }

        // An order that filled completely is no longer live anywhere.
        for e in &self.events {
            if let BookEvent::Filled(f) = e {
                for id in [f.maker_order, f.taker_order] {
                    if let Some(m) = self.order_market.get(&id) {
                        if self.books.get(m).map(|b| b.order(id).is_none()).unwrap_or(true) {
                            self.order_market.remove(&id);
                        }
                    }
                }
            }
        }

        self.latency.record(elapsed_nanos);
        &self.events
    }

    fn cancel_one(&mut self, id: OrderId) {
        let market = self.order_market.get(&id).copied();
        let cancelled = market
            .and_then(|m| self.books.get_mut(&m))
            .and_then(|b| b.cancel(id));
        self.seq += 1;
        match cancelled {
            Some(order) => {
                self.order_market.remove(&id);
                self.stats.cancels += 1;
                self.events.push(BookEvent::Cancelled {
                    seq: self.seq,
                    order: id,
                    market: order.market,
                    remaining: order.remaining,
                });
            }
            None => self.events.push(BookEvent::Rejected {
                seq: self.seq,
                order: id,
                market: market.unwrap_or(MarketId(0)),
                reason: RejectReason::UnknownOrder,
            }),
        }
    }

    /// Assert every book's invariants. Debug builds and tests.
    pub fn debug_check(&self) {
        for b in self.books.values() {
            b.debug_check();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use edge_core::types::{Price, Side};

    const A: MarketId = MarketId(1);
    const B: MarketId = MarketId(2);
    const CENT: i64 = 10_000;

    fn engine() -> MatchingEngine {
        let mut e = MatchingEngine::new();
        e.add_market(A, CENT);
        e.add_market(B, CENT);
        e
    }

    fn order(e: &mut MatchingEngine, m: MarketId, s: u16, side: Side, cents: i64, qty: i64) -> Order {
        let id = e.next_order_id();
        Order::limit(id, m, StrategyId(s), side, Price::from_cents(cents), Qty(qty))
    }

    #[test]
    fn sequence_numbers_are_global_across_markets() {
        let mut e = engine();
        let o1 = order(&mut e, A, 1, Side::Buy, 40, 10);
        let o2 = order(&mut e, B, 1, Side::Buy, 40, 10);
        let s1 = e.apply(Command::Submit(o1), Ts::ZERO, 0)[0].seq();
        let s2 = e.apply(Command::Submit(o2), Ts::ZERO, 0)[0].seq();
        assert!(s2 > s1, "a second market must continue the same sequence");
    }

    #[test]
    fn markets_do_not_leak_into_each_other() {
        let mut e = engine();
        let o1 = order(&mut e, A, 1, Side::Sell, 40, 10);
        e.apply(Command::Submit(o1), Ts::ZERO, 0);
        let o2 = order(&mut e, B, 2, Side::Buy, 90, 10);
        let ev = e.apply(Command::Submit(o2), Ts::ZERO, 0);
        assert!(
            ev.iter().all(|x| x.as_fill().is_none()),
            "a bid in market B must not hit an offer in market A"
        );
        assert_eq!(e.book(A).unwrap().best_ask(), Some(Price::from_cents(40)));
    }

    #[test]
    fn an_unknown_market_is_rejected_not_a_panic() {
        let mut e = engine();
        let o = order(&mut e, MarketId(99), 1, Side::Buy, 40, 10);
        let ev = e.apply(Command::Submit(o), Ts::ZERO, 0);
        assert!(matches!(
            ev[0],
            BookEvent::Rejected {
                reason: RejectReason::MarketNotTradable,
                ..
            }
        ));
    }

    #[test]
    fn cancel_needs_only_an_order_id() {
        let mut e = engine();
        let o = order(&mut e, B, 1, Side::Buy, 40, 10);
        let id = o.id;
        e.apply(Command::Submit(o), Ts::ZERO, 0);
        let ev = e.apply(Command::Cancel(id), Ts::ZERO, 0);
        assert!(matches!(
            ev[0],
            BookEvent::Cancelled { market, .. } if market == B
        ));
        assert!(e.book(B).unwrap().is_empty());
    }

    #[test]
    fn cancelling_twice_is_rejected_the_second_time() {
        let mut e = engine();
        let o = order(&mut e, A, 1, Side::Buy, 40, 10);
        let id = o.id;
        e.apply(Command::Submit(o), Ts::ZERO, 0);
        e.apply(Command::Cancel(id), Ts::ZERO, 0);
        let ev = e.apply(Command::Cancel(id), Ts::ZERO, 0);
        assert!(matches!(
            ev[0],
            BookEvent::Rejected {
                reason: RejectReason::UnknownOrder,
                ..
            }
        ));
    }

    #[test]
    fn cancel_all_for_strategy_spares_the_others() {
        let mut e = engine();
        for (m, s, c) in [(A, 1u16, 40), (A, 1, 41), (B, 1, 42), (A, 2, 30)] {
            let o = order(&mut e, m, s, Side::Buy, c, 10);
            e.apply(Command::Submit(o), Ts::ZERO, 0);
        }
        let ev = e.apply(Command::CancelAllForStrategy(StrategyId(1)), Ts::ZERO, 0);
        assert_eq!(ev.len(), 3, "three orders belonged to strategy 1");
        assert!(ev.iter().all(|x| matches!(x, BookEvent::Cancelled { .. })));
        // Strategy 2's order survives, and is still the best bid in A.
        assert_eq!(e.book(A).unwrap().best_bid(), Some(Price::from_cents(30)));
        assert!(e.book(B).unwrap().is_empty());
        e.debug_check();
    }

    #[test]
    fn a_strategys_filled_orders_do_not_break_cancel_all() {
        let mut e = engine();
        let resting = order(&mut e, A, 1, Side::Sell, 40, 10);
        let resting_id = resting.id;
        e.apply(Command::Submit(resting), Ts::ZERO, 0);
        let taker = order(&mut e, A, 2, Side::Buy, 40, 10);
        e.apply(Command::Submit(taker), Ts::ZERO, 0);

        // Strategy 1's only order is gone, filled rather than cancelled. The
        // lazily-pruned index still lists it; cancelling must cope.
        let ev = e.apply(Command::CancelAllForStrategy(StrategyId(1)), Ts::ZERO, 0);
        assert!(
            ev.iter().all(|x| matches!(
                x,
                BookEvent::Rejected {
                    reason: RejectReason::UnknownOrder,
                    ..
                }
            )),
            "a stale index entry must reject, not panic"
        );
        assert_eq!(ev[0].order_id(), resting_id);
        e.debug_check();
    }

    #[test]
    fn the_kill_switch_empties_every_book() {
        let mut e = engine();
        for (m, c) in [(A, 40), (A, 41), (B, 42)] {
            let o = order(&mut e, m, 1, Side::Buy, c, 10);
            e.apply(Command::Submit(o), Ts::ZERO, 0);
        }
        let n = e.apply(Command::CancelAll, Ts::ZERO, 0).len();
        assert_eq!(n, 3);
        assert!(e.book(A).unwrap().is_empty() && e.book(B).unwrap().is_empty());
        e.debug_check();
    }

    #[test]
    fn clear_market_leaves_other_markets_alone() {
        let mut e = engine();
        for (m, c) in [(A, 40), (B, 42)] {
            let o = order(&mut e, m, 1, Side::Buy, c, 10);
            e.apply(Command::Submit(o), Ts::ZERO, 0);
        }
        e.apply(Command::ClearMarket(A), Ts::ZERO, 0);
        assert!(e.book(A).unwrap().is_empty());
        assert_eq!(e.book(B).unwrap().best_bid(), Some(Price::from_cents(42)));
    }

    #[test]
    fn reduce_reports_the_new_resting_size() {
        let mut e = engine();
        let o = order(&mut e, A, 1, Side::Buy, 40, 10);
        let id = o.id;
        e.apply(Command::Submit(o), Ts::ZERO, 0);
        let ev = e.apply(Command::Reduce(id, Qty(4)), Ts::ZERO, 0);
        assert!(matches!(
            ev[0],
            BookEvent::Accepted { resting_qty, .. } if resting_qty == Qty(4)
        ));
        assert_eq!(e.book(A).unwrap().best_bid_qty(), Qty(4));

        // To zero is a cancel.
        let ev = e.apply(Command::Reduce(id, Qty(0)), Ts::ZERO, 0);
        assert!(matches!(ev[0], BookEvent::Cancelled { .. }));
        assert!(e.book(A).unwrap().is_empty());
    }

    #[test]
    fn stats_track_the_session() {
        let mut e = engine();
        let maker = order(&mut e, A, 1, Side::Sell, 40, 10);
        e.apply(Command::Submit(maker), Ts::ZERO, 0);
        let taker = order(&mut e, A, 2, Side::Buy, 40, 4);
        e.apply(Command::Submit(taker), Ts::ZERO, 0);

        let s = e.stats();
        assert_eq!(s.commands, 2);
        assert_eq!(s.fills, 1);
        assert_eq!(s.volume, 4);
        assert_eq!(s.orders_accepted, 1, "only the resting maker was accepted");
    }

    #[test]
    fn order_ids_are_unique_and_monotonic() {
        let mut e = engine();
        let ids: Vec<OrderId> = (0..100).map(|_| e.next_order_id()).collect();
        assert!(ids.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(ids.len(), 100);
    }

    #[test]
    fn latency_is_recorded_per_command() {
        let mut e = engine();
        for i in 0..100 {
            let o = order(&mut e, A, 1, Side::Buy, 1 + (i % 90), 10);
            e.apply(Command::Submit(o), Ts::ZERO, 250 + i as u64);
        }
        let l = e.latency();
        assert_eq!(l.count, 100);
        assert!(l.p50_ns > 0 && l.max_ns >= 349);
    }

    #[test]
    fn a_replayed_command_stream_reproduces_the_state_exactly() {
        // The property the whole design exists for: same input, same output.
        let build = || {
            let mut e = engine();
            let mut log = Vec::new();
            let mut rng: u64 = 0x9E37_79B9_7F4A_7C15;
            let mut next = move || {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                rng
            };
            let mut live: Vec<OrderId> = Vec::new();
            for _ in 0..2000 {
                if next() % 4 == 0 && !live.is_empty() {
                    let id = live.swap_remove((next() as usize) % live.len());
                    for ev in e.apply(Command::Cancel(id), Ts::ZERO, 0) {
                        log.push(format!("{ev:?}"));
                    }
                } else {
                    let m = if next() % 2 == 0 { A } else { B };
                    let side = if next() % 2 == 0 { Side::Buy } else { Side::Sell };
                    let o = order(
                        &mut e,
                        m,
                        (next() % 3) as u16 + 1,
                        side,
                        1 + (next() % 98) as i64,
                        1 + (next() % 20) as i64,
                    );
                    let id = o.id;
                    for ev in e.apply(Command::Submit(o), Ts::ZERO, 0) {
                        log.push(format!("{ev:?}"));
                    }
                    live.push(id);
                }
            }
            e.debug_check();
            (log, e.sequence(), e.stats())
        };

        let first = build();
        let second = build();
        assert_eq!(first.0, second.0, "event streams diverged");
        assert_eq!(first.1, second.1, "sequence counters diverged");
        assert_eq!(first.2, second.2, "statistics diverged");
        assert!(first.2.fills > 0, "the fixture should actually trade");
    }
}
