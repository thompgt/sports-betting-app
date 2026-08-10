//! The limit order book.
//!
//! # Design
//!
//! Two structures, chosen for the shape of this market rather than inherited
//! from equities practice:
//!
//! **Price levels are a flat array indexed by tick.** A binary contract trades
//! in `(0, $1)` on a fixed grid, so the entire price space is at most 999 slots.
//! Finding the best bid is `leading_zeros` over a 16-word bitmap
//! ([`TickBitset`]) rather than a tree descent, and inserting at a price is a
//! direct index rather than a lookup. There is no `O(log n)` anywhere in the hot
//! path.
//!
//! **Orders live in an intrusive doubly-linked list over a slab.** Each price
//! level is a FIFO queue threaded through a single `Vec` of nodes with a free
//! list. Cancelling is `O(1)` — unlink and return the slot — with no allocation
//! and no scan. This matters more than raw match speed: a market maker cancels
//! and re-quotes far more often than it trades, so cancel is the real hot path.
//!
//! The cost of the flat array is memory proportional to the tick grid per
//! market (~24 KB at a tenth-of-a-cent tick), which is why books are kept only
//! for markets actually being traded rather than for the whole venue catalogue.
//!
//! # Invariants
//!
//! - The book never contains a crossed price. Incoming orders match on entry,
//!   so anything resting is non-marketable against the other side.
//! - `bids`/`asks` bitmaps agree exactly with which levels are non-empty.
//! - A level's `qty` equals the sum of the `remaining` of its orders.
//! - Every id in `index` refers to an occupied slab slot, and vice versa.
//!
//! [`debug_check`](OrderBook::debug_check) asserts all four; the tests call it
//! after every mutation.

use std::collections::HashMap;

use edge_core::types::{MICROS, MarketId, Notional, OrderId, Price, Qty, Side, Ts};

use crate::bitset::TickBitset;
use crate::order::{BookEvent, Fill, Order, RejectReason, SelfTradePrevention, TimeInForce};

const NIL: u32 = u32::MAX;

#[derive(Debug, Clone, Copy)]
struct Node {
    order: Order,
    prev: u32,
    next: u32,
    occupied: bool,
}

#[derive(Debug, Clone, Copy)]
struct Level {
    head: u32,
    tail: u32,
    qty: i64,
    count: u32,
}

impl Level {
    const EMPTY: Level = Level { head: NIL, tail: NIL, qty: 0, count: 0 };

    #[inline]
    fn is_empty(&self) -> bool {
        self.head == NIL
    }
}

/// One side of the book at one price.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LevelView {
    pub price: Price,
    pub qty: Qty,
    pub orders: u32,
}

#[derive(Debug)]
pub struct OrderBook {
    market: MarketId,
    tick_size: i64,
    n_ticks: usize,
    levels: Vec<Level>,
    bids: TickBitset,
    asks: TickBitset,
    nodes: Vec<Node>,
    free: Vec<u32>,
    index: HashMap<OrderId, u32>,
    last_trade: Option<Price>,
}

impl OrderBook {
    /// `tick_size` is in micro-dollars and must divide $1 exactly.
    pub fn new(market: MarketId, tick_size: i64) -> Self {
        assert!(tick_size > 0 && MICROS % tick_size == 0, "tick must divide $1");
        let n_ticks = (MICROS / tick_size) as usize;
        OrderBook {
            market,
            tick_size,
            n_ticks,
            levels: vec![Level::EMPTY; n_ticks],
            bids: TickBitset::new(n_ticks),
            asks: TickBitset::new(n_ticks),
            nodes: Vec::with_capacity(1024),
            free: Vec::new(),
            index: HashMap::new(),
            last_trade: None,
        }
    }

    #[inline]
    pub fn market(&self) -> MarketId {
        self.market
    }

    #[inline]
    pub fn tick_size(&self) -> i64 {
        self.tick_size
    }

    #[inline]
    fn tick(&self, p: Price) -> Option<usize> {
        let m = p.micros();
        if m <= 0 || m >= MICROS || m % self.tick_size != 0 {
            return None;
        }
        Some((m / self.tick_size) as usize)
    }

    #[inline]
    fn price_at(&self, tick: usize) -> Price {
        Price(tick as i64 * self.tick_size)
    }

    #[inline]
    fn side_bitmap(&self, side: Side) -> &TickBitset {
        match side {
            Side::Buy => &self.bids,
            Side::Sell => &self.asks,
        }
    }

    // -- top of book ------------------------------------------------------

    pub fn best_bid(&self) -> Option<Price> {
        self.bids.highest().map(|t| self.price_at(t))
    }

    pub fn best_ask(&self) -> Option<Price> {
        self.asks.lowest().map(|t| self.price_at(t))
    }

    pub fn best(&self, side: Side) -> Option<Price> {
        match side {
            Side::Buy => self.best_bid(),
            Side::Sell => self.best_ask(),
        }
    }

    pub fn qty_at(&self, side: Side, price: Price) -> Qty {
        match self.tick(price) {
            Some(t) if self.side_bitmap(side).get(t) => Qty(self.levels[t].qty),
            _ => Qty::ZERO,
        }
    }

    pub fn best_bid_qty(&self) -> Qty {
        self.best_bid().map(|p| self.qty_at(Side::Buy, p)).unwrap_or(Qty::ZERO)
    }

    pub fn best_ask_qty(&self) -> Qty {
        self.best_ask().map(|p| self.qty_at(Side::Sell, p)).unwrap_or(Qty::ZERO)
    }

    /// Arithmetic mid. `None` when either side is empty — and a one-sided book
    /// is common in prediction markets, so callers must handle it rather than
    /// substituting a last trade price and pretending it is a quote.
    pub fn mid(&self) -> Option<Price> {
        match (self.best_bid(), self.best_ask()) {
            (Some(b), Some(a)) => Some(Price((b.micros() + a.micros()) / 2)),
            _ => None,
        }
    }

    pub fn spread(&self) -> Option<Price> {
        match (self.best_bid(), self.best_ask()) {
            (Some(b), Some(a)) => Some(a - b),
            _ => None,
        }
    }

    /// Size-weighted mid. A better one-tick-ahead predictor than the arithmetic
    /// mid: when the bid is three times the size of the offer, the next trade is
    /// far more likely to lift the offer, and the fair value sits nearer to it.
    pub fn microprice(&self) -> Option<Price> {
        let (b, a) = (self.best_bid()?, self.best_ask()?);
        let (bq, aq) = (self.best_bid_qty().get() as f64, self.best_ask_qty().get() as f64);
        let total = bq + aq;
        if total <= 0.0 {
            return self.mid();
        }
        // Weight each side by the *opposite* side's size.
        Some(Price(((b.micros() as f64 * aq + a.micros() as f64 * bq) / total).round() as i64))
    }

    /// Queue imbalance over the top `depth` levels, in `[-1, 1]`. Positive means
    /// bid-heavy. The primary microstructure feature for short-horizon direction.
    pub fn imbalance(&self, depth: usize) -> Option<f64> {
        let bid: i64 = self.depth(Side::Buy, depth).iter().map(|l| l.qty.get()).sum();
        let ask: i64 = self.depth(Side::Sell, depth).iter().map(|l| l.qty.get()).sum();
        let total = bid + ask;
        if total <= 0 {
            return None;
        }
        Some((bid - ask) as f64 / total as f64)
    }

    /// Best `n` levels on a side, best price first.
    pub fn depth(&self, side: Side, n: usize) -> Vec<LevelView> {
        let bm = self.side_bitmap(side);
        let ticks: Vec<usize> = match side {
            Side::Buy => bm.iter_descending().take(n).collect(),
            Side::Sell => bm.iter_ascending().take(n).collect(),
        };
        ticks
            .into_iter()
            .map(|t| LevelView {
                price: self.price_at(t),
                qty: Qty(self.levels[t].qty),
                orders: self.levels[t].count,
            })
            .collect()
    }

    pub fn total_qty(&self, side: Side) -> Qty {
        Qty(self.side_bitmap(side).iter_ascending().map(|t| self.levels[t].qty).sum())
    }

    pub fn last_trade(&self) -> Option<Price> {
        self.last_trade
    }

    pub fn order(&self, id: OrderId) -> Option<&Order> {
        self.index.get(&id).map(|&i| &self.nodes[i as usize].order)
    }

    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Cost of immediately taking `qty` contracts, walking the book level by
    /// level. Returns `None` if the book cannot fill the whole size.
    ///
    /// This is what a strategy must consult before deciding an edge is real. An
    /// edge measured against the top of book that requires ten times the
    /// top-of-book size to capture is not an edge, and comparing against the
    /// touch price is how a backtest manufactures profit that evaporates live.
    pub fn sweep_cost(&self, side: Side, qty: Qty) -> Option<(Notional, Price)> {
        let want = qty.get().abs();
        if want == 0 {
            return Some((Notional::ZERO, Price::ZERO));
        }
        // Taking on `side` consumes the opposite side of the book.
        let opposite = side.opposite();
        let bm = self.side_bitmap(opposite);
        let ticks: Box<dyn Iterator<Item = usize> + '_> = match opposite {
            Side::Buy => Box::new(bm.iter_descending()),
            Side::Sell => Box::new(bm.iter_ascending()),
        };
        let mut left = want;
        let mut cost: i64 = 0;
        for t in ticks {
            let avail = self.levels[t].qty;
            let take = avail.min(left);
            cost += take * self.price_at(t).micros();
            left -= take;
            if left == 0 {
                break;
            }
        }
        if left > 0 {
            return None;
        }
        Some((Notional(cost), Price(cost / want)))
    }

    // -- mutation ---------------------------------------------------------

    fn alloc(&mut self, order: Order) -> u32 {
        if let Some(i) = self.free.pop() {
            self.nodes[i as usize] = Node { order, prev: NIL, next: NIL, occupied: true };
            i
        } else {
            self.nodes.push(Node { order, prev: NIL, next: NIL, occupied: true });
            (self.nodes.len() - 1) as u32
        }
    }

    /// Append to a price level's FIFO tail. Time priority is the list order.
    fn rest(&mut self, order: Order) -> Result<(), RejectReason> {
        let t = self.tick(order.price).ok_or(RejectReason::InvalidPrice)?;
        if self.index.contains_key(&order.id) {
            return Err(RejectReason::DuplicateOrderId);
        }
        let remaining = order.remaining.get();
        let id = order.id;
        let side = order.side;
        let node = self.alloc(order);

        let level = &mut self.levels[t];
        if level.is_empty() {
            level.head = node;
            level.tail = node;
        } else {
            let old_tail = level.tail;
            self.nodes[old_tail as usize].next = node;
            self.nodes[node as usize].prev = old_tail;
            self.levels[t].tail = node;
        }
        let level = &mut self.levels[t];
        level.qty += remaining;
        level.count += 1;

        match side {
            Side::Buy => self.bids.set(t),
            Side::Sell => self.asks.set(t),
        }
        self.index.insert(id, node);
        Ok(())
    }

    /// Unlink a node and free its slot.
    fn unlink(&mut self, node: u32) -> Order {
        let n = self.nodes[node as usize];
        let order = n.order;
        let t = self.tick(order.price).expect("a resting order is always on the grid");

        if n.prev != NIL {
            self.nodes[n.prev as usize].next = n.next;
        } else {
            self.levels[t].head = n.next;
        }
        if n.next != NIL {
            self.nodes[n.next as usize].prev = n.prev;
        } else {
            self.levels[t].tail = n.prev;
        }

        let level = &mut self.levels[t];
        level.qty -= order.remaining.get();
        level.count -= 1;
        if level.is_empty() {
            *level = Level::EMPTY;
            match order.side {
                Side::Buy => self.bids.clear(t),
                Side::Sell => self.asks.clear(t),
            }
        }

        self.nodes[node as usize].occupied = false;
        self.nodes[node as usize].prev = NIL;
        self.nodes[node as usize].next = NIL;
        self.free.push(node);
        self.index.remove(&order.id);
        order
    }

    /// Remove a resting order. `O(1)`.
    pub fn cancel(&mut self, id: OrderId) -> Option<Order> {
        let node = *self.index.get(&id)?;
        Some(self.unlink(node))
    }

    /// Reduce a resting order's size without losing queue position.
    ///
    /// Increasing size would require going to the back of the queue, which is a
    /// cancel-replace and is deliberately not offered here — silently losing
    /// priority is the kind of surprise that quietly costs a market maker its
    /// entire edge.
    pub fn reduce(&mut self, id: OrderId, new_qty: Qty) -> Result<Qty, RejectReason> {
        let node = *self.index.get(&id).ok_or(RejectReason::UnknownOrder)?;
        let cur = self.nodes[node as usize].order.remaining;
        if new_qty.get() <= 0 {
            self.unlink(node);
            return Ok(Qty::ZERO);
        }
        if new_qty.get() >= cur.get() {
            return Err(RejectReason::InvalidQty);
        }
        let delta = cur.get() - new_qty.get();
        let price = self.nodes[node as usize].order.price;
        let t = self.tick(price).expect("resting order is on the grid");
        self.nodes[node as usize].order.remaining = new_qty;
        self.levels[t].qty -= delta;
        Ok(new_qty)
    }

    /// How much of `qty` could be filled immediately at prices crossing `limit`.
    /// Used to decide a fill-or-kill before touching the book.
    pub fn fillable(&self, side: Side, limit: Price) -> Qty {
        let opposite = side.opposite();
        let bm = self.side_bitmap(opposite);
        let ticks: Box<dyn Iterator<Item = usize> + '_> = match opposite {
            Side::Buy => Box::new(bm.iter_descending()),
            Side::Sell => Box::new(bm.iter_ascending()),
        };
        let mut total = 0;
        for t in ticks {
            if !side.crosses(limit, self.price_at(t)) {
                break;
            }
            total += self.levels[t].qty;
        }
        Qty(total)
    }

    /// Would this order take liquidity if submitted?
    pub fn would_cross(&self, side: Side, limit: Price) -> bool {
        match self.best(side.opposite()) {
            Some(book) => side.crosses(limit, book),
            None => false,
        }
    }

    // -- matching ---------------------------------------------------------

    /// Submit an order: validate, match against the book, then rest, cancel or
    /// expire the remainder according to its time-in-force.
    ///
    /// `seq` is a global counter the caller owns, so events are totally ordered
    /// across every market in the engine. Nothing here reads a clock.
    pub fn submit(&mut self, mut order: Order, seq: &mut u64, out: &mut Vec<BookEvent>) {
        let market = self.market;
        let mut next = || {
            *seq += 1;
            *seq
        };

        macro_rules! reject {
            ($why:expr) => {{
                out.push(BookEvent::Rejected {
                    seq: next(),
                    order: order.id,
                    market,
                    reason: $why,
                });
                return;
            }};
        }

        if order.market != market {
            reject!(RejectReason::MarketNotTradable);
        }
        if order.remaining.get() <= 0 || order.qty.get() <= 0 {
            reject!(RejectReason::InvalidQty);
        }
        if self.index.contains_key(&order.id) {
            reject!(RejectReason::DuplicateOrderId);
        }
        let limit = order.effective_limit();
        if order.order_type == crate::order::OrderType::Limit && self.tick(order.price).is_none() {
            reject!(RejectReason::InvalidPrice);
        }

        if order.tif == TimeInForce::PostOnly && self.would_cross(order.side, limit) {
            reject!(RejectReason::WouldCross);
        }
        if order.tif == TimeInForce::Fok && self.fillable(order.side, limit) < order.remaining {
            reject!(RejectReason::InsufficientLiquidity);
        }

        let mut stopped_by_stp = false;

        'sweep: while order.remaining.get() > 0 {
            let opposite = order.side.opposite();
            let Some(best_tick) = (match opposite {
                Side::Buy => self.bids.highest(),
                Side::Sell => self.asks.lowest(),
            }) else {
                break;
            };
            let level_price = self.price_at(best_tick);
            if !order.side.crosses(limit, level_price) {
                break;
            }

            // Walk this level's FIFO queue from the front.
            while order.remaining.get() > 0 {
                let head = self.levels[best_tick].head;
                if head == NIL {
                    break;
                }
                let maker = self.nodes[head as usize].order;

                if maker.strategy == order.strategy && order.stp != SelfTradePrevention::Allow {
                    match order.stp {
                        SelfTradePrevention::CancelResting => {
                            let removed = self.unlink(head);
                            out.push(BookEvent::Cancelled {
                                seq: next(),
                                order: removed.id,
                                market,
                                remaining: removed.remaining,
                            });
                            continue;
                        }
                        SelfTradePrevention::CancelIncoming => {
                            stopped_by_stp = true;
                            break 'sweep;
                        }
                        SelfTradePrevention::DecrementBoth => {
                            let overlap = order.remaining.min(maker.remaining);
                            order.remaining -= overlap;
                            let left = maker.remaining - overlap;
                            if left.get() <= 0 {
                                let removed = self.unlink(head);
                                out.push(BookEvent::Cancelled {
                                    seq: next(),
                                    order: removed.id,
                                    market,
                                    remaining: Qty::ZERO,
                                });
                            } else {
                                let _ = self.reduce(maker.id, left);
                            }
                            continue;
                        }
                        SelfTradePrevention::Allow => unreachable!(),
                    }
                }

                let traded = order.remaining.min(maker.remaining);
                // The maker set the price; the taker accepted it.
                out.push(BookEvent::Filled(Fill {
                    seq: next(),
                    market,
                    price: level_price,
                    qty: traded,
                    taker_order: order.id,
                    maker_order: maker.id,
                    taker_strategy: order.strategy,
                    maker_strategy: maker.strategy,
                    taker_side: order.side,
                    ts: order.ts,
                }));
                self.last_trade = Some(level_price);
                order.remaining -= traded;

                let maker_left = maker.remaining - traded;
                if maker_left.get() <= 0 {
                    self.unlink(head);
                } else {
                    let t = best_tick;
                    self.nodes[head as usize].order.remaining = maker_left;
                    self.levels[t].qty -= traded.get();
                }
            }
        }

        if order.remaining.get() <= 0 {
            return;
        }

        if stopped_by_stp || !order.tif.rests() {
            out.push(BookEvent::Expired {
                seq: next(),
                order: order.id,
                market,
                remaining: order.remaining,
            });
            return;
        }

        // A market order that has exhausted the book has nothing to rest at.
        if order.order_type == crate::order::OrderType::Market {
            out.push(BookEvent::Expired {
                seq: next(),
                order: order.id,
                market,
                remaining: order.remaining,
            });
            return;
        }

        let resting = order.remaining;
        match self.rest(order) {
            Ok(()) => out.push(BookEvent::Accepted {
                seq: next(),
                order: order.id,
                market,
                resting_qty: resting,
            }),
            Err(reason) => {
                out.push(BookEvent::Rejected { seq: next(), order: order.id, market, reason })
            }
        }
    }

    /// Drop every resting order, reporting each. Used on a venue disconnect,
    /// when the book must be rebuilt from a fresh snapshot, and by the kill
    /// switch.
    pub fn clear(&mut self, seq: &mut u64, ts: Ts, out: &mut Vec<BookEvent>) {
        let _ = ts;
        let ids: Vec<OrderId> = self.index.keys().copied().collect();
        for id in ids {
            if let Some(o) = self.cancel(id) {
                *seq += 1;
                out.push(BookEvent::Cancelled {
                    seq: *seq,
                    order: o.id,
                    market: self.market,
                    remaining: o.remaining,
                });
            }
        }
    }

    /// Assert every structural invariant. `O(n)`, for tests and for a debug
    /// build's periodic self-check — a book that has silently desynchronised
    /// from its bitmaps will quote prices that do not exist.
    pub fn debug_check(&self) {
        assert!(
            match (self.best_bid(), self.best_ask()) {
                (Some(b), Some(a)) => b < a,
                _ => true,
            },
            "book is crossed: {:?} / {:?}",
            self.best_bid(),
            self.best_ask()
        );

        let mut counted = 0usize;
        for t in 0..self.n_ticks {
            let level = &self.levels[t];
            let occupied = self.bids.get(t) || self.asks.get(t);
            assert_eq!(occupied, !level.is_empty(), "bitmap disagrees with level {t}");
            assert!(!(self.bids.get(t) && self.asks.get(t)), "level {t} is on both sides");
            if level.is_empty() {
                assert_eq!(level.qty, 0, "empty level {t} has qty");
                continue;
            }
            let mut sum = 0i64;
            let mut n = 0u32;
            let mut cur = level.head;
            let mut prev = NIL;
            while cur != NIL {
                let node = &self.nodes[cur as usize];
                assert!(node.occupied, "level {t} links a freed slot");
                assert_eq!(node.prev, prev, "back-link broken at level {t}");
                assert_eq!(
                    self.index.get(&node.order.id),
                    Some(&cur),
                    "index disagrees with list at level {t}"
                );
                sum += node.order.remaining.get();
                n += 1;
                prev = cur;
                cur = node.next;
            }
            assert_eq!(prev, level.tail, "tail pointer wrong at level {t}");
            assert_eq!(sum, level.qty, "level {t} qty is stale");
            assert_eq!(n, level.count, "level {t} count is stale");
            counted += n as usize;
        }
        assert_eq!(counted, self.index.len(), "orphaned index entries");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::order::{OrderType, TimeInForce};
    use edge_core::types::{MarketId, OrderId, StrategyId};

    const MKT: MarketId = MarketId(7);
    const CENT: i64 = 10_000;

    struct Fixture {
        book: OrderBook,
        seq: u64,
        next_id: u64,
    }

    impl Fixture {
        fn new() -> Self {
            Fixture { book: OrderBook::new(MKT, CENT), seq: 0, next_id: 0 }
        }

        fn submit(
            &mut self,
            strategy: u16,
            side: Side,
            cents: i64,
            qty: i64,
        ) -> (OrderId, Vec<BookEvent>) {
            self.submit_with(strategy, side, cents, qty, |o| o)
        }

        fn submit_with(
            &mut self,
            strategy: u16,
            side: Side,
            cents: i64,
            qty: i64,
            f: impl FnOnce(Order) -> Order,
        ) -> (OrderId, Vec<BookEvent>) {
            self.next_id += 1;
            let id = OrderId(self.next_id);
            let order = f(Order::limit(
                id,
                MKT,
                StrategyId(strategy),
                side,
                Price::from_cents(cents),
                Qty(qty),
            ));
            let mut out = Vec::new();
            self.book.submit(order, &mut self.seq, &mut out);
            self.book.debug_check();
            (id, out)
        }

        fn fills(events: &[BookEvent]) -> Vec<Fill> {
            events.iter().filter_map(|e| e.as_fill().copied()).collect()
        }
    }

    #[test]
    fn a_resting_order_becomes_the_touch() {
        let mut f = Fixture::new();
        let (_, ev) = f.submit(1, Side::Buy, 40, 100);
        assert!(matches!(ev[0], BookEvent::Accepted { .. }));
        assert_eq!(f.book.best_bid(), Some(Price::from_cents(40)));
        assert_eq!(f.book.best_bid_qty(), Qty(100));
        assert_eq!(f.book.best_ask(), None);
        assert_eq!(f.book.mid(), None, "a one-sided book has no mid");
        assert_eq!(f.book.spread(), None);
    }

    #[test]
    fn price_priority_beats_arrival_order() {
        let mut f = Fixture::new();
        f.submit(1, Side::Sell, 45, 10);
        f.submit(1, Side::Sell, 43, 10); // later, but better
        assert_eq!(f.book.best_ask(), Some(Price::from_cents(43)));

        let (_, ev) = f.submit(2, Side::Buy, 50, 10);
        let fills = Fixture::fills(&ev);
        assert_eq!(fills.len(), 1);
        assert_eq!(fills[0].price, Price::from_cents(43), "must hit the better offer first");
    }

    #[test]
    fn time_priority_is_fifo_within_a_level() {
        let mut f = Fixture::new();
        let (first, _) = f.submit(1, Side::Sell, 45, 10);
        let (second, _) = f.submit(1, Side::Sell, 45, 10);

        let (_, ev) = f.submit(2, Side::Buy, 45, 15);
        let fills = Fixture::fills(&ev);
        assert_eq!(fills.len(), 2);
        assert_eq!(fills[0].maker_order, first, "the earlier order fills first");
        assert_eq!(fills[0].qty, Qty(10));
        assert_eq!(fills[1].maker_order, second);
        assert_eq!(fills[1].qty, Qty(5));
        assert_eq!(f.book.qty_at(Side::Sell, Price::from_cents(45)), Qty(5));
    }

    #[test]
    fn the_maker_price_is_the_trade_price() {
        let mut f = Fixture::new();
        f.submit(1, Side::Sell, 40, 10);
        // Aggressor is willing to pay 60c but the resting offer is 40c.
        let (_, ev) = f.submit(2, Side::Buy, 60, 10);
        let fills = Fixture::fills(&ev);
        assert_eq!(
            fills[0].price,
            Price::from_cents(40),
            "the taker must not be charged their own limit"
        );
        assert_eq!(f.book.last_trade(), Some(Price::from_cents(40)));
    }

    #[test]
    fn an_aggressive_order_sweeps_multiple_levels() {
        let mut f = Fixture::new();
        f.submit(1, Side::Sell, 40, 10);
        f.submit(1, Side::Sell, 41, 10);
        f.submit(1, Side::Sell, 42, 10);

        let (_, ev) = f.submit(2, Side::Buy, 42, 25);
        let fills = Fixture::fills(&ev);
        assert_eq!(fills.len(), 3);
        assert_eq!(
            fills.iter().map(|x| x.price.micros()).collect::<Vec<_>>(),
            vec![400_000, 410_000, 420_000],
            "levels are consumed best-first"
        );
        assert_eq!(fills.iter().map(|x| x.qty.get()).sum::<i64>(), 25);
        assert_eq!(f.book.best_ask(), Some(Price::from_cents(42)));
        assert_eq!(f.book.best_ask_qty(), Qty(5));
    }

    #[test]
    fn the_book_never_ends_up_crossed() {
        let mut f = Fixture::new();
        f.submit(1, Side::Sell, 40, 10);
        // A 45c bid for more than is offered: it takes 10 and rests 20 at 45c,
        // which would cross if the offer had not been consumed first.
        f.submit(2, Side::Buy, 45, 30);
        f.book.debug_check();
        assert_eq!(f.book.best_bid(), Some(Price::from_cents(45)));
        assert_eq!(f.book.best_ask(), None);
        assert_eq!(f.book.best_bid_qty(), Qty(20));
    }

    #[test]
    fn cancel_removes_the_level_when_it_empties() {
        let mut f = Fixture::new();
        let (a, _) = f.submit(1, Side::Buy, 40, 10);
        let (b, _) = f.submit(1, Side::Buy, 40, 10);
        assert_eq!(f.book.qty_at(Side::Buy, Price::from_cents(40)), Qty(20));

        assert_eq!(f.book.cancel(a).unwrap().remaining, Qty(10));
        f.book.debug_check();
        assert_eq!(f.book.qty_at(Side::Buy, Price::from_cents(40)), Qty(10));

        f.book.cancel(b);
        f.book.debug_check();
        assert_eq!(f.book.best_bid(), None);
        assert!(f.book.is_empty());
    }

    #[test]
    fn cancelling_an_unknown_order_is_none_not_a_panic() {
        let mut f = Fixture::new();
        assert!(f.book.cancel(OrderId(999)).is_none());
    }

    #[test]
    fn cancelled_slots_are_reused() {
        let mut f = Fixture::new();
        for _ in 0..100 {
            let (id, _) = f.submit(1, Side::Buy, 40, 1);
            f.book.cancel(id);
        }
        f.book.debug_check();
        assert!(f.book.is_empty());
        // 100 submit/cancel cycles must not grow the slab past a handful.
        assert!(f.book.nodes.len() <= 2, "slab grew to {}", f.book.nodes.len());
    }

    #[test]
    fn reduce_keeps_queue_position_and_refuses_to_grow() {
        let mut f = Fixture::new();
        let (first, _) = f.submit(1, Side::Sell, 45, 10);
        f.submit(1, Side::Sell, 45, 10);

        assert_eq!(f.book.reduce(first, Qty(4)).unwrap(), Qty(4));
        f.book.debug_check();
        assert_eq!(f.book.qty_at(Side::Sell, Price::from_cents(45)), Qty(14));

        // Growing would silently forfeit priority, so it is rejected.
        assert_eq!(f.book.reduce(first, Qty(20)), Err(RejectReason::InvalidQty));
        assert_eq!(f.book.reduce(OrderId(404), Qty(1)), Err(RejectReason::UnknownOrder));

        // The reduced order is still at the front.
        let (_, ev) = f.submit(2, Side::Buy, 45, 4);
        assert_eq!(Fixture::fills(&ev)[0].maker_order, first);
    }

    #[test]
    fn reducing_to_zero_cancels() {
        let mut f = Fixture::new();
        let (id, _) = f.submit(1, Side::Sell, 45, 10);
        assert_eq!(f.book.reduce(id, Qty(0)).unwrap(), Qty::ZERO);
        f.book.debug_check();
        assert!(f.book.is_empty());
    }

    #[test]
    fn post_only_is_rejected_rather_than_taking() {
        let mut f = Fixture::new();
        f.submit(1, Side::Sell, 45, 10);
        let (_, ev) = f.submit_with(2, Side::Buy, 45, 10, |o| o.with_tif(TimeInForce::PostOnly));
        assert!(matches!(ev[0], BookEvent::Rejected { reason: RejectReason::WouldCross, .. }));
        assert_eq!(f.book.best_ask_qty(), Qty(10), "the book is untouched");

        // One tick away it rests happily.
        let (_, ev) = f.submit_with(2, Side::Buy, 44, 10, |o| o.with_tif(TimeInForce::PostOnly));
        assert!(matches!(ev[0], BookEvent::Accepted { .. }));
    }

    #[test]
    fn ioc_fills_what_it_can_and_expires_the_rest() {
        let mut f = Fixture::new();
        f.submit(1, Side::Sell, 45, 10);
        let (_, ev) = f.submit_with(2, Side::Buy, 45, 25, |o| o.with_tif(TimeInForce::Ioc));
        assert_eq!(Fixture::fills(&ev).len(), 1);
        assert!(matches!(
            ev.last().unwrap(),
            BookEvent::Expired { remaining, .. } if *remaining == Qty(15)
        ));
        assert!(f.book.is_empty(), "an IOC never rests");
    }

    #[test]
    fn fok_is_all_or_nothing() {
        let mut f = Fixture::new();
        f.submit(1, Side::Sell, 45, 10);

        let (_, ev) = f.submit_with(2, Side::Buy, 45, 25, |o| o.with_tif(TimeInForce::Fok));
        assert!(matches!(
            ev[0],
            BookEvent::Rejected { reason: RejectReason::InsufficientLiquidity, .. }
        ));
        assert_eq!(f.book.best_ask_qty(), Qty(10), "a failed FOK touches nothing");

        let (_, ev) = f.submit_with(2, Side::Buy, 45, 10, |o| o.with_tif(TimeInForce::Fok));
        assert_eq!(Fixture::fills(&ev).len(), 1);
        assert!(f.book.is_empty());
    }

    #[test]
    fn a_market_order_takes_everything_available_then_expires() {
        let mut f = Fixture::new();
        f.submit(1, Side::Sell, 40, 5);
        f.submit(1, Side::Sell, 90, 5);
        let (_, ev) = f.submit_with(2, Side::Buy, 1, 20, |o| {
            o.with_type(OrderType::Market).with_tif(TimeInForce::Ioc)
        });
        let fills = Fixture::fills(&ev);
        assert_eq!(fills.len(), 2, "a market order ignores its own limit price");
        assert_eq!(fills[1].price, Price::from_cents(90));
        assert!(matches!(
            ev.last().unwrap(),
            BookEvent::Expired { remaining, .. } if *remaining == Qty(10)
        ));
        assert!(f.book.is_empty(), "a market order never rests");
    }

    #[test]
    fn off_grid_and_out_of_range_prices_are_rejected() {
        let mut f = Fixture::new();
        for bad in [Price(405_000), Price::ZERO, Price::ONE, Price(-10_000)] {
            f.next_id += 1;
            let order =
                Order::limit(OrderId(f.next_id), MKT, StrategyId(1), Side::Buy, bad, Qty(10));
            let mut out = Vec::new();
            f.book.submit(order, &mut f.seq, &mut out);
            assert!(
                matches!(out[0], BookEvent::Rejected { reason: RejectReason::InvalidPrice, .. }),
                "price {bad} should be rejected, got {:?}",
                out[0]
            );
        }
        assert!(f.book.is_empty());
    }

    #[test]
    fn zero_and_duplicate_orders_are_rejected() {
        let mut f = Fixture::new();
        let (_, ev) = f.submit(1, Side::Buy, 40, 0);
        assert!(matches!(ev[0], BookEvent::Rejected { reason: RejectReason::InvalidQty, .. }));

        let (id, _) = f.submit(1, Side::Buy, 40, 10);
        let mut out = Vec::new();
        let dup = Order::limit(id, MKT, StrategyId(1), Side::Buy, Price::from_cents(41), Qty(5));
        f.book.submit(dup, &mut f.seq, &mut out);
        assert!(matches!(
            out[0],
            BookEvent::Rejected { reason: RejectReason::DuplicateOrderId, .. }
        ));
    }

    #[test]
    fn self_trade_prevention_cancels_the_resting_order_by_default() {
        let mut f = Fixture::new();
        let (resting, _) = f.submit(1, Side::Sell, 45, 10);
        let (_, ev) = f.submit(1, Side::Buy, 45, 10);

        assert!(Fixture::fills(&ev).is_empty(), "a strategy must not trade with itself");
        assert!(ev.iter().any(|e| matches!(
            e, BookEvent::Cancelled { order, .. } if *order == resting
        )));
        // The incoming order rests, since its own liquidity is gone.
        assert_eq!(f.book.best_bid(), Some(Price::from_cents(45)));
    }

    #[test]
    fn self_trade_prevention_can_cancel_the_incoming_order_instead() {
        let mut f = Fixture::new();
        f.submit(1, Side::Sell, 45, 10);
        let (_, ev) = f
            .submit_with(1, Side::Buy, 45, 10, |o| o.with_stp(SelfTradePrevention::CancelIncoming));
        assert!(Fixture::fills(&ev).is_empty());
        assert!(matches!(ev.last().unwrap(), BookEvent::Expired { .. }));
        assert_eq!(f.book.best_ask_qty(), Qty(10), "the resting order survives");
    }

    #[test]
    fn decrement_both_removes_the_overlap_without_a_trade() {
        let mut f = Fixture::new();
        f.submit(1, Side::Sell, 45, 10);
        let (_, ev) =
            f.submit_with(1, Side::Buy, 45, 4, |o| o.with_stp(SelfTradePrevention::DecrementBoth));
        assert!(Fixture::fills(&ev).is_empty());
        assert_eq!(f.book.best_ask_qty(), Qty(6), "both sides shrink by 4");
        assert!(f.book.best_bid().is_none());
    }

    #[test]
    fn different_strategies_may_trade_with_each_other() {
        let mut f = Fixture::new();
        f.submit(1, Side::Sell, 45, 10);
        let (_, ev) = f.submit(2, Side::Buy, 45, 10);
        assert_eq!(Fixture::fills(&ev).len(), 1);
    }

    #[test]
    fn microprice_leans_toward_the_thin_side() {
        let mut f = Fixture::new();
        f.submit(1, Side::Buy, 40, 900);
        f.submit(1, Side::Sell, 50, 100);
        assert_eq!(f.book.mid(), Some(Price::from_cents(45)));
        let micro = f.book.microprice().unwrap();
        assert!(
            micro > Price::from_cents(45),
            "heavy bid should pull fair value toward the offer, got {micro}"
        );
    }

    #[test]
    fn imbalance_is_signed_and_bounded() {
        let mut f = Fixture::new();
        f.submit(1, Side::Buy, 40, 300);
        f.submit(1, Side::Sell, 50, 100);
        let i = f.book.imbalance(5).unwrap();
        assert!((i - 0.5).abs() < 1e-12, "got {i}");

        let empty = OrderBook::new(MKT, CENT);
        assert!(empty.imbalance(5).is_none());
    }

    #[test]
    fn sweep_cost_is_size_aware() {
        let mut f = Fixture::new();
        f.submit(1, Side::Sell, 40, 10);
        f.submit(1, Side::Sell, 50, 10);

        // Ten contracts clear at the touch.
        let (cost, avg) = f.book.sweep_cost(Side::Buy, Qty(10)).unwrap();
        assert_eq!(avg, Price::from_cents(40));
        assert!((cost.dollars() - 4.0).abs() < 1e-9);

        // Twenty pay the average of both levels — this is the number a strategy
        // must judge an edge against, not the 40c touch.
        let (cost, avg) = f.book.sweep_cost(Side::Buy, Qty(20)).unwrap();
        assert_eq!(avg, Price::from_cents(45));
        assert!((cost.dollars() - 9.0).abs() < 1e-9);

        assert!(f.book.sweep_cost(Side::Buy, Qty(21)).is_none(), "not enough depth");
    }

    #[test]
    fn depth_is_ordered_best_first() {
        let mut f = Fixture::new();
        for c in [38, 39, 40] {
            f.submit(1, Side::Buy, c, 10);
        }
        for c in [50, 51, 52] {
            f.submit(1, Side::Sell, c, 10);
        }
        let bids = f.book.depth(Side::Buy, 2);
        assert_eq!(bids[0].price, Price::from_cents(40));
        assert_eq!(bids[1].price, Price::from_cents(39));
        let asks = f.book.depth(Side::Sell, 2);
        assert_eq!(asks[0].price, Price::from_cents(50));
        assert_eq!(asks[1].price, Price::from_cents(51));
        assert_eq!(f.book.total_qty(Side::Buy), Qty(30));
    }

    #[test]
    fn clearing_the_book_reports_every_order() {
        let mut f = Fixture::new();
        for c in [38, 39, 40] {
            f.submit(1, Side::Buy, c, 10);
        }
        let mut out = Vec::new();
        f.book.clear(&mut f.seq, Ts::ZERO, &mut out);
        assert_eq!(out.len(), 3);
        assert!(out.iter().all(|e| matches!(e, BookEvent::Cancelled { .. })));
        assert!(f.book.is_empty());
        f.book.debug_check();
    }

    #[test]
    fn event_sequence_numbers_are_strictly_increasing() {
        let mut f = Fixture::new();
        f.submit(1, Side::Sell, 40, 10);
        f.submit(1, Side::Sell, 41, 10);
        let (_, ev) = f.submit(2, Side::Buy, 41, 30);
        let seqs: Vec<u64> = ev.iter().map(|e| e.seq()).collect();
        assert!(seqs.windows(2).all(|w| w[0] < w[1]), "{seqs:?}");
    }

    #[test]
    fn a_tenth_of_a_cent_grid_works() {
        let mut book = OrderBook::new(MKT, 1_000);
        let mut seq = 0;
        let mut out = Vec::new();
        book.submit(
            Order::limit(OrderId(1), MKT, StrategyId(1), Side::Buy, Price(404_000), Qty(10)),
            &mut seq,
            &mut out,
        );
        assert!(matches!(out[0], BookEvent::Accepted { .. }));
        assert_eq!(book.best_bid(), Some(Price(404_000)));
        book.debug_check();
    }

    #[test]
    fn a_randomised_session_preserves_every_invariant() {
        // Deterministic LCG rather than a dependency: the point is coverage of
        // interleavings, not statistical quality.
        let mut rng: u64 = 0x2545_F491_4F6C_DD1D;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };

        let mut f = Fixture::new();
        let mut live: Vec<OrderId> = Vec::new();

        for _ in 0..4000 {
            match next() % 10 {
                0..=6 => {
                    let side = if next() % 2 == 0 { Side::Buy } else { Side::Sell };
                    let cents = 1 + (next() % 98) as i64;
                    let qty = 1 + (next() % 50) as i64;
                    let strategy = (next() % 3) as u16 + 1;
                    let (id, _) = f.submit(strategy, side, cents, qty);
                    if f.book.order(id).is_some() {
                        live.push(id);
                    }
                }
                7 | 8 => {
                    if !live.is_empty() {
                        let i = (next() as usize) % live.len();
                        let id = live.swap_remove(i);
                        f.book.cancel(id);
                        f.book.debug_check();
                    }
                }
                _ => {
                    if !live.is_empty() {
                        let i = (next() as usize) % live.len();
                        let id = live[i];
                        if let Some(o) = f.book.order(id) {
                            let cur = o.remaining.get();
                            if cur > 1 {
                                let _ = f.book.reduce(id, Qty(cur / 2));
                                f.book.debug_check();
                            }
                        }
                    }
                }
            }
            live.retain(|id| f.book.order(*id).is_some());
        }
        f.book.debug_check();
    }
}
