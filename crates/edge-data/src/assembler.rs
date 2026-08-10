//! From a venue's updates to the engine's books.
//!
//! This is the boundary where venue vocabulary stops. Tickers become interned
//! [`MarketId`]s, event keys become [`EventId`]s, and price levels become an
//! [`OrderBook`] — the same type the matching engine and every strategy already
//! consume, so a live feed and a backtest present identically.
//!
//! ## Levels are not orders
//!
//! A venue publishes aggregate depth: "180 contracts bid at 47c". The engine's
//! book is order-level. The assembler bridges that by holding exactly one
//! synthetic order per occupied price level and resizing it, which keeps one
//! book type across the whole system rather than paying for a second one.
//!
//! Those synthetic orders are posted `PostOnly`, so the book will reject rather
//! than *match* if the assembler ever tries to insert a crossing level. That is
//! not defensive decoration: an out-of-order delta genuinely can put a new bid
//! above the standing best ask, and silently matching it would invent trades
//! that never happened and hand the strategies a book that no venue ever
//! published. Crossed levels are removed explicitly first, as stale.
//!
//! ## Gaps are not recoverable, and pretending otherwise is worse
//!
//! Venue sequence numbers exist so a dropped message is detectable. When one
//! goes missing the local book is wrong by an unknown amount and no amount of
//! subsequent deltas will fix it. The assembler marks the market **stale** and
//! keeps it stale until a full snapshot arrives. A stale book is not a degraded
//! book to be traded more carefully — it is a book whose best bid may be a
//! price nobody is offering.

use std::collections::HashMap;

use edge_book::OrderBook;
use edge_book::order::{BookEvent, Order, TimeInForce};
use edge_core::market::{MarketRegistry, MarketStatus};
use edge_core::types::{EventId, MarketId, OrderId, Price, Qty, Side, StrategyId, Ts, VenueId};

use crate::source::{Listing, VenueUpdate};

/// The synthetic owner of every level posted by the assembler. Nothing else in
/// the system uses this id, so a book's market-data levels are always
/// distinguishable from the engine's own resting orders.
pub const FEED: StrategyId = StrategyId(u16::MAX);

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AssemblerConfig {
    /// Depth kept per side. Venues publish far more than any strategy reads,
    /// and the deep tail is both the least reliable and the least tradable
    /// part of the book.
    pub max_levels: usize,
    /// How long a market may go without an update before it is considered
    /// stale. A quiet market and a dead socket look identical otherwise.
    pub stale_after_secs: f64,
}

impl Default for AssemblerConfig {
    fn default() -> Self {
        AssemblerConfig { max_levels: 25, stale_after_secs: 30.0 }
    }
}

/// What the assembler concluded, for the runtime to act on.
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    /// A market was interned for the first time.
    Registered {
        market: MarketId,
        event: EventId,
    },
    /// The book changed.
    Book {
        market: MarketId,
        ts: Ts,
    },
    Trade {
        market: MarketId,
        price: Price,
        qty: Qty,
        taker: Side,
        ts: Ts,
    },
    Status {
        market: MarketId,
        status: MarketStatus,
        ts: Ts,
    },
    Settled {
        market: MarketId,
        outcome: bool,
        ts: Ts,
    },
    /// A sequence number was skipped. The book is wrong and the caller must
    /// re-snapshot; until it does, the market is stale and untradable.
    Gap {
        market: MarketId,
        expected: u64,
        got: u64,
    },
    /// The book was believed and is now trusted again after a snapshot.
    Recovered {
        market: MarketId,
    },
    /// An update named a ticker no listing has been seen for. Not an error —
    /// venues routinely stream a market before publishing its metadata — but
    /// the update is dropped rather than guessed at.
    Unknown {
        ticker: String,
    },
}

#[derive(Debug)]
struct BookState {
    book: OrderBook,
    /// Highest sequence number applied.
    seq: u64,
    last_update: Ts,
    /// Set by a gap, cleared only by a full snapshot.
    stale: bool,
    /// The synthetic order holding each occupied level.
    levels: HashMap<(Side, i64), OrderId>,
}

/// Venue updates in, engine state out.
#[derive(Debug)]
pub struct Assembler {
    venue: VenueId,
    cfg: AssemblerConfig,
    registry: MarketRegistry,
    books: HashMap<MarketId, BookState>,
    next_order: u64,
    seq: u64,
}

impl Assembler {
    pub fn new(venue: VenueId, cfg: AssemblerConfig) -> Self {
        Assembler {
            venue,
            cfg,
            registry: MarketRegistry::new(),
            books: HashMap::new(),
            next_order: 1,
            seq: 0,
        }
    }

    pub fn registry(&self) -> &MarketRegistry {
        &self.registry
    }

    pub fn book(&self, market: MarketId) -> Option<&OrderBook> {
        self.books.get(&market).map(|s| &s.book)
    }

    pub fn market_of(&self, ticker: &str) -> Option<MarketId> {
        self.registry.by_ticker(self.venue, ticker)
    }

    /// Is this market's book untrustworthy — either explicitly gapped, or
    /// simply silent for longer than the configured timeout?
    ///
    /// The two are one question because they have one answer: do not trade it.
    pub fn is_stale(&self, market: MarketId, now: Ts) -> bool {
        match self.books.get(&market) {
            None => true,
            Some(s) => {
                s.stale || (now.0 - s.last_update.0) as f64 / 1e9 > self.cfg.stale_after_secs
            }
        }
    }

    /// Every market whose book can currently be believed.
    pub fn fresh_markets(&self, now: Ts) -> Vec<MarketId> {
        self.books.keys().copied().filter(|m| !self.is_stale(*m, now)).collect()
    }

    fn next_order_id(&mut self) -> OrderId {
        self.next_order += 1;
        OrderId(self.next_order)
    }

    /// Intern a listing, creating or refreshing its spec and book.
    pub fn register(&mut self, listing: &Listing) -> (MarketId, bool) {
        let event = self.registry.intern_event(listing.event_key.clone());
        let existing = self.registry.by_ticker(self.venue, &listing.ticker);
        let spec = listing.to_spec(existing.unwrap_or(MarketId(0)), event, self.venue);
        let id = self.registry.register(spec);
        let is_new = existing.is_none();
        if is_new {
            self.books.insert(
                id,
                BookState {
                    book: OrderBook::new(id, listing.tick_size),
                    seq: 0,
                    last_update: Ts::ZERO,
                    // Nothing has been published yet, so there is nothing to
                    // trust. A market is stale until its first snapshot.
                    stale: true,
                    levels: HashMap::new(),
                },
            );
        }
        (id, is_new)
    }

    /// Apply one update, appending whatever it implies to `out`.
    pub fn apply(&mut self, update: VenueUpdate, out: &mut Vec<Event>) {
        let ticker = match update.ticker() {
            Some(t) => t.to_string(),
            None => return, // a heartbeat says only that the socket is alive
        };

        if let VenueUpdate::Listing(l) = &update {
            let (market, is_new) = self.register(l);
            if is_new {
                let event = self.registry.get(market).map(|s| s.event_id).unwrap_or(EventId(0));
                out.push(Event::Registered { market, event });
            }
            return;
        }

        let Some(market) = self.market_of(&ticker) else {
            out.push(Event::Unknown { ticker });
            return;
        };

        match update {
            VenueUpdate::Listing(_) => unreachable!("handled above"),

            VenueUpdate::Book { book, .. } => {
                let book = book.normalise();
                // A crossed snapshot was assembled from inconsistent parts.
                // Applying it would produce a book no venue published.
                if book.is_crossed() {
                    if let Some(s) = self.books.get_mut(&market) {
                        s.stale = true;
                    }
                    out.push(Event::Gap { market, expected: 0, got: book.seq });
                    return;
                }
                let ts = book.ts;
                let was_stale = self.books.get(&market).map(|s| s.stale).unwrap_or(true);
                self.replace_book(market, &book);
                if was_stale {
                    out.push(Event::Recovered { market });
                }
                out.push(Event::Book { market, ts });
            }

            VenueUpdate::Level { side, price, qty, seq, ts, .. } => {
                let Some(state) = self.books.get_mut(&market) else { return };

                // Sequence numbers of zero mean the venue does not publish
                // them; there is then nothing to verify and nothing to gap on.
                if seq > 0 {
                    if seq <= state.seq {
                        return; // already applied, or arrived out of order
                    }
                    if state.seq > 0 && seq > state.seq + 1 {
                        let expected = state.seq + 1;
                        state.stale = true;
                        out.push(Event::Gap { market, expected, got: seq });
                        return;
                    }
                    state.seq = seq;
                }

                // A gapped book must not be patched. Every delta applied to it
                // makes it look more current while staying just as wrong.
                if state.stale {
                    return;
                }
                state.last_update = ts;
                self.set_level(market, side, price, qty);
                out.push(Event::Book { market, ts });
            }

            VenueUpdate::Trade { price, qty, taker, ts, .. } => {
                if let Some(s) = self.books.get_mut(&market) {
                    s.last_update = ts;
                }
                out.push(Event::Trade { market, price, qty, taker, ts });
            }

            VenueUpdate::Status { status, ts, .. } => {
                if let Some(spec) = self.registry.get_mut(market) {
                    spec.status = status;
                }
                // A halted market's quotes are not prices. Clearing the book
                // rather than freezing it stops a strategy from trading against
                // the last quote before the halt.
                if !status.is_tradable() {
                    self.clear_book(market);
                    if let Some(s) = self.books.get_mut(&market) {
                        s.stale = true;
                    }
                }
                out.push(Event::Status { market, status, ts });
            }

            VenueUpdate::Settled { outcome, ts, .. } => {
                if let Some(spec) = self.registry.get_mut(market) {
                    spec.status = MarketStatus::Settled;
                }
                self.clear_book(market);
                out.push(Event::Settled { market, outcome, ts });
            }

            VenueUpdate::Heartbeat { .. } => {}
        }
    }

    /// Replace a book wholesale from a snapshot. This is the only operation
    /// that clears staleness, because it is the only one that does not depend
    /// on the previous state being right.
    fn replace_book(&mut self, market: MarketId, snapshot: &crate::source::BookSnapshot) {
        self.clear_book(market);
        for l in snapshot.bids.iter().take(self.cfg.max_levels) {
            self.set_level(market, Side::Buy, l.price, l.qty);
        }
        for l in snapshot.asks.iter().take(self.cfg.max_levels) {
            self.set_level(market, Side::Sell, l.price, l.qty);
        }
        if let Some(s) = self.books.get_mut(&market) {
            s.seq = snapshot.seq;
            s.last_update = snapshot.ts;
            s.stale = false;
        }
    }

    fn clear_book(&mut self, market: MarketId) {
        let mut seq = self.seq;
        if let Some(s) = self.books.get_mut(&market) {
            let mut events = Vec::new();
            s.book.clear(&mut seq, Ts::ZERO, &mut events);
            s.levels.clear();
        }
        self.seq = seq;
    }

    /// Make the resting quantity at one price equal `qty`.
    fn set_level(&mut self, market: MarketId, side: Side, price: Price, qty: Qty) {
        // Anything on the other side at a price this level would trade through
        // is, by definition, no longer there. Remove it before inserting, or
        // the book would hold a crossed state that no venue ever published.
        if qty.get() > 0 {
            self.drop_crossed(market, side, price);
        }

        let existing =
            self.books.get(&market).and_then(|s| s.levels.get(&(side, price.0)).copied());

        if qty.get() <= 0 {
            if let (Some(id), Some(s)) = (existing, self.books.get_mut(&market)) {
                s.book.cancel(id);
                s.levels.remove(&(side, price.0));
            }
            return;
        }

        if let Some(id) = existing {
            let current = self.books.get(&market).map(|s| s.book.qty_at(side, price));
            if current == Some(qty) {
                return;
            }
            // `reduce` preserves queue position but cannot grow an order.
            // Queue position is meaningless for aggregated market data, so a
            // growing level is simply replaced.
            let shrank = self
                .books
                .get_mut(&market)
                .map(|s| s.book.reduce(id, qty).is_ok())
                .unwrap_or(false);
            if shrank {
                return;
            }
            if let Some(s) = self.books.get_mut(&market) {
                s.book.cancel(id);
                s.levels.remove(&(side, price.0));
            }
        }

        let id = self.next_order_id();
        let mut seq = self.seq;
        if let Some(s) = self.books.get_mut(&market) {
            let order =
                Order::limit(id, market, FEED, side, price, qty).with_tif(TimeInForce::PostOnly);
            let mut events: Vec<BookEvent> = Vec::new();
            s.book.submit(order, &mut seq, &mut events);
            // PostOnly makes a crossing insert a rejection rather than a trade.
            // `drop_crossed` should have made that impossible; if it did not,
            // the level is simply absent, which is the safe direction to fail.
            if s.book.qty_at(side, price).get() > 0 {
                s.levels.insert((side, price.0), id);
            }
        }
        self.seq = seq;
    }

    /// Remove opposite-side levels that `price` would trade through.
    fn drop_crossed(&mut self, market: MarketId, side: Side, price: Price) {
        let doomed: Vec<(Side, i64)> = match self.books.get(&market) {
            None => return,
            Some(s) => s
                .levels
                .keys()
                .filter(|(ls, lp)| *ls == side.opposite() && side.crosses(price, Price(*lp)))
                .copied()
                .collect(),
        };
        if let Some(s) = self.books.get_mut(&market) {
            for key in doomed {
                if let Some(id) = s.levels.remove(&key) {
                    s.book.cancel(id);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::{BookSnapshot, Level};

    const V: VenueId = VenueId(1);

    fn cents(c: i64) -> Price {
        Price::from_cents(c)
    }

    fn snapshot(bids: &[(i64, i64)], asks: &[(i64, i64)], seq: u64) -> BookSnapshot {
        BookSnapshot {
            bids: bids.iter().map(|(p, q)| Level::new(cents(*p), Qty(*q))).collect(),
            asks: asks.iter().map(|(p, q)| Level::new(cents(*p), Qty(*q))).collect(),
            seq,
            ts: Ts::from_secs(100),
        }
    }

    /// An assembler with one registered market, ticker `T`.
    fn fixture() -> (Assembler, MarketId) {
        let mut a = Assembler::new(V, AssemblerConfig::default());
        let (m, _) = a.register(&Listing::new("T", "evt"));
        (a, m)
    }

    fn apply(a: &mut Assembler, u: VenueUpdate) -> Vec<Event> {
        let mut out = Vec::new();
        a.apply(u, &mut out);
        out
    }

    fn book(
        a: &mut Assembler,
        m: MarketId,
        bids: &[(i64, i64)],
        asks: &[(i64, i64)],
        seq: u64,
    ) -> Vec<Event> {
        let _ = m;
        apply(a, VenueUpdate::Book { ticker: "T".into(), book: snapshot(bids, asks, seq) })
    }

    fn level(a: &mut Assembler, side: Side, price: i64, qty: i64, seq: u64) -> Vec<Event> {
        apply(
            a,
            VenueUpdate::Level {
                ticker: "T".into(),
                side,
                price: cents(price),
                qty: Qty(qty),
                seq,
                ts: Ts::from_secs(101),
            },
        )
    }

    #[test]
    fn a_listing_is_interned_once_and_refreshed_thereafter() {
        let mut a = Assembler::new(V, AssemblerConfig::default());
        let events = apply(&mut a, VenueUpdate::Listing(Listing::new("T", "evt")));
        assert!(matches!(events.as_slice(), [Event::Registered { .. }]));

        let mut again = Listing::new("T", "evt");
        again.title = "renamed".into();
        let events = apply(&mut a, VenueUpdate::Listing(again));
        assert!(events.is_empty(), "a catalogue refresh is not a new market");
        assert_eq!(a.registry().len(), 1);
        assert_eq!(a.registry().get(a.market_of("T").unwrap()).unwrap().title, "renamed");
    }

    #[test]
    fn markets_on_one_event_across_venues_share_an_event_id() {
        let mut a = Assembler::new(V, AssemblerConfig::default());
        let (one, _) = a.register(&Listing::new("YES", "nba:bos@nyk"));
        let (other, _) = a.register(&Listing::new("NO", "nba:bos@nyk"));
        assert_ne!(one, other);
        assert_eq!(
            a.registry().get(one).unwrap().event_id,
            a.registry().get(other).unwrap().event_id
        );
    }

    #[test]
    fn a_snapshot_builds_the_book() {
        let (mut a, m) = fixture();
        book(&mut a, m, &[(45, 100), (44, 50)], &[(47, 80), (48, 30)], 10);
        let b = a.book(m).unwrap();
        assert_eq!(b.best_bid(), Some(cents(45)));
        assert_eq!(b.best_ask(), Some(cents(47)));
        assert_eq!(b.qty_at(Side::Buy, cents(45)), Qty(100));
        assert_eq!(b.qty_at(Side::Sell, cents(48)), Qty(30));
    }

    #[test]
    fn a_market_is_untradable_until_its_first_snapshot() {
        let (a, m) = fixture();
        assert!(a.is_stale(m, Ts::from_secs(100)), "nothing has been published yet");
    }

    #[test]
    fn a_snapshot_replaces_rather_than_merges() {
        let (mut a, m) = fixture();
        book(&mut a, m, &[(45, 100)], &[(47, 80)], 10);
        book(&mut a, m, &[(40, 5)], &[(60, 5)], 11);
        let b = a.book(m).unwrap();
        assert_eq!(b.best_bid(), Some(cents(40)));
        assert_eq!(b.qty_at(Side::Buy, cents(45)), Qty::ZERO, "the old level is gone");
    }

    #[test]
    fn a_level_update_resizes_in_both_directions_and_removes_at_zero() {
        let (mut a, m) = fixture();
        book(&mut a, m, &[(45, 100)], &[(47, 80)], 10);

        level(&mut a, Side::Buy, 45, 60, 11);
        assert_eq!(a.book(m).unwrap().qty_at(Side::Buy, cents(45)), Qty(60));

        level(&mut a, Side::Buy, 45, 250, 12);
        assert_eq!(a.book(m).unwrap().qty_at(Side::Buy, cents(45)), Qty(250), "levels can grow");

        level(&mut a, Side::Buy, 45, 0, 13);
        assert_eq!(a.book(m).unwrap().qty_at(Side::Buy, cents(45)), Qty::ZERO);
        assert_eq!(a.book(m).unwrap().best_bid(), None);
    }

    #[test]
    fn a_new_level_inside_the_spread_becomes_the_touch() {
        let (mut a, m) = fixture();
        book(&mut a, m, &[(45, 100)], &[(47, 80)], 10);
        level(&mut a, Side::Buy, 46, 20, 11);
        assert_eq!(a.book(m).unwrap().best_bid(), Some(cents(46)));
    }

    #[test]
    fn a_bid_arriving_through_the_ask_removes_the_ask_rather_than_trading_it() {
        // The book must never end up crossed, and the assembler must never
        // invent a trade that no venue reported.
        let (mut a, m) = fixture();
        book(&mut a, m, &[(45, 100)], &[(47, 80), (48, 30)], 10);
        level(&mut a, Side::Buy, 48, 25, 11);

        let b = a.book(m).unwrap();
        assert_eq!(b.best_bid(), Some(cents(48)));
        assert_eq!(b.best_ask(), None, "both asks were traded through and are stale");
        assert!(b.best_bid().zip(b.best_ask()).is_none_or(|(bid, ask)| bid < ask));
    }

    #[test]
    fn a_skipped_sequence_number_makes_the_market_stale() {
        let (mut a, m) = fixture();
        book(&mut a, m, &[(45, 100)], &[(47, 80)], 10);
        assert!(!a.is_stale(m, Ts::from_secs(101)));

        let events = level(&mut a, Side::Buy, 45, 60, 15);
        assert_eq!(events, vec![Event::Gap { market: m, expected: 11, got: 15 }]);
        assert!(a.is_stale(m, Ts::from_secs(101)));
        assert_eq!(
            a.book(m).unwrap().qty_at(Side::Buy, cents(45)),
            Qty(100),
            "the gapped update was not applied"
        );
    }

    #[test]
    fn a_gapped_book_is_not_patched_by_the_deltas_that_follow() {
        // Applying them would make the book look more current while leaving it
        // just as wrong, which is worse than an obviously stale book.
        let (mut a, m) = fixture();
        book(&mut a, m, &[(45, 100)], &[(47, 80)], 10);
        level(&mut a, Side::Buy, 45, 60, 15);
        level(&mut a, Side::Buy, 45, 70, 16);
        assert_eq!(a.book(m).unwrap().qty_at(Side::Buy, cents(45)), Qty(100));
        assert!(a.is_stale(m, Ts::from_secs(101)));
    }

    #[test]
    fn only_a_snapshot_restores_a_gapped_market() {
        let (mut a, m) = fixture();
        book(&mut a, m, &[(45, 100)], &[(47, 80)], 10);
        level(&mut a, Side::Buy, 45, 60, 15);

        let events = book(&mut a, m, &[(46, 10)], &[(49, 10)], 20);
        assert!(events.contains(&Event::Recovered { market: m }));
        assert!(!a.is_stale(m, Ts::from_secs(101)));
        assert_eq!(a.book(m).unwrap().best_bid(), Some(cents(46)));
    }

    #[test]
    fn a_replayed_or_reordered_update_is_dropped_not_gapped() {
        let (mut a, m) = fixture();
        book(&mut a, m, &[(45, 100)], &[(47, 80)], 10);
        level(&mut a, Side::Buy, 45, 60, 11);
        let events = level(&mut a, Side::Buy, 45, 999, 11);
        assert!(events.is_empty());
        assert_eq!(a.book(m).unwrap().qty_at(Side::Buy, cents(45)), Qty(60));
        assert!(!a.is_stale(m, Ts::from_secs(101)));
    }

    #[test]
    fn a_venue_without_sequence_numbers_still_works() {
        let (mut a, m) = fixture();
        book(&mut a, m, &[(45, 100)], &[(47, 80)], 0);
        level(&mut a, Side::Buy, 45, 60, 0);
        level(&mut a, Side::Buy, 44, 10, 0);
        assert!(!a.is_stale(m, Ts::from_secs(101)));
        assert_eq!(a.book(m).unwrap().qty_at(Side::Buy, cents(45)), Qty(60));
    }

    #[test]
    fn a_crossed_snapshot_is_refused_rather_than_applied() {
        let (mut a, m) = fixture();
        let events = book(&mut a, m, &[(60, 10)], &[(50, 10)], 10);
        assert!(matches!(events.as_slice(), [Event::Gap { .. }]));
        assert!(a.is_stale(m, Ts::from_secs(101)));
    }

    #[test]
    fn a_halt_empties_the_book_rather_than_freezing_it() {
        // A halted market's last quote is not a price, and leaving it visible
        // invites a strategy to trade against it.
        let (mut a, m) = fixture();
        book(&mut a, m, &[(45, 100)], &[(47, 80)], 10);
        apply(
            &mut a,
            VenueUpdate::Status {
                ticker: "T".into(),
                status: MarketStatus::Halted,
                ts: Ts::from_secs(102),
            },
        );
        assert_eq!(a.book(m).unwrap().best_bid(), None);
        assert!(a.is_stale(m, Ts::from_secs(102)));
        assert_eq!(a.registry().get(m).unwrap().status, MarketStatus::Halted);
    }

    #[test]
    fn settlement_closes_the_market_out() {
        let (mut a, m) = fixture();
        book(&mut a, m, &[(45, 100)], &[(47, 80)], 10);
        let events = apply(
            &mut a,
            VenueUpdate::Settled { ticker: "T".into(), outcome: true, ts: Ts::from_secs(103) },
        );
        assert_eq!(
            events,
            vec![Event::Settled { market: m, outcome: true, ts: Ts::from_secs(103) }]
        );
        assert_eq!(a.registry().get(m).unwrap().status, MarketStatus::Settled);
        assert!(a.book(m).unwrap().is_empty());
    }

    #[test]
    fn silence_is_indistinguishable_from_a_dead_socket_and_treated_as_such() {
        let (mut a, m) = fixture();
        book(&mut a, m, &[(45, 100)], &[(47, 80)], 10);
        assert!(!a.is_stale(m, Ts::from_secs(120)));
        assert!(a.is_stale(m, Ts::from_secs(200)), "30s of nothing is not a healthy feed");
        assert!(a.fresh_markets(Ts::from_secs(200)).is_empty());
    }

    #[test]
    fn an_update_for_an_unlisted_ticker_is_reported_not_guessed_at() {
        let mut a = Assembler::new(V, AssemblerConfig::default());
        let events = apply(
            &mut a,
            VenueUpdate::Book { ticker: "NOPE".into(), book: snapshot(&[(45, 1)], &[], 1) },
        );
        assert_eq!(events, vec![Event::Unknown { ticker: "NOPE".into() }]);
    }

    #[test]
    fn depth_beyond_the_configured_limit_is_discarded() {
        let mut a = Assembler::new(V, AssemblerConfig { max_levels: 2, stale_after_secs: 30.0 });
        let (m, _) = a.register(&Listing::new("T", "evt"));
        book(&mut a, m, &[(45, 1), (44, 1), (43, 1), (42, 1)], &[(47, 1)], 1);
        assert_eq!(a.book(m).unwrap().depth(Side::Buy, 10).len(), 2);
        assert_eq!(a.book(m).unwrap().best_bid(), Some(cents(45)), "the near side is kept");
    }

    #[test]
    fn a_heartbeat_changes_nothing() {
        let (mut a, m) = fixture();
        book(&mut a, m, &[(45, 100)], &[(47, 80)], 10);
        let events = apply(&mut a, VenueUpdate::Heartbeat { ts: Ts::from_secs(101) });
        assert!(events.is_empty());
        assert_eq!(a.book(m).unwrap().best_bid(), Some(cents(45)));
    }

    #[test]
    fn the_book_stays_internally_consistent_through_a_long_random_walk() {
        use edge_core::rng::Rng;
        let (mut a, m) = fixture();
        book(&mut a, m, &[(45, 100)], &[(47, 80)], 1);
        let mut rng = Rng::new(7);
        for seq in 2..2_000u64 {
            let side = if rng.bernoulli(0.5) { Side::Buy } else { Side::Sell };
            let price = 20 + rng.below(60) as i64;
            let qty = rng.below(200) as i64;
            level(&mut a, side, price, qty, seq);
            let b = a.book(m).unwrap();
            b.debug_check();
            if let (Some(bid), Some(ask)) = (b.best_bid(), b.best_ask()) {
                assert!(bid < ask, "book crossed at seq {seq}: {bid} / {ask}");
            }
        }
    }
}
