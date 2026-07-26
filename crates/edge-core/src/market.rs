//! Market definitions and the registry that interns them.
//!
//! The registry is the piece that makes cross-venue trading possible at all.
//! Kalshi calls a market `KXNBA-25DEC25-BOS`, Polymarket calls the same thing a
//! 77-character condition id, and a sportsbook calls it "Boston Celtics". They
//! are one *event* with several tradable *markets* on it, and the arbitrage and
//! consensus layers both need that relationship expressed explicitly rather than
//! rediscovered by string matching on every tick.
//!
//! Resolving messy venue names onto a shared `EventId` is a separate concern and
//! lives in the ingestion layer; by the time anything reaches the engine the
//! mapping is already an integer.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::fees::FeeModel;
use crate::types::{EventId, Leg, MarketId, Price, Qty, Ts, VenueId};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MarketStatus {
    /// Listed but not yet accepting orders.
    #[default]
    PreOpen,
    Open,
    /// Temporarily not trading — a venue outage or a pending news event. Quotes
    /// are stale and must not be treated as prices.
    Halted,
    /// Trading has ended, resolution pending.
    Closed,
    /// Resolved. The engine settles positions and stops quoting.
    Settled,
}

impl MarketStatus {
    #[inline]
    pub const fn is_tradable(self) -> bool {
        matches!(self, MarketStatus::Open)
    }
}

/// Everything the engine needs to know about one tradable contract.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MarketSpec {
    pub id: MarketId,
    /// The real-world event. Markets sharing this are the same bet.
    pub event_id: EventId,
    pub venue: VenueId,
    /// Venue-native identifier, for order routing and for humans.
    pub ticker: String,
    pub title: String,
    /// Which leg of the binary this contract pays on.
    pub leg: Leg,
    /// Minimum price increment, in micro-dollars. An order not on this grid is
    /// rejected or silently re-priced by the venue, so the engine rounds first.
    pub tick_size: i64,
    pub min_qty: Qty,
    /// Venue order-size ceiling, where one exists.
    pub max_qty: Option<Qty>,
    pub fee: FeeModel,
    /// When trading stops. Drives time-to-resolution features and the decision
    /// to stop quoting near the close.
    pub closes_at: Option<Ts>,
    pub status: MarketStatus,
}

impl MarketSpec {
    /// A minimal spec, for tests and for venues that expose little metadata.
    pub fn new(id: MarketId, event_id: EventId, venue: VenueId, ticker: impl Into<String>) -> Self {
        let ticker = ticker.into();
        MarketSpec {
            id,
            event_id,
            venue,
            title: ticker.clone(),
            ticker,
            leg: Leg::Yes,
            tick_size: 10_000, // one cent
            min_qty: Qty(1),
            max_qty: None,
            fee: FeeModel::None,
            closes_at: None,
            status: MarketStatus::Open,
        }
    }

    /// Coerce a price onto the venue's tick grid.
    #[inline]
    pub fn round_price(&self, p: Price) -> Price {
        p.round_to_tick(self.tick_size)
    }

    /// Would the venue accept an order of this size?
    pub fn qty_is_valid(&self, q: Qty) -> bool {
        let a = q.abs();
        a >= self.min_qty && self.max_qty.map(|m| a <= m).unwrap_or(true)
    }

    /// Seconds until trading closes, or `None` when the market has no scheduled
    /// close. Negative values are clamped to zero.
    pub fn seconds_to_close(&self, now: Ts) -> Option<f64> {
        self.closes_at.map(|c| now.elapsed_to(c) as f64 / 1e9)
    }
}

/// Interning table mapping venue-native identifiers onto engine integers.
///
/// Not thread-safe by design: it is built during ingestion and then shared
/// read-only with the trading threads, which is both simpler and faster than
/// locking a map on every tick.
#[derive(Debug, Default)]
pub struct MarketRegistry {
    specs: Vec<MarketSpec>,
    by_venue_ticker: HashMap<(VenueId, String), MarketId>,
    by_event: HashMap<EventId, Vec<MarketId>>,
    event_keys: HashMap<String, EventId>,
}

impl MarketRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Intern an event key (however the ingestion layer chose to canonicalise
    /// it) into a stable `EventId`.
    pub fn intern_event(&mut self, key: impl Into<String>) -> EventId {
        let key = key.into();
        let next = EventId(self.event_keys.len() as u64);
        *self.event_keys.entry(key).or_insert(next)
    }

    /// Register a market, or return the existing id if this venue/ticker pair is
    /// already known. Re-registering updates the mutable fields (status, fees,
    /// close time) without invalidating the id, so a venue republishing its
    /// catalogue does not churn the engine's identifiers.
    pub fn register(&mut self, mut spec: MarketSpec) -> MarketId {
        let key = (spec.venue, spec.ticker.clone());
        if let Some(&existing) = self.by_venue_ticker.get(&key) {
            let idx = existing.get() as usize;
            spec.id = existing;
            spec.event_id = self.specs[idx].event_id;
            self.specs[idx] = spec;
            return existing;
        }
        let id = MarketId(self.specs.len() as u64);
        spec.id = id;
        let event_id = spec.event_id;
        self.by_venue_ticker.insert(key, id);
        self.by_event.entry(event_id).or_default().push(id);
        self.specs.push(spec);
        id
    }

    #[inline]
    pub fn get(&self, id: MarketId) -> Option<&MarketSpec> {
        self.specs.get(id.get() as usize)
    }

    pub fn get_mut(&mut self, id: MarketId) -> Option<&mut MarketSpec> {
        self.specs.get_mut(id.get() as usize)
    }

    pub fn by_ticker(&self, venue: VenueId, ticker: &str) -> Option<MarketId> {
        self.by_venue_ticker.get(&(venue, ticker.to_string())).copied()
    }

    /// Every market listed on the same event, across all venues. The input to
    /// cross-venue arbitrage.
    pub fn markets_for_event(&self, event: EventId) -> &[MarketId] {
        self.by_event.get(&event).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Events listed on more than one venue — the only ones where cross-venue
    /// arbitrage is even possible.
    pub fn multi_venue_events(&self) -> Vec<EventId> {
        self.by_event
            .iter()
            .filter(|(_, ms)| {
                let mut venues: Vec<VenueId> =
                    ms.iter().filter_map(|m| self.get(*m)).map(|s| s.venue).collect();
                venues.sort_unstable();
                venues.dedup();
                venues.len() > 1
            })
            .map(|(e, _)| *e)
            .collect()
    }

    pub fn iter(&self) -> impl Iterator<Item = &MarketSpec> {
        self.specs.iter()
    }

    pub fn len(&self) -> usize {
        self.specs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.specs.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registration_assigns_stable_ids() {
        let mut r = MarketRegistry::new();
        let e = r.intern_event("NBA:2025-12-25:BOS@LAL");
        let a = r.register(MarketSpec::new(MarketId(0), e, VenueId(1), "KXNBA-BOS"));
        let b = r.register(MarketSpec::new(MarketId(0), e, VenueId(2), "0xabc123"));
        assert_ne!(a, b);
        assert_eq!(r.get(a).unwrap().ticker, "KXNBA-BOS");
        assert_eq!(r.len(), 2);
    }

    #[test]
    fn re_registering_updates_without_churning_the_id() {
        let mut r = MarketRegistry::new();
        let e = r.intern_event("evt");
        let mut spec = MarketSpec::new(MarketId(0), e, VenueId(1), "TICK");
        let id = r.register(spec.clone());

        spec.status = MarketStatus::Halted;
        spec.fee = FeeModel::KALSHI_STANDARD;
        let again = r.register(spec);

        assert_eq!(id, again, "a catalogue refresh must not renumber markets");
        assert_eq!(r.len(), 1);
        assert_eq!(r.get(id).unwrap().status, MarketStatus::Halted);
    }

    #[test]
    fn interning_an_event_key_twice_is_idempotent() {
        let mut r = MarketRegistry::new();
        assert_eq!(r.intern_event("same"), r.intern_event("same"));
        assert_ne!(r.intern_event("same"), r.intern_event("other"));
    }

    #[test]
    fn the_same_event_across_venues_is_discoverable() {
        let mut r = MarketRegistry::new();
        let shared = r.intern_event("NBA:BOS@LAL");
        let solo = r.intern_event("NBA:NYK@MIA");
        r.register(MarketSpec::new(MarketId(0), shared, VenueId(1), "K1"));
        r.register(MarketSpec::new(MarketId(0), shared, VenueId(2), "P1"));
        r.register(MarketSpec::new(MarketId(0), solo, VenueId(1), "K2"));

        assert_eq!(r.markets_for_event(shared).len(), 2);
        assert_eq!(r.multi_venue_events(), vec![shared]);
    }

    #[test]
    fn two_markets_on_one_venue_are_not_cross_venue() {
        let mut r = MarketRegistry::new();
        let e = r.intern_event("evt");
        r.register(MarketSpec::new(MarketId(0), e, VenueId(1), "A"));
        r.register(MarketSpec::new(MarketId(0), e, VenueId(1), "B"));
        assert!(r.multi_venue_events().is_empty());
    }

    #[test]
    fn price_is_snapped_to_the_venue_grid() {
        let mut spec = MarketSpec::new(MarketId(0), EventId(0), VenueId(1), "T");
        spec.tick_size = 10_000;
        assert_eq!(spec.round_price(Price(374_000)), Price::from_cents(37));
        spec.tick_size = 1_000;
        assert_eq!(spec.round_price(Price(374_400)), Price(374_000));
    }

    #[test]
    fn size_limits_are_enforced_both_ways() {
        let mut spec = MarketSpec::new(MarketId(0), EventId(0), VenueId(1), "T");
        spec.min_qty = Qty(10);
        spec.max_qty = Some(Qty(1000));
        assert!(!spec.qty_is_valid(Qty(9)));
        assert!(spec.qty_is_valid(Qty(10)));
        assert!(spec.qty_is_valid(Qty(-500)), "a short position is sized by magnitude");
        assert!(!spec.qty_is_valid(Qty(1001)));
    }

    #[test]
    fn only_open_markets_are_tradable() {
        assert!(MarketStatus::Open.is_tradable());
        for s in [
            MarketStatus::PreOpen,
            MarketStatus::Halted,
            MarketStatus::Closed,
            MarketStatus::Settled,
        ] {
            assert!(!s.is_tradable(), "{s:?} must not be tradable");
        }
    }

    #[test]
    fn time_to_close_never_goes_negative() {
        let mut spec = MarketSpec::new(MarketId(0), EventId(0), VenueId(1), "T");
        spec.closes_at = Some(Ts::from_secs(1_000));
        assert_eq!(spec.seconds_to_close(Ts::from_secs(400)), Some(600.0));
        assert_eq!(spec.seconds_to_close(Ts::from_secs(2_000)), Some(0.0));
        spec.closes_at = None;
        assert_eq!(spec.seconds_to_close(Ts::ZERO), None);
    }
}
