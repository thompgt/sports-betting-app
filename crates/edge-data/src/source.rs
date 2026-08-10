//! What a venue publishes, and how to ask it politely.
//!
//! Every adapter — Kalshi, Polymarket, the simulator — reduces to the same
//! handful of statements in [`VenueUpdate`]. Downstream code is written once
//! against that vocabulary and never against a venue's JSON.
//!
//! Updates are keyed by the venue's own ticker rather than by `MarketId`,
//! because a source has no business knowing the engine's interning table. The
//! translation happens once, in [`crate::assembler`].
//!
//! [`Guard`] is the other half: the composition of the rate limiter, the retry
//! scheduler, and the circuit breaker into a single "make this call, sensibly"
//! wrapper. Each of those is individually simple and individually useless — the
//! value is entirely in how they interact, and getting that interaction wrong
//! is how a client wedges itself against a venue that has already recovered.

use std::future::Future;
use std::sync::Mutex;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use edge_core::fees::FeeModel;
use edge_core::market::{MarketSpec, MarketStatus};
use edge_core::rng::Rng;
use edge_core::types::{Leg, MarketId, Price, Qty, Side, Ts, VenueId};
use serde::{Deserialize, Serialize};

use crate::backoff::{Decision, RetryPolicy};
use crate::breaker::{BreakerConfig, CircuitBreaker};
use crate::error::{DataError, Result};
use crate::limiter::RateLimiter;

/// Wall-clock now, as the engine counts time.
///
/// The only clock read in this crate. Everything below it takes `Ts` as an
/// argument, which is what makes a recorded session replayable.
pub fn now() -> Ts {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(d) => Ts(d.as_nanos().min(i64::MAX as u128) as i64),
        Err(_) => Ts::ZERO,
    }
}

/// One aggregated price level, as market data rather than as an order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Level {
    pub price: Price,
    pub qty: Qty,
}

impl Level {
    pub fn new(price: Price, qty: Qty) -> Self {
        Level { price, qty }
    }
}

/// A full depth snapshot.
///
/// `seq` is the venue's own sequence number where it publishes one. It is what
/// makes a delta stream verifiable; without it a dropped message is invisible
/// and the local book silently diverges from the real one.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BookSnapshot {
    /// Descending by price.
    pub bids: Vec<Level>,
    /// Ascending by price.
    pub asks: Vec<Level>,
    pub seq: u64,
    pub ts: Ts,
}

impl BookSnapshot {
    /// Sort both sides and drop empty levels, so an adapter can hand over
    /// whatever order the venue sent.
    pub fn normalise(mut self) -> Self {
        self.bids.retain(|l| l.qty.get() > 0);
        self.asks.retain(|l| l.qty.get() > 0);
        self.bids.sort_by_key(|l| std::cmp::Reverse(l.price));
        self.asks.sort_by_key(|l| l.price);
        self
    }

    pub fn best_bid(&self) -> Option<Price> {
        self.bids.first().map(|l| l.price)
    }

    pub fn best_ask(&self) -> Option<Price> {
        self.asks.first().map(|l| l.price)
    }

    /// Is the best bid at or above the best ask? A venue snapshot never should
    /// be, and one that is means the payload was assembled from inconsistent
    /// parts — worth refusing rather than trading against.
    pub fn is_crossed(&self) -> bool {
        matches!((self.best_bid(), self.best_ask()), (Some(b), Some(a)) if b >= a)
    }
}

/// A tradable contract as one venue describes it, before it is interned.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Listing {
    /// Venue-native identifier, used for routing and for humans.
    pub ticker: String,
    pub title: String,
    /// The canonical event key from [`crate::resolve`]. Markets sharing this
    /// are the same bet, and this is the only thing tying venues together.
    pub event_key: String,
    /// Which outcome of that event this contract pays on, canonical across
    /// venues. `None` means the adapter cannot say, in which case the ticker is
    /// used - two venues then never pair, which is the safe direction to fail.
    pub outcome_key: Option<String>,
    pub leg: Leg,
    pub tick_size: i64,
    pub min_qty: Qty,
    pub max_qty: Option<Qty>,
    pub fee: FeeModel,
    pub closes_at: Option<Ts>,
    pub status: MarketStatus,
}

impl Listing {
    /// A minimal listing, for the simulator and for tests.
    pub fn new(ticker: impl Into<String>, event_key: impl Into<String>) -> Self {
        let ticker = ticker.into();
        Listing {
            title: ticker.clone(),
            ticker,
            event_key: event_key.into(),
            outcome_key: None,
            leg: Leg::Yes,
            tick_size: 10_000,
            min_qty: Qty(1),
            max_qty: None,
            fee: FeeModel::None,
            closes_at: None,
            status: MarketStatus::Open,
        }
    }

    /// The engine-side spec, once the event key has been interned.
    pub fn to_spec(&self, id: MarketId, event: edge_core::types::EventId, venue: VenueId) -> MarketSpec {
        MarketSpec {
            id,
            event_id: event,
            venue,
            ticker: self.ticker.clone(),
            title: self.title.clone(),
            outcome: self
                .outcome_key
                .clone()
                .unwrap_or_else(|| self.ticker.clone()),
            leg: self.leg,
            tick_size: self.tick_size,
            min_qty: self.min_qty,
            max_qty: self.max_qty,
            fee: self.fee,
            closes_at: self.closes_at,
            status: self.status,
        }
    }
}

/// Everything a venue can tell us, in one vocabulary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum VenueUpdate {
    /// A market appeared, or its metadata changed.
    Listing(Listing),
    /// Full depth. Replaces whatever was held for this market.
    Book { ticker: String, book: BookSnapshot },
    /// One level changed. `qty` is the new resting total at that price, not a
    /// delta — every venue worth integrating publishes absolutes, because a
    /// delta stream with no snapshot to anchor it cannot be verified.
    Level { ticker: String, side: Side, price: Price, qty: Qty, seq: u64, ts: Ts },
    Trade { ticker: String, price: Price, qty: Qty, taker: Side, ts: Ts },
    Status { ticker: String, status: MarketStatus, ts: Ts },
    /// Resolved. `outcome` is whether the YES side paid.
    Settled { ticker: String, outcome: bool, ts: Ts },
    /// Proof of life on an otherwise silent stream. A market with no trades is
    /// indistinguishable from a dead socket without one.
    Heartbeat { ts: Ts },
}

impl VenueUpdate {
    pub fn ticker(&self) -> Option<&str> {
        match self {
            VenueUpdate::Listing(l) => Some(&l.ticker),
            VenueUpdate::Book { ticker, .. }
            | VenueUpdate::Level { ticker, .. }
            | VenueUpdate::Trade { ticker, .. }
            | VenueUpdate::Status { ticker, .. }
            | VenueUpdate::Settled { ticker, .. } => Some(ticker),
            VenueUpdate::Heartbeat { .. } => None,
        }
    }

    pub fn ts(&self) -> Option<Ts> {
        match self {
            VenueUpdate::Book { book, .. } => Some(book.ts),
            VenueUpdate::Level { ts, .. }
            | VenueUpdate::Trade { ts, .. }
            | VenueUpdate::Status { ts, .. }
            | VenueUpdate::Settled { ts, .. }
            | VenueUpdate::Heartbeat { ts } => Some(*ts),
            VenueUpdate::Listing(_) => None,
        }
    }
}

/// A venue we can pull market data from.
///
/// Deliberately pull-shaped for catalogue and snapshot, push-shaped for the
/// stream. Polling suits things that change on the scale of minutes and
/// tolerates a dropped response; the book does not.
#[async_trait]
pub trait MarketSource: Send + Sync {
    fn venue(&self) -> VenueId;

    /// Short stable name, for logs and metrics.
    fn name(&self) -> &str;

    /// The tradable universe. Called on a slow cadence.
    async fn listings(&self) -> Result<Vec<Listing>>;

    /// Point-in-time depth for the named tickers.
    async fn snapshot(&self, tickers: &[String]) -> Result<Vec<VenueUpdate>>;

    /// Stream updates into `sink` until the connection drops or `sink` is
    /// closed, then return.
    ///
    /// Returning `Ok(())` means the venue closed the stream cleanly and the
    /// caller should reconnect; the default implementation reports that this
    /// source has no stream, which is correct for a REST-only venue and lets
    /// the runtime fall back to polling.
    async fn stream(
        &self,
        _tickers: &[String],
        _sink: tokio::sync::mpsc::Sender<VenueUpdate>,
    ) -> Result<()> {
        Err(DataError::Config(format!("{} has no streaming endpoint", self.name())))
    }
}

/// Rate limiting, retrying, and circuit breaking, composed.
///
/// The ordering inside [`Guard::call`] is the part that matters:
///
/// 1. **Limiter first, breaker second.** Asking the breaker for permission
///    consumes a half-open probe slot, and a probe that is then blocked by the
///    limiter never reports back — so the breaker waits forever for a verdict
///    that is not coming and the circuit never closes. Checking the cheap,
///    purely local gate first avoids taking a permit we cannot use.
/// 2. **A throttle is not a venue failure.** Declining to send says nothing
///    about the venue's health, so it must not count against the breaker.
/// 3. **A `429` is.** It also means our model of the quota is wrong, so the
///    limiter is pushed into debt rather than merely consulted.
#[derive(Debug)]
pub struct Guard {
    venue: String,
    policy: RetryPolicy,
    state: Mutex<GuardState>,
}

#[derive(Debug)]
struct GuardState {
    limiter: RateLimiter,
    breaker: CircuitBreaker,
    rng: Rng,
}

impl Guard {
    pub fn new(venue: impl Into<String>, limiter: RateLimiter, policy: RetryPolicy) -> Self {
        Guard::with_breaker(venue, limiter, policy, BreakerConfig::default())
    }

    pub fn with_breaker(
        venue: impl Into<String>,
        limiter: RateLimiter,
        policy: RetryPolicy,
        breaker: BreakerConfig,
    ) -> Self {
        let venue = venue.into();
        // Seeded from the venue name so two processes against the same venue
        // jitter differently, and one process replays identically.
        let seed = venue.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
            (h ^ b as u64).wrapping_mul(0x1000_0000_01b3)
        });
        Guard {
            venue,
            policy,
            state: Mutex::new(GuardState {
                limiter,
                breaker: CircuitBreaker::new(breaker),
                rng: Rng::new(seed),
            }),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, GuardState> {
        // A panic inside a critical section here would leave the limiter's
        // token count slightly wrong, which is not worth killing the process
        // over.
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn breaker_state(&self) -> crate::breaker::State {
        self.lock().breaker.state()
    }

    /// Whether a call would be attempted right now, without attempting one.
    /// For health endpoints; do not gate real calls on this, since the answer
    /// is stale the moment it is returned.
    pub fn is_healthy(&self) -> bool {
        self.lock().breaker.state() != crate::breaker::State::Open
    }

    /// Run `op`, retrying transient failures on the configured schedule.
    ///
    /// `op` is called afresh for each attempt, so it must be idempotent — which
    /// is why [`RetryPolicy::none`] exists for the calls that are not.
    pub async fn call<F, Fut, T>(&self, mut op: F) -> Result<T>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = Result<T>>,
    {
        let started = Instant::now();
        let mut retry = self.policy.start();

        loop {
            let outcome = match self.admit() {
                Err(gate) => Err(gate),
                Ok(()) => {
                    let r = op().await;
                    self.report(&r);
                    r
                }
            };

            let err = match outcome {
                Ok(v) => return Ok(v),
                Err(e) => e,
            };

            let delay = {
                let mut s = self.lock();
                match retry.on_error(&err, started.elapsed(), &mut s.rng) {
                    Decision::Retry(d) => d,
                    Decision::GiveUp(_) => return Err(err),
                }
            };
            tokio::time::sleep(delay).await;
        }
    }

    /// The two gates, in the order that keeps them from deadlocking each other.
    fn admit(&self) -> Result<()> {
        let t = now();
        let mut s = self.lock();
        if let Err(wait) = s.limiter.acquire_one(t) {
            return Err(DataError::Throttled(wait));
        }
        if let Err(retry_in) = s.breaker.allow(t) {
            // The permit is already spent. Handing it back would let a caller
            // hammering an open circuit mint free quota for the moment it
            // reopens, which is exactly when the venue can least afford it.
            return Err(DataError::CircuitOpen { venue: self.venue.clone(), retry_in });
        }
        Ok(())
    }

    fn report<T>(&self, r: &Result<T>) {
        let t = now();
        let mut s = self.lock();
        match r {
            Ok(_) => s.breaker.on_success(t),
            Err(e) => {
                if let DataError::RateLimited { .. } = e {
                    // The venue's count differs from ours and the venue is
                    // right. Fall behind our own schedule rather than argue.
                    s.limiter.penalise(t, 5.0);
                }
                if e.counts_against_health() {
                    s.breaker.on_failure(t);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::time::Duration;

    fn guard(policy: RetryPolicy) -> Guard {
        Guard::new("test", RateLimiter::unlimited(), policy)
    }

    fn fast() -> RetryPolicy {
        RetryPolicy {
            max_attempts: 5,
            budget: None,
            backoff: crate::backoff::Backoff {
                initial: Duration::from_millis(1),
                max: Duration::from_millis(2),
                multiplier: 1.0,
                jitter: crate::backoff::Jitter::None,
            },
        }
    }

    fn transient() -> DataError {
        DataError::Http { venue: "test".into(), status: 503, detail: String::new() }
    }

    #[tokio::test]
    async fn a_call_that_works_is_not_retried() {
        let calls = AtomicU32::new(0);
        let g = guard(fast());
        let v: i32 = g
            .call(|| {
                calls.fetch_add(1, Ordering::SeqCst);
                async { Ok(7) }
            })
            .await
            .unwrap();
        assert_eq!(v, 7);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn a_transient_failure_is_retried_until_it_works() {
        let calls = AtomicU32::new(0);
        let g = guard(fast());
        let v: i32 = g
            .call(|| {
                let n = calls.fetch_add(1, Ordering::SeqCst);
                async move { if n < 2 { Err(transient()) } else { Ok(1) } }
            })
            .await
            .unwrap();
        assert_eq!(v, 1);
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn a_permanent_failure_is_attempted_exactly_once() {
        let calls = AtomicU32::new(0);
        let g = guard(fast());
        let r: Result<i32> = g
            .call(|| {
                calls.fetch_add(1, Ordering::SeqCst);
                async { Err(DataError::Auth { venue: "test".into(), detail: "bad key".into() }) }
            })
            .await;
        assert!(r.is_err());
        assert_eq!(calls.load(Ordering::SeqCst), 1, "retrying a bad key burns quota forever");
    }

    #[tokio::test]
    async fn a_dead_venue_stops_being_called_at_all() {
        let calls = AtomicU32::new(0);
        let g = Guard::with_breaker(
            "test",
            RateLimiter::unlimited(),
            RetryPolicy { max_attempts: 1, budget: None, ..fast() },
            BreakerConfig { consecutive_failures: 3, ..BreakerConfig::default() },
        );
        for _ in 0..3 {
            let _: Result<i32> = g
                .call(|| {
                    calls.fetch_add(1, Ordering::SeqCst);
                    async { Err(transient()) }
                })
                .await;
        }
        assert_eq!(calls.load(Ordering::SeqCst), 3);

        let r: Result<i32> = g
            .call(|| {
                calls.fetch_add(1, Ordering::SeqCst);
                async { Err(transient()) }
            })
            .await;
        assert!(matches!(r, Err(DataError::CircuitOpen { .. })));
        assert_eq!(calls.load(Ordering::SeqCst), 3, "an open circuit costs nothing");
        assert!(!g.is_healthy());
    }

    #[tokio::test]
    async fn the_limiter_is_consulted_before_the_breaker_spends_a_probe() {
        // If the breaker granted a half-open probe that the limiter then
        // blocked, nothing would ever report the probe's verdict and the
        // circuit would stay open forever. So a throttled call must not reach
        // the breaker at all.
        let g = Guard::with_breaker(
            "test",
            RateLimiter::per_second(1.0, now()),
            RetryPolicy { max_attempts: 1, budget: None, ..fast() },
            BreakerConfig::default(),
        );
        let _: Result<i32> = g.call(|| async { Ok(0) }).await;
        let r: Result<i32> = g.call(|| async { Ok(0) }).await;
        assert!(matches!(r, Err(DataError::Throttled(_))));
        assert!(g.is_healthy(), "declining to send says nothing about the venue");
    }

    #[tokio::test]
    async fn a_throttle_from_the_venue_puts_the_limiter_into_debt() {
        let g = Guard::new(
            "test",
            RateLimiter::per_second(100.0, now()),
            RetryPolicy { max_attempts: 1, budget: None, ..fast() },
        );
        let _: Result<i32> = g
            .call(|| async {
                Err(DataError::RateLimited { venue: "test".into(), retry_after: None })
            })
            .await;
        let available = g.lock().limiter.available(now());
        assert!(available < 99.0, "a 429 means our model of the quota is wrong: {available}");
    }

    #[test]
    fn a_snapshot_sorts_itself_and_drops_empty_levels() {
        let s = BookSnapshot {
            bids: vec![
                Level::new(Price::from_cents(40), Qty(5)),
                Level::new(Price::from_cents(45), Qty(3)),
                Level::new(Price::from_cents(44), Qty(0)),
            ],
            asks: vec![
                Level::new(Price::from_cents(52), Qty(2)),
                Level::new(Price::from_cents(48), Qty(9)),
            ],
            seq: 1,
            ts: Ts::ZERO,
        }
        .normalise();
        assert_eq!(s.best_bid(), Some(Price::from_cents(45)));
        assert_eq!(s.best_ask(), Some(Price::from_cents(48)));
        assert_eq!(s.bids.len(), 2, "the zero level is not a level");
        assert!(!s.is_crossed());
    }

    #[test]
    fn a_crossed_snapshot_is_recognisable_as_nonsense() {
        let s = BookSnapshot {
            bids: vec![Level::new(Price::from_cents(60), Qty(1))],
            asks: vec![Level::new(Price::from_cents(50), Qty(1))],
            seq: 0,
            ts: Ts::ZERO,
        };
        assert!(s.is_crossed(), "a venue book never crosses; this payload is inconsistent");
    }

    #[test]
    fn an_update_round_trips_through_json() {
        let u = VenueUpdate::Level {
            ticker: "KXNBA-BOS".into(),
            side: Side::Buy,
            price: Price::from_cents(47),
            qty: Qty(120),
            seq: 9,
            ts: Ts::from_secs(5),
        };
        let json = serde_json::to_string(&u).unwrap();
        assert_eq!(serde_json::from_str::<VenueUpdate>(&json).unwrap(), u);
    }

    #[test]
    fn a_listing_becomes_a_spec_without_losing_anything() {
        let mut l = Listing::new("KXNBA-BOS", "nba:2026-05-27:bos@nyk");
        l.fee = FeeModel::KALSHI_STANDARD;
        l.closes_at = Some(Ts::from_secs(1_000));
        let spec = l.to_spec(MarketId(3), edge_core::types::EventId(1), VenueId(2));
        assert_eq!(spec.ticker, "KXNBA-BOS");
        assert_eq!(spec.closes_at, Some(Ts::from_secs(1_000)));
        assert_eq!(spec.venue, VenueId(2));
    }
}
