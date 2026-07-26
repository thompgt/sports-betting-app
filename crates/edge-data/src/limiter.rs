//! Client-side rate limiting.
//!
//! Venues publish request quotas and enforce them with `429`s. Discovering the
//! limit by being throttled is expensive twice over: the rejected request is
//! wasted, and the venue's own backoff is far more punitive than one imposed
//! voluntarily. So the client meters itself.
//!
//! The limiter is a **pure state machine** — it takes the current time as an
//! argument rather than reading a clock. That keeps it deterministic under test
//! and makes it replayable in a backtest, which is the same reason
//! [`edge_core`] has no clock either.
//!
//! Real venues publish more than one quota at a time ("10/second and
//! 600/minute"), and those are not equivalent: the per-second bucket allows a
//! burst the per-minute bucket must still amortise. A [`RateLimiter`] therefore
//! holds a *set* of buckets and is only ready when every one of them is.

use std::time::Duration;

use edge_core::types::Ts;

/// One token bucket: `rate` tokens accrue per second, capped at `burst`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bucket {
    rate: f64,
    burst: f64,
    tokens: f64,
    last: Ts,
}

impl Bucket {
    /// A bucket allowing `rate` requests per second with room for `burst` of
    /// them at once. Both are clamped to sane positives; a zero rate would
    /// deadlock the caller forever rather than fail visibly.
    pub fn new(rate: f64, burst: f64, now: Ts) -> Self {
        let rate = if rate.is_finite() && rate > 0.0 { rate } else { f64::MIN_POSITIVE };
        let burst = if burst.is_finite() && burst >= 1.0 { burst } else { 1.0 };
        Bucket { rate, burst, tokens: burst, last: now }
    }

    /// `n` requests per second, allowing a one-second burst.
    pub fn per_second(n: f64, now: Ts) -> Self {
        Bucket::new(n, n.max(1.0), now)
    }

    /// `n` requests per minute, allowing a ten-second burst. Venues that quote a
    /// per-minute figure invariably tolerate short bursts inside it; refusing to
    /// burst at all would leave most of the quota unused.
    pub fn per_minute(n: f64, now: Ts) -> Self {
        Bucket::new(n / 60.0, (n / 6.0).max(1.0), now)
    }

    fn refill(&mut self, now: Ts) {
        let dt = (now.0 - self.last.0).max(0) as f64 / 1e9;
        if dt > 0.0 {
            self.tokens = (self.tokens + dt * self.rate).min(self.burst);
            self.last = now;
        }
    }

    /// How long until `cost` tokens are available. Zero means now.
    fn wait_for(&self, cost: f64) -> Duration {
        let deficit = cost - self.tokens;
        if deficit <= 0.0 {
            return Duration::ZERO;
        }
        Duration::from_secs_f64((deficit / self.rate).min(3_600.0))
    }
}

/// Metering across every quota a venue enforces at once.
#[derive(Debug, Clone, PartialEq)]
pub struct RateLimiter {
    buckets: Vec<Bucket>,
}

impl RateLimiter {
    /// A limiter enforcing all of `buckets` simultaneously. With no buckets it
    /// permits everything, which is the right behaviour for a simulator.
    pub fn new(buckets: Vec<Bucket>) -> Self {
        RateLimiter { buckets }
    }

    /// The common case: a single per-second quota.
    pub fn per_second(n: f64, now: Ts) -> Self {
        RateLimiter::new(vec![Bucket::per_second(n, now)])
    }

    /// Unmetered. For in-process sources that cannot be throttled.
    pub fn unlimited() -> Self {
        RateLimiter::new(Vec::new())
    }

    /// Take `cost` tokens if every bucket can pay, otherwise report how long the
    /// caller must wait — the longest wait across the buckets, since being ready
    /// on one quota is no use while another is exhausted.
    ///
    /// Nothing is deducted on refusal. A caller that sleeps and retries is not
    /// charged twice for the same request.
    pub fn acquire(&mut self, now: Ts, cost: f64) -> Result<(), Duration> {
        let cost = cost.max(0.0);
        for b in &mut self.buckets {
            b.refill(now);
        }
        let wait = self
            .buckets
            .iter()
            .map(|b| b.wait_for(cost))
            .max()
            .unwrap_or(Duration::ZERO);
        if wait > Duration::ZERO {
            return Err(wait);
        }
        for b in &mut self.buckets {
            b.tokens -= cost;
        }
        Ok(())
    }

    /// One request's worth.
    pub fn acquire_one(&mut self, now: Ts) -> Result<(), Duration> {
        self.acquire(now, 1.0)
    }

    /// Spend tokens whether or not they are available, driving the bucket
    /// negative.
    ///
    /// This is the response to a `429`: the venue has told us its count differs
    /// from ours, and the honest reaction is to fall behind our own schedule
    /// rather than to argue. Without it a client that mis-models the quota
    /// throttles at exactly the wrong moment forever.
    pub fn penalise(&mut self, now: Ts, cost: f64) {
        for b in &mut self.buckets {
            b.refill(now);
            b.tokens -= cost.max(0.0);
        }
    }

    /// Tokens available in the tightest bucket, for metrics.
    pub fn available(&mut self, now: Ts) -> f64 {
        for b in &mut self.buckets {
            b.refill(now);
        }
        self.buckets.iter().map(|b| b.tokens).fold(f64::INFINITY, f64::min)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: i64 = 1_000_000_000;

    #[test]
    fn a_fresh_bucket_is_full_and_spends_down() {
        let mut l = RateLimiter::per_second(5.0, Ts::ZERO);
        for _ in 0..5 {
            assert!(l.acquire_one(Ts::ZERO).is_ok());
        }
        assert!(l.acquire_one(Ts::ZERO).is_err(), "the burst is finite");
    }

    #[test]
    fn refusal_reports_the_wait_and_that_wait_is_enough() {
        let mut l = RateLimiter::per_second(2.0, Ts::ZERO);
        l.acquire(Ts::ZERO, 2.0).unwrap();
        let wait = l.acquire_one(Ts::ZERO).unwrap_err();
        assert!((wait.as_secs_f64() - 0.5).abs() < 1e-9);

        let later = Ts(Ts::ZERO.0 + (wait.as_secs_f64() * 1e9) as i64);
        assert!(l.acquire_one(later).is_ok(), "waiting the advertised time must suffice");
    }

    #[test]
    fn a_refused_request_is_not_charged() {
        let mut l = RateLimiter::per_second(1.0, Ts::ZERO);
        l.acquire_one(Ts::ZERO).unwrap();
        for _ in 0..10 {
            assert!(l.acquire_one(Ts::ZERO).is_err());
        }
        // Ten refusals must not have deepened the hole: one second still buys one.
        assert!(l.acquire_one(Ts(S)).is_ok());
    }

    #[test]
    fn tokens_accrue_but_never_past_the_burst() {
        let mut l = RateLimiter::per_second(10.0, Ts::ZERO);
        l.acquire(Ts::ZERO, 10.0).unwrap();
        assert!((l.available(Ts(100 * S)) - 10.0).abs() < 1e-9, "idling does not bank credit");
    }

    #[test]
    fn every_quota_binds_not_just_the_first() {
        // 100/second but only 60/minute: the per-second bucket would wave
        // through a burst the per-minute one cannot afford.
        let mut l = RateLimiter::new(vec![
            Bucket::per_second(100.0, Ts::ZERO),
            Bucket::per_minute(60.0, Ts::ZERO),
        ]);
        let mut ok = 0;
        for _ in 0..100 {
            if l.acquire_one(Ts::ZERO).is_ok() {
                ok += 1;
            }
        }
        assert_eq!(ok, 10, "the minute quota's 10-request burst is what binds");
    }

    #[test]
    fn a_penalty_puts_the_client_behind_its_own_schedule() {
        let mut l = RateLimiter::per_second(1.0, Ts::ZERO);
        l.penalise(Ts::ZERO, 3.0);
        assert!(l.acquire_one(Ts(S)).is_err(), "the debt is repaid before new requests");
        assert!(l.acquire_one(Ts(3 * S)).is_ok());
    }

    #[test]
    fn an_unlimited_limiter_never_refuses() {
        let mut l = RateLimiter::unlimited();
        for _ in 0..10_000 {
            assert!(l.acquire_one(Ts::ZERO).is_ok());
        }
    }

    #[test]
    fn a_degenerate_rate_cannot_deadlock_the_caller() {
        // A misconfigured zero rate must not hand back an infinite wait.
        let mut l = RateLimiter::new(vec![Bucket::new(0.0, 0.0, Ts::ZERO)]);
        l.acquire_one(Ts::ZERO).unwrap();
        let wait = l.acquire_one(Ts::ZERO).unwrap_err();
        assert!(wait.as_secs_f64().is_finite() && wait.as_secs_f64() <= 3_600.0);
    }

    #[test]
    fn time_running_backwards_does_not_mint_tokens() {
        let mut l = RateLimiter::per_second(1.0, Ts(10 * S));
        l.acquire_one(Ts(10 * S)).unwrap();
        assert!(l.acquire_one(Ts::ZERO).is_err(), "a clock step back is not free quota");
    }
}
