//! Retry scheduling.
//!
//! Two failure modes bracket this problem. Retrying too eagerly turns a venue's
//! brief wobble into a self-inflicted denial of service, and because every
//! client in the world backs off on the same schedule, an unjittered
//! exponential curve re-synchronises the whole herd onto the same instant.
//! Retrying too patiently means a trading system sits blind through an outage
//! that ended thirty seconds ago.
//!
//! The scheduler here is deterministic given its RNG, holds no clock, and
//! decides only *whether and when* — the caller does the sleeping. That keeps
//! the policy testable without waiting in real time, and lets a backtest replay
//! a recorded outage with the identical retry pattern.

use std::time::Duration;

use edge_core::rng::Rng;

use crate::error::DataError;

/// How much randomness to mix into the exponential curve.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Jitter {
    /// Exactly the exponential delay. Predictable, and synchronises clients.
    None,
    /// Uniform on `[0, base]`. Spreads the herd hardest, at the cost of
    /// occasionally retrying almost immediately.
    #[default]
    Full,
    /// Uniform on `[base/2, base]`. Keeps a floor under the delay while still
    /// breaking up the herd — the right default when the floor matters.
    Equal,
    /// Uniform on `[initial, prev * 3]`, capped. Converges on a healthy venue
    /// faster than plain exponential because a lucky short delay is not
    /// immediately punished by doubling from it.
    Decorrelated,
}

/// The exponential curve and how it is randomised.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Backoff {
    pub initial: Duration,
    pub max: Duration,
    pub multiplier: f64,
    pub jitter: Jitter,
}

impl Default for Backoff {
    fn default() -> Self {
        Backoff {
            initial: Duration::from_millis(200),
            max: Duration::from_secs(30),
            multiplier: 2.0,
            jitter: Jitter::Equal,
        }
    }
}

impl Backoff {
    /// The delay before attempt number `attempt` (0-based: `0` is the wait after
    /// the first failure), given the previous delay actually used.
    pub fn delay(&self, attempt: u32, prev: Duration, rng: &mut Rng) -> Duration {
        let initial = self.initial.as_secs_f64().max(0.0);
        let max = self.max.as_secs_f64().max(initial);
        let mult = if self.multiplier.is_finite() && self.multiplier >= 1.0 {
            self.multiplier
        } else {
            1.0
        };

        let secs = match self.jitter {
            Jitter::Decorrelated => {
                let lo = initial;
                let hi = (prev.as_secs_f64().max(initial) * 3.0).min(max);
                if hi <= lo { lo } else { rng.uniform(lo, hi) }
            }
            other => {
                // `powi` on a u32 exponent overflows to inf long before it
                // matters; the min against `max` absorbs that.
                let base = (initial * mult.powi(attempt.min(64) as i32)).min(max);
                match other {
                    Jitter::None => base,
                    Jitter::Full => rng.uniform(0.0, base),
                    Jitter::Equal => rng.uniform(base / 2.0, base),
                    Jitter::Decorrelated => unreachable!(),
                }
            }
        };

        Duration::from_secs_f64(secs.clamp(0.0, max))
    }
}

/// What the retry loop should do next.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    /// Sleep this long, then try again.
    Retry(Duration),
    /// Stop. The error the caller already holds is the final answer.
    GiveUp(GiveUp),
}

/// Why a retry loop stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GiveUp {
    /// The failure would recur identically — bad credentials, a schema change,
    /// a typo in the config.
    Permanent,
    /// Out of attempts.
    Exhausted,
    /// The next delay would run past the caller's deadline. Sleeping into a
    /// timeout that has already been decided is pure latency.
    Deadline,
}

/// Attempt limits and the overall time budget.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RetryPolicy {
    /// Total attempts including the first. `1` disables retrying.
    pub max_attempts: u32,
    pub backoff: Backoff,
    /// Wall-clock ceiling for the whole sequence, if any. A market-data poll
    /// that arrives after the next poll was due is worthless, so this is
    /// usually the binding constraint rather than the attempt count.
    pub budget: Option<Duration>,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        RetryPolicy {
            max_attempts: 4,
            backoff: Backoff::default(),
            budget: Some(Duration::from_secs(20)),
        }
    }
}

impl RetryPolicy {
    /// No retrying at all. For calls whose side effects are not idempotent —
    /// order placement, most obviously, where a "timeout" may well mean the
    /// order rested.
    pub fn none() -> Self {
        RetryPolicy { max_attempts: 1, backoff: Backoff::default(), budget: None }
    }

    /// Start a sequence.
    pub fn start(&self) -> Retry {
        Retry { policy: *self, attempts: 0, prev: Duration::ZERO }
    }
}

/// One retry sequence in progress.
#[derive(Debug, Clone, PartialEq)]
pub struct Retry {
    policy: RetryPolicy,
    attempts: u32,
    prev: Duration,
}

impl Retry {
    /// Attempts made so far.
    pub fn attempts(&self) -> u32 {
        self.attempts
    }

    /// Record a failure and decide what to do about it. `elapsed` is how long
    /// the sequence has been running, which is what the budget is measured
    /// against.
    pub fn on_error(&mut self, err: &DataError, elapsed: Duration, rng: &mut Rng) -> Decision {
        self.attempts += 1;

        if !err.is_transient() {
            return Decision::GiveUp(GiveUp::Permanent);
        }
        if self.attempts >= self.policy.max_attempts {
            return Decision::GiveUp(GiveUp::Exhausted);
        }

        let mut delay = self.policy.backoff.delay(self.attempts - 1, self.prev, rng);

        // A venue that states when to come back knows more than our curve does.
        // Take the later of the two: obeying its instruction is mandatory,
        // beating our own schedule is not.
        if let Some(after) = err.retry_after() {
            delay = delay.max(after);
        }

        if let Some(budget) = self.policy.budget
            && elapsed + delay >= budget
        {
            return Decision::GiveUp(GiveUp::Deadline);
        }

        self.prev = delay;
        Decision::Retry(delay)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transient() -> DataError {
        DataError::Http { venue: "v".into(), status: 503, detail: String::new() }
    }

    fn permanent() -> DataError {
        DataError::Auth { venue: "v".into(), detail: "bad key".into() }
    }

    fn unbounded() -> RetryPolicy {
        RetryPolicy { max_attempts: 100, budget: None, ..RetryPolicy::default() }
    }

    #[test]
    fn a_permanent_failure_is_not_retried_even_once() {
        let mut r = RetryPolicy::default().start();
        let mut rng = Rng::new(1);
        assert_eq!(
            r.on_error(&permanent(), Duration::ZERO, &mut rng),
            Decision::GiveUp(GiveUp::Permanent)
        );
    }

    #[test]
    fn attempts_are_capped() {
        let policy = RetryPolicy { max_attempts: 3, budget: None, ..RetryPolicy::default() };
        let mut r = policy.start();
        let mut rng = Rng::new(2);
        assert!(matches!(r.on_error(&transient(), Duration::ZERO, &mut rng), Decision::Retry(_)));
        assert!(matches!(r.on_error(&transient(), Duration::ZERO, &mut rng), Decision::Retry(_)));
        assert_eq!(
            r.on_error(&transient(), Duration::ZERO, &mut rng),
            Decision::GiveUp(GiveUp::Exhausted),
            "three attempts means two retries"
        );
    }

    #[test]
    fn a_single_attempt_policy_never_retries() {
        let mut r = RetryPolicy::none().start();
        let mut rng = Rng::new(3);
        assert_eq!(
            r.on_error(&transient(), Duration::ZERO, &mut rng),
            Decision::GiveUp(GiveUp::Exhausted)
        );
    }

    #[test]
    fn the_delay_grows_and_then_stops_growing() {
        let backoff = Backoff {
            initial: Duration::from_millis(100),
            max: Duration::from_secs(2),
            multiplier: 2.0,
            jitter: Jitter::None,
        };
        let mut rng = Rng::new(4);
        let d: Vec<f64> =
            (0..8).map(|a| backoff.delay(a, Duration::ZERO, &mut rng).as_secs_f64()).collect();
        assert!((d[0] - 0.1).abs() < 1e-9);
        assert!((d[1] - 0.2).abs() < 1e-9);
        assert!((d[2] - 0.4).abs() < 1e-9);
        assert!(d.iter().all(|x| *x <= 2.0 + 1e-9), "the cap binds: {d:?}");
        assert!((d[7] - 2.0).abs() < 1e-9);
    }

    #[test]
    fn jitter_stays_inside_its_advertised_band() {
        let mut rng = Rng::new(5);
        let base = Backoff {
            initial: Duration::from_secs(1),
            max: Duration::from_secs(100),
            multiplier: 2.0,
            jitter: Jitter::Full,
        };
        for _ in 0..500 {
            let d = base.delay(3, Duration::ZERO, &mut rng).as_secs_f64();
            assert!((0.0..=8.0).contains(&d), "full jitter out of band: {d}");
        }
        let equal = Backoff { jitter: Jitter::Equal, ..base };
        for _ in 0..500 {
            let d = equal.delay(3, Duration::ZERO, &mut rng).as_secs_f64();
            assert!((4.0..=8.0).contains(&d), "equal jitter out of band: {d}");
        }
    }

    #[test]
    fn jitter_actually_spreads_the_herd() {
        // The whole point: two clients failing at the same instant must not
        // come back at the same instant.
        let b = Backoff { jitter: Jitter::Full, ..Backoff::default() };
        let mut a = Rng::new(11);
        let mut c = Rng::new(12);
        let hits = (0..200)
            .filter(|_| b.delay(2, Duration::ZERO, &mut a) == b.delay(2, Duration::ZERO, &mut c))
            .count();
        assert!(hits < 5, "{hits} collisions is not jitter");
    }

    #[test]
    fn decorrelated_jitter_stays_between_the_floor_and_the_cap() {
        let b = Backoff {
            initial: Duration::from_millis(100),
            max: Duration::from_secs(5),
            multiplier: 2.0,
            jitter: Jitter::Decorrelated,
        };
        let mut rng = Rng::new(6);
        let mut prev = Duration::ZERO;
        for _ in 0..200 {
            let d = b.delay(0, prev, &mut rng);
            assert!(d >= Duration::from_millis(100) && d <= Duration::from_secs(5), "{d:?}");
            prev = d;
        }
    }

    #[test]
    fn a_venues_retry_after_wins_when_it_is_longer() {
        let mut r = unbounded().start();
        let mut rng = Rng::new(7);
        let err = DataError::RateLimited {
            venue: "v".into(),
            retry_after: Some(Duration::from_secs(60)),
        };
        match r.on_error(&err, Duration::ZERO, &mut rng) {
            Decision::Retry(d) => assert!(d >= Duration::from_secs(60), "ignored the venue: {d:?}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn we_do_not_sleep_past_a_deadline_we_have_already_missed() {
        let policy = RetryPolicy {
            max_attempts: 10,
            budget: Some(Duration::from_secs(5)),
            backoff: Backoff {
                initial: Duration::from_secs(4),
                jitter: Jitter::None,
                ..Backoff::default()
            },
        };
        let mut r = policy.start();
        let mut rng = Rng::new(8);
        assert_eq!(
            r.on_error(&transient(), Duration::from_secs(2), &mut rng),
            Decision::GiveUp(GiveUp::Deadline),
            "2s spent + 4s wait overruns a 5s budget; sleeping into it is pure latency"
        );
    }

    #[test]
    fn the_schedule_is_reproducible_from_its_seed() {
        let run = || {
            let mut r = unbounded().start();
            let mut rng = Rng::new(99);
            (0..5).map(|_| r.on_error(&transient(), Duration::ZERO, &mut rng)).collect::<Vec<_>>()
        };
        assert_eq!(run(), run(), "a replayed outage must retry identically");
    }
}
