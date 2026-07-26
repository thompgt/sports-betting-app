//! Circuit breaking.
//!
//! Retrying protects one call from one bad moment. It does nothing about a
//! venue that is simply down, where every call will fail after burning its full
//! retry schedule first — so the poll loop slows to the pace of its timeouts,
//! the failures pile up behind it, and the system stops trading the *other*
//! venues that were fine. The breaker's job is to make that failure cheap: once
//! a venue has proven itself unhealthy, calls to it fail instantly and for free
//! until there is evidence it recovered.
//!
//! Two trip conditions, because they catch different outages. A run of
//! consecutive failures catches a hard down — connection refused, every time. A
//! failure *rate* over a rolling window catches the more common and more
//! dangerous case: a venue that answers 60% of requests, which never produces a
//! long enough run to trip a consecutive counter but is entirely unusable for
//! trading.
//!
//! Like the rest of this module the breaker is a pure state machine over an
//! explicit `Ts`, so its behaviour under a recorded outage is reproducible.

use std::collections::VecDeque;
use std::time::Duration;

use edge_core::types::Ts;

/// Which of the three states the breaker is in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    /// Healthy. Calls pass through.
    Closed,
    /// Tripped. Calls are refused without being attempted.
    Open,
    /// Cooling-off has elapsed and a limited number of probes are allowed
    /// through to test whether the venue came back.
    HalfOpen,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BreakerConfig {
    /// Consecutive failures that trip the breaker outright.
    pub consecutive_failures: u32,
    /// Failure fraction over the rolling window that trips it.
    pub failure_rate: f64,
    /// How many recent calls the rate is measured over.
    pub window: usize,
    /// Calls needed in the window before the rate is trusted. Without this, one
    /// failure out of one call reads as 100% and trips instantly.
    pub min_samples: u32,
    /// First cooling-off period.
    pub open_for: Duration,
    /// Ceiling on the escalated cooling-off period.
    pub max_open_for: Duration,
    /// Growth factor applied each time the breaker re-trips without having
    /// recovered in between. A venue in a multi-hour outage should be probed
    /// every few minutes, not every five seconds.
    pub open_multiplier: f64,
    /// Probes allowed through at once while half-open.
    pub half_open_probes: u32,
    /// Consecutive probe successes required to close again.
    pub successes_to_close: u32,
}

impl Default for BreakerConfig {
    fn default() -> Self {
        BreakerConfig {
            consecutive_failures: 5,
            failure_rate: 0.5,
            window: 20,
            min_samples: 10,
            open_for: Duration::from_secs(5),
            max_open_for: Duration::from_secs(300),
            open_multiplier: 2.0,
            half_open_probes: 1,
            successes_to_close: 2,
        }
    }
}

/// Per-venue health gate.
#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    cfg: BreakerConfig,
    state: State,
    /// Recent outcomes, `true` for failure. Bounded by `cfg.window`.
    window: VecDeque<bool>,
    failures: u32,
    consecutive: u32,
    /// When the cooling-off period ends.
    until: Ts,
    /// Successive trips without an intervening recovery, driving escalation.
    trips: u32,
    probes_in_flight: u32,
    probe_successes: u32,
    opened_count: u64,
}

impl CircuitBreaker {
    pub fn new(cfg: BreakerConfig) -> Self {
        CircuitBreaker {
            cfg,
            state: State::Closed,
            window: VecDeque::with_capacity(cfg.window.max(1)),
            failures: 0,
            consecutive: 0,
            until: Ts::ZERO,
            trips: 0,
            probes_in_flight: 0,
            probe_successes: 0,
            opened_count: 0,
        }
    }

    pub fn state(&self) -> State {
        self.state
    }

    /// How many times the breaker has tripped over its lifetime. A metric worth
    /// alerting on: a breaker that flaps is describing a venue that is neither
    /// up nor down, which is the hardest kind to trade against.
    pub fn trip_count(&self) -> u64 {
        self.opened_count
    }

    /// Failure fraction over the rolling window, or `0.0` before there is
    /// enough evidence to have an opinion.
    pub fn observed_failure_rate(&self) -> f64 {
        if self.window.is_empty() {
            return 0.0;
        }
        self.failures as f64 / self.window.len() as f64
    }

    /// May a call be attempted? On refusal, how long until it is worth asking
    /// again.
    ///
    /// Takes `&mut self` because reaching the end of the cooling-off period is
    /// itself a state transition, and a caller asking permission is exactly
    /// when that should be noticed.
    pub fn allow(&mut self, now: Ts) -> Result<(), Duration> {
        match self.state {
            State::Closed => Ok(()),
            State::Open => {
                let remaining = (self.until.0 - now.0).max(0);
                if remaining > 0 {
                    return Err(Duration::from_nanos(remaining as u64));
                }
                self.state = State::HalfOpen;
                self.probes_in_flight = 0;
                self.probe_successes = 0;
                self.take_probe()
            }
            State::HalfOpen => self.take_probe(),
        }
    }

    fn take_probe(&mut self) -> Result<(), Duration> {
        if self.probes_in_flight < self.cfg.half_open_probes.max(1) {
            self.probes_in_flight += 1;
            Ok(())
        } else {
            // Another probe is already deciding this for everyone. Come back
            // shortly rather than piling on the venue we just declared sick.
            Err(Duration::from_millis(100))
        }
    }

    pub fn on_success(&mut self, now: Ts) {
        match self.state {
            State::HalfOpen => {
                self.probes_in_flight = self.probes_in_flight.saturating_sub(1);
                self.probe_successes += 1;
                if self.probe_successes >= self.cfg.successes_to_close.max(1) {
                    self.close();
                }
            }
            _ => {
                self.record(false);
                self.consecutive = 0;
                let _ = now;
            }
        }
    }

    pub fn on_failure(&mut self, now: Ts) {
        match self.state {
            State::HalfOpen => {
                // The probe is the evidence. One failure is enough — the venue
                // is still sick and the next cooling-off period is longer.
                self.probes_in_flight = self.probes_in_flight.saturating_sub(1);
                self.trip(now);
            }
            State::Open => {}
            State::Closed => {
                self.record(true);
                self.consecutive += 1;
                if self.should_trip() {
                    self.trip(now);
                }
            }
        }
    }

    fn should_trip(&self) -> bool {
        if self.consecutive >= self.cfg.consecutive_failures.max(1) {
            return true;
        }
        self.window.len() as u32 >= self.cfg.min_samples
            && self.observed_failure_rate() >= self.cfg.failure_rate
    }

    fn record(&mut self, failed: bool) {
        let cap = self.cfg.window.max(1);
        if self.window.len() >= cap
            && let Some(old) = self.window.pop_front()
            && old
        {
            self.failures -= 1;
        }
        self.window.push_back(failed);
        if failed {
            self.failures += 1;
        }
    }

    fn trip(&mut self, now: Ts) {
        self.state = State::Open;
        self.opened_count += 1;
        self.trips += 1;
        self.probes_in_flight = 0;
        self.probe_successes = 0;

        let base = self.cfg.open_for.as_secs_f64().max(0.0);
        let mult = if self.cfg.open_multiplier.is_finite() && self.cfg.open_multiplier >= 1.0 {
            self.cfg.open_multiplier
        } else {
            1.0
        };
        let cap = self.cfg.max_open_for.as_secs_f64().max(base);
        let secs = (base * mult.powi((self.trips - 1).min(32) as i32)).min(cap);
        self.until = Ts(now.0 + (secs * 1e9) as i64);

        // The window described the venue before it was cut off. Keeping it
        // would trip the breaker again on the first failure after recovery.
        self.window.clear();
        self.failures = 0;
        self.consecutive = 0;
    }

    fn close(&mut self) {
        self.state = State::Closed;
        self.window.clear();
        self.failures = 0;
        self.consecutive = 0;
        self.probes_in_flight = 0;
        self.probe_successes = 0;
        self.trips = 0; // recovered: the next outage starts from the short delay
    }
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        CircuitBreaker::new(BreakerConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: i64 = 1_000_000_000;

    fn at(secs: f64) -> Ts {
        Ts((secs * 1e9) as i64)
    }

    fn cfg() -> BreakerConfig {
        BreakerConfig {
            consecutive_failures: 3,
            failure_rate: 0.5,
            window: 10,
            min_samples: 4,
            open_for: Duration::from_secs(5),
            max_open_for: Duration::from_secs(60),
            open_multiplier: 2.0,
            half_open_probes: 1,
            successes_to_close: 2,
        }
    }

    /// Drive `n` failures through a closed breaker.
    fn fail(b: &mut CircuitBreaker, n: usize, now: Ts) {
        for _ in 0..n {
            let _ = b.allow(now);
            b.on_failure(now);
        }
    }

    #[test]
    fn a_healthy_venue_is_never_gated() {
        let mut b = CircuitBreaker::new(cfg());
        for i in 0..1_000 {
            let now = Ts(i * S);
            assert!(b.allow(now).is_ok());
            b.on_success(now);
        }
        assert_eq!(b.state(), State::Closed);
        assert_eq!(b.trip_count(), 0);
    }

    #[test]
    fn a_run_of_failures_trips_it() {
        let mut b = CircuitBreaker::new(cfg());
        fail(&mut b, 3, Ts::ZERO);
        assert_eq!(b.state(), State::Open);
        assert!(b.allow(Ts::ZERO).is_err(), "an open circuit costs nothing to hit");
    }

    #[test]
    fn a_venue_that_half_works_trips_on_rate_not_on_a_run() {
        // Alternating success/failure never produces a run of three, but 50%
        // is unusable for trading and must still trip.
        let mut b = CircuitBreaker::new(cfg());
        for i in 0..8 {
            let now = Ts(i * S);
            let _ = b.allow(now);
            if i % 2 == 0 { b.on_failure(now) } else { b.on_success(now) }
        }
        assert_eq!(b.state(), State::Open, "50% failure is an outage");
    }

    #[test]
    fn one_failure_out_of_one_call_is_not_evidence() {
        let mut b = CircuitBreaker::new(cfg());
        fail(&mut b, 1, Ts::ZERO);
        assert_eq!(b.state(), State::Closed, "100% of one sample proves nothing");
    }

    #[test]
    fn the_refusal_reports_a_usable_wait() {
        let mut b = CircuitBreaker::new(cfg());
        fail(&mut b, 3, Ts::ZERO);
        let wait = b.allow(at(1.0)).unwrap_err();
        assert!((wait.as_secs_f64() - 4.0).abs() < 1e-3, "{wait:?}");
        assert!(b.allow(at(5.0)).is_ok(), "the advertised wait must actually be enough");
    }

    #[test]
    fn recovery_needs_more_than_one_lucky_probe() {
        let mut b = CircuitBreaker::new(cfg());
        fail(&mut b, 3, Ts::ZERO);
        b.allow(at(6.0)).unwrap();
        b.on_success(at(6.0));
        assert_eq!(b.state(), State::HalfOpen, "one success is not recovery");
        b.allow(at(6.1)).unwrap();
        b.on_success(at(6.1));
        assert_eq!(b.state(), State::Closed);
    }

    #[test]
    fn only_one_probe_gets_through_at_a_time() {
        let mut b = CircuitBreaker::new(cfg());
        fail(&mut b, 3, Ts::ZERO);
        assert!(b.allow(at(6.0)).is_ok(), "the first probe goes");
        assert!(b.allow(at(6.0)).is_err(), "the rest wait for its verdict");
    }

    #[test]
    fn a_failed_probe_reopens_immediately_and_waits_longer() {
        let mut b = CircuitBreaker::new(cfg());
        fail(&mut b, 3, Ts::ZERO);
        b.allow(at(6.0)).unwrap();
        b.on_failure(at(6.0));
        assert_eq!(b.state(), State::Open);
        let wait = b.allow(at(6.0)).unwrap_err();
        assert!((wait.as_secs_f64() - 10.0).abs() < 1e-3, "escalated to 2x: {wait:?}");
    }

    #[test]
    fn escalation_is_capped() {
        let mut b = CircuitBreaker::new(cfg());
        fail(&mut b, 3, Ts::ZERO);
        let mut t = 6.0;
        for _ in 0..20 {
            // Probe, fail, wait out the (growing) cooling-off, repeat.
            t += 200.0;
            b.allow(at(t)).unwrap();
            b.on_failure(at(t));
        }
        let wait = b.allow(at(t)).unwrap_err();
        assert!(wait <= Duration::from_secs(60), "cap breached: {wait:?}");
    }

    #[test]
    fn recovering_resets_the_escalation() {
        let mut b = CircuitBreaker::new(cfg());
        fail(&mut b, 3, Ts::ZERO);
        b.allow(at(6.0)).unwrap();
        b.on_failure(at(6.0)); // escalate to 10s
        b.allow(at(17.0)).unwrap();
        b.on_success(at(17.0));
        b.allow(at(17.1)).unwrap();
        b.on_success(at(17.1));
        assert_eq!(b.state(), State::Closed);

        fail(&mut b, 3, at(20.0));
        let wait = b.allow(at(20.0)).unwrap_err();
        assert!(
            (wait.as_secs_f64() - 5.0).abs() < 1e-3,
            "a recovered venue starts from the short delay: {wait:?}"
        );
    }

    #[test]
    fn the_history_from_before_the_outage_does_not_re_trip_it() {
        let mut b = CircuitBreaker::new(cfg());
        fail(&mut b, 3, Ts::ZERO);
        b.allow(at(6.0)).unwrap();
        b.on_success(at(6.0));
        b.allow(at(6.1)).unwrap();
        b.on_success(at(6.1));
        assert_eq!(b.state(), State::Closed);

        // A single post-recovery failure must not instantly re-trip on a window
        // still full of the old outage.
        let _ = b.allow(at(7.0));
        b.on_failure(at(7.0));
        assert_eq!(b.state(), State::Closed);
    }

    #[test]
    fn the_rate_window_forgets_old_failures() {
        let mut b = CircuitBreaker::new(BreakerConfig {
            consecutive_failures: 100,
            window: 4,
            min_samples: 4,
            failure_rate: 0.75,
            ..cfg()
        });
        for i in 0..2 {
            let _ = b.allow(Ts(i * S));
            b.on_failure(Ts(i * S));
        }
        for i in 2..8 {
            let _ = b.allow(Ts(i * S));
            b.on_success(Ts(i * S));
        }
        assert_eq!(b.state(), State::Closed);
        assert_eq!(b.observed_failure_rate(), 0.0, "the old failures aged out");
    }

    #[test]
    fn failures_reported_while_open_are_ignored() {
        // In-flight calls land after the trip. They are evidence about the past.
        let mut b = CircuitBreaker::new(cfg());
        fail(&mut b, 3, Ts::ZERO);
        let before = b.trip_count();
        for _ in 0..10 {
            b.on_failure(Ts::ZERO);
        }
        assert_eq!(b.trip_count(), before, "an open circuit cannot trip again");
    }
}
