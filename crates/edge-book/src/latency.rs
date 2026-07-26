//! Latency instrumentation.
//!
//! The number that matters for a trading loop is not the mean — it is the tail.
//! A matching path that averages 200ns but spends 50µs at the 99.9th percentile
//! will miss exactly the fills that were worth having, because the moments when
//! the system is slowest are the moments when the market is busiest. A mean
//! hides that completely.
//!
//! This is a log-linear histogram in the HdrHistogram style: buckets are powers
//! of two subdivided linearly, giving constant relative error across seven
//! orders of magnitude with no allocation and no sorting on the recording path.

use serde::{Deserialize, Serialize};

/// Linear subdivisions per power of two. Sixteen gives ~6% worst-case relative
/// error, which is finer than the measurement noise on a nanosecond timer.
const SUB_BITS: u32 = 4;
const SUB: usize = 1 << SUB_BITS;
const BUCKETS: usize = 64 * SUB;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyHistogram {
    counts: Vec<u64>,
    total: u64,
    sum: u128,
    min: u64,
    max: u64,
}

impl Default for LatencyHistogram {
    fn default() -> Self {
        Self::new()
    }
}

impl LatencyHistogram {
    pub fn new() -> Self {
        LatencyHistogram {
            counts: vec![0; BUCKETS],
            total: 0,
            sum: 0,
            min: u64::MAX,
            max: 0,
        }
    }

    #[inline]
    fn bucket(v: u64) -> usize {
        if v < SUB as u64 {
            return v as usize;
        }
        // Index within the power-of-two band, then the band offset.
        let exp = 63 - v.leading_zeros() as usize;
        let shift = exp - SUB_BITS as usize;
        let sub = ((v >> shift) as usize) & (SUB - 1);
        ((exp - SUB_BITS as usize + 1) << SUB_BITS) + sub
    }

    /// Lower bound of a bucket, used when reconstructing a percentile.
    fn bucket_value(i: usize) -> u64 {
        if i < SUB {
            return i as u64;
        }
        let band = (i >> SUB_BITS) - 1;
        let sub = (i & (SUB - 1)) as u64;
        let shift = band + SUB_BITS as usize - SUB_BITS as usize;
        ((SUB as u64) + sub) << shift
    }

    /// Record one observation, in nanoseconds.
    #[inline]
    pub fn record(&mut self, nanos: u64) {
        let b = Self::bucket(nanos).min(BUCKETS - 1);
        self.counts[b] += 1;
        self.total += 1;
        self.sum += nanos as u128;
        self.min = self.min.min(nanos);
        self.max = self.max.max(nanos);
    }

    #[inline]
    pub fn count(&self) -> u64 {
        self.total
    }

    pub fn mean(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.sum as f64 / self.total as f64
    }

    pub fn min(&self) -> u64 {
        if self.total == 0 { 0 } else { self.min }
    }

    pub fn max(&self) -> u64 {
        self.max
    }

    /// Value at a percentile, `q` in `[0, 1]`.
    pub fn percentile(&self, q: f64) -> u64 {
        if self.total == 0 {
            return 0;
        }
        let target = (q.clamp(0.0, 1.0) * self.total as f64).ceil().max(1.0) as u64;
        let mut cumulative = 0u64;
        for (i, &c) in self.counts.iter().enumerate() {
            cumulative += c;
            if cumulative >= target {
                return Self::bucket_value(i).max(self.min).min(self.max);
            }
        }
        self.max
    }

    pub fn snapshot(&self) -> LatencySnapshot {
        LatencySnapshot {
            count: self.total,
            mean_ns: self.mean(),
            min_ns: self.min(),
            p50_ns: self.percentile(0.50),
            p95_ns: self.percentile(0.95),
            p99_ns: self.percentile(0.99),
            p999_ns: self.percentile(0.999),
            max_ns: self.max,
        }
    }

    pub fn reset(&mut self) {
        self.counts.iter_mut().for_each(|c| *c = 0);
        self.total = 0;
        self.sum = 0;
        self.min = u64::MAX;
        self.max = 0;
    }
}

/// A point-in-time summary, for metrics endpoints and logs.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LatencySnapshot {
    pub count: u64,
    pub mean_ns: f64,
    pub min_ns: u64,
    pub p50_ns: u64,
    pub p95_ns: u64,
    pub p99_ns: u64,
    pub p999_ns: u64,
    pub max_ns: u64,
}

impl std::fmt::Display for LatencySnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "n={} mean={:.0}ns p50={}ns p99={}ns p99.9={}ns max={}ns",
            self.count, self.mean_ns, self.p50_ns, self.p99_ns, self.p999_ns, self.max_ns
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_histogram_reports_zeroes_not_garbage() {
        let h = LatencyHistogram::new();
        assert_eq!(h.count(), 0);
        assert_eq!(h.mean(), 0.0);
        assert_eq!(h.percentile(0.99), 0);
        assert_eq!(h.min(), 0);
    }

    #[test]
    fn small_values_are_recorded_exactly() {
        let mut h = LatencyHistogram::new();
        for v in 0..16u64 {
            h.record(v);
        }
        assert_eq!(h.count(), 16);
        assert_eq!(h.min(), 0);
        assert_eq!(h.max(), 15);
        assert_eq!(h.percentile(1.0), 15);
    }

    #[test]
    fn percentiles_land_within_the_relative_error_bound() {
        let mut h = LatencyHistogram::new();
        for v in 1..=10_000u64 {
            h.record(v);
        }
        for (q, expect) in [(0.5, 5_000.0), (0.9, 9_000.0), (0.99, 9_900.0)] {
            let got = h.percentile(q) as f64;
            let err = (got - expect).abs() / expect;
            assert!(err < 0.07, "p{q}: got {got}, expected ~{expect} ({err:.3} error)");
        }
    }

    #[test]
    fn the_tail_is_not_hidden_by_the_mean() {
        // This is the whole point: 99,000 fast samples and 1,000 slow ones.
        let mut h = LatencyHistogram::new();
        for _ in 0..99_000 {
            h.record(200);
        }
        for _ in 0..1_000 {
            h.record(50_000);
        }
        assert!(h.mean() < 1_000.0, "the mean looks fine: {}", h.mean());
        assert!(
            h.percentile(0.995) > 40_000,
            "the tail must still show: {}",
            h.percentile(0.995)
        );
        assert_eq!(h.max(), 50_000);
    }

    #[test]
    fn values_span_seven_orders_of_magnitude() {
        let mut h = LatencyHistogram::new();
        for e in 0..7 {
            h.record(10u64.pow(e));
        }
        assert_eq!(h.count(), 7);
        assert_eq!(h.max(), 1_000_000);
        assert_eq!(h.min(), 1);
        assert!(h.percentile(1.0) >= 900_000);
    }

    #[test]
    fn reset_clears_everything() {
        let mut h = LatencyHistogram::new();
        h.record(1_000);
        h.reset();
        assert_eq!(h.count(), 0);
        assert_eq!(h.max(), 0);
        assert_eq!(h.percentile(0.5), 0);
    }

    #[test]
    fn snapshot_is_monotone_across_percentiles() {
        let mut h = LatencyHistogram::new();
        for v in 1..=1000u64 {
            h.record(v * 7);
        }
        let s = h.snapshot();
        assert!(s.min_ns <= s.p50_ns);
        assert!(s.p50_ns <= s.p95_ns);
        assert!(s.p95_ns <= s.p99_ns);
        assert!(s.p99_ns <= s.p999_ns);
        assert!(s.p999_ns <= s.max_ns);
    }
}
