//! Online statistics, scoring rules and risk metrics.
//!
//! Everything in this module is **streaming**. The engine sees a market update
//! every few hundred milliseconds across thousands of markets and never has the
//! whole history in memory, so an estimator that requires two passes or a stored
//! series is unusable in the hot path. The accumulators here update in constant
//! time and constant space.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Streaming moments
// ---------------------------------------------------------------------------

/// Welford's online mean and variance.
///
/// The naive `Σx²/n − (Σx/n)²` formulation catastrophically cancels when the
/// variance is small relative to the mean — which is precisely the regime here,
/// since contract prices live in `[0,1]` and move in fractions of a cent.
/// Welford is stable in that regime.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Welford {
    count: u64,
    mean: f64,
    m2: f64,
}

impl Welford {
    pub const fn new() -> Self {
        Welford {
            count: 0,
            mean: 0.0,
            m2: 0.0,
        }
    }

    pub fn push(&mut self, x: f64) {
        if !x.is_finite() {
            return;
        }
        self.count += 1;
        let delta = x - self.mean;
        self.mean += delta / self.count as f64;
        self.m2 += delta * (x - self.mean);
    }

    #[inline]
    pub const fn count(&self) -> u64 {
        self.count
    }

    #[inline]
    pub const fn mean(&self) -> f64 {
        self.mean
    }

    /// Sample variance (Bessel-corrected). Zero until two observations exist.
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            self.m2 / (self.count - 1) as f64
        }
    }

    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Standard error of the mean — the number that says whether an observed
    /// edge is distinguishable from zero.
    pub fn std_err(&self) -> f64 {
        if self.count < 2 {
            f64::INFINITY
        } else {
            self.std_dev() / (self.count as f64).sqrt()
        }
    }

    /// Merge another accumulator (Chan et al.), for combining per-shard stats.
    pub fn merge(&mut self, other: &Welford) {
        if other.count == 0 {
            return;
        }
        if self.count == 0 {
            *self = *other;
            return;
        }
        let n_a = self.count as f64;
        let n_b = other.count as f64;
        let n = n_a + n_b;
        let delta = other.mean - self.mean;
        self.mean += delta * n_b / n;
        self.m2 += other.m2 + delta * delta * n_a * n_b / n;
        self.count += other.count;
    }
}

/// Exponentially weighted mean and variance.
///
/// Preferred over a simple average anywhere the quantity is non-stationary —
/// realised volatility, fill rates, quote intensity — because a prediction
/// market's behaviour close to resolution has nothing in common with its
/// behaviour a month out, and an unweighted mean will keep insisting otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Ewma {
    alpha: f64,
    mean: f64,
    var: f64,
    initialised: bool,
}

impl Ewma {
    /// `alpha` is the weight on each new observation, in `(0, 1]`.
    pub fn new(alpha: f64) -> Self {
        Ewma {
            alpha: alpha.clamp(1e-9, 1.0),
            mean: 0.0,
            var: 0.0,
            initialised: false,
        }
    }

    /// Construct from a half-life in observations, which is usually the more
    /// natural way to think about it than a raw decay constant.
    pub fn from_half_life(half_life: f64) -> Self {
        let alpha = if half_life <= 0.0 {
            1.0
        } else {
            1.0 - (-std::f64::consts::LN_2 / half_life).exp()
        };
        Ewma::new(alpha)
    }

    pub fn push(&mut self, x: f64) {
        if !x.is_finite() {
            return;
        }
        if !self.initialised {
            self.mean = x;
            self.initialised = true;
            return;
        }
        let delta = x - self.mean;
        self.mean += self.alpha * delta;
        // West's incremental EW variance.
        self.var = (1.0 - self.alpha) * (self.var + self.alpha * delta * delta);
    }

    #[inline]
    pub const fn mean(&self) -> f64 {
        self.mean
    }

    #[inline]
    pub const fn variance(&self) -> f64 {
        self.var
    }

    pub fn std_dev(&self) -> f64 {
        self.var.sqrt()
    }

    #[inline]
    pub const fn is_ready(&self) -> bool {
        self.initialised
    }
}

/// A fixed-capacity ring of recent observations, for the statistics that
/// genuinely need a window rather than a decay (min, max, quantiles).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RollingWindow {
    buf: Vec<f64>,
    head: usize,
    len: usize,
}

impl RollingWindow {
    pub fn new(capacity: usize) -> Self {
        RollingWindow {
            buf: vec![0.0; capacity.max(1)],
            head: 0,
            len: 0,
        }
    }

    pub fn push(&mut self, x: f64) {
        if !x.is_finite() {
            return;
        }
        let cap = self.buf.len();
        self.buf[self.head] = x;
        self.head = (self.head + 1) % cap;
        if self.len < cap {
            self.len += 1;
        }
    }

    #[inline]
    pub const fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn is_full(&self) -> bool {
        self.len == self.buf.len()
    }

    /// Observations oldest-first.
    pub fn iter(&self) -> impl Iterator<Item = f64> + '_ {
        let cap = self.buf.len();
        let start = (self.head + cap - self.len) % cap;
        (0..self.len).map(move |i| self.buf[(start + i) % cap])
    }

    /// Most recent observation.
    pub fn last(&self) -> Option<f64> {
        if self.len == 0 {
            return None;
        }
        Some(self.buf[(self.head + self.buf.len() - 1) % self.buf.len()])
    }

    pub fn mean(&self) -> f64 {
        if self.len == 0 {
            return 0.0;
        }
        self.iter().sum::<f64>() / self.len as f64
    }

    pub fn std_dev(&self) -> f64 {
        if self.len < 2 {
            return 0.0;
        }
        let m = self.mean();
        (self.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (self.len - 1) as f64).sqrt()
    }

    pub fn min(&self) -> Option<f64> {
        self.iter().reduce(f64::min)
    }

    pub fn max(&self) -> Option<f64> {
        self.iter().reduce(f64::max)
    }

    /// Z-score of the most recent observation against the window. The signal
    /// the mean-reversion strategy trades on.
    pub fn z_score(&self) -> Option<f64> {
        let last = self.last()?;
        let sd = self.std_dev();
        if sd <= 0.0 {
            return None;
        }
        Some((last - self.mean()) / sd)
    }

    pub fn quantile(&self, q: f64) -> Option<f64> {
        if self.len == 0 {
            return None;
        }
        let mut v: Vec<f64> = self.iter().collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        Some(quantile_sorted(&v, q))
    }
}

/// Linear-interpolated quantile of an already-sorted slice.
pub fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let pos = q.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let w = pos - lo as f64;
        sorted[lo] * (1.0 - w) + sorted[hi] * w
    }
}

// ---------------------------------------------------------------------------
// Scoring rules
// ---------------------------------------------------------------------------

/// Brier score for a single binary forecast: `(p − outcome)²`. Lower is better,
/// 0.25 is what you get by always saying 50%.
#[inline]
pub fn brier(forecast: f64, outcome: bool) -> f64 {
    let y = if outcome { 1.0 } else { 0.0 };
    (forecast - y).powi(2)
}

/// Negative log-likelihood of a binary forecast. Punishes confident errors far
/// harder than Brier does, which is the behaviour you want from a model that is
/// about to be sized by Kelly.
#[inline]
pub fn log_loss(forecast: f64, outcome: bool) -> f64 {
    let p = forecast.clamp(1e-15, 1.0 - 1e-15);
    if outcome { -p.ln() } else { -(1.0 - p).ln() }
}

/// Running forecast quality: Brier, log loss, calibration and discrimination.
///
/// A prediction-market model is only worth sizing if it is *calibrated* — when
/// it says 30%, the thing must happen about 30% of the time. Tracked live so a
/// model that drifts out of calibration can be caught and de-weighted before it
/// has lost much.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct ForecastScore {
    pub n: u64,
    brier_sum: f64,
    log_loss_sum: f64,
    forecast_sum: f64,
    outcome_sum: f64,
}

impl ForecastScore {
    pub const fn new() -> Self {
        ForecastScore {
            n: 0,
            brier_sum: 0.0,
            log_loss_sum: 0.0,
            forecast_sum: 0.0,
            outcome_sum: 0.0,
        }
    }

    pub fn push(&mut self, forecast: f64, outcome: bool) {
        if !forecast.is_finite() {
            return;
        }
        self.n += 1;
        self.brier_sum += brier(forecast, outcome);
        self.log_loss_sum += log_loss(forecast, outcome);
        self.forecast_sum += forecast;
        self.outcome_sum += if outcome { 1.0 } else { 0.0 };
    }

    pub fn brier(&self) -> f64 {
        if self.n == 0 { f64::NAN } else { self.brier_sum / self.n as f64 }
    }

    pub fn log_loss(&self) -> f64 {
        if self.n == 0 { f64::NAN } else { self.log_loss_sum / self.n as f64 }
    }

    /// Mean forecast minus base rate. A persistently non-zero value means the
    /// model is systematically over- or under-confident and its probabilities
    /// should not be fed to Kelly until it is recalibrated.
    pub fn bias(&self) -> f64 {
        if self.n == 0 {
            return f64::NAN;
        }
        (self.forecast_sum - self.outcome_sum) / self.n as f64
    }

    /// Brier skill score against the base rate. Positive means the model beats
    /// simply predicting the historical frequency; zero or below means it adds
    /// nothing and should not be traded.
    pub fn skill_score(&self) -> f64 {
        if self.n == 0 {
            return f64::NAN;
        }
        let base = self.outcome_sum / self.n as f64;
        let reference = base * (1.0 - base);
        if reference <= 0.0 {
            return 0.0;
        }
        1.0 - self.brier() / reference
    }
}

// ---------------------------------------------------------------------------
// PnL / risk metrics
// ---------------------------------------------------------------------------

/// Annualised Sharpe ratio of a return series.
///
/// `periods_per_year` scales a per-period series to annual terms. Reported for
/// comparability, but treat it sceptically on a strategy holding binary
/// contracts: the return distribution is violently non-normal, so Sharpe
/// understates tail risk. Read it alongside max drawdown, never alone.
pub fn sharpe(returns: &[f64], periods_per_year: f64) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }
    let mut w = Welford::new();
    for &r in returns {
        w.push(r);
    }
    let sd = w.std_dev();
    if sd <= 0.0 {
        return 0.0;
    }
    w.mean() / sd * periods_per_year.sqrt()
}

/// Sortino ratio — Sharpe penalising only downside deviation. More honest than
/// Sharpe for a strategy whose upside is deliberately skewed.
pub fn sortino(returns: &[f64], periods_per_year: f64) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }
    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let downside: f64 = returns.iter().filter(|r| **r < 0.0).map(|r| r * r).sum();
    let dd = (downside / returns.len() as f64).sqrt();
    if dd <= 0.0 {
        return 0.0;
    }
    mean / dd * periods_per_year.sqrt()
}

/// Largest peak-to-trough decline in an equity curve, as a positive fraction.
pub fn max_drawdown(equity: &[f64]) -> f64 {
    let mut peak = f64::NEG_INFINITY;
    let mut worst: f64 = 0.0;
    for &e in equity {
        if e > peak {
            peak = e;
        }
        if peak > 0.0 {
            worst = worst.max((peak - e) / peak);
        }
    }
    worst
}

/// Standard normal CDF, via the complementary error function.
pub fn norm_cdf(x: f64) -> f64 {
    0.5 * erfc(-x / std::f64::consts::SQRT_2)
}

/// Complementary error function (Numerical Recipes' Chebyshev fit, ~1.2e-7 max
/// relative error — far tighter than any uncertainty in the inputs here).
fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 2.0 / (2.0 + z);
    let ty = 4.0 * t - 2.0;
    const COF: [f64; 28] = [
        -1.3026537197817094,
        6.419_697_923_564_902e-1,
        1.9476473204185836e-2,
        -9.561_514_786_808_631e-3,
        -9.46595344482036e-4,
        3.66839497852761e-4,
        4.2523324806907e-5,
        -2.0278578112534e-5,
        -1.624290004647e-6,
        1.303655835580e-6,
        1.5626441722e-8,
        -8.5238095915e-8,
        6.529054439e-9,
        5.059343495e-9,
        -9.91364156e-10,
        -2.27365122e-10,
        9.6467911e-11,
        2.394038e-12,
        -6.886027e-12,
        8.94487e-13,
        3.13092e-13,
        -1.12708e-13,
        3.81e-16,
        7.106e-15,
        -1.523e-15,
        -9.4e-17,
        1.21e-16,
        -2.8e-17,
    ];
    let mut d = 0.0;
    let mut dd = 0.0;
    for j in (1..COF.len()).rev() {
        let tmp = d;
        d = ty * d - dd + COF[j];
        dd = tmp;
    }
    let ans = t * (-z * z + 0.5 * (COF[0] + ty * d) - dd).exp();
    if x >= 0.0 { ans } else { 2.0 - ans }
}

/// Inverse standard normal CDF (Acklam's rational approximation refined by one
/// Halley step). Used to turn a VaR confidence level into a z-score.
pub fn norm_ppf(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    const P_LOW: f64 = 0.02425;

    let x = if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= 1.0 - P_LOW {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };

    // One Halley refinement takes the ~1e-9 approximation to machine precision.
    let e = norm_cdf(x) - p;
    let u = e * (2.0 * std::f64::consts::PI).sqrt() * (x * x / 2.0).exp();
    x - u / (1.0 + x * u / 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn welford_matches_the_textbook_answer() {
        let mut w = Welford::new();
        for x in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            w.push(x);
        }
        assert_eq!(w.count(), 8);
        assert!((w.mean() - 5.0).abs() < 1e-12);
        // Sample variance of that set is 32/7.
        assert!((w.variance() - 32.0 / 7.0).abs() < 1e-12);
    }

    #[test]
    fn welford_is_stable_where_the_naive_formula_is_not() {
        // Large mean, tiny variance: the naive sum-of-squares formula loses all
        // precision here and can even return a negative variance.
        let mut w = Welford::new();
        for i in 0..1000 {
            w.push(1e9 + (i % 2) as f64);
        }
        assert!(w.variance() > 0.24 && w.variance() < 0.26, "got {}", w.variance());
    }

    #[test]
    fn welford_merges() {
        let data: Vec<f64> = (0..100).map(|i| i as f64 * 0.37).collect();
        let mut all = Welford::new();
        for &x in &data {
            all.push(x);
        }
        let mut a = Welford::new();
        let mut b = Welford::new();
        for (i, &x) in data.iter().enumerate() {
            if i < 40 { a.push(x) } else { b.push(x) }
        }
        a.merge(&b);
        assert!((a.mean() - all.mean()).abs() < 1e-9);
        assert!((a.variance() - all.variance()).abs() < 1e-9);
        assert_eq!(a.count(), all.count());
    }

    #[test]
    fn welford_ignores_non_finite_input() {
        let mut w = Welford::new();
        w.push(1.0);
        w.push(f64::NAN);
        w.push(f64::INFINITY);
        w.push(3.0);
        assert_eq!(w.count(), 2);
        assert!((w.mean() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn ewma_tracks_a_level_shift_faster_than_a_mean() {
        let mut e = Ewma::from_half_life(5.0);
        for _ in 0..50 {
            e.push(0.30);
        }
        assert!((e.mean() - 0.30).abs() < 1e-6);
        for _ in 0..10 {
            e.push(0.70);
        }
        // Two half-lives covers 75% of a 0.40 gap, landing at ~0.60.
        assert!(e.mean() > 0.55, "EWMA stuck at {}", e.mean());
    }

    #[test]
    fn ewma_half_life_is_what_it_claims() {
        let mut e = Ewma::from_half_life(10.0);
        e.push(0.0);
        for _ in 0..10 {
            e.push(1.0);
        }
        assert!((e.mean() - 0.5).abs() < 0.02, "got {}", e.mean());
    }

    #[test]
    fn rolling_window_evicts_oldest_first() {
        let mut w = RollingWindow::new(3);
        for x in [1.0, 2.0, 3.0, 4.0] {
            w.push(x);
        }
        assert_eq!(w.len(), 3);
        assert_eq!(w.iter().collect::<Vec<_>>(), vec![2.0, 3.0, 4.0]);
        assert_eq!(w.last(), Some(4.0));
        assert_eq!(w.min(), Some(2.0));
        assert_eq!(w.max(), Some(4.0));
    }

    #[test]
    fn z_score_flags_an_outlier() {
        let mut w = RollingWindow::new(20);
        for _ in 0..19 {
            w.push(0.50);
        }
        w.push(0.50);
        // A flat window has no dispersion, so no z-score exists — and reporting
        // one would be an infinite signal.
        assert!(w.z_score().is_none());

        let mut w = RollingWindow::new(20);
        for i in 0..19 {
            w.push(0.50 + (i % 3) as f64 * 0.001);
        }
        w.push(0.60);
        assert!(w.z_score().unwrap() > 3.0);
    }

    #[test]
    fn quantiles_interpolate() {
        let v = [1.0, 2.0, 3.0, 4.0];
        assert!((quantile_sorted(&v, 0.5) - 2.5).abs() < 1e-12);
        assert!((quantile_sorted(&v, 0.0) - 1.0).abs() < 1e-12);
        assert!((quantile_sorted(&v, 1.0) - 4.0).abs() < 1e-12);
    }

    #[test]
    fn brier_and_log_loss_reward_being_right() {
        assert!(brier(0.9, true) < brier(0.5, true));
        assert!(log_loss(0.9, true) < log_loss(0.5, true));
        // Log loss punishes a confident miss far harder than Brier does.
        assert!(log_loss(0.01, true) / log_loss(0.5, true) > 6.0);
        assert!(brier(0.01, true) / brier(0.5, true) < 4.0);
    }

    #[test]
    fn log_loss_never_returns_infinity() {
        assert!(log_loss(0.0, true).is_finite());
        assert!(log_loss(1.0, false).is_finite());
    }

    #[test]
    fn a_perfectly_calibrated_forecaster_scores_well() {
        let mut s = ForecastScore::new();
        // Says 30% and is right 30% of the time.
        for i in 0..1000 {
            s.push(0.30, i % 10 < 3);
        }
        assert!(s.bias().abs() < 1e-9, "bias {}", s.bias());
        assert!((s.brier() - 0.21).abs() < 1e-9);
    }

    #[test]
    fn skill_score_exposes_a_useless_model() {
        // Always predicts the base rate: calibrated, but no discrimination.
        let mut s = ForecastScore::new();
        for i in 0..1000 {
            s.push(0.5, i % 2 == 0);
        }
        assert!(s.skill_score().abs() < 1e-9, "got {}", s.skill_score());

        // A model that actually knows the answer.
        let mut good = ForecastScore::new();
        for i in 0..1000 {
            let outcome = i % 2 == 0;
            good.push(if outcome { 0.95 } else { 0.05 }, outcome);
        }
        assert!(good.skill_score() > 0.9);
    }

    #[test]
    fn max_drawdown_finds_the_worst_trough() {
        let equity = [100.0, 120.0, 90.0, 130.0, 65.0, 140.0];
        // Worst is 130 -> 65, a 50% decline.
        assert!((max_drawdown(&equity) - 0.5).abs() < 1e-12);
        assert_eq!(max_drawdown(&[100.0, 110.0, 120.0]), 0.0);
    }

    #[test]
    fn sharpe_is_zero_without_dispersion() {
        assert_eq!(sharpe(&[0.01; 10], 252.0), 0.0);
        assert_eq!(sharpe(&[], 252.0), 0.0);
    }

    #[test]
    fn sortino_ignores_upside_volatility() {
        let steady = [0.01, 0.01, -0.01, 0.01];
        let spiky = [0.01, 0.10, -0.01, 0.01];
        // The upside spike raises Sortino (more mean, same downside) while
        // hurting Sharpe.
        assert!(sortino(&spiky, 252.0) > sortino(&steady, 252.0));
    }

    #[test]
    fn normal_cdf_known_values() {
        assert!((norm_cdf(0.0) - 0.5).abs() < 1e-12);
        assert!((norm_cdf(1.96) - 0.975).abs() < 1e-5);
        assert!((norm_cdf(-1.96) - 0.025).abs() < 1e-5);
        assert!((norm_cdf(-8.0)).abs() < 1e-14);
    }

    #[test]
    fn norm_ppf_inverts_norm_cdf() {
        for p in [0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999] {
            let x = norm_ppf(p);
            assert!((norm_cdf(x) - p).abs() < 1e-9, "p={p} x={x}");
        }
        assert!((norm_ppf(0.95) - 1.644_853_6).abs() < 1e-6);
        assert!((norm_ppf(0.99) - 2.326_347_9).abs() < 1e-6);
    }
}
