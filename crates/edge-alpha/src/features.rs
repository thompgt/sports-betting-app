//! Microstructure feature extraction.
//!
//! Features are computed **incrementally**. The engine watches thousands of
//! markets and sees an update every few hundred milliseconds; anything that
//! requires re-reading a history buffer per market per tick does not fit in the
//! budget. Every accumulator here updates in constant time and constant space.
//!
//! Two properties of prediction markets shape the feature set and separate it
//! from a generic equities one:
//!
//! **Price is a probability.** A move from 2c to 4c doubles the implied odds; a
//! move from 50c to 52c barely changes them. Returns are therefore taken in
//! log-odds rather than in price, so a feature means the same thing across the
//! whole range instead of overweighting the middle.
//!
//! **Every market has a deadline.** Volatility rises and edges compress as
//! resolution approaches, and behaviour a month out has nothing in common with
//! behaviour in the final hour. Time to resolution is a first-class feature
//! rather than an afterthought.

use edge_book::book::OrderBook;
use edge_core::market::MarketSpec;
use edge_core::stats::{Ewma, RollingWindow};
use edge_core::types::{Prob, Side, Ts};

pub const N_FEATURES: usize = 16;

/// Human-readable names, in the same order as [`Features::values`]. Used for
/// model introspection and for the dashboard — a model whose weights cannot be
/// attributed to named inputs cannot be debugged.
pub const FEATURE_NAMES: [&str; N_FEATURES] = [
    "mid",
    "microprice_edge",
    "spread",
    "imbalance_top",
    "imbalance_depth",
    "momentum_fast",
    "momentum_slow",
    "trend",
    "volatility",
    "z_score",
    "order_flow",
    "log_depth",
    "depth_skew",
    "time_decay",
    "certainty",
    "bias",
];

/// One market's state, as the model sees it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Features {
    values: [f64; N_FEATURES],
    /// Mid price at extraction, carried alongside so a caller can compare a
    /// prediction against the price it was made at without recomputing.
    pub mid: f64,
}

impl Features {
    /// Build a vector directly. For backtests replaying stored features and for
    /// tests; the live path goes through [`FeatureExtractor::extract`].
    pub fn from_values(values: [f64; N_FEATURES], mid: f64) -> Self {
        Features { values, mid }
    }

    #[inline]
    pub fn values(&self) -> &[f64; N_FEATURES] {
        &self.values
    }

    pub fn named(&self) -> impl Iterator<Item = (&'static str, f64)> + '_ {
        FEATURE_NAMES.iter().copied().zip(self.values.iter().copied())
    }

    pub fn get(&self, name: &str) -> Option<f64> {
        FEATURE_NAMES.iter().position(|n| *n == name).map(|i| self.values[i])
    }

    /// Every value finite. A single NaN reaching the model corrupts its weights
    /// permanently, so this is checked before any update rather than trusted.
    pub fn is_valid(&self) -> bool {
        self.values.iter().all(|v| v.is_finite())
    }
}

/// Rolling per-market state feeding [`Features`].
#[derive(Debug, Clone)]
pub struct FeatureExtractor {
    mid_fast: Ewma,
    mid_slow: Ewma,
    /// EWMA of squared log-odds returns — realised volatility.
    variance: Ewma,
    window: RollingWindow,
    /// Signed traded volume, decayed. Positive means buyers are lifting offers.
    flow: Ewma,
    last_logit: Option<f64>,
    updates: u64,
    trades: u64,
    volume: i64,
}

impl Default for FeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureExtractor {
    pub fn new() -> Self {
        FeatureExtractor {
            mid_fast: Ewma::from_half_life(10.0),
            mid_slow: Ewma::from_half_life(120.0),
            variance: Ewma::from_half_life(60.0),
            window: RollingWindow::new(120),
            flow: Ewma::from_half_life(40.0),
            last_logit: None,
            updates: 0,
            trades: 0,
            volume: 0,
        }
    }

    /// Observations seen. The model should not be trusted on a market whose
    /// extractor has barely warmed up.
    #[inline]
    pub fn updates(&self) -> u64 {
        self.updates
    }

    #[inline]
    pub fn is_warm(&self) -> bool {
        self.updates >= 20
    }

    pub fn trades(&self) -> u64 {
        self.trades
    }

    pub fn volume(&self) -> i64 {
        self.volume
    }

    /// Fold in a book update.
    pub fn on_book(&mut self, book: &OrderBook) {
        let Some(mid) = book.microprice().or_else(|| book.mid()) else {
            return;
        };
        let p = Prob::clamped_open(mid.dollars());
        let logit = p.logit();

        if let Some(prev) = self.last_logit {
            let r = logit - prev;
            // Squared return, not absolute: volatility is what sizing and
            // spread-setting both depend on.
            self.variance.push(r * r);
        }
        self.last_logit = Some(logit);
        self.mid_fast.push(logit);
        self.mid_slow.push(logit);
        self.window.push(logit);
        self.updates += 1;
    }

    /// Fold in a trade. `side` is the aggressor's.
    pub fn on_trade(&mut self, side: Side, qty: i64) {
        let signed = qty.abs() as f64 * side.sign() as f64;
        self.flow.push(signed);
        self.trades += 1;
        self.volume += qty.abs();
    }

    /// Build a feature vector. `None` when the book is one-sided or empty, in
    /// which case there is no mid to reason about and inventing one would put a
    /// fabricated price into the model.
    pub fn extract(&self, book: &OrderBook, spec: &MarketSpec, now: Ts) -> Option<Features> {
        let mid_price = book.mid()?;
        let micro = book.microprice()?;
        let mid = Prob::clamped_open(mid_price.dollars());
        let logit = mid.logit();

        let spread = book.spread().map(|s| s.dollars()).unwrap_or(0.0);
        let imbalance_top = book.imbalance(1).unwrap_or(0.0);
        let imbalance_depth = book.imbalance(5).unwrap_or(0.0);

        let momentum_fast = if self.mid_fast.is_ready() {
            logit - self.mid_fast.mean()
        } else {
            0.0
        };
        let momentum_slow = if self.mid_slow.is_ready() {
            logit - self.mid_slow.mean()
        } else {
            0.0
        };
        let trend = if self.mid_slow.is_ready() {
            self.mid_fast.mean() - self.mid_slow.mean()
        } else {
            0.0
        };

        let volatility = self.variance.mean().max(0.0).sqrt();
        let z_score = self.window.z_score().unwrap_or(0.0).clamp(-10.0, 10.0);

        // Normalise flow so the feature is a rate rather than an accumulating
        // total, which would otherwise grow without bound on an active market.
        let flow_scale = self.flow.std_dev().max(1.0);
        let order_flow = (self.flow.mean() / flow_scale).clamp(-10.0, 10.0);

        let bid_depth: i64 = book.depth(Side::Buy, 5).iter().map(|l| l.qty.get()).sum();
        let ask_depth: i64 = book.depth(Side::Sell, 5).iter().map(|l| l.qty.get()).sum();
        let total_depth = (bid_depth + ask_depth) as f64;
        let log_depth = (1.0 + total_depth).ln();
        let depth_skew = if total_depth > 0.0 {
            (bid_depth as f64 - ask_depth as f64) / total_depth
        } else {
            0.0
        };

        // Decays from 1 toward 0 as resolution approaches, on a one-day scale.
        // Markets with no scheduled close sit at 1, which is the right prior for
        // "resolution is not imminent".
        let time_decay = match spec.seconds_to_close(now) {
            Some(s) => 1.0 - (-(s / 86_400.0)).exp(),
            None => 1.0,
        };

        let values = [
            mid.get(),
            micro.dollars() - mid_price.dollars(),
            spread,
            imbalance_top,
            imbalance_depth,
            momentum_fast,
            momentum_slow,
            trend,
            volatility,
            z_score,
            order_flow,
            log_depth,
            depth_skew,
            time_decay,
            (mid.get() - 0.5).abs() * 2.0,
            1.0,
        ];

        let f = Features {
            values,
            mid: mid.get(),
        };
        f.is_valid().then_some(f)
    }

    pub fn reset(&mut self) {
        *self = FeatureExtractor::new();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use edge_book::order::Order;
    use edge_core::market::MarketSpec;
    use edge_core::types::{EventId, MarketId, OrderId, Price, Qty, StrategyId, VenueId};

    const M: MarketId = MarketId(0);

    struct Fx {
        book: OrderBook,
        spec: MarketSpec,
        seq: u64,
        next: u64,
        ex: FeatureExtractor,
    }

    impl Fx {
        fn new() -> Self {
            Fx {
                book: OrderBook::new(M, 10_000),
                spec: MarketSpec::new(M, EventId(0), VenueId(1), "T"),
                seq: 0,
                next: 0,
                ex: FeatureExtractor::new(),
            }
        }

        fn rest(&mut self, side: Side, cents: i64, qty: i64) {
            self.next += 1;
            let mut out = Vec::new();
            self.book.submit(
                Order::limit(
                    OrderId(self.next),
                    M,
                    StrategyId(1),
                    side,
                    Price::from_cents(cents),
                    Qty(qty),
                ),
                &mut self.seq,
                &mut out,
            );
            self.ex.on_book(&self.book);
        }

        fn quote(&mut self, bid: i64, ask: i64, qty: i64) {
            self.rest(Side::Buy, bid, qty);
            self.rest(Side::Sell, ask, qty);
        }

        fn features(&self) -> Option<Features> {
            self.ex.extract(&self.book, &self.spec, Ts::from_secs(0))
        }
    }

    #[test]
    fn a_one_sided_book_yields_no_features() {
        let mut f = Fx::new();
        f.rest(Side::Buy, 40, 100);
        assert!(
            f.features().is_none(),
            "there is no mid to reason about, and inventing one feeds the model a fiction"
        );
    }

    #[test]
    fn a_two_sided_book_yields_valid_features() {
        let mut f = Fx::new();
        f.quote(40, 50, 100);
        let x = f.features().unwrap();
        assert!(x.is_valid());
        assert_eq!(x.values().len(), N_FEATURES);
        assert!((x.mid - 0.45).abs() < 1e-9);
        assert!((x.get("spread").unwrap() - 0.10).abs() < 1e-9);
        assert_eq!(x.get("bias"), Some(1.0));
    }

    #[test]
    fn every_feature_has_a_name() {
        let mut f = Fx::new();
        f.quote(40, 50, 100);
        let x = f.features().unwrap();
        assert_eq!(x.named().count(), N_FEATURES);
        assert!(x.get("nonexistent").is_none());
        // Names are unique, or attribution is meaningless.
        let mut names = FEATURE_NAMES.to_vec();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), N_FEATURES);
    }

    #[test]
    fn imbalance_reflects_the_heavier_side() {
        let mut f = Fx::new();
        f.rest(Side::Buy, 40, 900);
        f.rest(Side::Sell, 50, 100);
        let x = f.features().unwrap();
        assert!(x.get("imbalance_top").unwrap() > 0.5);
        assert!(x.get("depth_skew").unwrap() > 0.5);
        // The microprice leans toward the thin side, so its edge over mid is positive.
        assert!(x.get("microprice_edge").unwrap() > 0.0);
    }

    #[test]
    fn momentum_is_positive_after_a_rally() {
        let mut f = Fx::new();
        // Settle at a 45c mid.
        for _ in 0..30 {
            f.ex.on_book(&f.book);
        }
        f.quote(40, 50, 100);
        for _ in 0..40 {
            f.ex.on_book(&f.book);
        }
        let before = f.features().unwrap();
        assert!(before.get("momentum_fast").unwrap().abs() < 0.05);

        // Move the whole market up.
        f.book = OrderBook::new(M, 10_000);
        f.quote(70, 80, 100);
        let after = f.features().unwrap();
        assert!(
            after.get("momentum_fast").unwrap() > 0.1,
            "momentum should register the move: {}",
            after.get("momentum_fast").unwrap()
        );
        assert!(
            after.get("momentum_fast").unwrap() > after.get("momentum_slow").unwrap() * 0.5,
            "the fast average should have moved further than the slow one"
        );
    }

    #[test]
    fn returns_are_measured_in_log_odds_not_price() {
        // The point of the choice: 2c to 4c is a bigger move than 50c to 52c,
        // because it doubles the odds. A price-space return would say they are
        // identical.
        let mut a = Fx::new();
        a.quote(1, 3, 100); // mid 2c
        a.book = OrderBook::new(M, 10_000);
        a.quote(3, 5, 100); // mid 4c

        let mut b = Fx::new();
        b.quote(49, 51, 100); // mid 50c
        b.book = OrderBook::new(M, 10_000);
        b.quote(51, 53, 100); // mid 52c

        let va = a.features().unwrap().get("volatility").unwrap();
        let vb = b.features().unwrap().get("volatility").unwrap();
        assert!(
            va > vb * 5.0,
            "a 2c move at the tail must dwarf the same move at the middle: {va} vs {vb}"
        );
    }

    #[test]
    fn volatility_rises_with_movement_and_stays_flat_without_it() {
        let mut quiet = Fx::new();
        quiet.quote(45, 55, 100);
        for _ in 0..50 {
            quiet.ex.on_book(&quiet.book);
        }
        let calm = quiet.features().unwrap().get("volatility").unwrap();
        assert!(calm < 1e-9, "a static book has no volatility, got {calm}");

        let mut noisy = Fx::new();
        for i in 0..50 {
            noisy.book = OrderBook::new(M, 10_000);
            let centre = 40 + (i % 20);
            noisy.quote(centre, centre + 5, 100);
        }
        assert!(noisy.features().unwrap().get("volatility").unwrap() > 0.0);
    }

    #[test]
    fn order_flow_signs_with_the_aggressor_and_stays_bounded() {
        let mut f = Fx::new();
        f.quote(40, 50, 100);
        for _ in 0..40 {
            f.ex.on_trade(Side::Buy, 500);
        }
        let x = f.features().unwrap().get("order_flow").unwrap();
        assert!(x > 0.0, "buy aggression should read positive, got {x}");
        assert!(x.abs() <= 10.0, "flow must stay bounded, got {x}");

        for _ in 0..200 {
            f.ex.on_trade(Side::Sell, 500);
        }
        assert!(f.features().unwrap().get("order_flow").unwrap() < 0.0);
    }

    #[test]
    fn a_huge_trade_stream_does_not_let_flow_run_away() {
        // Without normalisation this feature grows without bound on an active
        // market and eventually dominates every weight in the model.
        let mut f = Fx::new();
        f.quote(40, 50, 100);
        for _ in 0..10_000 {
            f.ex.on_trade(Side::Buy, 1_000_000);
        }
        let x = f.features().unwrap();
        assert!(x.is_valid());
        assert!(x.get("order_flow").unwrap().abs() <= 10.0);
    }

    #[test]
    fn time_decay_falls_toward_resolution() {
        let mut f = Fx::new();
        f.quote(45, 55, 100);
        f.spec.closes_at = Some(Ts::from_secs(86_400));

        let far = f.ex.extract(&f.book, &f.spec, Ts::from_secs(0)).unwrap();
        let near = f.ex.extract(&f.book, &f.spec, Ts::from_secs(86_000)).unwrap();
        assert!(far.get("time_decay").unwrap() > near.get("time_decay").unwrap());
        assert!(near.get("time_decay").unwrap() < 0.01);

        // A market with no scheduled close sits at the "not imminent" prior.
        f.spec.closes_at = None;
        assert_eq!(
            f.ex.extract(&f.book, &f.spec, Ts::ZERO).unwrap().get("time_decay"),
            Some(1.0)
        );
    }

    #[test]
    fn certainty_peaks_at_the_extremes() {
        let mut mid = Fx::new();
        mid.quote(49, 51, 100);
        let mut tail = Fx::new();
        tail.quote(2, 4, 100);
        assert!(mid.features().unwrap().get("certainty").unwrap() < 0.05);
        assert!(tail.features().unwrap().get("certainty").unwrap() > 0.9);
    }

    #[test]
    fn extreme_prices_never_produce_a_non_finite_feature() {
        // 1c and 99c are real, tradable prices, and the log-odds transform is
        // where an unguarded implementation produces an infinity.
        for (bid, ask) in [(1, 2), (98, 99), (1, 99)] {
            let mut f = Fx::new();
            for _ in 0..30 {
                f.quote(bid, ask, 100);
                f.book = OrderBook::new(M, 10_000);
            }
            f.quote(bid, ask, 100);
            let x = f.features().unwrap();
            assert!(x.is_valid(), "{bid}/{ask} produced {:?}", x.values());
        }
    }

    #[test]
    fn warmth_gates_a_cold_market() {
        let mut f = Fx::new();
        assert!(!f.ex.is_warm());
        f.quote(45, 55, 100);
        for _ in 0..30 {
            f.ex.on_book(&f.book);
        }
        assert!(f.ex.is_warm());
        assert!(f.ex.updates() >= 30);

        f.ex.reset();
        assert!(!f.ex.is_warm());
        assert_eq!(f.ex.updates(), 0);
    }

    #[test]
    fn z_score_is_clamped_rather_than_exploding() {
        let mut f = Fx::new();
        for _ in 0..100 {
            f.quote(45, 55, 100);
            f.book = OrderBook::new(M, 10_000);
        }
        f.quote(1, 3, 100);
        let z = f.features().unwrap().get("z_score").unwrap();
        assert!(z.abs() <= 10.0 && z.is_finite(), "z = {z}");
    }
}
