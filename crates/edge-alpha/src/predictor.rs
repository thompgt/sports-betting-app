//! Online price prediction.
//!
//! The model here is deliberately modest, and the shape of it matters more than
//! the capacity. Three decisions define it:
//!
//! **It predicts the residual, not the price.** The market logit enters as a
//! fixed offset, so the weights only ever learn *how the market is wrong*. A
//! model that learns nothing outputs the market price exactly. Contrast the
//! usual approach — regress features onto the outcome from scratch — where an
//! untrained model outputs something arbitrary and the trading layer has to
//! guess whether to believe it.
//!
//! **It is calibrated online.** A trading signal is only useful if 30% means
//! 30%. A Platt-style affine correction on the model's residual score is fitted
//! alongside the weights, so systematic over-confidence is squeezed out
//! continuously rather than discovered in a post-mortem.
//!
//! **It earns its weight.** The blend against the market price is governed by
//! *demonstrated out-of-sample skill*: the model's Brier score against the
//! market's, over predictions that were made strictly before their outcomes were
//! known. No skill, no weight, no trades. This is the single most important
//! property in the crate — it is what stops a plausible-looking model from
//! quietly financing its own overfitting.
//!
//! Learning is AdaGrad on log loss. Per-coordinate rates matter here because the
//! features are wildly different in scale and in how often they are non-zero:
//! `order_flow` on a quiet market is zero for hours, then informative, and a
//! single global learning rate either crawls on it or diverges on `mid`.

use edge_core::stats::{ForecastScore, Welford};
use edge_core::types::Prob;
use serde::{Deserialize, Serialize};

use crate::features::{FEATURE_NAMES, Features, N_FEATURES};

/// Index of the constant term. It is passed through standardisation unchanged —
/// a constant has zero variance, and standardising it would erase the intercept.
pub const BIAS_IDX: usize = N_FEATURES - 1;

#[inline]
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Running per-feature mean and variance, used to put every input on a
/// comparable scale before it reaches the weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Standardizer {
    stats: Vec<Welford>,
    clip: f64,
}

impl Default for Standardizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Standardizer {
    pub fn new() -> Self {
        Standardizer {
            stats: vec![Welford::new(); N_FEATURES],
            clip: 5.0,
        }
    }

    pub fn observe(&mut self, f: &Features) {
        for (s, v) in self.stats.iter_mut().zip(f.values().iter()) {
            s.push(*v);
        }
    }

    pub fn count(&self) -> u64 {
        self.stats[0].count()
    }

    /// Standardise, clipping to `±clip` standard deviations. Clipping is not
    /// cosmetic: one bad tick with a 60-sigma spread would otherwise move every
    /// weight at once and take hours to unlearn.
    pub fn transform(&self, f: &Features) -> [f64; N_FEATURES] {
        let mut z = [0.0; N_FEATURES];
        for (i, zi) in z.iter_mut().enumerate() {
            if i == BIAS_IDX {
                *zi = f.values()[i];
                continue;
            }
            let s = &self.stats[i];
            let sd = s.std_dev();
            // Below a handful of observations the estimated moments are noise;
            // feeding them in would make early updates chase their own scaling.
            *zi = if s.count() < 10 || !sd.is_finite() || sd <= 1e-9 {
                0.0
            } else {
                ((f.values()[i] - s.mean()) / sd).clamp(-self.clip, self.clip)
            };
        }
        z
    }

    pub fn mean(&self, i: usize) -> f64 {
        self.stats[i].mean()
    }

    pub fn std_dev(&self, i: usize) -> f64 {
        self.stats[i].std_dev()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PredictorConfig {
    /// AdaGrad base rate for the feature weights.
    pub learning_rate: f64,
    /// L2 penalty. Non-zero by default: with sixteen correlated microstructure
    /// features and a non-stationary target, unpenalised weights drift.
    pub l2: f64,
    /// Learning rate for the calibration parameters. An order of magnitude
    /// slower than the weights so calibration tracks the model rather than
    /// fighting it.
    pub calibration_rate: f64,
    /// Cap on the residual the model may add, in log-odds. Two log-odds is
    /// roughly 50c → 88c: a hard bound on how far a single model is allowed to
    /// disagree with a liquid market, which is the difference between an edge
    /// and a bug.
    pub max_residual: f64,
    /// Resolved predictions required before the model may reach full weight.
    /// Until then its blend weight is scaled down proportionally.
    pub confidence_samples: f64,
    /// Minimum standardiser observations before the model predicts anything
    /// other than the market price.
    pub warmup: u64,
}

impl Default for PredictorConfig {
    fn default() -> Self {
        PredictorConfig {
            learning_rate: 0.05,
            l2: 1e-4,
            calibration_rate: 0.005,
            max_residual: 2.0,
            confidence_samples: 200.0,
            warmup: 50,
        }
    }
}

/// One forecast, carrying everything needed to learn from it later.
///
/// Predictions are made now and resolved hours or days later, so the state that
/// the update needs travels with the prediction rather than being reconstructed
/// from a market that has since moved.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Prediction {
    /// Standardised inputs the forecast was made from.
    pub z: [f64; N_FEATURES],
    /// Market log-odds at prediction time.
    pub market_logit: f64,
    /// Raw residual score, before calibration.
    pub score: f64,
    /// The market's own implied probability.
    pub market: Prob,
    /// The model's calibrated probability, before blending.
    pub model: Prob,
    /// What the system will actually trade against: model and market blended by
    /// demonstrated skill.
    pub fair: Prob,
    /// Blend weight applied to the model, in `[0, 1]`.
    pub weight: f64,
}

impl Prediction {
    /// Signed disagreement with the market, in probability. This — not the
    /// model probability — is what a strategy sizes on.
    pub fn edge(&self) -> f64 {
        self.fair.get() - self.market.get()
    }

    /// True when the model is contributing nothing and the forecast is just the
    /// market price echoed back.
    pub fn is_market_echo(&self) -> bool {
        self.weight <= 0.0
    }
}

/// Online logistic model over microstructure features, anchored to the market.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Predictor {
    cfg: PredictorConfig,
    weights: Vec<f64>,
    /// AdaGrad accumulated squared gradient, per coordinate.
    accum: Vec<f64>,
    /// Platt slope and intercept on the residual score.
    cal_a: f64,
    cal_b: f64,
    standardizer: Standardizer,
    /// Score of the model's own probability, out of sample.
    model_score: ForecastScore,
    /// Score of the market price over the same predictions. The benchmark the
    /// model has to beat before it is allowed to move a price.
    market_score: ForecastScore,
    /// Score of what was actually traded on, for reporting.
    blended_score: ForecastScore,
    updates: u64,
}

impl Default for Predictor {
    fn default() -> Self {
        Self::new(PredictorConfig::default())
    }
}

impl Predictor {
    pub fn new(cfg: PredictorConfig) -> Self {
        Predictor {
            cfg,
            weights: vec![0.0; N_FEATURES],
            accum: vec![0.0; N_FEATURES],
            cal_a: 1.0,
            cal_b: 0.0,
            standardizer: Standardizer::new(),
            model_score: ForecastScore::new(),
            market_score: ForecastScore::new(),
            blended_score: ForecastScore::new(),
            updates: 0,
        }
    }

    pub fn config(&self) -> &PredictorConfig {
        &self.cfg
    }

    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Weights paired with their feature names, largest magnitude first. The
    /// first thing to look at when a model starts behaving strangely.
    pub fn attribution(&self) -> Vec<(&'static str, f64)> {
        let mut v: Vec<_> = FEATURE_NAMES.iter().copied().zip(self.weights.iter().copied()).collect();
        v.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
        v
    }

    pub fn updates(&self) -> u64 {
        self.updates
    }

    pub fn model_score(&self) -> &ForecastScore {
        &self.model_score
    }

    pub fn market_score(&self) -> &ForecastScore {
        &self.market_score
    }

    pub fn blended_score(&self) -> &ForecastScore {
        &self.blended_score
    }

    pub fn calibration(&self) -> (f64, f64) {
        (self.cal_a, self.cal_b)
    }

    /// Fractional Brier improvement over the market, out of sample. Zero or
    /// negative means the model has demonstrated nothing.
    pub fn skill_vs_market(&self) -> f64 {
        if self.model_score.n == 0 {
            return 0.0;
        }
        let market = self.market_score.brier();
        if !market.is_finite() || market <= 0.0 {
            return 0.0;
        }
        (market - self.model_score.brier()) / market
    }

    /// How much of the model's opinion is currently trusted, in `[0, 1]`.
    ///
    /// Two independent gates, multiplied: measured skill against the market, and
    /// how many resolutions that measurement rests on. A model that has beaten
    /// the market on nine samples is not yet allowed to trade like one that has
    /// done it on nine hundred.
    pub fn weight(&self) -> f64 {
        if self.standardizer.count() < self.cfg.warmup {
            return 0.0;
        }
        let skill = self.skill_vs_market().clamp(0.0, 1.0);
        if skill <= 0.0 {
            return 0.0;
        }
        let n = self.model_score.n as f64;
        let confidence = n / (n + self.cfg.confidence_samples);
        skill * confidence
    }

    /// Forecast, and fold the inputs into the running feature statistics.
    ///
    /// Takes `&mut self` because the standardiser learns from every observation,
    /// including ones that never resolve. That is intentional — feature *scale*
    /// can be estimated from far more data than feature *usefulness* can.
    pub fn predict(&mut self, f: &Features, market: Prob) -> Prediction {
        self.standardizer.observe(f);
        self.predict_static(f, market)
    }

    /// Forecast without updating any state. For backtests and for scoring the
    /// same observation twice without double-counting it.
    pub fn predict_static(&self, f: &Features, market: Prob) -> Prediction {
        let z = self.standardizer.transform(f);
        let score: f64 = self.weights.iter().zip(z.iter()).map(|(w, x)| w * x).sum();
        let market_logit = market.logit();

        let residual = (self.cal_a * score + self.cal_b).clamp(-self.cfg.max_residual, self.cfg.max_residual);
        let model = Prob::from_logit(market_logit + residual);

        let weight = self.weight();
        // Blend in log-odds, not in probability. Averaging 2c and 4c linearly
        // gives 3c, which is a different thing from the average of the two
        // beliefs; in log-odds the blend respects the geometry of the quantity
        // being averaged.
        let fair = Prob::from_logit(market_logit + weight * residual);

        Prediction {
            z,
            market_logit,
            score,
            market,
            model,
            fair,
            weight,
        }
    }

    /// Learn from a resolved prediction.
    ///
    /// `outcome` is whether the market resolved YES. This is the only place
    /// weights move, and it is driven exclusively by realised outcomes — never
    /// by subsequent prices, which would let the model learn to chase the
    /// market it is supposed to be disagreeing with.
    pub fn learn(&mut self, p: &Prediction, outcome: bool) {
        if !p.z.iter().all(|v| v.is_finite()) || !p.market_logit.is_finite() {
            return;
        }
        let y = if outcome { 1.0 } else { 0.0 };

        // Base weights, against the uncalibrated residual.
        let p_raw = sigmoid(p.market_logit + p.score.clamp(-self.cfg.max_residual, self.cfg.max_residual));
        let err = p_raw - y;
        for i in 0..N_FEATURES {
            let g = err * p.z[i] + self.cfg.l2 * self.weights[i];
            self.accum[i] += g * g;
            self.weights[i] -= self.cfg.learning_rate * g / (self.accum[i].sqrt() + 1e-8);
        }

        // Calibration, treating the score as a fixed input.
        let cal_err = sigmoid(p.market_logit + self.cal_a * p.score + self.cal_b) - y;
        self.cal_a -= self.cfg.calibration_rate * cal_err * p.score;
        self.cal_b -= self.cfg.calibration_rate * cal_err;
        // A negative slope would invert the model's own signal, which is never
        // the right repair for a bad model — de-weighting it is.
        self.cal_a = self.cal_a.clamp(0.0, 5.0);
        self.cal_b = self.cal_b.clamp(-1.0, 1.0);

        self.model_score.push(p.model.get(), outcome);
        self.market_score.push(p.market.get(), outcome);
        self.blended_score.push(p.fair.get(), outcome);
        self.updates += 1;
    }

    /// Predict and immediately learn, for backtests over already-resolved data.
    pub fn observe_resolved(&mut self, f: &Features, market: Prob, outcome: bool) -> Prediction {
        let p = self.predict(f, market);
        self.learn(&p, outcome);
        p
    }

    /// Drop everything learned, keeping the configuration. Used when a model is
    /// deliberately retired — for instance at the start of a new season, where
    /// the previous regime's weights are worse than no weights.
    pub fn reset(&mut self) {
        *self = Predictor::new(self.cfg);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use edge_core::rng::Rng;

    /// Build a feature vector directly, bypassing the book. `signal` is placed
    /// in the `order_flow` slot; everything else is noise.
    fn synth(rng: &mut Rng, mid: f64, signal: f64) -> Features {
        let mut v = [0.0; N_FEATURES];
        for x in v.iter_mut() {
            *x = rng.normal();
        }
        v[0] = mid;
        v[10] = signal;
        v[13] = 1.0;
        v[BIAS_IDX] = 1.0;
        Features::from_values(v, mid)
    }

    #[test]
    fn the_bias_slot_is_where_we_think_it_is() {
        assert_eq!(FEATURE_NAMES[BIAS_IDX], "bias");
    }

    #[test]
    fn an_untrained_model_returns_the_market_price_exactly() {
        let mut p = Predictor::default();
        let mut rng = Rng::new(1);
        let f = synth(&mut rng, 0.42, 0.0);
        let out = p.predict(&f, Prob::new(0.42).unwrap());

        assert!(out.is_market_echo());
        assert!((out.fair.get() - 0.42).abs() < 1e-12);
        assert_eq!(out.edge(), 0.0);
    }

    #[test]
    fn standardisation_leaves_the_intercept_alone() {
        let mut s = Standardizer::new();
        let mut rng = Rng::new(7);
        for _ in 0..50 {
            let sig = rng.normal();
            s.observe(&synth(&mut rng, 0.5, sig));
        }
        let z = s.transform(&synth(&mut rng, 0.5, 0.0));
        assert_eq!(z[BIAS_IDX], 1.0);
    }

    #[test]
    fn standardisation_puts_features_on_a_common_scale() {
        let mut s = Standardizer::new();
        let mut rng = Rng::new(11);
        for _ in 0..2_000 {
            let mut f = [0.0; N_FEATURES];
            // Deliberately absurd scale: this is the case standardisation exists for.
            f[10] = rng.gaussian(500.0, 250.0);
            f[BIAS_IDX] = 1.0;
            s.observe(&Features::from_values(f, 0.5));
        }
        let mut f = [0.0; N_FEATURES];
        f[10] = 750.0;
        f[BIAS_IDX] = 1.0;
        let z = s.transform(&Features::from_values(f, 0.5));
        assert!((z[10] - 1.0).abs() < 0.2, "expected ~1 sigma, got {}", z[10]);
    }

    #[test]
    fn standardisation_survives_a_constant_feature() {
        let mut s = Standardizer::new();
        for _ in 0..50 {
            s.observe(&Features::from_values([3.0; N_FEATURES], 0.5));
        }
        let z = s.transform(&Features::from_values([3.0; N_FEATURES], 0.5));
        assert!(z.iter().take(BIAS_IDX).all(|v| *v == 0.0));
    }

    #[test]
    fn standardisation_clips_a_wild_observation() {
        let mut s = Standardizer::new();
        let mut rng = Rng::new(3);
        for _ in 0..500 {
            let mut f = [0.0; N_FEATURES];
            f[10] = rng.normal();
            s.observe(&Features::from_values(f, 0.5));
        }
        let mut f = [0.0; N_FEATURES];
        f[10] = 1e6;
        let z = s.transform(&Features::from_values(f, 0.5));
        assert_eq!(z[10], 5.0);
    }

    #[test]
    fn a_model_with_no_skill_never_earns_weight() {
        // The market is right; the feature is pure noise. Nothing here should
        // ever move a price, however long it trains.
        let mut p = Predictor::default();
        let mut rng = Rng::new(21);
        for _ in 0..3_000 {
            let truth = rng.uniform(0.15, 0.85);
            let sig = rng.normal();
            let f = synth(&mut rng, truth, sig);
            p.observe_resolved(&f, Prob::new(truth).unwrap(), rng.bernoulli(truth));
        }
        assert!(p.skill_vs_market() <= 0.02, "skill {}", p.skill_vs_market());
        assert!(p.weight() < 0.05, "weight {}", p.weight());
    }

    #[test]
    fn a_model_that_beats_the_market_earns_weight_and_moves_the_price() {
        // The market quotes a biased price; the signal feature carries exactly
        // the information the market is missing.
        let mut p = Predictor::default();
        let mut rng = Rng::new(33);
        for _ in 0..6_000 {
            let signal = rng.normal();
            let quoted = rng.uniform(0.25, 0.75);
            let truth = Prob::from_logit(Prob::new(quoted).unwrap().logit() + 0.9 * signal);
            let f = synth(&mut rng, quoted, signal);
            p.observe_resolved(&f, Prob::new(quoted).unwrap(), rng.bernoulli(truth.get()));
        }

        assert!(p.skill_vs_market() > 0.05, "skill {}", p.skill_vs_market());
        assert!(p.weight() > 0.1, "weight {}", p.weight());
        assert!(
            p.blended_score().brier() < p.market_score().brier(),
            "blended {} vs market {}",
            p.blended_score().brier(),
            p.market_score().brier()
        );

        // And it learned the right feature, not a coincidence elsewhere.
        assert_eq!(p.attribution()[0].0, "order_flow");
    }

    #[test]
    fn the_residual_is_bounded_however_extreme_the_input() {
        let mut p = Predictor::new(PredictorConfig {
            max_residual: 1.0,
            ..Default::default()
        });
        // Force a large weight by hand, then check the clamp holds.
        let mut rng = Rng::new(5);
        for _ in 0..100 {
            let sig = rng.normal();
            p.predict(&synth(&mut rng, 0.5, sig), Prob::new(0.5).unwrap());
        }
        p.weights[10] = 50.0;
        let out = p.predict(&synth(&mut rng, 0.5, 5.0), Prob::new(0.5).unwrap());
        assert!(out.model.logit() - out.market_logit <= 1.0 + 1e-9);
    }

    #[test]
    fn learning_ignores_a_corrupted_prediction() {
        let mut p = Predictor::default();
        let mut rng = Rng::new(9);
        let mut bad = p.predict(&synth(&mut rng, 0.5, 1.0), Prob::new(0.5).unwrap());
        bad.z[3] = f64::NAN;
        p.learn(&bad, true);
        assert_eq!(p.updates(), 0);
        assert!(p.weights().iter().all(|w| *w == 0.0));
    }

    #[test]
    fn calibration_slope_never_inverts_the_signal() {
        let mut p = Predictor::default();
        let mut rng = Rng::new(13);
        // Train against outcomes that systematically contradict the score.
        for _ in 0..2_000 {
            let f = synth(&mut rng, 0.5, 3.0);
            let mut pred = p.predict(&f, Prob::new(0.5).unwrap());
            pred.score = 2.0;
            p.learn(&pred, false);
        }
        let (a, _) = p.calibration();
        assert!(a >= 0.0, "slope {a}");
    }

    #[test]
    fn a_reset_model_is_indistinguishable_from_a_new_one() {
        let mut p = Predictor::default();
        let mut rng = Rng::new(17);
        for _ in 0..500 {
            let sig = rng.normal();
            let f = synth(&mut rng, 0.5, sig);
            p.observe_resolved(&f, Prob::new(0.5).unwrap(), rng.bernoulli(0.5));
        }
        p.reset();
        assert_eq!(p.updates(), 0);
        assert_eq!(p.weight(), 0.0);
        assert_eq!(p.calibration(), (1.0, 0.0));
        assert!(p.weights().iter().all(|w| *w == 0.0));
    }

    #[test]
    fn predicting_statically_does_not_change_the_model() {
        let mut p = Predictor::default();
        let mut rng = Rng::new(19);
        for _ in 0..100 {
            let sig = rng.normal();
            p.predict(&synth(&mut rng, 0.5, sig), Prob::new(0.5).unwrap());
        }
        let before = p.standardizer.count();
        let f = synth(&mut rng, 0.5, 1.0);
        let a = p.predict_static(&f, Prob::new(0.5).unwrap());
        let b = p.predict_static(&f, Prob::new(0.5).unwrap());
        assert_eq!(p.standardizer.count(), before);
        assert_eq!(a.score, b.score);
    }

    #[test]
    fn a_prediction_round_trips_through_json() {
        let mut p = Predictor::default();
        let mut rng = Rng::new(23);
        let out = p.predict(&synth(&mut rng, 0.6, 0.5), Prob::new(0.6).unwrap());
        let s = serde_json::to_string(&out).unwrap();
        let back: Prediction = serde_json::from_str(&s).unwrap();
        assert_eq!(out, back);
    }

    #[test]
    fn a_trained_model_round_trips_through_json() {
        let mut p = Predictor::default();
        let mut rng = Rng::new(29);
        for _ in 0..300 {
            let sig = rng.normal();
            let f = synth(&mut rng, 0.5, sig);
            p.observe_resolved(&f, Prob::new(0.5).unwrap(), rng.bernoulli(0.5));
        }
        let s = serde_json::to_string(&p).unwrap();
        let back: Predictor = serde_json::from_str(&s).unwrap();
        // Compared to a tolerance rather than bit-for-bit: JSON is a decimal
        // format and a weight that survives to the last few ULP is more than a
        // restarted model needs.
        for (a, b) in back.weights().iter().zip(p.weights()) {
            assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        }
        assert_eq!(back.updates(), p.updates());
        assert_eq!(back.calibration(), p.calibration());
    }
}
