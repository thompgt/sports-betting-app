//! Combining several venues' opinions into one fair-value estimate.
//!
//! No single venue's devigged price is the truth. The estimate the strategies
//! trade against is a *pool* of every venue quoting the same event, and how that
//! pool is formed matters more than most of the rest of the pricing stack.
//!
//! Three choices here differ from the obvious implementation:
//!
//! 1. **Devig first, then pool.** Averaging raw quoted prices across venues
//!    averages their margins together and produces a fair value contaminated by
//!    whichever venue charges most. Each source is devigged on its own before it
//!    is allowed into the pool.
//!
//! 2. **Pool in log-odds, not in probability.** The arithmetic mean of
//!    probabilities is systematically under-confident: averaging 5% and 15%
//!    gives 10%, but the two sources actually agree the event is unlikely and
//!    the pooled estimate should sit nearer the geometric middle. Log-odds
//!    pooling is the standard fix and is what the forecasting literature
//!    consistently finds beats the linear average.
//!
//! 3. **Disagreement is an output, not noise.** When venues disagree the fair
//!    value is less certain, and position sizing should shrink accordingly. The
//!    dispersion of the pool is returned so it can be fed straight into
//!    [`crate::ev::KellyPolicy::estimate_sd`]. This closes the loop that the
//!    predecessor Python service left open: it averaged books and then sized as
//!    though the average were exact.

use serde::{Deserialize, Serialize};

use crate::devig::{DevigMethod, devig};
use crate::error::{EdgeError, Result};
use crate::types::{Prob, VenueId};

/// One venue's quote on a market's full outcome set, before devigging.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceQuote {
    pub venue: VenueId,
    /// Raw implied probabilities, one per outcome, in a consistent order across
    /// every source in the pool.
    pub implied: Vec<Prob>,
    /// Relative credibility of this venue. A sharp, high-limit venue that moves
    /// first deserves more weight than a soft recreational book that copies it —
    /// pooling them equally throws away the main thing you know about them.
    pub weight: f64,
}

impl SourceQuote {
    pub fn new(venue: VenueId, implied: Vec<Prob>, weight: f64) -> Self {
        SourceQuote { venue, implied, weight: weight.max(0.0) }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ConsensusConfig {
    pub method: DevigMethod,
    /// Fraction to trim from each tail of the per-outcome estimates before
    /// pooling, in `[0, 0.5)`. A single stale or fat-fingered venue can drag a
    /// mean a long way; trimming makes the pool robust to it.
    pub trim: f64,
    /// Minimum number of sources before a consensus is considered meaningful.
    /// One venue agreeing with itself is not a consensus, and trading a
    /// "consensus" of one is trading that venue's margin.
    pub min_sources: usize,
    /// Extremisation exponent applied in log-odds space. Pooled forecasts are
    /// known to be under-confident because sources share information; a value
    /// slightly above 1 corrects for it. Above ~1.5 it is wishful thinking.
    pub extremise: f64,
}

impl Default for ConsensusConfig {
    fn default() -> Self {
        ConsensusConfig { method: DevigMethod::Power, trim: 0.1, min_sources: 2, extremise: 1.0 }
    }
}

/// The pooled fair value plus everything needed to judge how much to trust it.
#[derive(Debug, Clone, PartialEq)]
pub struct Consensus {
    /// Pooled fair probabilities, summing to 1.
    pub fair: Vec<Prob>,
    /// How many sources survived validation and entered the pool.
    pub n_sources: usize,
    /// Cross-source standard deviation of the fair probability, per outcome.
    /// Feed the max of these into Kelly shrinkage.
    pub dispersion: Vec<f64>,
    /// Mean overround across sources — the typical margin being charged.
    pub mean_overround: f64,
    /// Widest gap between any two sources on any outcome. A large value usually
    /// means one source is stale, not that there is a large edge, and is worth
    /// gating on before trading.
    pub max_disagreement: f64,
}

impl Consensus {
    /// The estimation error to hand to [`crate::ev::KellyPolicy`].
    ///
    /// With few sources, cross-source dispersion understates true uncertainty —
    /// two venues can agree and still both be wrong — so a floor is applied that
    /// decays as the pool grows.
    pub fn estimate_sd(&self) -> f64 {
        let observed = self.dispersion.iter().copied().fold(0.0_f64, f64::max);
        let floor = 0.02 / (self.n_sources.max(1) as f64).sqrt();
        observed.max(floor)
    }

    pub fn fair_decimals(&self) -> Vec<f64> {
        self.fair.iter().map(|p| 1.0 / p.get().max(Prob::EPS)).collect()
    }
}

/// Devig each source and pool them into a single fair-value estimate.
pub fn consensus(sources: &[SourceQuote], cfg: &ConsensusConfig) -> Result<Consensus> {
    if sources.is_empty() {
        return Err(EdgeError::Empty("consensus requires at least one source"));
    }
    let n_outcomes = sources[0].implied.len();
    if n_outcomes == 0 {
        return Err(EdgeError::Empty("sources quote no outcomes"));
    }

    // Devig each source independently, dropping any the solver rejects rather
    // than letting one malformed venue take down the whole market's pricing.
    let mut fair_by_source: Vec<(Vec<f64>, f64)> = Vec::with_capacity(sources.len());
    let mut overround_sum = 0.0;
    for s in sources {
        if s.implied.len() != n_outcomes || s.weight <= 0.0 {
            continue;
        }
        let Ok(d) = devig(&s.implied, cfg.method) else {
            continue;
        };
        overround_sum += d.overround;
        fair_by_source.push((d.fair.iter().map(|p| p.get()).collect(), s.weight));
    }

    if fair_by_source.len() < cfg.min_sources.max(1) {
        return Err(EdgeError::DegenerateMarket("too few usable sources to form a consensus"));
    }

    let n_sources = fair_by_source.len();
    let mut pooled_logits = Vec::with_capacity(n_outcomes);
    let mut dispersion = Vec::with_capacity(n_outcomes);
    let mut max_disagreement: f64 = 0.0;

    for i in 0..n_outcomes {
        let mut points: Vec<(f64, f64)> =
            fair_by_source.iter().map(|(f, w)| (Prob::clamped_open(f[i]).logit(), *w)).collect();

        // Dispersion and disagreement are measured on the untrimmed pool, in
        // probability space, because that is the scale sizing cares about.
        let probs: Vec<f64> = fair_by_source.iter().map(|(f, _)| f[i]).collect();
        let mean_p = probs.iter().sum::<f64>() / n_sources as f64;
        let sd = if n_sources > 1 {
            (probs.iter().map(|p| (p - mean_p).powi(2)).sum::<f64>() / (n_sources - 1) as f64)
                .sqrt()
        } else {
            0.0
        };
        dispersion.push(sd);
        let (lo, hi) = probs.iter().fold((f64::MAX, f64::MIN), |(l, h), &p| (l.min(p), h.max(p)));
        max_disagreement = max_disagreement.max(hi - lo);

        // Trim symmetrically, then take the weighted mean of what remains.
        points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let k = ((points.len() as f64) * cfg.trim.clamp(0.0, 0.49)).floor() as usize;
        let kept = if points.len() > 2 * k { &points[k..points.len() - k] } else { &points[..] };

        let wsum: f64 = kept.iter().map(|(_, w)| w).sum();
        let logit = if wsum > 0.0 {
            kept.iter().map(|(l, w)| l * w).sum::<f64>() / wsum
        } else {
            kept.iter().map(|(l, _)| l).sum::<f64>() / kept.len() as f64
        };
        pooled_logits.push(logit * cfg.extremise);
    }

    // Pooling per-outcome breaks the sum-to-one constraint; restore it.
    let raw: Vec<f64> = pooled_logits.iter().map(|&l| Prob::from_logit(l).get()).collect();
    let total: f64 = raw.iter().sum();
    let fair: Vec<Prob> = if total > 0.0 {
        raw.iter().map(|p| Prob::clamped(p / total)).collect()
    } else {
        vec![Prob::clamped(1.0 / n_outcomes as f64); n_outcomes]
    };

    Ok(Consensus {
        fair,
        n_sources,
        dispersion,
        mean_overround: overround_sum / n_sources as f64,
        max_disagreement,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::odds::{american_to_decimal, decimal_to_implied};

    fn quote(venue: u16, american: &[i32], weight: f64) -> SourceQuote {
        SourceQuote::new(
            VenueId(venue),
            american
                .iter()
                .map(|&a| decimal_to_implied(american_to_decimal(a).unwrap()).unwrap())
                .collect(),
            weight,
        )
    }

    fn sums_to_one(c: &Consensus) {
        let s: f64 = c.fair.iter().map(|p| p.get()).sum();
        assert!((s - 1.0).abs() < 1e-9, "sums to {s}");
    }

    #[test]
    fn agreeing_sources_pool_to_their_shared_view() {
        let sources = vec![
            quote(1, &[-110, -110], 1.0),
            quote(2, &[-110, -110], 1.0),
            quote(3, &[-110, -110], 1.0),
        ];
        let c = consensus(&sources, &ConsensusConfig::default()).unwrap();
        sums_to_one(&c);
        assert!((c.fair[0].get() - 0.5).abs() < 1e-9);
        assert_eq!(c.n_sources, 3);
        assert!(c.max_disagreement < 1e-12);
        assert!(c.dispersion.iter().all(|d| *d < 1e-12));
    }

    #[test]
    fn disagreement_shows_up_as_estimation_uncertainty() {
        let tight = vec![quote(1, &[-110, -110], 1.0), quote(2, &[-112, -108], 1.0)];
        let wide = vec![quote(1, &[-110, -110], 1.0), quote(2, &[-250, 200], 1.0)];
        let a = consensus(&tight, &ConsensusConfig::default()).unwrap();
        let b = consensus(&wide, &ConsensusConfig::default()).unwrap();
        assert!(b.max_disagreement > a.max_disagreement);
        assert!(
            b.estimate_sd() > a.estimate_sd(),
            "wider disagreement must size smaller: {} vs {}",
            b.estimate_sd(),
            a.estimate_sd()
        );
    }

    #[test]
    fn weight_pulls_the_pool_toward_the_sharp_source() {
        let sources = vec![
            quote(1, &[-200, 170], 5.0), // sharp, heavily weighted
            quote(2, &[-110, -110], 1.0),
        ];
        let c = consensus(&sources, &ConsensusConfig::default()).unwrap();
        sums_to_one(&c);
        // The sharp source says ~0.65; equal weighting would land near 0.58.
        assert!(c.fair[0].get() > 0.60, "got {}", c.fair[0]);
    }

    #[test]
    fn trimming_rejects_a_stale_outlier() {
        // Four sources agree, one is badly stale. Without trimming the outlier
        // drags the pool; with it, it should barely register.
        let mut sources: Vec<SourceQuote> = (1..=4).map(|v| quote(v, &[-110, -110], 1.0)).collect();
        sources.push(quote(5, &[-2000, 1000], 1.0));

        let untrimmed =
            consensus(&sources, &ConsensusConfig { trim: 0.0, ..Default::default() }).unwrap();
        let trimmed =
            consensus(&sources, &ConsensusConfig { trim: 0.2, ..Default::default() }).unwrap();

        let err_untrimmed = (untrimmed.fair[0].get() - 0.5).abs();
        let err_trimmed = (trimmed.fair[0].get() - 0.5).abs();
        assert!(
            err_trimmed < err_untrimmed,
            "trimming should reduce outlier influence: {err_trimmed} vs {err_untrimmed}"
        );
    }

    #[test]
    fn log_odds_pooling_beats_the_linear_average_on_longshots() {
        // Two sources say 5% and 15%. The linear average is 10%; log-odds
        // pooling lands below it, which is the correct behaviour when both
        // sources agree the event is unlikely.
        let sources = vec![
            SourceQuote::new(
                VenueId(1),
                vec![Prob::new(0.05).unwrap(), Prob::new(0.95).unwrap()],
                1.0,
            ),
            SourceQuote::new(
                VenueId(2),
                vec![Prob::new(0.15).unwrap(), Prob::new(0.85).unwrap()],
                1.0,
            ),
        ];
        let c = consensus(&sources, &ConsensusConfig { trim: 0.0, ..Default::default() }).unwrap();
        sums_to_one(&c);
        assert!(
            c.fair[0].get() < 0.10,
            "log-odds pool should sit below the linear average, got {}",
            c.fair[0]
        );
        assert!(c.fair[0].get() > 0.05);
    }

    #[test]
    fn extremisation_sharpens_the_pool() {
        let sources = vec![quote(1, &[-200, 170], 1.0), quote(2, &[-190, 165], 1.0)];
        let plain = consensus(
            &sources,
            &ConsensusConfig { trim: 0.0, extremise: 1.0, ..Default::default() },
        )
        .unwrap();
        let sharp = consensus(
            &sources,
            &ConsensusConfig { trim: 0.0, extremise: 1.3, ..Default::default() },
        )
        .unwrap();
        assert!(sharp.fair[0].get() > plain.fair[0].get());
        sums_to_one(&sharp);
    }

    #[test]
    fn a_single_source_is_not_a_consensus() {
        let sources = vec![quote(1, &[-110, -110], 1.0)];
        assert!(consensus(&sources, &ConsensusConfig::default()).is_err());
        // ...unless the caller explicitly accepts one.
        let cfg = ConsensusConfig { min_sources: 1, ..Default::default() };
        assert!(consensus(&sources, &cfg).is_ok());
    }

    #[test]
    fn a_broken_source_is_dropped_not_fatal() {
        let mut sources = vec![quote(1, &[-110, -110], 1.0), quote(2, &[-110, -110], 1.0)];
        // A venue quoting certainty — the devigger rejects it.
        sources.push(SourceQuote::new(VenueId(3), vec![Prob::ONE, Prob::new(0.5).unwrap()], 1.0));
        // ...and one with the wrong number of outcomes.
        sources.push(SourceQuote::new(VenueId(4), vec![Prob::HALF], 1.0));

        let c = consensus(&sources, &ConsensusConfig::default()).unwrap();
        assert_eq!(c.n_sources, 2, "only the two well-formed sources should pool");
        assert!((c.fair[0].get() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn three_way_market_pools_and_normalises() {
        let sources = vec![
            quote(1, &[-150, 260, 550], 1.0),
            quote(2, &[-140, 250, 600], 1.0),
            quote(3, &[-160, 270, 520], 1.0),
        ];
        let c = consensus(&sources, &ConsensusConfig::default()).unwrap();
        sums_to_one(&c);
        assert_eq!(c.fair.len(), 3);
        assert!(c.fair[0].get() > c.fair[1].get());
        assert!(c.fair[1].get() > c.fair[2].get());
        assert!(c.mean_overround > 1.0);
    }

    #[test]
    fn empty_input_is_an_error() {
        assert!(consensus(&[], &ConsensusConfig::default()).is_err());
    }
}
