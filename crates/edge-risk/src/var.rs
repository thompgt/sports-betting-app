//! Value at Risk for a portfolio of binary contracts.
//!
//! # Why the textbook approach is wrong here
//!
//! Parametric VaR assumes returns are normal. A binary contract's payoff is
//! Bernoulli: it pays $1 or $0, and nothing in between ever happens. The normal
//! approximation does not merely lose precision on that distribution, it
//! describes a different one — it puts mass where no outcome exists and, far
//! more dangerously, it thins the very tail VaR is supposed to measure. A
//! portfolio of twenty correlated binaries has a realistic path where all twenty
//! resolve against you; the normal approximation prices that path at
//! effectively zero.
//!
//! So the honest measure here is a **Monte Carlo over resolution outcomes**,
//! and the correlation structure is the whole game:
//!
//! - Markets on the **same event** are not merely correlated, they are
//!   *deterministically linked*. Two contracts on the same game share one
//!   resolution. They get one shared latent draw, not two correlated ones.
//! - Markets on **different events** are correlated through a single common
//!   factor. Prediction-market positions are usually concentrated in a theme —
//!   one sport, one election, one week — and a strategy that finds an edge in
//!   one NBA game has usually found the same edge in nine others. Treating them
//!   as independent understates portfolio risk enormously.
//!
//! [`parametric_var`] is kept for comparison and for the fast path, clearly
//! labelled as the approximation it is.

use std::collections::HashMap;

use edge_core::rng::Rng;
use edge_core::stats::{norm_ppf, quantile_sorted};
use edge_core::types::{EventId, MarketId, Price};

use crate::position::Portfolio;

/// A tail-risk estimate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VarResult {
    /// Confidence level, e.g. `0.95`.
    pub confidence: f64,
    /// Loss not expected to be exceeded at `confidence`, as a positive number.
    pub var: f64,
    /// Expected loss *given* that VaR is exceeded — the average of the tail.
    /// Always the more informative figure: VaR says where the cliff is, CVaR
    /// says how far the fall is.
    pub cvar: f64,
    /// Mean profit and loss across paths. Positive means the book is expected
    /// to make money at current marks.
    pub expected_pnl: f64,
    /// Worst single path observed.
    pub worst: f64,
    /// Best single path observed.
    pub best: f64,
    /// Probability of ending down at all.
    pub prob_loss: f64,
    pub paths: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VarConfig {
    pub confidence: f64,
    pub paths: usize,
    /// Correlation between *different* events, in `[0, 1)`. Zero treats them as
    /// independent, which is almost always too generous.
    pub cross_event_correlation: f64,
    pub seed: u64,
}

impl Default for VarConfig {
    fn default() -> Self {
        VarConfig {
            confidence: 0.95,
            paths: 20_000,
            // A default of 0.3 rather than 0: positions found by one strategy on
            // one venue in one week are related, and pretending otherwise is the
            // most common way a portfolio risk number is quietly wrong.
            cross_event_correlation: 0.30,
            seed: 0xC0FF_EE00_1234_5678,
        }
    }
}

/// Simulate resolution across the whole portfolio.
///
/// Marks are read as probabilities: a contract trading at 40c is taken to have
/// a 40% chance of resolving YES. That is the market's own estimate, which is
/// the right prior for a risk calculation even when the strategy disagrees —
/// risk should be measured against the market's view, not against the view that
/// motivated the position.
pub fn monte_carlo_var(
    portfolio: &Portfolio,
    marks: &HashMap<MarketId, Price>,
    cfg: &VarConfig,
) -> VarResult {
    let positions: Vec<_> =
        portfolio.open_positions().filter_map(|p| marks.get(&p.market).map(|m| (*p, *m))).collect();

    if positions.is_empty() || cfg.paths == 0 {
        return VarResult {
            confidence: cfg.confidence,
            var: 0.0,
            cvar: 0.0,
            expected_pnl: 0.0,
            worst: 0.0,
            best: 0.0,
            prob_loss: 0.0,
            paths: 0,
        };
    }

    // Group markets by the event that resolves them. Markets sharing an event
    // share a single latent draw, so they resolve consistently with each other.
    let mut event_index: HashMap<EventId, usize> = HashMap::new();
    let mut market_group: Vec<usize> = Vec::with_capacity(positions.len());
    for (p, _) in &positions {
        let group = match portfolio.event_of(p.market) {
            Some(e) => {
                let next = event_index.len();
                *event_index.entry(e).or_insert(next)
            }
            // A market with no known event is its own group — conservative in
            // that it does not accidentally net against anything else.
            None => event_index.len() + market_group.len() + 1_000_000,
        };
        market_group.push(group);
    }
    let mut group_ids: Vec<usize> = market_group.clone();
    group_ids.sort_unstable();
    group_ids.dedup();
    let group_slot: HashMap<usize, usize> =
        group_ids.iter().enumerate().map(|(slot, id)| (*id, slot)).collect();
    let n_groups = group_ids.len();

    let rho = cfg.cross_event_correlation.clamp(0.0, 0.999);
    let (w_common, w_idio) = (rho.sqrt(), (1.0 - rho).sqrt());

    let mut rng = Rng::new(cfg.seed);
    let mut pnls = Vec::with_capacity(cfg.paths);
    let mut latents = vec![0.0; n_groups];

    for _ in 0..cfg.paths {
        // One market-wide factor, then one latent per event.
        let common = rng.normal();
        for l in latents.iter_mut() {
            *l = w_common * common + w_idio * rng.normal();
        }

        let mut pnl = 0.0;
        for (i, (pos, mark)) in positions.iter().enumerate() {
            let p_yes = mark.dollars().clamp(1e-9, 1.0 - 1e-9);
            // Threshold the shared latent at the market's own implied
            // probability, so the marginal resolution rate matches the price.
            let slot = group_slot[&market_group[i]];
            let yes = latents[slot] < norm_ppf(p_yes);

            let held_wins = if pos.qty.get() > 0 { yes } else { !yes };
            // VaR is a distribution over hypothetical futures, so it works
            // in dollars; the exact integer basis is converted once, here.
            let cost = pos.capital_at_risk().dollars();
            pnl += if held_wins { pos.max_gain().dollars() } else { -cost };
        }
        pnls.push(pnl);
    }

    pnls.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = pnls.len();
    let alpha = 1.0 - cfg.confidence.clamp(0.5, 0.999_9);
    let cutoff = quantile_sorted(&pnls, alpha);
    let tail_len = ((alpha * n as f64).round() as usize).max(1);
    let cvar_mean = pnls[..tail_len].iter().sum::<f64>() / tail_len as f64;
    let losses = pnls.iter().filter(|x| **x < 0.0).count();

    VarResult {
        confidence: cfg.confidence,
        var: (-cutoff).max(0.0),
        cvar: (-cvar_mean).max(0.0),
        expected_pnl: pnls.iter().sum::<f64>() / n as f64,
        worst: pnls[0],
        best: pnls[n - 1],
        prob_loss: losses as f64 / n as f64,
        paths: n,
    }
}

/// Normal-approximation VaR, for comparison and for a cheap intraday estimate.
///
/// Each position contributes a Bernoulli variance of `q² · p(1−p)` where `q` is
/// the contract count, plus a pairwise covariance term from the common factor.
///
/// **This number is unreliable in both directions and must not be used to set a
/// limit.** Where the assumed correlation is below the truth it understates the
/// tail, pricing a realistic all-lose path at nearly zero. Where it is above,
/// the normal tail runs past the hard bound and the function reports a loss
/// larger than the entire capital committed — which is not merely imprecise but
/// impossible, since the market is prepaid. The result is deliberately left
/// unclamped: a parametric VaR exceeding [`Portfolio::capital_at_risk`] is a
/// useful signal that the approximation has broken down, and hiding it behind a
/// clamp would make the number look trustworthy exactly when it is not.
pub fn parametric_var(
    portfolio: &Portfolio,
    marks: &HashMap<MarketId, Price>,
    confidence: f64,
    correlation: f64,
) -> VarResult {
    let positions: Vec<_> = portfolio
        .open_positions()
        .filter_map(|p| marks.get(&p.market).map(|m| (*p, m.dollars().clamp(1e-9, 1.0 - 1e-9))))
        .collect();

    if positions.is_empty() {
        return VarResult {
            confidence,
            var: 0.0,
            cvar: 0.0,
            expected_pnl: 0.0,
            worst: 0.0,
            best: 0.0,
            prob_loss: 0.0,
            paths: 0,
        };
    }

    let mut mean = 0.0;
    let mut sds = Vec::with_capacity(positions.len());
    for (pos, p_yes) in &positions {
        let p_win = if pos.qty.get() > 0 { *p_yes } else { 1.0 - *p_yes };
        let win = pos.max_gain().dollars();
        let lose = -pos.capital_at_risk().dollars();
        mean += p_win * win + (1.0 - p_win) * lose;
        // Bernoulli sd scaled by the payoff range.
        sds.push((win - lose).abs() * (p_win * (1.0 - p_win)).sqrt());
    }

    let rho = correlation.clamp(0.0, 1.0);
    let sum_sq: f64 = sds.iter().map(|s| s * s).sum();
    let sum: f64 = sds.iter().sum();
    // Var = Σσ² + ρ·(Σσ)² − ρ·Σσ², the equicorrelated case.
    let variance = sum_sq + rho * (sum * sum - sum_sq);
    let sd = variance.max(0.0).sqrt();

    let z = norm_ppf(1.0 - confidence.clamp(0.5, 0.999_9));
    let var = -(mean + z * sd);
    // Closed-form normal CVaR: E[X | X < q] = μ − σ·φ(z)/α.
    let alpha = 1.0 - confidence;
    let phi = (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let cvar = -(mean - sd * phi / alpha.max(1e-9));

    VarResult {
        confidence,
        var: var.max(0.0),
        cvar: cvar.max(0.0),
        expected_pnl: mean,
        worst: -portfolio.capital_at_risk().dollars(),
        best: positions.iter().map(|(p, _)| p.max_gain().dollars()).sum(),
        prob_loss: f64::NAN,
        paths: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use edge_core::types::{EventId, MarketId, Notional, Qty, Side};

    fn portfolio_with(n: usize, same_event: bool) -> (Portfolio, HashMap<MarketId, Price>) {
        let mut pf = Portfolio::new(Notional::from_dollars(10_000.0));
        let mut marks = HashMap::new();
        for i in 0..n {
            let m = MarketId(i as u64);
            let e = if same_event { EventId(0) } else { EventId(i as u64) };
            pf.set_event(m, e);
            pf.apply_fill(m, Side::Buy, Price::from_cents(50), Qty(100), Notional::ZERO);
            marks.insert(m, Price::from_cents(50));
        }
        (pf, marks)
    }

    #[test]
    fn an_empty_portfolio_has_no_risk() {
        let pf = Portfolio::new(Notional::from_dollars(1_000.0));
        let r = monte_carlo_var(&pf, &HashMap::new(), &VarConfig::default());
        assert_eq!(r.var, 0.0);
        assert_eq!(r.cvar, 0.0);
        assert_eq!(r.paths, 0);
    }

    #[test]
    fn var_never_exceeds_capital_at_risk() {
        // The hard bound a prepaid market gives us. Any VaR above it is a bug.
        let (pf, marks) = portfolio_with(10, false);
        let r = monte_carlo_var(&pf, &marks, &VarConfig::default());
        assert!(
            r.var <= pf.capital_at_risk().dollars() + 1e-6,
            "VaR {} exceeded the {} that was actually paid",
            r.var,
            pf.capital_at_risk()
        );
        assert!(r.cvar <= pf.capital_at_risk().dollars() + 1e-6);
        assert!(r.worst >= -pf.capital_at_risk().dollars() - 1e-6);
    }

    #[test]
    fn cvar_is_always_at_least_var() {
        let (pf, marks) = portfolio_with(8, false);
        let r = monte_carlo_var(&pf, &marks, &VarConfig::default());
        assert!(r.cvar >= r.var - 1e-9, "CVaR {} < VaR {}", r.cvar, r.var);
    }

    #[test]
    fn a_fairly_priced_book_has_roughly_zero_expected_pnl() {
        // Ten positions bought at exactly the mark: no edge, so no drift.
        let (pf, marks) = portfolio_with(10, false);
        let r = monte_carlo_var(&pf, &marks, &VarConfig::default());
        assert!(r.expected_pnl.abs() < 5.0, "expected pnl {}", r.expected_pnl);
    }

    #[test]
    fn markets_on_one_event_resolve_together() {
        // Ten positions on one event are one bet: every path is all-win or
        // all-lose, so VaR is the full capital at risk.
        let (pf, marks) = portfolio_with(10, true);
        let r = monte_carlo_var(&pf, &marks, &VarConfig::default());
        assert!(
            (r.var - pf.capital_at_risk().dollars()).abs() < 1.0,
            "linked markets must risk everything together: VaR {} vs {}",
            r.var,
            pf.capital_at_risk()
        );
        // And only two distinct outcomes are possible.
        assert!((r.worst + 500.0).abs() < 1e-6, "worst {}", r.worst);
        assert!((r.best - 500.0).abs() < 1e-6, "best {}", r.best);
    }

    #[test]
    fn ten_independent_events_diversify() {
        // The same capital spread across ten unrelated events must risk far
        // less than ten positions on one event.
        let (linked, m1) = portfolio_with(10, true);
        let (spread, m2) = portfolio_with(10, false);
        let cfg = VarConfig { cross_event_correlation: 0.0, ..Default::default() };
        let a = monte_carlo_var(&linked, &m1, &cfg);
        let b = monte_carlo_var(&spread, &m2, &cfg);
        assert!(b.var < a.var * 0.75, "diversification did nothing: {} vs {}", b.var, a.var);
    }

    #[test]
    fn correlation_between_events_raises_var() {
        let (pf, marks) = portfolio_with(20, false);
        let low = monte_carlo_var(
            &pf,
            &marks,
            &VarConfig { cross_event_correlation: 0.0, ..Default::default() },
        );
        let high = monte_carlo_var(
            &pf,
            &marks,
            &VarConfig { cross_event_correlation: 0.8, ..Default::default() },
        );
        assert!(
            high.var > low.var * 1.2,
            "correlation must increase tail risk: {} vs {}",
            high.var,
            low.var
        );
    }

    #[test]
    fn the_normal_approximation_can_report_an_impossible_loss() {
        // The reason the Monte Carlo exists, stated as sharply as it can be.
        // A prepaid portfolio cannot lose more than it paid. The Monte Carlo
        // respects that bound because it simulates the actual payoff; the
        // normal approximation does not know the bound exists and will happily
        // report a 99% VaR larger than the total capital committed.
        let (pf, marks) = portfolio_with(20, true);
        let bound = pf.capital_at_risk().dollars();
        let mc = monte_carlo_var(&pf, &marks, &VarConfig::default());
        let par = parametric_var(&pf, &marks, 0.99, 0.30);

        assert!(mc.var <= bound + 1e-6, "Monte Carlo VaR {} exceeded {bound}", mc.var);
        assert!(
            par.var > bound,
            "the parametric model should overshoot the hard bound here: {} vs {bound}",
            par.var
        );
    }

    #[test]
    fn results_are_reproducible() {
        let (pf, marks) = portfolio_with(6, false);
        let cfg = VarConfig::default();
        let a = monte_carlo_var(&pf, &marks, &cfg);
        let b = monte_carlo_var(&pf, &marks, &cfg);
        assert_eq!(a, b, "a risk number that moves between runs is unusable");

        // A different seed draws a different sample. VaR itself is a quantile
        // of a discrete payoff distribution and often lands on the same value,
        // so the mean is the thing to compare.
        let c = monte_carlo_var(&pf, &marks, &VarConfig { seed: 99, ..cfg });
        assert_ne!(a.expected_pnl, c.expected_pnl);
        assert!((a.expected_pnl - c.expected_pnl).abs() < 5.0, "but they should agree closely");
    }

    #[test]
    fn a_higher_confidence_demands_a_larger_var() {
        let (pf, marks) = portfolio_with(15, false);
        let c95 =
            monte_carlo_var(&pf, &marks, &VarConfig { confidence: 0.95, ..Default::default() });
        let c99 =
            monte_carlo_var(&pf, &marks, &VarConfig { confidence: 0.99, ..Default::default() });
        assert!(c99.var >= c95.var);
    }

    #[test]
    fn a_locked_arbitrage_carries_no_risk_at_all() {
        // YES for 48c on one venue and NO for 48c on another: 96c committed for
        // a payout of exactly $1 whichever way the event resolves. Every
        // simulated path must profit, and VaR must be zero — not small, zero.
        let mut pf = Portfolio::new(Notional::from_dollars(1_000.0));
        pf.set_event(MarketId(0), EventId(0));
        pf.set_event(MarketId(1), EventId(0));
        pf.apply_fill(MarketId(0), Side::Buy, Price::from_cents(48), Qty(100), Notional::ZERO);
        // Selling YES at 52c is buying the NO leg at 48c.
        pf.apply_fill(MarketId(1), Side::Sell, Price::from_cents(52), Qty(100), Notional::ZERO);
        let marks = HashMap::from([
            (MarketId(0), Price::from_cents(48)),
            (MarketId(1), Price::from_cents(48)),
        ]);

        let r = monte_carlo_var(&pf, &marks, &VarConfig::default());
        assert_eq!(r.var, 0.0, "a locked arbitrage has no downside");
        assert_eq!(r.cvar, 0.0);
        assert_eq!(r.prob_loss, 0.0);
        assert!(r.worst > 0.0, "even the worst path profits, got {}", r.worst);
        assert!((r.expected_pnl - 4.0).abs() < 1e-6, "4c per pair on 100 pairs");
    }

    #[test]
    fn markets_on_one_event_quoted_inconsistently_carry_residual_risk() {
        // The same two legs, but the venues disagree by 4c on where the event
        // stands. The shared latent still links them, and the disagreement
        // surfaces as a band of paths where both legs lose. That is not a
        // modelling artefact to smooth away — it is the risk of trusting two
        // marks that cannot both be right.
        let mut pf = Portfolio::new(Notional::from_dollars(1_000.0));
        pf.set_event(MarketId(0), EventId(0));
        pf.set_event(MarketId(1), EventId(0));
        pf.apply_fill(MarketId(0), Side::Buy, Price::from_cents(48), Qty(100), Notional::ZERO);
        pf.apply_fill(MarketId(1), Side::Sell, Price::from_cents(52), Qty(100), Notional::ZERO);
        let marks = HashMap::from([
            (MarketId(0), Price::from_cents(48)),
            (MarketId(1), Price::from_cents(52)),
        ]);

        // The losing band is about 4% of paths, so it is invisible at 95%
        // confidence and only shows up at 99% — which is itself the argument
        // for reading CVaR rather than VaR alone.
        let at_95 = monte_carlo_var(&pf, &marks, &VarConfig::default());
        assert!(at_95.prob_loss > 0.0 && at_95.prob_loss < 0.10, "prob_loss {}", at_95.prob_loss);
        assert_eq!(at_95.var, 0.0, "the 5% quantile still profits");
        assert!(at_95.cvar > 0.0, "but the tail beyond it does not");

        let at_99 =
            monte_carlo_var(&pf, &marks, &VarConfig { confidence: 0.99, ..Default::default() });
        assert!(at_99.var > 90.0, "the losing band is a 96c-per-pair loss: {}", at_99.var);
        // The position is still fair on average.
        assert!(at_99.expected_pnl.abs() < 2.0, "expected pnl {}", at_99.expected_pnl);
    }

    #[test]
    fn unmarked_positions_are_excluded_rather_than_guessed() {
        let (pf, _) = portfolio_with(5, false);
        let r = monte_carlo_var(&pf, &HashMap::new(), &VarConfig::default());
        assert_eq!(r.paths, 0);
    }

    #[test]
    fn the_parametric_path_is_finite_and_signed_correctly() {
        let (pf, marks) = portfolio_with(10, false);
        let r = parametric_var(&pf, &marks, 0.95, 0.3);
        assert!(r.var.is_finite() && r.var >= 0.0);
        assert!(r.cvar >= r.var);
        assert!(r.expected_pnl.abs() < 5.0);
    }
}
