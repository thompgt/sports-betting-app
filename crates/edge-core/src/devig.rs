//! Removing bookmaker margin from a quoted market ("devigging").
//!
//! A market's quoted prices imply probabilities that sum to more than one. The
//! excess is the operator's margin, and you cannot judge whether a price is
//! good until you have stripped it out. *How* you strip it out is a modelling
//! choice with real consequences, because the margin is not spread evenly: books
//! systematically load more of it onto longshots (the favourite–longshot bias).
//!
//! Four methods are provided, in increasing order of how seriously they take
//! that fact:
//!
//! | Method | Model of the margin | Fails on |
//! |---|---|---|
//! | [`Additive`](DevigMethod::Additive) | equal absolute charge per outcome | longshots, which go negative |
//! | [`Multiplicative`](DevigMethod::Multiplicative) | equal proportional charge | ignores favourite–longshot bias |
//! | [`Power`](DevigMethod::Power) | margin scales with `p^k` | assumes a single global exponent |
//! | [`Shin`](DevigMethod::Shin) | margin as insurance against informed flow | needs a sensible overround |
//!
//! Every solver here is **bracketed**. An unsafeguarded Newton–Raphson — the
//! obvious implementation, and the one the predecessor Python service used —
//! can step outside the domain and diverge on a market with an extreme longshot,
//! which in production means either a panic or, worse, a silently wrong fair
//! price that the strategy layer then trades against. The root here is always
//! trapped in an interval that provably contains it, and a Newton step is only
//! accepted when it stays inside.

use serde::{Deserialize, Serialize};

use crate::error::{EdgeError, Result};
use crate::types::Prob;

/// How to allocate the removed margin across outcomes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum DevigMethod {
    /// `π_i = p_i / Σp`. Cheap, and biased: it leaves too much probability on
    /// longshots, so it will report phantom edges on favourites.
    Multiplicative,
    /// Solve `Σ p_i^k = 1`. The workhorse — removes proportionally more margin
    /// from longshots, which is what books actually do.
    #[default]
    Power,
    /// Shin (1993): treats the margin as the book's protection against informed
    /// traders, solving for the implied insider fraction `z`.
    Shin,
    /// `π_i = p_i - (Σp − 1)/n`. Included for comparison and for the rare
    /// two-way market where the margin genuinely is flat. Clamped at zero.
    Additive,
}

impl DevigMethod {
    pub fn as_str(self) -> &'static str {
        match self {
            DevigMethod::Multiplicative => "multiplicative",
            DevigMethod::Power => "power",
            DevigMethod::Shin => "shin",
            DevigMethod::Additive => "additive",
        }
    }
}

/// A devigged market, with the diagnostics needed to audit the result.
#[derive(Debug, Clone, PartialEq)]
pub struct Devigged {
    /// Fair probabilities, summing to 1 within floating-point tolerance.
    pub fair: Vec<Prob>,
    /// Raw sum of implied probabilities before devigging.
    pub overround: f64,
    /// The solved free parameter: `k` for power, `z` for Shin, unused (0) otherwise.
    /// Worth logging — a `k` far from 1 on a low-margin market signals bad input.
    pub parameter: f64,
    pub method: DevigMethod,
    /// Iterations the solver used. Persisted so a market that is consistently
    /// hard to solve can be found later.
    pub iterations: u32,
}

impl Devigged {
    /// Fair decimal odds for each outcome.
    pub fn fair_decimals(&self) -> Vec<f64> {
        self.fair.iter().map(|p| 1.0 / p.get().max(Prob::EPS)).collect()
    }

    /// Margin removed, as a fraction of stake.
    pub fn margin(&self) -> f64 {
        self.overround - 1.0
    }
}

const MAX_ITER: u32 = 200;
const TOL: f64 = 1e-12;

/// Strip the margin from a set of raw implied probabilities.
///
/// The input is one market's outcomes as quoted by *one* book. To combine
/// several books, devig each independently and then use
/// [`crate::consensus`] — averaging raw prices across books mixes their
/// different margins together and is not the same thing.
pub fn devig(raw: &[Prob], method: DevigMethod) -> Result<Devigged> {
    if raw.is_empty() {
        return Err(EdgeError::Empty("devig requires at least one outcome"));
    }
    let p: Vec<f64> = raw.iter().map(|x| x.get()).collect();
    let sum: f64 = p.iter().sum();

    if !sum.is_finite() || sum <= 0.0 {
        return Err(EdgeError::DegenerateMarket("implied probabilities sum to zero"));
    }

    // A single outcome quoted at certainty carries no information to
    // redistribute; any solver would divide by zero chasing it.
    if p.iter().any(|&x| x >= 1.0 - Prob::EPS) {
        if p.len() == 1 {
            return Ok(Devigged {
                fair: vec![Prob::ONE],
                overround: sum,
                parameter: 1.0,
                method,
                iterations: 0,
            });
        }
        return Err(EdgeError::DegenerateMarket(
            "an outcome is quoted at certainty, leaving no margin to redistribute",
        ));
    }

    // Below one, there is no margin to remove — the book set is internally
    // arbitrageable. Normalising is the only defensible reading, and the caller
    // learns the real story from `overround`.
    if sum < 1.0 {
        return Ok(Devigged {
            fair: normalise(&p),
            overround: sum,
            parameter: 1.0,
            method,
            iterations: 0,
        });
    }

    if (sum - 1.0).abs() < 1e-12 {
        return Ok(Devigged {
            fair: raw.to_vec(),
            overround: sum,
            parameter: 1.0,
            method,
            iterations: 0,
        });
    }

    match method {
        DevigMethod::Multiplicative => Ok(Devigged {
            fair: normalise(&p),
            overround: sum,
            parameter: 1.0,
            method,
            iterations: 0,
        }),
        DevigMethod::Additive => Ok(devig_additive(&p, sum, method)),
        DevigMethod::Power => devig_power(&p, sum, method),
        DevigMethod::Shin => devig_shin(&p, sum, method),
    }
}

fn normalise(p: &[f64]) -> Vec<Prob> {
    let s: f64 = p.iter().sum();
    p.iter().map(|x| Prob::clamped(x / s)).collect()
}

fn devig_additive(p: &[f64], sum: f64, method: DevigMethod) -> Devigged {
    let n = p.len() as f64;
    let per = (sum - 1.0) / n;
    // Subtracting a flat charge can push a longshot below zero, which is exactly
    // the failure mode this method is known for. Clamp and renormalise so the
    // result is at least a valid distribution.
    let shifted: Vec<f64> = p.iter().map(|x| (x - per).max(0.0)).collect();
    Devigged {
        fair: normalise(&shifted),
        overround: sum,
        parameter: per,
        method,
        iterations: 0,
    }
}

/// Solve `Σ p_i^k = 1` for `k`.
///
/// `f(k) = Σ p_i^k − 1` is strictly decreasing in `k` when every `p_i < 1`, so
/// the root is unique and can be trapped. We bracket it first, then run Newton
/// steps that fall back to bisection whenever they would leave the bracket.
fn devig_power(p: &[f64], sum: f64, method: DevigMethod) -> Result<Devigged> {
    let f = |k: f64| -> f64 { p.iter().map(|&x| x.max(Prob::EPS).powf(k)).sum::<f64>() - 1.0 };

    // Overround > 1 means f(1) > 0, and f decreases without bound in k, so the
    // root lies above 1. Expand the upper end until the sign flips.
    let mut lo = 1.0_f64;
    let mut hi = 2.0_f64;
    let f_lo = f(lo);
    let mut f_hi = f(hi);
    let mut expansions = 0;
    while f_hi > 0.0 {
        lo = hi;
        hi *= 2.0;
        f_hi = f(hi);
        expansions += 1;
        if expansions > 60 {
            return Err(EdgeError::Convergence {
                method: "power devig bracketing",
                iterations: expansions,
                residual: f_hi,
            });
        }
    }
    debug_assert!(f_lo >= 0.0 && f_hi <= 0.0);
    let _ = f_lo;

    let mut k = 0.5 * (lo + hi);
    let mut iterations = 0;
    for i in 1..=MAX_ITER {
        iterations = i;
        let fk = f(k);
        if fk.abs() < TOL {
            break;
        }
        // Maintain the bracket using the sign of f at the current iterate.
        if fk > 0.0 { lo = k } else { hi = k }

        // f'(k) = Σ p_i^k · ln(p_i), strictly negative here.
        let dfk: f64 = p
            .iter()
            .map(|&x| {
                let c = x.max(Prob::EPS);
                c.powf(k) * c.ln()
            })
            .sum();

        let next = if dfk.abs() > 1e-300 { k - fk / dfk } else { f64::NAN };
        // Accept the Newton step only when it stays strictly inside the bracket;
        // otherwise bisect. This is what makes the solver total.
        k = if next.is_finite() && next > lo && next < hi {
            next
        } else {
            0.5 * (lo + hi)
        };

        if (hi - lo).abs() < TOL {
            break;
        }
    }

    let fair: Vec<f64> = p.iter().map(|&x| x.max(Prob::EPS).powf(k)).collect();
    Ok(Devigged {
        fair: normalise(&fair),
        overround: sum,
        parameter: k,
        method,
        iterations,
    })
}

/// Shin's model: solve `Σ π_i(z) = 1` where
/// `π_i = (√(z² + 4(1−z)·p_i²/Σp) − z) / (2(1−z))`.
///
/// `z` is the fraction of volume the book believes is informed. It is bounded in
/// `[0, 1)` and the sum is monotone decreasing in it, so plain bisection is both
/// sufficient and unconditionally safe — Newton buys nothing here.
fn devig_shin(p: &[f64], sum: f64, method: DevigMethod) -> Result<Devigged> {
    let pi_sum = |z: f64| -> f64 {
        let denom = 2.0 * (1.0 - z);
        p.iter()
            .map(|&x| {
                let inner = z * z + 4.0 * (1.0 - z) * x * x / sum;
                (inner.max(0.0).sqrt() - z) / denom
            })
            .sum::<f64>()
            - 1.0
    };

    let mut lo = 0.0_f64;
    let mut hi = 1.0 - 1e-9;
    let g_lo = pi_sum(lo);
    let g_hi = pi_sum(hi);

    // No sign change means the model has no admissible insider fraction for
    // this quote set. Rather than return something invented, degrade to the
    // multiplicative result and report z = 0 so the caller can see it happened.
    if !(g_lo.is_finite() && g_hi.is_finite()) || g_lo * g_hi > 0.0 {
        return Ok(Devigged {
            fair: normalise(p),
            overround: sum,
            parameter: 0.0,
            method,
            iterations: 0,
        });
    }

    let mut z = 0.0;
    let mut iterations = 0;
    for i in 1..=MAX_ITER {
        iterations = i;
        z = 0.5 * (lo + hi);
        let g = pi_sum(z);
        if g.abs() < TOL || (hi - lo) < TOL {
            break;
        }
        if g > 0.0 { lo = z } else { hi = z }
    }

    let denom = 2.0 * (1.0 - z);
    let fair: Vec<f64> = p
        .iter()
        .map(|&x| {
            let inner = z * z + 4.0 * (1.0 - z) * x * x / sum;
            ((inner.max(0.0).sqrt() - z) / denom).max(0.0)
        })
        .collect();

    Ok(Devigged {
        fair: normalise(&fair),
        overround: sum,
        parameter: z,
        method,
        iterations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::odds::{american_to_decimal, decimal_to_implied, overround};

    fn imp(american: &[i32]) -> Vec<Prob> {
        american
            .iter()
            .map(|&a| decimal_to_implied(american_to_decimal(a).unwrap()).unwrap())
            .collect()
    }

    fn sums_to_one(d: &Devigged) {
        let s: f64 = d.fair.iter().map(|p| p.get()).sum();
        assert!((s - 1.0).abs() < 1e-9, "fair probabilities sum to {s}");
    }

    #[test]
    fn every_method_returns_a_distribution() {
        let raw = imp(&[-110, -110]);
        for m in [
            DevigMethod::Multiplicative,
            DevigMethod::Power,
            DevigMethod::Shin,
            DevigMethod::Additive,
        ] {
            let d = devig(&raw, m).unwrap();
            sums_to_one(&d);
            assert!(d.fair.iter().all(|p| p.get() > 0.0 && p.get() < 1.0));
        }
    }

    #[test]
    fn symmetric_market_devigs_to_a_coin_flip() {
        let raw = imp(&[-110, -110]);
        for m in [
            DevigMethod::Multiplicative,
            DevigMethod::Power,
            DevigMethod::Shin,
            DevigMethod::Additive,
        ] {
            let d = devig(&raw, m).unwrap();
            assert!((d.fair[0].get() - 0.5).abs() < 1e-9, "{m:?} gave {}", d.fair[0]);
        }
    }

    #[test]
    fn power_removes_more_margin_from_the_longshot() {
        // The whole reason to prefer the power method: on a lopsided market it
        // must shave the longshot harder than a proportional rescale does.
        let raw = imp(&[-400, 300]);
        let mult = devig(&raw, DevigMethod::Multiplicative).unwrap();
        let power = devig(&raw, DevigMethod::Power).unwrap();
        assert!(power.parameter > 1.0, "k = {}", power.parameter);
        // Index 1 is the longshot (+300).
        assert!(
            power.fair[1].get() < mult.fair[1].get(),
            "power {} should sit below multiplicative {}",
            power.fair[1],
            mult.fair[1]
        );
        // ...and correspondingly more on the favourite.
        assert!(power.fair[0].get() > mult.fair[0].get());
    }

    #[test]
    fn power_solves_a_three_way_market() {
        // Soccer 1X2, the case multiplicative devigging handles worst.
        let raw = imp(&[-150, 260, 550]);
        assert!(overround(&raw) > 1.0, "fixture must actually carry margin");
        let d = devig(&raw, DevigMethod::Power).unwrap();
        sums_to_one(&d);
        let residual: f64 = raw.iter().map(|p| p.get().powf(d.parameter)).sum::<f64>() - 1.0;
        assert!(residual.abs() < 1e-9, "solver residual {residual:e}");
    }

    #[test]
    fn power_survives_an_extreme_longshot() {
        // A 200-to-1 longshot beside a heavy favourite. This is the input that
        // makes an unsafeguarded Newton iteration wander off; the bracket holds it.
        let raw = imp(&[-2000, 1200, 20000]);
        assert!(overround(&raw) > 1.0);
        let d = devig(&raw, DevigMethod::Power).unwrap();
        sums_to_one(&d);
        assert!(d.parameter.is_finite());
        assert!(d.iterations < MAX_ITER);
    }

    #[test]
    fn power_handles_a_large_field() {
        // A 20-runner race with a fat overround — the multi-way stress case.
        let raw: Vec<Prob> = (0..20)
            .map(|i| Prob::new(0.09 - 0.003 * i as f64).unwrap())
            .collect();
        let d = devig(&raw, DevigMethod::Power).unwrap();
        sums_to_one(&d);
        assert!(d.overround > 1.0);
    }

    #[test]
    fn additive_never_emits_a_negative_probability() {
        // The flat charge (~0.68 points) exceeds the longshot's own implied
        // probability (~0.66 points), so a naive subtraction goes negative.
        let raw = imp(&[-800, 700, 15000]);
        let d = devig(&raw, DevigMethod::Additive).unwrap();
        assert!(d.parameter > raw[2].get(), "fixture must trigger the clamp");
        sums_to_one(&d);
        assert!(d.fair.iter().all(|p| p.get() >= 0.0));
    }

    #[test]
    fn shin_finds_a_positive_insider_fraction() {
        let raw = imp(&[-400, 300]);
        let d = devig(&raw, DevigMethod::Shin).unwrap();
        sums_to_one(&d);
        assert!(d.parameter > 0.0 && d.parameter < 1.0, "z = {}", d.parameter);
        // Like the power method, Shin should shade the longshot below the
        // proportional answer.
        let mult = devig(&raw, DevigMethod::Multiplicative).unwrap();
        assert!(d.fair[1].get() < mult.fair[1].get());
    }

    #[test]
    fn underround_is_reported_not_hidden() {
        // Prices that sum below one are an arbitrage, not a margin. We normalise
        // but the caller must still be able to see what happened.
        let raw = vec![Prob::new(0.45).unwrap(), Prob::new(0.45).unwrap()];
        let d = devig(&raw, DevigMethod::Power).unwrap();
        sums_to_one(&d);
        assert!(d.overround < 1.0);
        assert!(d.margin() < 0.0);
    }

    #[test]
    fn fair_market_is_left_alone() {
        let raw = vec![Prob::new(0.6).unwrap(), Prob::new(0.4).unwrap()];
        let d = devig(&raw, DevigMethod::Power).unwrap();
        assert!((d.fair[0].get() - 0.6).abs() < 1e-12);
    }

    #[test]
    fn certainty_and_emptiness_are_errors_not_panics() {
        assert!(devig(&[], DevigMethod::Power).is_err());
        let raw = vec![Prob::ONE, Prob::new(0.1).unwrap()];
        assert!(matches!(
            devig(&raw, DevigMethod::Power),
            Err(EdgeError::DegenerateMarket(_))
        ));
    }

    #[test]
    fn power_matches_the_reference_python_implementation() {
        // Parity check against the value the predecessor service produced for
        // this market, so the migration is verifiable rather than asserted.
        let raw = imp(&[-110, -110]);
        let d = devig(&raw, DevigMethod::Power).unwrap();
        assert!((d.overround - 1.047_619_047_619).abs() < 1e-9);
        assert!((d.fair[0].get() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn devigging_is_monotone_in_the_input() {
        // A book that raises one side's price must not lower that side's fair
        // probability. Violations here would let a strategy trade a phantom edge.
        let a = devig(&imp(&[-110, -110]), DevigMethod::Power).unwrap();
        let b = devig(&imp(&[-130, -110]), DevigMethod::Power).unwrap();
        assert!(b.fair[0].get() > a.fair[0].get());
    }
}
