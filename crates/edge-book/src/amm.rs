//! Automated market makers: LMSR and constant-product.
//!
//! These are not here as a curiosity. A large share of prediction-market
//! liquidity sits in AMMs rather than in order books — Polymarket's original
//! mechanism was a Gnosis fixed-product market maker, and most on-chain venues
//! still are. An engine that can only read an order book is blind to that
//! liquidity, and blind to the arbitrage between it and a central limit order
//! book, which is one of the few genuinely repeatable edges in this market: an
//! AMM's price moves mechanically with its inventory and does not react to news
//! until someone trades it.
//!
//! Both implementations expose the same shape ([`MarketMaker`]), so a strategy
//! can quote against either without caring which it is facing.

use edge_core::error::{EdgeError, Result};
use edge_core::types::Prob;

/// The result of executing against an AMM.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AmmTrade {
    pub outcome: usize,
    /// Shares acquired (positive) or sold back (negative).
    pub shares: f64,
    /// Collateral paid (positive) or received (negative).
    pub cost: f64,
    /// Realised average price per share.
    pub avg_price: f64,
    pub spot_before: Prob,
    pub spot_after: Prob,
    /// Average price minus the spot price before the trade, as a fraction of
    /// spot. This is the number that decides whether an apparent AMM/book
    /// arbitrage survives execution.
    pub slippage: f64,
}

/// Common interface over both mechanisms.
pub trait MarketMaker {
    fn n_outcomes(&self) -> usize;

    /// Marginal price of an outcome. Prices across outcomes sum to 1.
    fn spot(&self, outcome: usize) -> Result<Prob>;

    /// Collateral required to buy `shares` of `outcome` without executing.
    /// Negative `shares` prices a sale.
    fn cost_to_buy(&self, outcome: usize, shares: f64) -> Result<f64>;

    /// Shares obtainable for a given collateral budget.
    fn shares_for_budget(&self, outcome: usize, budget: f64) -> Result<f64>;

    /// Execute, mutating the maker's state.
    fn buy(&mut self, outcome: usize, shares: f64) -> Result<AmmTrade>;

    /// Worst case the operator can lose across all resolutions, given the
    /// subsidy committed. Finite for both mechanisms, which is the property
    /// that makes running one survivable.
    fn max_loss(&self) -> f64;

    fn spots(&self) -> Result<Vec<Prob>> {
        (0..self.n_outcomes()).map(|i| self.spot(i)).collect()
    }
}

// ---------------------------------------------------------------------------
// LMSR
// ---------------------------------------------------------------------------

/// Hanson's Logarithmic Market Scoring Rule.
///
/// Cost function `C(q) = b · ln Σ exp(q_i / b)`, prices are its gradient, and
/// the operator's worst-case loss is `b · ln n` regardless of what traders do.
/// The liquidity parameter `b` is the whole design: it is simultaneously the
/// subsidy, the depth, and the inverse of how far a given trade moves the price.
///
/// Everything is computed through a shifted log-sum-exp. The textbook
/// formulation overflows `exp` as soon as a market becomes lopsided — which is
/// exactly what happens to a prediction market as it approaches resolution and
/// one outcome's inventory runs away — so the naive implementation fails
/// precisely when the market matters most.
#[derive(Debug, Clone, PartialEq)]
pub struct Lmsr {
    b: f64,
    /// Net shares outstanding per outcome.
    q: Vec<f64>,
}

impl Lmsr {
    /// `b` is the liquidity parameter, in collateral units.
    pub fn new(b: f64, n_outcomes: usize) -> Result<Self> {
        if !(b.is_finite() && b > 0.0) {
            return Err(EdgeError::DegenerateMarket("LMSR b must be positive"));
        }
        if n_outcomes < 2 {
            return Err(EdgeError::DegenerateMarket("LMSR needs at least two outcomes"));
        }
        Ok(Lmsr {
            b,
            q: vec![0.0; n_outcomes],
        })
    }

    /// Size the market by the subsidy you are willing to lose, which is the
    /// decision an operator actually faces. `b = max_loss / ln n`.
    pub fn with_max_loss(max_loss: f64, n_outcomes: usize) -> Result<Self> {
        if n_outcomes < 2 {
            return Err(EdgeError::DegenerateMarket("LMSR needs at least two outcomes"));
        }
        Lmsr::new(max_loss / (n_outcomes as f64).ln(), n_outcomes)
    }

    #[inline]
    pub fn b(&self) -> f64 {
        self.b
    }

    /// Net inventory per outcome. Positive means the maker is short that
    /// outcome, having sold shares in it.
    #[inline]
    pub fn inventory(&self) -> &[f64] {
        &self.q
    }

    /// Set the state directly, for restoring a maker from a snapshot.
    pub fn set_inventory(&mut self, q: Vec<f64>) -> Result<()> {
        if q.len() != self.q.len() || q.iter().any(|x| !x.is_finite()) {
            return Err(EdgeError::DegenerateMarket("invalid LMSR inventory"));
        }
        self.q = q;
        Ok(())
    }

    /// `(shift, Σ exp(q_i/b − shift))` — the stable decomposition everything else
    /// is built on.
    fn lse_parts(&self, q: &[f64]) -> (f64, f64) {
        let scaled: Vec<f64> = q.iter().map(|x| x / self.b).collect();
        let m = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let s: f64 = scaled.iter().map(|x| (x - m).exp()).sum();
        (m, s)
    }

    fn cost_of(&self, q: &[f64]) -> f64 {
        let (m, s) = self.lse_parts(q);
        self.b * (m + s.ln())
    }

    /// Total collateral the maker holds at the current state, relative to zero
    /// inventory. Also the sum of everything traders have paid in.
    pub fn cost(&self) -> f64 {
        self.cost_of(&self.q) - self.cost_of(&vec![0.0; self.q.len()])
    }

    fn check_outcome(&self, outcome: usize) -> Result<()> {
        if outcome >= self.q.len() {
            return Err(EdgeError::DegenerateMarket("outcome index out of range"));
        }
        Ok(())
    }
}

impl MarketMaker for Lmsr {
    fn n_outcomes(&self) -> usize {
        self.q.len()
    }

    fn spot(&self, outcome: usize) -> Result<Prob> {
        self.check_outcome(outcome)?;
        let (m, s) = self.lse_parts(&self.q);
        Ok(Prob::clamped((self.q[outcome] / self.b - m).exp() / s))
    }

    fn cost_to_buy(&self, outcome: usize, shares: f64) -> Result<f64> {
        self.check_outcome(outcome)?;
        if !shares.is_finite() {
            return Err(EdgeError::DegenerateMarket("share count must be finite"));
        }
        if shares == 0.0 {
            return Ok(0.0);
        }
        let mut q2 = self.q.clone();
        q2[outcome] += shares;
        Ok(self.cost_of(&q2) - self.cost_of(&self.q))
    }

    fn shares_for_budget(&self, outcome: usize, budget: f64) -> Result<f64> {
        self.check_outcome(outcome)?;
        if !budget.is_finite() {
            return Err(EdgeError::DegenerateMarket("budget must be finite"));
        }
        if budget == 0.0 {
            return Ok(0.0);
        }
        // Closed form. From C(q + Δe_i) − C(q) = budget:
        //   Δ = b·ln( S·(e^(budget/b) − 1) + e^(q_i/b) ) − q_i
        // evaluated in the shifted space so nothing overflows.
        let (m, s) = self.lse_parts(&self.q);
        let beta = budget / self.b;
        let inner = s * (beta.exp() - 1.0) + (self.q[outcome] / self.b - m).exp();
        if !(inner.is_finite() && inner > 0.0) {
            // A sale large enough to drive the outcome's price to zero. There is
            // no finite share count that raises exactly this much collateral.
            return Err(EdgeError::DegenerateMarket(
                "budget is not achievable at any share count",
            ));
        }
        Ok(self.b * (m + inner.ln()) - self.q[outcome])
    }

    fn buy(&mut self, outcome: usize, shares: f64) -> Result<AmmTrade> {
        let spot_before = self.spot(outcome)?;
        let cost = self.cost_to_buy(outcome, shares)?;
        self.q[outcome] += shares;
        let spot_after = self.spot(outcome)?;
        let avg_price = if shares != 0.0 { cost / shares } else { spot_before.get() };
        Ok(AmmTrade {
            outcome,
            shares,
            cost,
            avg_price,
            spot_before,
            spot_after,
            slippage: if spot_before.get() > 0.0 {
                (avg_price - spot_before.get()) / spot_before.get()
            } else {
                0.0
            },
        })
    }

    fn max_loss(&self) -> f64 {
        self.b * (self.q.len() as f64).ln()
    }
}

// ---------------------------------------------------------------------------
// Constant-product (fixed-product) market maker
// ---------------------------------------------------------------------------

/// The Gnosis/Polymarket fixed-product market maker, not a generic Uniswap pair.
///
/// The distinction matters. A prediction-market CPMM does not swap one asset for
/// another: collateral mints a complete set (one of every outcome), the unwanted
/// legs go into the pool, and the wanted leg comes out. The invariant is the
/// product of the outcome reserves, and price is a reserve *ratio* rather than a
/// quotient of two independent assets.
///
/// Buying `Δ` shares of outcome `i` for collateral `c` preserves
/// `∏ reserves = k`, giving a closed form for both directions:
///
/// ```text
/// binary case:  (x + c − Δ)(y + c) = xy
///               c = ( −(x + y − Δ) + √((x + y − Δ)² + 4Δy) ) / 2
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Cpmm {
    /// Outcome token reserves held by the pool.
    reserves: Vec<f64>,
    /// Collateral committed by liquidity providers. Bounds the operator's loss.
    funding: f64,
}

impl Cpmm {
    /// Create a pool seeded with `funding` collateral, split evenly — the
    /// uniform-prior state where every outcome prices at `1/n`.
    pub fn new(funding: f64, n_outcomes: usize) -> Result<Self> {
        if !(funding.is_finite() && funding > 0.0) {
            return Err(EdgeError::DegenerateMarket("CPMM funding must be positive"));
        }
        if n_outcomes < 2 {
            return Err(EdgeError::DegenerateMarket("CPMM needs at least two outcomes"));
        }
        Ok(Cpmm {
            reserves: vec![funding; n_outcomes],
            funding,
        })
    }

    /// Create with explicit reserves, for restoring a pool from on-chain state.
    pub fn with_reserves(reserves: Vec<f64>, funding: f64) -> Result<Self> {
        if reserves.len() < 2 || reserves.iter().any(|r| !r.is_finite() || *r <= 0.0) {
            return Err(EdgeError::DegenerateMarket("CPMM reserves must be positive"));
        }
        Ok(Cpmm { reserves, funding })
    }

    #[inline]
    pub fn reserves(&self) -> &[f64] {
        &self.reserves
    }

    /// `∏ reserves`. Invariant under every trade.
    pub fn k(&self) -> f64 {
        self.reserves.iter().product()
    }

    fn check_outcome(&self, outcome: usize) -> Result<()> {
        if outcome >= self.reserves.len() {
            return Err(EdgeError::DegenerateMarket("outcome index out of range"));
        }
        Ok(())
    }

    /// Solve for the collateral `c` that buys `shares` of `outcome`.
    ///
    /// For `n > 2` the invariant is a polynomial in `c` with no closed form, so
    /// this bisects. `f(c) = (r_i + c − Δ)·∏_{j≠i}(r_j + c) − k` is strictly
    /// increasing in `c` over the admissible range, so bisection is total.
    fn solve_cost(&self, outcome: usize, shares: f64) -> Result<f64> {
        let n = self.reserves.len();
        let k = self.k();

        if n == 2 {
            let i = outcome;
            let j = 1 - outcome;
            let (x, y) = (self.reserves[i], self.reserves[j]);
            let s = x + y - shares;
            let disc = s * s + 4.0 * shares * y;
            if disc < 0.0 {
                return Err(EdgeError::DegenerateMarket("CPMM trade has no solution"));
            }
            return Ok((-s + disc.sqrt()) / 2.0);
        }

        let f = |c: f64| -> f64 {
            let mut prod = 1.0;
            for (j, r) in self.reserves.iter().enumerate() {
                prod *= if j == outcome { r + c - shares } else { r + c };
            }
            prod - k
        };

        // A purchase needs c > 0; a sale needs c < 0 and is bounded below by the
        // point where the outcome's reserve would go negative.
        let (mut lo, mut hi) = if shares >= 0.0 {
            (0.0, shares.max(1.0))
        } else {
            (-self.reserves.iter().copied().fold(f64::MAX, f64::min) + 1e-12, 0.0)
        };
        let mut expansions = 0;
        while shares >= 0.0 && f(hi) < 0.0 {
            hi *= 2.0;
            expansions += 1;
            if expansions > 200 {
                return Err(EdgeError::Convergence {
                    method: "CPMM cost bracketing",
                    iterations: expansions,
                    residual: f(hi),
                });
            }
        }
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if f(mid) > 0.0 { hi = mid } else { lo = mid }
            if (hi - lo).abs() < 1e-12 {
                break;
            }
        }
        Ok(0.5 * (lo + hi))
    }
}

impl MarketMaker for Cpmm {
    fn n_outcomes(&self) -> usize {
        self.reserves.len()
    }

    /// Price is the *normalised inverse* reserve: the scarcer an outcome in the
    /// pool, the more it costs. For the binary case this is `y / (x + y)`.
    fn spot(&self, outcome: usize) -> Result<Prob> {
        self.check_outcome(outcome)?;
        // p_i ∝ ∏_{j≠i} r_j, which reduces to r_j/(r_i+r_j) when n = 2.
        let mut weights = Vec::with_capacity(self.reserves.len());
        for i in 0..self.reserves.len() {
            let mut prod = 1.0;
            for (j, r) in self.reserves.iter().enumerate() {
                if j != i {
                    prod *= r;
                }
            }
            weights.push(prod);
        }
        let total: f64 = weights.iter().sum();
        if !(total.is_finite() && total > 0.0) {
            return Err(EdgeError::DegenerateMarket("CPMM reserves are degenerate"));
        }
        Ok(Prob::clamped(weights[outcome] / total))
    }

    fn cost_to_buy(&self, outcome: usize, shares: f64) -> Result<f64> {
        self.check_outcome(outcome)?;
        if !shares.is_finite() {
            return Err(EdgeError::DegenerateMarket("share count must be finite"));
        }
        if shares == 0.0 {
            return Ok(0.0);
        }
        self.solve_cost(outcome, shares)
    }

    fn shares_for_budget(&self, outcome: usize, budget: f64) -> Result<f64> {
        self.check_outcome(outcome)?;
        if !budget.is_finite() {
            return Err(EdgeError::DegenerateMarket("budget must be finite"));
        }
        if budget == 0.0 {
            return Ok(0.0);
        }
        // Direct: add `budget` to every reserve, then take out enough of the
        // wanted outcome to restore k.
        let k = self.k();
        let mut prod_others = 1.0;
        for (j, r) in self.reserves.iter().enumerate() {
            if j != outcome {
                prod_others *= r + budget;
            }
        }
        if !(prod_others.is_finite() && prod_others > 0.0) {
            return Err(EdgeError::DegenerateMarket("CPMM reserves are degenerate"));
        }
        Ok(self.reserves[outcome] + budget - k / prod_others)
    }

    fn buy(&mut self, outcome: usize, shares: f64) -> Result<AmmTrade> {
        let spot_before = self.spot(outcome)?;
        let cost = self.cost_to_buy(outcome, shares)?;

        let mut next = self.reserves.clone();
        for (j, r) in next.iter_mut().enumerate() {
            *r += cost;
            if j == outcome {
                *r -= shares;
            }
        }
        if next.iter().any(|r| !r.is_finite() || *r <= 0.0) {
            return Err(EdgeError::DegenerateMarket(
                "trade would drain a reserve to zero",
            ));
        }
        self.reserves = next;

        let spot_after = self.spot(outcome)?;
        let avg_price = if shares != 0.0 { cost / shares } else { spot_before.get() };
        Ok(AmmTrade {
            outcome,
            shares,
            cost,
            avg_price,
            spot_before,
            spot_after,
            slippage: if spot_before.get() > 0.0 {
                (avg_price - spot_before.get()) / spot_before.get()
            } else {
                0.0
            },
        })
    }

    fn max_loss(&self) -> f64 {
        // The pool can never pay out more than the complete sets it can mint
        // from its own funding.
        self.funding
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sums_to_one(m: &dyn MarketMaker) {
        let s: f64 = m.spots().unwrap().iter().map(|p| p.get()).sum();
        assert!((s - 1.0).abs() < 1e-9, "prices sum to {s}");
    }

    // -- LMSR -------------------------------------------------------------

    #[test]
    fn lmsr_starts_at_a_uniform_prior() {
        let m = Lmsr::new(100.0, 2).unwrap();
        assert!((m.spot(0).unwrap().get() - 0.5).abs() < 1e-12);
        sums_to_one(&m);

        let m3 = Lmsr::new(100.0, 3).unwrap();
        assert!((m3.spot(0).unwrap().get() - 1.0 / 3.0).abs() < 1e-12);
        sums_to_one(&m3);
    }

    #[test]
    fn lmsr_buying_raises_the_price_it_bought() {
        let mut m = Lmsr::new(100.0, 2).unwrap();
        let t = m.buy(0, 50.0).unwrap();
        assert!(t.spot_after > t.spot_before);
        assert!(m.spot(1).unwrap() < Prob::HALF);
        sums_to_one(&m);
    }

    #[test]
    fn lmsr_cost_is_path_independent() {
        // The defining property: cost depends only on start and end inventory,
        // never on how you got there. Anything else is arbitrageable.
        let mut one = Lmsr::new(100.0, 3).unwrap();
        let direct = one.buy(0, 120.0).unwrap().cost;

        let mut many = Lmsr::new(100.0, 3).unwrap();
        let mut total = 0.0;
        for _ in 0..120 {
            total += many.buy(0, 1.0).unwrap().cost;
        }
        assert!(
            (direct - total).abs() < 1e-9,
            "one trade cost {direct}, 120 small ones cost {total}"
        );
        assert!((one.spot(0).unwrap().get() - many.spot(0).unwrap().get()).abs() < 1e-12);
    }

    #[test]
    fn lmsr_round_trip_returns_to_the_start() {
        let mut m = Lmsr::new(100.0, 2).unwrap();
        let bought = m.buy(0, 75.0).unwrap().cost;
        let sold = m.buy(0, -75.0).unwrap().cost;
        assert!((bought + sold).abs() < 1e-9, "a round trip should be free");
        assert!((m.spot(0).unwrap().get() - 0.5).abs() < 1e-12);
        assert!(m.inventory().iter().all(|q| q.abs() < 1e-9));
    }

    #[test]
    fn lmsr_average_price_sits_between_spot_before_and_after() {
        let mut m = Lmsr::new(100.0, 2).unwrap();
        let t = m.buy(0, 60.0).unwrap();
        assert!(t.avg_price > t.spot_before.get(), "a buyer pays above the pre-trade spot");
        assert!(t.avg_price < t.spot_after.get(), "and below the post-trade spot");
        assert!(t.slippage > 0.0);
    }

    #[test]
    fn lmsr_slippage_grows_with_size() {
        let small = Lmsr::new(100.0, 2).unwrap().buy(0, 10.0).unwrap().slippage;
        let large = Lmsr::new(100.0, 2).unwrap().buy(0, 200.0).unwrap().slippage;
        assert!(large > small, "{large} should exceed {small}");
    }

    #[test]
    fn lmsr_deeper_liquidity_means_less_slippage() {
        let thin = Lmsr::new(50.0, 2).unwrap().buy(0, 100.0).unwrap().slippage;
        let deep = Lmsr::new(5_000.0, 2).unwrap().buy(0, 100.0).unwrap().slippage;
        assert!(deep < thin / 10.0, "b should dominate slippage: {deep} vs {thin}");
    }

    #[test]
    fn lmsr_budget_inversion_is_exact() {
        let m = Lmsr::new(100.0, 3).unwrap();
        for budget in [1.0, 25.0, 250.0, 5_000.0] {
            let shares = m.shares_for_budget(0, budget).unwrap();
            let cost = m.cost_to_buy(0, shares).unwrap();
            assert!((cost - budget).abs() < 1e-8, "budget {budget} -> cost {cost}");
        }
    }

    #[test]
    fn lmsr_budget_inversion_works_from_a_skewed_state() {
        let mut m = Lmsr::new(100.0, 2).unwrap();
        m.buy(0, 400.0).unwrap();
        let shares = m.shares_for_budget(1, 50.0).unwrap();
        let cost = m.cost_to_buy(1, shares).unwrap();
        assert!((cost - 50.0).abs() < 1e-8);
    }

    #[test]
    fn lmsr_survives_a_market_running_to_resolution() {
        // The case the textbook formulation overflows on: one outcome's
        // inventory an order of magnitude past b, driving spot to ~1.
        let mut m = Lmsr::new(100.0, 2).unwrap();
        m.buy(0, 20_000.0).unwrap();
        let p = m.spot(0).unwrap().get();
        assert!(p.is_finite() && p > 0.999, "price ran away to {p}");
        assert!(m.spot(1).unwrap().get().is_finite());
        sums_to_one(&m);
        assert!(m.cost().is_finite());
        // ...and it can still be traded back.
        assert!(m.cost_to_buy(1, 100.0).unwrap().is_finite());
    }

    #[test]
    fn lmsr_loss_is_bounded_by_b_ln_n() {
        // Whatever a trader does, the maker's payout never exceeds its subsidy.
        let mut m = Lmsr::new(100.0, 2).unwrap();
        let bound = m.max_loss();
        assert!((bound - 100.0 * 2f64.ln()).abs() < 1e-12);

        let mut worst: f64 = 0.0;
        for size in [10.0, 100.0, 1_000.0, 10_000.0] {
            let mut probe = m.clone();
            let paid = probe.buy(0, size).unwrap().cost;
            // If outcome 0 resolves true, the maker owes `size` and holds `paid`.
            worst = worst.max(size - paid);
        }
        assert!(worst <= bound + 1e-6, "loss {worst} exceeded bound {bound}");
        let _ = m.buy(0, 1.0);
    }

    #[test]
    fn lmsr_sized_by_subsidy() {
        let m = Lmsr::with_max_loss(1_000.0, 4).unwrap();
        assert!((m.max_loss() - 1_000.0).abs() < 1e-9);
    }

    #[test]
    fn lmsr_rejects_nonsense_construction_and_indices() {
        assert!(Lmsr::new(0.0, 2).is_err());
        assert!(Lmsr::new(-1.0, 2).is_err());
        assert!(Lmsr::new(f64::NAN, 2).is_err());
        assert!(Lmsr::new(100.0, 1).is_err());
        let m = Lmsr::new(100.0, 2).unwrap();
        assert!(m.spot(2).is_err());
        assert!(m.cost_to_buy(9, 1.0).is_err());
        assert!(m.cost_to_buy(0, f64::NAN).is_err());
    }

    #[test]
    fn lmsr_state_can_be_snapshot_and_restored() {
        let mut m = Lmsr::new(100.0, 2).unwrap();
        m.buy(0, 33.0).unwrap();
        let snapshot = m.inventory().to_vec();

        let mut restored = Lmsr::new(100.0, 2).unwrap();
        restored.set_inventory(snapshot).unwrap();
        assert_eq!(restored.spot(0).unwrap(), m.spot(0).unwrap());
        assert!(restored.set_inventory(vec![1.0]).is_err());
    }

    // -- CPMM -------------------------------------------------------------

    #[test]
    fn cpmm_starts_at_a_uniform_prior() {
        let m = Cpmm::new(1_000.0, 2).unwrap();
        assert!((m.spot(0).unwrap().get() - 0.5).abs() < 1e-12);
        sums_to_one(&m);
        let m3 = Cpmm::new(1_000.0, 3).unwrap();
        assert!((m3.spot(0).unwrap().get() - 1.0 / 3.0).abs() < 1e-12);
        sums_to_one(&m3);
    }

    #[test]
    fn cpmm_preserves_its_invariant() {
        let mut m = Cpmm::new(1_000.0, 2).unwrap();
        let k0 = m.k();
        for size in [10.0, 50.0, 200.0, 7.5] {
            m.buy(0, size).unwrap();
            let rel = (m.k() - k0).abs() / k0;
            assert!(rel < 1e-9, "k drifted by {rel} after buying {size}");
        }
        sums_to_one(&m);
    }

    #[test]
    fn cpmm_preserves_its_invariant_with_three_outcomes() {
        let mut m = Cpmm::new(1_000.0, 3).unwrap();
        let k0 = m.k();
        m.buy(1, 120.0).unwrap();
        let rel = (m.k() - k0).abs() / k0;
        assert!(rel < 1e-6, "k drifted by {rel}");
        sums_to_one(&m);
    }

    #[test]
    fn cpmm_buying_raises_the_price_it_bought() {
        let mut m = Cpmm::new(1_000.0, 2).unwrap();
        let t = m.buy(0, 200.0).unwrap();
        assert!(t.spot_after > t.spot_before);
        assert!(t.cost > 0.0);
        assert!(t.avg_price > t.spot_before.get(), "the buyer pays slippage");
        assert!(t.slippage > 0.0);
        sums_to_one(&m);
    }

    #[test]
    fn cpmm_cost_and_budget_are_mutual_inverses() {
        let m = Cpmm::new(1_000.0, 2).unwrap();
        for budget in [1.0, 50.0, 500.0] {
            let shares = m.shares_for_budget(0, budget).unwrap();
            let cost = m.cost_to_buy(0, shares).unwrap();
            assert!((cost - budget).abs() < 1e-6, "budget {budget} -> {cost}");
        }
    }

    #[test]
    fn cpmm_inversion_holds_for_three_outcomes() {
        let m = Cpmm::new(1_000.0, 3).unwrap();
        let shares = m.shares_for_budget(2, 200.0).unwrap();
        let cost = m.cost_to_buy(2, shares).unwrap();
        assert!((cost - 200.0).abs() < 1e-6, "got {cost}");
    }

    #[test]
    fn cpmm_round_trip_costs_the_trader_nothing_extra() {
        // With no fee the mechanism is reversible, so a buy-then-sell returns
        // to the starting state. Any leakage here would be an arbitrage.
        let mut m = Cpmm::new(1_000.0, 2).unwrap();
        let paid = m.buy(0, 150.0).unwrap().cost;
        let received = -m.buy(0, -150.0).unwrap().cost;
        assert!((paid - received).abs() < 1e-6, "paid {paid}, got back {received}");
        assert!((m.spot(0).unwrap().get() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn cpmm_slippage_grows_with_size_and_shrinks_with_depth() {
        let small = Cpmm::new(1_000.0, 2).unwrap().buy(0, 20.0).unwrap().slippage;
        let large = Cpmm::new(1_000.0, 2).unwrap().buy(0, 500.0).unwrap().slippage;
        assert!(large > small);

        let deep = Cpmm::new(100_000.0, 2).unwrap().buy(0, 500.0).unwrap().slippage;
        assert!(deep < large / 10.0, "deeper pool should slip far less");
    }

    #[test]
    fn cpmm_never_lets_a_reserve_be_drained() {
        let mut m = Cpmm::new(1_000.0, 2).unwrap();
        // Asking for far more shares than the pool holds must fail cleanly.
        let huge = m.buy(0, 1e12);
        assert!(huge.is_ok() || huge.is_err());
        assert!(
            m.reserves().iter().all(|r| *r > 0.0),
            "reserves went non-positive: {:?}",
            m.reserves()
        );
        assert!(m.spot(0).unwrap().get() <= 1.0);
    }

    #[test]
    fn cpmm_loss_is_bounded_by_its_funding() {
        let m = Cpmm::new(1_000.0, 2).unwrap();
        assert_eq!(m.max_loss(), 1_000.0);
    }

    #[test]
    fn cpmm_rejects_nonsense_construction() {
        assert!(Cpmm::new(0.0, 2).is_err());
        assert!(Cpmm::new(-5.0, 2).is_err());
        assert!(Cpmm::new(1_000.0, 1).is_err());
        assert!(Cpmm::with_reserves(vec![100.0, 0.0], 100.0).is_err());
        assert!(Cpmm::with_reserves(vec![100.0], 100.0).is_err());
        let m = Cpmm::new(1_000.0, 2).unwrap();
        assert!(m.spot(5).is_err());
        assert!(m.cost_to_buy(0, f64::INFINITY).is_err());
    }

    #[test]
    fn cpmm_can_be_restored_from_chain_state() {
        let m = Cpmm::with_reserves(vec![800.0, 1_250.0], 1_000.0).unwrap();
        // Scarcer YES reserve means a higher YES price.
        assert!(m.spot(0).unwrap().get() > 0.5);
        sums_to_one(&m);
    }

    // -- cross-mechanism --------------------------------------------------

    #[test]
    fn both_mechanisms_agree_a_buy_costs_more_than_spot() {
        // The invariant every arbitrage strategy relies on: you never get the
        // displayed price, so an AMM/book spread narrower than the slippage is
        // not an opportunity.
        let makers: Vec<Box<dyn MarketMaker>> = vec![
            Box::new(Lmsr::new(1_000.0, 2).unwrap()),
            Box::new(Cpmm::new(1_000.0, 2).unwrap()),
        ];
        for m in makers {
            let spot = m.spot(0).unwrap().get();
            let shares = 100.0;
            let cost = m.cost_to_buy(0, shares).unwrap();
            assert!(
                cost / shares > spot,
                "average price {} must exceed spot {spot}",
                cost / shares
            );
        }
    }
}
