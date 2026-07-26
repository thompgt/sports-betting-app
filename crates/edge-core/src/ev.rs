//! Expected value, position sizing, and arbitrage.
//!
//! Everything here is **net of fees by construction**. There is no function that
//! computes a gross edge, because a gross edge is not a number anyone should act
//! on: on a 50c contract, Kalshi's taker fee alone is 1.75c, which exceeds the
//! entire edge in the large majority of opportunities a scanner surfaces. The
//! predecessor Python service computed EV gross and would have reported a
//! long tail of "edges" that lose money on contact.

use serde::{Deserialize, Serialize};

use crate::error::{EdgeError, Result};
use crate::fees::{FeeModel, Liquidity};
use crate::types::{Price, Prob, Qty, Side};

// ---------------------------------------------------------------------------
// Expected value
// ---------------------------------------------------------------------------

/// Expected profit per $1 staked at `decimal_odds` when the true probability of
/// winning is `fair`. The sportsbook form; `0.05` means +5% EV.
pub fn ev_per_dollar(decimal_odds: f64, fair: Prob) -> Result<f64> {
    if !decimal_odds.is_finite() || decimal_odds <= 1.0 {
        return Err(EdgeError::InvalidOdds {
            context: "decimal odds must exceed 1.0",
            value: decimal_odds,
        });
    }
    let p = fair.get();
    Ok(p * (decimal_odds - 1.0) - (1.0 - p))
}

/// The full assessment of one tradable price against one fair-value estimate.
///
/// This is the unit the strategy layer and the persistence layer both speak in,
/// so it carries the inputs alongside the conclusion — an edge you cannot
/// reconstruct later is an edge you cannot audit.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EdgeAssessment {
    pub side: Side,
    /// The price as quoted by the venue.
    pub price: Price,
    /// What one contract actually costs to acquire on this side, after fees.
    pub cost_per_contract: f64,
    /// Probability that the contract acquired on this side settles at $1.
    pub win_prob: f64,
    /// Model fair value of the YES leg, carried through unchanged for auditing.
    pub fair: Prob,
    /// Expected profit in dollars per contract, net of fees.
    pub ev_per_contract: f64,
    /// Expected profit per dollar of capital committed. The comparable figure
    /// across markets at different price levels.
    pub ev_per_dollar: f64,
    /// Full-Kelly fraction of bankroll. Sizing applies a multiplier to this.
    pub kelly: f64,
    /// Fair probability minus all-in cost, in probability points. The cleanest
    /// single number for ranking, and the one CLV is later measured against.
    pub edge: f64,
}

impl EdgeAssessment {
    #[inline]
    pub fn is_profitable(&self) -> bool {
        self.ev_per_contract > 0.0
    }
}

/// Assess buying or selling a binary contract at `price` against fair value `fair`.
///
/// `fair` is always the probability of the **YES** leg. Selling YES is treated
/// as buying NO at `1 − price`, which is what it is economically and what the
/// venue does internally.
///
/// `qty` matters because fee schedules are not linear in size — Kalshi rounds
/// the fee up to a whole cent per order, so a one-lot pays proportionally far
/// more than a thousand-lot.
pub fn assess(
    price: Price,
    fair: Prob,
    side: Side,
    qty: Qty,
    fee: &FeeModel,
    liquidity: Liquidity,
) -> Result<EdgeAssessment> {
    if !price.is_tradable() {
        return Err(EdgeError::InvalidPrice(price.micros()));
    }
    let fee_per = fee.fee_per_contract(price, qty, liquidity);

    // Reduce both sides to the same shape: pay `cost`, receive $1 with
    // probability `win_prob`.
    let (cost, win_prob) = match side {
        Side::Buy => (price.dollars() + fee_per, fair.get()),
        Side::Sell => (price.complement().dollars() + fee_per, 1.0 - fair.get()),
    };

    // Fees can push the all-in cost of a near-certain contract past $1, at which
    // point there is no bet to size — only a guaranteed loss.
    if cost <= 0.0 || cost >= 1.0 {
        return Ok(EdgeAssessment {
            side,
            price,
            cost_per_contract: cost,
            win_prob,
            fair,
            ev_per_contract: win_prob - cost,
            ev_per_dollar: if cost > 0.0 { (win_prob - cost) / cost } else { 0.0 },
            kelly: 0.0,
            edge: win_prob - cost,
        });
    }

    let ev_per_contract = win_prob - cost;
    let ev_per_dollar = ev_per_contract / cost;
    // f* = (p − c) / (1 − c) for a contract costing c that pays $1.
    let kelly = ((win_prob - cost) / (1.0 - cost)).max(0.0);

    Ok(EdgeAssessment {
        side,
        price,
        cost_per_contract: cost,
        win_prob,
        fair,
        ev_per_contract,
        ev_per_dollar,
        kelly,
        edge: win_prob - cost,
    })
}

/// Assess both directions and return whichever is better, if either is positive.
///
/// Useful because a mispriced market is tradable from exactly one side, and
/// which side it is depends on the sign of the error.
pub fn best_side(
    bid: Price,
    ask: Price,
    fair: Prob,
    qty: Qty,
    fee: &FeeModel,
    liquidity: Liquidity,
) -> Result<Option<EdgeAssessment>> {
    // You buy at the ask and sell at the bid, never the other way around. Using
    // a mid or a single price here is the classic way to manufacture an edge
    // that does not exist.
    let mut best: Option<EdgeAssessment> = None;
    if ask.is_tradable() {
        let a = assess(ask, fair, Side::Buy, qty, fee, liquidity)?;
        if a.is_profitable() {
            best = Some(a);
        }
    }
    if bid.is_tradable() {
        let b = assess(bid, fair, Side::Sell, qty, fee, liquidity)?;
        if b.is_profitable() && best.map(|x| b.ev_per_dollar > x.ev_per_dollar).unwrap_or(true) {
            best = Some(b);
        }
    }
    Ok(best)
}

// ---------------------------------------------------------------------------
// Sizing
// ---------------------------------------------------------------------------

/// How aggressively to act on the Kelly fraction.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct KellyPolicy {
    /// Multiplier on full Kelly. Full Kelly is the growth-optimal bet *only if
    /// your probability estimate is exactly right*; it is not, so anything above
    /// about a half is reckless. Quarter Kelly gives up ~6% of long-run growth
    /// for a large reduction in drawdown.
    pub fraction: f64,
    /// Hard ceiling on any single position as a share of bankroll, applied after
    /// the multiplier. Catches the case where a model error produces a
    /// near-certain edge that Kelly would happily size at 80% of the account.
    pub max_fraction: f64,
    /// Shrink the edge toward zero to account for estimation error, expressed as
    /// the standard deviation of the fair-probability estimate. See
    /// [`shrunk_kelly`].
    pub estimate_sd: f64,
}

impl Default for KellyPolicy {
    fn default() -> Self {
        KellyPolicy {
            fraction: 0.25,
            max_fraction: 0.05,
            estimate_sd: 0.02,
        }
    }
}

impl KellyPolicy {
    /// Kelly stake as a fraction of bankroll, after shrinkage, the fractional
    /// multiplier and the cap.
    pub fn stake_fraction(&self, a: &EdgeAssessment) -> f64 {
        if !a.is_profitable() {
            return 0.0;
        }
        let shrunk = shrunk_kelly(a, self.estimate_sd);
        (shrunk * self.fraction).clamp(0.0, self.max_fraction)
    }

    /// Number of contracts to trade given a bankroll.
    pub fn size(&self, a: &EdgeAssessment, bankroll: f64) -> Qty {
        let f = self.stake_fraction(a);
        if f <= 0.0 || a.cost_per_contract <= 0.0 {
            return Qty::ZERO;
        }
        // Flooring a value that "should" be exactly 2000 but arrives as
        // 1999.9999999999995 costs a contract on every clean-numbered size.
        Qty((bankroll * f / a.cost_per_contract + 1e-9).floor() as i64)
    }
}

/// Kelly adjusted for the fact that the fair probability is an estimate.
///
/// Treating an uncertain `p` as certain systematically oversizes: the Kelly
/// objective is concave in `p`, so error in either direction costs growth, and
/// the penalty is asymmetric against you. Subtracting one standard deviation of
/// estimation error from the edge is a crude but robust correction, and it has
/// the right behaviour at the extremes — a marginal edge inside the noise is
/// sized at zero rather than at "small but positive".
pub fn shrunk_kelly(a: &EdgeAssessment, estimate_sd: f64) -> f64 {
    if estimate_sd <= 0.0 {
        return a.kelly;
    }
    let adjusted_p = (a.win_prob - estimate_sd).max(0.0);
    let c = a.cost_per_contract;
    if c <= 0.0 || c >= 1.0 {
        return 0.0;
    }
    ((adjusted_p - c) / (1.0 - c)).max(0.0)
}

/// Classic Kelly for a bet at decimal odds. Retained for the sportsbook side of
/// the arbitrage strategies, where prices are quoted as odds rather than as a
/// contract price.
pub fn kelly_decimal(decimal_odds: f64, fair: Prob) -> f64 {
    if !decimal_odds.is_finite() || decimal_odds <= 1.0 {
        return 0.0;
    }
    let b = decimal_odds - 1.0;
    let p = fair.get();
    (((b * p) - (1.0 - p)) / b).max(0.0)
}

// ---------------------------------------------------------------------------
// Arbitrage
// ---------------------------------------------------------------------------

/// A set of simultaneous positions that profit regardless of the outcome.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Arbitrage {
    /// Share of total capital to allocate to each leg, summing to 1.
    pub stakes: Vec<f64>,
    /// Guaranteed return on total capital committed, net of fees.
    pub profit_pct: f64,
    /// Sum of all-in costs. Below 1 is what makes the arbitrage exist.
    pub total_cost: f64,
}

/// Detect arbitrage across the best available price for each outcome.
///
/// `costs` are the all-in per-contract costs of covering each outcome, already
/// including fees — see [`FeeModel::effective_price`]. They must cover a
/// mutually exclusive and exhaustive outcome set, which in practice means one
/// leg per outcome, each at whichever venue quotes it best.
pub fn arbitrage(costs: &[f64]) -> Option<Arbitrage> {
    if costs.is_empty() || costs.iter().any(|c| !c.is_finite() || *c <= 0.0) {
        return None;
    }
    let total: f64 = costs.iter().sum();
    if total >= 1.0 {
        return None;
    }
    Some(Arbitrage {
        // Stake proportional to cost so every outcome returns the same $1/total.
        stakes: costs.iter().map(|c| c / total).collect(),
        profit_pct: 1.0 / total - 1.0,
        total_cost: total,
    })
}

/// The prediction-market special case: buy YES on one venue and NO on another.
///
/// Returns the guaranteed profit per matched pair of contracts, in dollars, or
/// `None` when there is none. This is the highest-conviction opportunity the
/// engine can find — it does not depend on any probability estimate being right,
/// only on both fills landing.
pub fn cross_venue_arb(
    yes_ask: Price,
    no_ask: Price,
    qty: Qty,
    yes_fee: &FeeModel,
    no_fee: &FeeModel,
    liquidity: Liquidity,
) -> Option<f64> {
    if !yes_ask.is_tradable() || !no_ask.is_tradable() {
        return None;
    }
    let cost = yes_ask.dollars()
        + yes_fee.fee_per_contract(yes_ask, qty, liquidity)
        + no_ask.dollars()
        + no_fee.fee_per_contract(no_ask, qty, liquidity);
    // The pair settles at exactly $1 whichever way the event resolves.
    let profit = 1.0 - cost;
    if profit > 0.0 { Some(profit) } else { None }
}

// ---------------------------------------------------------------------------
// Closing line value
// ---------------------------------------------------------------------------

/// Closing line value: how much better the price taken was than the price the
/// market settled on, in probability points.
///
/// This is the only honest short-horizon scorecard a prediction-market strategy
/// has. Realised PnL over a few hundred binary events is almost entirely noise;
/// consistently beating the close is not, because the closing price is the
/// market's best estimate and beating it repeatedly requires genuine
/// information. Positive CLV does not guarantee profit — fees can still eat it —
/// but negative CLV guarantees its absence.
pub fn clv(entry: Price, close: Price, side: Side) -> f64 {
    let e = entry.dollars();
    let c = close.dollars();
    match side {
        // Bought at 40c, closed at 45c: the market came to us, +5 points.
        Side::Buy => c - e,
        Side::Sell => e - c,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FREE: FeeModel = FeeModel::None;

    #[test]
    fn ev_of_a_fair_bet_is_zero() {
        // 2.0 decimal at a true coin flip.
        assert!(ev_per_dollar(2.0, Prob::HALF).unwrap().abs() < 1e-12);
    }

    #[test]
    fn ev_of_a_priced_in_market_is_zero() {
        // Buying at exactly fair value, with no fee, must be a wash.
        let a = assess(
            Price::from_cents(40),
            Prob::new(0.40).unwrap(),
            Side::Buy,
            Qty(100),
            &FREE,
            Liquidity::Taker,
        )
        .unwrap();
        assert!(a.ev_per_contract.abs() < 1e-12);
        assert_eq!(a.kelly, 0.0);
        assert!(!a.is_profitable());
    }

    #[test]
    fn buying_below_fair_is_positive_ev() {
        let a = assess(
            Price::from_cents(40),
            Prob::new(0.50).unwrap(),
            Side::Buy,
            Qty(100),
            &FREE,
            Liquidity::Taker,
        )
        .unwrap();
        assert!((a.ev_per_contract - 0.10).abs() < 1e-12);
        assert!((a.ev_per_dollar - 0.25).abs() < 1e-12);
        // f* = (0.5 - 0.4) / 0.6
        assert!((a.kelly - 0.1 / 0.6).abs() < 1e-12);
    }

    #[test]
    fn selling_above_fair_is_positive_ev() {
        // Market says 60c, we think 50c: sell YES, which is buying NO at 40c.
        let a = assess(
            Price::from_cents(60),
            Prob::new(0.50).unwrap(),
            Side::Sell,
            Qty(100),
            &FREE,
            Liquidity::Taker,
        )
        .unwrap();
        assert!((a.cost_per_contract - 0.40).abs() < 1e-12);
        assert!((a.ev_per_contract - 0.10).abs() < 1e-12);
        assert!(a.is_profitable());
    }

    #[test]
    fn fees_can_erase_a_real_edge() {
        // A 1c gross edge at 50c, where Kalshi's taker fee is 1.75c.
        let fair = Prob::new(0.51).unwrap();
        let price = Price::from_cents(50);
        let gross = assess(price, fair, Side::Buy, Qty(100), &FREE, Liquidity::Taker).unwrap();
        assert!(gross.is_profitable(), "1c gross edge should look good pre-fee");

        let net = assess(
            price,
            fair,
            Side::Buy,
            Qty(100),
            &FeeModel::KALSHI_STANDARD,
            Liquidity::Taker,
        )
        .unwrap();
        assert!(
            !net.is_profitable(),
            "1c edge must not survive a 1.75c fee, got EV {}",
            net.ev_per_contract
        );
        assert_eq!(net.kelly, 0.0);
    }

    #[test]
    fn maker_and_taker_are_priced_differently() {
        let fair = Prob::new(0.53).unwrap();
        let price = Price::from_cents(50);
        let taker = assess(
            price,
            fair,
            Side::Buy,
            Qty(100),
            &FeeModel::KALSHI_STANDARD,
            Liquidity::Taker,
        )
        .unwrap();
        let maker = assess(
            price,
            fair,
            Side::Buy,
            Qty(100),
            &FeeModel::KALSHI_STANDARD,
            Liquidity::Maker,
        )
        .unwrap();
        assert!(maker.ev_per_contract > taker.ev_per_contract);
    }

    #[test]
    fn best_side_uses_the_correct_side_of_the_spread() {
        // Fair 50c inside a 45/55 market: neither side is worth taking.
        let none = best_side(
            Price::from_cents(45),
            Price::from_cents(55),
            Prob::new(0.50).unwrap(),
            Qty(100),
            &FREE,
            Liquidity::Taker,
        )
        .unwrap();
        assert!(none.is_none(), "a fair price inside the spread is not an edge");

        // Fair 70c against the same market: lift the 55c offer.
        let buy = best_side(
            Price::from_cents(45),
            Price::from_cents(55),
            Prob::new(0.70).unwrap(),
            Qty(100),
            &FREE,
            Liquidity::Taker,
        )
        .unwrap()
        .expect("70c fair vs a 55c offer is an edge");
        assert_eq!(buy.side, Side::Buy);
        assert_eq!(buy.price, Price::from_cents(55));

        // Fair 20c: hit the 45c bid.
        let sell = best_side(
            Price::from_cents(45),
            Price::from_cents(55),
            Prob::new(0.20).unwrap(),
            Qty(100),
            &FREE,
            Liquidity::Taker,
        )
        .unwrap()
        .expect("20c fair vs a 45c bid is an edge");
        assert_eq!(sell.side, Side::Sell);
    }

    #[test]
    fn kelly_policy_caps_an_absurd_edge() {
        // A "90% certain" 10c contract — almost always a data error, and full
        // Kelly would stake most of the account on it.
        let a = assess(
            Price::from_cents(10),
            Prob::new(0.90).unwrap(),
            Side::Buy,
            Qty(100),
            &FREE,
            Liquidity::Taker,
        )
        .unwrap();
        assert!(a.kelly > 0.8, "full Kelly here is enormous: {}", a.kelly);
        let policy = KellyPolicy::default();
        assert!(policy.stake_fraction(&a) <= policy.max_fraction);
    }

    #[test]
    fn shrinkage_zeroes_an_edge_inside_the_noise() {
        // A 1c edge with a 2c estimation error is not an edge.
        let a = assess(
            Price::from_cents(50),
            Prob::new(0.51).unwrap(),
            Side::Buy,
            Qty(100),
            &FREE,
            Liquidity::Taker,
        )
        .unwrap();
        assert!(a.kelly > 0.0);
        assert_eq!(shrunk_kelly(&a, 0.02), 0.0);
        // A 10c edge survives the same shrinkage.
        let big = assess(
            Price::from_cents(50),
            Prob::new(0.60).unwrap(),
            Side::Buy,
            Qty(100),
            &FREE,
            Liquidity::Taker,
        )
        .unwrap();
        assert!(shrunk_kelly(&big, 0.02) > 0.0);
    }

    #[test]
    fn sizing_converts_a_fraction_into_contracts() {
        let a = assess(
            Price::from_cents(50),
            Prob::new(0.70).unwrap(),
            Side::Buy,
            Qty(100),
            &FREE,
            Liquidity::Taker,
        )
        .unwrap();
        let policy = KellyPolicy {
            fraction: 0.25,
            max_fraction: 1.0,
            estimate_sd: 0.0,
        };
        // f* = 0.2/0.5 = 0.4; quarter Kelly = 0.10 of a $10,000 bankroll = $1,000
        // at 50c a contract = 2,000 contracts.
        assert_eq!(policy.size(&a, 10_000.0), Qty(2_000));
    }

    #[test]
    fn no_stake_without_an_edge() {
        let a = assess(
            Price::from_cents(60),
            Prob::new(0.50).unwrap(),
            Side::Buy,
            Qty(100),
            &FREE,
            Liquidity::Taker,
        )
        .unwrap();
        assert_eq!(KellyPolicy::default().size(&a, 10_000.0), Qty::ZERO);
    }

    #[test]
    fn arbitrage_found_and_priced() {
        // Cover both sides for 96c total.
        let arb = arbitrage(&[0.46, 0.50]).expect("96c total is an arbitrage");
        assert!((arb.profit_pct - (1.0 / 0.96 - 1.0)).abs() < 1e-12);
        assert!((arb.stakes.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        // Both legs return the same amount, which is the point.
        assert!((arb.stakes[0] / 0.46 - arb.stakes[1] / 0.50).abs() < 1e-12);
    }

    #[test]
    fn no_arbitrage_when_the_book_is_sane() {
        assert!(arbitrage(&[0.52, 0.51]).is_none());
        assert!(arbitrage(&[]).is_none());
        assert!(arbitrage(&[0.0, 0.5]).is_none());
    }

    #[test]
    fn cross_venue_arb_survives_or_dies_on_fees() {
        // YES at 46c on one venue, NO at 50c on another: 4c gross.
        let gross = cross_venue_arb(
            Price::from_cents(46),
            Price::from_cents(50),
            Qty(100),
            &FREE,
            &FREE,
            Liquidity::Taker,
        );
        assert!((gross.unwrap() - 0.04).abs() < 1e-12);

        // The same trade with Kalshi taker fees on both legs is a loser:
        // ~1.74c + ~1.75c of fee against 4c of gross... still positive, barely.
        let netted = cross_venue_arb(
            Price::from_cents(46),
            Price::from_cents(50),
            Qty(100),
            &FeeModel::KALSHI_STANDARD,
            &FeeModel::KALSHI_STANDARD,
            Liquidity::Taker,
        );
        assert!(
            netted.is_none() || netted.unwrap() < 0.01,
            "fees must consume most of a 4c cross-venue arb"
        );

        // A 1c gross arb is unambiguously destroyed.
        assert!(
            cross_venue_arb(
                Price::from_cents(49),
                Price::from_cents(50),
                Qty(100),
                &FeeModel::KALSHI_STANDARD,
                &FeeModel::KALSHI_STANDARD,
                Liquidity::Taker,
            )
            .is_none()
        );
    }

    #[test]
    fn clv_signs_correctly_for_both_sides() {
        // Bought at 40c, market closed at 45c: we got the better of it.
        assert!((clv(Price::from_cents(40), Price::from_cents(45), Side::Buy) - 0.05).abs() < 1e-12);
        // Sold at 40c and it closed at 45c: the market went against us.
        assert!((clv(Price::from_cents(40), Price::from_cents(45), Side::Sell) + 0.05).abs() < 1e-12);
    }

    #[test]
    fn untradable_prices_are_rejected() {
        assert!(
            assess(Price::ZERO, Prob::HALF, Side::Buy, Qty(1), &FREE, Liquidity::Taker).is_err()
        );
        assert!(assess(Price::ONE, Prob::HALF, Side::Buy, Qty(1), &FREE, Liquidity::Taker).is_err());
    }
}
