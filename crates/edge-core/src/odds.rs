//! Conversions between the several ways a market can quote the same thing.
//!
//! Sportsbooks quote American or decimal odds; prediction-market venues quote a
//! contract price in cents. They are the same statement about probability wearing
//! different clothes, and the arbitrage strategies need to compare a Kalshi
//! contract against a sportsbook line, so every representation has to land on a
//! common footing: [`Prob`].

use crate::error::{EdgeError, Result};
use crate::types::{Price, Prob};

/// American ("moneyline") odds to decimal odds.
///
/// `-110` -> `1.909…`, `+150` -> `2.5`.
pub fn american_to_decimal(odds: i32) -> Result<f64> {
    // American odds have no representation between -100 and +100 exclusive;
    // a feed emitting one is malformed, and treating it as valid would produce
    // decimal odds below 1.0 and a probability above 1.
    if odds >= 100 {
        Ok(odds as f64 / 100.0 + 1.0)
    } else if odds <= -100 {
        Ok(100.0 / (-odds) as f64 + 1.0)
    } else {
        Err(EdgeError::InvalidOdds {
            context: "american odds must be >= +100 or <= -100",
            value: odds as f64,
        })
    }
}

/// Decimal odds to American odds, rounded to the nearest whole unit.
pub fn decimal_to_american(decimal: f64) -> Result<i32> {
    if !decimal.is_finite() || decimal <= 1.0 {
        return Err(EdgeError::InvalidOdds {
            context: "decimal odds must exceed 1.0",
            value: decimal,
        });
    }
    let b = decimal - 1.0;
    Ok(if b >= 1.0 {
        (b * 100.0).round() as i32
    } else {
        -((100.0 / b).round() as i32)
    })
}

/// Fractional odds (`numerator/denominator`, e.g. 5/2) to decimal.
pub fn fractional_to_decimal(numerator: f64, denominator: f64) -> Result<f64> {
    if !numerator.is_finite() || !denominator.is_finite() || denominator <= 0.0 || numerator <= 0.0 {
        return Err(EdgeError::InvalidOdds {
            context: "fractional odds must be positive",
            value: numerator,
        });
    }
    Ok(numerator / denominator + 1.0)
}

/// Implied probability of decimal odds, *before* removing the bookmaker margin.
pub fn decimal_to_implied(decimal: f64) -> Result<Prob> {
    if !decimal.is_finite() || decimal <= 1.0 {
        return Err(EdgeError::InvalidOdds {
            context: "decimal odds must exceed 1.0",
            value: decimal,
        });
    }
    Prob::new(1.0 / decimal)
}

/// Decimal odds implied by a probability.
pub fn implied_to_decimal(p: Prob) -> Result<f64> {
    p.to_decimal_odds()
}

/// Decimal odds equivalent to buying a contract at `price` and holding to settlement.
///
/// This is the bridge that lets a Kalshi contract be compared with a sportsbook
/// line: paying 40c for a $1 payoff is 2.5 decimal odds.
pub fn price_to_decimal(price: Price) -> Result<f64> {
    if !price.is_tradable() {
        return Err(EdgeError::InvalidPrice(price.micros()));
    }
    Ok(1.0 / price.dollars())
}

/// The contract price equivalent to a set of decimal odds.
pub fn decimal_to_price(decimal: f64) -> Result<Price> {
    Ok(decimal_to_implied(decimal)?.to_price())
}

/// Total implied probability across a market's outcomes.
///
/// Above 1.0 this is the bookmaker's margin (the *overround*); below 1.0 the
/// market is internally arbitrageable.
pub fn overround(implied: &[Prob]) -> f64 {
    implied.iter().map(|p| p.get()).sum()
}

/// Bookmaker margin as a fraction, i.e. `overround - 1`. Negative means the
/// book set of prices can be backed for a guaranteed profit.
pub fn margin(implied: &[Prob]) -> f64 {
    overround(implied) - 1.0
}

/// The *hold* — the share of total stake the book expects to keep. This is the
/// number sportsbooks quote about themselves, and it is not the same as the
/// margin: a 4.76% overround is a 4.55% hold.
pub fn hold(implied: &[Prob]) -> f64 {
    let o = overround(implied);
    if o <= 0.0 { 0.0 } else { 1.0 - 1.0 / o }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn american_decimal_known_values() {
        assert!((american_to_decimal(-110).unwrap() - 1.909_090_909).abs() < 1e-9);
        assert!((american_to_decimal(150).unwrap() - 2.5).abs() < 1e-12);
        assert!((american_to_decimal(100).unwrap() - 2.0).abs() < 1e-12);
        assert!((american_to_decimal(-100).unwrap() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn american_rejects_the_impossible_gap() {
        // No book quotes -99 or +42; a feed that does is corrupt, and silently
        // accepting it yields an implied probability above 1.
        assert!(american_to_decimal(0).is_err());
        assert!(american_to_decimal(99).is_err());
        assert!(american_to_decimal(-99).is_err());
    }

    #[test]
    fn american_decimal_roundtrip() {
        for a in [-500, -250, -110, 100, 150, 300, 1200] {
            let d = american_to_decimal(a).unwrap();
            assert_eq!(decimal_to_american(d).unwrap(), a, "roundtrip failed for {a}");
        }
    }

    #[test]
    fn even_money_has_two_spellings_and_we_pick_one() {
        // -100 and +100 are the same bet. The roundtrip cannot preserve both,
        // and the industry convention is to render even money as +100.
        assert_eq!(american_to_decimal(-100).unwrap(), american_to_decimal(100).unwrap());
        assert_eq!(decimal_to_american(2.0).unwrap(), 100);
    }

    #[test]
    fn price_and_odds_describe_the_same_bet() {
        // 40c for a $1 payoff == 2.5 decimal == 40% implied.
        let p = Price::from_cents(40);
        assert!((price_to_decimal(p).unwrap() - 2.5).abs() < 1e-12);
        assert_eq!(decimal_to_price(2.5).unwrap(), p);
        assert!((p.implied_prob().get() - 0.40).abs() < 1e-12);
    }

    #[test]
    fn untradable_prices_are_rejected() {
        assert!(price_to_decimal(Price::ZERO).is_err());
        assert!(price_to_decimal(Price::ONE).is_err());
    }

    #[test]
    fn overround_and_hold_differ() {
        // A standard -110/-110 two-way market.
        let imp: Vec<Prob> = [-110, -110]
            .iter()
            .map(|&a| decimal_to_implied(american_to_decimal(a).unwrap()).unwrap())
            .collect();
        assert!((overround(&imp) - 1.047_619_048).abs() < 1e-8);
        assert!((margin(&imp) - 0.047_619_048).abs() < 1e-8);
        assert!((hold(&imp) - 0.045_454_545).abs() < 1e-8);
    }

    #[test]
    fn fractional_conversion() {
        assert!((fractional_to_decimal(5.0, 2.0).unwrap() - 3.5).abs() < 1e-12);
        assert!(fractional_to_decimal(1.0, 0.0).is_err());
    }
}
