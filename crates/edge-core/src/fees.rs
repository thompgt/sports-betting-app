//! Venue fee models.
//!
//! Fees are not a rounding error in this business — they are usually larger than
//! the edge. Kalshi's trading fee peaks at 1.75c per contract at a 50c price,
//! which is more than the entire gross edge on most opportunities a scanner will
//! surface. Any expected value computed before fees is fiction, so the fee model
//! lives in the core next to the EV calculation and every EV function takes one.

use serde::{Deserialize, Serialize};

use crate::types::{Notional, Price, Qty, Side};

/// Whether an order added liquidity to the book or removed it. Several venues
/// price the two differently, and a market-making strategy's entire economics
/// depend on landing on the maker side.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Liquidity {
    Maker,
    Taker,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FeeModel {
    /// No trading fee. The correct model for a paper-trading sanity check and
    /// for venues that charge only on settlement.
    #[default]
    None,

    /// Kalshi's published schedule: `fee = ceil(rate × C × P × (1 − P))` in
    /// cents, charged on takers only. The `P(1−P)` shape means the fee is
    /// heaviest exactly where prediction markets are most liquid, around 50c.
    Kalshi {
        /// Published as 0.07 for most markets, 0.035 for a few.
        taker_rate: f64,
        /// Kalshi has historically rebated or zero-rated maker fills; keep it
        /// configurable rather than assuming.
        maker_rate: f64,
    },

    /// A flat rate in basis points of notional traded, applied per side.
    Bps { maker_bps: f64, taker_bps: f64 },

    /// No fee on the trade, a percentage cut of *net winnings* at settlement.
    /// Charging this at trade time would overstate the cost of a losing
    /// position, so it is deliberately zero here and applied in settlement.
    WinningsOnly { rate: f64 },
}

impl FeeModel {
    /// Kalshi's standard schedule as published at the time of writing.
    ///
    /// Fee schedules change; this is a default, not a guarantee. The live
    /// adapter should override it from venue configuration.
    pub const KALSHI_STANDARD: FeeModel = FeeModel::Kalshi {
        taker_rate: 0.07,
        maker_rate: 0.0,
    };

    /// Fee charged to execute `qty` contracts at `price`, in micro-dollars.
    /// Always non-negative.
    pub fn trade_fee(&self, price: Price, qty: Qty, liquidity: Liquidity) -> Notional {
        let c = qty.get().abs() as f64;
        if c == 0.0 {
            return Notional::ZERO;
        }
        let p = price.dollars().clamp(0.0, 1.0);

        match *self {
            FeeModel::None | FeeModel::WinningsOnly { .. } => Notional::ZERO,

            FeeModel::Kalshi {
                taker_rate,
                maker_rate,
            } => {
                let rate = match liquidity {
                    Liquidity::Maker => maker_rate,
                    Liquidity::Taker => taker_rate,
                };
                if rate <= 0.0 {
                    return Notional::ZERO;
                }
                // The venue rounds the whole-order fee up to the next cent, so
                // the fee is computed on the order and not per contract.
                let dollars = rate * c * p * (1.0 - p);
                // Nudge before rounding up: an exact 1.75c comes out of binary
                // floating point as 1.7500000000000002, and a bare `ceil` would
                // charge a whole extra cent on every round-number order.
                let cents = (dollars * 100.0 - 1e-9).ceil();
                Notional((cents * 10_000.0) as i64)
            }

            FeeModel::Bps {
                maker_bps,
                taker_bps,
            } => {
                let bps = match liquidity {
                    Liquidity::Maker => maker_bps,
                    Liquidity::Taker => taker_bps,
                };
                let notional = p * c;
                Notional::from_dollars(notional * bps / 10_000.0)
            }
        }
    }

    /// Fee charged at settlement on a position that finished in the money.
    /// `profit` is net winnings before this cut.
    pub fn settlement_fee(&self, profit: Notional) -> Notional {
        match *self {
            FeeModel::WinningsOnly { rate } if profit.0 > 0 => {
                Notional((profit.0 as f64 * rate).round() as i64)
            }
            _ => Notional::ZERO,
        }
    }

    /// Fee expressed per contract, in dollars — the form the EV calculation wants.
    pub fn fee_per_contract(&self, price: Price, qty: Qty, liquidity: Liquidity) -> f64 {
        let q = qty.get().abs();
        if q == 0 {
            return 0.0;
        }
        self.trade_fee(price, qty, liquidity).dollars() / q as f64
    }

    /// The all-in price of acquiring one contract on `side` at `price`, i.e. the
    /// price you should actually be comparing against fair value.
    ///
    /// A buy is made worse by the fee, a sell is too — fees are always a cost,
    /// so they widen the effective spread from both directions.
    pub fn effective_price(&self, price: Price, side: Side, qty: Qty, liquidity: Liquidity) -> f64 {
        let f = self.fee_per_contract(price, qty, liquidity);
        match side {
            Side::Buy => price.dollars() + f,
            Side::Sell => price.dollars() - f,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_fee_model_is_free() {
        let f = FeeModel::None;
        assert_eq!(
            f.trade_fee(Price::from_cents(50), Qty(100), Liquidity::Taker),
            Notional::ZERO
        );
    }

    #[test]
    fn kalshi_fee_peaks_at_the_midpoint() {
        let f = FeeModel::KALSHI_STANDARD;
        let mid = f.trade_fee(Price::from_cents(50), Qty(100), Liquidity::Taker);
        let tail = f.trade_fee(Price::from_cents(10), Qty(100), Liquidity::Taker);
        assert!(mid > tail, "fee should be largest where P(1-P) is largest");
        // 0.07 * 100 * 0.5 * 0.5 = $1.75
        assert!((mid.dollars() - 1.75).abs() < 1e-9, "got {mid}");
    }

    #[test]
    fn kalshi_fee_rounds_up_to_the_cent() {
        let f = FeeModel::KALSHI_STANDARD;
        // 0.07 * 1 * 0.5 * 0.5 = $0.0175 -> rounds up to $0.02
        let fee = f.trade_fee(Price::from_cents(50), Qty(1), Liquidity::Taker);
        assert!((fee.dollars() - 0.02).abs() < 1e-9, "got {fee}");
    }

    #[test]
    fn kalshi_makers_pay_nothing_by_default() {
        let f = FeeModel::KALSHI_STANDARD;
        assert_eq!(
            f.trade_fee(Price::from_cents(50), Qty(100), Liquidity::Maker),
            Notional::ZERO
        );
    }

    #[test]
    fn fee_is_never_negative_at_the_boundaries() {
        let f = FeeModel::KALSHI_STANDARD;
        for c in [1, 25, 50, 75, 99] {
            let fee = f.trade_fee(Price::from_cents(c), Qty(500), Liquidity::Taker);
            assert!(fee.0 >= 0, "negative fee at {c}c");
        }
    }

    #[test]
    fn bps_fee_scales_with_notional() {
        let f = FeeModel::Bps {
            maker_bps: 0.0,
            taker_bps: 100.0, // 1%
        };
        // 100 contracts at 50c = $50 notional, 1% = $0.50
        let fee = f.trade_fee(Price::from_cents(50), Qty(100), Liquidity::Taker);
        assert!((fee.dollars() - 0.50).abs() < 1e-9, "got {fee}");
    }

    #[test]
    fn winnings_fee_is_charged_at_settlement_not_on_the_trade() {
        let f = FeeModel::WinningsOnly { rate: 0.02 };
        assert_eq!(
            f.trade_fee(Price::from_cents(50), Qty(100), Liquidity::Taker),
            Notional::ZERO
        );
        assert!((f.settlement_fee(Notional::from_dollars(100.0)).dollars() - 2.0).abs() < 1e-9);
        // A loss is not taxed.
        assert_eq!(f.settlement_fee(Notional::from_dollars(-50.0)), Notional::ZERO);
    }

    #[test]
    fn fees_widen_the_spread_from_both_sides() {
        let f = FeeModel::KALSHI_STANDARD;
        let p = Price::from_cents(50);
        let buy = f.effective_price(p, Side::Buy, Qty(100), Liquidity::Taker);
        let sell = f.effective_price(p, Side::Sell, Qty(100), Liquidity::Taker);
        assert!(buy > p.dollars());
        assert!(sell < p.dollars());
    }
}
