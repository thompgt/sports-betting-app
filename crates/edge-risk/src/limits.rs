//! Pre-trade limits and the vocabulary of a risk decision.

use std::fmt;

use edge_core::types::{Notional, Qty};
use serde::{Deserialize, Serialize};

/// Why an order was refused or cut down.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RiskBreach {
    /// The kill switch is engaged. Nothing opens until it is cleared by hand.
    KillSwitchActive,
    /// Too many orders in too short a window.
    RateLimit,
    /// Not enough uncommitted cash, after the reserve.
    InsufficientCash,
    /// Position in this market would exceed its contract cap.
    PositionSize,
    /// Capital at risk in this market would exceed its cap.
    PositionCost,
    /// Capital at risk across every market on this event would exceed its cap.
    EventConcentration,
    /// Total capital at risk would exceed the portfolio cap.
    PortfolioCost,
    /// A single order larger than the per-order cap.
    OrderSize,
    /// Already holding as many distinct markets as allowed.
    TooManyMarkets,
    /// The order's price is outside the tradable range.
    InvalidPrice,
    /// The market is not currently marked, so risk cannot be evaluated. Refusing
    /// to trade an unmarked market is deliberate: a stale or missing price is
    /// the most common cause of a large accidental position.
    NoMark,
}

impl fmt::Display for RiskBreach {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            RiskBreach::KillSwitchActive => "kill switch active",
            RiskBreach::RateLimit => "order rate limit exceeded",
            RiskBreach::InsufficientCash => "insufficient cash",
            RiskBreach::PositionSize => "position contract limit",
            RiskBreach::PositionCost => "position cost limit",
            RiskBreach::EventConcentration => "event concentration limit",
            RiskBreach::PortfolioCost => "portfolio cost limit",
            RiskBreach::OrderSize => "order size limit",
            RiskBreach::TooManyMarkets => "too many open markets",
            RiskBreach::InvalidPrice => "price outside tradable range",
            RiskBreach::NoMark => "market has no current mark",
        })
    }
}

/// What the risk engine decided about a proposed order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskDecision {
    Approve(Qty),
    /// Allowed, but smaller. Preferred over rejection wherever a limit is about
    /// *size* rather than about permission: a strategy asking for 1,000 when 300
    /// is allowed should trade 300, not nothing. Rejecting outright throws away
    /// real edge and — worse — trains strategies to ask for less than they want.
    Resize(Qty, RiskBreach),
    Reject(RiskBreach),
}

impl RiskDecision {
    /// Contracts actually permitted. Zero for a rejection.
    pub fn qty(&self) -> Qty {
        match self {
            RiskDecision::Approve(q) | RiskDecision::Resize(q, _) => *q,
            RiskDecision::Reject(_) => Qty::ZERO,
        }
    }

    pub fn is_allowed(&self) -> bool {
        self.qty().get() > 0
    }

    pub fn breach(&self) -> Option<RiskBreach> {
        match self {
            RiskDecision::Approve(_) => None,
            RiskDecision::Resize(_, b) | RiskDecision::Reject(b) => Some(*b),
        }
    }
}

/// Why trading was halted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KillReason {
    /// Cumulative loss since the session anchor exceeded its limit.
    DailyLoss,
    /// Equity fell too far from its high-water mark.
    Drawdown,
    /// Market data stopped arriving. Trading on a frozen book is the fastest
    /// way to accumulate a position nobody wants.
    StaleData,
    /// Tripped by an operator.
    Manual,
    /// The venue rejected too many orders in a row, which usually means the
    /// engine's view of the account or the market is wrong.
    VenueRejections,
}

impl fmt::Display for KillReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            KillReason::DailyLoss => "daily loss limit breached",
            KillReason::Drawdown => "maximum drawdown breached",
            KillReason::StaleData => "market data is stale",
            KillReason::Manual => "manually halted",
            KillReason::VenueRejections => "too many consecutive venue rejections",
        })
    }
}

/// The complete set of pre-trade constraints.
///
/// Defaults are deliberately conservative — small enough that a
/// misconfiguration is survivable and an operator has to consciously raise them
/// before risking real size.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RiskLimits {
    /// Maximum absolute contracts in any one market.
    pub max_position_contracts: i64,
    /// Maximum capital at risk in any one market.
    pub max_position_cost: Notional,
    /// Maximum capital at risk across every market resolving on one event.
    /// The limit that actually binds — ten markets on ten legs of one game are
    /// one bet, and a per-market limit does not constrain it at all.
    pub max_event_cost: Notional,
    /// Maximum total capital at risk.
    pub max_portfolio_cost: Notional,
    /// Maximum cost of any single order.
    pub max_order_cost: Notional,
    /// Cash never to be committed, so there is always something left to close
    /// positions and pay fees with.
    pub min_cash_reserve: Notional,
    /// Loss since the session anchor that trips the kill switch.
    pub max_daily_loss: Notional,
    /// Fractional decline from peak equity that trips the kill switch.
    pub max_drawdown: f64,
    /// Sustained order submission rate.
    pub max_orders_per_second: f64,
    /// Burst allowance above the sustained rate.
    pub order_burst: f64,
    /// Maximum number of distinct markets held at once.
    pub max_open_markets: usize,
    /// How old a mark may be before the market is untradable, in seconds.
    pub max_mark_age_secs: f64,
    /// Consecutive venue rejections before halting.
    pub max_consecutive_rejects: u32,
}

impl Default for RiskLimits {
    fn default() -> Self {
        RiskLimits {
            max_position_contracts: 1_000,
            max_position_cost: Notional::from_dollars(250.0),
            max_event_cost: Notional::from_dollars(500.0),
            max_portfolio_cost: Notional::from_dollars(2_500.0),
            max_order_cost: Notional::from_dollars(250.0),
            min_cash_reserve: Notional::from_dollars(100.0),
            max_daily_loss: Notional::from_dollars(500.0),
            max_drawdown: 0.20,
            max_orders_per_second: 10.0,
            order_burst: 50.0,
            max_open_markets: 50,
            max_mark_age_secs: 30.0,
            max_consecutive_rejects: 10,
        }
    }
}

impl RiskLimits {
    /// Limits scaled to a bankroll, for an operator who would rather state one
    /// number than twelve.
    pub fn for_bankroll(bankroll: Notional) -> Self {
        let frac = |f: f64| Notional((bankroll.0 as f64 * f).round() as i64);
        RiskLimits {
            max_position_contracts: (bankroll.dollars() * 0.05 / 0.5).round() as i64,
            max_position_cost: frac(0.02),
            max_event_cost: frac(0.05),
            max_portfolio_cost: frac(0.50),
            max_order_cost: frac(0.02),
            min_cash_reserve: frac(0.05),
            max_daily_loss: frac(0.05),
            max_drawdown: 0.20,
            ..Default::default()
        }
    }

    /// Reject a configuration that cannot be satisfied, rather than discovering
    /// it one order at a time.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.max_position_cost > self.max_event_cost {
            return Err("per-market cost limit exceeds the per-event limit");
        }
        if self.max_event_cost > self.max_portfolio_cost {
            return Err("per-event cost limit exceeds the portfolio limit");
        }
        if !(0.0..=1.0).contains(&self.max_drawdown) {
            return Err("max drawdown must be a fraction in [0, 1]");
        }
        if self.max_orders_per_second <= 0.0 {
            return Err("order rate must be positive");
        }
        if self.max_position_contracts <= 0 {
            return Err("position contract limit must be positive");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decisions_expose_their_size_and_cause() {
        assert_eq!(RiskDecision::Approve(Qty(10)).qty(), Qty(10));
        assert!(RiskDecision::Approve(Qty(10)).breach().is_none());

        let r = RiskDecision::Resize(Qty(3), RiskBreach::PositionCost);
        assert_eq!(r.qty(), Qty(3));
        assert!(r.is_allowed());
        assert_eq!(r.breach(), Some(RiskBreach::PositionCost));

        let x = RiskDecision::Reject(RiskBreach::KillSwitchActive);
        assert_eq!(x.qty(), Qty::ZERO);
        assert!(!x.is_allowed());
        assert_eq!(x.breach(), Some(RiskBreach::KillSwitchActive));
    }

    #[test]
    fn default_limits_are_self_consistent() {
        RiskLimits::default().validate().unwrap();
    }

    #[test]
    fn bankroll_scaled_limits_are_self_consistent() {
        for bankroll in [500.0, 10_000.0, 1_000_000.0] {
            RiskLimits::for_bankroll(Notional::from_dollars(bankroll))
                .validate()
                .unwrap_or_else(|e| panic!("bankroll {bankroll}: {e}"));
        }
    }

    #[test]
    fn an_incoherent_configuration_is_caught_up_front() {
        let mut l = RiskLimits::default();
        l.max_position_cost = Notional(l.max_event_cost.0 * 2);
        assert!(l.validate().is_err());

        let mut l = RiskLimits::default();
        l.max_event_cost = Notional(l.max_portfolio_cost.0 * 2);
        assert!(l.validate().is_err());

        assert!(RiskLimits { max_drawdown: 1.5, ..Default::default() }.validate().is_err());

        assert!(
            RiskLimits { max_orders_per_second: 0.0, ..Default::default() }.validate().is_err()
        );
    }

    #[test]
    fn breaches_and_kill_reasons_are_printable() {
        // Operators read these in logs at three in the morning.
        assert_eq!(RiskBreach::EventConcentration.to_string(), "event concentration limit");
        assert_eq!(KillReason::DailyLoss.to_string(), "daily loss limit breached");
    }
}
