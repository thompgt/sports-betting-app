//! # edge-risk
//!
//! Position accounting, pre-trade limits, tail risk, and the kill switch.
//!
//! The organising fact of this crate is that **prediction markets are prepaid**.
//! There is no margin, no borrow and no liquidation: taking the downside of an
//! event means buying the opposing contract outright. A position's maximum loss
//! is therefore exactly what was paid for it, known at trade time, with no
//! assumptions about volatility or correlation. Where an equities risk system
//! *estimates* capital at risk, this one knows it — and the limits are written
//! against that exact number rather than against a modelled one.
//!
//! Tail risk still needs modelling, because a portfolio of correlated binaries
//! can lose everything at once. [`var`] does that with a Monte Carlo over
//! resolution outcomes rather than a normal approximation, which for a Bernoulli
//! payoff does not merely lose precision but describes a different distribution
//! — and thins exactly the tail it is meant to measure.

#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

pub mod engine;
pub mod limits;
pub mod position;
pub mod var;

pub use engine::{RiskEngine, RiskStats};
pub use limits::{KillReason, RiskBreach, RiskDecision, RiskLimits};
pub use position::{Portfolio, Position};
pub use var::{VarConfig, VarResult, monte_carlo_var, parametric_var};
