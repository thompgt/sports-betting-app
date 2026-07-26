//! # edge-core
//!
//! Domain types and quantitative primitives for prediction-market trading.
//!
//! This crate is deliberately **pure**: no I/O, no clock, no async, no global
//! state. Everything is a function of its arguments. That is what lets the
//! backtester and the live engine share one implementation and produce
//! bit-identical results from the same input stream — the single most valuable
//! property a trading system can have, because it means a backtest is evidence
//! about the code that will actually run rather than about a parallel
//! reimplementation of it.
//!
//! ## Layout
//!
//! - [`types`] — prices as integers, validated probabilities, ids, sides
//! - [`odds`] — American / decimal / fractional / contract-price conversions
//! - [`devig`] — removing bookmaker margin, four models, all bracketed solvers
//! - [`consensus`] — pooling several venues into one fair value, with dispersion
//! - [`fees`] — venue fee schedules, because a gross edge is not an edge
//! - [`ev`] — expected value, Kelly sizing, arbitrage, closing line value
//! - [`stats`] — streaming moments, scoring rules, risk metrics
//! - [`market`] — market specifications and the cross-venue registry
//!
//! ## The pricing path
//!
//! ```
//! use edge_core::{consensus::*, devig::DevigMethod, ev::*, fees::*, types::*};
//!
//! // Two venues quote the same event. Devig each, then pool in log-odds.
//! let sources = vec![
//!     SourceQuote::new(VenueId(1), vec![Prob::new(0.55)?, Prob::new(0.50)?], 1.0),
//!     SourceQuote::new(VenueId(2), vec![Prob::new(0.57)?, Prob::new(0.48)?], 1.0),
//! ];
//! let view = consensus(&sources, &ConsensusConfig::default())?;
//!
//! // Judge a third venue's offer against that fair value, net of fees.
//! let edge = assess(
//!     Price::from_cents(48),
//!     view.fair[0],
//!     Side::Buy,
//!     Qty(500),
//!     &FeeModel::KALSHI_STANDARD,
//!     Liquidity::Taker,
//! )?;
//!
//! // Size it, shrinking for how much the venues disagreed.
//! let policy = KellyPolicy { estimate_sd: view.estimate_sd(), ..Default::default() };
//! let contracts = policy.size(&edge, 10_000.0);
//! # Ok::<(), edge_core::error::EdgeError>(())
//! ```

#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

pub mod consensus;
pub mod devig;
pub mod error;
pub mod ev;
pub mod fees;
pub mod market;
pub mod odds;
pub mod stats;
pub mod types;

pub use error::{EdgeError, Result};
pub use types::{
    ClientOrderId, EventId, Leg, MarketId, Notional, OrderId, Price, Prob, Qty, Side, StrategyId,
    Ts, VenueId,
};
