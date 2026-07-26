//! The strategies themselves.
//!
//! Each is an independent implementation of [`crate::strategy::Strategy`] with
//! its own configuration, and each is deliberately narrow — one idea, stated
//! plainly, testable in isolation. Combining them is the engine's job.
//!
//! | Strategy | Trades on | Liquidity |
//! |---|---|---|
//! | [`Arbitrage`] | mutually exclusive legs costing under $1 | takes |
//! | [`ValueTaker`] | model or consensus disagreeing with the touch | takes |
//! | [`QuoteMaker`] | the spread, leaned against inventory | makes |
//! | [`Momentum`] | a move that order flow confirms | takes |
//! | [`MeanReversion`] | a move that order flow does *not* confirm | makes |
//!
//! Two relationships between them are load-bearing rather than incidental:
//!
//! **Momentum and mean reversion are disjoint by construction.** One requires
//! flow confirmation and the other requires its absence, so they cannot fire on
//! the same book. Running a confirmation-free trend follower alongside a
//! confirmation-free fader is a machine for paying spread to yourself.
//!
//! **There is no separate "ML strategy".** The online predictor feeds
//! [`crate::strategy::MarketView::independent_fair`], which the value taker and
//! the maker already consume — a model that has earned weight moves fair value
//! and therefore moves quotes and triggers takes, while a model that has not
//! changes nothing anywhere. A parallel model-only strategy alongside that
//! would double-count one signal and size it twice.

pub mod arbitrage;
pub mod momentum;
pub mod quoting;
pub mod reversion;
pub mod value;

pub use arbitrage::{ArbConfig, Arbitrage};
pub use momentum::{Momentum, MomentumConfig};
pub use quoting::{QuoteConfig, QuoteMaker};
pub use reversion::{MeanReversion, ReversionConfig};
pub use value::{ValueConfig, ValueTaker};
