//! # edge-book
//!
//! Order book, matching engine and automated market makers.
//!
//! Like [`edge_core`], this crate is pure and deterministic: no clocks, no I/O,
//! no threads. Time and sequence numbers are supplied by the caller, so the same
//! input produces the same output whether it arrives from a live venue feed or
//! from a replayed journal.

#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

pub mod bitset;
pub mod book;
pub mod engine;
pub mod latency;
pub mod order;

pub use bitset::TickBitset;
pub use book::{LevelView, OrderBook};
pub use engine::{Command, EngineStats, MatchingEngine};
pub use latency::{LatencyHistogram, LatencySnapshot};
pub use order::{
    BookEvent, Fill, Order, OrderType, RejectReason, SelfTradePrevention, TimeInForce,
};
