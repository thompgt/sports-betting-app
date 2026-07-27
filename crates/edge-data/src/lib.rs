//! # edge-data
//!
//! Market data ingestion: getting prices out of venues and into the engine's
//! own vocabulary, without letting either the venues' unreliability or their
//! naming conventions leak downstream.
//!
//! Three concerns, deliberately separated:
//!
//! - **Resilience** — [`limiter`], [`backoff`], [`breaker`]. Every venue will
//!   throttle, time out, and fall over, and none of that should reach a
//!   strategy. All three are pure state machines over an explicit `Ts` with no
//!   clock of their own, so a recorded outage replays identically.
//! - **Resolution** — `resolve`. Kalshi's `KXNBA-25DEC25-BOS`, Polymarket's
//!   condition id, and a sportsbook's "Boston Celtics" are one event. Deciding
//!   that is fuzzy, fallible, and therefore quarantined here rather than
//!   rediscovered by string matching on every tick.
//! - **Sourcing** — `source` and `venues`. One trait over REST snapshots and
//!   streaming updates, with adapters behind it.

#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

pub mod assembler;
pub mod backoff;
pub mod breaker;
pub mod error;
pub mod limiter;
pub mod resolve;
pub mod similarity;
pub mod source;
pub mod time;

pub use backoff::{Backoff, Decision, Jitter, Retry, RetryPolicy};
pub use breaker::{BreakerConfig, CircuitBreaker, State as BreakerState};
pub use error::{DataError, Result};
pub use limiter::{Bucket, RateLimiter};
