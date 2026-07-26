//! The strategies themselves.
//!
//! Each is an independent implementation of [`crate::strategy::Strategy`] with
//! its own configuration, and each is deliberately narrow — one idea, stated
//! plainly, testable in isolation. Combining them is the engine's job, not
//! theirs.

pub mod momentum;
pub mod quoting;
pub mod reversion;
pub mod value;

pub use momentum::{Momentum, MomentumConfig};
pub use quoting::{QuoteConfig, QuoteMaker};
pub use reversion::{MeanReversion, ReversionConfig};
pub use value::{ValueConfig, ValueTaker};
