//! The strategies themselves.
//!
//! Each is an independent implementation of [`crate::strategy::Strategy`] with
//! its own configuration, and each is deliberately narrow — one idea, stated
//! plainly, testable in isolation. Combining them is the engine's job, not
//! theirs.

pub mod quoting;
pub mod value;

pub use quoting::{QuoteConfig, QuoteMaker};
pub use value::{ValueConfig, ValueTaker};
