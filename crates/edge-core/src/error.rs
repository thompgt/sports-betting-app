//! Error type shared by every quantitative primitive in the workspace.
//!
//! The engine treats bad market data as an expected, routine condition rather
//! than a bug: a venue can and will publish a crossed book, a zero price, or a
//! market whose outcome probabilities sum to less than one. Everything here is
//! therefore fallible and recoverable, never a panic.

use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Error)]
pub enum EdgeError {
    #[error("invalid odds ({context}): {value}")]
    InvalidOdds { context: &'static str, value: f64 },

    #[error("probability out of range: {0}")]
    InvalidProbability(f64),

    #[error("invalid price: {0}")]
    InvalidPrice(i64),

    #[error("invalid quantity: {0}")]
    InvalidQuantity(i64),

    #[error("{method} failed to converge after {iterations} iterations (residual {residual:e})")]
    Convergence {
        method: &'static str,
        iterations: u32,
        residual: f64,
    },

    #[error("degenerate market: {0}")]
    DegenerateMarket(&'static str),

    #[error("empty input where at least one element was required: {0}")]
    Empty(&'static str),
}

pub type Result<T> = std::result::Result<T, EdgeError>;
