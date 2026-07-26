//! # edge-alpha
//!
//! Signal generation: microstructure features, an online price predictor, and
//! (shortly) the strategies that trade on them.
//!
//! The organising principle of this crate is that a signal must **prove itself
//! against the market before it is allowed to move a price**. Every layer here
//! is built so that the untrained, uninformed, or broken case degenerates to
//! "quote the market back" rather than to "emit an arbitrary number":
//!
//! - [`features::FeatureExtractor`] returns `None` rather than inventing a mid
//!   for a one-sided book.
//! - [`predictor::Predictor`] anchors on the market log-odds and learns only the
//!   residual, so zero weights means zero disagreement.
//! - The model's blend weight is driven by its realised out-of-sample Brier
//!   score against the market's, so a model with no demonstrated skill has no
//!   influence and generates no trades.
//!
//! ```
//! use edge_alpha::features::{Features, N_FEATURES};
//! use edge_alpha::predictor::Predictor;
//! use edge_core::types::Prob;
//!
//! let mut model = Predictor::default();
//! let features = Features::from_values([0.0; N_FEATURES], 0.40);
//! let p = model.predict(&features, Prob::new(0.40).unwrap());
//!
//! // Nothing has been learned yet, so the forecast is the market, exactly.
//! assert!(p.is_market_echo());
//! assert_eq!(p.edge(), 0.0);
//! ```

#![forbid(unsafe_code)]

pub mod features;
pub mod predictor;
pub mod strategy;

pub use features::{FEATURE_NAMES, FeatureExtractor, Features, N_FEATURES};
pub use predictor::{Prediction, Predictor, PredictorConfig, Standardizer};
pub use strategy::{Action, MarketView, OrderIntent, RestingOrder, Strategy, StrategyStats};
