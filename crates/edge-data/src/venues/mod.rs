//! Venue adapters.
//!
//! Each one translates a venue's own schema into [`crate::source::VenueUpdate`]
//! and does nothing else. Decoding is kept in free functions over the venue's
//! payload types, separate from the HTTP and WebSocket plumbing, so the part
//! most likely to be wrong — the translation — is testable against recorded
//! payloads without a network.

pub mod sim;

pub use sim::{SimConfig, Simulator};
