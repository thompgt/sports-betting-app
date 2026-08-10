//! Failures the ingestion layer expects to see.
//!
//! Every variant here is a *routine* condition on a live venue connection, not
//! a bug. The distinction that matters operationally is not what went wrong but
//! whether retrying could plausibly succeed — [`DataError::is_transient`] is
//! what the retry loop and the circuit breaker both key off, so classification
//! lives with the error rather than being re-derived at each call site.

use std::time::Duration;

use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Error)]
pub enum DataError {
    /// The request never got an answer: DNS, TCP, TLS, timeout, reset.
    #[error("transport error talking to {venue}: {detail}")]
    Transport { venue: String, detail: String },

    /// The venue answered with a status we did not want.
    #[error("{venue} returned HTTP {status}: {detail}")]
    Http { venue: String, status: u16, detail: String },

    /// Throttled. `retry_after` is the venue's own instruction where it gave
    /// one; obeying it is strictly better than guessing.
    #[error("{venue} throttled the client")]
    RateLimited { venue: String, retry_after: Option<Duration> },

    /// The client metered itself and declined to send. Not a venue failure, and
    /// deliberately distinct from [`DataError::RateLimited`]: this one must not
    /// count against the circuit breaker, because nothing was tried.
    #[error("client-side rate limit: wait {0:?}")]
    Throttled(Duration),

    /// The breaker is open — a call was refused without being attempted.
    #[error("circuit open for {venue}, retry in {retry_in:?}")]
    CircuitOpen { venue: String, retry_in: Duration },

    /// Credentials missing, malformed, expired, or rejected.
    #[error("authentication failed for {venue}: {detail}")]
    Auth { venue: String, detail: String },

    /// The response arrived but did not mean what the schema says it should.
    #[error("could not decode {venue} {what}: {detail}")]
    Decode { venue: String, what: String, detail: String },

    /// Well-formed, but describing a market the engine cannot represent — a
    /// scalar payoff, a nonsensical tick size, a price outside \[0,1].
    #[error("{venue} market {ticker} is unusable: {reason}")]
    Unusable { venue: String, ticker: String, reason: String },

    /// A venue-native name that entity resolution could not place.
    #[error("could not resolve {what}: {detail}")]
    Unresolved { what: String, detail: String },

    /// The stream ended. Not fatal on its own — reconnection is the norm.
    #[error("{venue} stream closed: {detail}")]
    StreamClosed { venue: String, detail: String },

    /// Misconfiguration. Retrying this forever is the classic way to turn a
    /// typo into an outage, so it is explicitly permanent.
    #[error("configuration error: {0}")]
    Config(String),
}

impl DataError {
    /// Could the same call succeed if made again?
    ///
    /// The two interesting negatives are [`DataError::Auth`] and
    /// [`DataError::Config`]: both look like ordinary failures and both will
    /// fail identically forever, so retrying them burns quota and hides the
    /// real problem behind a wall of noise. Decoding failures are also
    /// permanent — the venue changed its schema and no number of retries will
    /// change it back.
    pub fn is_transient(&self) -> bool {
        match self {
            DataError::Transport { .. }
            | DataError::RateLimited { .. }
            | DataError::Throttled(_)
            | DataError::StreamClosed { .. } => true,
            DataError::Http { status, .. } => {
                // 408 timeout, 425 too early, 429 throttle, and the whole 5xx
                // family are the server's problem, not the request's.
                matches!(status, 408 | 425 | 429) || *status >= 500
            }
            DataError::CircuitOpen { .. }
            | DataError::Auth { .. }
            | DataError::Decode { .. }
            | DataError::Unusable { .. }
            | DataError::Unresolved { .. }
            | DataError::Config(_) => false,
        }
    }

    /// Should this failure count against the circuit breaker?
    ///
    /// Only evidence about the *venue* counts. A client-side throttle or an
    /// already-open circuit says nothing about whether the venue is healthy,
    /// and letting either trip the breaker creates a feedback loop in which the
    /// breaker's own refusals keep it open.
    pub fn counts_against_health(&self) -> bool {
        !matches!(
            self,
            DataError::Throttled(_)
                | DataError::CircuitOpen { .. }
                | DataError::Unresolved { .. }
                | DataError::Unusable { .. }
        )
    }

    /// The venue's own instruction about when to come back, if it gave one.
    pub fn retry_after(&self) -> Option<Duration> {
        match self {
            DataError::RateLimited { retry_after, .. } => *retry_after,
            DataError::Throttled(d) => Some(*d),
            DataError::CircuitOpen { retry_in, .. } => Some(*retry_in),
            _ => None,
        }
    }

    /// Short, stable label for metrics and logs. Deliberately not the `Display`
    /// text, which carries unbounded detail and would explode a metric's
    /// cardinality.
    pub fn kind(&self) -> &'static str {
        match self {
            DataError::Transport { .. } => "transport",
            DataError::Http { .. } => "http",
            DataError::RateLimited { .. } => "rate_limited",
            DataError::Throttled(_) => "throttled",
            DataError::CircuitOpen { .. } => "circuit_open",
            DataError::Auth { .. } => "auth",
            DataError::Decode { .. } => "decode",
            DataError::Unusable { .. } => "unusable",
            DataError::Unresolved { .. } => "unresolved",
            DataError::StreamClosed { .. } => "stream_closed",
            DataError::Config(_) => "config",
        }
    }
}

pub type Result<T> = std::result::Result<T, DataError>;

#[cfg(test)]
mod tests {
    use super::*;

    fn http(status: u16) -> DataError {
        DataError::Http { venue: "kalshi".into(), status, detail: String::new() }
    }

    #[test]
    fn server_faults_are_transient_and_client_faults_are_not() {
        for s in [408, 425, 429, 500, 502, 503, 504] {
            assert!(http(s).is_transient(), "{s} should be retried");
        }
        for s in [400, 401, 403, 404, 422] {
            assert!(!http(s).is_transient(), "{s} will fail identically forever");
        }
    }

    #[test]
    fn bad_credentials_are_never_retried() {
        let e = DataError::Auth { venue: "kalshi".into(), detail: "expired key".into() };
        assert!(!e.is_transient(), "retrying a bad key just burns quota");
        assert!(e.counts_against_health(), "but it is still a real failure of the connection");
    }

    #[test]
    fn a_schema_change_is_permanent() {
        let e = DataError::Decode {
            venue: "polymarket".into(),
            what: "book".into(),
            detail: "missing field `asks`".into(),
        };
        assert!(!e.is_transient());
    }

    #[test]
    fn the_breakers_own_refusals_cannot_keep_it_open() {
        let open = DataError::CircuitOpen { venue: "k".into(), retry_in: Duration::from_secs(5) };
        let throttled = DataError::Throttled(Duration::from_millis(50));
        assert!(!open.counts_against_health());
        assert!(!throttled.counts_against_health(), "we never asked the venue anything");
    }

    #[test]
    fn a_venues_own_retry_instruction_is_carried_through() {
        let e = DataError::RateLimited {
            venue: "kalshi".into(),
            retry_after: Some(Duration::from_secs(30)),
        };
        assert_eq!(e.retry_after(), Some(Duration::from_secs(30)));
        assert_eq!(http(500).retry_after(), None);
    }

    #[test]
    fn kinds_are_bounded_and_do_not_leak_detail() {
        let e =
            DataError::Transport { venue: "v".into(), detail: "connection reset by peer".into() };
        assert_eq!(e.kind(), "transport");
        assert!(!e.kind().contains("peer"));
    }
}
