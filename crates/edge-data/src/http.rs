//! HTTP plumbing, kept away from the venue adapters.
//!
//! The adapters translate JSON into [`crate::source::VenueUpdate`]. That
//! translation is the part that is actually hard and the part that breaks when
//! a venue changes a field, so it must be testable without a network — which
//! means it cannot be entangled with a client, a socket, or a retry loop.
//!
//! Hence [`Transport`]: one method, "fetch these bytes". A real venue gets
//! [`HttpTransport`]; a test gets [`MockTransport`] and a recorded payload.
//! Nothing above this line knows the difference.
//!
//! The other job here is turning a `reqwest` outcome into a [`DataError`] that
//! classifies itself correctly, because everything downstream — retry, circuit
//! breaking, alerting — keys off that classification rather than re-deriving it.

use std::collections::HashMap;
use std::fmt;
use std::sync::Mutex;
use std::time::Duration;

use async_trait::async_trait;

use crate::error::{DataError, Result};

/// Somewhere bytes come from.
#[async_trait]
pub trait Transport: Send + Sync + fmt::Debug {
    /// `path` is relative to the transport's base URL.
    async fn get(&self, path: &str, query: &[(&str, String)]) -> Result<Vec<u8>>;
}

/// Decode a JSON body, reporting which venue and which payload failed.
///
/// A schema change is a permanent error, so getting this label right is the
/// difference between an alert that names the broken field and a retry loop
/// that hides it.
pub fn decode<T: serde::de::DeserializeOwned>(venue: &str, what: &str, body: &[u8]) -> Result<T> {
    serde_json::from_slice(body).map_err(|e| DataError::Decode {
        venue: venue.to_string(),
        what: what.to_string(),
        detail: e.to_string(),
    })
}

/// A real HTTP client against one venue's REST API.
#[derive(Debug)]
pub struct HttpTransport {
    venue: String,
    base: String,
    client: reqwest::Client,
    headers: Vec<(String, String)>,
}

impl HttpTransport {
    pub fn new(venue: impl Into<String>, base: impl Into<String>) -> Result<Self> {
        let venue = venue.into();
        let client = reqwest::Client::builder()
            // A market-data request that has not answered in ten seconds has
            // been overtaken by the next poll; waiting longer only queues work
            // behind a request whose answer is already stale.
            .timeout(Duration::from_secs(10))
            .connect_timeout(Duration::from_secs(5))
            .user_agent(concat!("edge/", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|e| DataError::Config(format!("building {venue} client: {e}")))?;
        Ok(HttpTransport {
            venue,
            base: base.into().trim_end_matches('/').to_string(),
            client,
            headers: Vec::new(),
        })
    }

    /// Add a header sent with every request — an API key, typically.
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    fn transport_err(&self, e: reqwest::Error) -> DataError {
        DataError::Transport { venue: self.venue.clone(), detail: e.to_string() }
    }
}

/// Percent-encode a query component.
///
/// Venue tickers contain characters that are legal in a path and meaningful in
/// a query — Polymarket's comma-separated token ids most obviously — so this
/// keeps the unreserved set of RFC 3986 and escapes everything else.
fn percent_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            other => out.push_str(&format!("%{other:02X}")),
        }
    }
    out
}

/// A venue's `Retry-After`, which is either seconds or an HTTP date. Only the
/// numeric form is honoured; a date needs a parsed clock offset to be useful
/// and guessing at one is worse than falling back to our own schedule.
fn retry_after(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    headers
        .get(reqwest::header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()
        .map(Duration::from_secs)
}

#[async_trait]
impl Transport for HttpTransport {
    async fn get(&self, path: &str, query: &[(&str, String)]) -> Result<Vec<u8>> {
        let mut url = format!("{}/{}", self.base, path.trim_start_matches('/'));
        if !query.is_empty() {
            let q: Vec<String> =
                query.iter().map(|(k, v)| format!("{}={}", percent_encode(k), percent_encode(v))).collect();
            url.push('?');
            url.push_str(&q.join("&"));
        }
        let mut req = self.client.get(&url);
        for (k, v) in &self.headers {
            req = req.header(k.as_str(), v.as_str());
        }

        let resp = req.send().await.map_err(|e| self.transport_err(e))?;
        let status = resp.status();

        if status.is_success() {
            return resp.bytes().await.map(|b| b.to_vec()).map_err(|e| self.transport_err(e));
        }

        let after = retry_after(resp.headers());
        // The body of an error response is usually the only thing that says
        // *why*, and it is routinely more useful than the status. Truncated,
        // because some venues return an HTML error page.
        let body = resp.text().await.unwrap_or_default();
        let detail: String = body.chars().take(500).collect();

        Err(match status.as_u16() {
            429 => DataError::RateLimited { venue: self.venue.clone(), retry_after: after },
            401 | 403 => DataError::Auth { venue: self.venue.clone(), detail },
            code => DataError::Http { venue: self.venue.clone(), status: code, detail },
        })
    }
}

/// A transport that answers from a fixed table. For testing adapters against
/// recorded payloads.
#[derive(Debug, Default)]
pub struct MockTransport {
    responses: Mutex<HashMap<String, Result<Vec<u8>>>>,
    calls: Mutex<Vec<String>>,
}

impl MockTransport {
    pub fn new() -> Self {
        Self::default()
    }

    /// Answer `path` with `body`.
    pub fn with(self, path: impl Into<String>, body: impl Into<Vec<u8>>) -> Self {
        self.responses.lock().unwrap().insert(path.into(), Ok(body.into()));
        self
    }

    /// Answer `path` with a failure.
    pub fn failing(self, path: impl Into<String>, err: DataError) -> Self {
        self.responses.lock().unwrap().insert(path.into(), Err(err));
        self
    }

    /// Paths requested, in order.
    pub fn calls(&self) -> Vec<String> {
        self.calls.lock().unwrap().clone()
    }
}

#[async_trait]
impl Transport for MockTransport {
    async fn get(&self, path: &str, query: &[(&str, String)]) -> Result<Vec<u8>> {
        let key = path.trim_start_matches('/').to_string();
        self.calls.lock().unwrap().push(if query.is_empty() {
            key.clone()
        } else {
            let q: Vec<String> = query.iter().map(|(k, v)| format!("{k}={v}")).collect();
            format!("{key}?{}", q.join("&"))
        });
        let table = self.responses.lock().unwrap();
        match table.get(&key).or_else(|| table.get(path)) {
            Some(Ok(body)) => Ok(body.clone()),
            Some(Err(e)) => Err(e.clone()),
            None => Err(DataError::Http {
                venue: "mock".into(),
                status: 404,
                detail: format!("no recorded response for {key:?}"),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, serde::Deserialize, PartialEq)]
    struct Payload {
        markets: Vec<String>,
    }

    #[tokio::test]
    async fn a_mock_answers_what_it_was_given() {
        let t = MockTransport::new().with("markets", br#"{"markets":["A","B"]}"#.to_vec());
        let body = t.get("/markets", &[]).await.unwrap();
        let p: Payload = decode("mock", "markets", &body).unwrap();
        assert_eq!(p, Payload { markets: vec!["A".into(), "B".into()] });
    }

    #[tokio::test]
    async fn a_mock_records_the_query_it_was_asked() {
        let t = MockTransport::new().with("markets", b"{}".to_vec());
        let _ = t.get("/markets", &[("limit", "100".into()), ("status", "open".into())]).await;
        assert_eq!(t.calls(), vec!["markets?limit=100&status=open"]);
    }

    #[tokio::test]
    async fn an_unrecorded_path_fails_loudly_rather_than_returning_nothing() {
        let t = MockTransport::new();
        let err = t.get("/surprise", &[]).await.unwrap_err();
        assert!(matches!(err, DataError::Http { status: 404, .. }));
    }

    #[test]
    fn a_schema_change_names_the_venue_and_the_payload() {
        let err = decode::<Payload>("kalshi", "markets", b"{\"markets\": 3}").unwrap_err();
        match &err {
            DataError::Decode { venue, what, .. } => {
                assert_eq!(venue, "kalshi");
                assert_eq!(what, "markets");
            }
            other => panic!("{other:?}"),
        }
        assert!(!err.is_transient(), "no number of retries will change the schema back");
    }

    #[test]
    fn a_malformed_body_is_a_decode_error_not_a_panic() {
        assert!(decode::<Payload>("v", "markets", b"not json at all").is_err());
        assert!(decode::<Payload>("v", "markets", b"").is_err());
    }

    #[test]
    fn building_a_client_against_a_base_url_normalises_the_slash() {
        let t = HttpTransport::new("kalshi", "https://example.test/v2/").unwrap();
        assert_eq!(t.base, "https://example.test/v2");
    }

    #[tokio::test]
    async fn a_recorded_failure_is_replayed_verbatim() {
        let t = MockTransport::new().failing(
            "markets",
            DataError::RateLimited { venue: "v".into(), retry_after: Some(Duration::from_secs(3)) },
        );
        let err = t.get("markets", &[]).await.unwrap_err();
        assert_eq!(err.retry_after(), Some(Duration::from_secs(3)));
        assert!(err.is_transient());
    }
}
