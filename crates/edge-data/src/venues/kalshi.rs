//! Kalshi.
//!
//! A CFTC-regulated exchange with a genuine central limit order book, integer
//! cent prices, and a published fee schedule — which makes it the venue this
//! system's economics were designed around.
//!
//! ## The book is published as two bid stacks
//!
//! Kalshi does not publish bids and asks. It publishes the YES bids and the NO
//! bids, because on a binary market those are the same thing seen from either
//! end: a NO bid at 53c *is* an offer to sell YES at 47c, and a taker lifting
//! it buys YES at 47c. So a market's YES ask side is not missing from the
//! payload, it is the NO stack reflected through `100 − p`.
//!
//! Getting this backwards produces a book that looks plausible — right shape,
//! right magnitudes, sorted correctly — and is wrong by the width of the
//! spread on every level. A strategy fed that book sees phantom edge on both
//! sides simultaneously and trades into it continuously. It is the single
//! easiest way to lose money integrating this venue, so the reflection is a
//! named function with its own tests rather than an inline expression.
//!
//! ## What is not here
//!
//! Order placement and the RSA-PSS request signing it requires. Market data on
//! Kalshi is public; only the trading endpoints need credentials, and those
//! belong with execution rather than with ingestion.

use async_trait::async_trait;
use edge_core::fees::FeeModel;
use edge_core::market::MarketStatus;
use edge_core::types::{Leg, Price, Qty, Ts, VenueId};
use serde::Deserialize;

use crate::error::{DataError, Result};
use crate::http::{Transport, decode};
use crate::source::{BookSnapshot, Level, Listing, MarketSource, VenueUpdate};
use crate::time::parse_rfc3339;

pub const VENUE: &str = "kalshi";
pub const DEFAULT_BASE: &str = "https://api.elections.kalshi.com/trade-api/v2";

/// Kalshi quotes in whole cents.
const CENT: i64 = 10_000;

// ---------------------------------------------------------------------------
// Wire types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct MarketsPage {
    #[serde(default)]
    pub markets: Vec<RawMarket>,
    #[serde(default)]
    pub cursor: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RawMarket {
    pub ticker: String,
    #[serde(default)]
    pub event_ticker: String,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub status: String,
    /// Present on some responses and not others; absent means "one cent".
    #[serde(default)]
    pub tick_size: Option<i64>,
    #[serde(default)]
    pub close_time: Option<String>,
    #[serde(default)]
    pub result: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OrderbookResponse {
    pub orderbook: RawOrderbook,
}

/// Both stacks are bids: `yes` bids for YES, `no` bids for NO. Each entry is
/// `[price_in_cents, contracts]`. A null stack means that side is empty.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct RawOrderbook {
    #[serde(default)]
    pub yes: Option<Vec<[i64; 2]>>,
    #[serde(default)]
    pub no: Option<Vec<[i64; 2]>>,
}

// ---------------------------------------------------------------------------
// Translation
// ---------------------------------------------------------------------------

/// The reflection at the heart of this adapter: a NO bid at `p` cents is an
/// offer to sell YES at `100 − p` cents.
///
/// Kept separate and named because getting it wrong yields a book that looks
/// entirely reasonable and is wrong by a spread on every level.
pub fn no_bid_to_yes_ask(no_price_cents: i64) -> Option<Price> {
    let yes = 100 - no_price_cents;
    // A NO bid at 0c or 100c would imply a YES ask at 100c or 0c — outside the
    // tradable range, and always an artefact rather than a real order.
    (1..100).contains(&yes).then(|| Price::from_cents(yes))
}

/// Kalshi's two bid stacks as one two-sided book.
pub fn decode_book(raw: &RawOrderbook, ts: Ts) -> BookSnapshot {
    let bids = raw
        .yes
        .iter()
        .flatten()
        .filter(|e| e[1] > 0 && (1..100).contains(&e[0]))
        .map(|e| Level::new(Price::from_cents(e[0]), Qty(e[1])))
        .collect();

    let asks = raw
        .no
        .iter()
        .flatten()
        .filter(|e| e[1] > 0)
        .filter_map(|e| no_bid_to_yes_ask(e[0]).map(|p| Level::new(p, Qty(e[1]))))
        .collect();

    BookSnapshot { bids, asks, seq: 0, ts }.normalise()
}

/// Kalshi's market lifecycle, in the engine's vocabulary.
pub fn decode_status(status: &str) -> MarketStatus {
    match status.to_ascii_lowercase().as_str() {
        "active" | "open" => MarketStatus::Open,
        "initialized" | "unopened" => MarketStatus::PreOpen,
        "paused" | "halted" => MarketStatus::Halted,
        "closed" | "determined" => MarketStatus::Closed,
        "settled" | "finalized" => MarketStatus::Settled,
        // An unrecognised status is not an open market. Venues add lifecycle
        // states, and defaulting to tradable would have the engine quote into
        // whatever they invent next.
        _ => MarketStatus::Halted,
    }
}

/// The settled outcome, where the venue has published one.
pub fn decode_result(result: Option<&str>) -> Option<bool> {
    match result?.to_ascii_lowercase().as_str() {
        "yes" => Some(true),
        "no" => Some(false),
        _ => None,
    }
}

/// One market's metadata as a [`Listing`].
///
/// `event_key` is Kalshi's own event ticker. Mapping that onto a cross-venue
/// canonical key is [`crate::resolve`]'s job and needs team names this payload
/// does not carry; until that runs, a Kalshi market is grouped with the other
/// Kalshi markets on the same event, which is what makes single-venue
/// multi-outcome arbitrage work immediately.
pub fn decode_listing(m: &RawMarket) -> Result<Listing> {
    let closes_at = match m.close_time.as_deref() {
        None | Some("") => None,
        Some(s) => Some(parse_rfc3339(s).map_err(|e| DataError::Unusable {
            venue: VENUE.into(),
            ticker: m.ticker.clone(),
            reason: format!("close_time: {e}"),
        })?),
    };

    let tick_size = match m.tick_size {
        Some(t) if t > 0 => t * CENT,
        _ => CENT,
    };

    let event_key = if m.event_ticker.is_empty() {
        format!("{VENUE}:{}", m.ticker)
    } else {
        format!("{VENUE}:{}", m.event_ticker)
    };

    Ok(Listing {
        ticker: m.ticker.clone(),
        title: if m.title.is_empty() { m.ticker.clone() } else { m.title.clone() },
        event_key,
        // A Kalshi ticker already names one outcome of its event (an event
        // ticker plus the outcome suffix), so it is the canonical outcome key.
        outcome_key: Some(m.ticker.clone()),
        // Kalshi markets are quoted from the YES side; the NO side of the same
        // contract is the same market seen through the reflection above, not a
        // separate listing.
        leg: Leg::Yes,
        tick_size,
        min_qty: Qty(1),
        max_qty: None,
        fee: FeeModel::KALSHI_STANDARD,
        closes_at,
        status: decode_status(&m.status),
    })
}

// ---------------------------------------------------------------------------
// Adapter
// ---------------------------------------------------------------------------

/// Kalshi market data over REST.
#[derive(Debug)]
pub struct Kalshi<T: Transport> {
    venue: VenueId,
    transport: T,
    /// Series to fetch, e.g. `KXNBAGAME`. Empty means everything open, which
    /// is thousands of markets and rarely what anyone wants.
    series: Vec<String>,
    page_size: usize,
}

impl<T: Transport> Kalshi<T> {
    pub fn new(venue: VenueId, transport: T) -> Self {
        Kalshi { venue, transport, series: Vec::new(), page_size: 200 }
    }

    /// Restrict the catalogue to these series tickers.
    pub fn with_series(mut self, series: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.series = series.into_iter().map(Into::into).collect();
        self
    }

    async fn markets_page(&self, series: Option<&str>, cursor: Option<&str>) -> Result<MarketsPage> {
        let mut query: Vec<(&str, String)> =
            vec![("limit", self.page_size.to_string()), ("status", "open".into())];
        if let Some(s) = series {
            query.push(("series_ticker", s.to_string()));
        }
        if let Some(c) = cursor {
            query.push(("cursor", c.to_string()));
        }
        let body = self.transport.get("/markets", &query).await?;
        decode(VENUE, "markets", &body)
    }

    /// Walk the cursor to the end of a series' catalogue.
    async fn all_markets(&self, series: Option<&str>) -> Result<Vec<RawMarket>> {
        let mut out = Vec::new();
        let mut cursor: Option<String> = None;
        // Bounded: a cursor that never terminates is a venue bug, and looping
        // on it forever would look exactly like a hang.
        for _ in 0..100 {
            let page = self.markets_page(series, cursor.as_deref()).await?;
            let empty = page.markets.is_empty();
            out.extend(page.markets);
            cursor = page.cursor.filter(|c| !c.is_empty());
            if cursor.is_none() || empty {
                break;
            }
        }
        Ok(out)
    }

    pub async fn book(&self, ticker: &str) -> Result<BookSnapshot> {
        let body = self
            .transport
            .get(&format!("/markets/{ticker}/orderbook"), &[("depth", "25".into())])
            .await?;
        let resp: OrderbookResponse = decode(VENUE, "orderbook", &body)?;
        Ok(decode_book(&resp.orderbook, crate::source::now()))
    }
}

#[async_trait]
impl<T: Transport> MarketSource for Kalshi<T> {
    fn venue(&self) -> VenueId {
        self.venue
    }

    fn name(&self) -> &str {
        VENUE
    }

    async fn listings(&self) -> Result<Vec<Listing>> {
        let mut raw = Vec::new();
        if self.series.is_empty() {
            raw.extend(self.all_markets(None).await?);
        } else {
            for s in &self.series {
                raw.extend(self.all_markets(Some(s)).await?);
            }
        }

        // A market we cannot represent is skipped, not fatal. One malformed
        // close time should not cost us the whole catalogue.
        Ok(raw.iter().filter_map(|m| decode_listing(m).ok()).collect())
    }

    async fn snapshot(&self, tickers: &[String]) -> Result<Vec<VenueUpdate>> {
        let mut out = Vec::with_capacity(tickers.len());
        for t in tickers {
            match self.book(t).await {
                Ok(book) => out.push(VenueUpdate::Book { ticker: t.clone(), book }),
                // One market's book failing must not cost us the others: the
                // caller asked for a batch and a partial answer is useful.
                Err(e) if !e.is_transient() => continue,
                Err(e) => return Err(e),
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::MockTransport;

    const V: VenueId = VenueId(1);

    /// A response shaped like the real one, including the fields we ignore.
    const MARKETS: &str = r#"{
      "markets": [
        {"ticker": "KXNBAGAME-25DEC25BOSLAL-BOS",
         "event_ticker": "KXNBAGAME-25DEC25BOSLAL",
         "title": "Will the Celtics beat the Lakers?",
         "status": "active", "tick_size": 1,
         "close_time": "2025-12-26T03:00:00Z",
         "yes_bid": 47, "yes_ask": 49, "volume": 12000},
        {"ticker": "KXNBAGAME-25DEC25BOSLAL-LAL",
         "event_ticker": "KXNBAGAME-25DEC25BOSLAL",
         "title": "Will the Lakers beat the Celtics?",
         "status": "active", "close_time": "2025-12-26T03:00:00Z"}
      ],
      "cursor": ""
    }"#;

    const BOOK: &str = r#"{"orderbook": {
        "yes": [[45, 100], [44, 250], [43, 500]],
        "no":  [[51, 80],  [50, 120], [49, 300]]
    }}"#;

    fn kalshi(t: MockTransport) -> Kalshi<MockTransport> {
        Kalshi::new(V, t)
    }

    #[test]
    fn a_no_bid_is_a_yes_offer_at_the_complement() {
        // The whole adapter turns on this. A NO bid at 51c is someone offering
        // to sell YES at 49c.
        assert_eq!(no_bid_to_yes_ask(51), Some(Price::from_cents(49)));
        assert_eq!(no_bid_to_yes_ask(1), Some(Price::from_cents(99)));
        assert_eq!(no_bid_to_yes_ask(99), Some(Price::from_cents(1)));
    }

    #[test]
    fn a_degenerate_no_bid_is_discarded_rather_than_reflected_out_of_range() {
        assert_eq!(no_bid_to_yes_ask(0), None, "would imply a YES ask at 100c");
        assert_eq!(no_bid_to_yes_ask(100), None, "would imply a YES ask at 0c");
        assert_eq!(no_bid_to_yes_ask(-5), None);
    }

    #[test]
    fn the_two_bid_stacks_become_one_two_sided_book() {
        let raw: OrderbookResponse = serde_json::from_str(BOOK).unwrap();
        let book = decode_book(&raw.orderbook, Ts::from_secs(1));

        assert_eq!(book.best_bid(), Some(Price::from_cents(45)));
        // The tightest NO bid is 51c, which is the *cheapest* YES offer at 49c.
        assert_eq!(book.best_ask(), Some(Price::from_cents(49)));
        assert!(!book.is_crossed());
        assert_eq!(book.bids.len(), 3);
        assert_eq!(book.asks.len(), 3);
    }

    #[test]
    fn the_ask_stack_is_ordered_by_the_reflected_price_not_the_published_one() {
        // Kalshi's NO stack descends by NO price, which ascends by YES price
        // only after reflection. Sorting the raw numbers would put the *worst*
        // offer at the touch.
        let raw: OrderbookResponse = serde_json::from_str(BOOK).unwrap();
        let book = decode_book(&raw.orderbook, Ts::ZERO);
        let asks: Vec<i64> = book.asks.iter().map(|l| l.price.0 / CENT).collect();
        assert_eq!(asks, vec![49, 50, 51]);
        assert_eq!(book.asks[0].qty, Qty(80), "quantities travel with their own level");
    }

    #[test]
    fn an_empty_or_one_sided_book_decodes_without_inventing_a_side() {
        let empty = decode_book(&RawOrderbook::default(), Ts::ZERO);
        assert_eq!(empty.best_bid(), None);
        assert_eq!(empty.best_ask(), None);

        let raw: RawOrderbook = serde_json::from_str(r#"{"yes": [[45, 10]], "no": null}"#).unwrap();
        let book = decode_book(&raw, Ts::ZERO);
        assert_eq!(book.best_bid(), Some(Price::from_cents(45)));
        assert_eq!(book.best_ask(), None);
    }

    #[test]
    fn zero_size_levels_are_not_levels() {
        let raw: RawOrderbook =
            serde_json::from_str(r#"{"yes": [[45, 0], [44, 10]], "no": [[51, 0]]}"#).unwrap();
        let book = decode_book(&raw, Ts::ZERO);
        assert_eq!(book.best_bid(), Some(Price::from_cents(44)));
        assert_eq!(book.asks.len(), 0);
    }

    #[test]
    fn an_unknown_status_is_not_tradable() {
        assert_eq!(decode_status("active"), MarketStatus::Open);
        assert_eq!(decode_status("paused"), MarketStatus::Halted);
        assert_eq!(decode_status("settled"), MarketStatus::Settled);
        assert_eq!(decode_status("closed"), MarketStatus::Closed);
        assert!(
            !decode_status("some_new_state_kalshi_invented").is_tradable(),
            "defaulting to open would have the engine quote into anything they add"
        );
    }

    #[test]
    fn a_settled_result_decodes_to_the_yes_outcome() {
        assert_eq!(decode_result(Some("yes")), Some(true));
        assert_eq!(decode_result(Some("no")), Some(false));
        assert_eq!(decode_result(Some("")), None);
        assert_eq!(decode_result(None), None);
    }

    #[test]
    fn a_listing_carries_the_fee_schedule_and_the_close() {
        let page: MarketsPage = serde_json::from_str(MARKETS).unwrap();
        let l = decode_listing(&page.markets[0]).unwrap();
        assert_eq!(l.ticker, "KXNBAGAME-25DEC25BOSLAL-BOS");
        assert_eq!(l.tick_size, CENT);
        assert_eq!(l.fee, FeeModel::KALSHI_STANDARD);
        assert_eq!(l.closes_at, Some(parse_rfc3339("2025-12-26T03:00:00Z").unwrap()));
        assert_eq!(l.status, MarketStatus::Open);
    }

    #[test]
    fn markets_on_one_kalshi_event_share_an_event_key() {
        // Enough on its own for single-venue multi-outcome arbitrage, before
        // any cross-venue resolution has run.
        let page: MarketsPage = serde_json::from_str(MARKETS).unwrap();
        let a = decode_listing(&page.markets[0]).unwrap();
        let b = decode_listing(&page.markets[1]).unwrap();
        assert_eq!(a.event_key, b.event_key);
        assert_ne!(a.ticker, b.ticker);
    }

    #[test]
    fn a_missing_tick_size_defaults_to_a_cent() {
        let page: MarketsPage = serde_json::from_str(MARKETS).unwrap();
        assert_eq!(decode_listing(&page.markets[1]).unwrap().tick_size, CENT);
    }

    #[test]
    fn an_unparseable_close_time_makes_the_market_unusable_rather_than_undated() {
        // Silently dropping the close would leave every time-to-resolution
        // feature reading "never", which is worse than skipping the market.
        let m = RawMarket {
            ticker: "T".into(),
            event_ticker: "E".into(),
            title: String::new(),
            status: "active".into(),
            tick_size: None,
            close_time: Some("whenever".into()),
            result: None,
        };
        assert!(matches!(decode_listing(&m), Err(DataError::Unusable { .. })));
    }

    #[tokio::test]
    async fn the_catalogue_is_fetched_and_translated() {
        let k = kalshi(MockTransport::new().with("markets", MARKETS.as_bytes().to_vec()));
        let listings = k.listings().await.unwrap();
        assert_eq!(listings.len(), 2);
        assert!(k.transport.calls()[0].contains("status=open"));
    }

    #[tokio::test]
    async fn one_unusable_market_does_not_cost_us_the_catalogue() {
        let json = r#"{"markets": [
            {"ticker": "GOOD", "event_ticker": "E", "status": "active",
             "close_time": "2025-12-26T03:00:00Z"},
            {"ticker": "BAD", "event_ticker": "E", "status": "active",
             "close_time": "not-a-time"}
        ]}"#;
        let k = kalshi(MockTransport::new().with("markets", json.as_bytes().to_vec()));
        let listings = k.listings().await.unwrap();
        assert_eq!(listings.len(), 1);
        assert_eq!(listings[0].ticker, "GOOD");
    }

    #[tokio::test]
    async fn a_snapshot_returns_a_book_per_requested_ticker() {
        let t = MockTransport::new()
            .with("markets/A/orderbook", BOOK.as_bytes().to_vec())
            .with("markets/B/orderbook", BOOK.as_bytes().to_vec());
        let k = kalshi(t);
        let updates = k.snapshot(&["A".into(), "B".into()]).await.unwrap();
        assert_eq!(updates.len(), 2);
        assert_eq!(updates[0].ticker(), Some("A"));
    }

    #[tokio::test]
    async fn a_transient_failure_on_one_book_is_propagated_for_the_guard_to_retry() {
        let t = MockTransport::new()
            .with("markets/A/orderbook", BOOK.as_bytes().to_vec())
            .failing(
                "markets/B/orderbook",
                DataError::Http { venue: VENUE.into(), status: 503, detail: String::new() },
            );
        let k = kalshi(t);
        let err = k.snapshot(&["A".into(), "B".into()]).await.unwrap_err();
        assert!(err.is_transient());
    }

    #[tokio::test]
    async fn a_market_that_no_longer_exists_is_skipped_not_fatal() {
        let t = MockTransport::new().with("markets/A/orderbook", BOOK.as_bytes().to_vec());
        let k = kalshi(t);
        let updates = k.snapshot(&["A".into(), "GONE".into()]).await.unwrap();
        assert_eq!(updates.len(), 1, "a delisted market must not stall the others");
    }

    #[tokio::test]
    async fn a_schema_change_is_reported_as_a_decode_failure() {
        let k = kalshi(MockTransport::new().with("markets", br#"{"markets": "surprise"}"#.to_vec()));
        let err = k.listings().await.unwrap_err();
        assert!(matches!(err, DataError::Decode { .. }));
        assert!(!err.is_transient());
    }
}
