//! The pipeline, end to end, on recorded Kalshi payloads.
//!
//! Every crate here has thorough unit tests and, until this file, none of them
//! ran together. That is the gap worth closing: the assembler decides what a
//! market *is*, the strategy decides what to do about it, and the risk engine
//! decides whether that is allowed — and each of those three is only correct
//! relative to assumptions about the other two. A test that holds one at a time
//! cannot see a disagreement between them.
//!
//! So these tests drive the real chain — [`Kalshi`] over a mock transport, the
//! real [`edge_data::assembler::Assembler`], the real strategy, the real risk
//! engine — through [`Pipeline`], which is the same code `edge` runs.
//!
//! The assertions are deliberately about *refusal*. Confirming that a healthy
//! book produces quotes is the easy half and the half that is already covered;
//! what no single-crate test can establish is that a book the assembler
//! distrusts never reaches the strategy at all.

use edge_cli::Pipeline;
use edge_core::types::{Notional, Ts, VenueId};
use edge_data::http::MockTransport;
use edge_data::source::MarketSource;
use edge_data::venues::kalshi::Kalshi;

const VENUE: VenueId = VenueId(1);

/// Two markets on one event, shaped like the real `/markets` response.
const MARKETS: &str = r#"{
  "markets": [
    {"ticker": "KXNBAGAME-25DEC25BOSLAL-BOS",
     "event_ticker": "KXNBAGAME-25DEC25BOSLAL",
     "title": "Will the Celtics beat the Lakers?",
     "status": "active", "tick_size": 1,
     "close_time": "2035-12-26T03:00:00Z"},
    {"ticker": "KXNBAGAME-25DEC25BOSLAL-LAL",
     "event_ticker": "KXNBAGAME-25DEC25BOSLAL",
     "title": "Will the Lakers beat the Celtics?",
     "status": "active", "tick_size": 1,
     "close_time": "2035-12-26T03:00:00Z"}
  ],
  "cursor": ""
}"#;

const BOS: &str = "KXNBAGAME-25DEC25BOSLAL-BOS";
const LAL: &str = "KXNBAGAME-25DEC25BOSLAL-LAL";

/// A healthy two-sided book: YES bids up to 45c, NO bids up to 51c, which
/// reflects to a YES ask at 49c.
const HEALTHY: &str = r#"{"orderbook": {
    "yes": [[45, 100], [44, 250], [43, 500]],
    "no":  [[51, 80],  [50, 120], [49, 300]]
}}"#;

/// A book whose best YES bid (55c) sits above its best YES ask — the NO bid at
/// 51c reflects to a 49c offer. No venue publishes this; a payload that says so
/// was assembled from inconsistent parts.
const CROSSED: &str = r#"{"orderbook": {
    "yes": [[55, 100], [54, 250]],
    "no":  [[51, 80],  [50, 120]]
}}"#;

/// One-sided: bids only, so there is no mid to quote around.
const ONE_SIDED: &str = r#"{"orderbook": {"yes": [[45, 100], [44, 250]], "no": null}}"#;

fn kalshi(books: &[(&str, &str)]) -> Kalshi<MockTransport> {
    let mut t = MockTransport::new().with("markets", MARKETS.as_bytes().to_vec());
    for (ticker, body) in books {
        t = t.with(format!("markets/{ticker}/orderbook"), body.as_bytes().to_vec());
    }
    Kalshi::new(VENUE, t)
}

/// Register the catalogue and run `ticks` poll cycles a second apart.
async fn drive(venue: &Kalshi<MockTransport>, ticks: usize) -> Pipeline {
    let mut p = Pipeline::new(VENUE, Notional::from_dollars(10_000.0));
    let listings = venue.listings().await.expect("catalogue");
    p.register(&listings);

    let tickers: Vec<String> = listings.iter().map(|l| l.ticker.clone()).collect();
    for i in 0..ticks {
        let now = Ts::from_secs(1_000 + i as i64);
        let updates = venue.snapshot(&tickers, now).await.expect("snapshot");
        p.tick(updates, now);
    }
    p
}

#[tokio::test]
async fn a_healthy_book_makes_it_all_the_way_from_the_venue_to_an_order() {
    // The control. Without this the refusal tests below prove nothing: a
    // pipeline that never trades anything passes all of them trivially.
    let venue = kalshi(&[(BOS, HEALTHY), (LAL, HEALTHY)]);
    let p = drive(&venue, 5).await;

    assert_eq!(p.assembler.registry().len(), 2, "both markets interned");
    assert!(p.tally.intents > 0, "the strategy saw a quotable book");
    assert!(!p.placed.is_empty(), "and something reached the risk engine and passed");
    assert_eq!(p.tally.gaps, 0);
    assert!(p.reconciles(), "the ledger must close after a live session");
}

#[tokio::test]
async fn no_order_is_ever_placed_against_a_crossed_book() {
    // A crossed snapshot means the payload is internally inconsistent. The
    // assembler marks the market stale rather than applying it, and the point
    // of this test is that the *downstream* honours that: the strategy is never
    // offered the book, so no amount of apparent edge in it can be traded.
    let venue = kalshi(&[(BOS, CROSSED), (LAL, CROSSED)]);
    let p = drive(&venue, 5).await;

    assert!(p.tally.gaps > 0, "a crossed snapshot must be reported as a gap");
    assert_eq!(p.tally.intents, 0, "a crossed book must never reach the strategy");
    assert!(p.placed.is_empty(), "and must certainly never produce an order");
    assert_eq!(p.risk.portfolio().capital_at_risk(), Notional::ZERO);
    assert!(p.reconciles());
}

#[tokio::test]
async fn one_bad_market_does_not_stop_its_healthy_neighbour() {
    // The failure worth catching is the over-broad one: refusing the crossed
    // market by halting the whole poll. Markets on the same event are assessed
    // independently.
    let venue = kalshi(&[(BOS, CROSSED), (LAL, HEALTHY)]);
    let p = drive(&venue, 5).await;

    assert!(p.tally.gaps > 0);
    assert!(!p.placed.is_empty(), "the healthy market must still trade");

    let bos = p.assembler.market_of(BOS).expect("BOS interned");
    assert!(
        p.placed.iter().all(|o| o.market != bos),
        "no order may be placed on the crossed market"
    );
    assert!(p.reconciles());
}

#[tokio::test]
async fn a_book_that_stops_updating_stops_being_traded() {
    // Silence and a dead socket are indistinguishable, and both are answered
    // the same way. The assembler's staleness window is 30s; polls here are a
    // second apart, so this drives well past it with no fresh payload.
    let venue = kalshi(&[(BOS, HEALTHY), (LAL, HEALTHY)]);
    let mut p = Pipeline::new(VENUE, Notional::from_dollars(10_000.0));
    let listings = venue.listings().await.expect("catalogue");
    p.register(&listings);
    let tickers: Vec<String> = listings.iter().map(|l| l.ticker.clone()).collect();

    // One good poll, so there is a believable book to quote against.
    let updates = venue.snapshot(&tickers, Ts::from_secs(1_000)).await.expect("snapshot");
    p.tick(updates, Ts::from_secs(1_000));
    assert!(!p.placed.is_empty(), "the first poll must actually trade");

    // Then the feed goes quiet: ticks keep arriving, updates do not. Inside the
    // 30s window that is just a slow market and quoting continues, which is the
    // intended behaviour rather than something to assert against.
    for i in 1..=35 {
        p.tick(Vec::new(), Ts::from_secs(1_000 + i));
    }
    let after_window = p.placed.len();

    // Past it, silence is treated as a dead socket and the market stops being
    // quoted entirely.
    for i in 36..90 {
        p.tick(Vec::new(), Ts::from_secs(1_000 + i));
    }
    assert_eq!(
        p.placed.len(),
        after_window,
        "a feed silent past the staleness window must stop being quoted"
    );
    assert!(p.reconciles());
}

#[tokio::test]
async fn a_one_sided_book_is_not_quoted_around_an_invented_mid() {
    // There is no mid on a book with one side. Inventing one is how a maker
    // ends up quoting into a market that has no other side.
    let venue = kalshi(&[(BOS, ONE_SIDED), (LAL, ONE_SIDED)]);
    let p = drive(&venue, 5).await;

    assert_eq!(p.tally.gaps, 0, "one-sided is not inconsistent, just thin");
    assert!(p.placed.is_empty(), "but there is nothing to quote around");
    assert!(p.reconciles());
}

#[tokio::test]
async fn a_session_is_reproducible_from_the_same_payloads() {
    // The property the backtester depends on. Two runs over identical inputs
    // must produce identical orders — not merely similar totals — or a replay
    // proves nothing about the session it claims to reproduce.
    let a = drive(&kalshi(&[(BOS, HEALTHY), (LAL, HEALTHY)]), 8).await;
    let b = drive(&kalshi(&[(BOS, HEALTHY), (LAL, HEALTHY)]), 8).await;

    assert_eq!(a.placed, b.placed, "the same payloads must produce the same orders");
    assert_eq!(a.tally, b.tally);
    assert_eq!(a.risk.portfolio().cash, b.risk.portfolio().cash);
}

#[tokio::test]
async fn a_schema_change_stops_the_session_instead_of_emptying_the_universe() {
    // The whole-pipeline version of the adapter's own test: a decode failure
    // must not arrive downstream as "no markets have any depth today", which is
    // a perfectly plausible thing for the engine to see and act on.
    let venue = kalshi(&[(BOS, r#"{"orderbook": {"yes": "surprise"}}"#), (LAL, HEALTHY)]);
    let listings = venue.listings().await.expect("catalogue");
    let tickers: Vec<String> = listings.iter().map(|l| l.ticker.clone()).collect();

    let err = venue.snapshot(&tickers, Ts::from_secs(1_000)).await.unwrap_err();
    assert!(!err.is_transient(), "a schema change will not fix itself: {err}");
}
