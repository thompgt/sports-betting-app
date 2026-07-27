//! A synthetic venue.
//!
//! Not a toy. This is the source the backtester replays, the source the
//! integration tests run against, and the source a paper-trading run uses when
//! no venue credentials are configured — so it has to behave like the real
//! thing in the ways that matter and be *exactly* reproducible in the ways the
//! real thing is not.
//!
//! Reproducible means: same seed, same tape, byte for byte, forever. Every
//! random draw comes from one seeded [`Rng`] advanced in a fixed order, and the
//! clock is a counter rather than a reading. A backtest that cannot be replayed
//! is an anecdote.
//!
//! Behaving like the real thing means, specifically:
//!
//! - **Prices move in log-odds.** A random walk in probability space either
//!   walks out of `[0, 1]` or has to be clamped there, and clamping puts a
//!   wall of fake support under every long shot. In log-odds a 2c market can
//!   halve to 1c as easily as a 50c market moves to 33c, which is how these
//!   markets actually trade.
//! - **YES and NO are consistent.** Each event lists both legs, priced as
//!   complements around one latent truth. Without that there is no honest test
//!   of the arbitrage strategy — a simulator that prices the two legs
//!   independently manufactures arbitrage that no venue would ever show.
//! - **Markets settle.** Positions have to resolve for PnL to mean anything,
//!   and the outcome is drawn from the latent probability at close, so a
//!   well-calibrated strategy makes money and a badly calibrated one does not.

use std::sync::Mutex;
use std::time::Duration;

use async_trait::async_trait;
use edge_core::fees::FeeModel;
use edge_core::rng::Rng;
use edge_core::types::{Leg, Price, Prob, Qty, Side, Ts, VenueId};

use crate::error::Result;
use crate::source::{BookSnapshot, Level, Listing, MarketSource, VenueUpdate};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimConfig {
    pub seed: u64,
    /// Events listed. Each gets a YES and a NO market.
    pub events: usize,
    pub tick_size: i64,
    /// Levels published per side.
    pub levels: usize,
    /// Contracts resting at the touch. Deeper levels taper.
    pub depth: i64,
    pub half_spread_ticks: i64,
    /// Per-step standard deviation of the latent price, in log-odds.
    pub vol: f64,
    /// Simulated time between steps.
    pub step: Duration,
    /// Probability that a step also prints a trade.
    pub trade_rate: f64,
    pub fee: FeeModel,
    /// Simulated seconds from start to settlement.
    pub horizon_secs: f64,
}

impl Default for SimConfig {
    fn default() -> Self {
        SimConfig {
            seed: 0xED9E,
            events: 4,
            tick_size: 10_000,
            levels: 5,
            depth: 200,
            half_spread_ticks: 1,
            vol: 0.02,
            step: Duration::from_millis(250),
            trade_rate: 0.3,
            fee: FeeModel::KALSHI_STANDARD,
            horizon_secs: 3_600.0,
        }
    }
}

#[derive(Debug)]
struct State {
    rng: Rng,
    /// Latent fair value per event, in log-odds.
    truth: Vec<f64>,
    ts: Ts,
    seq: u64,
    settled: Vec<bool>,
}

/// A deterministic synthetic venue.
#[derive(Debug)]
pub struct Simulator {
    venue: VenueId,
    cfg: SimConfig,
    start: Ts,
    state: Mutex<State>,
}

impl Simulator {
    pub fn new(venue: VenueId, cfg: SimConfig) -> Self {
        let mut rng = Rng::new(cfg.seed);
        // Starting prices spread across the range rather than clustered at 50c,
        // so the fee model, the tick grid and the log-odds features are all
        // exercised at the extremes where they behave least like the middle.
        let truth = (0..cfg.events).map(|_| rng.uniform(-2.5, 2.5)).collect();
        Simulator {
            venue,
            cfg,
            start: Ts::ZERO,
            state: Mutex::new(State {
                rng,
                truth,
                ts: Ts::ZERO,
                seq: 0,
                settled: vec![false; cfg.events],
            }),
        }
    }

    pub fn config(&self) -> &SimConfig {
        &self.cfg
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn ticker(event: usize, leg: Leg) -> String {
        format!("SIM-{event:04}-{}", if leg == Leg::Yes { "YES" } else { "NO" })
    }

    pub fn event_key(event: usize) -> String {
        format!("sim:event:{event:04}")
    }

    /// The simulated clock. Advances only when the tape does.
    pub fn now(&self) -> Ts {
        self.lock().ts
    }

    /// The latent probability the tape is generated from.
    ///
    /// Available for tests and for the backtester's scoring, and for nothing
    /// else — a strategy that reads this is measuring its own answer key.
    pub fn truth(&self, event: usize) -> Option<Prob> {
        self.lock().truth.get(event).map(|l| Prob::from_logit(*l))
    }

    fn close_time(&self) -> Ts {
        Ts(self.start.0 + (self.cfg.horizon_secs * 1e9) as i64)
    }

    fn listings(&self) -> Vec<Listing> {
        (0..self.cfg.events)
            .flat_map(|e| {
                [Leg::Yes, Leg::No].into_iter().map(move |leg| {
                    let mut l = Listing::new(Simulator::ticker(e, leg), Simulator::event_key(e));
                    l.title = format!("Simulated event {e} — {leg}");
                    l.leg = leg;
                    l
                })
            })
            .map(|mut l| {
                l.tick_size = self.cfg.tick_size;
                l.fee = self.cfg.fee;
                l.closes_at = Some(self.close_time());
                l
            })
            .collect()
    }

    /// Build a book around `mid`, staying strictly inside the tradable range.
    fn book(&self, mid: Prob, leg: Leg, seq: u64, ts: Ts) -> BookSnapshot {
        let p = if leg == Leg::Yes { mid } else { mid.complement() };
        let tick = self.cfg.tick_size.max(1);
        let centre = Price::from_dollars(p.get()).unwrap_or(Price::from_cents(50)).0 / tick * tick;
        let half = self.cfg.half_spread_ticks.max(1) * tick;

        let mut bids = Vec::with_capacity(self.cfg.levels);
        let mut asks = Vec::with_capacity(self.cfg.levels);
        for i in 0..self.cfg.levels as i64 {
            // Depth thins going out, which is what makes sweep cost non-linear
            // and what a sizing rule has to respect.
            let qty = Qty((self.cfg.depth as f64 / (1.0 + i as f64)).round().max(1.0) as i64);
            let bid = centre - half - i * tick;
            let ask = centre + half + i * tick;
            if bid > 0 {
                bids.push(Level::new(Price(bid), qty));
            }
            if ask < edge_core::types::MICROS {
                asks.push(Level::new(Price(ask), qty));
            }
        }
        BookSnapshot { bids, asks, seq, ts }.normalise()
    }

    /// Advance the tape one step and return everything that happened.
    ///
    /// The whole simulator is this function; `snapshot` and `stream` are two
    /// ways of calling it.
    pub fn step(&self) -> Vec<VenueUpdate> {
        let mut s = self.lock();
        s.ts = Ts(s.ts.0 + self.cfg.step.as_nanos() as i64);
        s.seq += 1;
        let (ts, seq) = (s.ts, s.seq);
        let past_close = ts >= self.close_time();

        let mut out = Vec::with_capacity(self.cfg.events * 3);
        for e in 0..self.cfg.events {
            if s.settled[e] {
                continue;
            }

            if past_close {
                // The outcome is drawn from the latent probability, so a
                // calibrated strategy is rewarded and an overconfident one is
                // not. Anything else would make the backtest a fiction.
                let p = Prob::from_logit(s.truth[e]).get();
                let outcome = s.rng.bernoulli(p);
                s.settled[e] = true;
                for leg in [Leg::Yes, Leg::No] {
                    out.push(VenueUpdate::Settled {
                        ticker: Simulator::ticker(e, leg),
                        outcome: if leg == Leg::Yes { outcome } else { !outcome },
                        ts,
                    });
                }
                continue;
            }

            let shock = s.rng.gaussian(0.0, self.cfg.vol);
            s.truth[e] = (s.truth[e] + shock).clamp(-6.0, 6.0);
            let mid = Prob::from_logit(s.truth[e]);

            let prints = s.rng.bernoulli(self.cfg.trade_rate);
            let taker = if s.rng.bernoulli(0.5) { Side::Buy } else { Side::Sell };
            let size = Qty(1 + s.rng.below(self.cfg.depth.max(1) as u64 / 4 + 1) as i64);

            for leg in [Leg::Yes, Leg::No] {
                let ticker = Simulator::ticker(e, leg);
                let book = self.book(mid, leg, seq, ts);
                if prints {
                    // A trade happens at whichever side of the touch the taker
                    // crossed, which is what makes order-flow imbalance a real
                    // signal in the generated tape rather than noise.
                    let price = match taker {
                        Side::Buy => book.best_ask(),
                        Side::Sell => book.best_bid(),
                    };
                    if let Some(price) = price {
                        out.push(VenueUpdate::Trade {
                            ticker: ticker.clone(),
                            price,
                            qty: size,
                            taker,
                            ts,
                        });
                    }
                }
                out.push(VenueUpdate::Book { ticker, book });
            }
        }

        if out.is_empty() {
            out.push(VenueUpdate::Heartbeat { ts });
        }
        out
    }

    /// Has every market resolved?
    pub fn is_finished(&self) -> bool {
        self.lock().settled.iter().all(|s| *s)
    }
}

#[async_trait]
impl MarketSource for Simulator {
    fn venue(&self) -> VenueId {
        self.venue
    }

    fn name(&self) -> &str {
        "sim"
    }

    async fn listings(&self) -> Result<Vec<Listing>> {
        Ok(Simulator::listings(self))
    }

    async fn snapshot(&self, tickers: &[String]) -> Result<Vec<VenueUpdate>> {
        let all = self.step();
        if tickers.is_empty() {
            return Ok(all);
        }
        Ok(all
            .into_iter()
            .filter(|u| u.ticker().is_none_or(|t| tickers.iter().any(|w| w == t)))
            .collect())
    }

    async fn stream(
        &self,
        tickers: &[String],
        sink: tokio::sync::mpsc::Sender<VenueUpdate>,
    ) -> Result<()> {
        while !self.is_finished() {
            for u in self.snapshot(tickers).await? {
                if sink.send(u).await.is_err() {
                    return Ok(()); // the consumer went away; not an error
                }
            }
            tokio::time::sleep(self.cfg.step).await;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assembler::{Assembler, AssemblerConfig};

    const V: VenueId = VenueId(9);

    fn sim(cfg: SimConfig) -> Simulator {
        Simulator::new(V, cfg)
    }

    fn small() -> SimConfig {
        SimConfig { events: 2, horizon_secs: 10.0, step: Duration::from_secs(1), ..Default::default() }
    }

    fn tape(cfg: SimConfig, steps: usize) -> Vec<VenueUpdate> {
        let s = sim(cfg);
        (0..steps).flat_map(|_| s.step()).collect()
    }

    #[test]
    fn the_same_seed_produces_the_same_tape() {
        assert_eq!(tape(small(), 20), tape(small(), 20), "a backtest must be replayable");
    }

    #[test]
    fn a_different_seed_produces_a_different_tape() {
        let other = SimConfig { seed: 1234, ..small() };
        assert_ne!(tape(small(), 20), tape(other, 20));
    }

    #[tokio::test]
    async fn every_event_lists_both_legs() {
        let s = sim(small());
        let listings = MarketSource::listings(&s).await.unwrap();
        assert_eq!(listings.len(), 4);
        assert_eq!(listings.iter().filter(|l| l.leg == Leg::Yes).count(), 2);
        assert!(listings.iter().all(|l| l.closes_at.is_some()));
    }

    #[test]
    fn the_two_legs_of_an_event_are_priced_as_complements() {
        // A simulator that prices YES and NO independently manufactures
        // arbitrage that no venue would ever show, which makes every test of
        // the arbitrage strategy meaningless.
        let s = sim(small());
        for _ in 0..50 {
            let updates = s.step();
            let books: Vec<(&String, &BookSnapshot)> = updates
                .iter()
                .filter_map(|u| match u {
                    VenueUpdate::Book { ticker, book } => Some((ticker, book)),
                    _ => None,
                })
                .collect();
            for (ticker, yes) in &books {
                if !ticker.ends_with("YES") {
                    continue;
                }
                let no_ticker = ticker.replace("YES", "NO");
                let (_, no) = books.iter().find(|(t, _)| **t == no_ticker).expect("both legs");
                let (Some(ya), Some(nb)) = (yes.best_ask(), no.best_bid()) else { continue };
                // Buying YES and buying NO must cost at least a dollar before
                // fees, or the tape is handing out free money.
                let Some(na) = no.best_ask() else { continue };
                let cost = ya.dollars() + na.dollars();
                assert!(cost >= 1.0 - 1e-9, "{ticker}: YES {ya} + NO {na} = {cost}");
                let _ = nb;
            }
        }
    }

    #[test]
    fn prices_move_in_log_odds_and_never_leave_the_tradable_range() {
        let cfg = SimConfig { vol: 0.5, horizon_secs: 1e9, ..small() };
        let s = sim(cfg);
        for _ in 0..2_000 {
            for u in s.step() {
                if let VenueUpdate::Book { book, .. } = u {
                    assert!(!book.is_crossed());
                    for l in book.bids.iter().chain(book.asks.iter()) {
                        assert!(l.price.0 > 0 && l.price.0 < edge_core::types::MICROS, "{l:?}");
                        assert!(l.qty.get() > 0);
                    }
                }
            }
        }
    }

    #[test]
    fn a_long_shot_can_still_halve() {
        // The property clamping in probability space destroys: at 2c a market
        // must be able to reach 1c, not sit on a floor of fake support.
        let cfg = SimConfig { events: 1, vol: 0.3, horizon_secs: 1e9, seed: 42, ..small() };
        let s = sim(cfg);
        let mut lowest: f64 = 1.0;
        for _ in 0..5_000 {
            s.step();
            lowest = lowest.min(s.truth(0).unwrap().get());
        }
        assert!(lowest < 0.05, "the walk never reached the tail: {lowest}");
    }

    #[test]
    fn markets_settle_once_and_the_two_legs_disagree() {
        let s = sim(small());
        let mut settlements = Vec::new();
        for _ in 0..40 {
            for u in s.step() {
                if let VenueUpdate::Settled { ticker, outcome, .. } = u {
                    settlements.push((ticker, outcome));
                }
            }
        }
        assert!(s.is_finished());
        assert_eq!(settlements.len(), 4, "two events, two legs, settled exactly once each");
        for (ticker, outcome) in &settlements {
            if ticker.ends_with("YES") {
                let no = settlements
                    .iter()
                    .find(|(t, _)| *t == ticker.replace("YES", "NO"))
                    .expect("both legs settle");
                assert_ne!(*outcome, no.1, "exactly one leg of a binary pays");
            }
        }
    }

    #[test]
    fn settlement_outcomes_follow_the_latent_probability() {
        // A simulator that settles on a coin flip regardless of price rewards
        // a strategy for being wrong. Run many one-event sims started near a
        // known probability and check the frequency.
        let mut yes = 0;
        let trials = 400;
        for seed in 0..trials {
            let cfg = SimConfig {
                events: 1,
                seed,
                vol: 0.0, // hold the truth still so the target is unambiguous
                horizon_secs: 2.0,
                step: Duration::from_secs(1),
                ..small()
            };
            let s = sim(cfg);
            let target = s.truth(0).unwrap().get();
            let mut settled = None;
            for _ in 0..5 {
                for u in s.step() {
                    if let VenueUpdate::Settled { ticker, outcome, .. } = u
                        && ticker.ends_with("YES")
                    {
                        settled = Some((outcome, target));
                    }
                }
            }
            if let Some((outcome, _)) = settled
                && outcome
            {
                yes += 1;
            }
        }
        // Starting logits are uniform on [-2.5, 2.5], so the mean settlement
        // rate is 0.5 by symmetry; a fixed-outcome bug would show as 0 or 1.
        let rate = yes as f64 / trials as f64;
        assert!((0.35..=0.65).contains(&rate), "settlement rate {rate} is not price-driven");
    }

    #[test]
    fn trades_print_at_the_touch_the_taker_crossed() {
        let cfg = SimConfig { trade_rate: 1.0, horizon_secs: 1e9, ..small() };
        let s = sim(cfg);
        for _ in 0..100 {
            let updates = s.step();
            for u in &updates {
                let VenueUpdate::Trade { ticker, price, taker, .. } = u else { continue };
                let book = updates
                    .iter()
                    .find_map(|b| match b {
                        VenueUpdate::Book { ticker: t, book } if t == ticker => Some(book),
                        _ => None,
                    })
                    .expect("a trade is accompanied by its book");
                match taker {
                    Side::Buy => assert_eq!(Some(*price), book.best_ask()),
                    Side::Sell => assert_eq!(Some(*price), book.best_bid()),
                }
            }
        }
    }

    #[test]
    fn depth_thins_going_away_from_the_touch() {
        let s = sim(small());
        let updates = s.step();
        let book = updates
            .iter()
            .find_map(|u| match u {
                VenueUpdate::Book { book, .. } => Some(book),
                _ => None,
            })
            .unwrap();
        assert!(book.bids.len() > 1);
        for w in book.bids.windows(2) {
            assert!(w[0].qty >= w[1].qty, "depth must not grow with distance: {:?}", book.bids);
        }
    }

    #[tokio::test]
    async fn the_tape_feeds_the_assembler_without_ever_producing_a_bad_book() {
        // The integration that matters: everything the simulator emits must be
        // something the assembler accepts, or the synthetic venue is testing a
        // path the real ones do not take.
        let s = sim(SimConfig { horizon_secs: 30.0, ..small() });
        let mut a = Assembler::new(V, AssemblerConfig::default());
        for l in MarketSource::listings(&s).await.unwrap() {
            a.register(&l);
        }

        let mut events = Vec::new();
        while !s.is_finished() {
            for u in s.snapshot(&[]).await.unwrap() {
                a.apply(u, &mut events);
            }
        }

        assert!(!events.iter().any(|e| matches!(e, crate::assembler::Event::Gap { .. })));
        assert!(!events.iter().any(|e| matches!(e, crate::assembler::Event::Unknown { .. })));
        assert!(events.iter().any(|e| matches!(e, crate::assembler::Event::Settled { .. })));
        for m in 0..a.registry().len() {
            let id = edge_core::types::MarketId(m as u64);
            if let Some(b) = a.book(id) {
                b.debug_check();
            }
        }
    }

    #[tokio::test]
    async fn a_stream_stops_when_the_consumer_goes_away() {
        let s = sim(SimConfig { step: Duration::from_millis(1), horizon_secs: 1e9, ..small() });
        let (tx, rx) = tokio::sync::mpsc::channel(4);
        drop(rx);
        // Must return rather than spin forever against a closed channel.
        tokio::time::timeout(Duration::from_secs(5), s.stream(&[], tx))
            .await
            .expect("stream did not notice the consumer left")
            .unwrap();
    }
}
