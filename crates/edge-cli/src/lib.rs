//! The pipeline, as a thing that can be driven.
//!
//! Every crate in this workspace is a library, and each is tested thoroughly on
//! its own. That leaves the seams — the assembler's view of a market meeting the
//! strategy's, meeting the risk engine's — tested nowhere, which is where the
//! expensive bugs live.
//!
//! So the driver lives here rather than in `main.rs`, and it is generic over the
//! source of updates. The binary feeds it the deterministic simulator; the
//! integration tests feed it recorded Kalshi payloads through the real adapter.
//! Both run this exact code, which is the only reason the tests are worth
//! anything.
//!
//! ```text
//! MarketSource → Assembler → MarketView → Strategy → RiskEngine → fills
//! ```

#![forbid(unsafe_code)]

use edge_alpha::strategy::MarketView;
use edge_alpha::{QuoteConfig, QuoteMaker, Strategy};
use edge_core::fees::Liquidity;
use edge_core::types::{MarketId, Notional, Price, Qty, Side, StrategyId, Ts, VenueId};
use edge_data::assembler::{Assembler, AssemblerConfig, Event};
use edge_data::source::{Listing, VenueUpdate};
use edge_risk::engine::RiskEngine;
use edge_risk::limits::{RiskDecision, RiskLimits};

pub const MAKER: StrategyId = StrategyId(1);

/// An order the pipeline actually sent, after risk had its say.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Placed {
    pub market: MarketId,
    pub side: Side,
    pub price: Price,
    pub qty: Qty,
    pub ts: Ts,
}

/// What a session did, in counts.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Tally {
    pub intents: u64,
    pub approved: u64,
    pub resized: u64,
    pub rejected: u64,
    pub gaps: u64,
    pub settled: u64,
    /// Markets skipped this tick because their book could not be believed.
    pub skipped_stale: u64,
}

impl Tally {
    fn record(&mut self, d: &RiskDecision) {
        match d {
            RiskDecision::Approve(_) => self.approved += 1,
            RiskDecision::Resize(..) => self.resized += 1,
            RiskDecision::Reject(_) => self.rejected += 1,
        }
    }
}

/// Assembler, strategy and risk engine wired together.
#[derive(Debug)]
pub struct Pipeline {
    pub assembler: Assembler,
    pub maker: QuoteMaker,
    pub risk: RiskEngine,
    pub tally: Tally,
    /// Every order sent, in order. The integration tests assert on this: what
    /// matters is not that the pipeline ran but that it declined to trade the
    /// books it should not have.
    pub placed: Vec<Placed>,
    events: Vec<Event>,
    actions: Vec<edge_alpha::Action>,
}

impl Pipeline {
    pub fn new(venue: VenueId, bankroll: Notional) -> Self {
        Pipeline {
            assembler: Assembler::new(venue, AssemblerConfig::default()),
            maker: QuoteMaker::new(MAKER, QuoteConfig::default()),
            risk: RiskEngine::new(RiskLimits::for_bankroll(bankroll), bankroll),
            tally: Tally::default(),
            placed: Vec::new(),
            events: Vec::new(),
            actions: Vec::new(),
        }
    }

    /// Intern the catalogue.
    ///
    /// Nothing at all happens without this: an update naming a ticker with no
    /// listing is dropped rather than guessed at.
    pub fn register(&mut self, listings: &[Listing]) {
        for l in listings {
            let (market, _) = self.assembler.register(l);
            if let Some(spec) = self.assembler.registry().get(market) {
                let event = spec.event_id;
                self.risk.portfolio_mut().set_event(market, event);
            }
        }
    }

    /// One poll cycle: absorb `updates`, then quote whatever is still tradable.
    pub fn tick(&mut self, updates: Vec<VenueUpdate>, now: Ts) {
        self.events.clear();
        for u in updates {
            self.assembler.apply(u, &mut self.events);
        }

        for e in std::mem::take(&mut self.events) {
            match e {
                Event::Gap { .. } => self.tally.gaps += 1,
                Event::Settled { market, outcome, .. } => {
                    self.risk.on_settle(market, outcome);
                    self.maker.on_settle(market, outcome);
                    self.tally.settled += 1;
                }
                _ => {}
            }
        }

        let fresh = self.assembler.fresh_markets(now);
        self.tally.skipped_stale += (self.assembler.registry().len() - fresh.len()) as u64;

        for market in fresh {
            self.quote(market, now);
        }

        self.risk.update(now);
    }

    fn quote(&mut self, market: MarketId, now: Ts) {
        let Some(spec) = self.assembler.registry().get(market) else {
            return;
        };
        if !spec.status.is_tradable() {
            return;
        }
        let Some(book) = self.assembler.book(market) else {
            return;
        };
        // A one-sided book has no mid: nothing to mark and nothing to quote
        // around. Inventing one is how a maker ends up quoting into a market
        // that has no other side.
        let Some(mid) = book.mid() else {
            return;
        };

        // Mark before sizing. The risk engine refuses to size an unmarked
        // market on purpose — an absent or stale price is the most common cause
        // of a large accidental position.
        self.risk.set_mark(market, mid, now);

        let position = self.risk.portfolio().qty(market);
        let avg_cost =
            self.risk.portfolio().position(market).map(|p| p.avg_cost().dollars()).unwrap_or(0.0);
        let bankroll = self.risk.portfolio().cash.dollars();

        let view = MarketView {
            spec,
            book,
            // No model and no cross-venue consensus here, so the maker quotes
            // around the market's own mid. Deliberate: a predictor with no
            // demonstrated skill must not move a price, and wiring one in to
            // make the output livelier would stage the exact failure edge-alpha
            // is built to prevent.
            features: None,
            prediction: None,
            consensus: None,
            position,
            avg_cost,
            bankroll,
            resting: &[],
            now,
        };

        let mut actions = std::mem::take(&mut self.actions);
        actions.clear();
        self.maker.on_market(&view, &mut actions);
        self.tally.intents += actions.len() as u64;

        let fee_model = spec.fee;
        for action in &actions {
            let Some(intent) = action.intent() else {
                continue;
            };
            let per_contract = Notional::from_dollars(fee_model.fee_per_contract(
                intent.price,
                intent.qty,
                Liquidity::Maker,
            ));
            let decision =
                self.risk.check(market, intent.side, intent.price, intent.qty, per_contract, now);
            self.tally.record(&decision);

            let filled = decision.qty();
            if filled.get() > 0 {
                // Assumed filled at the quoted price. This is the line where a
                // real system would talk to a venue and wait to be told.
                let fee = Notional(per_contract.0 * filled.get());
                self.risk.on_fill(market, intent.side, intent.price, filled, fee, now);
                self.placed.push(Placed {
                    market,
                    side: intent.side,
                    price: intent.price,
                    qty: filled,
                    ts: now,
                });
            }
        }
        self.actions = actions;
    }

    /// Does the ledger close? Cash plus what the open positions cost must equal
    /// the starting bankroll plus everything realised — exactly, because none
    /// of the accounting is approximate.
    pub fn reconciles(&self) -> bool {
        let pf = self.risk.portfolio();
        pf.cash + pf.capital_at_risk() - pf.realized() == pf.starting_cash
    }
}
