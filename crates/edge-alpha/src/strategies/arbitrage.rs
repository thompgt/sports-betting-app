//! Locking a profit that does not depend on any forecast being right.
//!
//! Every other strategy in this crate is a bet that some estimate is better
//! than the market's. This one is not: if the YES leg can be bought somewhere
//! for 46c and the NO leg somewhere else for 51c, the pair settles at exactly
//! $1 whichever way the event resolves, and 3c is earned without any opinion
//! about the event at all. It is the highest-conviction opportunity the system
//! can find, and the only one whose expected value does not depend on a model.
//!
//! What it *does* depend on is both fills landing. That risk is real and it is
//! the entire reason this is harder than the arithmetic suggests, so the
//! implementation is built around it:
//!
//! **Size is the minimum across legs, never the maximum.** A 500-contract YES
//! offer against a 40-contract NO offer is a 40-contract arbitrage. Sizing on
//! the larger leg leaves 460 contracts of naked directional exposure acquired
//! at a price nobody thought was fair.
//!
//! **Both legs are IOC, and the margin must cover a leg missing.** The profit
//! floor is set well above zero specifically so that a fill on one leg and a
//! miss on the other is survivable — a one-cent arbitrage that legs out is a
//! loss much larger than one cent.
//!
//! **Fees are charged on every leg before anything is called an arbitrage.**
//! Almost every apparent cross-venue arbitrage in prediction markets is a fee
//! schedule that has not been subtracted yet.
//!
//! Two shapes are detected. *Complement*: a YES market and a NO market on the
//! same event, whose asks sum to less than a dollar. *Multi-outcome*: a set of
//! mutually exclusive markets covering every outcome, whose asks sum to less
//! than a dollar — the same trade with more legs, which is where a stale leg in
//! a three-way market usually hides.

use edge_core::ev;
use edge_core::fees::Liquidity;
use edge_core::types::{Leg, MarketId, Price, Qty, Side, StrategyId};
use serde::{Deserialize, Serialize};

use crate::strategy::{Action, EventView, MarketView, OrderIntent, StatsRecorder, Strategy, StrategyStats};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ArbConfig {
    /// Minimum guaranteed profit per matched contract, in dollars, after all
    /// fees. Not a preference — the margin has to be wide enough that legging
    /// out on one side is survivable, which a one-tick arbitrage never is.
    pub min_profit: f64,
    /// Never take more than this share of the quantity resting on any leg.
    pub max_touch_share: f64,
    /// Cap on contracts per arbitrage, however much is offered.
    pub max_size: i64,
    /// Cap on total capital committed to a single arbitrage, in dollars.
    pub max_notional: f64,
    /// Refuse inside this many seconds of resolution: legging risk rises
    /// sharply when books thin out at the close.
    pub min_seconds_left: f64,
}

impl Default for ArbConfig {
    fn default() -> Self {
        ArbConfig {
            min_profit: 0.01,
            max_touch_share: 1.0,
            max_size: 1_000,
            max_notional: 1_000.0,
            min_seconds_left: 60.0,
        }
    }
}

impl ArbConfig {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.min_profit <= 0.0 {
            return Err("a zero profit floor makes legging risk unbounded relative to the edge");
        }
        if !(0.0..=1.0).contains(&self.max_touch_share) {
            return Err("touch share must be a fraction");
        }
        Ok(())
    }
}

/// One leg of a detected arbitrage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArbLeg {
    pub market: MarketId,
    pub side: Side,
    pub price: Price,
    pub qty: Qty,
}

/// A complete, executable arbitrage.
#[derive(Debug, Clone, PartialEq)]
pub struct ArbOpportunity {
    pub legs: Vec<ArbLeg>,
    /// Total all-in cost per matched set, in dollars. Below $1 is what makes it
    /// an arbitrage.
    pub cost: f64,
    /// Guaranteed profit per matched set, after fees.
    pub profit_per_set: f64,
    pub qty: Qty,
}

impl ArbOpportunity {
    pub fn total_profit(&self) -> f64 {
        self.profit_per_set * self.qty.get() as f64
    }
}

#[derive(Debug, Clone)]
pub struct Arbitrage {
    id: StrategyId,
    cfg: ArbConfig,
    stats: StatsRecorder,
}

impl Arbitrage {
    pub fn new(id: StrategyId, cfg: ArbConfig) -> Self {
        Arbitrage {
            id,
            cfg,
            stats: StatsRecorder::default(),
        }
    }

    pub fn config(&self) -> &ArbConfig {
        &self.cfg
    }

    /// Cheapest way to acquire a contract paying $1 if this market's leg wins,
    /// with the quantity available at that price.
    ///
    /// A YES payout is bought by buying the YES market's ask; it is *also*
    /// bought by selling a NO market — which on a binary book is the same
    /// position. Only the direct purchase is used here, because the sell side
    /// of a leg carries different fees and different resting depth and
    /// conflating them is how a phantom arbitrage appears.
    fn cover(view: &MarketView<'_>, qty: Qty) -> Option<(Price, f64, Qty)> {
        let ask = view.book.best_ask()?;
        if !ask.is_tradable() {
            return None;
        }
        let fee = view.spec.fee.fee_per_contract(ask, qty, Liquidity::Taker);
        Some((ask, ask.dollars() + fee, view.book.qty_at(Side::Sell, ask)))
    }

    /// Find the best arbitrage across an event, if there is one.
    pub fn scan(&self, event: &EventView<'_>) -> Option<ArbOpportunity> {
        if !event.all_tradable() {
            return None;
        }
        if event
            .markets
            .iter()
            .any(|m| m.time_left() < self.cfg.min_seconds_left)
        {
            return None;
        }

        // Group the event's markets into an exhaustive outcome cover. A binary
        // event is covered by one YES and one NO; a multi-outcome event by one
        // market per outcome, taking the cheapest quote for each.
        let candidates = self.cover_set(event)?;

        // Probe at a nominal lot to learn the prices, then size, then re-price:
        // a fee schedule rounded per order is not linear in size.
        let probe = Qty(10);
        let mut legs = Vec::with_capacity(candidates.len());
        let mut costs = Vec::with_capacity(candidates.len());
        let mut available = i64::MAX;

        for m in &candidates {
            let (price, cost, depth) = Self::cover(m, probe)?;
            let usable = (depth.get() as f64 * self.cfg.max_touch_share).floor() as i64;
            available = available.min(usable);
            legs.push((m.spec.id, price));
            costs.push(cost);
        }

        ev::arbitrage(&costs)?;

        // Size on the *thinnest* leg. Anything more is naked exposure.
        let by_notional = (self.cfg.max_notional / costs.iter().sum::<f64>().max(1e-9)).floor() as i64;
        let qty = available.min(self.cfg.max_size).min(by_notional).max(0);
        if qty <= 0 {
            return None;
        }

        // Re-price at the size actually traded and re-check the floor.
        let mut sized_cost = 0.0;
        for (m, (_, price)) in candidates.iter().zip(legs.iter()) {
            let fee = m.spec.fee.fee_per_contract(*price, Qty(qty), Liquidity::Taker);
            sized_cost += price.dollars() + fee;
        }
        let profit = 1.0 - sized_cost;
        if profit < self.cfg.min_profit {
            return None;
        }

        Some(ArbOpportunity {
            legs: legs
                .into_iter()
                .map(|(market, price)| ArbLeg {
                    market,
                    side: Side::Buy,
                    price,
                    qty: Qty(qty),
                })
                .collect(),
            cost: sized_cost,
            profit_per_set: profit,
            qty: Qty(qty),
        })
    }

    /// Pick one market per outcome, cheapest first.
    ///
    /// Returns `None` unless the selection is genuinely exhaustive — a partial
    /// cover is not an arbitrage, it is a directional position that happens to
    /// look cheap.
    ///
    /// The unit of exhaustiveness is the **outcome**, not the event. Pairing the
    /// cheapest YES anywhere on an event against the cheapest NO anywhere is
    /// only sound when the event has a single outcome: on a three-way event
    /// YES("A wins") and NO("B wins") both lose when C wins, so the pair pays
    /// $0 and the "arbitrage" is a phantom. Both legs must therefore agree on
    /// `spec.outcome`.
    fn cover_set<'a>(&self, event: &'a EventView<'a>) -> Option<Vec<&'a MarketView<'a>>> {
        let cheapest = |leg: Leg, outcome: &str| -> Option<&'a MarketView<'a>> {
            event
                .leg(leg)
                .filter(|m| m.spec.outcome == outcome && m.book.best_ask().is_some())
                .min_by(|a, b| {
                    let x = a.book.best_ask().unwrap().micros();
                    let y = b.book.best_ask().unwrap().micros();
                    x.cmp(&y)
                })
        };

        // Distinct outcomes on this event, in first-seen order so the choice is
        // deterministic.
        let mut outcomes: Vec<&str> = Vec::new();
        for m in event.markets {
            if !outcomes.contains(&m.spec.outcome.as_str()) {
                outcomes.push(&m.spec.outcome);
            }
        }

        // Complement: YES and NO on the SAME outcome. The two legs need not be,
        // and usually are not, on the same venue — that is where cross-venue
        // arbitrage falls out. Take whichever outcome gives the cheapest pair.
        let mut best: Option<(i64, Vec<&'a MarketView<'a>>)> = None;
        for outcome in &outcomes {
            if let (Some(yes), Some(no)) = (cheapest(Leg::Yes, outcome), cheapest(Leg::No, outcome))
            {
                let cost =
                    yes.book.best_ask().unwrap().micros() + no.book.best_ask().unwrap().micros();
                if best.as_ref().is_none_or(|(b, _)| cost < *b) {
                    best = Some((cost, vec![yes, no]));
                }
            }
        }
        if let Some((_, legs)) = best {
            return Some(legs);
        }

        // Multi-outcome: one YES per distinct outcome, every outcome present.
        // Fewer than two outcomes cannot cover anything. Note this takes the
        // cheapest quote per outcome rather than every market on the event —
        // two venues listing the same outcome are one leg, not two, and
        // counting both inflates the cost of a real cover into a miss.
        if outcomes.len() >= 2 && event.markets.iter().all(|m| m.spec.leg == Leg::Yes) {
            let legs: Vec<_> = outcomes
                .iter()
                .filter_map(|o| cheapest(Leg::Yes, o))
                .collect();
            if legs.len() == outcomes.len() {
                return Some(legs);
            }
        }
        None
    }
}

impl Strategy for Arbitrage {
    fn id(&self) -> StrategyId {
        self.id
    }

    fn name(&self) -> &'static str {
        "arbitrage"
    }

    /// Nothing. Arbitrage is not visible one market at a time.
    fn on_market(&mut self, _view: &MarketView<'_>, _out: &mut Vec<Action>) {}

    fn on_event(&mut self, event: &EventView<'_>, out: &mut Vec<Action>) {
        let start = out.len();
        if let Some(opp) = self.scan(event) {
            for leg in &opp.legs {
                // IOC on every leg. A resting arbitrage leg is a free option
                // written to whoever moves the other market first.
                out.push(Action::Place(OrderIntent::take(
                    leg.market,
                    leg.side,
                    leg.price,
                    leg.qty,
                    "arbitrage",
                )));
            }
        }
        self.stats.record(&out[start..]);
    }

    fn on_fill(&mut self, fill: &edge_book::order::Fill, is_maker: bool) {
        self.stats.record_fill(fill, is_maker);
    }

    fn stats(&self) -> StrategyStats {
        self.stats.stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategy::harness::*;
    use edge_book::order::TimeInForce;
    use edge_core::types::{EventId, Ts, VenueId};

    const EVENT: EventId = EventId(1);

    /// One market on the shared event, with its own book and leg.
    struct Venue {
        sim: Sim,
    }

    impl Venue {
        /// A market on the event's single default outcome. Venues differ, the
        /// outcome does not — which is what makes a YES here and a NO there
        /// genuine complements.
        fn new(id: u64, venue: u16, leg: Leg, ask: i64, depth: i64) -> Self {
            Self::with_outcome(id, venue, leg, ask, depth, "OUTCOME")
        }

        /// A market on a named outcome, for events with more than one.
        fn with_outcome(
            id: u64,
            venue: u16,
            leg: Leg,
            ask: i64,
            depth: i64,
            outcome: &str,
        ) -> Self {
            let market = MarketId(id);
            let mut sim = Sim::for_market(market, EVENT, VenueId(venue));
            sim.spec.ticker = format!("M{id}");
            sim.spec.outcome = outcome.to_string();
            sim.spec.leg = leg;
            // A bid too, so the market counts as tradable.
            sim.rest(Side::Buy, (ask - 3).max(1), 100);
            sim.rest(Side::Sell, ask, depth);
            Venue { sim }
        }
    }

    fn scan(venues: &[Venue], cfg: ArbConfig) -> Option<ArbOpportunity> {
        let views: Vec<_> = venues.iter().map(|v| v.sim.plain()).collect();
        let ev = EventView {
            event: EVENT,
            markets: &views,
            bankroll: 10_000.0,
            now: Ts(0),
        };
        Arbitrage::new(StrategyId(5), cfg).scan(&ev)
    }

    #[test]
    fn the_default_configuration_is_coherent() {
        ArbConfig::default().validate().unwrap();
    }

    #[test]
    fn a_zero_profit_floor_is_rejected() {
        assert!(
            ArbConfig {
                min_profit: 0.0,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn yes_and_no_summing_below_a_dollar_is_bought_on_both_legs() {
        // 46c + 51c = 97c for a pair that settles at $1.
        let v = [
            Venue::new(1, 1, Leg::Yes, 46, 200),
            Venue::new(2, 2, Leg::No, 51, 200),
        ];
        let opp = scan(&v, ArbConfig::default()).expect("3c of arbitrage");
        assert_eq!(opp.legs.len(), 2);
        assert!(opp.legs.iter().all(|l| l.side == Side::Buy));
        assert!((opp.profit_per_set - 0.03).abs() < 1e-9, "{}", opp.profit_per_set);
        assert!((opp.cost - 0.97).abs() < 1e-9);
    }

    #[test]
    fn legs_are_sized_on_the_thinnest_side() {
        // 500 offered on one leg, 40 on the other: this is a 40-lot arbitrage.
        // Sizing on 500 would leave 460 contracts of naked exposure.
        let v = [
            Venue::new(1, 1, Leg::Yes, 46, 500),
            Venue::new(2, 2, Leg::No, 51, 40),
        ];
        let opp = scan(&v, ArbConfig::default()).unwrap();
        assert_eq!(opp.qty, Qty(40));
        assert!(opp.legs.iter().all(|l| l.qty == Qty(40)));
    }

    #[test]
    fn a_pair_summing_above_a_dollar_is_not_an_arbitrage() {
        let v = [
            Venue::new(1, 1, Leg::Yes, 52, 200),
            Venue::new(2, 2, Leg::No, 51, 200),
        ];
        assert!(scan(&v, ArbConfig::default()).is_none());
    }

    #[test]
    fn a_margin_too_thin_to_survive_legging_out_is_declined() {
        // 99c: one cent of profit. Real, and whether it is worth the risk of
        // being left holding one leg is exactly what the floor decides.
        let v = [
            Venue::new(1, 1, Leg::Yes, 48, 200),
            Venue::new(2, 2, Leg::No, 51, 200),
        ];
        assert!(
            scan(
                &v,
                ArbConfig {
                    min_profit: 0.02,
                    ..Default::default()
                }
            )
            .is_none(),
            "a two-cent floor should reject a one-cent arbitrage"
        );
        assert!(
            scan(&v, ArbConfig::default()).is_some(),
            "and the one-cent default should admit it"
        );
    }

    #[test]
    fn fees_are_subtracted_before_anything_is_called_an_arbitrage() {
        let mut v = [
            Venue::new(1, 1, Leg::Yes, 46, 200),
            Venue::new(2, 2, Leg::No, 51, 200),
        ];
        assert!(scan(&v, ArbConfig::default()).is_some(), "3c gross is there");

        // The same prices under Kalshi's taker schedule: ~1.5c a leg near the
        // middle of the range, and the arbitrage is gone.
        for m in v.iter_mut() {
            m.sim.spec.fee = edge_core::fees::FeeModel::KALSHI_STANDARD;
        }
        assert!(
            scan(&v, ArbConfig::default()).is_none(),
            "a gross arbitrage that fees consume is not an arbitrage"
        );
    }

    #[test]
    fn the_cheapest_venue_is_chosen_for_each_leg() {
        let v = [
            Venue::new(1, 1, Leg::Yes, 55, 200),
            Venue::new(2, 2, Leg::Yes, 46, 200), // cheaper YES
            Venue::new(3, 3, Leg::No, 51, 200),
        ];
        let opp = scan(&v, ArbConfig::default()).unwrap();
        assert!(opp.legs.iter().any(|l| l.market == MarketId(2)));
        assert!(!opp.legs.iter().any(|l| l.market == MarketId(1)));
    }

    #[test]
    fn a_three_way_market_priced_below_a_dollar_is_taken_on_every_leg() {
        let v = [
            Venue::with_outcome(1, 1, Leg::Yes, 30, 200, "A"),
            Venue::with_outcome(2, 1, Leg::Yes, 33, 200, "B"),
            Venue::with_outcome(3, 1, Leg::Yes, 34, 200, "C"),
        ];
        let opp = scan(&v, ArbConfig::default()).expect("97c across three outcomes");
        assert_eq!(opp.legs.len(), 3);
        assert!((opp.profit_per_set - 0.03).abs() < 1e-9);
    }

    #[test]
    fn a_yes_and_a_no_on_different_outcomes_are_not_an_arbitrage() {
        // The phantom. YES("A wins") at 30c and NO("B wins") at 40c sum to 70c
        // and look like 30c of free money, but both legs lose when C wins, so
        // the pair pays $0. Only legs on the SAME outcome are complements.
        let v = [
            Venue::with_outcome(1, 1, Leg::Yes, 30, 200, "A"),
            Venue::with_outcome(2, 2, Leg::No, 40, 200, "B"),
            Venue::with_outcome(3, 3, Leg::Yes, 35, 200, "C"),
        ];
        assert!(
            scan(&v, ArbConfig::default()).is_none(),
            "YES(A) + NO(B) on a three-way event is not a cover"
        );
    }

    #[test]
    fn a_no_leg_does_not_hide_a_genuine_multi_outcome_cover() {
        // Same three-way event, but NO(B) is cheap enough that YES(B) + NO(B)
        // is a real complement pair. That is the arbitrage, not the phantom.
        let v = [
            Venue::with_outcome(1, 1, Leg::Yes, 30, 200, "A"),
            Venue::with_outcome(2, 2, Leg::Yes, 33, 200, "B"),
            Venue::with_outcome(3, 3, Leg::No, 62, 200, "B"),
        ];
        let opp = scan(&v, ArbConfig::default()).expect("33c + 62c = 95c on outcome B");
        assert_eq!(opp.legs.len(), 2);
        assert!(opp.legs.iter().all(|l| l.market != MarketId(1)));
        assert!((opp.profit_per_set - 0.05).abs() < 1e-9);
    }

    #[test]
    fn two_venues_quoting_the_same_outcome_are_one_leg_not_two() {
        // A three-way cover where outcome A is listed on two venues. Counting
        // both would total 30 + 31 + 33 + 34 = $1.28 and miss a real 97c cover.
        let v = [
            Venue::with_outcome(1, 1, Leg::Yes, 30, 200, "A"),
            Venue::with_outcome(2, 2, Leg::Yes, 31, 200, "A"),
            Venue::with_outcome(3, 1, Leg::Yes, 33, 200, "B"),
            Venue::with_outcome(4, 1, Leg::Yes, 34, 200, "C"),
        ];
        let opp = scan(&v, ArbConfig::default()).expect("97c across three outcomes");
        assert_eq!(opp.legs.len(), 3);
        // The cheaper of the two A listings is the one taken.
        assert!(opp.legs.iter().any(|l| l.market == MarketId(1)));
        assert!(opp.legs.iter().all(|l| l.market != MarketId(2)));
    }

    #[test]
    fn a_partial_cover_is_never_traded() {
        // One leg of a binary event on its own is a directional bet, however
        // cheap it looks.
        let v = [Venue::new(1, 1, Leg::Yes, 20, 200)];
        assert!(scan(&v, ArbConfig::default()).is_none());
    }

    #[test]
    fn a_halted_leg_kills_the_whole_arbitrage() {
        let mut v = [
            Venue::new(1, 1, Leg::Yes, 46, 200),
            Venue::new(2, 2, Leg::No, 51, 200),
        ];
        v[1].sim.spec.status = edge_core::market::MarketStatus::Halted;
        assert!(scan(&v, ArbConfig::default()).is_none());
    }

    #[test]
    fn notional_caps_the_size_even_when_depth_allows_more() {
        let v = [
            Venue::new(1, 1, Leg::Yes, 46, 100_000),
            Venue::new(2, 2, Leg::No, 51, 100_000),
        ];
        let opp = scan(
            &v,
            ArbConfig {
                max_notional: 97.0,
                max_size: 1_000_000,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(opp.qty, Qty(100), "$97 of capital at 97c a set");
    }

    #[test]
    fn nothing_is_attempted_close_to_resolution() {
        let mut v = [
            Venue::new(1, 1, Leg::Yes, 46, 200),
            Venue::new(2, 2, Leg::No, 51, 200),
        ];
        v[0].sim.spec.closes_at = Some(Ts(30_000_000_000));
        assert!(scan(&v, ArbConfig::default()).is_none());
    }

    #[test]
    fn every_leg_is_sent_immediate_or_cancel() {
        let v = [
            Venue::new(1, 1, Leg::Yes, 46, 200),
            Venue::new(2, 2, Leg::No, 51, 200),
        ];
        let views: Vec<_> = v.iter().map(|x| x.sim.plain()).collect();
        let ev = EventView {
            event: EVENT,
            markets: &views,
            bankroll: 10_000.0,
            now: Ts(0),
        };
        let mut a = Arbitrage::new(StrategyId(5), ArbConfig::default());
        let mut out = Vec::new();
        a.on_event(&ev, &mut out);
        assert_eq!(out.len(), 2);
        for i in places(&out) {
            assert_eq!(i.tif, TimeInForce::Ioc, "a resting arb leg is a free option");
            assert_eq!(&*i.reason, "arbitrage");
        }
    }

    #[test]
    fn per_market_ticks_produce_nothing() {
        let v = Venue::new(1, 1, Leg::Yes, 46, 200);
        let mut a = Arbitrage::new(StrategyId(5), ArbConfig::default());
        assert!(v.sim.step(&mut a, &v.sim.plain()).is_empty());
    }
}
