//! Two-sided market making with inventory skew.
//!
//! The maker's problem is not "what is this worth" — it is "what is this worth
//! *given that someone wants to trade against me*". Three mechanisms handle
//! that here, and each of them exists because its absence is a known way to lose
//! money quoting:
//!
//! **Inventory skew.** The quotes are centred not on fair value but on a
//! *reservation price* shifted against the current position. Long inventory
//! lowers both sides, so the next trade is more likely to reduce the position
//! than grow it. A maker quoting symmetrically around fair accumulates whichever
//! side the market is trending away from and is, in effect, running a momentum
//! strategy with a negative expected return.
//!
//! **A spread floor that covers fees.** The half-spread never goes below the
//! round-trip fee plus a margin. Quoting inside your own cost base is the
//! purest form of paying for volume.
//!
//! **Queue preservation.** An existing quote within tolerance of the desired
//! price is left alone rather than cancelled and reposted one tick better.
//! Queue position is most of the value of a resting order; churning it away for
//! a marginal price improvement is a net loss on almost any book.
//!
//! Additionally the maker widens as resolution approaches and stops quoting the
//! side that would grow an already-maximal position — both cases where the
//! adverse selection a maker faces rises sharply.

use edge_core::fees::Liquidity;
use edge_core::types::{Price, Prob, Qty, Side, StrategyId};
use serde::{Deserialize, Serialize};

use crate::strategy::{Action, MarketView, OrderIntent, StatsRecorder, Strategy, StrategyStats};

#[cfg(test)]
use edge_book::order::TimeInForce;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct QuoteConfig {
    /// Contracts per quote at flat inventory.
    pub size: i64,
    /// Position at which the maker stops adding on that side entirely. Skew
    /// scales against this, so it sets the aggressiveness of the lean as well as
    /// the hard stop.
    pub max_inventory: i64,
    /// Half-spread floor, in probability. The maker never quotes tighter than
    /// this regardless of what the model says.
    pub min_half_spread: f64,
    /// Half-spread ceiling. Beyond this a quote is so far from the touch that it
    /// only ever fills on a dislocation — which is exactly when it should not.
    pub max_half_spread: f64,
    /// Multiple of realised volatility added to the half-spread.
    pub vol_multiple: f64,
    /// Maximum shift of the reservation price at full inventory, in probability.
    pub max_skew: f64,
    /// How far an existing quote may drift from the desired price before it is
    /// worth giving up queue position to reprice, in ticks.
    pub requote_ticks: i64,
    /// Stop quoting entirely inside this many seconds of resolution. Late in a
    /// market, informed flow dominates and a maker is mostly a counterparty to
    /// people who know something.
    pub stop_quoting_secs: f64,
    /// Cancel a quote that has rested longer than this, in seconds, even if its
    /// price is still good. A very old quote usually means the market has moved
    /// on and this one survived only because nobody wanted it.
    pub max_quote_age_secs: f64,
}

impl Default for QuoteConfig {
    fn default() -> Self {
        QuoteConfig {
            size: 25,
            max_inventory: 250,
            min_half_spread: 0.01,
            max_half_spread: 0.10,
            vol_multiple: 1.5,
            max_skew: 0.04,
            requote_ticks: 1,
            stop_quoting_secs: 300.0,
            max_quote_age_secs: 120.0,
        }
    }
}

/// One side's desired quote, exposed so the behaviour can be tested and
/// inspected without going through the intent machinery.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DesiredQuote {
    pub side: Side,
    pub price: Price,
    pub qty: Qty,
}

#[derive(Debug, Clone)]
pub struct QuoteMaker {
    id: StrategyId,
    cfg: QuoteConfig,
    stats: StatsRecorder,
}

impl QuoteMaker {
    pub fn new(id: StrategyId, cfg: QuoteConfig) -> Self {
        QuoteMaker {
            id,
            cfg,
            stats: StatsRecorder::default(),
        }
    }

    pub fn config(&self) -> &QuoteConfig {
        &self.cfg
    }

    /// Reservation price: fair value shifted against inventory.
    ///
    /// Computed in probability rather than log-odds because the skew is a
    /// statement about *capital at risk per contract*, which is linear in price.
    pub fn reservation(&self, fair: Prob, position: i64) -> f64 {
        let inv = (position as f64 / self.cfg.max_inventory.max(1) as f64).clamp(-1.0, 1.0);
        (fair.get() - inv * self.cfg.max_skew).clamp(0.001, 0.999)
    }

    /// Half-spread: a fee-covering floor, widened by volatility and by proximity
    /// to resolution.
    pub fn half_spread(&self, view: &MarketView<'_>, fair: Prob) -> f64 {
        let vol = view
            .features
            .and_then(|f| f.get("volatility"))
            .filter(|v| v.is_finite())
            .unwrap_or(0.0);
        // Feature volatility is in log-odds; convert to probability at the
        // current level, where the local derivative is p(1−p).
        let vol_prob = vol * fair.get() * (1.0 - fair.get());

        // Round-trip fees, sized on a nominal lot so a per-order rounding rule
        // is not amortised over an unrealistically large one. The round trip is
        // *maker in, taker out* — a maker gets filled passively but almost
        // always exits by crossing, and a floor built on the maker rate alone
        // (zero, on Kalshi) would quote a spread that cannot pay for its own
        // exit.
        let lot = Qty(self.cfg.size.max(1));
        let mid = Price::from_dollars(fair.get()).unwrap_or(Price::from_cents(50));
        let round_trip = view.spec.fee.fee_per_contract(mid, lot, Liquidity::Maker)
            + view.spec.fee.fee_per_contract(mid, lot, Liquidity::Taker);

        // The full spread is twice the half, so half the round trip is the
        // break-even half-spread.
        let base = self.cfg.min_half_spread.max(round_trip / 2.0) + self.cfg.vol_multiple * vol_prob;

        // Widen smoothly over the final hour rather than at a cliff, so a
        // market crossing the threshold does not produce a jump in quotes.
        let urgency = (3_600.0 / view.time_left().max(1.0)).min(4.0);
        (base * urgency.max(1.0)).clamp(self.cfg.min_half_spread, self.cfg.max_half_spread)
    }

    /// The quotes the maker wants right now, before reconciling with what it
    /// already has resting.
    pub fn desired(&self, view: &MarketView<'_>) -> Vec<DesiredQuote> {
        let mut out = Vec::new();
        if !view.is_tradable() || view.time_left() < self.cfg.stop_quoting_secs {
            return out;
        }
        let Some(fair) = view.fair() else { return out };
        let (Some(best_bid), Some(best_ask)) = (view.book.best_bid(), view.book.best_ask()) else {
            return out;
        };

        let r = self.reservation(fair, view.position.get());
        let half = self.half_spread(view, fair);
        let tick = view.spec.tick_size;
        let pos = view.position.get();
        let max_inv = self.cfg.max_inventory;

        // Size each side down as inventory approaches its cap on that side, and
        // to nothing at the cap. A hard on/off switch at the limit produces a
        // maker that is fully aggressive right up to the moment it is stuck.
        let room_buy = ((max_inv - pos) as f64 / max_inv.max(1) as f64).clamp(0.0, 1.0);
        let room_sell = ((max_inv + pos) as f64 / max_inv.max(1) as f64).clamp(0.0, 1.0);

        for (side, target, room) in [
            (Side::Buy, r - half, room_buy),
            (Side::Sell, r + half, room_sell),
        ] {
            let qty = (self.cfg.size as f64 * room).round() as i64;
            if qty <= 0 {
                continue;
            }
            let Ok(raw) = Price::from_dollars(target) else { continue };
            // Round away from the touch, so rounding never tightens a spread
            // that was deliberately chosen.
            let mut price = raw.round_to_tick_conservative(tick, side);

            // Never quote through the opposite side: a post-only order that
            // would cross is rejected, and a maker that generates rejections
            // instead of quotes is invisible to the market and noisy to the
            // venue.
            price = match side {
                Side::Buy => price.min(Price::from_micros(best_ask.micros() - tick)),
                Side::Sell => price.max(Price::from_micros(best_bid.micros() + tick)),
            };
            if !price.is_tradable() {
                continue;
            }
            out.push(DesiredQuote {
                side,
                price,
                qty: Qty(qty),
            });
        }
        out
    }
}

impl Strategy for QuoteMaker {
    fn id(&self) -> StrategyId {
        self.id
    }

    fn name(&self) -> &'static str {
        "quote-maker"
    }

    fn on_market(&mut self, view: &MarketView<'_>, out: &mut Vec<Action>) {
        let start = out.len();
        let desired = self.desired(view);
        let tick = view.spec.tick_size;
        let tolerance = self.cfg.requote_ticks.max(0) * tick;

        for side in [Side::Buy, Side::Sell] {
            let want = desired.iter().find(|d| d.side == side);
            let mut kept = false;

            for existing in view.resting_on(side) {
                let stale = existing.age_secs(view.now) > self.cfg.max_quote_age_secs;
                let in_place = want
                    .map(|w| (w.price.micros() - existing.price.micros()).abs() <= tolerance)
                    .unwrap_or(false);

                if !stale && in_place && !kept {
                    // Leave it: the queue position it has already earned is
                    // worth more than the fraction of a tick on offer.
                    kept = true;
                    continue;
                }
                out.push(Action::Cancel {
                    order_id: existing.id,
                    reason: if want.is_none() {
                        "no-quote".into()
                    } else if stale {
                        "stale-quote".into()
                    } else {
                        "reprice".into()
                    },
                });
            }

            if let Some(w) = want
                && !kept
            {
                out.push(Action::Place(OrderIntent::quote(
                    view.spec.id,
                    w.side,
                    w.price,
                    w.qty,
                    "make",
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
    use crate::strategy::RestingOrder;
    use edge_core::types::{OrderId, Ts};

    fn maker() -> QuoteMaker {
        QuoteMaker::new(StrategyId(1), QuoteConfig::default())
    }

    #[test]
    fn a_flat_maker_quotes_symmetrically_around_fair() {
        let sim = Sim::quoted(45, 55);
        let mut m = maker();
        let acts = sim.step(&mut m, &sim.plain());
        let ps = places(&acts);
        assert_eq!(ps.len(), 2);

        let bid = ps.iter().find(|p| p.side == Side::Buy).unwrap();
        let ask = ps.iter().find(|p| p.side == Side::Sell).unwrap();
        // Mid is 50c; the two quotes should straddle it evenly.
        let below = 0.50 - bid.price.dollars();
        let above = ask.price.dollars() - 0.50;
        assert!((below - above).abs() < 0.011, "bid {below} ask {above}");
        assert!(ps.iter().all(|p| p.tif == TimeInForce::PostOnly));
    }

    #[test]
    fn a_long_maker_leans_its_quotes_lower_on_both_sides() {
        let sim = Sim::quoted(45, 55);
        let mut m = maker();
        let flat = places(&sim.step(&mut m, &sim.plain()));

        let mut long = Sim::quoted(45, 55);
        long.position = Qty(200);
        let leaned = places(&sim.step(&mut m, &long.plain()));

        for side in [Side::Buy, Side::Sell] {
            let a = flat.iter().find(|p| p.side == side).unwrap().price;
            let b = leaned.iter().find(|p| p.side == side).unwrap().price;
            assert!(b < a, "{side:?}: leaned {b:?} should be below flat {a:?}");
        }
    }

    #[test]
    fn a_maker_at_its_inventory_cap_stops_adding_to_the_position() {
        let mut sim = Sim::quoted(45, 55);
        sim.position = Qty(250); // exactly max_inventory
        let mut m = maker();
        let ps = places(&sim.step(&mut m, &sim.plain()));

        assert!(!ps.iter().any(|p| p.side == Side::Buy), "should not buy more");
        assert!(ps.iter().any(|p| p.side == Side::Sell), "should still offer");
    }

    #[test]
    fn sizes_taper_toward_the_inventory_cap_rather_than_cutting_off() {
        let mut m = maker();
        let mut sizes = Vec::new();
        for pos in [0, 125, 200] {
            let mut sim = Sim::quoted(45, 55);
            sim.position = Qty(pos);
            let ps = places(&sim.step(&mut m, &sim.plain()));
            sizes.push(ps.iter().find(|p| p.side == Side::Buy).unwrap().qty.get());
        }
        assert!(sizes[0] > sizes[1] && sizes[1] > sizes[2], "{sizes:?}");
    }

    #[test]
    fn quotes_never_cross_the_touch() {
        // A one-tick market: naive symmetric quoting would post inside it.
        let sim = Sim::quoted(50, 51);
        let mut m = maker();
        let ps = places(&sim.step(&mut m, &sim.plain()));
        for p in &ps {
            match p.side {
                Side::Buy => assert!(p.price < Price::from_cents(51), "{p:?}"),
                Side::Sell => assert!(p.price > Price::from_cents(50), "{p:?}"),
            }
        }
    }

    #[test]
    fn the_half_spread_never_undercuts_the_fee() {
        let mut sim = Sim::quoted(45, 55);
        sim.spec.fee = edge_core::fees::FeeModel::KALSHI_STANDARD;
        let m = QuoteMaker::new(
            StrategyId(1),
            QuoteConfig {
                min_half_spread: 0.0001,
                ..Default::default()
            },
        );
        let fair = Prob::new(0.50).unwrap();
        // Kalshi rebates makers, so a floor built on the maker rate alone would
        // be zero. The round trip a maker actually runs is maker in, taker out.
        let round_trip = sim.spec.fee.fee_per_contract(Price::from_cents(50), Qty(25), Liquidity::Maker)
            + sim.spec.fee.fee_per_contract(Price::from_cents(50), Qty(25), Liquidity::Taker);
        assert!(round_trip > 0.0, "fixture should charge a fee");
        assert!(m.half_spread(&sim.plain(), fair) >= round_trip / 2.0);
    }

    #[test]
    fn an_existing_quote_at_the_right_price_keeps_its_queue_position() {
        let mut sim = Sim::quoted(45, 55);
        let mut m = maker();
        let first = places(&sim.step(&mut m, &sim.plain()));
        let bid = first.iter().find(|p| p.side == Side::Buy).unwrap();

        sim.resting = vec![RestingOrder {
            id: OrderId(1),
            side: Side::Buy,
            price: bid.price,
            remaining: Qty(25),
            ts: Ts(0),
        }];
        let acts = sim.step(&mut m, &sim.plain());
        assert!(
            !acts.iter().any(|a| matches!(a, Action::Cancel { order_id, .. } if *order_id == OrderId(1))),
            "a good quote should not be churned: {acts:?}"
        );
        assert!(
            !places(&acts).iter().any(|p| p.side == Side::Buy),
            "and should not be duplicated"
        );
    }

    #[test]
    fn a_quote_that_has_drifted_is_repriced() {
        let mut sim = Sim::quoted(45, 55);
        let mut m = maker();
        sim.resting = vec![RestingOrder {
            id: OrderId(1),
            side: Side::Buy,
            price: Price::from_cents(20), // nowhere near fair
            remaining: Qty(25),
            ts: Ts(0),
        }];
        let acts = sim.step(&mut m, &sim.plain());
        assert_eq!(acts.iter().filter(|a| a.reason() == "reprice").count(), 1);
        assert!(places(&acts).iter().any(|p| p.side == Side::Buy));
    }

    #[test]
    fn a_quote_that_has_rested_too_long_is_pulled_even_at_a_good_price() {
        let mut sim = Sim::quoted(45, 55);
        let mut m = maker();
        let bid = places(&sim.step(&mut m, &sim.plain()))
            .into_iter()
            .find(|p| p.side == Side::Buy)
            .unwrap();

        sim.now = Ts(200_000_000_000); // 200s
        sim.resting = vec![RestingOrder {
            id: OrderId(1),
            side: Side::Buy,
            price: bid.price,
            remaining: Qty(25),
            ts: Ts(0),
        }];
        let acts = sim.step(&mut m, &sim.plain());
        assert_eq!(acts.iter().filter(|a| a.reason() == "stale-quote").count(), 1);
    }

    #[test]
    fn the_maker_stands_down_near_resolution_and_pulls_what_it_has() {
        let mut sim = Sim::quoted(45, 55);
        sim.spec.closes_at = Some(Ts(60_000_000_000)); // 60s away
        sim.resting = vec![RestingOrder {
            id: OrderId(1),
            side: Side::Buy,
            price: Price::from_cents(45),
            remaining: Qty(25),
            ts: Ts(0),
        }];
        let mut m = maker();
        let acts = sim.step(&mut m, &sim.plain());
        assert!(places(&acts).is_empty(), "should not quote: {acts:?}");
        assert_eq!(acts.iter().filter(|a| a.reason() == "no-quote").count(), 1);
    }

    #[test]
    fn the_maker_widens_as_resolution_approaches() {
        let m = maker();
        let fair = Prob::new(0.5).unwrap();

        let far = Sim::quoted(45, 55);
        let wide_far = m.half_spread(&far.plain(), fair);

        let mut near = Sim::quoted(45, 55);
        near.spec.closes_at = Some(Ts(900_000_000_000)); // 15 minutes
        let wide_near = m.half_spread(&near.plain(), fair);

        assert!(wide_near > wide_far, "{wide_near} vs {wide_far}");
    }

    #[test]
    fn a_halted_market_gets_no_quotes() {
        let mut sim = Sim::quoted(45, 55);
        sim.spec.status = edge_core::market::MarketStatus::Halted;
        let mut m = maker();
        assert!(places(&sim.step(&mut m, &sim.plain())).is_empty());
    }
}
