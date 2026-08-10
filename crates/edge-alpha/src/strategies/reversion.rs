//! Fading a move that order flow does *not* confirm.
//!
//! This is the exact dual of [`crate::strategies::momentum`], and the pair only
//! makes sense read together: momentum requires flow confirmation, reversion
//! requires its absence. The two therefore cannot both fire on the same book,
//! which is the point — a system running both without that disjointness is just
//! paying spread to trade against itself.
//!
//! The trade is: a price has moved several standard deviations from its recent
//! range, *nobody is aggressively trading in the direction of the move*, and
//! resolution is far enough away that the range has time to reassert. That
//! combination is usually a thin book being pushed rather than information
//! arriving, and it reverts.
//!
//! Two design choices distinguish this from the naive version:
//!
//! **It fades passively, by improving the touch, rather than crossing.** A
//! reversion signal is a statement that the current price is too good; crossing
//! the spread to act on it gives back the very edge being claimed. Posting
//! inside means the fill happens only if the move continues one more tick — and
//! being filled reluctantly on a mean-reversion trade is a feature.
//!
//! **It refuses to fade in the endgame.** Close to resolution a large move
//! usually *is* the answer arriving, and the range it is departing from is
//! about to become meaningless. This is the single most expensive way to run
//! this strategy, so the guard is generous.

use edge_core::types::{Price, Qty, Side, StrategyId};
use serde::{Deserialize, Serialize};

use crate::strategy::{Action, MarketView, OrderIntent, StatsRecorder, Strategy, StrategyStats};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ReversionConfig {
    /// Standard deviations from the recent range before a fade is considered.
    pub min_z: f64,
    /// Maximum confirming order flow. Above this the move is being driven by
    /// aggressors and is momentum, not dislocation.
    pub max_flow: f64,
    pub base_size: i64,
    pub size_scale: f64,
    pub max_position: i64,
    /// Widest spread worth quoting into.
    pub max_spread: f64,
    /// Price band outside which the range statistics are dominated by the
    /// bounds rather than by behaviour.
    pub price_band: (f64, f64),
    /// Close the position once the price is back inside this many standard
    /// deviations.
    pub exit_z: f64,
    /// Do not fade inside this many seconds of resolution. Deliberately much
    /// longer than the momentum guard: late moves are usually the answer.
    pub min_seconds_left: f64,
    /// Post this many ticks inside the touch. One tick is the usual choice —
    /// it takes queue priority without giving up meaningful price.
    pub improve_ticks: i64,
}

impl Default for ReversionConfig {
    fn default() -> Self {
        ReversionConfig {
            min_z: 2.0,
            max_flow: 0.25,
            base_size: 20,
            size_scale: 2.5,
            max_position: 200,
            max_spread: 0.04,
            price_band: (0.10, 0.90),
            exit_z: 0.5,
            min_seconds_left: 1_800.0,
            improve_ticks: 1,
        }
    }
}

impl ReversionConfig {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.exit_z >= self.min_z {
            return Err("exit threshold must be inside the entry threshold");
        }
        if self.min_z <= 0.0 {
            return Err("entry threshold must be positive");
        }
        if self.price_band.0 >= self.price_band.1 {
            return Err("price band is empty");
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct MeanReversion {
    id: StrategyId,
    cfg: ReversionConfig,
    stats: StatsRecorder,
}

impl MeanReversion {
    pub fn new(id: StrategyId, cfg: ReversionConfig) -> Self {
        MeanReversion { id, cfg, stats: StatsRecorder::default() }
    }

    pub fn config(&self) -> &ReversionConfig {
        &self.cfg
    }

    /// The dislocation to fade, or `None`. Positive z means the price is high
    /// relative to its range, so the trade is to sell.
    pub fn signal(&self, view: &MarketView<'_>) -> Option<f64> {
        if !view.is_tradable() || view.time_left() < self.cfg.min_seconds_left {
            return None;
        }
        let f = view.features?;
        if f.mid < self.cfg.price_band.0 || f.mid > self.cfg.price_band.1 {
            return None;
        }
        if f.get("spread")? > self.cfg.max_spread {
            return None;
        }

        let z = f.get("z_score")?;
        let flow = f.get("order_flow")?;
        if !z.is_finite() || !flow.is_finite() || z.abs() < self.cfg.min_z {
            return None;
        }

        // The disjointness condition. Flow *confirming* the move in either
        // direction disqualifies the fade; this is momentum's territory.
        if flow.abs() > self.cfg.max_flow && flow.signum() == z.signum() {
            return None;
        }
        Some(z)
    }

    fn size_for(&self, z: f64) -> i64 {
        let scale = (z.abs() / self.cfg.min_z).clamp(1.0, self.cfg.size_scale);
        (self.cfg.base_size as f64 * scale).round() as i64
    }

    /// Where to rest the fade: one tick inside the touch on the side being
    /// traded, never through the opposite side.
    fn passive_price(&self, view: &MarketView<'_>, side: Side) -> Option<Price> {
        let bid = view.book.best_bid()?;
        let ask = view.book.best_ask()?;
        let tick = view.spec.tick_size;
        let step = self.cfg.improve_ticks.max(0) * tick;

        let p = match side {
            Side::Buy => Price::from_micros((bid.micros() + step).min(ask.micros() - tick)),
            Side::Sell => Price::from_micros((ask.micros() - step).max(bid.micros() + tick)),
        };
        p.is_tradable().then_some(p)
    }
}

impl Strategy for MeanReversion {
    fn id(&self) -> StrategyId {
        self.id
    }

    fn name(&self) -> &'static str {
        "mean-reversion"
    }

    fn on_market(&mut self, view: &MarketView<'_>, out: &mut Vec<Action>) {
        let start = out.len();
        let pos = view.position.get();
        let z = view.features.and_then(|f| f.get("z_score")).unwrap_or(0.0);

        // Take profit once the price is back in its range. Crossing is right
        // here: the reason for holding is gone, and a resting exit on a
        // reverted price is an invitation to be picked off on the next move.
        if pos != 0 && view.is_tradable() && z.abs() <= self.cfg.exit_z {
            let (side, price) = if pos > 0 {
                (Side::Sell, view.book.best_bid())
            } else {
                (Side::Buy, view.book.best_ask())
            };
            if let Some(price) = price {
                out.push(Action::Place(OrderIntent::take(
                    view.spec.id,
                    side,
                    price,
                    Qty(pos.abs()),
                    "reversion-complete",
                )));
                self.stats.record(&out[start..]);
                return;
            }
        }

        if let Some(z) = self.signal(view) {
            // High price relative to range → sell it.
            let side = if z > 0.0 { Side::Sell } else { Side::Buy };
            let already = pos != 0 && (pos > 0) == (side == Side::Buy);
            let has_quote = view.resting_on(side).next().is_some();

            if !already && !has_quote {
                let room = match side {
                    Side::Buy => self.cfg.max_position - pos,
                    Side::Sell => self.cfg.max_position + pos,
                };
                let qty = self.size_for(z).min(room).max(0);
                if let Some(price) = self.passive_price(view, side)
                    && qty > 0
                {
                    out.push(Action::Place(OrderIntent::quote(
                        view.spec.id,
                        side,
                        price,
                        Qty(qty),
                        "fade",
                    )));
                }
            }
        } else {
            // The dislocation is gone; a resting fade at a price nobody wants
            // is now just a free option written to the market.
            for o in view.resting {
                out.push(Action::Cancel { order_id: o.id, reason: "fade-expired".into() });
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
    use crate::strategy::RestingOrder;
    use crate::strategy::harness::*;
    use edge_book::order::TimeInForce;
    use edge_core::types::{OrderId, Ts};

    fn rev() -> MeanReversion {
        MeanReversion::new(StrategyId(4), ReversionConfig::default())
    }

    /// A dislocation with no confirming flow, far from resolution.
    fn dislocated(z: f64, flow: f64) -> crate::Features {
        feats(0.50, &[("z_score", z), ("order_flow", flow), ("spread", 0.02)])
    }

    #[test]
    fn the_default_configuration_is_coherent() {
        ReversionConfig::default().validate().unwrap();
    }

    #[test]
    fn a_high_unconfirmed_price_is_faded_by_offering_inside_the_ask() {
        let sim = Sim::quoted(49, 53);
        let f = dislocated(3.0, 0.05);
        let mut v = sim.plain();
        v.features = Some(&f);

        let mut r = rev();
        let acts = sim.step(&mut r, &v);
        let i = &places(&acts)[0];
        assert_eq!(i.side, Side::Sell);
        assert_eq!(i.price, Price::from_cents(52), "one tick inside the 53c ask");
        assert_eq!(i.tif, TimeInForce::PostOnly, "fading by crossing gives back the edge");
        assert_eq!(&*i.reason, "fade");
    }

    #[test]
    fn a_low_unconfirmed_price_is_faded_by_bidding_inside_the_bid() {
        let sim = Sim::quoted(47, 51);
        let f = dislocated(-3.0, -0.05);
        let mut v = sim.plain();
        v.features = Some(&f);

        let mut r = rev();
        let acts = sim.step(&mut r, &v);
        let i = &places(&acts)[0];
        assert_eq!(i.side, Side::Buy);
        assert_eq!(i.price, Price::from_cents(48));
    }

    #[test]
    fn a_move_that_aggressors_are_driving_is_left_to_momentum() {
        // This is the disjointness condition: the same book cannot be both a
        // momentum signal and a reversion signal.
        let sim = Sim::quoted(49, 53);
        let f = dislocated(3.0, 0.9);
        let mut v = sim.plain();
        v.features = Some(&f);
        assert!(rev().signal(&v).is_none());

        let m = crate::strategies::momentum::Momentum::new(
            StrategyId(3),
            crate::strategies::momentum::MomentumConfig { max_spread: 0.05, ..Default::default() },
        );
        let g =
            feats(0.50, &[("trend", 0.3), ("order_flow", 0.9), ("spread", 0.02), ("z_score", 3.0)]);
        let mut v2 = sim.plain();
        v2.features = Some(&g);
        assert!(m.signal(&v2).is_some(), "momentum owns this book");
        assert!(rev().signal(&v2).is_none(), "and reversion declines it");
    }

    #[test]
    fn flow_opposing_the_move_does_not_disqualify_the_fade() {
        // Heavy selling into a price that is already high is the fade
        // *confirming*, not contradicting.
        let sim = Sim::quoted(49, 53);
        let f = dislocated(3.0, -0.9);
        let mut v = sim.plain();
        v.features = Some(&f);
        assert!(rev().signal(&v).is_some());
    }

    #[test]
    fn a_small_dislocation_is_not_traded() {
        let sim = Sim::quoted(49, 53);
        let f = dislocated(1.2, 0.0);
        let mut v = sim.plain();
        v.features = Some(&f);
        assert!(rev().signal(&v).is_none());
    }

    #[test]
    fn nothing_is_faded_near_resolution() {
        // The expensive mistake: a large late move is usually the answer.
        let mut sim = Sim::quoted(49, 53);
        sim.spec.closes_at = Some(Ts(600_000_000_000)); // 10 minutes
        let f = dislocated(4.0, 0.0);
        let mut v = sim.plain();
        v.features = Some(&f);
        assert!(rev().signal(&v).is_none());
    }

    #[test]
    fn size_grows_with_the_dislocation_but_is_capped() {
        let r = rev();
        assert!(r.size_for(3.0) > r.size_for(2.0));
        assert_eq!(r.size_for(50.0), (r.config().base_size as f64 * r.config().size_scale) as i64);
    }

    #[test]
    fn a_reverted_price_closes_the_position_by_crossing() {
        let mut sim = Sim::quoted(49, 51);
        sim.position = Qty(40);
        let f = dislocated(0.1, 0.0);
        let mut v = sim.plain();
        v.features = Some(&f);

        let mut r = rev();
        let acts = sim.step(&mut r, &v);
        let i = &places(&acts)[0];
        assert_eq!(i.side, Side::Sell);
        assert_eq!(i.qty, Qty(40));
        assert_eq!(i.tif, TimeInForce::Ioc);
        assert_eq!(&*i.reason, "reversion-complete");
    }

    #[test]
    fn a_resting_fade_is_pulled_once_the_dislocation_passes() {
        let mut sim = Sim::quoted(49, 53);
        sim.resting = vec![RestingOrder {
            id: OrderId(7),
            side: Side::Sell,
            price: Price::from_cents(52),
            remaining: Qty(20),
            ts: Ts(0),
        }];
        let f = dislocated(0.8, 0.0); // no longer dislocated, not yet an exit
        let mut v = sim.plain();
        v.features = Some(&f);

        let mut r = rev();
        let acts = sim.step(&mut r, &v);
        assert_eq!(acts.len(), 1);
        assert_eq!(acts[0].reason(), "fade-expired");
    }

    #[test]
    fn an_existing_fade_is_not_duplicated() {
        let mut sim = Sim::quoted(49, 53);
        sim.resting = vec![RestingOrder {
            id: OrderId(7),
            side: Side::Sell,
            price: Price::from_cents(52),
            remaining: Qty(20),
            ts: Ts(0),
        }];
        let f = dislocated(3.0, 0.0);
        let mut v = sim.plain();
        v.features = Some(&f);

        let mut r = rev();
        assert!(places(&sim.step(&mut r, &v)).is_empty());
    }

    #[test]
    fn the_fade_never_quotes_through_the_opposite_side() {
        let sim = Sim::quoted(50, 51); // one-tick market
        let f = dislocated(3.0, 0.0);
        let mut v = sim.plain();
        v.features = Some(&f);

        let mut r = rev();
        let acts = sim.step(&mut r, &v);
        let i = &places(&acts)[0];
        assert!(i.price > Price::from_cents(50), "{i:?}");
    }

    #[test]
    fn a_market_with_no_features_is_left_alone() {
        let sim = Sim::quoted(49, 53);
        let mut r = rev();
        assert!(r.signal(&sim.plain()).is_none());
        assert!(sim.step(&mut r, &sim.plain()).is_empty());
    }
}
