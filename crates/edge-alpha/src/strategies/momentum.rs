//! Trading with a move that order flow confirms.
//!
//! Momentum in a prediction market is not the same object as momentum in an
//! equity. A contract's price is a probability with a hard deadline, and it
//! moves for exactly one interesting reason: information arrived. So the signal
//! this strategy wants is not "the price went up" — it is "the price went up
//! *and* aggressive buyers are still lifting offers", which is what information
//! being priced in looks like while it is still happening.
//!
//! That distinction is the whole strategy. A price drift with no confirming
//! flow is usually one participant walking a thin book, and following it means
//! buying their inventory at the top. Both conditions are therefore required,
//! and they must agree in sign.
//!
//! Three further guards, each closing off a way this idea loses money:
//!
//! - **No trading near the bounds.** At 3c, one tick is a 33% move in implied
//!   odds. Momentum measured in log-odds is enormous there and means nothing.
//! - **No trading through a wide spread.** The signal has to pay for the
//!   crossing, and a wide spread is also the market saying it does not know.
//! - **No pyramiding.** Positions are opened once and exited on decay, not
//!   added to while the move continues, which is how a trend follower turns a
//!   good run into a single large adverse position at the reversal.

use edge_core::types::{Qty, Side, StrategyId};
use serde::{Deserialize, Serialize};

use crate::strategy::{Action, MarketView, OrderIntent, StatsRecorder, Strategy, StrategyStats};

#[cfg(test)]
use edge_core::types::Price;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MomentumConfig {
    /// Minimum trend, in log-odds, to act on.
    pub min_trend: f64,
    /// Minimum normalised order-flow imbalance that must agree with the trend.
    /// Zero would disable the confirmation requirement, which is the entire
    /// point of the strategy — so it is validated against.
    pub min_flow: f64,
    /// Contracts at the minimum signal. Scales up to `size_scale`× from there.
    pub base_size: i64,
    pub size_scale: f64,
    pub max_position: i64,
    /// Widest spread worth crossing, in probability.
    pub max_spread: f64,
    /// Refuse to trade outside this price band. Outside it, a tick is a large
    /// change in odds and the signal is dominated by discretisation.
    pub price_band: (f64, f64),
    /// Trend at which an open position is closed. Below `min_trend` so a
    /// position is not opened and shut on consecutive ticks.
    pub exit_trend: f64,
    /// Refuse to open inside this many seconds of resolution.
    pub min_seconds_left: f64,
}

impl Default for MomentumConfig {
    fn default() -> Self {
        MomentumConfig {
            min_trend: 0.10,
            min_flow: 0.30,
            base_size: 20,
            size_scale: 3.0,
            max_position: 200,
            max_spread: 0.03,
            price_band: (0.10, 0.90),
            exit_trend: 0.03,
            min_seconds_left: 120.0,
        }
    }
}

impl MomentumConfig {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.min_flow <= 0.0 {
            return Err("flow confirmation cannot be disabled: an unconfirmed drift is not a signal");
        }
        if self.exit_trend >= self.min_trend {
            return Err("exit threshold must be below the entry threshold");
        }
        if self.price_band.0 >= self.price_band.1 {
            return Err("price band is empty");
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Momentum {
    id: StrategyId,
    cfg: MomentumConfig,
    stats: StatsRecorder,
}

impl Momentum {
    pub fn new(id: StrategyId, cfg: MomentumConfig) -> Self {
        Momentum {
            id,
            cfg,
            stats: StatsRecorder::default(),
        }
    }

    pub fn config(&self) -> &MomentumConfig {
        &self.cfg
    }

    /// The confirmed trend, or `None` when any precondition fails. Positive
    /// means upward and buyer-confirmed.
    pub fn signal(&self, view: &MarketView<'_>) -> Option<f64> {
        if !view.is_tradable() {
            return None;
        }
        let f = view.features?;
        let mid = f.mid;
        if mid < self.cfg.price_band.0 || mid > self.cfg.price_band.1 {
            return None;
        }
        if f.get("spread")? > self.cfg.max_spread {
            return None;
        }

        let trend = f.get("trend")?;
        let flow = f.get("order_flow")?;
        if !trend.is_finite() || !flow.is_finite() {
            return None;
        }

        // Both conditions, agreeing in sign. Either alone is noise.
        if trend.abs() < self.cfg.min_trend || flow.abs() < self.cfg.min_flow {
            return None;
        }
        if trend.signum() != flow.signum() {
            return None;
        }
        Some(trend)
    }

    fn size_for(&self, trend: f64) -> i64 {
        let scale = (trend.abs() / self.cfg.min_trend).clamp(1.0, self.cfg.size_scale);
        (self.cfg.base_size as f64 * scale).round() as i64
    }
}

impl Strategy for Momentum {
    fn id(&self) -> StrategyId {
        self.id
    }

    fn name(&self) -> &'static str {
        "momentum"
    }

    fn on_market(&mut self, view: &MarketView<'_>, out: &mut Vec<Action>) {
        let start = out.len();
        let pos = view.position.get();
        let trend = view.features.and_then(|f| f.get("trend")).unwrap_or(0.0);

        // Exit first: a decayed or reversed trend closes the position regardless
        // of whether a fresh entry also qualifies.
        if pos != 0
            && view.is_tradable()
            && (trend.abs() < self.cfg.exit_trend || trend.signum() != (pos as f64).signum())
        {
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
                    "momentum-decayed",
                )));
                self.stats.record(&out[start..]);
                return;
            }
        }

        if view.time_left() < self.cfg.min_seconds_left {
            self.stats.record(&out[start..]);
            return;
        }

        if let Some(trend) = self.signal(view) {
            let side = if trend > 0.0 { Side::Buy } else { Side::Sell };
            // No pyramiding: a position already open in the signal's direction
            // is left alone.
            let aligned = pos != 0 && (pos > 0) == (trend > 0.0);
            if !aligned {
                let price = match side {
                    Side::Buy => view.book.best_ask(),
                    Side::Sell => view.book.best_bid(),
                };
                let room = match side {
                    Side::Buy => self.cfg.max_position - pos,
                    Side::Sell => self.cfg.max_position + pos,
                };
                let qty = self.size_for(trend).min(room).max(0);
                if let Some(price) = price
                    && qty > 0
                {
                    out.push(Action::Place(OrderIntent::take(
                        view.spec.id,
                        side,
                        price,
                        Qty(qty),
                        "momentum",
                    )));
                }
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
    use edge_core::types::Ts;

    fn mom() -> Momentum {
        Momentum::new(StrategyId(3), MomentumConfig::default())
    }

    #[test]
    fn the_default_configuration_is_coherent() {
        MomentumConfig::default().validate().unwrap();
    }

    #[test]
    fn confirmation_cannot_be_configured_away() {
        assert!(
            MomentumConfig {
                min_flow: 0.0,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn a_rally_that_buyers_are_still_lifting_is_bought() {
        let sim = Sim::quoted(49, 51);
        let f = feats(0.50, &[("trend", 0.25), ("order_flow", 0.8), ("spread", 0.02)]);
        let mut v = sim.plain();
        v.features = Some(&f);

        let mut m = mom();
        let acts = sim.step(&mut m, &v);
        let i = &places(&acts)[0];
        assert_eq!(i.side, Side::Buy);
        assert_eq!(i.price, Price::from_cents(51));
        assert_eq!(&*i.reason, "momentum");
    }

    #[test]
    fn a_drift_with_no_confirming_flow_is_ignored() {
        let sim = Sim::quoted(49, 51);
        let f = feats(0.50, &[("trend", 0.40), ("order_flow", 0.05), ("spread", 0.02)]);
        let mut v = sim.plain();
        v.features = Some(&f);
        assert!(mom().signal(&v).is_none(), "unconfirmed drift is not a signal");
    }

    #[test]
    fn flow_that_disagrees_with_the_move_is_ignored() {
        // Price rising while aggressors sell: someone is being walked up, or
        // the print is stale. Either way it is not information being priced.
        let sim = Sim::quoted(49, 51);
        let f = feats(0.50, &[("trend", 0.30), ("order_flow", -0.9), ("spread", 0.02)]);
        let mut v = sim.plain();
        v.features = Some(&f);
        assert!(mom().signal(&v).is_none());
    }

    #[test]
    fn a_selloff_is_sold_into_the_bid() {
        let sim = Sim::quoted(49, 51);
        let f = feats(0.50, &[("trend", -0.30), ("order_flow", -0.7), ("spread", 0.02)]);
        let mut v = sim.plain();
        v.features = Some(&f);
        let mut m = mom();
        let acts = sim.step(&mut m, &v);
        let i = &places(&acts)[0];
        assert_eq!(i.side, Side::Sell);
        assert_eq!(i.price, Price::from_cents(49));
    }

    #[test]
    fn size_grows_with_the_signal_but_is_capped() {
        let m = mom();
        let base = m.size_for(0.10);
        let bigger = m.size_for(0.25);
        let huge = m.size_for(10.0);
        assert!(bigger > base);
        assert_eq!(huge, (m.config().base_size as f64 * m.config().size_scale) as i64);
    }

    #[test]
    fn nothing_is_traded_at_the_extremes_of_the_price_range() {
        // At 4c a one-tick move is a huge log-odds change and the signal is
        // discretisation, not information.
        let sim = Sim::quoted(3, 5);
        let f = feats(0.04, &[("trend", 0.9), ("order_flow", 0.9), ("spread", 0.02)]);
        let mut v = sim.plain();
        v.features = Some(&f);
        assert!(mom().signal(&v).is_none());
    }

    #[test]
    fn nothing_is_traded_through_a_wide_spread() {
        let sim = Sim::quoted(40, 60);
        let f = feats(0.50, &[("trend", 0.5), ("order_flow", 0.9), ("spread", 0.20)]);
        let mut v = sim.plain();
        v.features = Some(&f);
        assert!(mom().signal(&v).is_none());
    }

    #[test]
    fn a_position_is_not_pyramided_while_the_move_continues() {
        let mut sim = Sim::quoted(49, 51);
        sim.position = Qty(50);
        let f = feats(0.50, &[("trend", 0.30), ("order_flow", 0.9), ("spread", 0.02)]);
        let mut v = sim.plain();
        v.features = Some(&f);

        let mut m = mom();
        assert!(m.signal(&v).is_some(), "the signal is still live");
        assert!(places(&sim.step(&mut m, &v)).is_empty(), "but must not add");
    }

    #[test]
    fn a_decayed_trend_closes_the_position() {
        let mut sim = Sim::quoted(49, 51);
        sim.position = Qty(50);
        let f = feats(0.50, &[("trend", 0.01), ("order_flow", 0.9), ("spread", 0.02)]);
        let mut v = sim.plain();
        v.features = Some(&f);

        let mut m = mom();
        let acts = sim.step(&mut m, &v);
        let i = &places(&acts)[0];
        assert_eq!(i.side, Side::Sell);
        assert_eq!(i.qty, Qty(50));
        assert_eq!(&*i.reason, "momentum-decayed");
    }

    #[test]
    fn a_reversed_trend_closes_before_it_reopens() {
        let mut sim = Sim::quoted(49, 51);
        sim.position = Qty(50);
        let f = feats(0.50, &[("trend", -0.40), ("order_flow", -0.9), ("spread", 0.02)]);
        let mut v = sim.plain();
        v.features = Some(&f);

        let mut m = mom();
        let acts = sim.step(&mut m, &v);
        assert_eq!(acts.len(), 1, "one action: flatten, not flip in a single step");
        assert_eq!(places(&acts)[0].qty, Qty(50));
        assert_eq!(&*places(&acts)[0].reason, "momentum-decayed");
    }

    #[test]
    fn nothing_opens_close_to_resolution_but_positions_still_close() {
        let mut sim = Sim::quoted(49, 51);
        sim.spec.closes_at = Some(Ts(30_000_000_000));
        let f = feats(0.50, &[("trend", 0.30), ("order_flow", 0.9), ("spread", 0.02)]);
        let mut v = sim.plain();
        v.features = Some(&f);
        let mut m = mom();
        assert!(places(&sim.step(&mut m, &v)).is_empty());

        // ...but an open position is still allowed to exit on decay.
        sim.position = Qty(50);
        let g = feats(0.50, &[("trend", 0.0), ("order_flow", 0.0), ("spread", 0.02)]);
        let mut v = sim.plain();
        v.features = Some(&g);
        assert_eq!(places(&sim.step(&mut m, &v)).len(), 1);
    }

    #[test]
    fn a_market_with_no_features_is_left_alone() {
        let sim = Sim::quoted(49, 51);
        let mut m = mom();
        assert!(m.signal(&sim.plain()).is_none());
        assert!(sim.step(&mut m, &sim.plain()).is_empty());
    }
}
