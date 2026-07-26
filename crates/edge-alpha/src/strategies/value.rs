//! Taking liquidity when the market disagrees with independent evidence.
//!
//! This is the strategy the whole system exists to feed: everything upstream —
//! devigging, cross-venue consensus, the online model — produces a fair value,
//! and this decides whether that disagreement is worth crossing a spread for.
//!
//! Four rules govern it, and the first is the one that matters:
//!
//! **It will not trade against the mid.** Fair value must come from evidence
//! independent of this book ([`MarketView::independent_fair`]). Comparing a mid
//! to a touch always shows half a spread of apparent edge, so a taker anchored
//! on the mid crosses continuously and pays the spread every time. No model, no
//! consensus, no trade.
//!
//! **Edge is measured after fees, at the price actually available.** Not the
//! mid, not the last trade — the ask if buying, the bid if selling, with the
//! taker fee included. [`edge_core::ev::best_side`] does this arithmetic and
//! returns nothing when neither direction survives it.
//!
//! **Size is the smaller of Kelly and what is actually there.** Sizing past the
//! touch means the marginal contract is bought at a worse price than the one
//! the edge was computed at, which is how a backtested edge becomes a live loss.
//!
//! **Positions are exited when the thesis inverts**, not held to resolution out
//! of loyalty to the entry. The exit threshold is deliberately looser than the
//! entry so a position does not oscillate around the boundary paying fees.

use edge_core::ev::{self, KellyPolicy};
use edge_core::fees::Liquidity;
use edge_core::types::{Price, Prob, Qty, Side, StrategyId};
use serde::{Deserialize, Serialize};

use crate::strategy::{Action, MarketView, OrderIntent, StatsRecorder, Strategy, StrategyStats};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ValueConfig {
    /// Minimum post-fee edge in probability points before crossing. One cent is
    /// not an edge on a market that ticks in cents — it is a rounding error with
    /// a fee attached.
    pub min_edge: f64,
    /// Minimum expected profit per dollar committed. Screens out the case where
    /// a large absolute edge sits on a contract so expensive that the capital is
    /// better used elsewhere.
    pub min_ev_per_dollar: f64,
    pub kelly: KellyPolicy,
    /// Never take more than this share of the quantity resting at the touch.
    /// Consuming the whole level signals size and moves the price against the
    /// remainder of the order.
    pub max_touch_share: f64,
    /// Hard cap on the absolute position, independent of Kelly.
    pub max_position: i64,
    /// Adverse edge, in probability points, at which an open position is closed.
    /// Wider than `min_edge` on purpose — a position that flickers across the
    /// entry threshold should not be traded twice.
    pub exit_edge: f64,
    /// Refuse to open inside this many seconds of resolution. The model's own
    /// horizon is not that short, and the spread usually is not either.
    pub min_seconds_left: f64,
}

impl Default for ValueConfig {
    fn default() -> Self {
        ValueConfig {
            min_edge: 0.02,
            min_ev_per_dollar: 0.03,
            kelly: KellyPolicy::default(),
            max_touch_share: 0.5,
            max_position: 500,
            exit_edge: 0.04,
            min_seconds_left: 60.0,
        }
    }
}

/// What the taker concluded about a market, whether or not it acted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ValueSignal {
    pub side: Side,
    pub price: Price,
    pub qty: Qty,
    /// Post-fee edge in probability points.
    pub edge: f64,
    pub ev_per_dollar: f64,
    pub fair: Prob,
}

#[derive(Debug, Clone)]
pub struct ValueTaker {
    id: StrategyId,
    cfg: ValueConfig,
    stats: StatsRecorder,
}

impl ValueTaker {
    pub fn new(id: StrategyId, cfg: ValueConfig) -> Self {
        ValueTaker {
            id,
            cfg,
            stats: StatsRecorder::default(),
        }
    }

    pub fn config(&self) -> &ValueConfig {
        &self.cfg
    }

    /// Evaluate a market and return the trade worth doing, if any.
    ///
    /// Separated from `on_market` so the decision can be tested and logged
    /// without the intent plumbing, and so a dry-run mode can show what the
    /// system *would* have done.
    pub fn evaluate(&self, view: &MarketView<'_>) -> Option<ValueSignal> {
        if !view.is_tradable() || view.time_left() < self.cfg.min_seconds_left {
            return None;
        }
        // Independent, not `fair()`: see the module note.
        let fair = view.independent_fair()?;
        let bid = view.book.best_bid()?;
        let ask = view.book.best_ask()?;

        // Assess at a nominal lot first. Fee schedules are non-linear in size,
        // so the edge is size-dependent and has to be re-checked once sized.
        let probe = Qty(10);
        let a = ev::best_side(bid, ask, fair, probe, &view.spec.fee, Liquidity::Taker).ok()??;

        if a.edge < self.cfg.min_edge || a.ev_per_dollar < self.cfg.min_ev_per_dollar {
            return None;
        }

        let price = a.price;
        let wanted = self.cfg.kelly.size(&a, view.bankroll).get();

        // Available liquidity is on the far side of the trade: buying consumes
        // the ask, selling consumes the bid.
        let resting = view.book.qty_at(side_of_book(a.side), price).get();
        let available = (resting as f64 * self.cfg.max_touch_share).floor() as i64;

        // Room left before the position cap, in the direction being traded.
        let pos = view.position.get();
        let room = match a.side {
            Side::Buy => self.cfg.max_position - pos,
            Side::Sell => self.cfg.max_position + pos,
        };

        let qty = wanted.min(available).min(room).max(0);
        if qty <= 0 {
            return None;
        }

        // Re-assess at the real size: on a per-order rounded fee schedule a
        // one-lot and a hundred-lot have materially different economics, and
        // the edge that justified the trade must survive at the size traded.
        let sized = ev::assess(price, fair, a.side, Qty(qty), &view.spec.fee, Liquidity::Taker).ok()?;
        if sized.edge < self.cfg.min_edge || sized.ev_per_dollar < self.cfg.min_ev_per_dollar {
            return None;
        }

        Some(ValueSignal {
            side: a.side,
            price,
            qty: Qty(qty),
            edge: sized.edge,
            ev_per_dollar: sized.ev_per_dollar,
            fair,
        })
    }

    /// Whether an open position should be closed because the evidence moved
    /// against it, and by how much.
    pub fn exit(&self, view: &MarketView<'_>) -> Option<(Side, Price, Qty)> {
        let pos = view.position.get();
        if pos == 0 || !view.is_tradable() {
            return None;
        }
        let fair = view.independent_fair()?;

        // Close a long by selling into the bid, a short by buying the ask.
        let (side, price) = if pos > 0 {
            (Side::Sell, view.book.best_bid()?)
        } else {
            (Side::Buy, view.book.best_ask()?)
        };

        // The position's own edge, signed for the direction held. Positive means
        // the position is still justified.
        let held_edge = if pos > 0 {
            fair.get() - view.book.best_ask()?.dollars()
        } else {
            view.book.best_bid()?.dollars() - fair.get()
        };

        if held_edge > -self.cfg.exit_edge {
            return None;
        }
        Some((side, price, Qty(pos.abs())))
    }
}

/// Which side of the book a trade consumes.
fn side_of_book(taking: Side) -> Side {
    match taking {
        Side::Buy => Side::Sell,
        Side::Sell => Side::Buy,
    }
}

impl Strategy for ValueTaker {
    fn id(&self) -> StrategyId {
        self.id
    }

    fn name(&self) -> &'static str {
        "value-taker"
    }

    fn on_market(&mut self, view: &MarketView<'_>, out: &mut Vec<Action>) {
        let start = out.len();

        // Exit before entry: a market that wants both is one where the position
        // is wrong, and closing it is unambiguously the first move.
        if let Some((side, price, qty)) = self.exit(view) {
            out.push(Action::Place(OrderIntent::take(
                view.spec.id,
                side,
                price,
                qty,
                "exit-thesis-inverted",
            )));
        } else if let Some(sig) = self.evaluate(view) {
            out.push(Action::Place(OrderIntent::take(
                view.spec.id,
                sig.side,
                sig.price,
                sig.qty,
                "value",
            )));
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
    use edge_core::types::Ts;

    fn taker() -> ValueTaker {
        ValueTaker::new(StrategyId(2), ValueConfig::default())
    }

    fn p(x: f64) -> Prob {
        Prob::new(x).unwrap()
    }

    #[test]
    fn without_independent_evidence_nothing_is_ever_traded() {
        // The mid says 50c and the ask is 55c. A taker anchored on the mid would
        // see 5c of "edge" here; there is none.
        let sim = Sim::quoted(45, 55);
        let mut t = taker();
        assert!(t.evaluate(&sim.plain()).is_none());
        assert!(sim.step(&mut t, &sim.plain()).is_empty());
    }

    #[test]
    fn an_untrained_model_is_not_independent_evidence() {
        let sim = Sim::quoted(45, 55);
        let echo = crate::Prediction {
            z: [0.0; crate::N_FEATURES],
            market_logit: 0.0,
            score: 0.0,
            market: p(0.50),
            model: p(0.90),
            fair: p(0.50),
            weight: 0.0,
        };
        let mut t = taker();
        let v = sim.view(Some(&echo), None);
        assert!(t.evaluate(&v).is_none(), "a zero-weight model must not trade");
        assert!(sim.step(&mut t, &v).is_empty());
    }

    #[test]
    fn a_market_priced_below_consensus_is_bought_at_the_ask() {
        let sim = Sim::quoted(45, 55);
        let mut t = taker();
        let v = sim.view(None, Some(p(0.75)));
        let sig = t.evaluate(&v).expect("75c fair against a 55c ask is an edge");
        assert_eq!(sig.side, Side::Buy);
        assert_eq!(sig.price, Price::from_cents(55));
        assert!(sig.edge > 0.15, "edge {}", sig.edge);

        let acts = sim.step(&mut t, &v);
        let i = &places(&acts)[0];
        assert_eq!(i.tif, TimeInForce::Ioc, "a stale edge must not rest");
        assert_eq!(&*i.reason, "value");
    }

    #[test]
    fn a_market_priced_above_consensus_is_sold_at_the_bid() {
        let sim = Sim::quoted(45, 55);
        let t = taker();
        let sig = t.evaluate(&sim.view(None, Some(p(0.20)))).unwrap();
        assert_eq!(sig.side, Side::Sell);
        assert_eq!(sig.price, Price::from_cents(45));
    }

    #[test]
    fn a_thin_edge_is_left_alone() {
        // Fair 56c against a 55c ask: one cent, below the two-cent floor.
        let sim = Sim::quoted(45, 55);
        let t = taker();
        assert!(t.evaluate(&sim.view(None, Some(p(0.56)))).is_none());
    }

    #[test]
    fn fees_can_eat_an_edge_that_looks_real_gross() {
        let sim = Sim::quoted(45, 55);
        let t = ValueTaker::new(
            StrategyId(2),
            ValueConfig {
                min_edge: 0.02,
                min_ev_per_dollar: 0.0,
                ..Default::default()
            },
        );
        // 58c fair against a 55c ask is 3c gross — real, but thin.
        assert!(t.evaluate(&sim.view(None, Some(p(0.58)))).is_some());

        // The same edge under a punitive fee schedule is not a trade.
        let mut dear = Sim::quoted(45, 55);
        dear.spec.fee = edge_core::fees::FeeModel::Bps {
            maker_bps: 0.0,
            taker_bps: 400.0,
        };
        assert!(
            t.evaluate(&dear.view(None, Some(p(0.58)))).is_none(),
            "gross edge must not survive a fee that exceeds it"
        );
    }

    #[test]
    fn size_never_exceeds_what_is_resting_at_the_touch() {
        let mut sim = Sim::new();
        sim.rest(Side::Buy, 45, 500);
        sim.rest(Side::Sell, 55, 20); // only 20 offered
        let t = taker();
        let sig = t.evaluate(&sim.view(None, Some(p(0.90)))).unwrap();
        // Half of 20, by default max_touch_share.
        assert_eq!(sig.qty, Qty(10));
    }

    #[test]
    fn size_respects_the_position_cap() {
        let mut sim = Sim::quoted(45, 55);
        sim.position = Qty(495);
        let t = taker();
        let sig = t.evaluate(&sim.view(None, Some(p(0.90)))).unwrap();
        assert_eq!(sig.qty, Qty(5), "only 5 contracts of room left");

        sim.position = Qty(500);
        assert!(t.evaluate(&sim.view(None, Some(p(0.90)))).is_none());
    }

    #[test]
    fn size_scales_with_the_bankroll() {
        let mut small = Sim::quoted(45, 55);
        small.bankroll = 200.0;
        let big = Sim::quoted(45, 55);
        let t = taker();
        let a = t.evaluate(&small.view(None, Some(p(0.70)))).unwrap().qty;
        let b = t.evaluate(&big.view(None, Some(p(0.70)))).unwrap().qty;
        assert!(a < b, "{a:?} vs {b:?}");
    }

    #[test]
    fn a_position_is_closed_once_the_evidence_inverts() {
        let mut sim = Sim::quoted(45, 55);
        sim.position = Qty(100);
        let mut t = taker();

        // Still justified: fair well above the ask.
        assert!(t.exit(&sim.view(None, Some(p(0.80)))).is_none());

        // Inverted: fair far below where the position could be sold.
        let v = sim.view(None, Some(p(0.20)));
        let (side, price, qty) = t.exit(&v).expect("should close");
        assert_eq!(side, Side::Sell);
        assert_eq!(price, Price::from_cents(45));
        assert_eq!(qty, Qty(100));

        let acts = sim.step(&mut t, &v);
        assert_eq!(acts.len(), 1, "closing takes precedence over re-entering");
        assert_eq!(acts[0].reason(), "exit-thesis-inverted");
    }

    #[test]
    fn a_short_is_closed_by_buying_the_ask() {
        let mut sim = Sim::quoted(45, 55);
        sim.position = Qty(-80);
        let t = taker();
        let (side, price, qty) = t.exit(&sim.view(None, Some(p(0.90)))).unwrap();
        assert_eq!(side, Side::Buy);
        assert_eq!(price, Price::from_cents(55));
        assert_eq!(qty, Qty(80));
    }

    #[test]
    fn the_exit_band_is_wider_than_the_entry_band() {
        // A position whose edge has decayed to just past neutral is held, not
        // churned. Otherwise every position pays two spreads to end up flat.
        let mut sim = Sim::quoted(45, 55);
        sim.position = Qty(100);
        let t = taker();
        assert!(t.exit(&sim.view(None, Some(p(0.53)))).is_none());
    }

    #[test]
    fn nothing_opens_in_the_final_seconds() {
        let mut sim = Sim::quoted(45, 55);
        sim.spec.closes_at = Some(Ts(30_000_000_000)); // 30s
        let t = taker();
        assert!(t.evaluate(&sim.view(None, Some(p(0.90)))).is_none());
    }

    #[test]
    fn a_one_sided_book_is_not_traded() {
        let mut sim = Sim::new();
        sim.rest(Side::Sell, 55, 500);
        let t = taker();
        assert!(t.evaluate(&sim.view(None, Some(p(0.90)))).is_none());
    }
}
