//! Position and portfolio accounting.
//!
//! # Everything is prepaid
//!
//! Prediction markets do not let you short. On Kalshi and Polymarket alike,
//! taking the downside of an event means *buying the NO contract*, fully funded
//! up front. There is no margin, no borrow, and no liquidation.
//!
//! That is not a limitation to work around — it is the single most useful fact
//! about this asset class from a risk perspective, and modelling it faithfully
//! removes a whole category of risk machinery that a generic trading system
//! needs. A position's maximum loss is exactly what was paid for it, known at
//! trade time, with no assumptions about volatility or correlation. Where an
//! equities system estimates capital at risk, this one *knows* it.
//!
//! Positions are still stored with a signed quantity, because a strategy
//! reasons in terms of "long or short this event" and flipping between the two
//! should net rather than accumulate two positions. Negative quantity means
//! holding the NO leg; `avg_cost` is always the price paid **for the leg
//! actually held**, so `|qty| × avg_cost` is the capital at risk in both cases.

use std::collections::HashMap;

use edge_core::types::{EventId, MarketId, Price, Qty, Side};
use serde::{Deserialize, Serialize};

/// A holding in one market.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Position {
    pub market: MarketId,
    /// Signed. Positive holds YES, negative holds NO.
    pub qty: Qty,
    /// Average price paid per contract **of the leg held**, in dollars. For a
    /// short YES position this is the NO price, so it is always in `[0, 1]` and
    /// always equals the capital at risk per contract.
    pub avg_cost: f64,
    /// Closed-out profit and loss, net of the fees attributed to those closes.
    pub realized: f64,
    /// Total fees paid on this market, already deducted from `realized`.
    pub fees: f64,
    pub contracts_bought: i64,
    pub contracts_sold: i64,
}

impl Position {
    pub fn new(market: MarketId) -> Self {
        Position {
            market,
            qty: Qty::ZERO,
            avg_cost: 0.0,
            realized: 0.0,
            fees: 0.0,
            contracts_bought: 0,
            contracts_sold: 0,
        }
    }

    #[inline]
    pub fn is_flat(&self) -> bool {
        self.qty.get() == 0
    }

    /// Price of the leg held, given a YES mark.
    #[inline]
    fn leg_price(&self, yes_price: f64) -> f64 {
        if self.qty.get() >= 0 { yes_price } else { 1.0 - yes_price }
    }

    /// Mark-to-market value of the holding, in dollars.
    pub fn value(&self, mark: Price) -> f64 {
        self.qty.get().abs() as f64 * self.leg_price(mark.dollars())
    }

    /// Open profit and loss at `mark`.
    pub fn unrealized(&self, mark: Price) -> f64 {
        self.qty.get().abs() as f64 * (self.leg_price(mark.dollars()) - self.avg_cost)
    }

    /// The most this position can lose: everything paid for it.
    ///
    /// Exact, not estimated. This is what makes prediction-market risk
    /// tractable — the number is known at trade time and cannot be exceeded by
    /// any market move, gap, or correlation surprise.
    pub fn capital_at_risk(&self) -> f64 {
        self.qty.get().abs() as f64 * self.avg_cost
    }

    /// The most this position can make: the payout less what it cost.
    pub fn max_gain(&self) -> f64 {
        self.qty.get().abs() as f64 * (1.0 - self.avg_cost)
    }

    /// Apply a fill. `price` is always the YES price, whatever leg is traded.
    ///
    /// Returns realized profit and loss from the portion this fill closed.
    pub fn apply_fill(&mut self, side: Side, price: Price, qty: Qty, fee: f64) -> f64 {
        let size = qty.get().abs();
        if size == 0 {
            return 0.0;
        }
        let yes = price.dollars();
        self.fees += fee;

        match side {
            Side::Buy => self.contracts_bought += size,
            Side::Sell => self.contracts_sold += size,
        }

        // Direction this fill pushes the position, and the cost of the leg it
        // acquires if it is opening rather than closing.
        let signed = size * side.sign();
        let open_leg_price = if side == Side::Buy { yes } else { 1.0 - yes };

        let current = self.qty.get();
        let mut realized = 0.0;

        if current == 0 || (current > 0) == (signed > 0) {
            // Opening or adding: blend into the average.
            let total = current.abs() + size;
            self.avg_cost = (self.avg_cost * current.abs() as f64 + open_leg_price * size as f64)
                / total as f64;
            self.qty = Qty(current + signed);
        } else {
            // Reducing, closing, or flipping through zero.
            let closing = size.min(current.abs());
            // Closing the leg held means selling it, receiving its current price.
            let exit_price = if current > 0 { yes } else { 1.0 - yes };
            realized += closing as f64 * (exit_price - self.avg_cost);

            let remainder = size - closing;
            self.qty = Qty(current + signed);
            if remainder > 0 {
                // Flipped: the excess opens a fresh position on the other leg.
                self.avg_cost = open_leg_price;
            } else if self.qty.get() == 0 {
                self.avg_cost = 0.0;
            }
        }

        realized -= fee;
        self.realized += realized;
        realized
    }

    /// Resolve the market. `outcome` is whether YES settled true.
    ///
    /// Returns the profit and loss realised by settlement.
    pub fn settle(&mut self, outcome: bool) -> f64 {
        if self.is_flat() {
            return 0.0;
        }
        let settle_price = if outcome { Price::ONE } else { Price::ZERO };
        let pnl = self.unrealized(settle_price);
        self.realized += pnl;
        self.qty = Qty::ZERO;
        self.avg_cost = 0.0;
        pnl
    }
}

/// Every position, plus cash.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Portfolio {
    positions: HashMap<MarketId, Position>,
    /// Which event each market resolves on, so concentration can be measured
    /// against the thing that actually resolves rather than against a ticker.
    event_of: HashMap<MarketId, EventId>,
    /// Uncommitted collateral.
    pub cash: f64,
    /// Cash the account started with. The denominator for drawdown.
    pub starting_cash: f64,
    pub total_fees: f64,
    /// Highest equity ever marked. Drawdown is measured from here.
    peak_equity: f64,
}

impl Portfolio {
    pub fn new(starting_cash: f64) -> Self {
        Portfolio {
            positions: HashMap::new(),
            event_of: HashMap::new(),
            cash: starting_cash,
            starting_cash,
            total_fees: 0.0,
            peak_equity: starting_cash,
        }
    }

    /// Associate a market with the event it resolves on. Required for
    /// per-event concentration limits to mean anything.
    pub fn set_event(&mut self, market: MarketId, event: EventId) {
        self.event_of.insert(market, event);
    }

    pub fn position(&self, market: MarketId) -> Option<&Position> {
        self.positions.get(&market)
    }

    pub fn positions(&self) -> impl Iterator<Item = &Position> {
        self.positions.values()
    }

    /// Open positions only. Flat entries are retained for their realised
    /// history, so most callers want this.
    pub fn open_positions(&self) -> impl Iterator<Item = &Position> {
        self.positions.values().filter(|p| !p.is_flat())
    }

    pub fn qty(&self, market: MarketId) -> Qty {
        self.positions.get(&market).map(|p| p.qty).unwrap_or(Qty::ZERO)
    }

    /// Record a fill, moving cash and the position together.
    pub fn apply_fill(&mut self, market: MarketId, side: Side, price: Price, qty: Qty, fee: f64) {
        let size = qty.get().abs();
        if size == 0 {
            return;
        }
        let pos = self.positions.entry(market).or_insert_with(|| Position::new(market));
        let before = pos.qty.get();
        pos.apply_fill(side, price, qty, fee);
        let after = pos.qty.get();

        // Cash moves with the leg being acquired or released. Opening any
        // position is a full prepayment; closing returns the leg's market value.
        let yes = price.dollars();
        // A fill that flips the position through zero both closes the whole old
        // leg and opens a new one. Netting the absolute sizes would silently
        // report a single small close and lose the money on both sides of it.
        let flipped = before != 0 && after != 0 && (before > 0) != (after > 0);
        let (closed, opened) = if flipped {
            (before.abs(), after.abs())
        } else {
            ((before.abs() - after.abs()).max(0), (after.abs() - before.abs()).max(0))
        };
        let open_leg_price = if side == Side::Buy { yes } else { 1.0 - yes };
        let close_leg_price = if before > 0 { yes } else { 1.0 - yes };

        self.cash -= opened as f64 * open_leg_price;
        self.cash += closed as f64 * close_leg_price;
        self.cash -= fee;
        self.total_fees += fee;
    }

    /// Settle a market and release its capital back to cash.
    pub fn settle(&mut self, market: MarketId, outcome: bool) -> f64 {
        let Some(pos) = self.positions.get_mut(&market) else {
            return 0.0;
        };
        let held = pos.qty.get().abs();
        let cost = pos.capital_at_risk();
        let pnl = pos.settle(outcome);
        // The winning leg pays $1 per contract; the losing leg pays nothing.
        self.cash += cost + pnl;
        let _ = held;
        pnl
    }

    /// Total realised profit and loss, net of all fees.
    pub fn realized(&self) -> f64 {
        self.positions.values().map(|p| p.realized).sum()
    }

    /// Open profit and loss against a set of marks. Markets without a mark are
    /// valued at cost, which is neutral rather than optimistic.
    pub fn unrealized(&self, marks: &HashMap<MarketId, Price>) -> f64 {
        self.positions
            .values()
            .filter(|p| !p.is_flat())
            .map(|p| marks.get(&p.market).map(|m| p.unrealized(*m)).unwrap_or(0.0))
            .sum()
    }

    /// Cash plus the marked value of every open position.
    pub fn equity(&self, marks: &HashMap<MarketId, Price>) -> f64 {
        let held: f64 = self
            .positions
            .values()
            .filter(|p| !p.is_flat())
            .map(|p| match marks.get(&p.market) {
                Some(m) => p.value(*m),
                None => p.capital_at_risk(),
            })
            .sum();
        self.cash + held
    }

    /// Capital committed across all open positions — the exact maximum the
    /// portfolio can lose if every position resolves against it.
    pub fn capital_at_risk(&self) -> f64 {
        self.positions.values().map(|p| p.capital_at_risk()).sum()
    }

    /// Capital at risk on one event, summed across every market resolving on it.
    ///
    /// The number a concentration limit must be written against. Ten markets on
    /// ten different legs of the same game are one bet, and limiting each of
    /// them individually limits nothing.
    pub fn event_at_risk(&self, event: EventId) -> f64 {
        self.positions
            .values()
            .filter(|p| self.event_of.get(&p.market) == Some(&event))
            .map(|p| p.capital_at_risk())
            .sum()
    }

    pub fn event_of(&self, market: MarketId) -> Option<EventId> {
        self.event_of.get(&market).copied()
    }

    /// Signed exposure: positive when net long YES across the book.
    pub fn net_exposure(&self) -> f64 {
        self.positions.values().map(|p| p.qty.get() as f64 * p.avg_cost).sum()
    }

    /// Update the equity high-water mark. Call on every mark cycle.
    pub fn mark(&mut self, marks: &HashMap<MarketId, Price>) -> f64 {
        let e = self.equity(marks);
        if e > self.peak_equity {
            self.peak_equity = e;
        }
        e
    }

    pub fn peak_equity(&self) -> f64 {
        self.peak_equity
    }

    /// Fractional decline from the equity high-water mark.
    pub fn drawdown(&self, marks: &HashMap<MarketId, Price>) -> f64 {
        if self.peak_equity <= 0.0 {
            return 0.0;
        }
        ((self.peak_equity - self.equity(marks)) / self.peak_equity).max(0.0)
    }

    pub fn open_count(&self) -> usize {
        self.positions.values().filter(|p| !p.is_flat()).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use edge_core::types::MarketId;

    const M: MarketId = MarketId(1);
    const N: MarketId = MarketId(2);

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-9, "{a} != {b}");
    }

    #[test]
    fn a_long_position_records_what_it_paid() {
        let mut p = Position::new(M);
        p.apply_fill(Side::Buy, Price::from_cents(40), Qty(100), 0.0);
        assert_eq!(p.qty, Qty(100));
        approx(p.avg_cost, 0.40);
        approx(p.capital_at_risk(), 40.0);
        approx(p.max_gain(), 60.0);
        approx(p.value(Price::from_cents(40)), 40.0);
        approx(p.unrealized(Price::from_cents(40)), 0.0);
    }

    #[test]
    fn a_short_position_is_the_no_leg() {
        // Selling YES at 40c is buying NO at 60c. Capital at risk is the 60c
        // paid, not the 40c received.
        let mut p = Position::new(M);
        p.apply_fill(Side::Sell, Price::from_cents(40), Qty(100), 0.0);
        assert_eq!(p.qty, Qty(-100));
        approx(p.avg_cost, 0.60);
        approx(p.capital_at_risk(), 60.0);
        approx(p.max_gain(), 40.0);
    }

    #[test]
    fn averaging_in_blends_the_cost() {
        let mut p = Position::new(M);
        p.apply_fill(Side::Buy, Price::from_cents(40), Qty(100), 0.0);
        p.apply_fill(Side::Buy, Price::from_cents(60), Qty(100), 0.0);
        assert_eq!(p.qty, Qty(200));
        approx(p.avg_cost, 0.50);
        approx(p.capital_at_risk(), 100.0);
    }

    #[test]
    fn unrealized_moves_with_the_mark_on_both_legs() {
        let mut long = Position::new(M);
        long.apply_fill(Side::Buy, Price::from_cents(40), Qty(100), 0.0);
        approx(long.unrealized(Price::from_cents(55)), 15.0);
        approx(long.unrealized(Price::from_cents(30)), -10.0);

        let mut short = Position::new(M);
        short.apply_fill(Side::Sell, Price::from_cents(40), Qty(100), 0.0);
        // NO bought at 60c, YES falls to 30c so NO is worth 70c: +10.
        approx(short.unrealized(Price::from_cents(30)), 10.0);
        approx(short.unrealized(Price::from_cents(55)), -15.0);
    }

    #[test]
    fn closing_realizes_the_difference() {
        let mut p = Position::new(M);
        p.apply_fill(Side::Buy, Price::from_cents(40), Qty(100), 0.0);
        let realized = p.apply_fill(Side::Sell, Price::from_cents(55), Qty(60), 0.0);
        approx(realized, 9.0); // 60 contracts x 15c
        assert_eq!(p.qty, Qty(40));
        approx(p.avg_cost, 0.40);
        approx(p.realized, 9.0);
    }

    #[test]
    fn closing_out_entirely_leaves_no_cost_basis() {
        let mut p = Position::new(M);
        p.apply_fill(Side::Buy, Price::from_cents(40), Qty(100), 0.0);
        p.apply_fill(Side::Sell, Price::from_cents(50), Qty(100), 0.0);
        assert!(p.is_flat());
        approx(p.avg_cost, 0.0);
        approx(p.capital_at_risk(), 0.0);
        approx(p.realized, 10.0);
    }

    #[test]
    fn flipping_through_zero_closes_then_reopens() {
        let mut p = Position::new(M);
        p.apply_fill(Side::Buy, Price::from_cents(40), Qty(100), 0.0);
        // Sell 150: closes the 100 long at 50c (+$10), opens 50 short.
        let realized = p.apply_fill(Side::Sell, Price::from_cents(50), Qty(150), 0.0);
        approx(realized, 10.0);
        assert_eq!(p.qty, Qty(-50));
        approx(p.avg_cost, 0.50); // the NO leg at 1 - 0.50
        approx(p.capital_at_risk(), 25.0);
    }

    #[test]
    fn fees_reduce_realized_pnl() {
        let mut p = Position::new(M);
        p.apply_fill(Side::Buy, Price::from_cents(40), Qty(100), 1.75);
        let realized = p.apply_fill(Side::Sell, Price::from_cents(50), Qty(100), 1.75);
        approx(realized, 10.0 - 1.75);
        approx(p.realized, 10.0 - 3.50);
        approx(p.fees, 3.50);
    }

    #[test]
    fn settlement_pays_the_winning_leg() {
        let mut win = Position::new(M);
        win.apply_fill(Side::Buy, Price::from_cents(40), Qty(100), 0.0);
        approx(win.settle(true), 60.0);
        assert!(win.is_flat());

        let mut lose = Position::new(M);
        lose.apply_fill(Side::Buy, Price::from_cents(40), Qty(100), 0.0);
        approx(lose.settle(false), -40.0);

        // A short wins when YES loses.
        let mut short = Position::new(M);
        short.apply_fill(Side::Sell, Price::from_cents(40), Qty(100), 0.0);
        approx(short.settle(false), 40.0);
        // ...and loses its full 60c cost when YES wins.
        let mut short2 = Position::new(M);
        short2.apply_fill(Side::Sell, Price::from_cents(40), Qty(100), 0.0);
        approx(short2.settle(true), -60.0);
    }

    #[test]
    fn a_loss_never_exceeds_capital_at_risk() {
        // The defining property of a prepaid market, asserted directly.
        for (side, cents) in [(Side::Buy, 40), (Side::Sell, 40), (Side::Buy, 95), (Side::Sell, 5)] {
            for outcome in [true, false] {
                let mut p = Position::new(M);
                p.apply_fill(side, Price::from_cents(cents), Qty(100), 0.0);
                let risk = p.capital_at_risk();
                let pnl = p.settle(outcome);
                assert!(
                    pnl >= -risk - 1e-9,
                    "{side:?} at {cents}c resolving {outcome} lost {pnl}, more than the {risk} paid"
                );
            }
        }
    }

    #[test]
    fn settling_a_flat_position_is_a_no_op() {
        let mut p = Position::new(M);
        approx(p.settle(true), 0.0);
    }

    // -- portfolio --------------------------------------------------------

    #[test]
    fn cash_falls_by_what_a_position_cost() {
        let mut pf = Portfolio::new(1_000.0);
        pf.apply_fill(M, Side::Buy, Price::from_cents(40), Qty(100), 0.0);
        approx(pf.cash, 960.0);
        approx(pf.capital_at_risk(), 40.0);

        // A short costs the NO leg: 100 x 60c.
        pf.apply_fill(N, Side::Sell, Price::from_cents(40), Qty(100), 0.0);
        approx(pf.cash, 900.0);
    }

    #[test]
    fn equity_is_unchanged_by_a_trade_at_the_mark() {
        // Buying at the current mark converts cash into position value and
        // must leave equity flat. If it does not, the accounting is wrong.
        let mut pf = Portfolio::new(1_000.0);
        let marks = HashMap::from([(M, Price::from_cents(40))]);
        approx(pf.equity(&marks), 1_000.0);
        pf.apply_fill(M, Side::Buy, Price::from_cents(40), Qty(100), 0.0);
        approx(pf.equity(&marks), 1_000.0);
    }

    #[test]
    fn equity_tracks_the_mark() {
        let mut pf = Portfolio::new(1_000.0);
        pf.apply_fill(M, Side::Buy, Price::from_cents(40), Qty(100), 0.0);
        let up = HashMap::from([(M, Price::from_cents(60))]);
        approx(pf.equity(&up), 1_020.0);
        approx(pf.unrealized(&up), 20.0);
    }

    #[test]
    fn closing_returns_capital_to_cash() {
        let mut pf = Portfolio::new(1_000.0);
        pf.apply_fill(M, Side::Buy, Price::from_cents(40), Qty(100), 0.0);
        pf.apply_fill(M, Side::Sell, Price::from_cents(55), Qty(100), 0.0);
        approx(pf.cash, 1_015.0);
        approx(pf.realized(), 15.0);
        approx(pf.capital_at_risk(), 0.0);
        assert_eq!(pf.open_count(), 0);
    }

    #[test]
    fn settlement_returns_the_payout_to_cash() {
        let mut pf = Portfolio::new(1_000.0);
        pf.apply_fill(M, Side::Buy, Price::from_cents(40), Qty(100), 0.0);
        approx(pf.settle(M, true), 60.0);
        // Paid $40, received $100.
        approx(pf.cash, 1_060.0);
        approx(pf.capital_at_risk(), 0.0);

        let mut lost = Portfolio::new(1_000.0);
        lost.apply_fill(M, Side::Buy, Price::from_cents(40), Qty(100), 0.0);
        lost.settle(M, false);
        approx(lost.cash, 960.0);
    }

    #[test]
    fn fees_come_straight_out_of_cash() {
        let mut pf = Portfolio::new(1_000.0);
        pf.apply_fill(M, Side::Buy, Price::from_cents(40), Qty(100), 1.75);
        approx(pf.cash, 958.25);
        approx(pf.total_fees, 1.75);
    }

    #[test]
    fn event_concentration_aggregates_across_markets() {
        // Two markets, same game. Individually each is small; together they are
        // one bet, and only the event-level number says so.
        let mut pf = Portfolio::new(1_000.0);
        pf.set_event(M, EventId(9));
        pf.set_event(N, EventId(9));
        pf.apply_fill(M, Side::Buy, Price::from_cents(40), Qty(100), 0.0);
        pf.apply_fill(N, Side::Buy, Price::from_cents(30), Qty(100), 0.0);
        approx(pf.event_at_risk(EventId(9)), 70.0);
        approx(pf.event_at_risk(EventId(1)), 0.0);
    }

    #[test]
    fn net_exposure_signs_correctly() {
        let mut pf = Portfolio::new(1_000.0);
        pf.apply_fill(M, Side::Buy, Price::from_cents(40), Qty(100), 0.0);
        assert!(pf.net_exposure() > 0.0);
        pf.apply_fill(N, Side::Sell, Price::from_cents(40), Qty(200), 0.0);
        assert!(pf.net_exposure() < 0.0, "net short overall");
    }

    #[test]
    fn drawdown_measures_from_the_high_water_mark() {
        let mut pf = Portfolio::new(1_000.0);
        pf.apply_fill(M, Side::Buy, Price::from_cents(40), Qty(1000), 0.0);

        let up = HashMap::from([(M, Price::from_cents(60))]);
        pf.mark(&up);
        approx(pf.peak_equity(), 1_200.0);
        approx(pf.drawdown(&up), 0.0);

        let down = HashMap::from([(M, Price::from_cents(30))]);
        pf.mark(&down);
        // Equity 900 against a 1200 peak.
        approx(pf.drawdown(&down), 0.25);
        approx(pf.peak_equity(), 1_200.0);
    }

    #[test]
    fn an_unmarked_position_is_valued_at_cost_not_at_zero() {
        let mut pf = Portfolio::new(1_000.0);
        pf.apply_fill(M, Side::Buy, Price::from_cents(40), Qty(100), 0.0);
        // No marks at all: equity must not collapse to cash.
        approx(pf.equity(&HashMap::new()), 1_000.0);
    }

    #[test]
    fn a_full_round_trip_conserves_money() {
        // Buy, add, partially close, flip, settle. Cash plus realised must
        // reconcile against the starting bankroll at every step.
        let mut pf = Portfolio::new(1_000.0);
        pf.apply_fill(M, Side::Buy, Price::from_cents(40), Qty(100), 0.0);
        pf.apply_fill(M, Side::Buy, Price::from_cents(50), Qty(100), 0.0);
        pf.apply_fill(M, Side::Sell, Price::from_cents(60), Qty(50), 0.0);
        pf.apply_fill(M, Side::Sell, Price::from_cents(55), Qty(200), 0.0);
        pf.settle(M, false);

        assert!(pf.position(M).unwrap().is_flat());
        approx(pf.cash, 1_000.0 + pf.realized());
        approx(pf.capital_at_risk(), 0.0);
    }
}
