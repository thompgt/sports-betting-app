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
//! holding the NO leg.
//!
//! # Money is an integer
//!
//! Every monetary quantity here is a [`Notional`] — signed micro-dollars — for
//! the same reason [`Price`] is: a book and a PnL held in `f64` drift, and the
//! drift is invisible precisely where it matters. `0.1 + 0.2 != 0.3` is a
//! rounding curiosity in a report and a reconciliation break in a ledger.
//!
//! The consequence that shapes the code below is that a position stores its
//! **total cost basis**, not an average price. An average is a division, and a
//! division of integers either rounds — putting the error straight into the
//! number that limits are written against — or forces the basis to be a float
//! and reintroduces the drift. Storing the total makes every full close, flip
//! and settlement conserve money *exactly*; [`Position::avg_cost`] derives the
//! per-contract price on demand, for humans and for display.
//!
//! Partial closes remove cost basis proportionally, which is the one place a
//! rounding decision is unavoidable. It truncates, so any sub-micro-dollar
//! residue stays with the remaining position rather than being conjured into
//! realised PnL — money is never created, and a full close still zeroes the
//! basis to the micro-dollar.

use std::collections::HashMap;

use edge_core::types::{EventId, MICROS, MarketId, Notional, Price, Qty, Side};
use serde::{Deserialize, Serialize};

/// `a * b` in micro-dollars, via `i128` so an intermediate product cannot wrap.
/// A million contracts at a dollar is 10^12 micro-dollars, and the products
/// below multiply that by a quantity again.
#[inline]
fn mul(price_micros: i64, contracts: i64) -> Notional {
    Notional((price_micros as i128 * contracts as i128) as i64)
}

/// A holding in one market.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Position {
    pub market: MarketId,
    /// Signed. Positive holds YES, negative holds NO.
    pub qty: Qty,
    /// Total paid for the leg currently held — and therefore exactly the
    /// capital at risk, since a prepaid contract cannot lose more than it cost.
    ///
    /// Stored as a total rather than as an average price so that closes and
    /// flips conserve money exactly; see the module docs.
    cost_basis: Notional,
    /// Closed-out profit and loss, net of the fees attributed to those closes.
    realized: Notional,
    /// Total fees paid on this market, already deducted from `realized`.
    fees: Notional,
    pub contracts_bought: i64,
    pub contracts_sold: i64,
}

impl Position {
    pub fn new(market: MarketId) -> Self {
        Position {
            market,
            qty: Qty::ZERO,
            cost_basis: Notional::ZERO,
            realized: Notional::ZERO,
            fees: Notional::ZERO,
            contracts_bought: 0,
            contracts_sold: 0,
        }
    }

    #[inline]
    pub fn is_flat(&self) -> bool {
        self.qty.get() == 0
    }

    /// Closed-out profit and loss, net of fees.
    #[inline]
    pub fn realized(&self) -> Notional {
        self.realized
    }

    /// Fees paid on this market, already deducted from [`Position::realized`].
    #[inline]
    pub fn fees(&self) -> Notional {
        self.fees
    }

    /// Average price paid per contract **of the leg held**. For a short YES
    /// position this is the NO price, so it is always in `[0, 1]`.
    ///
    /// Derived, not stored: it is a reporting convenience, and rounding it into
    /// the stored state is what makes float accounting drift. Use
    /// [`Position::capital_at_risk`] for anything that has to reconcile.
    pub fn avg_cost(&self) -> Price {
        let held = self.qty.get().abs();
        if held == 0 {
            return Price::ZERO;
        }
        Price::from_micros(self.cost_basis.0 / held)
    }

    /// Price of the leg held, given a YES mark, in micro-dollars.
    #[inline]
    fn leg_price(&self, yes_micros: i64) -> i64 {
        if self.qty.get() >= 0 { yes_micros } else { MICROS - yes_micros }
    }

    /// Mark-to-market value of the holding.
    pub fn value(&self, mark: Price) -> Notional {
        mul(self.leg_price(mark.micros()), self.qty.get().abs())
    }

    /// Open profit and loss at `mark`.
    pub fn unrealized(&self, mark: Price) -> Notional {
        self.value(mark) - self.cost_basis
    }

    /// The most this position can lose: everything paid for it.
    ///
    /// Exact, not estimated. This is what makes prediction-market risk
    /// tractable — the number is known at trade time and cannot be exceeded by
    /// any market move, gap, or correlation surprise.
    #[inline]
    pub fn capital_at_risk(&self) -> Notional {
        self.cost_basis
    }

    /// The most this position can make: the payout less what it cost.
    pub fn max_gain(&self) -> Notional {
        mul(MICROS, self.qty.get().abs()) - self.cost_basis
    }

    /// Apply a fill. `price` is always the YES price, whatever leg is traded.
    ///
    /// Returns realized profit and loss from the portion this fill closed.
    pub fn apply_fill(&mut self, side: Side, price: Price, qty: Qty, fee: Notional) -> Notional {
        let size = qty.get().abs();
        if size == 0 {
            return Notional::ZERO;
        }
        let yes = price.micros();
        self.fees += fee;

        match side {
            Side::Buy => self.contracts_bought += size,
            Side::Sell => self.contracts_sold += size,
        }

        // Direction this fill pushes the position, and the price of the leg it
        // acquires if it is opening rather than closing.
        let signed = size * side.sign();
        let open_leg_price = if side == Side::Buy { yes } else { MICROS - yes };

        let current = self.qty.get();
        let mut realized = Notional::ZERO;

        if current == 0 || (current > 0) == (signed > 0) {
            // Opening or adding: the basis is simply what this leg cost.
            self.cost_basis += mul(open_leg_price, size);
            self.qty = Qty(current + signed);
        } else {
            // Reducing, closing, or flipping through zero.
            let held = current.abs();
            let closing = size.min(held);
            // Closing the leg held means selling it, receiving its current price.
            let exit_price = if current > 0 { yes } else { MICROS - yes };

            // Remove basis in proportion to the contracts closed. Truncating
            // leaves any sub-micro-dollar residue in the surviving basis rather
            // than releasing it as profit; a full close divides exactly and so
            // zeroes the basis outright.
            let removed =
                Notional((self.cost_basis.0 as i128 * closing as i128 / held as i128) as i64);
            realized += mul(exit_price, closing) - removed;
            self.cost_basis -= removed;

            let remainder = size - closing;
            self.qty = Qty(current + signed);
            if remainder > 0 {
                // Flipped: the excess opens a fresh position on the other leg.
                self.cost_basis = mul(open_leg_price, remainder);
            } else if self.qty.get() == 0 {
                self.cost_basis = Notional::ZERO;
            }
        }

        realized -= fee;
        self.realized += realized;
        realized
    }

    /// Resolve the market. `outcome` is whether YES settled true.
    ///
    /// Returns the profit and loss realised by settlement.
    pub fn settle(&mut self, outcome: bool) -> Notional {
        if self.is_flat() {
            return Notional::ZERO;
        }
        let settle_price = if outcome { Price::ONE } else { Price::ZERO };
        let pnl = self.unrealized(settle_price);
        self.realized += pnl;
        self.qty = Qty::ZERO;
        self.cost_basis = Notional::ZERO;
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
    pub cash: Notional,
    /// Cash the account started with. The denominator for drawdown.
    pub starting_cash: Notional,
    pub total_fees: Notional,
    /// Highest equity ever marked. Drawdown is measured from here.
    peak_equity: Notional,
}

impl Portfolio {
    pub fn new(starting_cash: Notional) -> Self {
        Portfolio {
            positions: HashMap::new(),
            event_of: HashMap::new(),
            cash: starting_cash,
            starting_cash,
            total_fees: Notional::ZERO,
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
    pub fn apply_fill(
        &mut self,
        market: MarketId,
        side: Side,
        price: Price,
        qty: Qty,
        fee: Notional,
    ) {
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
        let yes = price.micros();
        // A fill that flips the position through zero both closes the whole old
        // leg and opens a new one. Netting the absolute sizes would silently
        // report a single small close and lose the money on both sides of it.
        let flipped = before != 0 && after != 0 && (before > 0) != (after > 0);
        let (closed, opened) = if flipped {
            (before.abs(), after.abs())
        } else {
            ((before.abs() - after.abs()).max(0), (after.abs() - before.abs()).max(0))
        };
        let open_leg_price = if side == Side::Buy { yes } else { MICROS - yes };
        let close_leg_price = if before > 0 { yes } else { MICROS - yes };

        self.cash -= mul(open_leg_price, opened);
        self.cash += mul(close_leg_price, closed);
        self.cash -= fee;
        self.total_fees += fee;
    }

    /// Settle a market and release its capital back to cash.
    pub fn settle(&mut self, market: MarketId, outcome: bool) -> Notional {
        let Some(pos) = self.positions.get_mut(&market) else {
            return Notional::ZERO;
        };
        let cost = pos.capital_at_risk();
        let pnl = pos.settle(outcome);
        // The winning leg pays $1 per contract; the losing leg pays nothing.
        // Either way the capital that was locked up comes back adjusted by the
        // result, which is exactly cost + pnl.
        self.cash += cost + pnl;
        pnl
    }

    /// Total realised profit and loss, net of all fees.
    pub fn realized(&self) -> Notional {
        self.positions.values().map(|p| p.realized).fold(Notional::ZERO, |a, b| a + b)
    }

    /// Open profit and loss against a set of marks. Markets without a mark are
    /// valued at cost, which is neutral rather than optimistic.
    pub fn unrealized(&self, marks: &HashMap<MarketId, Price>) -> Notional {
        self.positions
            .values()
            .filter(|p| !p.is_flat())
            .map(|p| marks.get(&p.market).map(|m| p.unrealized(*m)).unwrap_or(Notional::ZERO))
            .fold(Notional::ZERO, |a, b| a + b)
    }

    /// Cash plus the marked value of every open position.
    pub fn equity(&self, marks: &HashMap<MarketId, Price>) -> Notional {
        let held = self
            .positions
            .values()
            .filter(|p| !p.is_flat())
            .map(|p| match marks.get(&p.market) {
                Some(m) => p.value(*m),
                None => p.capital_at_risk(),
            })
            .fold(Notional::ZERO, |a, b| a + b);
        self.cash + held
    }

    /// Capital committed across all open positions — the exact maximum the
    /// portfolio can lose if every position resolves against it.
    pub fn capital_at_risk(&self) -> Notional {
        self.positions.values().map(|p| p.capital_at_risk()).fold(Notional::ZERO, |a, b| a + b)
    }

    /// Capital at risk on one event, summed across every market resolving on it.
    ///
    /// The number a concentration limit must be written against. Ten markets on
    /// ten different legs of the same game are one bet, and limiting each of
    /// them individually limits nothing.
    pub fn event_at_risk(&self, event: EventId) -> Notional {
        self.positions
            .values()
            .filter(|p| self.event_of.get(&p.market) == Some(&event))
            .map(|p| p.capital_at_risk())
            .fold(Notional::ZERO, |a, b| a + b)
    }

    pub fn event_of(&self, market: MarketId) -> Option<EventId> {
        self.event_of.get(&market).copied()
    }

    /// Signed exposure: positive when net long YES across the book.
    pub fn net_exposure(&self) -> Notional {
        self.positions
            .values()
            .map(|p| if p.qty.get() >= 0 { p.capital_at_risk() } else { -p.capital_at_risk() })
            .fold(Notional::ZERO, |a, b| a + b)
    }

    /// Update the equity high-water mark. Call on every mark cycle.
    pub fn mark(&mut self, marks: &HashMap<MarketId, Price>) -> Notional {
        let e = self.equity(marks);
        if e > self.peak_equity {
            self.peak_equity = e;
        }
        e
    }

    pub fn peak_equity(&self) -> Notional {
        self.peak_equity
    }

    /// Fractional decline from the equity high-water mark.
    ///
    /// A ratio rather than an amount, so this one is legitimately a float.
    pub fn drawdown(&self, marks: &HashMap<MarketId, Price>) -> f64 {
        if self.peak_equity.0 <= 0 {
            return 0.0;
        }
        let peak = self.peak_equity.0 as f64;
        ((peak - self.equity(marks).0 as f64) / peak).max(0.0)
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

    /// Dollars as money. Every assertion below is exact: integer accounting
    /// that needs a tolerance is integer accounting that is wrong somewhere.
    fn d(x: f64) -> Notional {
        Notional::from_dollars(x)
    }

    fn cents(c: i64) -> Price {
        Price::from_cents(c)
    }

    #[test]
    fn a_long_position_records_what_it_paid() {
        let mut p = Position::new(M);
        p.apply_fill(Side::Buy, cents(40), Qty(100), Notional::ZERO);
        assert_eq!(p.qty, Qty(100));
        assert_eq!(p.avg_cost(), cents(40));
        assert_eq!(p.capital_at_risk(), d(40.0));
        assert_eq!(p.max_gain(), d(60.0));
        assert_eq!(p.value(cents(40)), d(40.0));
        assert_eq!(p.unrealized(cents(40)), Notional::ZERO);
    }

    #[test]
    fn a_short_position_is_the_no_leg() {
        // Selling YES at 40c is buying NO at 60c. Capital at risk is the 60c
        // paid, not the 40c received.
        let mut p = Position::new(M);
        p.apply_fill(Side::Sell, cents(40), Qty(100), Notional::ZERO);
        assert_eq!(p.qty, Qty(-100));
        assert_eq!(p.avg_cost(), cents(60));
        assert_eq!(p.capital_at_risk(), d(60.0));
        assert_eq!(p.max_gain(), d(40.0));
    }

    #[test]
    fn averaging_in_blends_the_cost() {
        let mut p = Position::new(M);
        p.apply_fill(Side::Buy, cents(40), Qty(100), Notional::ZERO);
        p.apply_fill(Side::Buy, cents(60), Qty(100), Notional::ZERO);
        assert_eq!(p.qty, Qty(200));
        assert_eq!(p.avg_cost(), cents(50));
        assert_eq!(p.capital_at_risk(), d(100.0));
    }

    #[test]
    fn unrealized_moves_with_the_mark_on_both_legs() {
        let mut long = Position::new(M);
        long.apply_fill(Side::Buy, cents(40), Qty(100), Notional::ZERO);
        assert_eq!(long.unrealized(cents(55)), d(15.0));
        assert_eq!(long.unrealized(cents(30)), d(-10.0));

        let mut short = Position::new(M);
        short.apply_fill(Side::Sell, cents(40), Qty(100), Notional::ZERO);
        // NO bought at 60c, YES falls to 30c so NO is worth 70c: +10.
        assert_eq!(short.unrealized(cents(30)), d(10.0));
        assert_eq!(short.unrealized(cents(55)), d(-15.0));
    }

    #[test]
    fn closing_realizes_the_difference() {
        let mut p = Position::new(M);
        p.apply_fill(Side::Buy, cents(40), Qty(100), Notional::ZERO);
        let realized = p.apply_fill(Side::Sell, cents(55), Qty(60), Notional::ZERO);
        assert_eq!(realized, d(9.0)); // 60 contracts x 15c
        assert_eq!(p.qty, Qty(40));
        assert_eq!(p.avg_cost(), cents(40));
        assert_eq!(p.realized(), d(9.0));
    }

    #[test]
    fn closing_out_entirely_leaves_no_cost_basis() {
        let mut p = Position::new(M);
        p.apply_fill(Side::Buy, cents(40), Qty(100), Notional::ZERO);
        p.apply_fill(Side::Sell, cents(50), Qty(100), Notional::ZERO);
        assert!(p.is_flat());
        assert_eq!(p.avg_cost(), Price::ZERO);
        assert_eq!(p.capital_at_risk(), Notional::ZERO);
        assert_eq!(p.realized(), d(10.0));
    }

    #[test]
    fn flipping_through_zero_closes_then_reopens() {
        let mut p = Position::new(M);
        p.apply_fill(Side::Buy, cents(40), Qty(100), Notional::ZERO);
        // Sell 150: closes the 100 long at 50c (+$10), opens 50 short.
        let realized = p.apply_fill(Side::Sell, cents(50), Qty(150), Notional::ZERO);
        assert_eq!(realized, d(10.0));
        assert_eq!(p.qty, Qty(-50));
        assert_eq!(p.avg_cost(), cents(50)); // the NO leg at 1 - 0.50
        assert_eq!(p.capital_at_risk(), d(25.0));
    }

    #[test]
    fn fees_reduce_realized_pnl() {
        let mut p = Position::new(M);
        p.apply_fill(Side::Buy, cents(40), Qty(100), d(1.75));
        let realized = p.apply_fill(Side::Sell, cents(50), Qty(100), d(1.75));
        assert_eq!(realized, d(10.0) - d(1.75));
        assert_eq!(p.realized(), d(10.0) - d(3.50));
        assert_eq!(p.fees(), d(3.50));
    }

    #[test]
    fn settlement_pays_the_winning_leg() {
        let mut win = Position::new(M);
        win.apply_fill(Side::Buy, cents(40), Qty(100), Notional::ZERO);
        assert_eq!(win.settle(true), d(60.0));
        assert!(win.is_flat());

        let mut lose = Position::new(M);
        lose.apply_fill(Side::Buy, cents(40), Qty(100), Notional::ZERO);
        assert_eq!(lose.settle(false), d(-40.0));

        // A short wins when YES loses.
        let mut short = Position::new(M);
        short.apply_fill(Side::Sell, cents(40), Qty(100), Notional::ZERO);
        assert_eq!(short.settle(false), d(40.0));
        // ...and loses its full 60c cost when YES wins.
        let mut short2 = Position::new(M);
        short2.apply_fill(Side::Sell, cents(40), Qty(100), Notional::ZERO);
        assert_eq!(short2.settle(true), d(-60.0));
    }

    #[test]
    fn a_loss_never_exceeds_capital_at_risk() {
        // The defining property of a prepaid market, asserted exactly — with
        // integer money there is no epsilon to hide behind.
        for (side, c) in [(Side::Buy, 40), (Side::Sell, 40), (Side::Buy, 95), (Side::Sell, 5)] {
            for outcome in [true, false] {
                let mut p = Position::new(M);
                p.apply_fill(side, cents(c), Qty(100), Notional::ZERO);
                let risk = p.capital_at_risk();
                let pnl = p.settle(outcome);
                assert!(
                    pnl >= -risk,
                    "{side:?} at {c}c resolving {outcome} lost {pnl}, more than the {risk} paid"
                );
            }
        }
    }

    #[test]
    fn settling_a_flat_position_is_a_no_op() {
        let mut p = Position::new(M);
        assert_eq!(p.settle(true), Notional::ZERO);
    }

    #[test]
    fn a_partial_close_never_conjures_money_out_of_rounding() {
        // Three contracts bought at a price whose basis does not divide by
        // three. The residue must stay with the surviving position rather than
        // being released as profit.
        let mut p = Position::new(M);
        p.apply_fill(Side::Buy, Price::from_micros(333_333), Qty(3), Notional::ZERO);
        let basis = p.capital_at_risk();
        assert_eq!(basis, Notional(999_999));

        let realized =
            p.apply_fill(Side::Sell, Price::from_micros(333_333), Qty(1), Notional::ZERO);
        // Sold at cost: no profit, to the micro-dollar.
        assert_eq!(realized, Notional::ZERO);
        assert_eq!(p.capital_at_risk() + Notional(333_333), basis, "a micro-dollar went missing");

        // And closing the rest zeroes the basis exactly.
        p.apply_fill(Side::Sell, Price::from_micros(333_333), Qty(2), Notional::ZERO);
        assert!(p.is_flat());
        assert_eq!(p.capital_at_risk(), Notional::ZERO);
    }

    // -- portfolio --------------------------------------------------------

    #[test]
    fn cash_falls_by_what_a_position_cost() {
        let mut pf = Portfolio::new(d(1_000.0));
        pf.apply_fill(M, Side::Buy, cents(40), Qty(100), Notional::ZERO);
        assert_eq!(pf.cash, d(960.0));
        assert_eq!(pf.capital_at_risk(), d(40.0));

        // A short costs the NO leg: 100 x 60c.
        pf.apply_fill(N, Side::Sell, cents(40), Qty(100), Notional::ZERO);
        assert_eq!(pf.cash, d(900.0));
    }

    #[test]
    fn equity_is_unchanged_by_a_trade_at_the_mark() {
        // Buying at the current mark converts cash into position value and
        // must leave equity flat. If it does not, the accounting is wrong.
        let mut pf = Portfolio::new(d(1_000.0));
        let marks = HashMap::from([(M, cents(40))]);
        assert_eq!(pf.equity(&marks), d(1_000.0));
        pf.apply_fill(M, Side::Buy, cents(40), Qty(100), Notional::ZERO);
        assert_eq!(pf.equity(&marks), d(1_000.0));
    }

    #[test]
    fn equity_tracks_the_mark() {
        let mut pf = Portfolio::new(d(1_000.0));
        pf.apply_fill(M, Side::Buy, cents(40), Qty(100), Notional::ZERO);
        let up = HashMap::from([(M, cents(60))]);
        assert_eq!(pf.equity(&up), d(1_020.0));
        assert_eq!(pf.unrealized(&up), d(20.0));
    }

    #[test]
    fn closing_returns_capital_to_cash() {
        let mut pf = Portfolio::new(d(1_000.0));
        pf.apply_fill(M, Side::Buy, cents(40), Qty(100), Notional::ZERO);
        pf.apply_fill(M, Side::Sell, cents(55), Qty(100), Notional::ZERO);
        assert_eq!(pf.cash, d(1_015.0));
        assert_eq!(pf.realized(), d(15.0));
        assert_eq!(pf.capital_at_risk(), Notional::ZERO);
        assert_eq!(pf.open_count(), 0);
    }

    #[test]
    fn settlement_returns_the_payout_to_cash() {
        let mut pf = Portfolio::new(d(1_000.0));
        pf.apply_fill(M, Side::Buy, cents(40), Qty(100), Notional::ZERO);
        assert_eq!(pf.settle(M, true), d(60.0));
        // Paid $40, received $100.
        assert_eq!(pf.cash, d(1_060.0));
        assert_eq!(pf.capital_at_risk(), Notional::ZERO);

        let mut lost = Portfolio::new(d(1_000.0));
        lost.apply_fill(M, Side::Buy, cents(40), Qty(100), Notional::ZERO);
        lost.settle(M, false);
        assert_eq!(lost.cash, d(960.0));
    }

    #[test]
    fn fees_come_straight_out_of_cash() {
        let mut pf = Portfolio::new(d(1_000.0));
        pf.apply_fill(M, Side::Buy, cents(40), Qty(100), d(1.75));
        assert_eq!(pf.cash, d(958.25));
        assert_eq!(pf.total_fees, d(1.75));
    }

    #[test]
    fn event_concentration_aggregates_across_markets() {
        // Two markets, same game. Individually each is small; together they are
        // one bet, and only the event-level number says so.
        let mut pf = Portfolio::new(d(1_000.0));
        pf.set_event(M, EventId(9));
        pf.set_event(N, EventId(9));
        pf.apply_fill(M, Side::Buy, cents(40), Qty(100), Notional::ZERO);
        pf.apply_fill(N, Side::Buy, cents(30), Qty(100), Notional::ZERO);
        assert_eq!(pf.event_at_risk(EventId(9)), d(70.0));
        assert_eq!(pf.event_at_risk(EventId(1)), Notional::ZERO);
    }

    #[test]
    fn net_exposure_signs_correctly() {
        let mut pf = Portfolio::new(d(1_000.0));
        pf.apply_fill(M, Side::Buy, cents(40), Qty(100), Notional::ZERO);
        assert!(pf.net_exposure() > Notional::ZERO);
        pf.apply_fill(N, Side::Sell, cents(40), Qty(200), Notional::ZERO);
        assert!(pf.net_exposure() < Notional::ZERO, "net short overall");
    }

    #[test]
    fn drawdown_measures_from_the_high_water_mark() {
        let mut pf = Portfolio::new(d(1_000.0));
        pf.apply_fill(M, Side::Buy, cents(40), Qty(1000), Notional::ZERO);

        let up = HashMap::from([(M, cents(60))]);
        pf.mark(&up);
        assert_eq!(pf.peak_equity(), d(1_200.0));
        assert_eq!(pf.drawdown(&up), 0.0);

        let down = HashMap::from([(M, cents(30))]);
        pf.mark(&down);
        // Equity 900 against a 1200 peak.
        assert_eq!(pf.drawdown(&down), 0.25);
        assert_eq!(pf.peak_equity(), d(1_200.0));
    }

    #[test]
    fn an_unmarked_position_is_valued_at_cost_not_at_zero() {
        let mut pf = Portfolio::new(d(1_000.0));
        pf.apply_fill(M, Side::Buy, cents(40), Qty(100), Notional::ZERO);
        // No marks at all: equity must not collapse to cash.
        assert_eq!(pf.equity(&HashMap::new()), d(1_000.0));
    }

    #[test]
    fn a_full_round_trip_conserves_money() {
        // Buy, add, partially close, flip, settle. Cash plus realised must
        // reconcile against the starting bankroll exactly — no epsilon, which
        // is the entire reason this ledger is not held in f64.
        let mut pf = Portfolio::new(d(1_000.0));
        pf.apply_fill(M, Side::Buy, cents(40), Qty(100), Notional::ZERO);
        pf.apply_fill(M, Side::Buy, cents(50), Qty(100), Notional::ZERO);
        pf.apply_fill(M, Side::Sell, cents(60), Qty(50), Notional::ZERO);
        pf.apply_fill(M, Side::Sell, cents(55), Qty(200), Notional::ZERO);
        pf.settle(M, false);

        assert!(pf.position(M).unwrap().is_flat());
        assert_eq!(pf.cash, d(1_000.0) + pf.realized());
        assert_eq!(pf.capital_at_risk(), Notional::ZERO);
    }

    #[test]
    fn a_long_random_walk_of_fills_conserves_money_to_the_micro_dollar() {
        // The property f64 accounting cannot hold. A thousand fills at prices
        // and fees that do not divide evenly, across five markets, and the
        // books still close with no tolerance at all.
        let mut pf = Portfolio::new(d(10_000.0));
        let mut price = 500_000i64;
        let mut seed = 12_345u64;
        for i in 0..1_000 {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let step = ((seed >> 33) % 20_001) as i64 - 10_000;
            price = (price + step).clamp(1_000, 999_000);
            let side = if (seed >> 20) & 1 == 0 { Side::Buy } else { Side::Sell };
            let qty = Qty(((seed >> 10) % 7) as i64 + 1);
            let fee = Notional(((seed >> 5) % 1_000) as i64);
            pf.apply_fill(MarketId(i % 5), side, Price::from_micros(price), qty, fee);
        }

        // Unwind everything by settling each market, then the books must close.
        for m in 0..5 {
            pf.settle(MarketId(m), m % 2 == 0);
        }
        assert_eq!(pf.capital_at_risk(), Notional::ZERO);
        assert_eq!(pf.cash, d(10_000.0) + pf.realized());
    }
}
