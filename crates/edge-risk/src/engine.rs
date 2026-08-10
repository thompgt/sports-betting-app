//! The risk engine: the only thing standing between a strategy and the account.
//!
//! Every order passes through [`RiskEngine::check`] before it reaches a venue.
//! Two behaviours here differ from a conventional pre-trade gate and both are
//! deliberate:
//!
//! **Reducing risk is always allowed.** An order that shrinks an existing
//! position is approved even when a limit is already breached. A gate that
//! blocks closing trades because a limit is exceeded traps the account in
//! exactly the position it was supposed to prevent — the failure mode is not
//! theoretical, it is the standard way an automated system turns a bad day into
//! a catastrophic one.
//!
//! **Size limits resize; permission limits reject.** A strategy asking for
//! 1,000 contracts when 300 fit should trade 300. Rejecting outright discards
//! real edge and trains strategies to ask for less than they want, which
//! degrades the whole system's information. Only limits about *permission* —
//! the kill switch, a stale mark, the rate limiter — refuse outright.

use std::collections::HashMap;

use edge_core::types::{MarketId, Price, Qty, Side, Ts};

use crate::limits::{KillReason, RiskBreach, RiskDecision, RiskLimits};
use crate::position::Portfolio;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RiskStats {
    pub checks: u64,
    pub approved: u64,
    pub resized: u64,
    pub rejected: u64,
    pub kills: u64,
}

#[derive(Debug)]
pub struct RiskEngine {
    limits: RiskLimits,
    portfolio: Portfolio,
    /// Latest mark per market, with the time it arrived. The timestamp is not
    /// optional: an unmarked or stale market is untradable, because a frozen
    /// price is the most common cause of a large accidental position.
    marks: HashMap<MarketId, (Price, Ts)>,
    kill: Option<KillReason>,
    /// Equity at the start of the session. The anchor for the daily loss limit.
    session_anchor: f64,
    tokens: f64,
    last_refill: Ts,
    consecutive_rejects: u32,
    stats: RiskStats,
}

impl RiskEngine {
    pub fn new(limits: RiskLimits, starting_cash: f64) -> Self {
        RiskEngine {
            portfolio: Portfolio::new(starting_cash),
            tokens: limits.order_burst,
            limits,
            marks: HashMap::new(),
            kill: None,
            session_anchor: starting_cash,
            last_refill: Ts::ZERO,
            consecutive_rejects: 0,
            stats: RiskStats::default(),
        }
    }

    pub fn portfolio(&self) -> &Portfolio {
        &self.portfolio
    }

    pub fn portfolio_mut(&mut self) -> &mut Portfolio {
        &mut self.portfolio
    }

    pub fn limits(&self) -> &RiskLimits {
        &self.limits
    }

    pub fn stats(&self) -> RiskStats {
        self.stats
    }

    pub fn kill_reason(&self) -> Option<KillReason> {
        self.kill
    }

    pub fn is_halted(&self) -> bool {
        self.kill.is_some()
    }

    /// Halt trading. Clearing it is a deliberate operator action — nothing
    /// resets the kill switch automatically, because whatever tripped it has
    /// not been investigated yet.
    pub fn trip(&mut self, reason: KillReason) {
        if self.kill.is_none() {
            self.kill = Some(reason);
            self.stats.kills += 1;
        }
    }

    pub fn reset_kill_switch(&mut self) {
        self.kill = None;
        self.consecutive_rejects = 0;
    }

    pub fn set_mark(&mut self, market: MarketId, price: Price, now: Ts) {
        self.marks.insert(market, (price, now));
    }

    pub fn mark(&self, market: MarketId) -> Option<Price> {
        self.marks.get(&market).map(|(p, _)| *p)
    }

    /// Marks as a plain map, for the portfolio and VaR calculations.
    pub fn marks(&self) -> HashMap<MarketId, Price> {
        self.marks.iter().map(|(m, (p, _))| (*m, *p)).collect()
    }

    /// Anchor the daily loss limit to current equity. Call at session rollover.
    pub fn roll_session(&mut self) {
        self.session_anchor = self.portfolio.equity(&self.marks());
    }

    pub fn session_pnl(&self) -> f64 {
        self.portfolio.equity(&self.marks()) - self.session_anchor
    }

    /// Re-evaluate the halt conditions. Call on every mark cycle.
    pub fn update(&mut self, now: Ts) {
        let marks = self.marks();
        self.portfolio.mark(&marks);

        if self.session_pnl() < -self.limits.max_daily_loss {
            self.trip(KillReason::DailyLoss);
        }
        if self.portfolio.drawdown(&marks) > self.limits.max_drawdown {
            self.trip(KillReason::Drawdown);
        }
        // Any open position whose mark has gone stale halts the whole engine
        // rather than just that market: a feed that has stopped for one market
        // has usually stopped for all of them.
        let max_age = (self.limits.max_mark_age_secs * 1e9) as i64;
        for p in self.portfolio.open_positions() {
            match self.marks.get(&p.market) {
                Some((_, seen)) if now.as_nanos() - seen.as_nanos() <= max_age => {}
                _ => {
                    self.kill.get_or_insert(KillReason::StaleData);
                    self.stats.kills += 1;
                    break;
                }
            }
        }
    }

    /// Record a venue rejection. Enough in a row means the engine's view of the
    /// world disagrees with the venue's, and continuing is guessing.
    pub fn on_venue_reject(&mut self) {
        self.consecutive_rejects += 1;
        if self.consecutive_rejects >= self.limits.max_consecutive_rejects {
            self.trip(KillReason::VenueRejections);
        }
    }

    pub fn on_venue_accept(&mut self) {
        self.consecutive_rejects = 0;
    }

    /// Book a fill.
    pub fn on_fill(
        &mut self,
        market: MarketId,
        side: Side,
        price: Price,
        qty: Qty,
        fee: f64,
        now: Ts,
    ) {
        self.portfolio.apply_fill(market, side, price, qty, fee);
        self.set_mark(market, price, now);
        self.on_venue_accept();
    }

    pub fn on_settle(&mut self, market: MarketId, outcome: bool) -> f64 {
        self.marks.remove(&market);
        self.portfolio.settle(market, outcome)
    }

    fn refill_tokens(&mut self, now: Ts) {
        if self.last_refill == Ts::ZERO {
            self.last_refill = now;
            return;
        }
        let elapsed = (now.as_nanos() - self.last_refill.as_nanos()).max(0) as f64 / 1e9;
        self.tokens = (self.tokens + elapsed * self.limits.max_orders_per_second)
            .min(self.limits.order_burst);
        self.last_refill = now;
    }

    /// Evaluate a proposed order.
    ///
    /// `price` is the YES price the order would trade at; `fee_per_contract` is
    /// the venue fee, which counts against the limits because it is money that
    /// leaves the account exactly like the premium does.
    pub fn check(
        &mut self,
        market: MarketId,
        side: Side,
        price: Price,
        qty: Qty,
        fee_per_contract: f64,
        now: Ts,
    ) -> RiskDecision {
        self.stats.checks += 1;
        let decision = self.evaluate(market, side, price, qty, fee_per_contract, now);
        match decision {
            RiskDecision::Approve(_) => self.stats.approved += 1,
            RiskDecision::Resize(..) => self.stats.resized += 1,
            RiskDecision::Reject(_) => self.stats.rejected += 1,
        }
        if decision.is_allowed() {
            self.tokens -= 1.0;
        }
        decision
    }

    fn evaluate(
        &mut self,
        market: MarketId,
        side: Side,
        price: Price,
        qty: Qty,
        fee_per_contract: f64,
        now: Ts,
    ) -> RiskDecision {
        let want = qty.get().abs();
        if want == 0 {
            return RiskDecision::Reject(RiskBreach::OrderSize);
        }
        if !price.is_tradable() {
            return RiskDecision::Reject(RiskBreach::InvalidPrice);
        }

        // How much of this order merely unwinds what is already held. Always
        // permitted — even under a kill switch, even over every limit.
        let held = self.portfolio.qty(market).get();
        let closes =
            if held != 0 && (held > 0) != (side == Side::Buy) { want.min(held.abs()) } else { 0 };

        if self.kill.is_some() {
            return if closes > 0 {
                if closes == want {
                    RiskDecision::Approve(Qty(closes))
                } else {
                    RiskDecision::Resize(Qty(closes), RiskBreach::KillSwitchActive)
                }
            } else {
                RiskDecision::Reject(RiskBreach::KillSwitchActive)
            };
        }

        self.refill_tokens(now);
        if self.tokens < 1.0 {
            return RiskDecision::Reject(RiskBreach::RateLimit);
        }

        // A market with no fresh mark cannot be risk-assessed. Closing is still
        // fine — getting out never needs a price opinion.
        let max_age = (self.limits.max_mark_age_secs * 1e9) as i64;
        let fresh = matches!(
            self.marks.get(&market),
            Some((_, seen)) if now.as_nanos() - seen.as_nanos() <= max_age
        );
        if !fresh && closes < want {
            return if closes > 0 {
                RiskDecision::Resize(Qty(closes), RiskBreach::NoMark)
            } else {
                RiskDecision::Reject(RiskBreach::NoMark)
            };
        }

        let opening = want - closes;
        if opening == 0 {
            return RiskDecision::Approve(Qty(want));
        }

        // Cost of one contract of the leg this order would acquire.
        let leg_price =
            if side == Side::Buy { price.dollars() } else { price.complement().dollars() };
        let unit_cost = leg_price + fee_per_contract;
        if unit_cost <= 0.0 {
            return RiskDecision::Reject(RiskBreach::InvalidPrice);
        }

        let mut allowed = opening;
        let mut binding = RiskBreach::OrderSize;
        let bind = |cap: i64, why: RiskBreach, allowed: &mut i64, binding: &mut RiskBreach| {
            let cap = cap.max(0);
            if cap < *allowed {
                *allowed = cap;
                *binding = why;
            }
        };

        // Cash, keeping the reserve intact.
        let spendable = (self.portfolio.cash - self.limits.min_cash_reserve).max(0.0);
        bind(
            (spendable / unit_cost).floor() as i64,
            RiskBreach::InsufficientCash,
            &mut allowed,
            &mut binding,
        );

        // Per-order notional.
        bind(
            (self.limits.max_order_cost / unit_cost).floor() as i64,
            RiskBreach::OrderSize,
            &mut allowed,
            &mut binding,
        );

        // Contract count in this market, after the closing portion.
        let post_close = (held.abs() - closes).max(0);
        bind(
            self.limits.max_position_contracts - post_close,
            RiskBreach::PositionSize,
            &mut allowed,
            &mut binding,
        );

        // Capital at risk in this market.
        let pos_cost = self.portfolio.position(market).map(|p| p.capital_at_risk()).unwrap_or(0.0);
        let freed =
            self.portfolio.position(market).map(|p| closes as f64 * p.avg_cost).unwrap_or(0.0);
        bind(
            ((self.limits.max_position_cost - (pos_cost - freed)) / unit_cost).floor() as i64,
            RiskBreach::PositionCost,
            &mut allowed,
            &mut binding,
        );

        // Capital at risk on the whole event.
        if let Some(event) = self.portfolio.event_of(market) {
            let at_risk = self.portfolio.event_at_risk(event) - freed;
            bind(
                ((self.limits.max_event_cost - at_risk) / unit_cost).floor() as i64,
                RiskBreach::EventConcentration,
                &mut allowed,
                &mut binding,
            );
        }

        // Capital at risk across the portfolio.
        let total_at_risk = self.portfolio.capital_at_risk() - freed;
        bind(
            ((self.limits.max_portfolio_cost - total_at_risk) / unit_cost).floor() as i64,
            RiskBreach::PortfolioCost,
            &mut allowed,
            &mut binding,
        );

        // Opening a position in a market not already held.
        if post_close == 0
            && self.portfolio.qty(market).get() == 0
            && self.portfolio.open_count() >= self.limits.max_open_markets
        {
            return if closes > 0 {
                RiskDecision::Resize(Qty(closes), RiskBreach::TooManyMarkets)
            } else {
                RiskDecision::Reject(RiskBreach::TooManyMarkets)
            };
        }

        let total = closes + allowed.max(0);
        if total == want {
            RiskDecision::Approve(Qty(total))
        } else if total > 0 {
            RiskDecision::Resize(Qty(total), binding)
        } else {
            RiskDecision::Reject(binding)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use edge_core::types::EventId;

    const M: MarketId = MarketId(1);
    const N: MarketId = MarketId(2);

    fn engine() -> RiskEngine {
        let limits = RiskLimits {
            max_position_contracts: 500,
            max_position_cost: 100.0,
            max_event_cost: 150.0,
            max_portfolio_cost: 400.0,
            max_order_cost: 100.0,
            min_cash_reserve: 50.0,
            max_daily_loss: 200.0,
            max_drawdown: 0.25,
            max_orders_per_second: 10.0,
            order_burst: 20.0,
            max_open_markets: 3,
            max_mark_age_secs: 30.0,
            max_consecutive_rejects: 3,
        };
        limits.validate().unwrap();
        let mut e = RiskEngine::new(limits, 1_000.0);
        e.set_mark(M, Price::from_cents(50), Ts::from_secs(1));
        e.set_mark(N, Price::from_cents(50), Ts::from_secs(1));
        e
    }

    fn now() -> Ts {
        Ts::from_secs(2)
    }

    #[test]
    fn an_order_inside_every_limit_is_approved_whole() {
        let mut e = engine();
        // 50 contracts at 50c = $25, inside all caps.
        let d = e.check(M, Side::Buy, Price::from_cents(50), Qty(50), 0.0, now());
        assert_eq!(d, RiskDecision::Approve(Qty(50)));
    }

    #[test]
    fn an_oversized_order_is_cut_down_not_refused() {
        let mut e = engine();
        // 1,000 at 50c would be $500; the per-order cap is $100, so 200 fit.
        let d = e.check(M, Side::Buy, Price::from_cents(50), Qty(1_000), 0.0, now());
        assert_eq!(d, RiskDecision::Resize(Qty(200), RiskBreach::OrderSize));
        assert!(d.is_allowed());
    }

    #[test]
    fn fees_count_against_the_limits() {
        let mut e = engine();
        // At 50c plus a 1.75c fee, $100 buys 193 contracts rather than 200.
        let d = e.check(M, Side::Buy, Price::from_cents(50), Qty(1_000), 0.0175, now());
        assert_eq!(d.qty(), Qty(193));
    }

    #[test]
    fn the_per_market_cost_limit_binds() {
        let mut e = engine();
        e.on_fill(M, Side::Buy, Price::from_cents(50), Qty(150), 0.0, now());
        // $75 already at risk against a $100 cap: 50 more contracts fit.
        let d = e.check(M, Side::Buy, Price::from_cents(50), Qty(200), 0.0, now());
        assert_eq!(d, RiskDecision::Resize(Qty(50), RiskBreach::PositionCost));
    }

    #[test]
    fn the_event_limit_binds_across_markets() {
        let mut e = engine();
        e.portfolio_mut().set_event(M, EventId(1));
        e.portfolio_mut().set_event(N, EventId(1));
        // $100 at risk on market M, all of it on event 1.
        e.on_fill(M, Side::Buy, Price::from_cents(50), Qty(200), 0.0, now());
        // The per-market cap on N is untouched, but the event cap leaves $50.
        let d = e.check(N, Side::Buy, Price::from_cents(50), Qty(200), 0.0, now());
        assert_eq!(d, RiskDecision::Resize(Qty(100), RiskBreach::EventConcentration));
    }

    #[test]
    fn the_portfolio_limit_binds_across_events() {
        let e = engine();
        let mut limits = *e.limits();
        limits.max_portfolio_cost = 120.0;
        let mut e = RiskEngine::new(limits, 1_000.0);
        e.set_mark(M, Price::from_cents(50), Ts::from_secs(1));
        e.set_mark(N, Price::from_cents(50), Ts::from_secs(1));
        e.on_fill(M, Side::Buy, Price::from_cents(50), Qty(200), 0.0, now());
        let d = e.check(N, Side::Buy, Price::from_cents(50), Qty(200), 0.0, now());
        assert_eq!(d, RiskDecision::Resize(Qty(40), RiskBreach::PortfolioCost));
    }

    #[test]
    fn the_cash_reserve_is_never_spent() {
        // Every other limit is raised out of the way so the reserve is the only
        // thing that can bind.
        let limits = RiskLimits {
            min_cash_reserve: 900.0,
            max_order_cost: 10_000.0,
            max_position_cost: 10_000.0,
            max_event_cost: 10_000.0,
            max_portfolio_cost: 10_000.0,
            ..Default::default()
        };
        let mut e = RiskEngine::new(limits, 1_000.0);
        e.set_mark(M, Price::from_cents(50), Ts::from_secs(1));
        // Only $100 is spendable, so 200 contracts at 50c.
        let d = e.check(M, Side::Buy, Price::from_cents(50), Qty(1_000), 0.0, now());
        assert_eq!(d, RiskDecision::Resize(Qty(200), RiskBreach::InsufficientCash));
    }

    #[test]
    fn a_short_is_costed_on_the_no_leg() {
        let mut e = engine();
        // Selling YES at 20c buys NO at 80c, so the $100 order cap allows 125.
        let d = e.check(M, Side::Sell, Price::from_cents(20), Qty(1_000), 0.0, now());
        assert_eq!(d.qty(), Qty(125));
    }

    #[test]
    fn closing_is_allowed_even_when_a_limit_is_breached() {
        let mut e = engine();
        e.on_fill(M, Side::Buy, Price::from_cents(50), Qty(200), 0.0, now());
        // Fully at the per-market cap. Getting out must still work.
        let d = e.check(M, Side::Sell, Price::from_cents(50), Qty(200), 0.0, now());
        assert_eq!(d, RiskDecision::Approve(Qty(200)));
    }

    #[test]
    fn closing_is_allowed_even_under_the_kill_switch() {
        // The failure this guards against: an automated system trapped in the
        // position its own risk gate was supposed to prevent.
        let mut e = engine();
        e.on_fill(M, Side::Buy, Price::from_cents(50), Qty(100), 0.0, now());
        e.trip(KillReason::Manual);
        let d = e.check(M, Side::Sell, Price::from_cents(50), Qty(100), 0.0, now());
        assert_eq!(d, RiskDecision::Approve(Qty(100)));

        // Opening is refused.
        let d = e.check(N, Side::Buy, Price::from_cents(50), Qty(10), 0.0, now());
        assert_eq!(d, RiskDecision::Reject(RiskBreach::KillSwitchActive));
    }

    #[test]
    fn a_flip_closes_freely_and_opens_under_the_limits() {
        let mut e = engine();
        e.trip(KillReason::Manual);
        e.portfolio_mut().apply_fill(M, Side::Buy, Price::from_cents(50), Qty(100), 0.0);
        // Sell 150: 100 closes, 50 would open a short — refused while halted.
        let d = e.check(M, Side::Sell, Price::from_cents(50), Qty(150), 0.0, now());
        assert_eq!(d, RiskDecision::Resize(Qty(100), RiskBreach::KillSwitchActive));
    }

    #[test]
    fn an_unmarked_market_cannot_be_opened_but_can_be_closed() {
        let mut e = engine();
        let unknown = MarketId(99);
        let d = e.check(unknown, Side::Buy, Price::from_cents(50), Qty(10), 0.0, now());
        assert_eq!(d, RiskDecision::Reject(RiskBreach::NoMark));

        e.portfolio_mut().apply_fill(unknown, Side::Buy, Price::from_cents(50), Qty(10), 0.0);
        let d = e.check(unknown, Side::Sell, Price::from_cents(50), Qty(10), 0.0, now());
        assert_eq!(d, RiskDecision::Approve(Qty(10)));
    }

    #[test]
    fn a_stale_mark_is_treated_as_no_mark() {
        let mut e = engine();
        e.set_mark(M, Price::from_cents(50), Ts::from_secs(1));
        // Thirty-one seconds later, past the limit.
        let d = e.check(M, Side::Buy, Price::from_cents(50), Qty(10), 0.0, Ts::from_secs(32));
        assert_eq!(d, RiskDecision::Reject(RiskBreach::NoMark));
    }

    #[test]
    fn the_rate_limiter_throttles_then_refills() {
        let mut e = engine();
        // Burst is 20.
        for i in 0..20 {
            let d = e.check(M, Side::Buy, Price::from_cents(50), Qty(1), 0.0, now());
            assert!(d.is_allowed(), "order {i} should be inside the burst");
        }
        let d = e.check(M, Side::Buy, Price::from_cents(50), Qty(1), 0.0, now());
        assert_eq!(d, RiskDecision::Reject(RiskBreach::RateLimit));

        // A second later, ten more tokens.
        let later = Ts::from_secs(3);
        assert!(e.check(M, Side::Buy, Price::from_cents(50), Qty(1), 0.0, later).is_allowed());
    }

    #[test]
    fn rejected_orders_do_not_consume_rate_tokens() {
        let mut e = engine();
        for _ in 0..50 {
            e.check(MarketId(77), Side::Buy, Price::from_cents(50), Qty(1), 0.0, now());
        }
        // All rejected for NoMark; the burst should be untouched.
        let d = e.check(M, Side::Buy, Price::from_cents(50), Qty(1), 0.0, now());
        assert!(d.is_allowed());
    }

    #[test]
    fn too_many_open_markets_stops_new_ones() {
        let mut e = engine();
        for i in 0..3 {
            let m = MarketId(100 + i);
            e.set_mark(m, Price::from_cents(50), Ts::from_secs(1));
            e.on_fill(m, Side::Buy, Price::from_cents(50), Qty(10), 0.0, now());
        }
        let m = MarketId(200);
        e.set_mark(m, Price::from_cents(50), Ts::from_secs(1));
        let d = e.check(m, Side::Buy, Price::from_cents(50), Qty(10), 0.0, now());
        assert_eq!(d, RiskDecision::Reject(RiskBreach::TooManyMarkets));

        // Adding to a market already held is still fine.
        let d = e.check(MarketId(100), Side::Buy, Price::from_cents(50), Qty(10), 0.0, now());
        assert!(d.is_allowed());
    }

    #[test]
    fn the_daily_loss_limit_halts_trading() {
        let mut e = engine();
        e.on_fill(M, Side::Buy, Price::from_cents(90), Qty(300), 0.0, now());
        // The position collapses.
        e.set_mark(M, Price::from_cents(10), now());
        e.update(now());
        assert_eq!(e.kill_reason(), Some(KillReason::DailyLoss));
        assert!(e.session_pnl() < -200.0);
    }

    #[test]
    fn the_drawdown_limit_halts_trading() {
        let limits = RiskLimits {
            max_daily_loss: 1e9, // out of the way, so only drawdown can trip
            max_drawdown: 0.10,
            ..Default::default()
        };
        let mut e = RiskEngine::new(limits, 1_000.0);
        e.set_mark(M, Price::from_cents(50), Ts::from_secs(1));
        e.on_fill(M, Side::Buy, Price::from_cents(50), Qty(1_000), 0.0, now());

        e.set_mark(M, Price::from_cents(70), now());
        e.update(now()); // peak equity 1,200
        assert!(!e.is_halted());

        e.set_mark(M, Price::from_cents(45), now());
        e.update(now()); // equity 950, a 21% drawdown
        assert_eq!(e.kill_reason(), Some(KillReason::Drawdown));
    }

    #[test]
    fn a_frozen_feed_halts_trading() {
        let mut e = engine();
        e.on_fill(M, Side::Buy, Price::from_cents(50), Qty(10), 0.0, Ts::from_secs(2));
        e.update(Ts::from_secs(3));
        assert!(!e.is_halted());
        e.update(Ts::from_secs(120));
        assert_eq!(e.kill_reason(), Some(KillReason::StaleData));
    }

    #[test]
    fn repeated_venue_rejections_halt_trading() {
        let mut e = engine();
        e.on_venue_reject();
        e.on_venue_reject();
        assert!(!e.is_halted());
        e.on_venue_accept(); // a success resets the run
        e.on_venue_reject();
        e.on_venue_reject();
        assert!(!e.is_halted());
        e.on_venue_reject();
        assert_eq!(e.kill_reason(), Some(KillReason::VenueRejections));
    }

    #[test]
    fn the_kill_switch_only_clears_by_hand() {
        let mut e = engine();
        e.trip(KillReason::DailyLoss);
        e.update(now());
        assert!(e.is_halted(), "nothing clears the switch on its own");
        e.reset_kill_switch();
        assert!(!e.is_halted());
        let d = e.check(M, Side::Buy, Price::from_cents(50), Qty(10), 0.0, now());
        assert!(d.is_allowed());
    }

    #[test]
    fn tripping_twice_counts_once() {
        let mut e = engine();
        e.trip(KillReason::DailyLoss);
        e.trip(KillReason::Drawdown);
        assert_eq!(e.kill_reason(), Some(KillReason::DailyLoss), "the first cause is kept");
        assert_eq!(e.stats().kills, 1);
    }

    #[test]
    fn rolling_the_session_re_anchors_the_daily_limit() {
        let mut e = engine();
        e.on_fill(M, Side::Buy, Price::from_cents(90), Qty(300), 0.0, now());
        e.set_mark(M, Price::from_cents(10), now());
        assert!(e.session_pnl() < -200.0);
        e.roll_session();
        assert!(e.session_pnl().abs() < 1e-9);
        e.update(now());
        assert!(!e.is_halted());
    }

    #[test]
    fn settlement_releases_capital_and_the_mark() {
        let mut e = engine();
        e.on_fill(M, Side::Buy, Price::from_cents(40), Qty(100), 0.0, now());
        let pnl = e.on_settle(M, true);
        assert!((pnl - 60.0).abs() < 1e-9);
        assert!(e.mark(M).is_none());
        assert!(e.portfolio().capital_at_risk() < 1e-9);
    }

    #[test]
    fn statistics_count_each_outcome_once() {
        let mut e = engine();
        e.check(M, Side::Buy, Price::from_cents(50), Qty(10), 0.0, now());
        e.check(M, Side::Buy, Price::from_cents(50), Qty(10_000), 0.0, now());
        e.check(MarketId(77), Side::Buy, Price::from_cents(50), Qty(10), 0.0, now());
        let s = e.stats();
        assert_eq!((s.checks, s.approved, s.resized, s.rejected), (3, 1, 1, 1));
    }

    #[test]
    fn zero_and_untradable_orders_are_refused() {
        let mut e = engine();
        assert_eq!(
            e.check(M, Side::Buy, Price::from_cents(50), Qty(0), 0.0, now()),
            RiskDecision::Reject(RiskBreach::OrderSize)
        );
        assert_eq!(
            e.check(M, Side::Buy, Price::ONE, Qty(10), 0.0, now()),
            RiskDecision::Reject(RiskBreach::InvalidPrice)
        );
        assert_eq!(
            e.check(M, Side::Buy, Price::ZERO, Qty(10), 0.0, now()),
            RiskDecision::Reject(RiskBreach::InvalidPrice)
        );
    }

    #[test]
    fn no_sequence_of_approved_orders_can_breach_the_portfolio_cap() {
        // The property that matters: limits must hold under repetition, not
        // just for one order in isolation.
        let mut e = engine();
        for i in 0..500 {
            let t = Ts::from_secs(10 + i);
            let m = MarketId(1 + (i % 3) as u64);
            e.set_mark(m, Price::from_cents(50), t);
            let d = e.check(m, Side::Buy, Price::from_cents(50), Qty(50), 0.0, t);
            if d.is_allowed() {
                e.on_fill(m, Side::Buy, Price::from_cents(50), d.qty(), 0.0, t);
            }
            assert!(
                e.portfolio().capital_at_risk() <= e.limits().max_portfolio_cost + 1e-6,
                "portfolio cap breached at step {i}: {}",
                e.portfolio().capital_at_risk()
            );
            assert!(e.portfolio().cash >= e.limits().min_cash_reserve - 1e-6);
        }
    }
}
