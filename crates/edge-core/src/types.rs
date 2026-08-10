//! Core value types.
//!
//! Two deliberate choices here shape everything downstream:
//!
//! 1. **Prices are integers.** A prediction-market contract settles at exactly
//!    $1 or $0, and venues quote it on a fixed tick grid (Kalshi: whole cents;
//!    Polymarket: tenths of a cent). Representing that as `f64` invites
//!    accumulated rounding error in the order book and in PnL, and makes
//!    equality comparison — which the matching engine depends on — unsound. We
//!    use signed micro-dollars, so $1.00 is `1_000_000` and every venue tick is
//!    exactly representable.
//!
//! 2. **Identifiers are interned integers.** The matching hot path compares and
//!    hashes ids constantly. `Copy` integers keep the book allocation-free;
//!    the human-readable venue ticker lives in a registry
//!    ([`crate::market::MarketRegistry`]) consulted only at the edges.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::error::{EdgeError, Result};

/// Micro-dollars per dollar. A contract's full settlement value.
pub const MICROS: i64 = 1_000_000;

// ---------------------------------------------------------------------------
// Time
// ---------------------------------------------------------------------------

/// Nanoseconds since the Unix epoch.
///
/// The engine is fully deterministic given its input event stream, which means
/// time has to be data rather than an ambient call to the system clock — the
/// backtester and the live runtime feed the identical `Ts` values through the
/// identical code.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct Ts(pub i64);

impl Ts {
    pub const ZERO: Ts = Ts(0);

    #[inline]
    pub const fn from_nanos(nanos: i64) -> Self {
        Ts(nanos)
    }

    #[inline]
    pub const fn from_millis(millis: i64) -> Self {
        Ts(millis * 1_000_000)
    }

    #[inline]
    pub const fn from_secs(secs: i64) -> Self {
        Ts(secs * 1_000_000_000)
    }

    #[inline]
    pub const fn as_nanos(self) -> i64 {
        self.0
    }

    #[inline]
    pub const fn as_millis(self) -> i64 {
        self.0 / 1_000_000
    }

    #[inline]
    pub const fn as_secs_f64(self) -> f64 {
        self.0 as f64 / 1e9
    }

    /// Elapsed time to a later timestamp, saturating at zero.
    #[inline]
    pub const fn elapsed_to(self, later: Ts) -> i64 {
        let d = later.0 - self.0;
        if d < 0 { 0 } else { d }
    }
}

impl fmt::Display for Ts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// Probability
// ---------------------------------------------------------------------------

/// A probability in `[0, 1]`, validated at construction.
///
/// Wrapping this rather than passing bare `f64` catches the single most common
/// class of bug in pricing code: a devigged probability vector that silently
/// went negative, or an inverted price that exceeded one.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Prob(f64);

impl Prob {
    pub const ZERO: Prob = Prob(0.0);
    pub const ONE: Prob = Prob(1.0);
    pub const HALF: Prob = Prob(0.5);

    /// Smallest probability we will quote or price against. Below this the
    /// implied decimal odds exceed 100,000:1, which no venue offers and which
    /// makes Kelly and log-odds numerically useless.
    pub const EPS: f64 = 1e-9;

    pub fn new(v: f64) -> Result<Self> {
        if !v.is_finite() || !(0.0..=1.0).contains(&v) {
            return Err(EdgeError::InvalidProbability(v));
        }
        Ok(Prob(v))
    }

    /// Construct without failing, clamping into range. Use where the input is
    /// known to be an approximation (a model output, an interpolation) rather
    /// than an assertion about the market.
    #[inline]
    pub fn clamped(v: f64) -> Self {
        if v.is_nan() {
            return Prob(0.5);
        }
        Prob(v.clamp(0.0, 1.0))
    }

    /// Clamp strictly inside the open interval, for use where a zero or one
    /// would produce an infinity (log-odds, Kelly, log-loss).
    #[inline]
    pub fn clamped_open(v: f64) -> Self {
        if v.is_nan() {
            return Prob(0.5);
        }
        Prob(v.clamp(Self::EPS, 1.0 - Self::EPS))
    }

    #[inline]
    pub const fn get(self) -> f64 {
        self.0
    }

    #[inline]
    pub fn complement(self) -> Self {
        Prob(1.0 - self.0)
    }

    /// Fair decimal odds implied by this probability.
    pub fn to_decimal_odds(self) -> Result<f64> {
        if self.0 <= Self::EPS {
            return Err(EdgeError::InvalidProbability(self.0));
        }
        Ok(1.0 / self.0)
    }

    /// Log-odds. Saturates rather than returning an infinity.
    #[inline]
    pub fn logit(self) -> f64 {
        let p = self.0.clamp(Self::EPS, 1.0 - Self::EPS);
        (p / (1.0 - p)).ln()
    }

    #[inline]
    pub fn from_logit(x: f64) -> Self {
        // Numerically stable both ways: for large negative x, exp(x) underflows
        // to zero instead of exp(-x) overflowing to infinity.
        let p = if x >= 0.0 {
            1.0 / (1.0 + (-x).exp())
        } else {
            let e = x.exp();
            e / (1.0 + e)
        };
        Prob(p)
    }

    /// The price at which this probability is fair value, rounded to nearest micro.
    #[inline]
    pub fn to_price(self) -> Price {
        Price((self.0 * MICROS as f64).round() as i64)
    }
}

impl fmt::Display for Prob {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.4}", self.0)
    }
}

impl Eq for Prob {}

#[allow(clippy::derive_ord_xor_partial_ord)]
impl Ord for Prob {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Safe: the constructors reject NaN, so a total order exists.
        self.0.partial_cmp(&other.0).unwrap_or(std::cmp::Ordering::Equal)
    }
}

// ---------------------------------------------------------------------------
// Price
// ---------------------------------------------------------------------------

/// A contract price in micro-dollars, where `1_000_000` is full settlement value.
///
/// Signed because spreads, edges and PnL deltas are all naturally expressed in
/// the same unit and are frequently negative.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct Price(pub i64);

impl Price {
    pub const ZERO: Price = Price(0);
    pub const ONE: Price = Price(MICROS);
    /// Tightest tradable price on any venue we support (one tenth of a cent).
    pub const MIN_TRADABLE: Price = Price(1_000);
    pub const MAX_TRADABLE: Price = Price(MICROS - 1_000);

    #[inline]
    pub const fn from_micros(m: i64) -> Self {
        Price(m)
    }

    #[inline]
    pub const fn from_cents(c: i64) -> Self {
        Price(c * 10_000)
    }

    pub fn from_dollars(d: f64) -> Result<Self> {
        if !d.is_finite() {
            return Err(EdgeError::InvalidPrice(0));
        }
        Ok(Price((d * MICROS as f64).round() as i64))
    }

    #[inline]
    pub const fn micros(self) -> i64 {
        self.0
    }

    #[inline]
    pub fn dollars(self) -> f64 {
        self.0 as f64 / MICROS as f64
    }

    /// Interpret the price as the market's implied probability of settlement.
    #[inline]
    pub fn implied_prob(self) -> Prob {
        Prob::clamped(self.dollars())
    }

    /// The complementary contract's price. A YES at 40c implies NO at 60c.
    #[inline]
    pub const fn complement(self) -> Price {
        Price(MICROS - self.0)
    }

    /// Is this a price a venue would actually accept an order at?
    #[inline]
    pub const fn is_tradable(self) -> bool {
        self.0 > 0 && self.0 < MICROS
    }

    /// Round to the nearest multiple of `tick`, which is how a venue will
    /// interpret any order we send regardless of what we asked for.
    #[inline]
    pub fn round_to_tick(self, tick: i64) -> Price {
        if tick <= 1 {
            return self;
        }
        let half = tick / 2;
        let r =
            if self.0 >= 0 { (self.0 + half) / tick * tick } else { (self.0 - half) / tick * tick };
        Price(r)
    }

    /// Round in the direction that is conservative for `side`: a buy rounds
    /// down (never pay more than intended), a sell rounds up.
    #[inline]
    pub fn round_to_tick_conservative(self, tick: i64, side: Side) -> Price {
        if tick <= 1 {
            return self;
        }
        let r = match side {
            Side::Buy => self.0.div_euclid(tick) * tick,
            Side::Sell => {
                self.0.div_euclid(tick) * tick + if self.0.rem_euclid(tick) != 0 { tick } else { 0 }
            }
        };
        Price(r)
    }

    /// Cost of `qty` contracts at this price, in micro-dollars.
    #[inline]
    pub const fn notional(self, qty: Qty) -> Notional {
        Notional(self.0 * qty.0)
    }
}

impl std::ops::Add for Price {
    type Output = Price;
    #[inline]
    fn add(self, rhs: Price) -> Price {
        Price(self.0 + rhs.0)
    }
}

impl std::ops::Sub for Price {
    type Output = Price;
    #[inline]
    fn sub(self, rhs: Price) -> Price {
        Price(self.0 - rhs.0)
    }
}

impl std::ops::Neg for Price {
    type Output = Price;
    #[inline]
    fn neg(self) -> Price {
        Price(-self.0)
    }
}

impl fmt::Display for Price {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.4}", self.dollars())
    }
}

// ---------------------------------------------------------------------------
// Quantity and notional
// ---------------------------------------------------------------------------

/// A number of contracts. Signed so a position can be expressed in the same type.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct Qty(pub i64);

impl Qty {
    pub const ZERO: Qty = Qty(0);

    #[inline]
    pub const fn new(n: i64) -> Self {
        Qty(n)
    }

    pub fn positive(n: i64) -> Result<Self> {
        if n <= 0 {
            return Err(EdgeError::InvalidQuantity(n));
        }
        Ok(Qty(n))
    }

    #[inline]
    pub const fn get(self) -> i64 {
        self.0
    }

    #[inline]
    pub const fn is_zero(self) -> bool {
        self.0 == 0
    }

    #[inline]
    pub const fn abs(self) -> Qty {
        Qty(self.0.abs())
    }

    #[inline]
    pub fn min(self, other: Qty) -> Qty {
        Qty(self.0.min(other.0))
    }
}

impl std::ops::Add for Qty {
    type Output = Qty;
    #[inline]
    fn add(self, rhs: Qty) -> Qty {
        Qty(self.0 + rhs.0)
    }
}

impl std::ops::Sub for Qty {
    type Output = Qty;
    #[inline]
    fn sub(self, rhs: Qty) -> Qty {
        Qty(self.0 - rhs.0)
    }
}

impl std::ops::AddAssign for Qty {
    #[inline]
    fn add_assign(&mut self, rhs: Qty) {
        self.0 += rhs.0;
    }
}

impl std::ops::SubAssign for Qty {
    #[inline]
    fn sub_assign(&mut self, rhs: Qty) {
        self.0 -= rhs.0;
    }
}

impl std::ops::Neg for Qty {
    type Output = Qty;
    #[inline]
    fn neg(self) -> Qty {
        Qty(-self.0)
    }
}

impl fmt::Display for Qty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Money, in micro-dollars. `Price * Qty` lands here.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct Notional(pub i64);

impl Notional {
    pub const ZERO: Notional = Notional(0);

    #[inline]
    pub fn from_dollars(d: f64) -> Self {
        Notional((d * MICROS as f64).round() as i64)
    }

    #[inline]
    pub fn dollars(self) -> f64 {
        self.0 as f64 / MICROS as f64
    }

    #[inline]
    pub const fn abs(self) -> Notional {
        Notional(self.0.abs())
    }
}

impl std::ops::Add for Notional {
    type Output = Notional;
    #[inline]
    fn add(self, rhs: Notional) -> Notional {
        Notional(self.0 + rhs.0)
    }
}

impl std::ops::Sub for Notional {
    type Output = Notional;
    #[inline]
    fn sub(self, rhs: Notional) -> Notional {
        Notional(self.0 - rhs.0)
    }
}

impl std::ops::AddAssign for Notional {
    #[inline]
    fn add_assign(&mut self, rhs: Notional) {
        self.0 += rhs.0;
    }
}

impl std::ops::SubAssign for Notional {
    #[inline]
    fn sub_assign(&mut self, rhs: Notional) {
        self.0 -= rhs.0;
    }
}

impl std::ops::Neg for Notional {
    type Output = Notional;
    #[inline]
    fn neg(self) -> Notional {
        Notional(-self.0)
    }
}

impl fmt::Display for Notional {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "${:.2}", self.dollars())
    }
}

// ---------------------------------------------------------------------------
// Sides
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Side {
    Buy,
    Sell,
}

impl Side {
    #[inline]
    pub const fn opposite(self) -> Side {
        match self {
            Side::Buy => Side::Sell,
            Side::Sell => Side::Buy,
        }
    }

    /// `+1` for a buy, `-1` for a sell — the sign this side applies to a position.
    #[inline]
    pub const fn sign(self) -> i64 {
        match self {
            Side::Buy => 1,
            Side::Sell => -1,
        }
    }

    /// Would an order on this side at `limit` trade against a resting `book` price?
    #[inline]
    pub const fn crosses(self, limit: Price, book: Price) -> bool {
        match self {
            Side::Buy => limit.0 >= book.0,
            Side::Sell => limit.0 <= book.0,
        }
    }
}

impl fmt::Display for Side {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Side::Buy => "BUY",
            Side::Sell => "SELL",
        })
    }
}

/// Which leg of a binary market a contract represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Leg {
    Yes,
    No,
}

impl Leg {
    #[inline]
    pub const fn opposite(self) -> Leg {
        match self {
            Leg::Yes => Leg::No,
            Leg::No => Leg::Yes,
        }
    }
}

impl fmt::Display for Leg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Leg::Yes => "YES",
            Leg::No => "NO",
        })
    }
}

// ---------------------------------------------------------------------------
// Identifiers
// ---------------------------------------------------------------------------

macro_rules! id_type {
    ($(#[$m:meta])* $name:ident, $inner:ty) => {
        $(#[$m])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(pub $inner);

        impl $name {
            #[inline]
            pub const fn new(v: $inner) -> Self {
                $name(v)
            }
            #[inline]
            pub const fn get(self) -> $inner {
                self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, concat!(stringify!($name), "({})"), self.0)
            }
        }
    };
}

id_type!(
    /// An interned tradable instrument: one venue's contract on one outcome.
    MarketId, u64
);
id_type!(
    /// A real-world event that several venues may each list a market on.
    /// Cross-venue arbitrage is expressed as two `MarketId`s sharing an `EventId`.
    EventId, u64
);
id_type!(
    /// Engine-assigned, monotonic, unique for the lifetime of a process.
    OrderId, u64
);
id_type!(
    /// Caller-assigned, used to correlate an order across a venue round trip.
    ClientOrderId, u64
);
id_type!(VenueId, u16);
id_type!(StrategyId, u16);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn price_dollar_roundtrip() {
        assert_eq!(Price::from_cents(37).micros(), 370_000);
        assert!((Price::from_cents(37).dollars() - 0.37).abs() < 1e-12);
        assert_eq!(Price::from_dollars(0.375).unwrap(), Price(375_000));
    }

    #[test]
    fn price_complement_is_involutive() {
        let p = Price::from_cents(41);
        assert_eq!(p.complement(), Price::from_cents(59));
        assert_eq!(p.complement().complement(), p);
    }

    #[test]
    fn tick_rounding_is_nearest() {
        let tick = 10_000; // one cent
        assert_eq!(Price(374_000).round_to_tick(tick), Price(370_000));
        assert_eq!(Price(376_000).round_to_tick(tick), Price(380_000));
        assert_eq!(Price(375_000).round_to_tick(tick), Price(380_000));
    }

    #[test]
    fn conservative_rounding_never_worsens_the_order() {
        let tick = 10_000;
        // A buy must not end up bidding higher than we asked.
        assert_eq!(Price(376_000).round_to_tick_conservative(tick, Side::Buy), Price(370_000));
        // A sell must not end up offering lower than we asked.
        assert_eq!(Price(374_000).round_to_tick_conservative(tick, Side::Sell), Price(380_000));
        // Exact ticks are untouched on both sides.
        assert_eq!(Price(370_000).round_to_tick_conservative(tick, Side::Sell), Price(370_000));
        assert_eq!(Price(370_000).round_to_tick_conservative(tick, Side::Buy), Price(370_000));
    }

    #[test]
    fn prob_rejects_out_of_range() {
        assert!(Prob::new(-0.01).is_err());
        assert!(Prob::new(1.01).is_err());
        assert!(Prob::new(f64::NAN).is_err());
        assert!(Prob::new(0.0).is_ok());
        assert!(Prob::new(1.0).is_ok());
    }

    #[test]
    fn logit_roundtrips() {
        for p in [0.01, 0.1, 0.5, 0.9, 0.99] {
            let round = Prob::from_logit(Prob::new(p).unwrap().logit()).get();
            assert!((round - p).abs() < 1e-12, "{p} -> {round}");
        }
    }

    #[test]
    fn logit_saturates_instead_of_producing_infinity() {
        assert!(Prob::ZERO.logit().is_finite());
        assert!(Prob::ONE.logit().is_finite());
        assert_eq!(Prob::from_logit(f64::NEG_INFINITY).get(), 0.0);
        assert_eq!(Prob::from_logit(f64::INFINITY).get(), 1.0);
    }

    #[test]
    fn side_crossing() {
        let bid = Price::from_cents(40);
        let ask = Price::from_cents(45);
        assert!(Side::Buy.crosses(Price::from_cents(45), ask));
        assert!(!Side::Buy.crosses(Price::from_cents(44), ask));
        assert!(Side::Sell.crosses(Price::from_cents(40), bid));
        assert!(!Side::Sell.crosses(Price::from_cents(41), bid));
    }

    #[test]
    fn notional_arithmetic() {
        let n = Price::from_cents(30).notional(Qty(100));
        assert_eq!(n.dollars(), 30.0);
    }
}
