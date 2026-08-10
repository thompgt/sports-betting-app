//! RFC 3339 timestamps, in and out.
//!
//! Every venue publishes close times, settlement times, and trade timestamps as
//! ISO strings, and the engine works in nanoseconds since the Unix epoch. This
//! is the only place that conversion happens.
//!
//! It is hand-rolled rather than pulled from a date library because the whole
//! requirement is one grammar and two calendar functions, and because the
//! failure mode of a lenient parser here is a market whose close time is a year
//! out — which silently poisons every time-to-resolution feature and every
//! near-close guard downstream. So parsing is strict: fixed widths, validated
//! ranges, and no guessing.
//!
//! Naive timestamps (no zone, as in the seed catalogue) are read as UTC. That
//! is an assumption rather than a fact, but it is the documented one, and the
//! alternative — rejecting them — would make the existing fixtures unusable.

use edge_core::types::Ts;

use crate::error::DataError;

const NANOS_PER_SEC: i64 = 1_000_000_000;

/// Days from 1970-01-01 to `y-m-d`, proleptic Gregorian.
///
/// Hinnant's civil-from-days algorithm: shifting the year to start in March
/// puts the leap day last, which is what makes the month-length polynomial
/// `(153*m + 2)/5` exact.
fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = y - if m <= 2 { 1 } else { 0 };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = if m > 2 { m - 3 } else { m + 9 } as i64;
    let doy = (153 * mp + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// Inverse of [`days_from_civil`].
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (y + if m <= 2 { 1 } else { 0 }, m, d)
}

fn days_in_month(y: i64, m: u32) -> u32 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if (y % 4 == 0 && y % 100 != 0) || y % 400 == 0 => 29,
        2 => 28,
        _ => 0,
    }
}

fn bad(input: &str, why: &str) -> DataError {
    DataError::Decode {
        venue: "-".into(),
        what: "timestamp".into(),
        detail: format!("{why}: {input:?}"),
    }
}

/// Parse a fixed-width unsigned integer, rejecting `+`, `-`, and short fields.
fn field(s: &str, width: usize, input: &str, what: &str) -> Result<(u32, usize), DataError> {
    let bytes = s.as_bytes();
    if bytes.len() < width || !bytes[..width].iter().all(|b| b.is_ascii_digit()) {
        return Err(bad(input, &format!("expected {width} digits for {what}")));
    }
    let v = s[..width].parse::<u32>().map_err(|_| bad(input, what))?;
    Ok((v, width))
}

/// Parse an RFC 3339 timestamp into nanoseconds since the Unix epoch.
///
/// Accepts `YYYY-MM-DDTHH:MM:SS[.fraction][Z|±HH:MM]`, with a space permitted
/// in place of the `T` and the offset optional (read as UTC). Leap seconds
/// (`:60`) are accepted and clamped to `:59`, because a venue emitting one is
/// not a reason to drop a market.
pub fn parse_rfc3339(input: &str) -> Result<Ts, DataError> {
    let s = input.trim();
    let (year, _) = {
        let b = s.as_bytes();
        if b.len() < 19 {
            return Err(bad(input, "too short for a date and time"));
        }
        field(s, 4, input, "year")?
    };
    let rest = &s[4..];
    let sep = |r: &str, c: u8, what: &str| -> Result<(), DataError> {
        if r.as_bytes().first() == Some(&c) { Ok(()) } else { Err(bad(input, what)) }
    };

    sep(rest, b'-', "expected '-' after the year")?;
    let (month, _) = field(&rest[1..], 2, input, "month")?;
    let rest = &rest[3..];
    sep(rest, b'-', "expected '-' after the month")?;
    let (day, _) = field(&rest[1..], 2, input, "day")?;
    let rest = &rest[3..];

    match rest.as_bytes().first() {
        Some(b'T') | Some(b't') | Some(b' ') => {}
        _ => return Err(bad(input, "expected 'T' between date and time")),
    }
    let rest = &rest[1..];

    let (hour, _) = field(rest, 2, input, "hour")?;
    sep(&rest[2..], b':', "expected ':' after the hour")?;
    let (minute, _) = field(&rest[3..], 2, input, "minute")?;
    sep(&rest[5..], b':', "expected ':' after the minute")?;
    let (second, _) = field(&rest[6..], 2, input, "second")?;
    let mut rest = &rest[8..];

    // Fractional seconds, to nanosecond resolution. Extra digits are truncated
    // rather than rounded: a timestamp must never move forward past an event
    // that actually preceded it.
    let mut nanos = 0i64;
    if rest.as_bytes().first() == Some(&b'.') {
        let digits: usize = rest[1..].bytes().take_while(u8::is_ascii_digit).count();
        if digits == 0 {
            return Err(bad(input, "'.' with no fractional digits"));
        }
        let taken = digits.min(9);
        let frac: i64 = rest[1..1 + taken].parse().map_err(|_| bad(input, "fraction"))?;
        nanos = frac * 10i64.pow(9 - taken as u32);
        rest = &rest[1 + digits..];
    }

    let offset_secs: i64 = match rest.as_bytes().first() {
        None => 0,
        Some(b'Z') | Some(b'z') if rest.len() == 1 => 0,
        Some(c @ (b'+' | b'-')) => {
            let sign = if *c == b'-' { -1 } else { 1 };
            let (oh, _) = field(&rest[1..], 2, input, "offset hour")?;
            // `+HHMM` and `+HH:MM` are both seen in the wild.
            let mm = if rest.as_bytes().get(3) == Some(&b':') { &rest[4..] } else { &rest[3..] };
            let (om, _) = field(mm, 2, input, "offset minute")?;
            if oh > 23 || om > 59 {
                return Err(bad(input, "offset out of range"));
            }
            sign * (oh as i64 * 3600 + om as i64 * 60)
        }
        _ => return Err(bad(input, "trailing characters after the time")),
    };

    if month == 0 || month > 12 || day == 0 || day > days_in_month(year as i64, month) {
        return Err(bad(input, "date does not exist"));
    }
    if hour > 23 || minute > 59 || second > 60 {
        return Err(bad(input, "time out of range"));
    }
    let second = second.min(59);

    let days = days_from_civil(year as i64, month, day);
    let secs =
        days * 86_400 + hour as i64 * 3600 + minute as i64 * 60 + second as i64 - offset_secs;

    secs.checked_mul(NANOS_PER_SEC)
        .and_then(|n| n.checked_add(nanos))
        .map(Ts)
        .ok_or_else(|| bad(input, "timestamp out of representable range"))
}

/// Render as `YYYY-MM-DDTHH:MM:SS.mmmZ` — always UTC, always milliseconds, so
/// log lines and journal entries sort lexicographically.
pub fn format_rfc3339(ts: Ts) -> String {
    let (mut secs, mut nanos) = (ts.0.div_euclid(NANOS_PER_SEC), ts.0.rem_euclid(NANOS_PER_SEC));
    if nanos < 0 {
        nanos += NANOS_PER_SEC;
        secs -= 1;
    }
    let days = secs.div_euclid(86_400);
    let tod = secs.rem_euclid(86_400);
    let (y, m, d) = civil_from_days(days);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}.{:03}Z",
        tod / 3600,
        (tod % 3600) / 60,
        tod % 60,
        nanos / 1_000_000
    )
}

/// Seconds since the Unix epoch as a `Ts`. Venues that publish integer epochs
/// (Polymarket, mostly) go through here.
pub fn from_unix_secs(secs: i64) -> Ts {
    Ts(secs.saturating_mul(NANOS_PER_SEC))
}

/// Milliseconds since the Unix epoch as a `Ts`.
pub fn from_unix_millis(ms: i64) -> Ts {
    Ts(ms.saturating_mul(1_000_000))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(s: &str) -> Ts {
        parse_rfc3339(s).unwrap_or_else(|e| panic!("{s:?}: {e}"))
    }

    #[test]
    fn the_epoch_is_zero() {
        assert_eq!(p("1970-01-01T00:00:00Z"), Ts::ZERO);
    }

    #[test]
    fn a_known_instant_matches_its_epoch_second() {
        // Locked-in values cross-checked by hand from an independent epoch
        // count; the point is to catch an off-by-one in the calendar code.
        assert_eq!(p("2026-05-27T20:00:00Z").0 / NANOS_PER_SEC, 1_779_912_000);
        assert_eq!(p("2000-03-01T00:00:00Z").0 / NANOS_PER_SEC, 951_868_800);
        assert_eq!(p("2024-02-29T12:00:00Z").0 / NANOS_PER_SEC, 1_709_208_000);
    }

    #[test]
    fn dates_before_the_epoch_go_negative_correctly() {
        assert_eq!(p("1969-12-31T23:59:59Z").0, -NANOS_PER_SEC);
        assert_eq!(p("1900-01-01T00:00:00Z").0 / NANOS_PER_SEC, -2_208_988_800);
    }

    #[test]
    fn an_offset_is_subtracted_not_added() {
        // 15:00 in New York is 20:00 UTC, which is *later* in epoch terms.
        assert_eq!(p("2026-05-27T15:00:00-05:00"), p("2026-05-27T20:00:00Z"));
        assert_eq!(p("2026-05-28T05:00:00+09:00"), p("2026-05-27T20:00:00Z"));
    }

    #[test]
    fn the_compact_offset_form_is_accepted() {
        assert_eq!(p("2026-05-27T15:00:00-0500"), p("2026-05-27T20:00:00Z"));
    }

    #[test]
    fn a_naive_timestamp_is_read_as_utc() {
        // The seed catalogue carries these.
        assert_eq!(p("2026-05-27T20:00:00"), p("2026-05-27T20:00:00Z"));
        assert_eq!(p("2026-05-27 20:00:00"), p("2026-05-27T20:00:00Z"));
    }

    #[test]
    fn fractional_seconds_survive_to_nanoseconds() {
        assert_eq!(p("1970-01-01T00:00:00.5Z").0, 500_000_000);
        assert_eq!(p("1970-01-01T00:00:00.000000001Z").0, 1);
        assert_eq!(p("1970-01-01T00:00:00.123456789Z").0, 123_456_789);
    }

    #[test]
    fn excess_precision_truncates_rather_than_rounds() {
        // Rounding up would place an event before a timestamp that precedes it.
        assert_eq!(p("1970-01-01T00:00:00.9999999999Z").0, 999_999_999);
    }

    #[test]
    fn a_leap_second_is_clamped_not_rejected() {
        assert_eq!(p("2016-12-31T23:59:60Z"), p("2016-12-31T23:59:59Z"));
    }

    #[test]
    fn impossible_dates_are_refused() {
        for s in [
            "2026-02-30T00:00:00Z",
            "2025-02-29T00:00:00Z", // 2025 is not a leap year
            "2026-13-01T00:00:00Z",
            "2026-00-01T00:00:00Z",
            "2026-01-00T00:00:00Z",
            "2026-01-01T24:00:00Z",
            "2026-01-01T00:60:00Z",
        ] {
            assert!(parse_rfc3339(s).is_err(), "{s} should not parse");
        }
    }

    #[test]
    fn malformed_input_is_refused_rather_than_guessed_at() {
        for s in [
            "",
            "2026-05-27",
            "26-05-27T20:00:00Z",
            "2026/05/27T20:00:00Z",
            "2026-05-27X20:00:00Z",
            "2026-05-27T20:00:00Zjunk",
            "2026-05-27T20:00:00.Z",
            "2026-05-27T20:00:00+99:00",
            "+026-05-27T20:00:00Z",
            "not a timestamp at all",
        ] {
            assert!(parse_rfc3339(s).is_err(), "{s:?} should not parse");
        }
    }

    #[test]
    fn formatting_round_trips_through_parsing() {
        for s in [
            "1970-01-01T00:00:00.000Z",
            "2026-05-27T20:00:00.000Z",
            "1999-12-31T23:59:59.999Z",
            "2024-02-29T00:00:00.000Z",
            "1900-01-01T00:00:00.000Z",
        ] {
            assert_eq!(format_rfc3339(p(s)), s);
        }
    }

    #[test]
    fn formatting_a_pre_epoch_instant_does_not_run_the_clock_backwards() {
        // Truncating division would render -0.5s as 1969-12-31T23:59:59.-500.
        assert_eq!(format_rfc3339(Ts(-500_000_000)), "1969-12-31T23:59:59.500Z");
    }

    #[test]
    fn the_calendar_is_its_own_inverse_across_four_centuries() {
        for day in (-40_000..40_000).step_by(7) {
            let (y, m, d) = civil_from_days(day);
            assert_eq!(days_from_civil(y, m, d), day, "{y}-{m}-{d}");
        }
    }

    #[test]
    fn epoch_helpers_agree_with_the_parser() {
        assert_eq!(from_unix_secs(1_779_912_000), p("2026-05-27T20:00:00Z"));
        assert_eq!(from_unix_millis(1_779_912_000_500), p("2026-05-27T20:00:00.5Z"));
    }
}
