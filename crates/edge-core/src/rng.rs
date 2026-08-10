//! A small, seedable, reproducible random number generator.
//!
//! Deliberately not a dependency. Monte Carlo VaR, the execution simulator and
//! the backtester all need randomness, and all three need it to be *exactly*
//! reproducible: a risk number that differs between two runs on the same
//! portfolio cannot be reasoned about, and a backtest you cannot re-run
//! bit-for-bit is an anecdote rather than evidence. Pinning the algorithm here
//! makes reproducibility a property of this repository rather than of whichever
//! version of a crate happened to be resolved.
//!
//! xoshiro256++ — 256-bit state, 2^256 period, passes BigCrush. Not
//! cryptographic, and nothing here should be used for anything that needs to be.

use serde::{Deserialize, Serialize};

/// xoshiro256++ seeded by SplitMix64.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rng {
    s: [u64; 4],
    /// Cached second variate from Box–Muller, which produces normals in pairs.
    spare_normal: Option<u64>,
}

impl Rng {
    /// Seed the generator. The same seed always produces the same stream.
    pub fn new(seed: u64) -> Self {
        // SplitMix64 to expand one word into the four the generator needs;
        // seeding xoshiro directly from a small integer leaves it correlated
        // for the first few outputs.
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        Rng { s: [next(), next(), next(), next()], spare_normal: None }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[0].wrapping_add(self.s[3]).rotate_left(23).wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform in `[0, 1)`, using the 53 bits an `f64` can actually represent.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform in `[low, high)`.
    #[inline]
    pub fn uniform(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.next_f64()
    }

    /// Uniform integer in `[0, n)`. Unbiased via rejection — the usual modulo
    /// reduction skews the low values, which matters when it is choosing which
    /// order to fill.
    pub fn below(&mut self, n: u64) -> u64 {
        if n == 0 {
            return 0;
        }
        let zone = u64::MAX - (u64::MAX % n);
        loop {
            let v = self.next_u64();
            if v < zone {
                return v % n;
            }
        }
    }

    #[inline]
    pub fn bernoulli(&mut self, p: f64) -> bool {
        self.next_f64() < p
    }

    /// Standard normal, Box–Muller in polar form.
    pub fn normal(&mut self) -> f64 {
        if let Some(bits) = self.spare_normal.take() {
            return f64::from_bits(bits);
        }
        loop {
            let u = self.uniform(-1.0, 1.0);
            let v = self.uniform(-1.0, 1.0);
            let s = u * u + v * v;
            if s > 0.0 && s < 1.0 {
                let f = (-2.0 * s.ln() / s).sqrt();
                self.spare_normal = Some((v * f).to_bits());
                return u * f;
            }
        }
    }

    #[inline]
    pub fn gaussian(&mut self, mean: f64, sd: f64) -> f64 {
        mean + sd * self.normal()
    }

    /// Exponential with the given rate. Models inter-arrival times of order flow.
    pub fn exponential(&mut self, rate: f64) -> f64 {
        if rate <= 0.0 {
            return f64::INFINITY;
        }
        -(1.0 - self.next_f64()).ln() / rate
    }

    /// Fisher–Yates.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.below(i as u64 + 1) as usize;
            items.swap(i, j);
        }
    }

    pub fn choose<'a, T>(&mut self, items: &'a [T]) -> Option<&'a T> {
        if items.is_empty() {
            return None;
        }
        items.get(self.below(items.len() as u64) as usize)
    }
}

impl Default for Rng {
    fn default() -> Self {
        Rng::new(0x2545_F491_4F6C_DD1D)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats::Welford;

    #[test]
    fn the_same_seed_gives_the_same_stream() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_diverge_immediately() {
        // SplitMix64 expansion exists so that adjacent seeds are not correlated
        // in their first outputs.
        let mut a = Rng::new(1);
        let mut b = Rng::new(2);
        let first_a: Vec<u64> = (0..4).map(|_| a.next_u64()).collect();
        let first_b: Vec<u64> = (0..4).map(|_| b.next_u64()).collect();
        assert!(first_a.iter().zip(&first_b).all(|(x, y)| x != y));
    }

    #[test]
    fn uniforms_stay_in_range_and_look_uniform() {
        let mut r = Rng::new(7);
        let mut w = Welford::new();
        for _ in 0..200_000 {
            let x = r.next_f64();
            assert!((0.0..1.0).contains(&x));
            w.push(x);
        }
        assert!((w.mean() - 0.5).abs() < 0.005, "mean {}", w.mean());
        // Variance of U(0,1) is 1/12.
        assert!((w.variance() - 1.0 / 12.0).abs() < 0.002);
    }

    #[test]
    fn bounded_integers_cover_their_range_without_bias() {
        let mut r = Rng::new(9);
        let mut counts = [0u32; 7];
        for _ in 0..70_000 {
            counts[r.below(7) as usize] += 1;
        }
        for (i, c) in counts.iter().enumerate() {
            assert!((*c as i64 - 10_000).abs() < 600, "bucket {i} got {c}, expected ~10000");
        }
        assert_eq!(r.below(0), 0);
        assert_eq!(r.below(1), 0);
    }

    #[test]
    fn normals_have_the_right_moments() {
        let mut r = Rng::new(11);
        let mut w = Welford::new();
        for _ in 0..200_000 {
            w.push(r.normal());
        }
        assert!(w.mean().abs() < 0.01, "mean {}", w.mean());
        assert!((w.std_dev() - 1.0).abs() < 0.01, "sd {}", w.std_dev());
    }

    #[test]
    fn normals_are_produced_in_pairs_without_losing_one() {
        // Box-Muller yields two variates; the cached one must be returned, not
        // silently discarded, or half the stream is thrown away.
        let mut a = Rng::new(5);
        let first: Vec<f64> = (0..10).map(|_| a.normal()).collect();
        let mut b = Rng::new(5);
        let second: Vec<f64> = (0..10).map(|_| b.normal()).collect();
        assert_eq!(first, second);
        assert!(first.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn gaussian_shifts_and_scales() {
        let mut r = Rng::new(13);
        let mut w = Welford::new();
        for _ in 0..100_000 {
            w.push(r.gaussian(0.4, 0.05));
        }
        assert!((w.mean() - 0.4).abs() < 0.001);
        assert!((w.std_dev() - 0.05).abs() < 0.001);
    }

    #[test]
    fn bernoulli_matches_its_probability() {
        let mut r = Rng::new(17);
        let hits = (0..100_000).filter(|_| r.bernoulli(0.3)).count();
        assert!((hits as i64 - 30_000).abs() < 600, "got {hits}");
        // The degenerate cases must be exact, not approximately exact.
        assert!(!r.bernoulli(0.0));
        assert!(r.bernoulli(1.0));
    }

    #[test]
    fn exponential_has_mean_one_over_rate() {
        let mut r = Rng::new(19);
        let mut w = Welford::new();
        for _ in 0..200_000 {
            w.push(r.exponential(4.0));
        }
        assert!((w.mean() - 0.25).abs() < 0.005, "mean {}", w.mean());
        assert!(r.exponential(0.0).is_infinite());
    }

    #[test]
    fn shuffle_permutes_without_losing_elements() {
        let mut r = Rng::new(23);
        let mut v: Vec<u32> = (0..100).collect();
        r.shuffle(&mut v);
        assert_ne!(v, (0..100).collect::<Vec<_>>());
        v.sort_unstable();
        assert_eq!(v, (0..100).collect::<Vec<_>>());

        // Degenerate sizes must not panic.
        r.shuffle(&mut Vec::<u32>::new());
        r.shuffle(&mut [1u32]);
    }

    #[test]
    fn choose_returns_none_only_when_empty() {
        let mut r = Rng::new(29);
        assert!(r.choose::<u8>(&[]).is_none());
        assert_eq!(r.choose(&[7u8]), Some(&7));
    }

    #[test]
    fn state_round_trips_through_serde() {
        // A backtest must be resumable from a checkpoint at the exact point in
        // the random stream it stopped.
        let mut r = Rng::new(31);
        for _ in 0..100 {
            r.next_u64();
        }
        let saved = serde_json::to_string(&r).unwrap();
        let mut restored: Rng = serde_json::from_str(&saved).unwrap();
        assert_eq!(r.next_u64(), restored.next_u64());
    }
}
