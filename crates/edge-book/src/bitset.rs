//! A dense bitset over the price grid, used to find the best bid and offer.
//!
//! Prediction markets have a property equity markets do not: the price grid is
//! **small and bounded**. A contract trades in `(0, $1)` on a fixed tick, so
//! there are at most 999 possible prices even at a tenth-of-a-cent tick. That
//! makes a general-purpose ordered map — the standard order-book data structure,
//! and what a port of the reference C++ implementation would use — the wrong
//! tool. A `BTreeMap` pays a pointer chase per level and `O(log n)` to find the
//! best price; here the whole occupancy of one side of the book fits in 16
//! machine words, and the best price is a single `leading_zeros` instruction.

/// Occupancy bitmap over tick indices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TickBitset {
    words: Vec<u64>,
    n_bits: usize,
    count: usize,
}

const BITS: usize = 64;

impl TickBitset {
    pub fn new(n_bits: usize) -> Self {
        TickBitset { words: vec![0; n_bits.div_ceil(BITS).max(1)], n_bits, count: 0 }
    }

    #[inline]
    pub const fn len(&self) -> usize {
        self.n_bits
    }

    /// Number of set bits, maintained incrementally.
    #[inline]
    pub const fn count(&self) -> usize {
        self.count
    }

    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    #[inline]
    pub fn get(&self, i: usize) -> bool {
        if i >= self.n_bits {
            return false;
        }
        self.words[i / BITS] & (1u64 << (i % BITS)) != 0
    }

    #[inline]
    pub fn set(&mut self, i: usize) {
        if i >= self.n_bits {
            return;
        }
        let (w, b) = (i / BITS, i % BITS);
        if self.words[w] & (1u64 << b) == 0 {
            self.words[w] |= 1u64 << b;
            self.count += 1;
        }
    }

    #[inline]
    pub fn clear(&mut self, i: usize) {
        if i >= self.n_bits {
            return;
        }
        let (w, b) = (i / BITS, i % BITS);
        if self.words[w] & (1u64 << b) != 0 {
            self.words[w] &= !(1u64 << b);
            self.count -= 1;
        }
    }

    pub fn clear_all(&mut self) {
        self.words.iter_mut().for_each(|w| *w = 0);
        self.count = 0;
    }

    /// Highest set bit — the best bid.
    pub fn highest(&self) -> Option<usize> {
        for (wi, &w) in self.words.iter().enumerate().rev() {
            if w != 0 {
                return Some(wi * BITS + (BITS - 1 - w.leading_zeros() as usize));
            }
        }
        None
    }

    /// Lowest set bit — the best offer.
    pub fn lowest(&self) -> Option<usize> {
        for (wi, &w) in self.words.iter().enumerate() {
            if w != 0 {
                return Some(wi * BITS + w.trailing_zeros() as usize);
            }
        }
        None
    }

    /// Highest set bit strictly below `i`. Walks the book downward one level at
    /// a time while an aggressive order sweeps.
    pub fn highest_below(&self, i: usize) -> Option<usize> {
        if i == 0 {
            return None;
        }
        let i = i.min(self.n_bits);
        let (mut wi, b) = ((i - 1) / BITS, (i - 1) % BITS);
        // Mask off everything at or above `i` in the starting word.
        let mask = if b == BITS - 1 { u64::MAX } else { (1u64 << (b + 1)) - 1 };
        let mut w = self.words[wi] & mask;
        loop {
            if w != 0 {
                return Some(wi * BITS + (BITS - 1 - w.leading_zeros() as usize));
            }
            if wi == 0 {
                return None;
            }
            wi -= 1;
            w = self.words[wi];
        }
    }

    /// Lowest set bit strictly above `i`.
    pub fn lowest_above(&self, i: usize) -> Option<usize> {
        let start = i + 1;
        if start >= self.n_bits {
            return None;
        }
        let (mut wi, b) = (start / BITS, start % BITS);
        let mut w = self.words[wi] & !((1u64 << b) - 1);
        loop {
            if w != 0 {
                return Some(wi * BITS + w.trailing_zeros() as usize);
            }
            wi += 1;
            if wi >= self.words.len() {
                return None;
            }
            w = self.words[wi];
        }
    }

    /// Set bits in ascending order.
    pub fn iter_ascending(&self) -> impl Iterator<Item = usize> + '_ {
        let mut next = self.lowest();
        std::iter::from_fn(move || {
            let cur = next?;
            next = self.lowest_above(cur);
            Some(cur)
        })
    }

    /// Set bits in descending order.
    pub fn iter_descending(&self) -> impl Iterator<Item = usize> + '_ {
        let mut next = self.highest();
        std::iter::from_fn(move || {
            let cur = next?;
            next = self.highest_below(cur);
            Some(cur)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_and_clear_maintain_the_count() {
        let mut b = TickBitset::new(200);
        assert!(b.is_empty());
        b.set(5);
        b.set(5); // idempotent
        b.set(199);
        assert_eq!(b.count(), 2);
        assert!(b.get(5) && b.get(199) && !b.get(6));
        b.clear(5);
        b.clear(5);
        assert_eq!(b.count(), 1);
        assert!(!b.is_empty());
        b.clear(199);
        assert!(b.is_empty());
    }

    #[test]
    fn out_of_range_access_is_a_no_op_not_a_panic() {
        let mut b = TickBitset::new(64);
        b.set(64);
        b.set(usize::MAX);
        b.clear(1000);
        assert!(b.is_empty());
        assert!(!b.get(64));
    }

    #[test]
    fn extremes_are_found_across_word_boundaries() {
        let mut b = TickBitset::new(300);
        for i in [3, 63, 64, 127, 128, 299] {
            b.set(i);
        }
        assert_eq!(b.lowest(), Some(3));
        assert_eq!(b.highest(), Some(299));
    }

    #[test]
    fn empty_bitset_has_no_extremes() {
        let b = TickBitset::new(128);
        assert_eq!(b.lowest(), None);
        assert_eq!(b.highest(), None);
        assert_eq!(b.highest_below(64), None);
        assert_eq!(b.lowest_above(0), None);
    }

    #[test]
    fn neighbour_search_is_strict_and_crosses_words() {
        let mut b = TickBitset::new(300);
        for i in [10, 63, 64, 65, 200] {
            b.set(i);
        }
        // Strictly below / above: the bit at the pivot itself never counts.
        assert_eq!(b.highest_below(64), Some(63));
        assert_eq!(b.highest_below(63), Some(10));
        assert_eq!(b.highest_below(10), None);
        assert_eq!(b.lowest_above(63), Some(64));
        assert_eq!(b.lowest_above(65), Some(200));
        assert_eq!(b.lowest_above(200), None);
        assert_eq!(b.highest_below(0), None);
    }

    #[test]
    fn iteration_visits_every_bit_in_order() {
        let mut b = TickBitset::new(300);
        let bits = [1usize, 63, 64, 130, 299];
        for &i in &bits {
            b.set(i);
        }
        assert_eq!(b.iter_ascending().collect::<Vec<_>>(), bits.to_vec());
        let mut rev = bits.to_vec();
        rev.reverse();
        assert_eq!(b.iter_descending().collect::<Vec<_>>(), rev);
    }

    #[test]
    fn a_dense_bitset_round_trips() {
        // Every tick on a tenth-of-a-cent grid.
        let mut b = TickBitset::new(1000);
        for i in 0..1000 {
            b.set(i);
        }
        assert_eq!(b.count(), 1000);
        assert_eq!(b.lowest(), Some(0));
        assert_eq!(b.highest(), Some(999));
        assert_eq!(b.iter_ascending().count(), 1000);
        b.clear_all();
        assert!(b.is_empty());
    }
}
