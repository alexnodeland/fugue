//! Random draws that come out the same on every target.
//!
//! `rand` 0.8 samples a `usize` range through `Standard::<usize>`, which reads
//! `next_u32` on 32-bit targets (wasm32 among them) and `next_u64` on 64-bit
//! ones. So `rng.gen_range(0..n)` with `n: usize` picks a different index, and
//! consumes a different number of words, on wasm than on a 64-bit host, and a
//! seeded run stops being reproducible across them.
//!
//! [`gen_index`] draws the index as a `u64` instead. On a 64-bit target that
//! is exactly the draw a `usize` range always made: `rand` implements
//! `UniformInt` for `usize` and `u64` with the same zone, the same rejection
//! loop and the same widening multiply, one `next_u64` per attempt. So native
//! output is unchanged and 32-bit targets now match it.
//!
//! Rule for library code: never draw a `usize` with `gen_range` or
//! `gen::<usize>()`. Use [`gen_index`].

use rand::Rng;

/// A uniform index in `0..n`, drawn identically on 32- and 64-bit targets.
///
/// Panics if `n == 0`, as `rng.gen_range(0..0)` does.
#[inline]
pub(crate) fn gen_index<R: Rng + ?Sized>(rng: &mut R, n: usize) -> usize {
    rng.gen_range(0..n as u64) as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{RngCore, SeedableRng};

    /// Every `n` the comparison covers: small counts, powers of two and their
    /// neighbours (where the rejection zone changes), and, on 64-bit, counts
    /// past `u32::MAX` up to `usize::MAX`, where rejection is frequent.
    fn counts() -> Vec<usize> {
        let mut ns: Vec<usize> = vec![1, 2, 3, 4, 5, 6, 7, 10, 17, 100, 255, 256, 257, 1000, 4096];
        ns.extend([
            65_535,
            65_536,
            1_000_003,
            (1 << 31) - 1,
            1 << 31,
            (1 << 31) + 1,
        ]);
        ns.push(u32::MAX as usize);
        #[cfg(target_pointer_width = "64")]
        ns.extend([
            1usize << 32,
            (1usize << 32) + 1,
            (1usize << 32) * 3 + 7,
            1usize << 40,
            usize::MAX / 3,
            (1usize << 63) - 1,
            1usize << 63,
            (1usize << 63) + 1,
            usize::MAX - 1,
            usize::MAX,
        ]);
        ns
    }

    /// On 64-bit, `gen_index` is the draw `gen_range(0..n)` over `usize`
    /// always made: same index, and the same number of words consumed, which
    /// the next raw word after each draw proves. This is what keeps every
    /// seeded result and golden value in the suite where it was.
    #[test]
    #[cfg(target_pointer_width = "64")]
    fn matches_the_old_usize_draw_on_64_bit() {
        for seed in 0..64u64 {
            let mut old = StdRng::seed_from_u64(seed);
            let mut new = StdRng::seed_from_u64(seed);
            for &n in &counts() {
                for _ in 0..8 {
                    let a: usize = old.gen_range(0..n);
                    let b = gen_index(&mut new, n);
                    assert_eq!(a, b, "seed {seed}, n {n}");
                    assert_eq!(
                        old.next_u64(),
                        new.next_u64(),
                        "seed {seed}, n {n}: words consumed differ"
                    );
                }
            }
        }
    }

    /// Works through `dyn RngCore`, as `PopulationKernel::sweep` hands it.
    #[test]
    fn works_through_a_trait_object() {
        let mut rng = StdRng::seed_from_u64(7);
        let dyn_rng: &mut dyn RngCore = &mut rng;
        for n in 1..50 {
            assert!(gen_index(dyn_rng, n) < n);
        }
    }

    /// The draw for a fixed seed, pinned. These constants hold on every target;
    /// a change to them is a change to every seeded run in the crate.
    #[test]
    fn pinned_sequence() {
        let mut rng = StdRng::seed_from_u64(42);
        let got: Vec<usize> = [2, 3, 7, 10, 100, 1000, 65_536, u32::MAX as usize]
            .iter()
            .map(|&n| gen_index(&mut rng, n))
            .collect();
        assert_eq!(got, vec![1, 1, 4, 4, 3, 414, 8603, 13_967_647]);
        assert_eq!(rng.next_u64(), 17_195_042_692_806_716_983);
    }

    #[test]
    #[should_panic]
    fn empty_range_panics() {
        let mut rng = StdRng::seed_from_u64(0);
        gen_index(&mut rng, 0);
    }
}
