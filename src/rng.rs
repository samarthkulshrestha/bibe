//! Process-wide seedable RNG for weight initialization and dropout.
//!
//! All random parameter initialization draws from this thread-local generator
//! so a single [`seed`] call makes a run reproducible without threading an RNG
//! through every layer constructor. Without seeding it is initialized from
//! system entropy, preserving randomized behavior by default.

use std::cell::RefCell;

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

thread_local! {
    static RNG: RefCell<StdRng> = RefCell::new({
        let entropy: u64 = rand::rng().random();
        StdRng::seed_from_u64(entropy)
    });
}

/// Reseed the generator on the current thread, making subsequent random
/// initialization deterministic.
pub fn seed(seed: u64) {
    RNG.with(|r| *r.borrow_mut() = StdRng::seed_from_u64(seed));
}

/// Run `f` with a mutable borrow of the thread-local generator.
pub(crate) fn with_rng<T>(f: impl FnOnce(&mut StdRng) -> T) -> T {
    RNG.with(|r| f(&mut r.borrow_mut()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_seed_makes_randn_reproducible() {
        seed(123);
        let a = Tensor::randn(&[6]);
        seed(123);
        let b = Tensor::randn(&[6]);
        assert_eq!(a.data, b.data, "same seed must reproduce the same weights");
    }

    #[test]
    fn test_different_seeds_differ() {
        seed(1);
        let a = Tensor::xaviern(&[4, 4]);
        seed(2);
        let b = Tensor::xaviern(&[4, 4]);
        assert_ne!(a.data, b.data, "different seeds should differ");
    }
}
