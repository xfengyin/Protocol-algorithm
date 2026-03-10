//! Random number generation utilities

use rand::Rng;
use rand_xoshiro::Xoshiro256PlusPlus;

/// Create seeded RNG for reproducibility
pub fn create_seeded_rng(seed: u64) -> Xoshiro256PlusPlus {
    Xoshiro256PlusPlus::seed_from_u64(seed)
}

/// Generate random float in range [min, max)
pub fn random_range<R: Rng>(rng: &mut R, min: f64, max: f64) -> f64 {
    rng.gen_range(min..=max)
}

/// Generate random integer in range [min, max)
pub fn random_range_int<R: Rng>(rng: &mut R, min: usize, max: usize) -> usize {
    rng.gen_range(min..=max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seeded_rng() {
        let mut rng1 = create_seeded_rng(42);
        let mut rng2 = create_seeded_rng(42);
        
        // Should produce same sequence
        assert_eq!(rng1.gen_range(0..100), rng2.gen_range(0..100));
    }
}
