//! Utility functions for Protocol-algorithm

use rand::Rng;

/// Generate random position in area
pub fn random_position<R: Rng>(rng: &mut R, width: f64, height: f64) -> (f64, f64) {
    let x = rng.gen_range(0.0..=width);
    let y = rng.gen_range(0.0..=height);
    (x, y)
}

/// Calculate distance between two points
pub fn distance(x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    let dx = x1 - x2;
    let dy = y1 - y2;
    (dx * dx + dy * dy).sqrt()
}

/// Calculate centroid of points
pub fn centroid(points: &[(f64, f64)]) -> (f64, f64) {
    if points.is_empty() {
        return (0.0, 0.0);
    }
    
    let sum_x: f64 = points.iter().map(|p| p.0).sum();
    let sum_y: f64 = points.iter().map(|p| p.1).sum();
    let n = points.len() as f64;
    
    (sum_x / n, sum_y / n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn test_distance() {
        assert_eq!(distance(0.0, 0.0, 3.0, 4.0), 5.0);
    }

    #[test]
    fn test_centroid() {
        let points = vec![(0.0, 0.0), (2.0, 0.0), (0.0, 2.0), (2.0, 2.0)];
        let c = centroid(&points);
        assert!((c.0 - 1.0).abs() < 0.001);
        assert!((c.1 - 1.0).abs() < 0.001);
    }
}
