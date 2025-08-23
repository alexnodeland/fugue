use fugue::*;
use rand::{rngs::StdRng, SeedableRng};

#[test]
fn normal_normalization() {
    let n = Normal::new(0.0, 1.0).unwrap();
    let mut rng = StdRng::seed_from_u64(42);

    // Sample many points and check they're reasonable
    let mut samples = Vec::new();
    for _ in 0..1000 {
        samples.push(n.sample(&mut rng));
    }

    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;

    assert!((mean - 0.0).abs() < 0.1); // Should be close to true mean
    assert!((variance - 1.0).abs() < 0.2); // Should be close to true variance
}

#[test]
fn bernoulli_properties() {
    let b = Bernoulli::new(0.7).unwrap();

    // Valid outcomes - now using natural bool values
    let lp_false = b.log_prob(&false); // P(false) = 1-p = 0.3
    let lp_true = b.log_prob(&true); // P(true) = p = 0.7
    println!("lp_false: {}, lp_true: {}", lp_false, lp_true);
    // For p=0.7: P(false) = 0.3, P(true) = 0.7
    assert!(lp_false.is_finite());
    assert!(lp_true.is_finite());
    assert!(lp_true > lp_false); // p=0.7 so P(true) > P(false)

    // No invalid outcomes for bool - only true/false are valid
    // This test is no longer needed since Bernoulli only accepts bool values

    // Sampling should produce bools
    let mut rng = StdRng::seed_from_u64(123);
    let mut count_true = 0;
    let n_samples = 1000;

    for _ in 0..n_samples {
        let x = b.sample(&mut rng);
        // x is now naturally bool
        if x {
            count_true += 1;
        }
    }

    let empirical_p = count_true as f64 / n_samples as f64;
    assert!((empirical_p - 0.7).abs() < 0.05); // Should be close to true p
}

#[test]
fn categorical_properties() {
    let probs = vec![0.2, 0.3, 0.5];
    let c = Categorical::new(probs.clone()).unwrap();

    // Valid outcomes - now using usize directly
    for i in 0..3 {
        let lp = c.log_prob(&i);
        assert!(lp.is_finite());
        assert!((lp - probs[i].ln()).abs() < 1e-12);
    }

    // Invalid outcome - index out of bounds
    assert_eq!(c.log_prob(&5), f64::NEG_INFINITY);

    // Sampling should respect probabilities
    let mut rng = StdRng::seed_from_u64(456);
    let mut counts = vec![0; 3];
    let n_samples = 1000;

    for _ in 0..n_samples {
        let idx = c.sample(&mut rng);
        // idx is now naturally usize
        assert!(idx < 3);
        counts[idx] += 1;
    }

    for i in 0..3 {
        let empirical_p = counts[i] as f64 / n_samples as f64;
        assert!((empirical_p - probs[i]).abs() < 0.05);
    }
}

#[test]
fn beta_support() {
    let b = Beta::new(2.0, 3.0).unwrap();

    // Support is (0, 1)
    assert_eq!(b.log_prob(&0.0), f64::NEG_INFINITY);
    assert_eq!(b.log_prob(&1.0), f64::NEG_INFINITY);
    assert_eq!(b.log_prob(&-0.1), f64::NEG_INFINITY);
    assert_eq!(b.log_prob(&1.1), f64::NEG_INFINITY);

    // Inside support should be finite
    assert!(b.log_prob(&0.5).is_finite());

    // Sampling should stay in support
    let mut rng = StdRng::seed_from_u64(789);
    for _ in 0..100 {
        let x = b.sample(&mut rng);
        assert!(x > 0.0 && x < 1.0);
    }
}

#[test]
fn gamma_support() {
    let g = Gamma::new(2.0, 1.0).unwrap();

    // Support is (0, ∞)
    assert_eq!(g.log_prob(&0.0), f64::NEG_INFINITY);
    assert_eq!(g.log_prob(&-0.1), f64::NEG_INFINITY);

    // Positive values should be finite
    assert!(g.log_prob(&0.1).is_finite());
    assert!(g.log_prob(&1.0).is_finite());
    assert!(g.log_prob(&10.0).is_finite());

    // Sampling should be positive
    let mut rng = StdRng::seed_from_u64(101112);
    for _ in 0..100 {
        let x = g.sample(&mut rng);
        assert!(x > 0.0);
    }
}

#[test]
fn binomial_support() {
    let b = Binomial::new(10, 0.3).unwrap();

    // Valid outcomes: 0, 1, ..., n (now using u64 directly)
    for k in 0..=10 {
        let lp = b.log_prob(&k);
        assert!(lp.is_finite());
    }

    // Invalid outcome: k > n
    assert_eq!(b.log_prob(&15), f64::NEG_INFINITY);

    // Sampling should be in range
    let mut rng = StdRng::seed_from_u64(131415);
    for _ in 0..100 {
        let x = b.sample(&mut rng);
        // x is now naturally u64
        assert!(x <= 10);
    }
}

#[test]
fn poisson_support() {
    let p = Poisson::new(2.0).unwrap();

    // Valid outcomes: 0, 1, 2, ... (now using u64 directly)
    for k in 0..20 {
        let lp = p.log_prob(&k);
        assert!(lp.is_finite());
    }

    // No invalid outcomes for u64 - all non-negative integers are valid

    // Sampling should be non-negative integers
    let mut rng = StdRng::seed_from_u64(161718);
    for _ in 0..100 {
        let x = p.sample(&mut rng);
        // x is now naturally u64 - always non-negative integer by definition
        assert!(x < 1000); // Just a sanity check for reasonable values
    }
}
