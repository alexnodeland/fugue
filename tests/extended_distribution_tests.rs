use fugue::*;
use rand::{rngs::StdRng, SeedableRng};

#[test]
fn normal_normalization() {
    let n = Normal{mu: 0.0, sigma: 1.0};
    let mut rng = StdRng::seed_from_u64(42);
    
    // Sample many points and check they're reasonable
    let mut samples = Vec::new();
    for _ in 0..1000 {
        samples.push(n.sample(&mut rng));
    }
    
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;
    
    assert!((mean - 0.0).abs() < 0.1);  // Should be close to true mean
    assert!((variance - 1.0).abs() < 0.2);  // Should be close to true variance
}

#[test]
fn bernoulli_properties() {
    let b = Bernoulli{p: 0.7};
    
    // Valid outcomes
    let lp0 = b.log_prob(0.0);
    let lp1 = b.log_prob(1.0);
    assert!(lp0.is_finite());
    assert!(lp1.is_finite());
    assert!(lp1 > lp0);  // p=0.7 so P(1) > P(0)
    
    // Invalid outcomes
    assert_eq!(b.log_prob(0.5), f64::NEG_INFINITY);
    assert_eq!(b.log_prob(2.0), f64::NEG_INFINITY);
    
    // Sampling should produce 0s and 1s
    let mut rng = StdRng::seed_from_u64(123);
    let mut count_ones = 0;
    let n_samples = 1000;
    
    for _ in 0..n_samples {
        let x = b.sample(&mut rng);
        assert!(x == 0.0 || x == 1.0);
        if x == 1.0 { count_ones += 1; }
    }
    
    let empirical_p = count_ones as f64 / n_samples as f64;
    assert!((empirical_p - 0.7).abs() < 0.05);  // Should be close to true p
}

#[test]
fn categorical_properties() {
    let probs = vec![0.2, 0.3, 0.5];
    let c = Categorical{probs: probs.clone()};
    
    // Valid outcomes
    for i in 0..3 {
        let lp = c.log_prob(i as f64);
        assert!(lp.is_finite());
        assert!((lp - probs[i].ln()).abs() < 1e-12);
    }
    
    // Invalid outcomes
    assert_eq!(c.log_prob(3.0), f64::NEG_INFINITY);
    assert_eq!(c.log_prob(-1.0), f64::NEG_INFINITY);
    assert_eq!(c.log_prob(1.5), f64::NEG_INFINITY);
    
    // Sampling should respect probabilities
    let mut rng = StdRng::seed_from_u64(456);
    let mut counts = vec![0; 3];
    let n_samples = 1000;
    
    for _ in 0..n_samples {
        let x = c.sample(&mut rng);
        let idx = x as usize;
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
    let b = Beta{alpha: 2.0, beta: 3.0};
    
    // Support is (0, 1)
    assert_eq!(b.log_prob(0.0), f64::NEG_INFINITY);
    assert_eq!(b.log_prob(1.0), f64::NEG_INFINITY);
    assert_eq!(b.log_prob(-0.1), f64::NEG_INFINITY);
    assert_eq!(b.log_prob(1.1), f64::NEG_INFINITY);
    
    // Inside support should be finite
    assert!(b.log_prob(0.5).is_finite());
    
    // Sampling should stay in support
    let mut rng = StdRng::seed_from_u64(789);
    for _ in 0..100 {
        let x = b.sample(&mut rng);
        assert!(x > 0.0 && x < 1.0);
    }
}

#[test]
fn gamma_support() {
    let g = Gamma{shape: 2.0, rate: 1.0};
    
    // Support is (0, ∞)
    assert_eq!(g.log_prob(0.0), f64::NEG_INFINITY);
    assert_eq!(g.log_prob(-0.1), f64::NEG_INFINITY);
    
    // Positive values should be finite
    assert!(g.log_prob(0.1).is_finite());
    assert!(g.log_prob(1.0).is_finite());
    assert!(g.log_prob(10.0).is_finite());
    
    // Sampling should be positive
    let mut rng = StdRng::seed_from_u64(101112);
    for _ in 0..100 {
        let x = g.sample(&mut rng);
        assert!(x > 0.0);
    }
}

#[test]
fn binomial_support() {
    let b = Binomial{n: 10, p: 0.3};
    
    // Valid outcomes: 0, 1, ..., n
    for k in 0..=10 {
        let lp = b.log_prob(k as f64);
        assert!(lp.is_finite());
    }
    
    // Invalid outcomes
    assert_eq!(b.log_prob(11.0), f64::NEG_INFINITY);
    assert_eq!(b.log_prob(-1.0), f64::NEG_INFINITY);
    assert_eq!(b.log_prob(5.5), f64::NEG_INFINITY);
    
    // Sampling should be in range
    let mut rng = StdRng::seed_from_u64(131415);
    for _ in 0..100 {
        let x = b.sample(&mut rng);
        assert!(x >= 0.0 && x <= 10.0);
        assert!((x - x.round()).abs() < 1e-12);  // Should be integer
    }
}

#[test]
fn poisson_support() {
    let p = Poisson{lambda: 2.0};
    
    // Valid outcomes: 0, 1, 2, ...
    for k in 0..20 {
        let lp = p.log_prob(k as f64);
        assert!(lp.is_finite());
    }
    
    // Invalid outcomes
    assert_eq!(p.log_prob(-1.0), f64::NEG_INFINITY);
    assert_eq!(p.log_prob(2.5), f64::NEG_INFINITY);
    
    // Sampling should be non-negative integers
    let mut rng = StdRng::seed_from_u64(161718);
    for _ in 0..100 {
        let x = p.sample(&mut rng);
        assert!(x >= 0.0);
        assert!((x - x.round()).abs() < 1e-12);  // Should be integer
    }
}
