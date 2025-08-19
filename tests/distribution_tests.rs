use monadic_ppl::*;
use rand::{rngs::StdRng, SeedableRng};

#[test]
fn normal_log_prob_symmetry() {
    let n = Normal {
        mu: 0.0,
        sigma: 2.0,
    };
    let lp1 = n.log_prob(1.0);
    let lp2 = n.log_prob(-1.0);
    assert!((lp1 - lp2).abs() < 1e-12);
}

#[test]
fn uniform_support_and_sampling() {
    let u = Uniform {
        low: -1.0,
        high: 1.0,
    };
    // Outside support is -inf
    assert!(u.log_prob(2.0).is_infinite());
    let mut rng = StdRng::seed_from_u64(9);
    let x = u.sample(&mut rng);
    assert!(x >= -1.0 && x <= 1.0);
}

#[test]
fn exponential_support() {
    let e = Exponential { rate: 2.0 };
    assert!(e.log_prob(-0.1).is_infinite());
    assert!(e.log_prob(0.0).is_finite());
}
