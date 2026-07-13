//! Known-answer, boundary, regression, and constructor tests for the
//! distribution log-density implementations in `src/core/distribution.rs`.
//!
//! Reference constants are closed-form values (verified against the standard
//! `scipy.stats.*.logpdf` / `.logpmf` definitions); the Python expression that
//! produces each is shown in a comment. All comparisons use a 1e-9 tolerance,
//! which is well above the ~1e-12 agreement between `libm::lgamma` and the
//! reference `math.lgamma`, yet tight enough to catch a wrong constant,
//! parameterization (rate vs scale), or missing normalizer.
//!
//! Covers findings FG-06, FG-07, FG-08, FG-27, FG-28, FG-29, FG-30, FG-53.

use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

const TOL: f64 = 1e-9;

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < TOL,
        "expected {expected}, got {actual} (|Δ| = {})",
        (actual - expected).abs()
    );
}

// ---------------------------------------------------------------------------
// FG-06: interior-point known-answer tests for ALL distributions (2+ points).
// ---------------------------------------------------------------------------

#[test]
fn fg06_normal_interior_points() {
    // scipy.stats.norm(0,1).logpdf(0.0)
    close(
        Normal::new(0.0, 1.0).unwrap().log_prob(&0.0),
        -0.9189385332046727,
    );
    // scipy.stats.norm(1,2).logpdf(2.5)
    close(
        Normal::new(1.0, 2.0).unwrap().log_prob(&2.5),
        -1.893335713764618,
    );
}

#[test]
fn fg06_uniform_interior_points() {
    let u = Uniform::new(-2.0, 2.0).unwrap();
    // scipy.stats.uniform(-2,4).logpdf(x) = -ln(4) for x in [-2,2)
    close(u.log_prob(&0.0), -1.3862943611198906);
    close(u.log_prob(&1.5), -1.3862943611198906);
}

#[test]
fn fg06_lognormal_interior_points() {
    // scipy.stats.lognorm(s=1, scale=exp(0)).logpdf(1.0)
    close(
        LogNormal::new(0.0, 1.0).unwrap().log_prob(&1.0),
        -0.9189385332046727,
    );
    // scipy.stats.lognorm(s=1, scale=exp(0)).logpdf(2.0)
    close(
        LogNormal::new(0.0, 1.0).unwrap().log_prob(&2.0),
        -1.8523122207237186,
    );
    // scipy.stats.lognorm(s=0.75, scale=exp(0.5)).logpdf(1.5)
    close(
        LogNormal::new(0.5, 0.75).unwrap().log_prob(&1.5),
        -1.044665431781057,
    );
}

#[test]
fn fg06_exponential_interior_points() {
    // scipy.stats.expon(scale=1/2).logpdf(1.0)  -> ln(2) - 2
    close(
        Exponential::new(2.0).unwrap().log_prob(&1.0),
        -1.3068528194400546,
    );
    // scipy.stats.expon(scale=1/0.5).logpdf(3.0) -> ln(0.5) - 0.5*3
    close(
        Exponential::new(0.5).unwrap().log_prob(&3.0),
        -2.1931471805599454,
    );
}

#[test]
fn fg06_bernoulli_interior_points() {
    let b = Bernoulli::new(0.3).unwrap();
    close(b.log_prob(&true), -1.2039728043259361); // ln(0.3)
    close(b.log_prob(&false), -0.35667494393873245); // ln(0.7)
}

#[test]
fn fg06_categorical_interior_points() {
    let c = Categorical::new(vec![0.2, 0.3, 0.5]).unwrap();
    close(c.log_prob(&1), -1.2039728043259361); // ln(0.3)
    close(c.log_prob(&2), -std::f64::consts::LN_2); // ln(0.5) = -ln(2)
}

#[test]
fn fg06_beta_interior_points() {
    // scipy.stats.beta(2,3).logpdf(0.5)
    close(
        Beta::new(2.0, 3.0).unwrap().log_prob(&0.5),
        0.4054651081081637,
    );
    // scipy.stats.beta(2,5).logpdf(0.3)
    close(
        Beta::new(2.0, 5.0).unwrap().log_prob(&0.3),
        0.7705248015812911,
    );
}

#[test]
fn fg06_gamma_interior_points() {
    // scipy.stats.gamma(a=2, scale=1/1).logpdf(1.0) = -1.0
    close(Gamma::new(2.0, 1.0).unwrap().log_prob(&1.0), -1.0);
    // scipy.stats.gamma(a=3, scale=1/2).logpdf(1.5)
    close(
        Gamma::new(3.0, 2.0).unwrap().log_prob(&1.5),
        -0.8027754226637804,
    );
}

#[test]
fn fg06_binomial_interior_points() {
    // scipy.stats.binom(10,0.5).logpmf(5)
    close(
        Binomial::new(10, 0.5).unwrap().log_prob(&5),
        -1.4020427180880324,
    );
    // scipy.stats.binom(20,0.3).logpmf(7)
    close(
        Binomial::new(20, 0.3).unwrap().log_prob(&7),
        -1.8062926549204255,
    );
}

#[test]
fn fg06_poisson_interior_points() {
    // scipy.stats.poisson(3).logpmf(2)
    close(Poisson::new(3.0).unwrap().log_prob(&2), -1.4959226032237254);
    // scipy.stats.poisson(4).logpmf(7)
    close(Poisson::new(4.0).unwrap().log_prob(&7), -2.821100833226181);
}

// ---------------------------------------------------------------------------
// FG-07 / FG-08 / FG-30: previously-guarded points now return finite densities.
// Each of these was returned as -inf by the pre-fix bogus "overflow guards".
// ---------------------------------------------------------------------------

#[test]
fn fg07_gamma_large_argument_is_finite() {
    // Pre-fix: rate*x = 800 > 700 -> -inf (fired across the whole mass of any
    // Gamma with mean > ~700). scipy.stats.gamma(a=2, scale=1).logpdf(800).
    let lp = Gamma::new(2.0, 1.0).unwrap().log_prob(&800.0);
    assert!(lp.is_finite(), "FG-07: expected finite, got {lp}");
    close(lp, -793.315388272332);
}

#[test]
fn fg08_normal_large_residual_is_finite() {
    // Pre-fix: |z| = 50 > 37 -> -inf. scipy.stats.norm(0,0.001).logpdf(0.05).
    let lp = Normal::new(0.0, 0.001).unwrap().log_prob(&0.05);
    assert!(lp.is_finite(), "FG-08: expected finite, got {lp}");
    close(lp, -1244.0111832542225);

    // scipy.stats.norm(0,1).logpdf(40.0) (|z| = 40 > 37).
    close(
        Normal::new(0.0, 1.0).unwrap().log_prob(&40.0),
        -800.9189385332047,
    );
}

#[test]
fn fg08_lognormal_tight_sigma_is_finite() {
    // Pre-fix: |z| = ln(1.05)/0.001 ~= 48.8 > 37 -> -inf.
    // scipy.stats.lognorm(s=0.001, scale=exp(0)).logpdf(1.05).
    let lp = LogNormal::new(0.0, 0.001).unwrap().log_prob(&1.05);
    assert!(lp.is_finite(), "FG-08: expected finite, got {lp}");
    close(lp, -1184.3000332584572);
}

#[test]
fn fg30_exponential_large_argument_is_finite() {
    // Pre-fix: rate*x = 800 > 700 -> -inf. scipy.stats.expon(scale=1/2).logpdf(400).
    let lp = Exponential::new(2.0).unwrap().log_prob(&400.0);
    assert!(lp.is_finite(), "FG-30: expected finite, got {lp}");
    close(lp, -799.3068528194401);
}

// ---------------------------------------------------------------------------
// FG-27: Beta boundary semantics (matching scipy.stats.beta.logpdf).
// ---------------------------------------------------------------------------

#[test]
fn fg27_beta_subnormal_interior_no_longer_clipped() {
    // Pre-fix: line 813's hard 1e-100 cutoff returned -inf here.
    // scipy.stats.beta(0.5,0.5).logpdf(1e-100) -> large POSITIVE (density diverges).
    let lp = Beta::new(0.5, 0.5).unwrap().log_prob(&1e-100);
    assert!(lp.is_finite(), "FG-27: expected finite positive, got {lp}");
    close(lp, 113.98452476385289);

    // scipy.stats.beta(2,2).logpdf(1e-100) -> finite (large negative), also
    // wrongly returned as -inf pre-fix.
    let lp2 = Beta::new(2.0, 2.0).unwrap().log_prob(&1e-100);
    assert!(lp2.is_finite(), "FG-27: expected finite, got {lp2}");
    close(lp2, -228.46674983017652);
}

#[test]
fn fg27_beta_endpoint_limits() {
    // alpha == 1 at x = 0 -> finite ln(beta); scipy.stats.beta(1,5).logpdf(0.0).
    close(
        Beta::new(1.0, 5.0).unwrap().log_prob(&0.0),
        1.6094379124341003,
    ); // ln(5)
       // beta == 1 at x = 1 -> finite ln(alpha); scipy.stats.beta(3,1).logpdf(1.0).
    close(
        Beta::new(3.0, 1.0).unwrap().log_prob(&1.0),
        1.0986122886681098,
    ); // ln(3)

    // shape param > 1 -> density is 0 at that endpoint -> -inf.
    assert_eq!(
        Beta::new(2.0, 5.0).unwrap().log_prob(&0.0),
        f64::NEG_INFINITY
    );
    assert_eq!(
        Beta::new(2.0, 5.0).unwrap().log_prob(&1.0),
        f64::NEG_INFINITY
    );

    // shape param < 1 -> density diverges at that endpoint -> +inf.
    assert_eq!(Beta::new(0.5, 3.0).unwrap().log_prob(&0.0), f64::INFINITY);
    assert_eq!(Beta::new(2.0, 0.5).unwrap().log_prob(&1.0), f64::INFINITY);

    // Genuinely outside support is still -inf and never NaN.
    assert_eq!(
        Beta::new(2.0, 3.0).unwrap().log_prob(&-0.1),
        f64::NEG_INFINITY
    );
    assert_eq!(
        Beta::new(2.0, 3.0).unwrap().log_prob(&1.1),
        f64::NEG_INFINITY
    );
}

// ---------------------------------------------------------------------------
// FG-28: Binomial (and Bernoulli/Poisson) boundary parameters must be exact,
// never NaN.
// ---------------------------------------------------------------------------

#[test]
fn fg28_binomial_degenerate_p_is_exact_not_nan() {
    // p = 0: all mass on k = 0.
    let b0 = Binomial::new(5, 0.0).unwrap();
    assert!(!b0.log_prob(&0).is_nan(), "FG-28: p=0,k=0 must not be NaN");
    close(b0.log_prob(&0), 0.0);
    assert_eq!(b0.log_prob(&1), f64::NEG_INFINITY);
    assert_eq!(b0.log_prob(&5), f64::NEG_INFINITY);

    // p = 1: all mass on k = n.
    let b1 = Binomial::new(5, 1.0).unwrap();
    assert!(!b1.log_prob(&5).is_nan(), "FG-28: p=1,k=n must not be NaN");
    close(b1.log_prob(&5), 0.0);
    assert_eq!(b1.log_prob(&3), f64::NEG_INFINITY);
    assert_eq!(b1.log_prob(&0), f64::NEG_INFINITY);
}

#[test]
fn fg28_bernoulli_and_poisson_boundaries_no_nan() {
    // Bernoulli already branches on p<=0 / p>=1; confirm no 0*ln(0) NaN.
    let bern0 = Bernoulli::new(0.0).unwrap();
    close(bern0.log_prob(&false), 0.0);
    assert_eq!(bern0.log_prob(&true), f64::NEG_INFINITY);
    assert!(!bern0.log_prob(&false).is_nan());

    let bern1 = Bernoulli::new(1.0).unwrap();
    close(bern1.log_prob(&true), 0.0);
    assert_eq!(bern1.log_prob(&false), f64::NEG_INFINITY);
    assert!(!bern1.log_prob(&true).is_nan());

    // Poisson enforces lambda > 0 at construction, so it has no degenerate
    // boundary parameter; confirm the k=0 term is the finite -lambda, not NaN.
    let p = Poisson::new(3.0).unwrap();
    close(p.log_prob(&0), -3.0);
    assert!(!p.log_prob(&0).is_nan());
}

// ---------------------------------------------------------------------------
// FG-29: infallible convenience constructors for statically-valid cases.
// ---------------------------------------------------------------------------

#[test]
fn fg29_infallible_constructors() {
    let z = Normal::standard();
    assert_eq!((z.mu(), z.sigma()), (0.0, 1.0));
    close(z.log_prob(&0.0), -0.9189385332046727);

    let u = Uniform::unit();
    assert_eq!((u.low(), u.high()), (0.0, 1.0));
    close(u.log_prob(&0.5), 0.0); // -ln(1) = 0

    let prior = Beta::uniform_prior();
    assert_eq!((prior.alpha(), prior.beta()), (1.0, 1.0));
    // Beta(1,1) is Uniform(0,1): density 1 everywhere in (0,1).
    close(prior.log_prob(&0.3), 0.0);
    close(prior.log_prob(&0.5), 0.0);

    let coin = Bernoulli::fair();
    assert_eq!(coin.p(), 0.5);
    close(coin.log_prob(&true), -std::f64::consts::LN_2); // ln(0.5) = -ln(2)
}

// ---------------------------------------------------------------------------
// FG-53: Categorical caches the CDF, validates once, samples via binary search.
// ---------------------------------------------------------------------------

#[test]
fn fg53_categorical_log_prob_and_revalidate() {
    let c = Categorical::new(vec![0.1, 0.2, 0.3, 0.4]).unwrap();
    close(c.log_prob(&0), (0.1f64).ln());
    close(c.log_prob(&3), (0.4f64).ln());
    // Out-of-bounds index is a clean -inf (no panic, no NaN).
    assert_eq!(c.log_prob(&4), f64::NEG_INFINITY);
    // The cached invariant can be re-asserted on demand.
    assert!(c.revalidate().is_ok());

    // A zero-probability category returns -inf.
    let c2 = Categorical::new(vec![0.0, 1.0]).unwrap();
    assert_eq!(c2.log_prob(&0), f64::NEG_INFINITY);
    close(c2.log_prob(&1), 0.0);
}

#[test]
fn fg53_categorical_sample_matches_probabilities() {
    // Seeded so the test is deterministic. With N = 200_000 draws the sample
    // proportion of the rarest category (p = 0.1) has std ~= sqrt(0.1*0.9/N)
    // ~= 6.7e-4, so a 5e-3 tolerance is > 7 standard errors — tight enough to
    // catch a broken binary search / off-by-one, loose enough to never flake.
    let probs = vec![0.1, 0.2, 0.3, 0.4];
    let c = Categorical::new(probs.clone()).unwrap();
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let n = 200_000usize;
    let mut counts = [0usize; 4];
    for _ in 0..n {
        let i = c.sample(&mut rng);
        assert!(i < 4, "sample out of range: {i}");
        counts[i] += 1;
    }
    for (k, &p) in probs.iter().enumerate() {
        let freq = counts[k] as f64 / n as f64;
        assert!(
            (freq - p).abs() < 5e-3,
            "category {k}: empirical {freq} vs expected {p}"
        );
    }
}
