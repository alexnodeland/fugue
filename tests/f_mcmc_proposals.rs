//! Statistical known-answer regressions for the Metropolis-Hastings proposal
//! machinery (audit findings FG-02, FG-10, FG-41, FG-42).
//!
//! Every test is seeded (`StdRng::seed_from_u64`) and its tolerance is justified
//! in comments. Each targets a specific pre-fix bias so it fails on the
//! unremediated sampler and passes after the fix.

use fugue::inference::mh::SiteProposal;
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::HashMap;

/// Sample mean of the `f64` values at `addr` across a chain of `(A, Trace)`.
fn f64_values<A>(samples: &[(A, Trace)], addr: &Address) -> Vec<f64> {
    samples
        .iter()
        .filter_map(|(_, t)| t.get_f64(addr))
        .collect()
}

// FG-02 (CRITICAL): the log-space random-walk proposal must include the
// Jacobian/Hastings correction +(ln x' − ln x). Target Gamma(shape=3, rate=2),
// whose mean is shape/rate = 1.5. With no likelihood the posterior equals the
// prior, so the chain samples Gamma(3,2) directly, and the positive-support
// site is auto-routed to the log-space walk (FG-42). Without the correction the
// coded kernel targets π(x)/x ∝ Gamma(2,2), whose mean is (shape-1)/rate = 1.0
// (the documented pre-fix value); with the correction the mean is 1.5.
#[test]
fn fg02_log_space_walk_targets_gamma_mean() {
    let model_fn = || sample(addr!("x"), Gamma::new(3.0, 2.0).unwrap());
    let mut rng = StdRng::seed_from_u64(20260710);
    let samples = adaptive_mcmc_chain(&mut rng, model_fn, 12_000, 3_000);
    let xs = f64_values(&samples, &addr!("x"));
    assert!(!xs.is_empty());

    let mean = xs.iter().sum::<f64>() / xs.len() as f64;
    let var = xs.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (xs.len() - 1) as f64;
    let sd = var.sqrt();
    // Autocorrelation-aware standard error of the mean.
    let ess = effective_sample_size_mcmc(&xs);
    let se = sd / ess.sqrt();

    // Target mean 1.5 within 3·SE. The pre-fix mean (1.0) is many SE away and
    // would fail this bound; the extra sanity bound documents that explicitly.
    assert!(
        (mean - 1.5).abs() < 3.0 * se,
        "Gamma(3,2) posterior mean {mean:.4} not within 3·SE ({:.4}) of 1.5 (ess={ess:.1})",
        3.0 * se
    );
    assert!(
        mean > 1.25,
        "mean {mean:.4} collapsed toward the pre-fix Gamma(2,2) mean of 1.0"
    );
}

// FG-02 via the explicit override API (FG-42): forcing LogSpace on an
// arbitrarily-named address must still hit the correct mean, proving the
// correction is wired through the override path too.
#[test]
fn fg02_log_space_override_targets_gamma_mean() {
    let model_fn = || sample(addr!("theta"), Gamma::new(3.0, 2.0).unwrap());
    let mut overrides: HashMap<Address, SiteProposal> = HashMap::new();
    overrides.insert(addr!("theta"), SiteProposal::LogSpace);

    let mut rng = StdRng::seed_from_u64(13371337);
    let samples = adaptive_mcmc_chain_with_overrides(&mut rng, model_fn, 12_000, 3_000, &overrides);
    let xs = f64_values(&samples, &addr!("theta"));

    let mean = xs.iter().sum::<f64>() / xs.len() as f64;
    let var = xs.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (xs.len() - 1) as f64;
    let se = var.sqrt() / effective_sample_size_mcmc(&xs).sqrt();
    assert!(
        (mean - 1.5).abs() < 3.0 * se,
        "override LogSpace mean {mean:.4} not within 3·SE of 1.5"
    );
}

// FG-42: proposal choice must come from the distribution's support, not from
// address-name substrings. The pre-fix heuristic routed any address containing
// the letter "p" whose current value fell in [0,1] to a reflected walk confined
// to [0,1] — a one-way absorbing trap that permanently confines an
// unbounded-support parameter and breaks ergodicity. Here the target is
// Normal(0.5, 2.0) (full support, ~80% of its mass outside [0,1]) at an address
// literally named "p". Once the pre-fix chain wandered into [0,1] it could never
// leave, collapsing the samples into [0,1]; the support-based selection uses a
// Gaussian walk and explores the whole line.
#[test]
fn fg42_name_heuristic_no_longer_traps_unbounded_parameter() {
    let model_fn = || sample(addr!("p"), Normal::new(0.5, 2.0).unwrap());
    let mut rng = StdRng::seed_from_u64(31415);
    let samples = adaptive_mcmc_chain(&mut rng, model_fn, 20_000, 4_000);
    let xs = f64_values(&samples, &addr!("p"));

    let outside =
        xs.iter().filter(|&&x| !(0.0..=1.0).contains(&x)).count() as f64 / xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / xs.len() as f64;
    let sd = (xs.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (xs.len() - 1) as f64).sqrt();

    // Normal(0.5, 2.0): P(x ∉ [0,1]) = 1 - (Φ(0.25) - Φ(-0.25)) ≈ 0.803
    // (scipy: 1 - (norm.cdf(0.25) - norm.cdf(-0.25))). A [0,1]-trapped chain would
    // put ~0% outside and have sd « 0.3; require a majority outside and sd near 2.
    assert!(
        outside > 0.5,
        "only {:.2} of samples fell outside [0,1]; parameter appears trapped",
        outside
    );
    assert!(
        sd > 1.0,
        "sample sd {sd:.3} far below the target sd of 2.0 (trapped?)"
    );
}

// FG-10: categorical/usize sites are proposed by resampling from the site's
// prior, which makes acceptance reduce to the likelihood ratio and can never
// miss the support. Three-category mixture-indicator model:
//   z ~ Categorical([0.2, 0.3, 0.5]);  y=1.0 ~ Normal(means[z], 1.0), means=[0,1,2].
// The exact posterior is P(z=k) ∝ prior[k]·N(1; means[k], 1). The 1/sqrt(2π)
// cancels, leaving weights ∝ [0.2·e^{-1/2}, 0.3·1, 0.5·e^{-1/2}]
//   = [0.12130614, 0.3, 0.30326535], sum 0.72457149
//   → P = [0.167416, 0.414036, 0.418548]  (scipy: w=[0.2*exp(-0.5),0.3,0.5*exp(-0.5)]; w/ w.sum()).
// The pre-fix asymmetric UniformCategoricalProposal over-weighted higher indices.
#[test]
fn fg10_categorical_prior_resample_recovers_posterior() {
    let model_fn = || {
        sample(addr!("z"), Categorical::new(vec![0.2, 0.3, 0.5]).unwrap()).and_then(|z| {
            let means = [0.0, 1.0, 2.0];
            observe(addr!("y"), Normal::new(means[z], 1.0).unwrap(), 1.0).map(move |_| z)
        })
    };
    let mut rng = StdRng::seed_from_u64(555);
    let samples = adaptive_mcmc_chain(&mut rng, model_fn, 40_000, 5_000);

    let mut counts = [0usize; 3];
    for (_z, t) in &samples {
        if let Some(z) = t.get_usize(&addr!("z")) {
            counts[z] += 1;
        }
    }
    let n = samples.len() as f64;
    let emp = [
        counts[0] as f64 / n,
        counts[1] as f64 / n,
        counts[2] as f64 / n,
    ];
    let expected = [0.167416_f64, 0.414036, 0.418548];
    for k in 0..3 {
        // Tolerance 0.02: with 40k dependent draws the Monte Carlo error on each
        // category probability is well under this, while the pre-fix bias on the
        // top categories exceeded it.
        assert!(
            (emp[k] - expected[k]).abs() < 0.02,
            "P(z={k}) = {:.4}, expected {:.4}",
            emp[k],
            expected[k]
        );
    }
}

// FG-41: the reflected discrete walk for count-valued (u64) latents must yield a
// symmetric kernel, so single-site MH recovers the correct stationary law even
// near the 0 boundary. Target Poisson(1) (no likelihood): mean 1, P(k=0)=e^{-1}.
// Poisson(1) concentrates its mass at k∈{0,1,2}, so the boundary behavior is
// exercised heavily; a factor-2 boundary asymmetry (naive |x+δ|) would skew the
// recovered mass at 0.
#[test]
fn fg41_discrete_walk_recovers_poisson_at_boundary() {
    let model_fn = || sample(addr!("k"), Poisson::new(1.0).unwrap());
    let mut rng = StdRng::seed_from_u64(24680);
    let samples = adaptive_mcmc_chain(&mut rng, model_fn, 40_000, 5_000);

    let ks: Vec<u64> = samples
        .iter()
        .filter_map(|(_, t)| t.get_u64(&addr!("k")))
        .collect();
    let n = ks.len() as f64;
    let mean = ks.iter().map(|&k| k as f64).sum::<f64>() / n;
    let p0 = ks.iter().filter(|&&k| k == 0).count() as f64 / n;

    // Poisson(1): mean = 1, P(0) = e^{-1} = 0.367879 (scipy: poisson.pmf(0,1)).
    assert!(
        (mean - 1.0).abs() < 0.05,
        "Poisson(1) mean {mean:.4} != 1.0"
    );
    assert!(
        (p0 - 0.367879).abs() < 0.03,
        "Poisson(1) P(k=0) {p0:.4} != 0.3679 (boundary asymmetry?)"
    );
}
