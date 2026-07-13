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

/// Exact categorical posterior P(z=k) ∝ prior[k]·N(y; means[k], sigma) for a
/// mixture-indicator model, normalized. The `1/(sigma·√2π)` factor cancels.
fn exact_categorical_posterior(prior: &[f64], means: &[f64], y: f64, sigma: f64) -> Vec<f64> {
    let w: Vec<f64> = prior
        .iter()
        .zip(means)
        .map(|(&p, &m)| p * (-0.5 * ((y - m) / sigma).powi(2)).exp())
        .collect();
    let z: f64 = w.iter().sum();
    w.iter().map(|&wi| wi / z).collect()
}

/// Run the mixture-indicator chain and return the empirical P(z=k) over `k=0..K`.
fn run_categorical_chain(
    prior: Vec<f64>,
    means: Vec<f64>,
    y: f64,
    sigma: f64,
    n_samples: usize,
    n_warmup: usize,
    seed: u64,
) -> Vec<f64> {
    let k = prior.len();
    let model_fn = move || {
        let means = means.clone();
        sample(addr!("z"), Categorical::new(prior.clone()).unwrap()).and_then(move |z| {
            // Guard out-of-support proposals with a -inf factor instead of indexing
            // `means[z]`. The prior only ever draws z ∈ [0, K), so this branch is
            // never taken by the (fixed) prior-resample proposal; it exists solely
            // so the PRE-FIX asymmetric proposal — which could draw z ≥ K — is
            // cleanly *rejected* (exposing the stationary bias) rather than
            // panicking on an out-of-bounds index.
            if z < means.len() {
                observe(addr!("y"), Normal::new(means[z], sigma).unwrap(), y).map(move |_| z)
            } else {
                factor(f64::NEG_INFINITY).map(move |_| z)
            }
        })
    };
    let mut rng = StdRng::seed_from_u64(seed);
    let samples = adaptive_mcmc_chain(&mut rng, model_fn, n_samples, n_warmup);
    let mut counts = vec![0usize; k];
    for (_z, t) in &samples {
        if let Some(z) = t.get_usize(&addr!("z")) {
            counts[z] += 1;
        }
    }
    let n = samples.len() as f64;
    counts.iter().map(|&c| c as f64 / n).collect()
}

// FG-10: categorical/usize sites are proposed by resampling from the site's
// PRIOR (an independence proposal), so acceptance reduces to the likelihood ratio
// with NO Hastings correction and every category is directly reachable. The
// pre-fix `UniformCategoricalProposal { n_categories: None }` instead drew from
// `[0, max(current+5, 10))` — a current-DEPENDENT range whose asymmetry was left
// uncorrected, biasing the stationary law and systematically over-weighting high
// indices.
//
// This regression uses K=8 categories (the pre-fix K=3 test could not expose the
// bug — its whole support sits inside the flat `max_val=10` window, so the
// asymmetry barely moved the three probabilities). Uniform prior, means 0..7,
// observation y=6, sigma=1.5, so the true posterior places ~0.57 of its mass on
// the high indices k∈{6,7} where the pre-fix bias is largest.
//
// Exact posterior (uniform prior cancels; weights ∝ exp(-0.5·((6-k)/1.5)^2),
// normalized — reproducible in scipy):
//   [0.000105, 0.001215, 0.008981, 0.042549, 0.129253, 0.251750, 0.314397, 0.251750]
// The audit measured the pre-fix chain's L1 error ≈ 0.05 with per-category error
// up to ~0.03 on k=5/7; our simulation of the exact pre-fix kernel reproduces
// L1 ≈ 0.07 (per-cat 0.029 at k=7), so the tolerances below fail on the pre-fix
// code and pass on the prior-resample fix (whose MC error here is < 0.006/cat).
#[test]
fn fg10_categorical_prior_resample_recovers_posterior() {
    let prior = vec![1.0 / 8.0; 8];
    let means: Vec<f64> = (0..8).map(|k| k as f64).collect();
    let expected = exact_categorical_posterior(&prior, &means, 6.0, 1.5);
    assert!((expected[6] + expected[7] - 0.5661).abs() < 1e-3);

    let emp = run_categorical_chain(prior, means, 6.0, 1.5, 80_000, 8_000, 555);

    let l1: f64 = emp.iter().zip(&expected).map(|(a, b)| (a - b).abs()).sum();
    // Aggregate L1: fixed ≈ 0.012, pre-fix ≈ 0.07.
    assert!(
        l1 < 0.03,
        "categorical posterior L1 error {l1:.4} too large"
    );
    for k in 0..8 {
        // Per-category: fixed < 0.006, pre-fix up to 0.029 (k=5,7).
        assert!(
            (emp[k] - expected[k]).abs() < 0.02,
            "P(z={k}) = {:.4}, expected {:.4} (L1={l1:.4})",
            emp[k],
            expected[k]
        );
    }
}

// FG-10 (reachability of top categories): with K=12 > the pre-fix heuristic
// ceiling `max(current+5, 10)`, the old proposal could reach the top categories
// only by slowly climbing through intermediate ones, so it grossly under- or
// mis-weighted them; the audit flagged category k=11 as biased by ~0.02. The
// prior-resample proposal draws directly from `[0, K)`, so every category —
// including the last — is proposed in one step. Uniform prior over 12 categories,
// means 0..11, y=10, sigma=1.5; the true posterior puts ~0.57 of its mass on
// k∈{10,11}.
//
// Exact posterior (last four entries): k=8:0.129252, k=9:0.251748,
// k=10:0.314395, k=11:0.251748.
#[test]
fn fg10_categorical_top_categories_reachable_for_large_k() {
    let prior = vec![1.0 / 12.0; 12];
    let means: Vec<f64> = (0..12).map(|k| k as f64).collect();
    let expected = exact_categorical_posterior(&prior, &means, 10.0, 1.5);

    let emp = run_categorical_chain(prior, means, 10.0, 1.5, 80_000, 8_000, 20260711);

    // The top category (index 11, above the pre-fix flat window) must be recovered
    // to within Monte Carlo error. Fixed ≈ 0.006; the pre-fix bias here is ≈ 0.02.
    assert!(
        (emp[11] - expected[11]).abs() < 0.015,
        "P(z=11) = {:.4}, expected {:.4}: top category not properly reachable",
        emp[11],
        expected[11]
    );
    // And the whole high-index tail is unbiased.
    let l1: f64 = emp.iter().zip(&expected).map(|(a, b)| (a - b).abs()).sum();
    assert!(
        l1 < 0.03,
        "K=12 categorical posterior L1 error {l1:.4} too large"
    );
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
