//! Statistical validation and testing utilities for inference algorithms.
//!
//! This module provides tools for validating the correctness of inference
//! implementations using known theoretical results and simulation studies.

use crate::addr;
use crate::core::distribution::*;

use crate::inference::mcmc_utils::effective_sample_size_mcmc;
use crate::runtime::trace::{ChoiceValue, Trace};
use rand::Rng;

/// Kolmogorov-Smirnov test for distribution correctness.
///
/// Tests whether samples from our distribution implementation match
/// the theoretical distribution using the two-sample KS test.
pub fn ks_test_distribution<R: Rng>(
    rng: &mut R,
    dist: &dyn Distribution<f64>,
    reference_samples: &[f64],
    n_samples: usize,
    alpha: f64,
) -> bool {
    // Generate samples from our implementation
    let mut our_samples = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        our_samples.push(dist.sample(rng));
    }

    // Sort both sample sets
    our_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut ref_sorted = reference_samples.to_vec();
    ref_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Compute KS statistic
    let ks_stat = ks_statistic(&our_samples, &ref_sorted);

    // Critical value for two-sample KS test
    let n1 = our_samples.len() as f64;
    let n2 = ref_sorted.len() as f64;
    let critical_value = (-0.5 * alpha.ln()).sqrt() * ((n1 + n2) / (n1 * n2)).sqrt();

    ks_stat < critical_value
}

/// Compute two-sample Kolmogorov-Smirnov statistic.
fn ks_statistic(sample1: &[f64], sample2: &[f64]) -> f64 {
    let n1 = sample1.len() as f64;
    let n2 = sample2.len() as f64;

    let mut max_diff: f64 = 0.0;
    let mut i1 = 0;
    let mut i2 = 0;

    while i1 < sample1.len() && i2 < sample2.len() {
        let cdf1 = (i1 + 1) as f64 / n1;
        let cdf2 = (i2 + 1) as f64 / n2;

        max_diff = max_diff.max((cdf1 - cdf2).abs());

        if sample1[i1] <= sample2[i2] {
            i1 += 1;
        } else {
            i2 += 1;
        }
    }

    max_diff
}

/// Configuration for conjugate normal model validation.
#[derive(Debug, Clone)]
pub struct ConjugateNormalConfig {
    /// Prior mean
    pub prior_mu: f64,
    /// Prior standard deviation
    pub prior_sigma: f64,
    /// Likelihood standard deviation
    pub likelihood_sigma: f64,
    /// Observed data point
    pub observation: f64,
    /// Number of MCMC samples
    pub n_samples: usize,
    /// Number of warmup/burn-in samples
    pub n_warmup: usize,
}

/// Test MCMC implementation against known analytical posterior.
///
/// For conjugate models where the posterior is known analytically,
/// this validates that MCMC produces the correct distribution.
pub fn test_conjugate_normal_model<R: Rng>(
    rng: &mut R,
    mcmc_fn: impl Fn(&mut R, usize, usize) -> Vec<(f64, Trace)>,
    config: ConjugateNormalConfig,
) -> ValidationResult {
    // Analytical posterior for normal-normal conjugate model
    let prior_precision = 1.0 / (config.prior_sigma * config.prior_sigma);
    let likelihood_precision = 1.0 / (config.likelihood_sigma * config.likelihood_sigma);

    let posterior_precision = prior_precision + likelihood_precision;
    let posterior_variance = 1.0 / posterior_precision;
    let posterior_mu = posterior_variance
        * (prior_precision * config.prior_mu + likelihood_precision * config.observation);

    let samples = mcmc_fn(rng, config.n_samples, config.n_warmup);
    validate_against_analytical_posterior(
        &samples,
        &addr!("mu"),
        posterior_mu,
        posterior_variance,
        config.n_samples,
    )
}

/// Configuration for conjugate Beta-Bernoulli model validation.
///
/// FG-15: complements [`ConjugateNormalConfig`] with the other textbook
/// conjugate pair (a bounded-support, non-symmetric posterior) so the
/// harness isn't validated on Normal-Normal alone.
#[derive(Debug, Clone)]
pub struct ConjugateBetaBernoulliConfig {
    /// Prior alpha (pseudo-count of prior successes).
    pub prior_alpha: f64,
    /// Prior beta (pseudo-count of prior failures).
    pub prior_beta: f64,
    /// Observed i.i.d. Bernoulli outcomes.
    pub observations: Vec<bool>,
    /// Number of MCMC samples.
    pub n_samples: usize,
    /// Number of warmup/burn-in samples.
    pub n_warmup: usize,
}

/// Test MCMC implementation against the known analytical Beta-Bernoulli
/// conjugate posterior.
///
/// For `theta ~ Beta(a, b)` and `n` i.i.d. `Bernoulli(theta)` observations
/// with `s` successes, the exact posterior is `Beta(a + s, b + n - s)`, with
/// mean `(a+s)/(a+b+n)` and variance `(a+s)(b+n-s) / ((a+b+n)^2 (a+b+n+1))`.
/// `mcmc_fn`'s model is expected to sample the success probability at
/// address `"theta"` (mirroring [`test_conjugate_normal_model`]'s use of
/// `"mu"`).
pub fn test_conjugate_beta_bernoulli_model<R: Rng>(
    rng: &mut R,
    mcmc_fn: impl Fn(&mut R, usize, usize) -> Vec<(f64, Trace)>,
    config: ConjugateBetaBernoulliConfig,
) -> ValidationResult {
    let successes = config.observations.iter().filter(|&&b| b).count() as f64;
    let n = config.observations.len() as f64;
    let post_alpha = config.prior_alpha + successes;
    let post_beta = config.prior_beta + (n - successes);
    let post_sum = post_alpha + post_beta;

    let posterior_mu = post_alpha / post_sum;
    let posterior_variance = (post_alpha * post_beta) / (post_sum * post_sum * (post_sum + 1.0));

    let samples = mcmc_fn(rng, config.n_samples, config.n_warmup);
    validate_against_analytical_posterior(
        &samples,
        &addr!("theta"),
        posterior_mu,
        posterior_variance,
        config.n_samples,
    )
}

/// Shared scoring logic for the conjugate-model validation harnesses:
/// extract the `f64` trace values at `address`, compare their sample
/// mean/variance to the supplied analytical posterior mean/variance within
/// 2 Monte Carlo standard errors (computed from the effective sample size),
/// and check the chain achieved at least 10% sampling efficiency.
fn validate_against_analytical_posterior(
    samples: &[(f64, Trace)],
    address: &crate::core::address::Address,
    posterior_mu: f64,
    posterior_variance: f64,
    n_samples: usize,
) -> ValidationResult {
    let posterior_sigma = posterior_variance.sqrt();

    let param_samples: Vec<f64> = samples
        .iter()
        .filter_map(|(_, trace)| trace.choices.get(address))
        .filter_map(|choice| match choice.value {
            ChoiceValue::F64(val) => Some(val),
            _ => None,
        })
        .collect();

    if param_samples.is_empty() {
        return ValidationResult::Failed("No samples extracted".to_string());
    }

    // Compute sample statistics
    let sample_mean = param_samples.iter().sum::<f64>() / param_samples.len() as f64;
    let sample_var = param_samples
        .iter()
        .map(|&x| (x - sample_mean).powi(2))
        .sum::<f64>()
        / (param_samples.len() - 1) as f64;
    let sample_sigma = sample_var.sqrt();

    // Compute effective sample size
    let ess = effective_sample_size_mcmc(&param_samples);

    // Check if estimates are within reasonable bounds (2 standard errors)
    let se_mean = posterior_sigma / (ess.sqrt());
    let se_var = posterior_variance * (2.0 / ess).sqrt();

    let mean_error = (sample_mean - posterior_mu).abs();
    let var_error = (sample_var - posterior_variance).abs();

    let mean_ok = mean_error < 2.0 * se_mean;
    let var_ok = var_error < 2.0 * se_var;
    let ess_ok = ess > n_samples as f64 * 0.1; // At least 10% efficiency

    ValidationResult::Success {
        mean_error,
        var_error,
        effective_sample_size: ess,
        mean_within_bounds: mean_ok,
        var_within_bounds: var_ok,
        ess_adequate: ess_ok,
        posterior_mu,
        posterior_sigma,
        sample_mean,
        sample_sigma,
    }
}

/// Result of statistical validation test.
#[derive(Debug)]
pub enum ValidationResult {
    Success {
        mean_error: f64,
        var_error: f64,
        effective_sample_size: f64,
        mean_within_bounds: bool,
        var_within_bounds: bool,
        ess_adequate: bool,
        posterior_mu: f64,
        posterior_sigma: f64,
        sample_mean: f64,
        sample_sigma: f64,
    },
    Failed(String),
}

impl ValidationResult {
    pub fn is_valid(&self) -> bool {
        match self {
            ValidationResult::Success {
                mean_within_bounds,
                var_within_bounds,
                ess_adequate,
                ..
            } => *mean_within_bounds && *var_within_bounds && *ess_adequate,
            ValidationResult::Failed(_) => false,
        }
    }

    pub fn print_summary(&self) {
        match self {
            ValidationResult::Success {
                mean_error,
                var_error,
                effective_sample_size,
                posterior_mu,
                posterior_sigma,
                sample_mean,
                sample_sigma,
                mean_within_bounds,
                var_within_bounds,
                ess_adequate,
            } => {
                println!("Validation Results:");
                println!(
                    "  True posterior: N({:.4}, {:.4})",
                    posterior_mu, posterior_sigma
                );
                println!(
                    "  Sample estimates: N({:.4}, {:.4})",
                    sample_mean, sample_sigma
                );
                println!(
                    "  Mean error: {:.6} ({})",
                    mean_error,
                    if *mean_within_bounds { "PASS" } else { "FAIL" }
                );
                println!(
                    "  Var error: {:.6} ({})",
                    var_error,
                    if *var_within_bounds { "PASS" } else { "FAIL" }
                );
                println!(
                    "  ESS: {:.1} ({})",
                    effective_sample_size,
                    if *ess_adequate { "PASS" } else { "FAIL" }
                );
                println!(
                    "  Overall: {}",
                    if self.is_valid() { "PASS" } else { "FAIL" }
                );
            }
            ValidationResult::Failed(msg) => {
                println!("Validation FAILED: {}", msg);
            }
        }
    }
}

#[cfg(test)]
mod tests_more {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn ks_test_edge_thresholds_and_print_summary() {
        let mut rng = StdRng::seed_from_u64(50);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let ref_samples: Vec<f64> = (0..200).map(|_| normal.sample(&mut rng)).collect();
        let ok = ks_test_distribution(&mut rng, &normal, &ref_samples, 200, 0.05);
        assert!(ok);

        // ValidationResult print_summary coverage
        let res = ValidationResult::Success {
            mean_error: 0.0,
            var_error: 0.0,
            effective_sample_size: 10.0,
            mean_within_bounds: true,
            var_within_bounds: true,
            ess_adequate: true,
            posterior_mu: 0.0,
            posterior_sigma: 1.0,
            sample_mean: 0.0,
            sample_sigma: 1.0,
        };
        res.print_summary();
        assert!(res.is_valid());
    }
}

#[cfg(test)]
mod validation_tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_normal_distribution() {
        let mut rng = StdRng::seed_from_u64(42);
        let normal = Normal::new(0.0, 1.0).unwrap();

        // Generate reference samples using a different method
        let reference: Vec<f64> = (0..1000)
            .map(|_| {
                let u1: f64 = rng.gen();
                let u2: f64 = rng.gen();
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
            })
            .collect();

        let is_valid = ks_test_distribution(&mut rng, &normal, &reference, 1000, 0.05);
        assert!(is_valid, "Normal distribution failed KS test");
    }
}
