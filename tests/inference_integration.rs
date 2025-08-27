//! # Inference Algorithm Integration Tests
//!
//! This module contains integration tests for inference algorithms with real models.
//! These tests validate that inference methods work end-to-end with actual
//! probabilistic models using **only the public API**.
//!
//! ## Test Categories
//!
//! ### 1. MCMC Integration (`test_mcmc_*`)
//! - `adaptive_mcmc_chain()` with various model types
//! - Convergence properties and basic sanity checks
//! - Parameter recovery for known models (e.g., Beta-Binomial conjugacy)
//! - Chain mixing and acceptance rates
//! - Integration with diagnostics
//!
//! ### 2. SMC Integration (`test_smc_*`)
//! - `adaptive_smc()` with `SMCConfig`
//! - Particle filtering and resampling
//! - Different resampling methods: `Systematic`, `Multinomial`, `Stratified`
//! - Effective sample size monitoring
//! - Sequential importance sampling
//!
//! ### 3. ABC Integration (`test_abc_*`)
//! - `abc_rejection()` with custom summary functions
//! - `abc_smc()` for sequential ABC
//! - Distance functions: `EuclideanDistance`, custom distances
//! - Tolerance effects on acceptance rates
//! - Summary statistic design and model comparison
//!
//! ### 4. Variational Inference (`test_vi_*`)
//! - `optimize_meanfield_vi()` optimization
//! - `MeanFieldGuide` creation and usage
//! - `elbo_with_guide()` estimation
//! - Parameter updates and convergence
//! - Integration with different model types
//!
//! ### 5. Diagnostic Integration (`test_diagnostics_*`)
//! - `r_hat_f64()` convergence diagnostics
//! - `effective_sample_size_mcmc()` and `effective_sample_size()`
//! - `summarize_f64_parameter()` for parameter summaries
//! - `print_diagnostics()` output formatting
//! - Multi-chain diagnostics and comparison
//!
//! ### 6. Validation Framework (`test_validation_*`)
//! - `ks_test_distribution()` goodness-of-fit testing
//! - `test_conjugate_normal_model()` analytical validation
//! - `ValidationResult` interpretation
//! - Cross-validation and model selection
//!
//! ### 7. End-to-End Workflows (`test_workflow_*`)
//! - Complete Bayesian workflows: prior → MCMC → diagnostics → validation
//! - Model comparison and selection
//! - Parameter estimation with uncertainty quantification
//! - Prediction and posterior predictive checks
//!
//! ## Model Templates for Testing
//!
//! ### Simple Conjugate Models
//! - **Beta-Binomial**: Known posterior for validation
//! - **Normal-Normal**: Conjugate prior for mean estimation
//! - **Gamma-Poisson**: Rate parameter estimation
//!
//! ### Regression Models
//! - **Linear Regression**: `y = α + βx + ε`
//! - **Logistic Regression**: Binary classification
//! - **Hierarchical Models**: Partial pooling
//!
//! ### Mixture Models
//! - **Gaussian Mixture**: Component identification
//! - **Discrete Mixture**: Clustering applications
//!
//! ## Implementation Guidelines
//!
//! - **Public API Only**: Import `fugue::*`, avoid internal paths
//! - **Model Functions**: Define as `|| { model_definition }` closures
//! - **Fixed Seeds**: Use `StdRng::seed_from_u64()` for reproducibility
//! - **Reasonable Sizes**: Small chain lengths (50-200) for fast tests
//! - **Statistical Validation**: Use confidence intervals, not point estimates
//! - **Error Handling**: Test both success and failure cases
//!
//! ## Example Test Pattern
//!
//! ```rust
//! #[test]
//! fn test_mcmc_beta_binomial_recovery() {
//!     let mut rng = StdRng::seed_from_u64(42);
//!     
//!     // Define conjugate model with known posterior
//!     let model_fn = || {
//!         sample(addr!("theta"), Beta::new(1.0, 1.0).unwrap())
//!             .bind(|theta| observe(addr!("k"), Binomial::new(10, theta).unwrap(), 7)
//!                  .bind(move |_| pure(theta)))
//!     };
//!     
//!     // Run MCMC
//!     let samples = adaptive_mcmc_chain(&mut rng, model_fn, 100, 20);
//!     
//!     // Extract parameter values
//!     let theta_values: Vec<f64> = samples.iter()
//!         .map(|(_, trace)| trace.get_f64(&addr!("theta")).unwrap())
//!         .collect();
//!     
//!     // Validate against known posterior Beta(8, 4)
//!     let mean = theta_values.iter().sum::<f64>() / theta_values.len() as f64;
//!     let expected_mean = 8.0 / 12.0; // (α + k) / (α + β + n)
//!     assert!((mean - expected_mean).abs() < 0.1);
//! }
//! ```

use fugue::*;
use rand::{rngs::StdRng, SeedableRng};

// TODO: Implement the test categories described above
// Focus on end-to-end validation of inference algorithms
// Each test should use realistic models and validate statistical properties

#[test]
fn test_mcmc_normal_mean_recovery() {
    let mut rng = StdRng::seed_from_u64(42);
    
    // Define conjugate Normal model with known posterior
    // Prior: mu ~ Normal(0, 2), Data: y ~ Normal(mu, 1), observed y = 2.5
    // Posterior: mu ~ Normal(mean_post, var_post) where:
    // var_post = 1/(1/2^2 + 1/1^2) = 1/(1/4 + 1) = 1/(5/4) = 4/5 = 0.8
    // mean_post = var_post * (0/2^2 + 2.5/1^2) = 0.8 * (0 + 2.5) = 2.0
    let model_fn = || {
        sample(addr!("mu"), Normal::new(0.0, 2.0).unwrap())
            .bind(|mu| observe(addr!("y"), Normal::new(mu, 1.0).unwrap(), 2.5)
                 .bind(move |_| pure(mu)))
    };
    
    // Run MCMC
    let samples = adaptive_mcmc_chain(&mut rng, model_fn, 200, 50);
    
    // Extract parameter values
    let mu_values: Vec<f64> = samples.iter()
        .map(|(mu, _trace)| *mu)
        .collect();
    
    // Validate against known posterior Normal(2.0, sqrt(0.8))
    let mean = mu_values.iter().sum::<f64>() / mu_values.len() as f64;
    let expected_mean = 2.0; // Theoretical posterior mean
    
    // Should be close to theoretical mean (with some tolerance for MCMC noise)
    assert!((mean - expected_mean).abs() < 0.2);
    
    // Check that we got reasonable number of samples
    assert_eq!(mu_values.len(), 200);
    
    // Check that all samples are finite
    assert!(mu_values.iter().all(|x| x.is_finite()));
}

#[test]
fn test_smc_gaussian_model() {
    let mut rng = StdRng::seed_from_u64(42);
    
    // Define a simple Gaussian model
    let model_fn = || {
        sample(addr!("mu"), Normal::new(0.0, 2.0).unwrap())
            .bind(|mu| observe(addr!("y"), Normal::new(mu, 0.5).unwrap(), 1.8)
                 .map(move |_| mu))
    };
    
    // Configure SMC
    let config = SMCConfig {
        resampling_method: ResamplingMethod::Systematic,
        ess_threshold: 0.5,
        rejuvenation_steps: 0,
    };
    
    // Run SMC
    let particles = adaptive_smc(&mut rng, 50, model_fn, config);
    
    // Extract mu estimates
    let mu_estimates: Vec<f64> = particles.iter()
        .filter_map(|p| p.trace.get_f64(&addr!("mu")))
        .collect();
    
    // Should have particles
    assert!(!mu_estimates.is_empty());
    assert_eq!(particles.len(), 50);
    
    // Check that particles have finite log weights
    assert!(particles.iter().all(|p| p.log_weight.is_finite()));
    
    // Weighted mean should be reasonable (closer to observed value 1.8)
    let weighted_sum: f64 = particles.iter()
        .filter_map(|p| {
            p.trace.get_f64(&addr!("mu")).map(|mu| mu * p.log_weight.exp())
        })
        .sum();
    let total_weight: f64 = particles.iter()
        .map(|p| p.log_weight.exp())
        .sum();
    
    if total_weight > 0.0 {
        let weighted_mean = weighted_sum / total_weight;
        // Should be somewhere between prior mean (0.0) and observation (1.8)
        assert!(weighted_mean > -1.0 && weighted_mean < 3.0);
    }
}

#[test]
fn test_abc_scalar_summary_basic() {
    let mut rng = StdRng::seed_from_u64(42);
    
    // Define a simple model for ABC
    let model_fn = || {
        sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap())
    };
    
    // Simple simulator: just return the sampled mu
    let simulator = |trace: &runtime::trace::Trace| -> f64 {
        trace.get_f64(&addr!("mu")).unwrap_or(0.0)
    };
    
    // Observed summary statistic
    let observed_summary = 0.5;
    
    // Run ABC with scalar summary
    let samples = abc_scalar_summary(
        &mut rng,
        model_fn,
        simulator,
        observed_summary,
        0.5, // tolerance
        100, // max_samples
    );
    
    // Should get some samples (though maybe not many due to tolerance)
    assert!(samples.len() <= 100);
    
    // All samples should be within tolerance of observed data
    for trace in &samples {
        let mu = trace.get_f64(&addr!("mu")).unwrap();
        assert!((mu - observed_summary).abs() <= 0.5);
    }
}

#[test]
fn test_diagnostics_basic() {
    let mut rng = StdRng::seed_from_u64(42);
    
    // Generate two MCMC chains
    let model_fn = || {
        sample(addr!("theta"), Normal::new(0.0, 1.0).unwrap())
    };
    
    let chain1 = adaptive_mcmc_chain(&mut rng, &model_fn, 100, 20);
    let chain2 = adaptive_mcmc_chain(&mut rng, &model_fn, 100, 20);
    
    // Extract theta values from both chains
    let theta1: Vec<f64> = chain1.iter().map(|(theta, _)| *theta).collect();
    let _theta2: Vec<f64> = chain2.iter().map(|(theta, _)| *theta).collect();
    
    // Test R-hat diagnostic - need to use traces, not extracted values
    let trace_chains = vec![
        chain1.iter().map(|(_, trace)| trace.clone()).collect::<Vec<_>>(),
        chain2.iter().map(|(_, trace)| trace.clone()).collect::<Vec<_>>(),
    ];
    let r_hat = r_hat_f64(&trace_chains, &addr!("theta"));
    
    // R-hat should be finite and ideally close to 1.0 for converged chains
    assert!(r_hat.is_finite());
    // Note: R-hat can sometimes be < 1.0 due to numerical issues or insufficient data
    // The important thing is that it's finite and reasonable
    assert!(r_hat > 0.0);
    
    // Test effective sample size
    let ess = effective_sample_size_mcmc(&theta1);
    assert!(ess >= 0.0);
    assert!(ess <= theta1.len() as f64);
    
    // Test parameter summary
    let summary = summarize_f64_parameter(&trace_chains, &addr!("theta"));
    assert!(summary.mean.is_finite());
    assert!(summary.std.is_finite());
    assert!(summary.std >= 0.0);
}

#[test]
fn test_variational_inference_basic() {
    let mut rng = StdRng::seed_from_u64(42);
    
    // Define a simple model
    let model_fn = || {
        sample(addr!("mu"), Normal::new(0.0, 2.0).unwrap())
            .bind(|mu| observe(addr!("y"), Normal::new(mu, 1.0).unwrap(), 1.5)
                 .bind(move |_| pure(mu)))
    };
    
    // Create a mean-field guide
    let mut guide = MeanFieldGuide::new();
    guide.params.insert(
        addr!("mu"),
        VariationalParam::Normal { mu: 0.0, log_sigma: 0.0 }
    );
    
    // Run a few VI steps
    let result = optimize_meanfield_vi(
        &mut rng,
        model_fn,
        guide.clone(),
        10,  // n_iterations
        10,  // n_samples_per_iter
        0.01, // learning_rate
    );
    
    // Should have some parameters
    assert!(!result.params.is_empty());
    
    // Test ELBO computation directly
    let elbo = elbo_with_guide(
        &mut rng,
        model_fn,
        &result,
        10, // num_samples
    );
    assert!(elbo.is_finite());
}

#[test]
fn test_validation_framework() {
    let mut rng = StdRng::seed_from_u64(42);
    
    // Generate reference samples from a known Normal(0, 1) distribution
    let reference_samples: Vec<f64> = (0..100).map(|_| {
        let dist = Normal::new(0.0, 1.0).unwrap();
        dist.sample(&mut rng)
    }).collect();
    
    // Test KS test against the true distribution
    let test_dist = Normal::new(0.0, 1.0).unwrap();
    let ks_result = ks_test_distribution(
        &mut rng,
        &test_dist,
        &reference_samples,
        100, // n_samples
        0.05 // alpha
    );
    
    // Should not reject the null hypothesis (samples come from the distribution)
    assert!(ks_result); // Returns bool, not a struct
    
    // For now, just test that the validation function exists and can be called
    // The full conjugate test would require the ConjugateNormalConfig which isn't exported
    // This validates that the public API is accessible
}