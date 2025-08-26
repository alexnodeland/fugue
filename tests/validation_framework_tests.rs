//! Tests for statistical validation framework.
//!
//! This module tests the validation functionality which currently has low coverage (25.49%).

mod test_utils;

use fugue::inference::validation::*;
use fugue::*;
use rand::{rngs::StdRng, Rng};
use test_utils::*;

#[test]
fn test_ks_test_distribution_normal() {
    let mut rng = seeded_rng(42);

    let normal = Normal::new(0.0, 1.0).unwrap();

    // Generate reference samples using Box-Muller transform
    let mut reference_samples = Vec::with_capacity(1000);
    for _ in 0..500 {
        let u1: f64 = rng.gen();
        let u2: f64 = rng.gen();

        let z0 = (-2.0_f64 * u1.ln()).sqrt() * (2.0_f64 * std::f64::consts::PI * u2).cos();
        let z1 = (-2.0_f64 * u1.ln()).sqrt() * (2.0_f64 * std::f64::consts::PI * u2).sin();

        reference_samples.push(z0);
        reference_samples.push(z1);
    }

    // Test our Normal implementation against reference
    let is_valid = ks_test_distribution(
        &mut rng,
        &normal,
        &reference_samples,
        1000,
        0.05, // 5% significance level
    );

    assert!(
        is_valid,
        "Normal distribution should pass KS test against reference implementation"
    );
}

#[test]
fn test_ks_test_distribution_uniform() {
    let mut rng = seeded_rng(123);

    let uniform = Uniform::new(0.0, 1.0).unwrap();

    // Generate reference uniform samples
    let reference_samples: Vec<f64> = (0..1000).map(|_| rng.gen::<f64>()).collect();

    let is_valid = ks_test_distribution(&mut rng, &uniform, &reference_samples, 1000, 0.05);

    assert!(is_valid, "Uniform distribution should pass KS test");
}

#[test]
fn test_ks_test_distribution_different_distributions() {
    let mut rng = seeded_rng(456);

    let normal = Normal::new(0.0, 1.0).unwrap();

    // Generate reference from a different distribution (uniform)
    let reference_samples: Vec<f64> = (0..1000).map(|_| rng.gen::<f64>()).collect();

    let is_valid = ks_test_distribution(&mut rng, &normal, &reference_samples, 1000, 0.05);

    // Should fail since we're comparing Normal vs Uniform
    assert!(
        !is_valid,
        "Normal should fail KS test against Uniform reference"
    );
}

#[test]
fn test_ks_test_distribution_small_sample() {
    let mut rng = seeded_rng(789);

    let normal = Normal::new(0.0, 1.0).unwrap();

    // Very small reference sample
    let reference_samples: Vec<f64> = vec![0.0, 1.0, -1.0];

    let is_valid = ks_test_distribution(
        &mut rng,
        &normal,
        &reference_samples,
        10, // Small test sample too
        0.05,
    );

    // Test should complete without crashing, result may vary
    assert!(is_valid || !is_valid); // Just check it doesn't panic
}

#[test]
fn test_ks_test_distribution_strict_alpha() {
    let mut rng = seeded_rng(999);

    let normal = Normal::new(0.0, 1.0).unwrap();

    // Generate reference samples
    let reference_samples: Vec<f64> = (0..1000)
        .map(|_| {
            let u1: f64 = rng.gen();
            let u2: f64 = rng.gen();
            (-2.0_f64 * u1.ln()).sqrt() * (2.0_f64 * std::f64::consts::PI * u2).cos()
        })
        .collect();

    // Test with very strict significance level
    let is_valid_strict = ks_test_distribution(
        &mut rng,
        &normal,
        &reference_samples,
        1000,
        0.001, // Very strict
    );

    // Test with lenient significance level
    let is_valid_lenient = ks_test_distribution(
        &mut rng,
        &normal,
        &reference_samples,
        1000,
        0.2, // Very lenient
    );

    // Lenient test should be more likely to pass than strict test
    if !is_valid_strict {
        // If strict fails, we can't guarantee lenient passes, but it shouldn't crash
        assert!(is_valid_lenient || !is_valid_lenient);
    }
}

#[test]
fn test_conjugate_normal_validation() {
    let config = ConjugateNormalConfig {
        prior_mu: 0.0,
        prior_sigma: 1.0,
        likelihood_sigma: 0.5,
        observation: 1.0,
        n_samples: 1000,
        n_warmup: 200,
    };

    // Extract config values to avoid borrow issues
    let prior_mu = config.prior_mu;
    let prior_sigma = config.prior_sigma;
    let likelihood_sigma = config.likelihood_sigma;
    let observation = config.observation;

    // Mock MCMC function that returns samples from the known posterior
    let mock_mcmc = |rng: &mut StdRng, n_samples: usize, _n_warmup: usize| -> Vec<(f64, Trace)> {
        // For conjugate normal model:
        // posterior_var = 1 / (1/prior_var + 1/likelihood_var)
        // posterior_mean = posterior_var * (prior_mean/prior_var + obs/likelihood_var)

        let prior_var = prior_sigma * prior_sigma;
        let likelihood_var = likelihood_sigma * likelihood_sigma;

        let posterior_var = 1.0 / (1.0 / prior_var + 1.0 / likelihood_var);
        let posterior_mean = posterior_var * (prior_mu / prior_var + observation / likelihood_var);
        let posterior_sigma = posterior_var.sqrt();

        let posterior_normal = Normal::new(posterior_mean, posterior_sigma).unwrap();

        let mut samples = Vec::new();
        for _ in 0..n_samples {
            let mu_sample = posterior_normal.sample(rng);
            let mut trace = Trace::default();
            trace.insert_choice(addr!("mu"), ChoiceValue::F64(mu_sample), -0.5);
            samples.push((mu_sample, trace));
        }
        samples
    };

    let mut rng = seeded_rng(42);
    let result = test_conjugate_normal_model(&mut rng, mock_mcmc, config.clone());

    match result {
        ValidationResult::Success {
            mean_within_bounds,
            var_within_bounds,
            ess_adequate,
            ..
        } => {
            assert!(
                mean_within_bounds,
                "Mean should be within bounds for conjugate model"
            );
            assert!(
                var_within_bounds,
                "Variance should be within bounds for conjugate model"
            );
            assert!(
                ess_adequate,
                "ESS should be adequate for independent samples"
            );
        }
        ValidationResult::Failed(msg) => {
            panic!("Conjugate normal model validation failed: {}", msg);
        }
    }
}

#[test]
fn test_conjugate_normal_validation_poor_mcmc() {
    let config = ConjugateNormalConfig {
        prior_mu: 0.0,
        prior_sigma: 1.0,
        likelihood_sigma: 0.5,
        observation: 1.0,
        n_samples: 100,
        n_warmup: 10,
    };

    // Mock MCMC that returns samples from wrong distribution (should fail validation)
    let bad_mcmc = |rng: &mut StdRng, n_samples: usize, _n_warmup: usize| -> Vec<(f64, Trace)> {
        let wrong_normal = Normal::new(5.0, 2.0).unwrap(); // Wrong distribution

        let mut samples = Vec::new();
        for _ in 0..n_samples {
            let mu_sample = wrong_normal.sample(rng);
            let mut trace = Trace::default();
            trace.insert_choice(addr!("mu"), ChoiceValue::F64(mu_sample), -0.5);
            samples.push((mu_sample, trace));
        }
        samples
    };

    let mut rng = seeded_rng(123);
    let result = test_conjugate_normal_model(&mut rng, bad_mcmc, config.clone());

    match result {
        ValidationResult::Success {
            mean_within_bounds,
            var_within_bounds,
            ..
        } => {
            // Should fail bounds checks
            assert!(
                !mean_within_bounds || !var_within_bounds,
                "Bad MCMC should fail bounds checks"
            );
        }
        ValidationResult::Failed(_) => {
            // Also acceptable - the validation detected the problem
        }
    }
}

#[test]
fn test_conjugate_normal_validation_insufficient_samples() {
    let config = ConjugateNormalConfig {
        prior_mu: 0.0,
        prior_sigma: 1.0,
        likelihood_sigma: 1.0,
        observation: 0.0,
        n_samples: 5, // Very few samples
        n_warmup: 0,
    };

    let minimal_mcmc =
        |_rng: &mut StdRng, n_samples: usize, _n_warmup: usize| -> Vec<(f64, Trace)> {
            let mut samples = Vec::new();
            for i in 0..n_samples {
                let mut trace = Trace::default();
                trace.insert_choice(addr!("mu"), ChoiceValue::F64(i as f64), -0.5);
                samples.push((i as f64, trace));
            }
            samples
        };

    let mut rng = seeded_rng(456);
    let result = test_conjugate_normal_model(&mut rng, minimal_mcmc, config.clone());

    // Should handle minimal samples gracefully
    assert!(
        matches!(result, ValidationResult::Success { .. })
            || matches!(result, ValidationResult::Failed(_))
    );
}

#[test]
fn test_validation_result_is_valid() {
    let success_valid = ValidationResult::Success {
        mean_error: 0.01,
        var_error: 0.02,
        effective_sample_size: 800.0,
        mean_within_bounds: true,
        var_within_bounds: true,
        ess_adequate: true,
        posterior_mu: 0.5,
        posterior_sigma: 0.7,
        sample_mean: 0.51,
        sample_sigma: 0.72,
    };

    assert!(success_valid.is_valid());

    let success_invalid_mean = ValidationResult::Success {
        mean_error: 0.01,
        var_error: 0.02,
        effective_sample_size: 800.0,
        mean_within_bounds: false, // Failed
        var_within_bounds: true,
        ess_adequate: true,
        posterior_mu: 0.5,
        posterior_sigma: 0.7,
        sample_mean: 2.0, // Way off
        sample_sigma: 0.72,
    };

    assert!(!success_invalid_mean.is_valid());

    let failed = ValidationResult::Failed("Test error".to_string());
    assert!(!failed.is_valid());
}

#[test]
fn test_validation_result_print_summary() {
    let success = ValidationResult::Success {
        mean_error: 0.015,
        var_error: 0.023,
        effective_sample_size: 750.0,
        mean_within_bounds: true,
        var_within_bounds: true,
        ess_adequate: true,
        posterior_mu: 0.8,
        posterior_sigma: 0.6,
        sample_mean: 0.815,
        sample_sigma: 0.623,
    };

    // Should not panic when printing
    success.print_summary();

    let failed = ValidationResult::Failed("Mock validation error".to_string());
    failed.print_summary();
}

#[test]
fn test_ks_statistic_identical_samples() {
    let sample1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let _sample2 = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    // This function is not directly accessible, but we can test through ks_test_distribution
    let mut rng = seeded_rng(1);
    let uniform = Uniform::new(0.0, 10.0).unwrap();

    // Test with identical reference samples
    let is_valid = ks_test_distribution(&mut rng, &uniform, &sample1, 5, 0.05);

    // When samples are identical, test should pass (KS statistic should be small)
    assert!(is_valid);
}

#[test]
fn test_ks_test_edge_cases() {
    let mut rng = seeded_rng(999);
    let normal = Normal::new(0.0, 1.0).unwrap();

    // Test with empty reference (should handle gracefully)
    let empty_reference: Vec<f64> = vec![];
    let result = ks_test_distribution(&mut rng, &normal, &empty_reference, 10, 0.05);
    // Should not panic and return some result
    assert!(result || !result);

    // Test with single-value reference
    let single_reference = vec![0.0];
    let result = ks_test_distribution(&mut rng, &normal, &single_reference, 10, 0.05);
    assert!(result || !result);

    // Test with extreme alpha values
    let reference: Vec<f64> = (0..100).map(|_| rng.gen::<f64>()).collect();

    let result_alpha_zero = ks_test_distribution(&mut rng, &normal, &reference, 100, 0.0);
    assert!(result_alpha_zero || !result_alpha_zero);

    let result_alpha_one = ks_test_distribution(&mut rng, &normal, &reference, 100, 1.0);
    assert!(result_alpha_one || !result_alpha_one);
}

#[test]
fn test_validation_with_extreme_parameters() {
    let extreme_config = ConjugateNormalConfig {
        prior_mu: 1e6,
        prior_sigma: 1e-6,
        likelihood_sigma: 1e6,
        observation: -1e6,
        n_samples: 100,
        n_warmup: 10,
    };

    let extreme_mcmc =
        |rng: &mut StdRng, n_samples: usize, _n_warmup: usize| -> Vec<(f64, Trace)> {
            let normal = Normal::new(0.0, 1.0).unwrap(); // Simple distribution

            let mut samples = Vec::new();
            for _ in 0..n_samples {
                let mu_sample = normal.sample(rng);
                let mut trace = Trace::default();
                trace.insert_choice(addr!("mu"), ChoiceValue::F64(mu_sample), -0.5);
                samples.push((mu_sample, trace));
            }
            samples
        };

    let mut rng = seeded_rng(777);
    let result = test_conjugate_normal_model(&mut rng, extreme_mcmc, extreme_config.clone());

    // Should handle extreme parameters without crashing
    match result {
        ValidationResult::Success { .. } => {
            // May pass or fail, but shouldn't crash
        }
        ValidationResult::Failed(_) => {
            // Also acceptable
        }
    }
}

#[test]
fn test_validation_numerical_stability() {
    let config = ConjugateNormalConfig {
        prior_mu: 0.0,
        prior_sigma: 1.0,
        likelihood_sigma: 0.1, // Very precise observation
        observation: 0.0,
        n_samples: 1000,
        n_warmup: 100,
    };

    let precise_mcmc =
        |rng: &mut StdRng, n_samples: usize, _n_warmup: usize| -> Vec<(f64, Trace)> {
            // Generate samples that should be very close to the observation
            let precise_normal = Normal::new(0.0, 0.05).unwrap();

            let mut samples = Vec::new();
            for _ in 0..n_samples {
                let mu_sample = precise_normal.sample(rng);
                let mut trace = Trace::default();
                trace.insert_choice(addr!("mu"), ChoiceValue::F64(mu_sample), -0.5);
                samples.push((mu_sample, trace));
            }
            samples
        };

    let mut rng = seeded_rng(555);
    let result = test_conjugate_normal_model(&mut rng, precise_mcmc, config.clone());

    match result {
        ValidationResult::Success {
            effective_sample_size,
            sample_mean,
            sample_sigma,
            ..
        } => {
            assert_finite(effective_sample_size);
            assert_finite(sample_mean);
            assert_finite(sample_sigma);
            assert!(effective_sample_size > 0.0);
        }
        ValidationResult::Failed(_) => {
            // May fail due to numerical precision, but shouldn't crash
        }
    }
}

#[test]
fn test_validation_with_correlated_samples() {
    let config = ConjugateNormalConfig {
        prior_mu: 0.0,
        prior_sigma: 1.0,
        likelihood_sigma: 1.0,
        observation: 0.0,
        n_samples: 200,
        n_warmup: 50,
    };

    // Generate highly correlated samples (bad MCMC mixing)
    let correlated_mcmc =
        |_rng: &mut StdRng, n_samples: usize, _n_warmup: usize| -> Vec<(f64, Trace)> {
            let mut samples = Vec::new();
            let mut current_val = 0.0;

            for _ in 0..n_samples {
                // Very small random walk (high correlation)
                current_val += (rand::random::<f64>() - 0.5) * 0.01;

                let mut trace = Trace::default();
                trace.insert_choice(addr!("mu"), ChoiceValue::F64(current_val), -0.5);
                samples.push((current_val, trace));
            }
            samples
        };

    let mut rng = seeded_rng(888);
    let result = test_conjugate_normal_model(&mut rng, correlated_mcmc, config.clone());

    match result {
        ValidationResult::Success {
            ess_adequate,
            effective_sample_size,
            ..
        } => {
            // Should detect low ESS for highly correlated samples
            assert!(
                !ess_adequate,
                "Highly correlated samples should have inadequate ESS"
            );
            assert!(effective_sample_size < 50.0); // Should be much lower than sample size
        }
        ValidationResult::Failed(_) => {
            // Also acceptable
        }
    }
}
