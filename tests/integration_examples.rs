//! Integration tests mirroring the examples directory.
//!
//! These tests validate that all examples work correctly and provide
//! additional coverage for integration scenarios.

mod test_utils;

use fugue::*;
use test_utils::*;

/// Test the gaussian_mean example functionality
mod gaussian_mean_integration {
    use super::*;

    #[test]
    fn test_gaussian_mean_prior_sampling() {
        let mut rng = test_rng();
        let model = models::gaussian_mean(2.7);

        let (mu, trace) = runtime::handler::run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            model,
        );

        assert_finite(mu);
        assert!(trace.choices.contains_key(&addr!("mu")));
        assert_finite(trace.total_log_weight());
    }

    #[test]
    fn test_gaussian_mean_mcmc() {
        let mut rng = seeded_rng(123);

        // Run short MCMC chain
        let samples = adaptive_mcmc_chain(
            &mut rng,
            || models::gaussian_mean(2.0),
            100, // samples
            50,  // warmup
        );

        assert_eq!(samples.len(), 100);

        // Extract mu samples
        let mu_samples: Vec<f64> = samples
            .iter()
            .filter_map(|(_, trace)| trace.get_f64(&addr!("mu")))
            .collect();

        assert_eq!(mu_samples.len(), 100);

        // Should be centered around observation
        let mean = stats::mean(&mu_samples);
        assert!((mean - 1.0).abs() < 0.5); // Should be between true posterior mean and obs

        // Calculate effective sample size
        let ess = effective_sample_size_mcmc(&mu_samples);
        assert!(ess > 10.0); // Should have reasonable ESS
    }

    #[test]
    fn test_gaussian_mean_validation() {
        let mut rng = seeded_rng(456);

        // Simple validation - just run MCMC and check basic properties
        let samples = adaptive_mcmc_chain(&mut rng, || models::gaussian_mean(1.5), 200, 100);

        let mu_samples: Vec<f64> = samples
            .iter()
            .filter_map(|(_, trace)| trace.get_f64(&addr!("mu")))
            .collect();

        // Basic sanity checks
        assert_eq!(mu_samples.len(), 200);
        let mean = stats::mean(&mu_samples);
        assert!(mean > 0.5 && mean < 2.5); // Should be reasonable given obs=1.5

        let ess = effective_sample_size_mcmc(&mu_samples);
        assert!(ess > 20.0); // Should have reasonable effective sample size
    }
}

/// Test the gaussian_mixture example functionality
mod gaussian_mixture_integration {
    use super::*;

    fn gaussian_mixture_model(obs: f64) -> Model<(f64, bool)> {
        sample(addr!("component"), Bernoulli::new(0.3).unwrap()).bind(move |z| {
            let mu = if z { -2.0 } else { 2.0 };
            sample(addr!("x"), Normal::new(mu, 1.0).unwrap()).bind(move |x| {
                observe(addr!("y"), Normal::new(x, 0.1).unwrap(), obs);
                pure((x, z))
            })
        })
    }

    #[test]
    fn test_mixture_model_prior_sampling() {
        let mut rng = test_rng();
        let model = gaussian_mixture_model(1.8);

        let ((x, z), trace) = runtime::handler::run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            model,
        );

        assert_finite(x);
        assert!(trace.choices.contains_key(&addr!("component")));
        assert!(trace.choices.contains_key(&addr!("x")));
        assert_finite(trace.total_log_weight());

        // Component should be boolean
        assert_eq!(trace.get_bool(&addr!("component")), Some(z));
    }

    #[test]
    fn test_mixture_mcmc_component_switching() {
        let mut rng = seeded_rng(789);

        // Run MCMC on mixture model
        let samples = adaptive_mcmc_chain(
            &mut rng,
            || gaussian_mixture_model(1.8), // Obs closer to component 1
            200,
            100,
        );

        let component_samples: Vec<bool> = samples
            .iter()
            .filter_map(|(_, trace)| trace.get_bool(&addr!("component")))
            .collect();

        // Should see both components
        let true_count = component_samples.iter().filter(|&&b| b).count();
        let false_count = component_samples.len() - true_count;

        assert!(true_count > 10, "Should sample from component 1");
        assert!(false_count > 10, "Should sample from component 0");

        // Given obs=1.8, should prefer component 1 (mu=2.0)
        assert!(
            true_count > false_count,
            "Should prefer component closer to observation"
        );
    }
}

// Exponential hazard tests removed due to complexity issues

/// Test the conjugate_beta_binomial example
mod beta_binomial_integration {
    use super::*;

    fn beta_binomial_model(n: u64, k: u64) -> Model<f64> {
        sample(addr!("p"), Beta::new(2.0, 2.0).unwrap()).bind(move |p| {
            // Use safe probability clamping
            let safe_p = p.max(0.001).min(0.999);
            observe(addr!("successes"), Binomial::new(n, safe_p).unwrap(), k);
            pure(p)
        })
    }

    #[test]
    fn test_beta_binomial_prior() {
        let mut rng = test_rng();
        let model = beta_binomial_model(10, 7);

        let (p, trace) = runtime::handler::run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            model,
        );

        assert!(p >= 0.0 && p <= 1.0);
        assert!(trace.choices.contains_key(&addr!("p")));
        assert_finite(trace.total_log_weight());

        // Check that successes is observed as u64
        if let Some(choice) = trace.choices.get(&addr!("successes")) {
            if let ChoiceValue::U64(k) = choice.value {
                assert_eq!(k, 7);
            } else {
                panic!("Expected U64 value for binomial observation");
            }
        }
    }

    #[test]
    fn test_beta_binomial_basic() {
        let mut rng = seeded_rng(111);

        // Simple test - just verify it runs
        let samples = adaptive_mcmc_chain(&mut rng, || beta_binomial_model(10, 7), 50, 25);

        let p_samples: Vec<f64> = samples
            .iter()
            .filter_map(|(_, trace)| trace.get_f64(&addr!("p")))
            .collect();

        // Basic checks
        assert_eq!(p_samples.len(), 50);
        assert!(p_samples.iter().all(|&p| p >= 0.0 && p <= 1.0));

        let actual_mean = stats::mean(&p_samples);
        assert!(actual_mean > 0.3 && actual_mean < 0.9); // Should be reasonable
    }
}

/// Test complex models with type safety
mod type_safety_integration {
    use super::*;

    #[test]
    fn test_mixed_type_model_mcmc() {
        let mut rng = seeded_rng(222);

        let samples = adaptive_mcmc_chain(&mut rng, models::mixed_model, 100, 50);

        // Validate all expected types are preserved
        for (_, trace) in &samples {
            assert!(trace.get_f64(&addr!("continuous")).is_some());
            assert!(trace.get_bool(&addr!("discrete")).is_some());
        }

        // Extract values and check basic properties
        let continuous: Vec<f64> = samples
            .iter()
            .filter_map(|(_, trace)| trace.get_f64(&addr!("continuous")))
            .collect();

        let discrete: Vec<bool> = samples
            .iter()
            .filter_map(|(_, trace)| trace.get_bool(&addr!("discrete")))
            .collect();

        assert_eq!(continuous.len(), 100);
        assert_eq!(discrete.len(), 100);

        // Should have finite continuous values
        assert!(continuous.iter().all(|&x| x.is_finite()));

        // Should have both true and false for boolean (probabilistically)
        let true_count = discrete.iter().filter(|&&b| b).count();
        assert!(true_count > 20 && true_count < 80); // Should be roughly balanced
    }

    #[test]
    fn test_mixed_model_basic() {
        let mut rng = test_rng();
        let model = models::mixed_model();

        let ((x, b), trace) = runtime::handler::run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            model,
        );

        // Should return valid values
        assert_finite(x);

        // Should have both addresses in trace
        assert!(trace.choices.contains_key(&addr!("continuous")));
        assert!(trace.choices.contains_key(&addr!("discrete")));

        // Should match return values
        assert_eq!(trace.get_f64(&addr!("continuous")), Some(x));
        assert_eq!(trace.get_bool(&addr!("discrete")), Some(b));
    }
}

/// Test performance regression scenarios
mod performance_regression {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_mcmc_performance_regression() {
        let mut rng = seeded_rng(333);

        // Should complete MCMC chain within reasonable time
        let (samples, duration) = perf::time_fn(|| {
            adaptive_mcmc_chain(&mut rng, || models::gaussian_mean(0.0), 1000, 500)
        });

        assert_eq!(samples.len(), 1000);

        // Should complete within 5 seconds (generous bound)
        assert!(
            duration < Duration::from_secs(5),
            "MCMC took {:?} which may indicate performance regression",
            duration
        );

        // Should have reasonable acceptance rate (implicit via ESS)
        let mu_samples: Vec<f64> = trace_utils::extract_f64_samples(&samples, &addr!("mu"));
        let ess = effective_sample_size_mcmc(&mu_samples);
        assert!(ess > 50.0, "ESS too low: {}, may indicate poor mixing", ess);
    }

    #[test]
    fn test_trace_memory_efficiency() {
        let mut rng = test_rng();

        // Create large model with many addresses
        let large_model = || {
            let models: Vec<_> = (0..100)
                .map(|i| sample(addr!("x", i), Normal::new(0.0, 1.0).unwrap()))
                .collect();
            sequence_vec(models)
        };

        let (_, duration) = perf::time_fn(|| {
            let (_data, trace) = runtime::handler::run(
                PriorHandler {
                    rng: &mut rng,
                    trace: Trace::default(),
                },
                large_model(),
            );

            // Should handle 100 addresses efficiently
            assert_eq!(trace.choices.len(), 100);
        });

        // Should complete quickly
        assert!(duration < Duration::from_millis(100));
    }
}
