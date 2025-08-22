//! Comprehensive tests for type-safe Metropolis-Hastings implementation.
//!
//! These tests ensure that the type-safe MH implementation preserves types
//! correctly and that all ProposalStrategy traits work as expected.

use fugue::*;
use fugue::inference::mh::{
    adaptive_mcmc_chain, adaptive_single_site_mh, GaussianWalkProposal, FlipProposal, 
    DiscreteWalkProposal, UniformCategoricalProposal, ProposalStrategy
};
use fugue::inference::mcmc_utils::DiminishingAdaptation;
use fugue::runtime::trace::{ChoiceValue, Trace};
use rand::{rngs::StdRng, SeedableRng, Rng};

#[test]
fn test_gaussian_walk_proposal_strategy() {
    let mut rng = StdRng::seed_from_u64(42);
    let strategy = GaussianWalkProposal;
    
    let current = 1.0;
    let scale = 0.5;
    
    // Test multiple proposals
    for _ in 0..100 {
        let proposed = strategy.propose(current, scale, &mut rng);
        assert!(proposed.is_finite());
        // Proposals should be in a reasonable range around current value
        assert!((proposed - current).abs() < 10.0 * scale);
    }
    
    // Test log proposal probability (should be 0 for symmetric proposal)
    let log_prob = strategy.log_proposal_prob(1.0, 2.0, 0.5);
    assert_eq!(log_prob, 0.0);
}

#[test]
fn test_flip_proposal_strategy() {
    let mut rng = StdRng::seed_from_u64(42);
    let strategy = FlipProposal;
    
    let mut flip_count = 0;
    let n_trials = 1000;
    
    // Test multiple proposals
    for _ in 0..n_trials {
        let current = rng.gen::<bool>();
        let proposed = strategy.propose(current, 1.0, &mut rng);
        
        if proposed != current {
            flip_count += 1;
        }
    }
    
    // Should flip about 50% of the time
    let flip_rate = flip_count as f64 / n_trials as f64;
    assert!(flip_rate > 0.3 && flip_rate < 0.7);
    
    // Test log proposal probability (should be 0 for symmetric proposal)
    let log_prob = strategy.log_proposal_prob(true, false, 1.0);
    assert_eq!(log_prob, 0.0);
}

#[test]
fn test_discrete_walk_proposal_strategy() {
    let mut rng = StdRng::seed_from_u64(42);
    let strategy = DiscreteWalkProposal;
    
    let current = 10u64;
    let scale = 2.0;
    
    // Test multiple proposals
    for _ in 0..100 {
        let proposed = strategy.propose(current, scale, &mut rng);
        // Should be non-negative
        assert!(proposed < u64::MAX);
        // Should be in reasonable range
        let diff = (proposed as i64 - current as i64).abs();
        assert!(diff < 20); // Reasonable bound for scale=2
    }
}

#[test]
fn test_uniform_categorical_proposal_strategy() {
    let mut rng = StdRng::seed_from_u64(42);
    
    // Test with known number of categories
    let strategy = UniformCategoricalProposal { n_categories: Some(5) };
    
    for _ in 0..100 {
        let current = 2usize;
        let proposed = strategy.propose(current, 1.0, &mut rng);
        assert!(proposed < 5); // Should be within bounds
    }
    
    // Test with unknown number of categories (uses heuristic)
    let strategy_unknown = UniformCategoricalProposal { n_categories: None };
    
    for _ in 0..100 {
        let current = 3usize;
        let proposed = strategy_unknown.propose(current, 1.0, &mut rng);
        // Should use heuristic: max((current + 5), 10)
        assert!(proposed < (current + 5).max(10));
    }
}

#[test]
fn test_type_safe_mh_preserves_f64_type() {
    let mut rng = StdRng::seed_from_u64(42);
    
    // Create a model with f64 parameter
    let model_fn = || {
        sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap())
            .map(|mu| mu)
    };
    
    // Initialize with prior sample
    let (_, mut current_trace) = runtime::handler::run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        model_fn(),
    );
    
    // Verify initial type
    let initial_choice = &current_trace.choices[&addr!("mu")];
    assert!(matches!(initial_choice.value, ChoiceValue::F64(_)));
    
    let mut adaptation = DiminishingAdaptation::new(0.44, 0.7);
    
    // Run several MH steps
    for _ in 0..10 {
        let (_, new_trace) = adaptive_single_site_mh(
            &mut rng,
            &model_fn,
            &current_trace,
            &mut adaptation,
        );
        
        // Verify type is preserved
        let choice = &new_trace.choices[&addr!("mu")];
        assert!(matches!(choice.value, ChoiceValue::F64(_)));
        
        // Verify it's a reasonable value
        if let ChoiceValue::F64(val) = choice.value {
            assert!(val.is_finite());
            assert!(val > -10.0 && val < 10.0); // Reasonable range
        }
        
        current_trace = new_trace;
    }
}

#[test]
fn test_type_safe_mh_preserves_bool_type() {
    let mut rng = StdRng::seed_from_u64(42);
    
    // Create a model with bool parameter
    let model_fn = || {
        sample(addr!("coin"), Bernoulli::new(0.7).unwrap())
            .map(|coin| coin)
    };
    
    // Initialize with prior sample
    let (_, mut current_trace) = runtime::handler::run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        model_fn(),
    );
    
    // Verify initial type
    let initial_choice = &current_trace.choices[&addr!("coin")];
    assert!(matches!(initial_choice.value, ChoiceValue::Bool(_)));
    
    let mut adaptation = DiminishingAdaptation::new(0.44, 0.7);
    
    // Run several MH steps
    for _ in 0..10 {
        let (_, new_trace) = adaptive_single_site_mh(
            &mut rng,
            &model_fn,
            &current_trace,
            &mut adaptation,
        );
        
        // Verify type is preserved
        let choice = &new_trace.choices[&addr!("coin")];
        assert!(matches!(choice.value, ChoiceValue::Bool(_)));
        
        current_trace = new_trace;
    }
}

#[test]
fn test_type_safe_mh_preserves_u64_type() {
    let mut rng = StdRng::seed_from_u64(42);
    
    // Create a model with u64 parameter
    let model_fn = || {
        sample(addr!("count"), Poisson::new(3.0).unwrap())
            .map(|count| count)
    };
    
    // Initialize with prior sample
    let (_, mut current_trace) = runtime::handler::run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        model_fn(),
    );
    
    // Verify initial type
    let initial_choice = &current_trace.choices[&addr!("count")];
    assert!(matches!(initial_choice.value, ChoiceValue::U64(_)));
    
    let mut adaptation = DiminishingAdaptation::new(0.44, 0.7);
    
    // Run several MH steps
    for _ in 0..10 {
        let (_, new_trace) = adaptive_single_site_mh(
            &mut rng,
            &model_fn,
            &current_trace,
            &mut adaptation,
        );
        
        // Verify type is preserved
        let choice = &new_trace.choices[&addr!("count")];
        assert!(matches!(choice.value, ChoiceValue::U64(_)));
        
        // Verify it's a reasonable value
        if let ChoiceValue::U64(val) = choice.value {
            assert!(val < 1000); // Reasonable upper bound for Poisson(3)
        }
        
        current_trace = new_trace;
    }
}

#[test]
fn test_type_safe_mh_preserves_usize_type() {
    let mut rng = StdRng::seed_from_u64(42);
    
    // Create a model with usize parameter
    let model_fn = || {
        sample(addr!("choice"), Categorical::new(vec![0.2, 0.3, 0.5]).unwrap())
            .map(|choice| choice)
    };
    
    // Initialize with prior sample
    let (_, mut current_trace) = runtime::handler::run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        model_fn(),
    );
    
    // Verify initial type
    let initial_choice = &current_trace.choices[&addr!("choice")];
    assert!(matches!(initial_choice.value, ChoiceValue::Usize(_)));
    
    let mut adaptation = DiminishingAdaptation::new(0.44, 0.7);
    
    // Run several MH steps
    for _ in 0..10 {
        let (_, new_trace) = adaptive_single_site_mh(
            &mut rng,
            &model_fn,
            &current_trace,
            &mut adaptation,
        );
        
        // Verify type is preserved
        let choice = &new_trace.choices[&addr!("choice")];
        assert!(matches!(choice.value, ChoiceValue::Usize(_)));
        
        // Verify it's a valid choice
        if let ChoiceValue::Usize(val) = choice.value {
            assert!(val < 20); // Should be reasonable given categorical heuristic
        }
        
        current_trace = new_trace;
    }
}

#[test]
fn test_mixed_type_model_mcmc() {
    let mut rng = StdRng::seed_from_u64(42);
    
    // Create a model with multiple parameter types
    let model_fn = || {
        prob! {
            let mu <- sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap());        // f64
            let coin <- sample(addr!("coin"), Bernoulli::new(0.6).unwrap());      // bool
            let count <- sample(addr!("count"), Poisson::new(2.0).unwrap());     // u64
            let choice <- sample(addr!("choice"), Categorical::new(vec![0.4, 0.6]).unwrap()); // usize
            
            // Add some observations to make it interesting
            observe(addr!("y"), Normal::new(mu, 0.5).unwrap(), 1.5);
            
            pure((mu, coin, count, choice))
        }
    };
    
    // Run adaptive MCMC chain
    let samples = adaptive_mcmc_chain(
        &mut rng,
        model_fn,
        20, // Small number for test
        5,  // Brief warmup
    );
    
    assert_eq!(samples.len(), 20);
    
    // Verify all samples preserve correct types
    for (_, trace) in &samples {
        // Check f64 parameter
        let mu_choice = &trace.choices[&addr!("mu")];
        assert!(matches!(mu_choice.value, ChoiceValue::F64(_)));
        
        // Check bool parameter
        let coin_choice = &trace.choices[&addr!("coin")];
        assert!(matches!(coin_choice.value, ChoiceValue::Bool(_)));
        
        // Check u64 parameter
        let count_choice = &trace.choices[&addr!("count")];
        assert!(matches!(count_choice.value, ChoiceValue::U64(_)));
        
        // Check usize parameter
        let choice_choice = &trace.choices[&addr!("choice")];
        assert!(matches!(choice_choice.value, ChoiceValue::Usize(_)));
        
        // Verify values are reasonable
        if let ChoiceValue::F64(mu) = mu_choice.value {
            assert!(mu.is_finite());
            assert!(mu > -5.0 && mu < 5.0); // Reasonable range given observation
        }
        
        if let ChoiceValue::U64(count) = count_choice.value {
            assert!(count < 50); // Reasonable upper bound
        }
        
        if let ChoiceValue::Usize(choice) = choice_choice.value {
            assert!(choice < 10); // Should be reasonable given heuristic
        }
    }
}

#[test]
fn test_type_safe_proposal_edge_cases() {
    let mut rng = StdRng::seed_from_u64(42);
    
    // Test extreme f64 values
    let mut trace = Trace::default();
    trace.insert_choice(addr!("extreme"), ChoiceValue::F64(1e10), -1.0);
    
    let mut adaptation = DiminishingAdaptation::new(0.44, 0.7);
    
    let model_fn = || pure(42.0); // Dummy model
    
    // This should not crash
    let (_, new_trace) = adaptive_single_site_mh(
        &mut rng,
        &model_fn,
        &trace,
        &mut adaptation,
    );
    
    let choice = &new_trace.choices[&addr!("extreme")];
    assert!(matches!(choice.value, ChoiceValue::F64(_)));
    if let ChoiceValue::F64(val) = choice.value {
        assert!(val.is_finite());
    }
}

#[test]
fn test_proposal_with_zero_scale() {
    let mut rng = StdRng::seed_from_u64(42);
    
    let current = 5.0;
    let scale = 0.0;
    
    let strategy = GaussianWalkProposal;
    let proposed = strategy.propose(current, scale, &mut rng);
    
    // With zero scale, proposal should be very close to current (but might not be exactly equal due to numerical precision)
    assert!((proposed - current).abs() < 1e-10);
}

#[test]
fn test_adaptation_with_type_safe_mh() {
    let mut rng = StdRng::seed_from_u64(42);
    
    // Model with observation to create acceptance/rejection dynamics
    let model_fn = || {
        sample(addr!("theta"), Normal::new(0.0, 1.0).unwrap())
            .bind(|theta| {
                observe(addr!("y"), Normal::new(theta, 0.1).unwrap(), 2.0)
                    .map(move |_| theta)
            })
    };
    
    // Initialize
    let (_, initial_trace) = runtime::handler::run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        model_fn(),
    );
    
    let mut adaptation = DiminishingAdaptation::new(0.44, 0.7);
    let mut current_trace = initial_trace;
    
    // Run many steps to test adaptation
    let mut accept_count = 0;
    let n_steps = 100;
    
    for _ in 0..n_steps {
        let old_value = if let ChoiceValue::F64(val) = current_trace.choices[&addr!("theta")].value {
            val
        } else {
            panic!("Expected F64");
        };
        
        let (_, new_trace) = adaptive_single_site_mh(
            &mut rng,
            &model_fn,
            &current_trace,
            &mut adaptation,
        );
        
        let new_value = if let ChoiceValue::F64(val) = new_trace.choices[&addr!("theta")].value {
            val
        } else {
            panic!("Expected F64");
        };
        
        if (new_value - old_value).abs() > 1e-10 {
            accept_count += 1;
        }
        
        current_trace = new_trace;
        
        // Verify type preservation
        assert!(matches!(current_trace.choices[&addr!("theta")].value, ChoiceValue::F64(_)));
    }
    
    // Should have some acceptance
    let acceptance_rate = accept_count as f64 / n_steps as f64;
    assert!(acceptance_rate > 0.1 && acceptance_rate < 0.9); // Reasonable bounds
    
    // Adaptation should have occurred
    let final_scale = adaptation.get_scale(&addr!("theta"));
    assert!(final_scale > 0.0);
    assert!(final_scale.is_finite());
}

#[test] 
fn test_mh_with_no_random_variables() {
    let mut rng = StdRng::seed_from_u64(42);
    
    // Model with no random variables
    let model_fn = || pure(42.0);
    
    let empty_trace = Trace::default();
    let mut adaptation = DiminishingAdaptation::new(0.44, 0.7);
    
    // Should handle gracefully
    let (result, trace) = adaptive_single_site_mh(
        &mut rng,
        &model_fn,
        &empty_trace,
        &mut adaptation,
    );
    
    assert_eq!(result, 42.0);
    assert!(trace.choices.is_empty());
}

#[test]
fn test_integration_type_safe_bayesian_regression() {
    let mut rng = StdRng::seed_from_u64(42);
    
    let model_fn = || {
        // Realistic Bayesian regression model demonstrating type safety
        let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_data = vec![2.1, 3.9, 6.1, 8.0, 9.9];
        // f64 parameters
        sample(addr!("slope"), Normal::new(0.0, 1.0).unwrap())
            .bind(|slope| {
                sample(addr!("intercept"), Normal::new(0.0, 1.0).unwrap())
                    .bind(move |intercept| {
                        // bool parameter for model selection
                        sample(addr!("use_noise"), Bernoulli::new(0.8).unwrap())
                            .bind(move |use_noise| {
                                // u64 parameter for robustness (number of outliers to expect)
                                sample(addr!("n_outliers"), Poisson::new(0.5).unwrap())
                                    .bind(move |n_outliers| {
                                        // usize parameter for noise model selection
                                        sample(addr!("noise_model"), Categorical::new(vec![0.7, 0.3]).unwrap())
                                            .bind(move |noise_model| {
                                                // Use type-safe values naturally
                                                let base_noise = if use_noise { 0.1 } else { 0.05 };
                                                let noise_multiplier = if noise_model == 0 { 1.0 } else { 2.0 };
                                                let final_noise = base_noise * noise_multiplier;
                                                
                                                // Add likelihood with observations
                                                let mut result = pure((slope, intercept, use_noise, n_outliers, noise_model));
                                                for (i, (&x, &y)) in x_data.iter().zip(y_data.iter()).enumerate() {
                                                    let y_pred = slope * x + intercept;
                                                    result = result.bind(move |res| {
                                                        observe(addr!("y", i), Normal::new(y_pred, final_noise).unwrap(), y)
                                                            .map(move |_| res)
                                                    });
                                                }
                                                result
                                            })
                                    })
                            })
                    })
            })
    };
    
    // Run adaptive MCMC
    let samples = adaptive_mcmc_chain(
        &mut rng,
        model_fn,
        100, // Reasonable number for integration test
        50,  // Warmup
    );
    
    assert_eq!(samples.len(), 100);
    
    // Extract type-safe values using new accessor methods
    let slopes: Vec<f64> = samples.iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("slope")))
        .collect();
    
    let use_noise_flags: Vec<bool> = samples.iter()
        .filter_map(|(_, trace)| trace.get_bool(&addr!("use_noise")))
        .collect();
    
    let n_outliers_counts: Vec<u64> = samples.iter()
        .filter_map(|(_, trace)| trace.get_u64(&addr!("n_outliers")))
        .collect();
    
    let noise_models: Vec<usize> = samples.iter()
        .filter_map(|(_, trace)| trace.get_usize(&addr!("noise_model")))
        .collect();
    
    // All samples should have all parameters
    assert_eq!(slopes.len(), 100);
    assert_eq!(use_noise_flags.len(), 100);
    assert_eq!(n_outliers_counts.len(), 100);
    assert_eq!(noise_models.len(), 100);
    
    // Check that values are reasonable given the linear relationship
    let mean_slope = slopes.iter().sum::<f64>() / slopes.len() as f64;
    assert!(mean_slope > 1.5 && mean_slope < 2.5); // True slope is ~2.0
    
    // Check type constraints
    assert!(use_noise_flags.iter().all(|&b| b == true || b == false)); // Trivial but shows type safety
    assert!(n_outliers_counts.iter().all(|&n| n < 20)); // Reasonable upper bound
    assert!(noise_models.iter().all(|&m| m < 2)); // Valid categorical indices
    
    // Check that we can use the returned values with their natural types
    for (i, &slope) in slopes.iter().enumerate() {
        let use_noise = use_noise_flags[i];
        let n_outliers = n_outliers_counts[i];
        let noise_model = noise_models[i];
        
        // Natural usage - compiler enforces correct types
        assert!(slope.is_finite());
        let _prediction = slope * 3.0; // f64 arithmetic works naturally
        
        if use_noise { // bool works naturally in conditionals
            assert!(noise_model < 2); // usize works naturally in comparisons
        }
        
        let _total_params = 2 + n_outliers; // u64 arithmetic works naturally
    }
}

#[test]
fn test_demonstrates_distribution_awareness_limitation() {
    // This test demonstrates the current limitation: 
    // Beta and Normal both get the same Gaussian proposals despite very different geometries
    
    let mut rng = StdRng::seed_from_u64(42);
    
    let beta_model = || {
        // Beta distribution is bounded [0,1] but gets unbounded Gaussian proposals
        sample(addr!("p"), Beta::new(2.0, 2.0).unwrap())
            .map(|p| p)
    };
    
    let normal_model = || {
        // Normal distribution is unbounded, so Gaussian proposals are appropriate  
        sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap())
            .map(|mu| mu)
    };
    
    // Both models work, but Beta may have poor acceptance rates due to boundary violations
    let beta_samples = adaptive_mcmc_chain(&mut rng, beta_model, 50, 20);
    let normal_samples = adaptive_mcmc_chain(&mut rng, normal_model, 50, 20);
    
    assert_eq!(beta_samples.len(), 50);
    assert_eq!(normal_samples.len(), 50);
    
    // Extract values
    let beta_values: Vec<f64> = beta_samples.iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("p")))
        .collect();
    
    let normal_values: Vec<f64> = normal_samples.iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("mu")))
        .collect();
    
    // Both should produce valid samples, but Beta might have more repeated values
    // due to boundary rejections (this is the limitation we're documenting)
    assert!(beta_values.iter().all(|&x| x >= 0.0 && x <= 1.0));
    assert!(normal_values.iter().all(|&x| x.is_finite()));
    
    // Note: In a production system, Beta would benefit from logit-transform proposals
    // while Normal works fine with Gaussian proposals. Currently both get Gaussian.
}
