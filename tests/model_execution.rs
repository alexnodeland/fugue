//! # Model Execution Integration Tests
//! 
//! This module contains integration tests for end-to-end model execution flows.
//! These tests validate that models can be defined, executed with different handlers,
//! and produce expected results using **only the public API**.
//!
//! ## Test Categories
//!
//! ### 1. Basic Model Execution (`test_basic_*`)
//! - Simple model creation and execution with `PriorHandler`
//! - Verify that `runtime::handler::run()` works correctly
//! - Test that traces contain expected addresses and values
//! - Validate log weight accumulation
//!
//! ### 2. Handler Compatibility (`test_handler_*`)
//! - Test all handler types: `PriorHandler`, `ReplayHandler`, `SafeReplayHandler`, 
//!   `ScoreGivenTrace`, `SafeScoreGivenTrace`
//! - Verify handlers produce consistent results for same models
//! - Test handler-specific behaviors (replay consistency, safe fallbacks)
//!
//! ### 3. Model Composition (`test_composition_*`)
//! - Test `bind`, `map`, `and_then` operations
//! - Test `zip` for combining models
//! - Test `sequence_vec` and `traverse_vec` for collections
//! - Verify composed models execute correctly end-to-end
//!
//! ### 4. Mixed Type Support (`test_mixed_types_*`)
//! - Models with `f64`, `bool`, `u64`, `usize` values
//! - Type-safe trace access for different value types
//! - Integration between continuous and discrete distributions
//!
//! ### 5. Factor and Guard Integration (`test_factor_guard_*`)
//! - Models with `factor()` statements affecting log weights
//! - Models with `guard()` conditions
//! - Integration of factors and guards with observations
//!
//! ### 6. Distribution Coverage (`test_distribution_*`)
//! - End-to-end execution with all distribution types:
//!   - Continuous: `Normal`, `Uniform`, `Exponential`, `Beta`, `Gamma`, `LogNormal`
//!   - Discrete: `Bernoulli`, `Poisson`, `Binomial`, `Categorical`
//! - Verify each distribution works in models and produces valid traces
//!
//! ### 7. Macro Integration (`test_macro_*`)
//! - `prob!` macro for model definition
//! - `addr!` macro for address creation
//! - `plate!` and `scoped_addr!` for structured addressing
//!
//! ## Implementation Guidelines
//!
//! - **Public API Only**: Use `fugue::*` imports, avoid `crate::` paths
//! - **Handler Creation**: Use struct literal syntax like examples:
//!   ```rust
//!   let handler = runtime::interpreters::PriorHandler {
//!       rng: &mut rng,
//!       trace: runtime::trace::Trace::default(),
//!   };
//!   ```
//! - **Address Creation**: Use `addr!("name")` macro, not `Address::new()`
//! - **Model Execution**: Use `runtime::handler::run(handler, model)`
//! - **Trace Access**: Use `trace.get_f64(&addr!("name"))` etc., returns `Option<T>`
//! - **Type Safety**: Test both successful access and type mismatches
//!
//! ## Expected Test Structure
//!
//! Each test should follow this pattern:
//! 1. Set up RNG with fixed seed for reproducibility
//! 2. Define model using public API
//! 3. Create appropriate handler
//! 4. Execute with `runtime::handler::run()`
//! 5. Assert on results and trace properties
//! 6. Test edge cases and error conditions

use fugue::*;
use rand::{rngs::StdRng, SeedableRng};

#[test]
fn test_basic_prior_sampling() {
    let mut rng = StdRng::seed_from_u64(42);
    
    // Define a simple model that samples from a normal distribution
    let model = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
    
    // Create a PriorHandler
    let handler = runtime::interpreters::PriorHandler {
        rng: &mut rng,
        trace: runtime::trace::Trace::default(),
    };
    
    // Execute the model
    let (value, trace) = runtime::handler::run(handler, model);
    
    // Check that the trace contains the expected address
    let x_value = trace.get_f64(&addr!("x"));
    assert!(x_value.is_some());
    
    // The sampled value should equal the trace value
    assert_eq!(value, x_value.unwrap());
    
    // Check that the total log weight is finite (should be 0.0 for pure prior sampling)
    let log_weight = trace.total_log_weight();
    assert!(log_weight.is_finite());
}

#[test]
fn test_model_with_observation_and_factor() {
    let mut rng = StdRng::seed_from_u64(42);
    
    // Define a model with sample, observe, and factor
    let model = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
        .bind(|x| observe(addr!("y"), Normal::new(x, 1.0).unwrap(), 0.5))
        .bind(|_| factor(-1.0));
    
    // Create a PriorHandler
    let handler = runtime::interpreters::PriorHandler {
        rng: &mut rng,
        trace: runtime::trace::Trace::default(),
    };
    
    // Execute the model
    let ((), trace) = runtime::handler::run(handler, model);
    
    // Check that the trace contains the expected address
    let x_value = trace.get_f64(&addr!("x"));
    assert!(x_value.is_some());
    
    // Check that all components of the log weight are finite
    assert!(trace.log_prior.is_finite());
    assert!(trace.log_likelihood.is_finite());
    assert!(trace.log_factors.is_finite());
    
    // The factor should contribute exactly -1.0
    assert!((trace.log_factors + 1.0).abs() < 1e-12);
    
    // Total log weight should be finite
    let log_weight = trace.total_log_weight();
    assert!(log_weight.is_finite());
}

#[test]
fn test_replay_and_score_handlers() {
    let mut rng = StdRng::seed_from_u64(42);
    
    // Define a simple model
    let model = || sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
        .bind(|x| observe(addr!("y"), Normal::new(x, 1.0).unwrap(), 0.5)
            .map(move |_| x));
    
    // First, run with PriorHandler to get a trace
    let prior_handler = runtime::interpreters::PriorHandler {
        rng: &mut rng,
        trace: runtime::trace::Trace::default(),
    };
    
    let (original_value, original_trace) = runtime::handler::run(prior_handler, model());
    
    // Now replay with ReplayHandler - should get same result
    let replay_handler = runtime::interpreters::ReplayHandler {
        rng: &mut rng,
        base: original_trace.clone(),
        trace: runtime::trace::Trace::default(),
    };
    
    let (replayed_value, replayed_trace) = runtime::handler::run(replay_handler, model());
    
    // Replayed value should match original
    assert_eq!(original_value, replayed_value);
    
    // Both traces should have the same x value
    let original_x = original_trace.get_f64(&addr!("x")).unwrap();
    let replayed_x = replayed_trace.get_f64(&addr!("x")).unwrap();
    assert_eq!(original_x, replayed_x);
    
    // Now test ScoreGivenTrace handler
    let score_handler = runtime::interpreters::ScoreGivenTrace {
        base: original_trace.clone(),
        trace: runtime::trace::Trace::default(),
    };
    
    let (scored_value, scored_trace) = runtime::handler::run(score_handler, model());
    
    // Scored value should match original (since it's deterministic replay)
    assert_eq!(scored_value, original_value);
    
    // Scored trace should have the same total log weight structure
    assert!(scored_trace.total_log_weight().is_finite());
    assert_eq!(scored_trace.get_f64(&addr!("x")).unwrap(), original_x);
}

#[test]
fn test_model_composition() {
    let mut rng = StdRng::seed_from_u64(42);
    
    // Test bind and map operations
    let model1 = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
        .bind(|x| sample(addr!("y"), Normal::new(x, 0.5).unwrap()))
        .map(|y| y * 2.0);
    
    let handler1 = runtime::interpreters::PriorHandler {
        rng: &mut rng,
        trace: runtime::trace::Trace::default(),
    };
    
    let (result1, trace1) = runtime::handler::run(handler1, model1);
    
    // Check that both addresses are in the trace
    let _x_val = trace1.get_f64(&addr!("x")).unwrap();
    let y_val = trace1.get_f64(&addr!("y")).unwrap();
    
    // Result should be y * 2.0
    assert_eq!(result1, y_val * 2.0);
    
    // Test zip operation
    let model_a = sample(addr!("a"), Normal::new(0.0, 1.0).unwrap());
    let model_b = sample(addr!("b"), Normal::new(1.0, 1.0).unwrap());
    let zipped_model = zip(model_a, model_b);
    
    let handler2 = runtime::interpreters::PriorHandler {
        rng: &mut rng,
        trace: runtime::trace::Trace::default(),
    };
    
    let ((a_result, b_result), trace2) = runtime::handler::run(handler2, zipped_model);
    
    // Check that both addresses are in the trace
    let a_val = trace2.get_f64(&addr!("a")).unwrap();
    let b_val = trace2.get_f64(&addr!("b")).unwrap();
    
    // Results should match trace values
    assert_eq!(a_result, a_val);
    assert_eq!(b_result, b_val);
    
    // Test sequence_vec
    let models = vec![
        sample(addr!("seq_0"), Normal::new(0.0, 1.0).unwrap()),
        sample(addr!("seq_1"), Normal::new(1.0, 1.0).unwrap()),
        sample(addr!("seq_2"), Normal::new(2.0, 1.0).unwrap()),
    ];
    let sequence_model = sequence_vec(models);
    
    let handler3 = runtime::interpreters::PriorHandler {
        rng: &mut rng,
        trace: runtime::trace::Trace::default(),
    };
    
    let (seq_results, trace3) = runtime::handler::run(handler3, sequence_model);
    
    // Check that all sequence addresses are in the trace
    assert_eq!(seq_results.len(), 3);
    assert_eq!(seq_results[0], trace3.get_f64(&addr!("seq_0")).unwrap());
    assert_eq!(seq_results[1], trace3.get_f64(&addr!("seq_1")).unwrap());
    assert_eq!(seq_results[2], trace3.get_f64(&addr!("seq_2")).unwrap());
}

#[test]
fn test_mixed_types() {
    let mut rng = StdRng::seed_from_u64(42);
    
    // Define a model with multiple value types
    let model = sample(addr!("f64_val"), Normal::new(0.0, 1.0).unwrap())
        .bind(|_| sample(addr!("bool_val"), Bernoulli::new(0.6).unwrap()))
        .bind(|_| sample(addr!("u64_val"), Poisson::new(3.0).unwrap()))
        .bind(|_| sample(addr!("usize_val"), Categorical::new(vec![0.3, 0.4, 0.3]).unwrap()))
        .map(|usize_val| (usize_val, "mixed_types_result"));
    
    let handler = runtime::interpreters::PriorHandler {
        rng: &mut rng,
        trace: runtime::trace::Trace::default(),
    };
    
    let ((usize_result, string_result), trace) = runtime::handler::run(handler, model);
    
    // Test type-safe trace access for different value types
    let f64_val = trace.get_f64(&addr!("f64_val"));
    assert!(f64_val.is_some());
    
    let bool_val = trace.get_bool(&addr!("bool_val"));
    assert!(bool_val.is_some());
    
    let u64_val = trace.get_u64(&addr!("u64_val"));
    assert!(u64_val.is_some());
    
    let usize_val = trace.get_usize(&addr!("usize_val"));
    assert!(usize_val.is_some());
    
    // The returned usize should match the trace value
    assert_eq!(usize_result, usize_val.unwrap());
    assert_eq!(string_result, "mixed_types_result");
    
    // Test type mismatches return None (not panicking)
    assert!(trace.get_f64(&addr!("bool_val")).is_none());
    assert!(trace.get_bool(&addr!("f64_val")).is_none());
    assert!(trace.get_u64(&addr!("usize_val")).is_none());
    assert!(trace.get_usize(&addr!("u64_val")).is_none());
    
    // Test result variants that return errors instead of panicking
    assert!(trace.get_f64_result(&addr!("bool_val")).is_err());
    assert!(trace.get_bool_result(&addr!("f64_val")).is_err());
    assert!(trace.get_u64_result(&addr!("usize_val")).is_err());
    assert!(trace.get_usize_result(&addr!("u64_val")).is_err());
    
    // Test missing addresses
    assert!(trace.get_f64(&addr!("missing")).is_none());
    assert!(trace.get_f64_result(&addr!("missing")).is_err());
}

#[test]
fn test_macro_integration() {
    let mut rng = StdRng::seed_from_u64(42);
    
    // Test using macros in model definition  
    let model = prob!(
        let x <- sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
        let y <- sample(addr!("y"), Normal::new(x, 0.5).unwrap());
        observe(addr!("obs"), Normal::new(x, 0.5).unwrap(), 0.3);
        factor(-0.5);
        pure(x + y)
    );
    
    let handler = runtime::interpreters::PriorHandler {
        rng: &mut rng,
        trace: runtime::trace::Trace::default(),
    };
    
    let (result, trace) = runtime::handler::run(handler, model);
    
    // Check that addresses are created correctly by macros
    let x_val = trace.get_f64(&addr!("x")).unwrap();
    let y_val = trace.get_f64(&addr!("y")).unwrap();
    
    // Result should be x + y
    assert_eq!(result, x_val + y_val);
    
    // Check that factor was applied
    assert!((trace.log_factors + 0.5).abs() < 1e-12);
    
    // Test scoped_addr macro with plate
    let plate_model = plate! { i in 0..3 =>
        sample(scoped_addr!("plate", "item", "{}", i), Normal::new(i as f64, 1.0).unwrap())
    };
    
    let handler2 = runtime::interpreters::PriorHandler {
        rng: &mut rng,
        trace: runtime::trace::Trace::default(),
    };
    
    let (plate_results, trace2) = runtime::handler::run(handler2, plate_model);
    
    // Check that all plate addresses are created correctly
    assert_eq!(plate_results.len(), 3);
    for i in 0..3 {
        let addr = scoped_addr!("plate", "item", "{}", i);
        let val = trace2.get_f64(&addr);
        assert!(val.is_some());
        assert_eq!(plate_results[i], val.unwrap());
    }
}

#[test]
fn test_factor_guard_integration() {
    let mut rng = StdRng::seed_from_u64(42);
    
    // Test guard() conditions
    let guard_model = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
        .bind(|x| guard(x > -2.0 && x < 2.0))  // Should usually pass for standard normal
        .bind(|_| factor(-0.5))
        .bind(|_| pure("guard_passed"));
    
    let handler = runtime::interpreters::PriorHandler {
        rng: &mut rng,
        trace: runtime::trace::Trace::default(),
    };
    
    let (result, trace) = runtime::handler::run(handler, guard_model);
    
    // If we get here, the guard passed
    assert_eq!(result, "guard_passed");
    
    // Check that both factor and guard affected the trace
    assert!(trace.log_factors.is_finite());
    assert!((trace.log_factors + 0.5).abs() < 1e-12);
    
    // The sampled x should be within the guard bounds
    let x_val = trace.get_f64(&addr!("x")).unwrap();
    assert!(x_val > -2.0 && x_val < 2.0);
    
    // Test complex model with factors, guards, and observations
    let complex_model = sample(addr!("mu"), Normal::new(0.0, 2.0).unwrap())
        .bind(|mu| guard(mu.abs() < 5.0)  // Reasonable bound
             .bind(move |_| sample(addr!("sigma"), Exponential::new(1.0).unwrap())
                  .bind(move |sigma| guard(sigma > 0.1 && sigma < 10.0)  // Reasonable sigma bounds
                       .bind(move |_| observe(addr!("y"), Normal::new(mu, sigma).unwrap(), 1.5))
                       .bind(move |_| factor(mu * 0.1))  // Small preference for positive mu
                       .map(move |_| (mu, sigma)))));
    
    let handler2 = runtime::interpreters::PriorHandler {
        rng: &mut rng,
        trace: runtime::trace::Trace::default(),
    };
    
    let ((mu_result, sigma_result), trace2) = runtime::handler::run(handler2, complex_model);
    
    // All guards should have passed
    assert!(mu_result.abs() < 5.0);
    assert!(sigma_result > 0.1 && sigma_result < 10.0);
    
    // Check trace components
    assert!(trace2.log_prior.is_finite());
    assert!(trace2.log_likelihood.is_finite());
    assert!(trace2.log_factors.is_finite());
    assert!(trace2.total_log_weight().is_finite());
    
    // Factor should be mu * 0.1
    assert!((trace2.log_factors - mu_result * 0.1).abs() < 1e-12);
}

#[test]
fn test_distribution_coverage() {
    let mut rng = StdRng::seed_from_u64(42);
    
    // Test all continuous distributions in models
    let continuous_model = sample(addr!("normal"), Normal::new(0.0, 1.0).unwrap())
        .bind(|_| sample(addr!("uniform"), Uniform::new(0.0, 1.0).unwrap()))
        .bind(|_| sample(addr!("exponential"), Exponential::new(1.0).unwrap()))
        .bind(|_| sample(addr!("beta"), Beta::new(2.0, 3.0).unwrap()))
        .bind(|_| sample(addr!("gamma"), Gamma::new(2.0, 1.0).unwrap()))
        .bind(|_| sample(addr!("lognormal"), LogNormal::new(0.0, 1.0).unwrap()));
    
    let handler1 = runtime::interpreters::PriorHandler {
        rng: &mut rng,
        trace: runtime::trace::Trace::default(),
    };
    
    let (lognormal_val, trace1) = runtime::handler::run(handler1, continuous_model);
    
    // Check that all continuous distributions produced finite values
    assert!(trace1.get_f64(&addr!("normal")).unwrap().is_finite());
    assert!(trace1.get_f64(&addr!("uniform")).unwrap().is_finite());
    assert!(trace1.get_f64(&addr!("exponential")).unwrap().is_finite());
    assert!(trace1.get_f64(&addr!("beta")).unwrap().is_finite());
    assert!(trace1.get_f64(&addr!("gamma")).unwrap().is_finite());
    assert!(trace1.get_f64(&addr!("lognormal")).unwrap().is_finite());
    assert_eq!(lognormal_val, trace1.get_f64(&addr!("lognormal")).unwrap());
    
    // Test all discrete distributions in models
    let discrete_model = sample(addr!("bernoulli"), Bernoulli::new(0.7).unwrap())
        .bind(|_| sample(addr!("poisson"), Poisson::new(3.0).unwrap()))
        .bind(|_| sample(addr!("binomial"), Binomial::new(10, 0.4).unwrap()))
        .bind(|_| sample(addr!("categorical"), Categorical::new(vec![0.2, 0.3, 0.5]).unwrap()));
    
    let handler2 = runtime::interpreters::PriorHandler {
        rng: &mut rng,
        trace: runtime::trace::Trace::default(),
    };
    
    let (categorical_val, trace2) = runtime::handler::run(handler2, discrete_model);
    
    // Check that all discrete distributions produced valid values
    let bernoulli_val = trace2.get_bool(&addr!("bernoulli")).unwrap();
    let poisson_val = trace2.get_u64(&addr!("poisson")).unwrap();
    let binomial_val = trace2.get_u64(&addr!("binomial")).unwrap();
    let categorical_result = trace2.get_usize(&addr!("categorical")).unwrap();
    
    assert!(bernoulli_val == true || bernoulli_val == false);
    assert!(poisson_val >= 0);
    assert!(binomial_val >= 0 && binomial_val <= 10);
    assert!(categorical_result < 3);  // Should be 0, 1, or 2
    assert_eq!(categorical_val, categorical_result);
    
    // Test mixed continuous and discrete in one model with observations
    let mixed_model = sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap())
        .bind(|mu| sample(addr!("success"), Bernoulli::new(0.6).unwrap())
             .bind(move |success| {
                 if success {
                     observe(addr!("obs"), Normal::new(mu, 0.5).unwrap(), 0.8)
                         .map(move |_| (mu, success))
                 } else {
                     observe(addr!("obs"), Normal::new(mu, 0.5).unwrap(), -0.3)
                         .map(move |_| (mu, success))
                 }
             }));
    
    let handler3 = runtime::interpreters::PriorHandler {
        rng: &mut rng,
        trace: runtime::trace::Trace::default(),
    };
    
    let ((mu_mixed, success_mixed), trace3) = runtime::handler::run(handler3, mixed_model);
    
    // Check integration between continuous and discrete
    assert!(mu_mixed.is_finite());
    assert!(success_mixed == true || success_mixed == false);
    assert!(trace3.log_likelihood.is_finite());
    assert_eq!(mu_mixed, trace3.get_f64(&addr!("mu")).unwrap());
    assert_eq!(success_mixed, trace3.get_bool(&addr!("success")).unwrap());
}

#[test]
fn test_handler_compatibility_complete() {
    let mut rng = StdRng::seed_from_u64(42);
    
    // Define a model that tests all handler capabilities
    let model = || sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
        .bind(|x| sample(addr!("y"), Bernoulli::new(0.6).unwrap())
             .bind(move |y| observe(addr!("obs"), Normal::new(x, 0.5).unwrap(), 0.5)
                  .bind(move |_| factor(if y { 0.1 } else { -0.1 }))
                  .map(move |_| (x, y))));
    
    // 1. Get baseline with PriorHandler
    let prior_handler = runtime::interpreters::PriorHandler {
        rng: &mut rng,
        trace: runtime::trace::Trace::default(),
    };
    
    let (baseline_result, baseline_trace) = runtime::handler::run(prior_handler, model());
    
    // 2. Test ReplayHandler (exact replay)
    let replay_handler = runtime::interpreters::ReplayHandler {
        rng: &mut rng,
        base: baseline_trace.clone(),
        trace: runtime::trace::Trace::default(),
    };
    
    let (replay_result, replay_trace) = runtime::handler::run(replay_handler, model());
    
    // ReplayHandler should produce identical results
    assert_eq!(baseline_result, replay_result);
    assert_eq!(baseline_trace.get_f64(&addr!("x")), replay_trace.get_f64(&addr!("x")));
    assert_eq!(baseline_trace.get_bool(&addr!("y")), replay_trace.get_bool(&addr!("y")));
    
    // 3. Test SafeReplayHandler (falls back to sampling for missing addresses)
    let safe_replay_handler = runtime::interpreters::SafeReplayHandler {
        rng: &mut rng,
        base: baseline_trace.clone(),
        trace: runtime::trace::Trace::default(),
        warn_on_mismatch: false,
    };
    
    let (safe_replay_result, safe_replay_trace) = runtime::handler::run(safe_replay_handler, model());
    
    // SafeReplayHandler should produce same results when all addresses exist
    assert_eq!(baseline_result, safe_replay_result);
    assert_eq!(baseline_trace.get_f64(&addr!("x")), safe_replay_trace.get_f64(&addr!("x")));
    assert_eq!(baseline_trace.get_bool(&addr!("y")), safe_replay_trace.get_bool(&addr!("y")));
    
    // 4. Test ScoreGivenTrace (deterministic scoring)
    let score_handler = runtime::interpreters::ScoreGivenTrace {
        base: baseline_trace.clone(),
        trace: runtime::trace::Trace::default(),
    };
    
    let (score_result, score_trace) = runtime::handler::run(score_handler, model());
    
    // ScoreGivenTrace should produce same results
    assert_eq!(baseline_result, score_result);
    assert_eq!(baseline_trace.get_f64(&addr!("x")), score_trace.get_f64(&addr!("x")));
    assert_eq!(baseline_trace.get_bool(&addr!("y")), score_trace.get_bool(&addr!("y")));
    
    // 5. Test SafeScoreGivenTrace (safe scoring with fallbacks)
    let safe_score_handler = runtime::interpreters::SafeScoreGivenTrace {
        base: baseline_trace.clone(),
        trace: runtime::trace::Trace::default(),
        warn_on_error: false,
    };
    
    let (safe_score_result, safe_score_trace) = runtime::handler::run(safe_score_handler, model());
    
    // SafeScoreGivenTrace should produce same results when all addresses exist
    assert_eq!(baseline_result, safe_score_result);
    assert_eq!(baseline_trace.get_f64(&addr!("x")), safe_score_trace.get_f64(&addr!("x")));
    assert_eq!(baseline_trace.get_bool(&addr!("y")), safe_score_trace.get_bool(&addr!("y")));
    
    // All handlers should produce finite log weights
    assert!(baseline_trace.total_log_weight().is_finite());
    assert!(replay_trace.total_log_weight().is_finite());
    assert!(safe_replay_trace.total_log_weight().is_finite());
    assert!(score_trace.total_log_weight().is_finite());
    assert!(safe_score_trace.total_log_weight().is_finite());
}

#[test]
fn test_model_composition_complete() {
    let mut rng = StdRng::seed_from_u64(42);
    
    // Test and_then operation (should be equivalent to bind)
    let and_then_model = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
        .and_then(|x| sample(addr!("y"), Normal::new(x, 0.5).unwrap()))
        .map(|y| y * 2.0);
    
    let handler1 = runtime::interpreters::PriorHandler {
        rng: &mut rng,
        trace: runtime::trace::Trace::default(),
    };
    
    let (and_then_result, trace1) = runtime::handler::run(handler1, and_then_model);
    
    // Check that and_then works like bind
    let _x_val = trace1.get_f64(&addr!("x")).unwrap();
    let y_val = trace1.get_f64(&addr!("y")).unwrap();
    assert_eq!(and_then_result, y_val * 2.0);
    
    // Test traverse_vec operation
    let data = vec![1.0, 2.0, 3.0];
    let traverse_model = traverse_vec(data.clone(), |x| {
        let idx = (x as usize).saturating_sub(1);  // Convert 1.0->0, 2.0->1, 3.0->2
        sample(addr!("traverse", idx), Normal::new(x, 0.5).unwrap())
    });
    
    let handler2 = runtime::interpreters::PriorHandler {
        rng: &mut rng,
        trace: runtime::trace::Trace::default(),
    };
    
    let (traverse_results, trace2) = runtime::handler::run(handler2, traverse_model);
    
    // Check that traverse_vec processes all elements
    assert_eq!(traverse_results.len(), 3);
    for (i, _original_val) in data.iter().enumerate() {
        let addr = addr!("traverse", i);
        let sampled_val = trace2.get_f64(&addr).unwrap();
        assert_eq!(traverse_results[i], sampled_val);
        // The sampled value should be reasonably close to the mean (original_val)
        // but we can't assert exact equality due to randomness
        assert!(sampled_val.is_finite());
    }
    
    // Test complex composition with multiple operations
    let complex_composition = sample(addr!("a"), Normal::new(0.0, 1.0).unwrap())
        .bind(|a| sample(addr!("b"), Normal::new(a, 0.5).unwrap())
             .map(move |b| (a, b)))
        .and_then(|(a, b)| zip(pure(a + b), pure(a - b)))
        .bind(|(sum, diff)| sequence_vec(vec![pure(sum), pure(diff), pure(sum * diff)]))
        .map(|results| results.iter().sum::<f64>());
    
    let handler3 = runtime::interpreters::PriorHandler {
        rng: &mut rng,
        trace: runtime::trace::Trace::default(),
    };
    
    let (complex_result, trace3) = runtime::handler::run(handler3, complex_composition);
    
    // Verify the complex composition worked correctly
    let a_val = trace3.get_f64(&addr!("a")).unwrap();
    let b_val = trace3.get_f64(&addr!("b")).unwrap();
    let sum = a_val + b_val;
    let diff = a_val - b_val;
    let expected_result = sum + diff + (sum * diff);
    
    assert!((complex_result - expected_result).abs() < 1e-12);
    assert!(complex_result.is_finite());
}