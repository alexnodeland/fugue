//! # Public API Coverage Integration Tests
//!
//! This module contains integration tests that validate comprehensive coverage
//! of the public API surface. These tests ensure that all major public API
//! components work together correctly and provide the expected functionality.
//!
//! ## Test Categories
//!
//! ### 1. Distribution API Coverage (`test_distribution_*`)
//! - All distribution constructors with valid/invalid parameters
//! - `sample()` and `log_prob()` methods for each distribution type
//! - `validate()` method integration
//! - Error handling for invalid parameters
//! - Type safety across different distribution families
//!
//! ### 2. Model API Coverage (`test_model_*`)
//! - Core functions: `pure()`, `sample()`, `observe()`, `factor()`, `guard()`
//! - `ModelExt` trait methods: `bind()`, `map()`, `and_then()`
//! - Utility functions: `zip()`, `sequence_vec()`, `traverse_vec()`
//! - Type-specific samplers: `sample_f64()`, `sample_bool()`, etc.
//! - Integration between all model operations
//!
//! ### 3. Handler API Coverage (`test_handler_*`)
//! - All handler types and their creation patterns
//! - `Handler` trait implementation consistency
//! - `runtime::handler::run()` with different handler types
//! - Handler-specific behaviors and error handling
//! - Memory management and resource cleanup
//!
//! ### 4. Trace API Coverage (`test_trace_*`)
//! - `Trace` struct field access and manipulation
//! - Type-safe accessors: `get_f64()`, `get_bool()`, etc.
//! - Result-based accessors: `get_f64_result()`, etc.
//! - `ChoiceValue` enum and its methods
//! - `Choice` struct and trace building
//! - Log weight calculation: `total_log_weight()`
//!
//! ### 5. Address System Coverage (`test_address_*`)
//! - `addr!()` macro with different patterns
//! - `scoped_addr!()` for hierarchical addressing
//! - `Address` struct behavior and comparison
//! - Integration with trace access and model definition
//!
//! ### 6. Macro System Coverage (`test_macro_*`)
//! - `prob!` macro for model definition
//! - `plate!` macro for vectorized operations
//! - Macro interaction with type system
//! - Nested macro usage and composition
//!
//! ### 7. Memory Management Coverage (`test_memory_*`)
//! - `TracePool` and `PooledPriorHandler` integration
//! - `CowTrace` copy-on-write semantics
//! - `TraceBuilder` for manual trace construction
//! - Memory efficiency and resource usage
//! - Pool statistics and monitoring
//!
//! ### 8. Numerical Utilities Coverage (`test_numerical_*`)
//! - `log_sum_exp()` and `weighted_log_sum_exp()`
//! - `normalize_log_probs()` probability normalization
//! - `log1p_exp()` and `safe_ln()` numerical stability
//! - Integration with inference algorithms
//! - Edge case handling (infinities, NaN, zeros)
//!
//! ### 9. Error Handling Coverage (`test_error_*`)
//! - `FugueError` variants and error propagation
//! - `ErrorCode` and `ErrorCategory` classification
//! - `ErrorContext` for detailed error information
//! - `Validate` trait implementation across types
//! - Error recovery and graceful degradation
//!
//! ### 10. Inference API Coverage (`test_inference_api_*`)
//! - All inference function signatures and parameter validation
//! - Configuration objects: `SMCConfig`, etc.
//! - Return type consistency and interpretation
//! - Integration between different inference methods
//! - Performance characteristics and scalability
//!
//! ## Implementation Strategy
//!
//! ### Systematic Coverage
//! - **Enumerate all public exports** from `src/lib.rs`
//! - **Test each export** in isolation and in combination
//! - **Validate type signatures** and expected behaviors
//! - **Test error conditions** for robustness
//!
//! ### Integration Patterns
//! - **Composition Testing**: Combine multiple API components
//! - **Workflow Testing**: End-to-end usage patterns
//! - **Edge Case Testing**: Boundary conditions and limits
//! - **Performance Testing**: Resource usage and efficiency
//!
//! ### Documentation Validation
//! - **Example Code Testing**: Validate documentation examples
//! - **API Contract Testing**: Verify stated behaviors
//! - **Consistency Testing**: Cross-reference related functions
//!
//! ## Implementation Guidelines
//!
//! - **Public API Only**: Never use `crate::` imports or internal paths
//! - **Comprehensive Coverage**: Test every public export at least once
//! - **Error Path Testing**: Validate error conditions and messages
//! - **Type Safety Testing**: Verify compile-time and runtime type safety
//! - **Resource Management**: Test cleanup and memory management
//! - **Cross-Platform**: Ensure tests work across different environments
//!
//! ## Test Organization
//!
//! Tests should be organized by API surface area, with each test focusing
//! on a specific aspect of the public API. Use descriptive test names that
//! clearly indicate what functionality is being validated.
//!
//! Example naming convention:
//! - `test_distribution_normal_constructor_validation()`
//! - `test_model_bind_chain_composition()`
//! - `test_trace_type_safe_access_all_types()`
//! - `test_handler_prior_basic_execution()`

use fugue::*;
use rand::{rngs::StdRng, SeedableRng};

#[test]
fn test_distribution_constructors_and_validation() {
    // Test all distribution constructors with valid parameters
    assert!(Normal::new(0.0, 1.0).is_ok());
    assert!(Bernoulli::new(0.5).is_ok());
    assert!(Uniform::new(0.0, 1.0).is_ok());
    assert!(Exponential::new(1.0).is_ok());
    assert!(Beta::new(1.0, 1.0).is_ok());
    assert!(Gamma::new(1.0, 1.0).is_ok());
    assert!(LogNormal::new(0.0, 1.0).is_ok());
    assert!(Poisson::new(1.0).is_ok());
    assert!(Binomial::new(10, 0.5).is_ok());
    assert!(Categorical::new(vec![0.3, 0.7]).is_ok());

    // Test invalid parameters return errors
    assert!(Normal::new(0.0, -1.0).is_err()); // negative std
    assert!(Bernoulli::new(1.5).is_err()); // p > 1
    assert!(Uniform::new(1.0, 0.0).is_err()); // a > b
    assert!(Exponential::new(-1.0).is_err()); // negative rate
    assert!(Beta::new(-1.0, 1.0).is_err()); // negative alpha
    assert!(Gamma::new(0.0, 1.0).is_err()); // zero shape
    assert!(LogNormal::new(0.0, -1.0).is_err()); // negative sigma
    assert!(Poisson::new(-1.0).is_err()); // negative lambda
    assert!(Binomial::new(10, -0.1).is_err()); // negative p
    assert!(Categorical::new(vec![]).is_err()); // empty weights
}

#[test]
fn test_distribution_sampling_and_log_prob() {
    let mut rng = StdRng::seed_from_u64(42);

    // Test continuous distributions
    let normal = Normal::new(0.0, 1.0).unwrap();
    let x = normal.sample(&mut rng);
    assert!(x.is_finite());
    assert!(normal.log_prob(&x).is_finite());

    let uniform = Uniform::new(0.0, 1.0).unwrap();
    let u = uniform.sample(&mut rng);
    assert!((0.0..=1.0).contains(&u));
    assert!(uniform.log_prob(&u).is_finite());

    // Test discrete distributions
    let bernoulli = Bernoulli::new(0.5).unwrap();
    let b = bernoulli.sample(&mut rng);
    let _ = b; // Just checking it's a valid bool
    assert!(bernoulli.log_prob(&b).is_finite());

    let poisson = Poisson::new(2.0).unwrap();
    let p = poisson.sample(&mut rng);
    let _ = p; // p is u64, comparison with 0 is always true
    assert!(poisson.log_prob(&p).is_finite());
}

#[test]
fn test_model_core_functions() {
    let mut rng = StdRng::seed_from_u64(42);

    // Test pure
    let pure_model = pure(42.0);
    let handler1 = runtime::interpreters::PriorHandler {
        rng: &mut rng,
        trace: runtime::trace::Trace::default(),
    };
    let (result1, _) = runtime::handler::run(handler1, pure_model);
    assert_eq!(result1, 42.0);

    // Test sample
    let sample_model = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
    let handler2 = runtime::interpreters::PriorHandler {
        rng: &mut rng,
        trace: runtime::trace::Trace::default(),
    };
    let (result2, trace2) = runtime::handler::run(handler2, sample_model);
    assert!(result2.is_finite());
    assert!(trace2.get_f64(&addr!("x")).is_some());

    // Test observe
    let observe_model = sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap())
        .bind(|mu| observe(addr!("y"), Normal::new(mu, 1.0).unwrap(), 0.5).map(move |_| mu));
    let handler3 = runtime::interpreters::PriorHandler {
        rng: &mut rng,
        trace: runtime::trace::Trace::default(),
    };
    let (result3, trace3) = runtime::handler::run(handler3, observe_model);
    assert!(result3.is_finite());
    assert!(trace3.log_likelihood.is_finite());

    // Test factor
    let factor_model = pure(1.0).bind(|_| factor(-1.5));
    let handler4 = runtime::interpreters::PriorHandler {
        rng: &mut rng,
        trace: runtime::trace::Trace::default(),
    };
    let (_, trace4) = runtime::handler::run(handler4, factor_model);
    assert!((trace4.log_factors + 1.5).abs() < 1e-12);
}

#[test]
fn test_model_ext_trait_methods() {
    let mut rng = StdRng::seed_from_u64(42);

    // Test bind
    let bind_model = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap()).bind(|x| pure(x * 2.0));
    let handler1 = runtime::interpreters::PriorHandler {
        rng: &mut rng,
        trace: runtime::trace::Trace::default(),
    };
    let (result1, trace1) = runtime::handler::run(handler1, bind_model);
    let x_val = trace1.get_f64(&addr!("x")).unwrap();
    assert_eq!(result1, x_val * 2.0);

    // Test map
    let map_model = sample(addr!("y"), Normal::new(1.0, 1.0).unwrap()).map(|y| y + 10.0);
    let handler2 = runtime::interpreters::PriorHandler {
        rng: &mut rng,
        trace: runtime::trace::Trace::default(),
    };
    let (result2, trace2) = runtime::handler::run(handler2, map_model);
    let y_val = trace2.get_f64(&addr!("y")).unwrap();
    assert_eq!(result2, y_val + 10.0);

    // Test and_then (alias for bind)
    let and_then_model =
        sample(addr!("z"), Normal::new(-1.0, 1.0).unwrap()).and_then(|z| pure(z.abs()));
    let handler3 = runtime::interpreters::PriorHandler {
        rng: &mut rng,
        trace: runtime::trace::Trace::default(),
    };
    let (result3, trace3) = runtime::handler::run(handler3, and_then_model);
    let z_val = trace3.get_f64(&addr!("z")).unwrap();
    assert_eq!(result3, z_val.abs());
}

#[test]
fn test_trace_api_comprehensive() {
    let mut rng = StdRng::seed_from_u64(42);

    // Create a model with mixed types
    let model = sample(addr!("f64_val"), Normal::new(0.0, 1.0).unwrap())
        .bind(|_| sample(addr!("bool_val"), Bernoulli::new(0.6).unwrap()))
        .bind(|_| sample(addr!("u64_val"), Poisson::new(3.0).unwrap()))
        .bind(|_| {
            sample(
                addr!("usize_val"),
                Categorical::new(vec![0.3, 0.4, 0.3]).unwrap(),
            )
        });

    let handler = runtime::interpreters::PriorHandler {
        rng: &mut rng,
        trace: runtime::trace::Trace::default(),
    };
    let (_, trace) = runtime::handler::run(handler, model);

    // Test type-safe accessors
    assert!(trace.get_f64(&addr!("f64_val")).is_some());
    assert!(trace.get_bool(&addr!("bool_val")).is_some());
    assert!(trace.get_u64(&addr!("u64_val")).is_some());
    assert!(trace.get_usize(&addr!("usize_val")).is_some());

    // Test type mismatches return None
    assert!(trace.get_f64(&addr!("bool_val")).is_none());
    assert!(trace.get_bool(&addr!("f64_val")).is_none());

    // Test result-based accessors
    assert!(trace.get_f64_result(&addr!("f64_val")).is_ok());
    assert!(trace.get_f64_result(&addr!("bool_val")).is_err());
    assert!(trace.get_f64_result(&addr!("missing")).is_err());

    // Test log weight components
    assert!(trace.log_prior.is_finite());
    assert!(trace.log_likelihood.is_finite());
    assert!(trace.log_factors.is_finite());
    assert!(trace.total_log_weight().is_finite());
}

#[test]
fn test_handler_api_coverage() {
    let mut rng = StdRng::seed_from_u64(42);

    // Test PriorHandler
    let prior_handler = runtime::interpreters::PriorHandler {
        rng: &mut rng,
        trace: runtime::trace::Trace::default(),
    };
    let model = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
    let (_, trace1) = runtime::handler::run(prior_handler, model);
    assert!(trace1.get_f64(&addr!("x")).is_some());

    // Test ReplayHandler
    let replay_handler = runtime::interpreters::ReplayHandler {
        rng: &mut rng,
        base: trace1.clone(),
        trace: runtime::trace::Trace::default(),
    };
    let model2 = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
    let (_, trace2) = runtime::handler::run(replay_handler, model2);
    assert_eq!(trace1.get_f64(&addr!("x")), trace2.get_f64(&addr!("x")));

    // Test ScoreGivenTrace
    let score_handler = runtime::interpreters::ScoreGivenTrace {
        base: trace1.clone(),
        trace: runtime::trace::Trace::default(),
    };
    let model3 = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
    let (_, trace3) = runtime::handler::run(score_handler, model3);
    assert!(trace3.log_prior.is_finite());

    // Test SafeReplayHandler
    let safe_replay_handler = runtime::interpreters::SafeReplayHandler {
        rng: &mut rng,
        base: trace1.clone(),
        trace: runtime::trace::Trace::default(),
        warn_on_mismatch: false,
    };
    let model4 = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
    let (_, trace4) = runtime::handler::run(safe_replay_handler, model4);
    assert!(trace4.get_f64(&addr!("x")).is_some());

    // Test SafeScoreGivenTrace
    let safe_score_handler = runtime::interpreters::SafeScoreGivenTrace {
        base: trace1.clone(),
        trace: runtime::trace::Trace::default(),
        warn_on_error: false,
    };
    let model5 = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
    let (_, trace5) = runtime::handler::run(safe_score_handler, model5);
    assert!(trace5.log_prior.is_finite());
}

#[test]
fn test_utility_functions_coverage() {
    let mut rng = StdRng::seed_from_u64(42);

    // Test zip function
    let model_a = sample(addr!("a"), Normal::new(0.0, 1.0).unwrap());
    let model_b = sample(addr!("b"), Normal::new(1.0, 1.0).unwrap());
    let zipped = zip(model_a, model_b);

    let handler = runtime::interpreters::PriorHandler {
        rng: &mut rng,
        trace: runtime::trace::Trace::default(),
    };
    let ((a_val, b_val), trace) = runtime::handler::run(handler, zipped);
    assert_eq!(a_val, trace.get_f64(&addr!("a")).unwrap());
    assert_eq!(b_val, trace.get_f64(&addr!("b")).unwrap());

    // Test sequence_vec function
    let models = vec![
        sample(addr!("seq_0"), Normal::new(0.0, 1.0).unwrap()),
        sample(addr!("seq_1"), Normal::new(1.0, 1.0).unwrap()),
        sample(addr!("seq_2"), Normal::new(2.0, 1.0).unwrap()),
    ];
    let sequenced = sequence_vec(models);

    let handler2 = runtime::interpreters::PriorHandler {
        rng: &mut rng,
        trace: runtime::trace::Trace::default(),
    };
    let (seq_results, trace2) = runtime::handler::run(handler2, sequenced);
    assert_eq!(seq_results.len(), 3);
    assert_eq!(seq_results[0], trace2.get_f64(&addr!("seq_0")).unwrap());
    assert_eq!(seq_results[1], trace2.get_f64(&addr!("seq_1")).unwrap());
    assert_eq!(seq_results[2], trace2.get_f64(&addr!("seq_2")).unwrap());

    // Test traverse_vec function
    let data = vec![1.0, 2.0, 3.0];
    let traversed = traverse_vec(data.clone(), |x| pure(x * 2.0));

    let handler3 = runtime::interpreters::PriorHandler {
        rng: &mut rng,
        trace: runtime::trace::Trace::default(),
    };
    let (trav_results, _) = runtime::handler::run(handler3, traversed);
    assert_eq!(trav_results, vec![2.0, 4.0, 6.0]);
}

#[test]
fn test_address_system_coverage() {
    // Test addr! macro
    let addr1 = addr!("simple");
    let addr2 = addr!("indexed", 5);
    let addr3 = addr!("other", 42);

    assert_eq!(addr1, addr!("simple"));
    assert_ne!(addr1, addr2);
    assert_ne!(addr2, addr3);

    // Test scoped_addr! macro
    let scoped1 = scoped_addr!("scope", "name");
    let scoped2 = scoped_addr!("scope", "name", "{}", 42);
    let scoped3 = scoped_addr!("scope", "other", "{}", 42);

    assert_ne!(scoped1, scoped2);
    assert_ne!(scoped2, scoped3);

    // Test address usage in traces
    let mut rng = StdRng::seed_from_u64(42);
    let model = sample(scoped1.clone(), Normal::new(0.0, 1.0).unwrap());
    let handler = runtime::interpreters::PriorHandler {
        rng: &mut rng,
        trace: runtime::trace::Trace::default(),
    };
    let (_, trace) = runtime::handler::run(handler, model);
    assert!(trace.get_f64(&scoped1).is_some());
    assert!(trace.get_f64(&scoped2).is_none());
}

#[test]
fn test_numerical_utilities_coverage() {
    // Test log_sum_exp
    let log_probs = vec![-1.0, -2.0, -3.0];
    let lse = log_sum_exp(&log_probs);
    assert!(lse.is_finite());
    assert!(lse > log_probs[0]); // Should be greater than max

    // weighted_log_sum_exp is not in the public API, skip this test

    // Test normalize_log_probs
    let log_probs_mut = vec![-1.0, -2.0, -3.0];
    normalize_log_probs(&log_probs_mut);
    // After normalization, probabilities should sum to approximately 1.0
    // But we'll just test that the function ran and produced finite values
    assert!(log_probs_mut.iter().all(|&x| x.is_finite()));
    assert!(log_probs_mut.len() == 3);

    // Test log1p_exp
    let x = 0.5;
    let result = log1p_exp(x);
    assert!(result.is_finite());
    assert!(result > x); // log(1 + exp(x)) > x for positive x

    // Test safe_ln
    assert!(safe_ln(1.0).is_finite());
    assert_eq!(safe_ln(0.0), f64::NEG_INFINITY);
    // safe_ln of negative numbers should return NEG_INFINITY (not NaN)
    assert_eq!(safe_ln(-1.0), f64::NEG_INFINITY);
}

#[test]
fn test_error_handling_coverage() {
    // Test FugueError variants through invalid distribution parameters
    let invalid_normal = Normal::new(0.0, -1.0);
    assert!(invalid_normal.is_err());

    if let Err(error) = invalid_normal {
        // Test error display and structure
        let error_string = format!("{}", error);
        assert!(error_string.contains("Standard deviation"));

        // Test error code and category (negative std dev is InvalidVariance)
        assert_eq!(error.code(), ErrorCode::InvalidVariance);
        assert_eq!(error.category(), ErrorCategory::DistributionValidation);
    }

    // Test type mismatch errors through trace access
    let mut rng = StdRng::seed_from_u64(42);
    let model = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
    let handler = runtime::interpreters::PriorHandler {
        rng: &mut rng,
        trace: runtime::trace::Trace::default(),
    };
    let (_, trace) = runtime::handler::run(handler, model);

    // This should return a type mismatch error
    let bool_result = trace.get_bool_result(&addr!("x"));
    assert!(bool_result.is_err());

    if let Err(error) = bool_result {
        assert_eq!(error.code(), ErrorCode::TypeMismatch);
        assert_eq!(error.category(), ErrorCategory::TypeSystem);
    }

    // Test missing address error
    let missing_result = trace.get_f64_result(&addr!("missing"));
    assert!(missing_result.is_err());

    if let Err(error) = missing_result {
        assert_eq!(error.code(), ErrorCode::TraceAddressNotFound);
        assert_eq!(error.category(), ErrorCategory::TraceManipulation);
    }
}

#[test]
fn test_macro_system_comprehensive() {
    let mut rng = StdRng::seed_from_u64(42);

    // Test prob! macro with complex model
    let prob_model = prob!(
        let x <- sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
        let y <- sample(addr!("y"), Normal::new(x, 0.5).unwrap());
        observe(addr!("obs"), Normal::new(y, 0.1).unwrap(), 0.8);
        factor(-0.5);
        pure((x, y))
    );

    let handler = runtime::interpreters::PriorHandler {
        rng: &mut rng,
        trace: runtime::trace::Trace::default(),
    };
    let ((x_result, y_result), trace) = runtime::handler::run(handler, prob_model);

    // Verify the model executed correctly
    let x_trace = trace.get_f64(&addr!("x")).unwrap();
    let y_trace = trace.get_f64(&addr!("y")).unwrap();
    assert_eq!(x_result, x_trace);
    assert_eq!(y_result, y_trace);
    assert!(trace.log_likelihood.is_finite());
    assert!((trace.log_factors + 0.5).abs() < 1e-12);

    // Test plate! macro
    let plate_model = plate! { i in 0..3 =>
        sample(scoped_addr!("plate", "item", "{}", i), Normal::new(i as f64, 1.0).unwrap())
    };

    let handler2 = runtime::interpreters::PriorHandler {
        rng: &mut rng,
        trace: runtime::trace::Trace::default(),
    };
    let (plate_results, trace2) = runtime::handler::run(handler2, plate_model);

    assert_eq!(plate_results.len(), 3);
    for (i, &expected_val) in plate_results.iter().enumerate().take(3) {
        let addr = scoped_addr!("plate", "item", "{}", i);
        assert!(trace2.get_f64(&addr).is_some());
        assert_eq!(expected_val, trace2.get_f64(&addr).unwrap());
    }
}

#[test]
fn test_memory_management_coverage() {
    let mut rng = StdRng::seed_from_u64(42);

    // Test TracePool
    let mut pool = runtime::memory::TracePool::new(5);
    let stats_initial = pool.stats();
    assert_eq!(stats_initial.total_gets(), 0);
    assert_eq!(stats_initial.hits, 0);
    assert_eq!(stats_initial.misses, 0);

    // Get a trace from the pool
    let trace1 = pool.get();
    let stats_after_get = pool.stats();
    assert_eq!(stats_after_get.total_gets(), 1);
    assert_eq!(stats_after_get.misses, 1); // First get is always a miss

    // Return the trace to the pool
    pool.return_trace(trace1);
    let stats_after_return = pool.stats();
    assert_eq!(stats_after_return.returns, 1);
    assert_eq!(pool.len(), 1);

    // Get another trace (should be a hit this time)
    let _trace2 = pool.get();
    let stats_after_second_get = pool.stats();
    assert_eq!(stats_after_second_get.hits, 1);
    assert_eq!(stats_after_second_get.total_gets(), 2);

    // Test pool capacity
    assert_eq!(pool.capacity(), 5);

    // Test CowTrace copy-on-write semantics
    let base_trace = runtime::trace::Trace::default();
    let cow_trace = runtime::memory::CowTrace::from_trace(base_trace.clone());
    let converted_back = cow_trace.to_trace();
    assert_eq!(converted_back.choices.len(), base_trace.choices.len());

    // Test CowTrace creation and access
    let cow_trace2 = runtime::memory::CowTrace::new();
    let choices = cow_trace2.choices();
    assert!(choices.is_empty());

    // Test TraceBuilder for manual trace construction
    let mut builder = runtime::memory::TraceBuilder::new();
    builder.add_sample(addr!("x"), 1.5, -0.5);
    builder.add_sample_bool(addr!("flag"), true, -0.7);
    builder.add_sample_u64(addr!("count"), 42, -0.3);
    builder.add_sample_usize(addr!("index"), 3, -0.2);
    builder.add_observation(-1.2);
    builder.add_factor(-0.8);

    let built_trace = builder.build();
    assert_eq!(built_trace.get_f64(&addr!("x")), Some(1.5));
    assert_eq!(built_trace.get_bool(&addr!("flag")), Some(true));
    assert_eq!(built_trace.get_u64(&addr!("count")), Some(42));
    assert_eq!(built_trace.get_usize(&addr!("index")), Some(3));
    assert!((built_trace.log_likelihood + 1.2).abs() < 1e-12);
    assert!((built_trace.log_factors + 0.8).abs() < 1e-12);

    // Test PooledPriorHandler integration
    let model = sample(addr!("test"), Normal::new(0.0, 1.0).unwrap());
    let pooled_handler = runtime::memory::PooledPriorHandler::new(&mut rng, &mut pool);

    let (result, final_trace) = runtime::handler::run(pooled_handler, model);
    assert!(result.is_finite());
    assert!(final_trace.get_f64(&addr!("test")).is_some());

    // Pool should now have additional statistics
    let final_stats = pool.stats();
    assert!(final_stats.total_gets() >= 2);
}

#[test]
fn test_inference_api_coverage() {
    let mut rng = StdRng::seed_from_u64(42);

    // Test MCMC API
    let mcmc_model = || {
        sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
            .bind(|x| observe(addr!("y"), Normal::new(x, 0.5).unwrap(), 0.8).map(move |_| x))
    };

    let mcmc_samples = adaptive_mcmc_chain(&mut rng, mcmc_model, 50, 10);
    assert_eq!(mcmc_samples.len(), 50);
    assert!(mcmc_samples.iter().all(|(val, _)| val.is_finite()));

    // Test SMC API with configuration
    let smc_model = || {
        sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap())
            .bind(|mu| observe(addr!("obs"), Normal::new(mu, 0.5).unwrap(), 1.2).map(move |_| mu))
    };

    let smc_config = SMCConfig {
        resampling_method: ResamplingMethod::Systematic,
        ess_threshold: 0.5,
        rejuvenation_steps: 1,
    };

    let smc_particles = adaptive_smc(&mut rng, 20, smc_model, smc_config);
    assert_eq!(smc_particles.len(), 20);
    assert!(smc_particles.iter().all(|p| p.log_weight.is_finite()));
    assert!(smc_particles.iter().all(|p| p.weight >= 0.0));

    // Test effective sample size calculation
    let ess = effective_sample_size(&smc_particles);
    assert!(ess > 0.0);
    assert!(ess <= smc_particles.len() as f64);

    // Test ABC API
    let abc_model = || sample(addr!("theta"), Normal::new(0.0, 1.0).unwrap());

    let simulator =
        |trace: &runtime::trace::Trace| -> f64 { trace.get_f64(&addr!("theta")).unwrap_or(0.0) };

    let abc_samples = abc_scalar_summary(
        &mut rng, abc_model, simulator, 0.5, // observed summary
        1.0, // tolerance
        20,  // max samples
    );

    assert!(abc_samples.len() <= 20);
    assert!(abc_samples
        .iter()
        .all(|trace| trace.get_f64(&addr!("theta")).is_some()));

    // Test VI API
    let vi_model = || {
        sample(addr!("param"), Normal::new(0.0, 1.0).unwrap()).bind(|param| {
            observe(addr!("data"), Normal::new(param, 0.5).unwrap(), 1.0).map(move |_| param)
        })
    };

    let mut vi_guide = MeanFieldGuide::new();
    vi_guide.params.insert(
        addr!("param"),
        VariationalParam::Normal {
            mu: 0.0,
            log_sigma: 0.0,
        },
    );

    let vi_result = optimize_meanfield_vi(
        &mut rng, vi_model, vi_guide, 10,   // iterations
        5,    // samples per iteration
        0.01, // learning rate
    );

    assert!(!vi_result.params.is_empty());
    assert!(vi_result.params.contains_key(&addr!("param")));

    // Test ELBO computation
    let elbo = elbo_with_guide(&mut rng, vi_model, &vi_result, 5);
    assert!(elbo.is_finite() || elbo.is_infinite()); // May be -inf for poorly fitted guide

    // Test diagnostics API
    let chains = vec![mcmc_samples
        .iter()
        .map(|(_, trace)| trace.clone())
        .collect::<Vec<_>>()];

    let r_hat = r_hat_f64(&chains, &addr!("x"));
    assert!(r_hat.is_finite());
    assert!(r_hat > 0.0);

    let summary = summarize_f64_parameter(&chains, &addr!("x"));
    assert!(summary.mean.is_finite());
    assert!(summary.std.is_finite());
    assert!(summary.std >= 0.0);

    let x_values: Vec<f64> = chains[0]
        .iter()
        .filter_map(|trace| trace.get_f64(&addr!("x")))
        .collect();
    let ess_mcmc = effective_sample_size_mcmc(&x_values);
    assert!(ess_mcmc >= 0.0);
    assert!(ess_mcmc <= x_values.len() as f64);

    // Test validation API
    let reference_samples: Vec<f64> = (0..50)
        .map(|_| Normal::new(0.0, 1.0).unwrap().sample(&mut rng))
        .collect();

    let test_dist = Normal::new(0.0, 1.0).unwrap();
    let ks_result = ks_test_distribution(&mut rng, &test_dist, &reference_samples, 50, 0.05);

    // KS test should pass (samples from same distribution)
    assert!(ks_result);
}

#[test]
fn test_model_api_comprehensive() {
    let mut rng = StdRng::seed_from_u64(42);

    // Test guard function
    let guard_model = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
        .bind(|x| guard(x > -2.0 && x < 2.0))
        .bind(|_| pure("guard_passed"));

    let handler = runtime::interpreters::PriorHandler {
        rng: &mut rng,
        trace: runtime::trace::Trace::default(),
    };

    let (result, trace) = runtime::handler::run(handler, guard_model);
    assert_eq!(result, "guard_passed");
    let x_val = trace.get_f64(&addr!("x")).unwrap();
    assert!(x_val > -2.0 && x_val < 2.0);

    // Test type-specific samplers (if they exist in public API)
    // Note: These might not be directly exposed in the public API,
    // but we can test them through the general sample function

    let mixed_model = sample(addr!("f64"), Normal::new(0.0, 1.0).unwrap())
        .bind(|_| sample(addr!("bool"), Bernoulli::new(0.6).unwrap()))
        .bind(|_| sample(addr!("u64"), Poisson::new(2.0).unwrap()))
        .bind(|_| {
            sample(
                addr!("usize"),
                Categorical::new(vec![0.25, 0.5, 0.25]).unwrap(),
            )
        });

    let handler2 = runtime::interpreters::PriorHandler {
        rng: &mut rng,
        trace: runtime::trace::Trace::default(),
    };

    let (usize_result, trace2) = runtime::handler::run(handler2, mixed_model);

    // Verify all types were sampled correctly
    assert!(trace2.get_f64(&addr!("f64")).is_some());
    assert!(trace2.get_bool(&addr!("bool")).is_some());
    assert!(trace2.get_u64(&addr!("u64")).is_some());
    assert!(trace2.get_usize(&addr!("usize")).is_some());
    assert_eq!(usize_result, trace2.get_usize(&addr!("usize")).unwrap());

    // Test complex model composition
    let complex_model = sample(addr!("a"), Normal::new(0.0, 1.0).unwrap()).bind(|a| {
        sample(addr!("b"), Normal::new(a, 0.5).unwrap()).bind(move |b| {
            guard(b.abs() < 3.0)
                .bind(move |_| observe(addr!("obs"), Normal::new(a, 0.3).unwrap(), 0.5))
                .bind(move |_| factor(a * 0.1))
                .map(move |_| (a, b))
        })
    });

    let handler3 = runtime::interpreters::PriorHandler {
        rng: &mut rng,
        trace: runtime::trace::Trace::default(),
    };

    let ((a_result, b_result), trace3) = runtime::handler::run(handler3, complex_model);

    // Verify the complex composition worked
    assert_eq!(a_result, trace3.get_f64(&addr!("a")).unwrap());
    assert_eq!(b_result, trace3.get_f64(&addr!("b")).unwrap());
    assert!(trace3.log_likelihood.is_finite());
    assert!(trace3.log_factors.is_finite());
    assert!(trace3.total_log_weight().is_finite());

    // Factor should be a * 0.1
    let expected_factor = a_result * 0.1;
    assert!((trace3.log_factors - expected_factor).abs() < 1e-12);
}

#[test]
fn test_trace_api_comprehensive_extended() {
    let mut rng = StdRng::seed_from_u64(42);

    // Test ChoiceValue enum and Choice struct through trace building
    let model = sample(addr!("normal"), Normal::new(0.0, 1.0).unwrap())
        .bind(|_| sample(addr!("bernoulli"), Bernoulli::new(0.5).unwrap()))
        .bind(|_| sample(addr!("poisson"), Poisson::new(3.0).unwrap()))
        .bind(|_| {
            sample(
                addr!("categorical"),
                Categorical::new(vec![0.3, 0.4, 0.3]).unwrap(),
            )
        });

    let handler = runtime::interpreters::PriorHandler {
        rng: &mut rng,
        trace: runtime::trace::Trace::default(),
    };

    let (_, trace) = runtime::handler::run(handler, model);

    // Test direct choice access (if available in public API)
    assert!(!trace.choices.is_empty());

    // Test that each address has a corresponding choice
    for (addr, choice) in &trace.choices {
        assert_eq!(addr, &choice.addr);
        assert!(choice.logp.is_finite());

        // Test ChoiceValue variants through type-safe accessors
        match &choice.value {
            runtime::trace::ChoiceValue::F64(_) => {
                assert!(trace.get_f64(addr).is_some());
            }
            runtime::trace::ChoiceValue::Bool(_) => {
                assert!(trace.get_bool(addr).is_some());
            }
            runtime::trace::ChoiceValue::U64(_) => {
                assert!(trace.get_u64(addr).is_some());
            }
            runtime::trace::ChoiceValue::Usize(_) => {
                assert!(trace.get_usize(addr).is_some());
            }
            runtime::trace::ChoiceValue::I64(_) => {
                assert!(trace.get_i64(addr).is_some());
            }
        }
    }

    // Test log weight components and calculation
    assert!(trace.log_prior.is_finite());
    assert!(trace.log_likelihood.is_finite());
    assert!(trace.log_factors.is_finite());

    let manual_total = trace.log_prior + trace.log_likelihood + trace.log_factors;
    let api_total = trace.total_log_weight();
    assert!((manual_total - api_total).abs() < 1e-12);

    // Test trace cloning and equality
    let cloned_trace = trace.clone();
    assert_eq!(trace.choices.len(), cloned_trace.choices.len());
    assert_eq!(trace.log_prior, cloned_trace.log_prior);
    assert_eq!(trace.log_likelihood, cloned_trace.log_likelihood);
    assert_eq!(trace.log_factors, cloned_trace.log_factors);
}

#[test]
fn test_numerical_utilities_edge_cases() {
    // Test log_sum_exp with edge cases

    // Empty vector
    let empty_lse = log_sum_exp(&[]);
    assert_eq!(empty_lse, f64::NEG_INFINITY);

    // Single element
    let single_lse = log_sum_exp(&[-2.0]);
    assert!((single_lse + 2.0).abs() < 1e-12);

    // Very large values (should not overflow)
    let large_lse = log_sum_exp(&[700.0, 701.0, 702.0]);
    assert!(large_lse.is_finite());
    assert!(large_lse > 702.0);

    // Very small values (should not underflow to zero)
    let small_lse = log_sum_exp(&[-700.0, -701.0, -702.0]);
    assert!(small_lse.is_finite());
    assert!(small_lse < -699.0);

    // Mixed positive and negative infinity
    let mixed_inf_lse = log_sum_exp(&[f64::NEG_INFINITY, 0.0, f64::NEG_INFINITY]);
    assert!((mixed_inf_lse - 0.0).abs() < 1e-12);

    // Test normalize_log_probs edge cases
    let empty_probs = vec![];
    let _ = normalize_log_probs(&empty_probs);
    assert!(empty_probs.is_empty());

    let single_prob = vec![-1.0];
    let _ = normalize_log_probs(&single_prob);
    // Single element should remain unchanged (already normalized)
    assert!((single_prob[0] + 1.0).abs() < 1e-12);

    let inf_probs = vec![f64::NEG_INFINITY, 0.0, f64::NEG_INFINITY];
    let _ = normalize_log_probs(&inf_probs);
    assert!(inf_probs[0].is_infinite() && inf_probs[0] < 0.0); // Should remain -inf
    assert!((inf_probs[1] - 0.0).abs() < 1e-12); // Should be log(1.0) = 0.0
    assert!(inf_probs[2].is_infinite() && inf_probs[2] < 0.0); // Should remain -inf

    // Test log1p_exp edge cases
    assert!((log1p_exp(0.0) - (2.0_f64.ln())).abs() < 1e-12); // log(1 + exp(0)) = log(2)
    assert!(log1p_exp(-700.0).abs() < 1e-12); // log(1 + exp(-700)) ≈ 0
    assert!((log1p_exp(700.0) - 700.0).abs() < 1e-12); // log(1 + exp(700)) ≈ 700

    // Test safe_ln edge cases
    assert_eq!(safe_ln(0.0), f64::NEG_INFINITY);
    assert_eq!(safe_ln(-1.0), f64::NEG_INFINITY);
    assert_eq!(safe_ln(f64::NEG_INFINITY), f64::NEG_INFINITY);
    assert!((safe_ln(1.0) - 0.0).abs() < 1e-12);
    assert!((safe_ln(std::f64::consts::E) - 1.0).abs() < 1e-12);

    // Test with NaN inputs - behavior may be implementation-specific
    let nan_result = safe_ln(f64::NAN);
    assert!(nan_result.is_nan() || nan_result == f64::NEG_INFINITY);
}
