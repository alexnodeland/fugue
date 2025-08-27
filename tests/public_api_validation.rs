//! # Public API Validation Integration Tests
//!
//! This module contains integration tests specifically focused on validating
//! that the public API works correctly and provides the expected interface.
//! These tests ensure API stability and usability.
//!
//! ## Test Categories
//!
//! ### 1. API Contract Validation (`test_api_contract_*`)
//! - Function signatures match documented interfaces
//! - Return types are as expected
//! - Error handling follows documented patterns
//! - Optional parameters work correctly
//!
//! ### 2. Public Export Validation (`test_public_exports_*`)
//! - All intended public items are accessible via `fugue::*`
//! - No internal implementation details are exposed
//! - Re-exports work correctly
//! - Module structure is as documented
//!
//! ### 3. API Consistency Validation (`test_consistency_*`)
//! - Similar functions have consistent interfaces
//! - Naming conventions are followed consistently
//! - Error types and messages are consistent
//! - Type safety is enforced consistently
//!
//! ### 4. Backwards Compatibility (`test_compatibility_*`)
//! - Existing code patterns continue to work
//! - Deprecation warnings are appropriate
//! - Migration paths are clear
//! - Breaking changes are documented
//!
//! ### 5. Ergonomics Validation (`test_ergonomics_*`)
//! - Common use cases are straightforward
//! - Type inference works as expected
//! - Error messages are helpful
//! - API is discoverable and intuitive
//!
//! ## Implementation Strategy
//!
//! ### Systematic Coverage
//! - **Enumerate Public API**: List all public exports from `lib.rs`
//! - **Test Each Export**: Validate functionality and interface
//! - **Cross-Reference Docs**: Ensure documentation matches implementation
//! - **User Perspective**: Test from user's point of view
//!
//! ### Validation Patterns
//! - **Interface Testing**: Verify function signatures and behavior
//! - **Integration Testing**: Test components working together
//! - **Error Testing**: Validate error conditions and messages
//! - **Edge Case Testing**: Test boundary conditions and limits
//!
//! ### Quality Metrics
//! - **Completeness**: All public API is tested
//! - **Correctness**: Behavior matches documentation
//! - **Usability**: Common patterns are easy to use
//! - **Robustness**: Error conditions are handled gracefully
//!
//! ## Public API Surface (from src/lib.rs)
//!
//! ### Core Types and Functions
//! ```rust
//! // Address system
//! pub use core::address::Address;
//! // addr! macro exported via #[macro_export]
//!
//! // Distributions
//! pub use core::distribution::{
//!     Bernoulli, Beta, Binomial, Categorical, Distribution, Exponential, 
//!     Gamma, LogNormal, Normal, Poisson, Uniform,
//! };
//!
//! // Model system
//! pub use core::model::{
//!     factor, guard, observe, pure, sample, sample_bool, sample_f64, 
//!     sample_u64, sample_usize, sequence_vec, traverse_vec, zip, 
//!     Model, ModelExt, SampleType,
//! };
//!
//! // Runtime system
//! pub use runtime::handler::Handler;
//! pub use runtime::interpreters::{
//!     PriorHandler, ReplayHandler, SafeReplayHandler, 
//!     SafeScoreGivenTrace, ScoreGivenTrace,
//! };
//! pub use runtime::trace::{Choice, ChoiceValue, Trace};
//!
//! // Inference algorithms
//! pub use inference::abc::{abc_rejection, abc_scalar_summary, abc_smc, ...};
//! pub use inference::diagnostics::{r_hat_f64, summarize_f64_parameter, ...};
//! pub use inference::mh::{adaptive_mcmc_chain, adaptive_single_site_mh};
//! pub use inference::smc::{adaptive_smc, effective_sample_size, ...};
//! pub use inference::validation::{ks_test_distribution, ...};
//! pub use inference::vi::{elbo_with_guide, optimize_meanfield_vi, ...};
//!
//! // Utilities
//! pub use core::numerical::{log1p_exp, log_sum_exp, normalize_log_probs, safe_ln};
//! pub use error::{ErrorCategory, ErrorCode, ErrorContext, FugueError, ...};
//! pub use runtime::memory::{CowTrace, PooledPriorHandler, TraceBuilder, TracePool};
//! ```
//!
//! ## Testing Guidelines
//!
//! ### Test Organization
//! - Group tests by API area (distributions, models, inference, etc.)
//! - Use descriptive test names that indicate what's being validated
//! - Include both positive and negative test cases
//! - Test integration between different API components
//!
//! ### Test Patterns
//! - **Construction Tests**: Verify objects can be created correctly
//! - **Method Tests**: Verify methods work as documented
//! - **Integration Tests**: Verify components work together
//! - **Error Tests**: Verify error conditions are handled correctly
//!
//! ### Example Test Structure
//! ```rust
//! #[test]
//! fn test_distribution_normal_public_interface() {
//!     // Test constructor
//!     let dist = Normal::new(0.0, 1.0).expect("Valid parameters");
//!     
//!     // Test sampling
//!     let mut rng = StdRng::seed_from_u64(42);
//!     let sample = dist.sample(&mut rng);
//!     assert!(sample.is_finite());
//!     
//!     // Test log_prob
//!     let log_p = dist.log_prob(&sample);
//!     assert!(log_p.is_finite());
//!     
//!     // Test validation
//!     assert!(dist.validate().is_ok());
//!     
//!     // Test error conditions
//!     assert!(Normal::new(0.0, -1.0).is_err());
//! }
//! ```

use fugue::*;
use rand::{rngs::StdRng, SeedableRng};

#[test]
fn test_api_contract_distribution_interfaces() {
    let mut rng = StdRng::seed_from_u64(42);
    
    // Test that all distributions implement the Distribution trait correctly
    let normal = Normal::new(0.0, 1.0).expect("Valid Normal parameters");
    let sample_n = normal.sample(&mut rng);
    let log_prob_n = normal.log_prob(&sample_n);
    assert!(sample_n.is_finite());
    assert!(log_prob_n.is_finite());
    
    let bernoulli = Bernoulli::new(0.5).expect("Valid Bernoulli parameters");
    let sample_b = bernoulli.sample(&mut rng);
    let log_prob_b = bernoulli.log_prob(&sample_b);
    assert!(sample_b == true || sample_b == false);
    assert!(log_prob_b.is_finite());
    
    let poisson = Poisson::new(2.0).expect("Valid Poisson parameters");
    let sample_p = poisson.sample(&mut rng);
    let log_prob_p = poisson.log_prob(&sample_p);
    assert!(sample_p >= 0);
    assert!(log_prob_p.is_finite());
    
    // Test error conditions return appropriate error types
    let invalid_normal = Normal::new(0.0, -1.0);
    assert!(invalid_normal.is_err());
    match invalid_normal {
        Err(error) => {
            assert_eq!(error.category(), ErrorCategory::DistributionValidation);
            assert_eq!(error.code(), ErrorCode::InvalidVariance);
        }
        Ok(_) => panic!("Expected error for negative variance"),
    }
}

#[test]
fn test_api_contract_model_composition() {
    let mut rng = StdRng::seed_from_u64(42);
    
    // Test that model composition functions have consistent interfaces
    let model1 = pure(42.0);
    let model2 = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
    let model3 = observe(addr!("y"), Normal::new(0.0, 1.0).unwrap(), 0.5);
    let model4 = factor(-1.0);
    
    // Test that all models can be run with handlers
    let handler1 = runtime::interpreters::PriorHandler {
        rng: &mut rng,
        trace: runtime::trace::Trace::default(),
    };
    let (result1, _) = runtime::handler::run(handler1, model1);
    assert_eq!(result1, 42.0);
    
    let handler2 = runtime::interpreters::PriorHandler {
        rng: &mut rng,
        trace: runtime::trace::Trace::default(),
    };
    let (result2, trace2) = runtime::handler::run(handler2, model2);
    assert!(result2.is_finite());
    assert!(trace2.get_f64(&addr!("x")).is_some());
    
    let handler3 = runtime::interpreters::PriorHandler {
        rng: &mut rng,
        trace: runtime::trace::Trace::default(),
    };
    let (_, trace3) = runtime::handler::run(handler3, model3);
    assert!(trace3.log_likelihood.is_finite());
    
    let handler4 = runtime::interpreters::PriorHandler {
        rng: &mut rng,
        trace: runtime::trace::Trace::default(),
    };
    let (_, trace4) = runtime::handler::run(handler4, model4);
    assert!((trace4.log_factors + 1.0).abs() < 1e-12);
}

#[test]
fn test_public_exports_accessibility() {
    // Test that all major public exports are accessible via fugue::*
    
    // Address system
    let _addr = addr!("test");
    let _address = Address("test".to_string());
    
    // Distributions - test construction to verify exports
    let _normal = Normal::new(0.0, 1.0).unwrap();
    let _bernoulli = Bernoulli::new(0.5).unwrap();
    let _uniform = Uniform::new(0.0, 1.0).unwrap();
    let _exponential = Exponential::new(1.0).unwrap();
    let _beta = Beta::new(1.0, 1.0).unwrap();
    let _gamma = Gamma::new(1.0, 1.0).unwrap();
    let _lognormal = LogNormal::new(0.0, 1.0).unwrap();
    let _poisson = Poisson::new(1.0).unwrap();
    let _binomial = Binomial::new(10, 0.5).unwrap();
    let _categorical = Categorical::new(vec![0.5, 0.5]).unwrap();
    
    // Model system functions
    let _pure_model = pure(1.0);
    let _sample_model = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
    let _observe_model = observe(addr!("y"), Normal::new(0.0, 1.0).unwrap(), 0.0);
    let _factor_model = factor(-0.5);
    
    // Model utilities
    let models = vec![pure(1.0), pure(2.0)];
    let _sequence_model = sequence_vec(models);
    let _zip_model = zip(pure(1.0), pure(2.0));
    let _traverse_model = traverse_vec(vec![1.0, 2.0], |x| pure(x * 2.0));
    
    // Runtime types
    let _trace = runtime::trace::Trace::default();
    
    // Error types
    let _error_code = ErrorCode::InvalidVariance;
    let _error_category = ErrorCategory::DistributionValidation;
    
    // Numerical utilities
    let _lse = log_sum_exp(&vec![-1.0, -2.0]);
    let _log1p = log1p_exp(0.5);
    let _safe = safe_ln(1.0);
    
    // All exports are accessible - test passes if it compiles
    assert!(true);
}

#[test]
fn test_api_consistency_error_handling() {
    // Test that error handling is consistent across the API
    
    // Distribution validation errors should have consistent structure
    let errors = vec![
        Normal::new(0.0, -1.0).unwrap_err(),
        Bernoulli::new(1.5).unwrap_err(),
        Uniform::new(1.0, 0.0).unwrap_err(),
        Exponential::new(-1.0).unwrap_err(),
    ];
    
    for error in errors {
        // All should be distribution validation errors
        assert_eq!(error.category(), ErrorCategory::DistributionValidation);
        
        // All should have meaningful error messages
        let message = format!("{}", error);
        assert!(!message.is_empty());
        assert!(message.len() > 10); // Should be descriptive
        
        // All should have specific error codes
        let code = error.code();
        assert!(matches!(code, 
            ErrorCode::InvalidMean | ErrorCode::InvalidVariance | 
            ErrorCode::InvalidProbability | ErrorCode::InvalidRange |
            ErrorCode::InvalidShape | ErrorCode::InvalidRate
        ));
    }
}

#[test]
fn test_api_consistency_naming_conventions() {
    // Test that naming conventions are followed consistently
    
    // Distribution constructors should all be named "new"
    let _n1 = Normal::new(0.0, 1.0);
    let _n2 = Bernoulli::new(0.5);
    let _n3 = Uniform::new(0.0, 1.0);
    let _n4 = Exponential::new(1.0);
    let _n5 = Beta::new(1.0, 1.0);
    let _n6 = Gamma::new(1.0, 1.0);
    let _n7 = LogNormal::new(0.0, 1.0);
    let _n8 = Poisson::new(1.0);
    let _n9 = Binomial::new(10, 0.5);
    let _n10 = Categorical::new(vec![0.5, 0.5]);
    
    // Model functions should have consistent naming
    let _pure = pure(1.0);
    let _sample = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
    let _observe = observe(addr!("y"), Normal::new(0.0, 1.0).unwrap(), 0.0);
    let _factor = factor(-0.5);
    
    // Trace accessors should have consistent naming patterns
    let trace = runtime::trace::Trace::default();
    let _f64_opt = trace.get_f64(&addr!("x"));
    let _bool_opt = trace.get_bool(&addr!("x"));
    let _u64_opt = trace.get_u64(&addr!("x"));
    let _usize_opt = trace.get_usize(&addr!("x"));
    
    let _f64_res = trace.get_f64_result(&addr!("x"));
    let _bool_res = trace.get_bool_result(&addr!("x"));
    let _u64_res = trace.get_u64_result(&addr!("x"));
    let _usize_res = trace.get_usize_result(&addr!("x"));
    
    // Naming is consistent - test passes if it compiles
    assert!(true);
}

#[test]
fn test_ergonomics_type_inference() {
    let mut rng = StdRng::seed_from_u64(42);
    
    // Test that type inference works well for common patterns
    
    // Should infer f64 from Normal distribution
    let model1 = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
    let handler1 = runtime::interpreters::PriorHandler {
        rng: &mut rng,
        trace: runtime::trace::Trace::default(),
    };
    let (result1, _) = runtime::handler::run(handler1, model1);
    let _: f64 = result1; // Should compile without explicit type annotation
    
    // Should infer bool from Bernoulli distribution
    let model2 = sample(addr!("b"), Bernoulli::new(0.5).unwrap());
    let handler2 = runtime::interpreters::PriorHandler {
        rng: &mut rng,
        trace: runtime::trace::Trace::default(),
    };
    let (result2, _) = runtime::handler::run(handler2, model2);
    let _: bool = result2; // Should compile without explicit type annotation
    
    // Should infer u64 from Poisson distribution
    let model3 = sample(addr!("p"), Poisson::new(2.0).unwrap());
    let handler3 = runtime::interpreters::PriorHandler {
        rng: &mut rng,
        trace: runtime::trace::Trace::default(),
    };
    let (result3, _) = runtime::handler::run(handler3, model3);
    let _: u64 = result3; // Should compile without explicit type annotation
    
    // Model composition should preserve types
    let composed = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
        .map(|x| x * 2.0)
        .bind(|x| pure(x + 1.0));
    let handler4 = runtime::interpreters::PriorHandler {
        rng: &mut rng,
        trace: runtime::trace::Trace::default(),
    };
    let (result4, _) = runtime::handler::run(handler4, composed);
    let _: f64 = result4; // Should infer f64 through the composition
    
    assert!(true);
}

#[test]
fn test_ergonomics_common_patterns() {
    let mut rng = StdRng::seed_from_u64(42);
    
    // Test that common usage patterns are ergonomic
    
    // Pattern 1: Simple Bayesian inference
    let bayesian_model = sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap())
        .bind(|mu| observe(addr!("y"), Normal::new(mu, 0.5).unwrap(), 1.2)
             .map(move |_| mu));
    
    let handler = runtime::interpreters::PriorHandler {
        rng: &mut rng,
        trace: runtime::trace::Trace::default(),
    };
    let (mu_sample, trace) = runtime::handler::run(handler, bayesian_model);
    assert!(mu_sample.is_finite());
    assert!(trace.log_likelihood.is_finite());
    
    // Pattern 2: Multiple parameters
    let multi_param = sample(addr!("a"), Normal::new(0.0, 1.0).unwrap())
        .bind(|a| sample(addr!("b"), Normal::new(a, 0.5).unwrap())
             .map(move |b| (a, b)));
    
    let handler2 = runtime::interpreters::PriorHandler {
        rng: &mut rng,
        trace: runtime::trace::Trace::default(),
    };
    let ((a_val, b_val), trace2) = runtime::handler::run(handler2, multi_param);
    assert!(a_val.is_finite());
    assert!(b_val.is_finite());
    assert_eq!(a_val, trace2.get_f64(&addr!("a")).unwrap());
    assert_eq!(b_val, trace2.get_f64(&addr!("b")).unwrap());
    
    // Pattern 3: Vectorized operations
    let vectorized = sequence_vec(vec![
        sample(addr!("x1"), Normal::new(0.0, 1.0).unwrap()),
        sample(addr!("x2"), Normal::new(1.0, 1.0).unwrap()),
        sample(addr!("x3"), Normal::new(2.0, 1.0).unwrap()),
    ]);
    
    let handler3 = runtime::interpreters::PriorHandler {
        rng: &mut rng,
        trace: runtime::trace::Trace::default(),
    };
    let (vec_results, trace3) = runtime::handler::run(handler3, vectorized);
    assert_eq!(vec_results.len(), 3);
    assert!(vec_results.iter().all(|x| x.is_finite()));
    assert!(trace3.get_f64(&addr!("x1")).is_some());
    assert!(trace3.get_f64(&addr!("x2")).is_some());
    assert!(trace3.get_f64(&addr!("x3")).is_some());
}

#[test]
fn test_api_contract_inference_algorithms() {
    let mut rng = StdRng::seed_from_u64(42);
    
    // Test that inference algorithms have consistent interfaces
    
    // MCMC interface
    let model_fn = || sample(addr!("theta"), Normal::new(0.0, 1.0).unwrap());
    let mcmc_samples = adaptive_mcmc_chain(&mut rng, model_fn, 50, 10);
    assert_eq!(mcmc_samples.len(), 50);
    assert!(mcmc_samples.iter().all(|(theta, _)| theta.is_finite()));
    
    // SMC interface  
    let smc_model_fn = || sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap());
    let smc_config = SMCConfig {
        resampling_method: ResamplingMethod::Systematic,
        ess_threshold: 0.5,
        rejuvenation_steps: 0,
    };
    let particles = adaptive_smc(&mut rng, 20, smc_model_fn, smc_config);
    assert_eq!(particles.len(), 20);
    assert!(particles.iter().all(|p| p.log_weight.is_finite()));
    
    // ABC interface
    let abc_model_fn = || sample(addr!("param"), Normal::new(0.0, 1.0).unwrap());
    let simulator = |trace: &runtime::trace::Trace| -> f64 {
        trace.get_f64(&addr!("param")).unwrap_or(0.0)
    };
    let abc_samples = abc_scalar_summary(
        &mut rng,
        abc_model_fn,
        simulator,
        0.0, // observed
        1.0, // tolerance
        20,  // max_samples
    );
    assert!(abc_samples.len() <= 20);
    
    // VI interface
    let vi_model_fn = || sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
    let mut guide = MeanFieldGuide::new();
    guide.params.insert(
        addr!("x"),
        VariationalParam::Normal { mu: 0.0, log_sigma: 0.0 }
    );
    let optimized_guide = optimize_meanfield_vi(
        &mut rng,
        vi_model_fn,
        guide,
        5,   // n_iterations
        10,  // n_samples_per_iter
        0.1, // learning_rate
    );
    assert!(!optimized_guide.params.is_empty());
}

#[test]
fn test_api_contract_memory_management() {
    // Test memory management APIs
    
    // TracePool interface
    let mut pool = TracePool::new(10); // max_size parameter required
    let stats_before = pool.stats();
    assert_eq!(stats_before.total_gets(), 0); // Use total_gets() method
    
    // Get a trace from the pool
    let trace = pool.get();
    let stats_after_get = pool.stats();
    assert_eq!(stats_after_get.misses, 1); // First get is always a miss
    
    // Return the trace
    pool.return_trace(trace); // Method is called return_trace
    let stats_after_return = pool.stats();
    assert_eq!(stats_after_return.returns, 1);
    
    // Pool capacity and length
    assert_eq!(pool.capacity(), 10);
    assert_eq!(pool.len(), 1); // One trace returned
    
    // CowTrace interface (copy-on-write semantics)
    let base_trace = runtime::trace::Trace::default();
    let cow_trace = CowTrace::from_trace(base_trace.clone()); // Use from_trace
    let converted_back = cow_trace.to_trace(); // Use to_trace method
    assert_eq!(converted_back.choices.len(), base_trace.choices.len());
    
    // Test CowTrace creation and choices access
    let cow_trace2 = CowTrace::new();
    let choices = cow_trace2.choices(); // Read-only access
    assert!(choices.is_empty());
    
    // TraceBuilder interface
    let mut builder = TraceBuilder::new();
    builder.add_sample(addr!("test"), 1.0, -0.5); // Use add_sample method
    let built_trace = builder.build();
    assert_eq!(built_trace.get_f64(&addr!("test")), Some(1.0));
    
    // Test other builder methods
    let mut builder2 = TraceBuilder::new();
    builder2.add_sample_bool(addr!("bool_test"), true, -0.7);
    builder2.add_observation(-1.2);
    builder2.add_factor(-0.3);
    let built_trace2 = builder2.build();
    assert_eq!(built_trace2.get_bool(&addr!("bool_test")), Some(true));
    assert!((built_trace2.log_likelihood + 1.2).abs() < 1e-12);
    assert!((built_trace2.log_factors + 0.3).abs() < 1e-12);
}

#[test]
fn test_compatibility_legacy_patterns() {
    // Test that common patterns from earlier versions still work
    let mut rng = StdRng::seed_from_u64(42);
    
    // Legacy pattern 1: Direct handler usage
    let model = sample(addr!("param"), Normal::new(0.0, 1.0).unwrap());
    let handler = runtime::interpreters::PriorHandler {
        rng: &mut rng,
        trace: runtime::trace::Trace::default(),
    };
    let (result, trace) = runtime::handler::run(handler, model);
    
    assert!(result.is_finite());
    assert!(trace.get_f64(&addr!("param")).is_some());
    
    // Legacy pattern 2: Manual trace building (if supported)
    let mut builder = runtime::memory::TraceBuilder::new();
    builder.add_sample(addr!("manual"), 2.5, -1.0);
    let manual_trace = builder.build();
    
    assert_eq!(manual_trace.get_f64(&addr!("manual")), Some(2.5));
    assert!((manual_trace.log_prior + 1.0).abs() < 1e-12);
    
    // Legacy pattern 3: Basic MCMC usage
    let legacy_mcmc_model = || sample(addr!("theta"), Normal::new(0.0, 1.0).unwrap());
    let legacy_samples = adaptive_mcmc_chain(&mut rng, legacy_mcmc_model, 20, 5);
    
    assert_eq!(legacy_samples.len(), 20);
    assert!(legacy_samples.iter().all(|(val, _)| val.is_finite()));
    
    // Legacy pattern 4: Simple model composition
    let legacy_composition = pure(1.0)
        .bind(|x| pure(x + 1.0))
        .map(|x| x * 2.0);
    
    let handler2 = runtime::interpreters::PriorHandler {
        rng: &mut rng,
        trace: runtime::trace::Trace::default(),
    };
    let (legacy_result, _) = runtime::handler::run(handler2, legacy_composition);
    assert_eq!(legacy_result, 4.0); // (1 + 1) * 2
}

#[test]
fn test_compatibility_api_stability() {
    // Test that core API signatures remain stable
    
    // Distribution constructors should maintain their signatures
    let _normal: Normal = Normal::new(0.0, 1.0).unwrap();
    let _bernoulli: Bernoulli = Bernoulli::new(0.5).unwrap();
    let _uniform: Uniform = Uniform::new(0.0, 1.0).unwrap();
    
    // Model functions should maintain their signatures
    let _pure_model: Model<i32> = pure(42);
    let _sample_model: Model<f64> = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
    let _observe_model: Model<()> = observe(addr!("y"), Normal::new(0.0, 1.0).unwrap(), 0.0);
    let _factor_model: Model<()> = factor(-0.5);
    
    // Trace accessors should maintain their signatures
    let trace = runtime::trace::Trace::default();
    let _f64_option: Option<f64> = trace.get_f64(&addr!("test"));
    let _bool_option: Option<bool> = trace.get_bool(&addr!("test"));
    let _f64_result: Result<f64, FugueError> = trace.get_f64_result(&addr!("test"));
    let _bool_result: Result<bool, FugueError> = trace.get_bool_result(&addr!("test"));
    
    // Handler types should be constructible
    let mut rng = StdRng::seed_from_u64(42);
    let _prior_handler = runtime::interpreters::PriorHandler {
        rng: &mut rng,
        trace: runtime::trace::Trace::default(),
    };
    
    // Inference functions should maintain their signatures
    let model_fn = || sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
    let _mcmc_samples: Vec<(f64, runtime::trace::Trace)> = adaptive_mcmc_chain(&mut rng, model_fn, 10, 2);
    
    // Error types should be accessible
    let _error_code: ErrorCode = ErrorCode::InvalidVariance;
    let _error_category: ErrorCategory = ErrorCategory::DistributionValidation;
    
    // API stability confirmed - test passes if it compiles
    assert!(true);
}

#[test]
fn test_api_contract_comprehensive_validation() {
    let mut rng = StdRng::seed_from_u64(42);
    
    // Comprehensive validation of all major API contracts
    
    // 1. All distributions should implement Distribution trait consistently
    let distributions: Vec<Box<dyn Distribution<f64>>> = vec![
        Box::new(Normal::new(0.0, 1.0).unwrap()),
        Box::new(Uniform::new(0.0, 1.0).unwrap()),
        Box::new(Exponential::new(1.0).unwrap()),
        Box::new(Beta::new(1.0, 1.0).unwrap()),
        Box::new(Gamma::new(1.0, 1.0).unwrap()),
        Box::new(LogNormal::new(0.0, 1.0).unwrap()),
    ];
    
    for dist in distributions {
        let sample = dist.sample(&mut rng);
        let log_prob = dist.log_prob(&sample);
        assert!(sample.is_finite());
        assert!(log_prob.is_finite());
    }
    
    // 2. All handler types should work with the same model
    let make_test_model = || sample(addr!("test"), Normal::new(0.0, 1.0).unwrap());
    
    // PriorHandler
    let prior_handler = runtime::interpreters::PriorHandler {
        rng: &mut rng,
        trace: runtime::trace::Trace::default(),
    };
    let (_, base_trace) = runtime::handler::run(prior_handler, make_test_model());
    
    // ReplayHandler
    let replay_handler = runtime::interpreters::ReplayHandler {
        rng: &mut rng,
        base: base_trace.clone(),
        trace: runtime::trace::Trace::default(),
    };
    let (_, replay_trace) = runtime::handler::run(replay_handler, make_test_model());
    assert_eq!(base_trace.get_f64(&addr!("test")), replay_trace.get_f64(&addr!("test")));
    
    // SafeReplayHandler
    let safe_replay_handler = runtime::interpreters::SafeReplayHandler {
        rng: &mut rng,
        base: base_trace.clone(),
        trace: runtime::trace::Trace::default(),
        warn_on_mismatch: false,
    };
    let (_, safe_replay_trace) = runtime::handler::run(safe_replay_handler, make_test_model());
    assert_eq!(base_trace.get_f64(&addr!("test")), safe_replay_trace.get_f64(&addr!("test")));
    
    // ScoreGivenTrace
    let score_handler = runtime::interpreters::ScoreGivenTrace {
        base: base_trace.clone(),
        trace: runtime::trace::Trace::default(),
    };
    let (_, score_trace) = runtime::handler::run(score_handler, make_test_model());
    assert_eq!(base_trace.get_f64(&addr!("test")), score_trace.get_f64(&addr!("test")));
    
    // SafeScoreGivenTrace
    let safe_score_handler = runtime::interpreters::SafeScoreGivenTrace {
        base: base_trace.clone(),
        trace: runtime::trace::Trace::default(),
        warn_on_error: false,
    };
    let (_, safe_score_trace) = runtime::handler::run(safe_score_handler, make_test_model());
    assert_eq!(base_trace.get_f64(&addr!("test")), safe_score_trace.get_f64(&addr!("test")));
    
    // 3. All inference algorithms should handle the same model type
    let inference_model = || sample(addr!("param"), Normal::new(0.0, 1.0).unwrap())
        .bind(|param| observe(addr!("obs"), Normal::new(param, 0.5).unwrap(), 0.5)
             .map(move |_| param));
    
    // MCMC
    let mcmc_samples = adaptive_mcmc_chain(&mut rng, &inference_model, 20, 5);
    assert_eq!(mcmc_samples.len(), 20);
    
    // SMC
    let smc_config = SMCConfig::default();
    let smc_particles = adaptive_smc(&mut rng, 15, &inference_model, smc_config);
    assert_eq!(smc_particles.len(), 15);
    
    // ABC
    let abc_model = || sample(addr!("param"), Normal::new(0.0, 1.0).unwrap());
    let simulator = |trace: &runtime::trace::Trace| trace.get_f64(&addr!("param")).unwrap_or(0.0);
    let abc_samples = abc_scalar_summary(&mut rng, abc_model, simulator, 0.5, 1.0, 10);
    assert!(abc_samples.len() <= 10);
    
    // VI
    let vi_model = || sample(addr!("param"), Normal::new(0.0, 1.0).unwrap());
    let mut guide = MeanFieldGuide::new();
    guide.params.insert(addr!("param"), VariationalParam::Normal { mu: 0.0, log_sigma: 0.0 });
    let optimized_guide = optimize_meanfield_vi(&mut rng, vi_model, guide, 5, 5, 0.1);
    assert!(!optimized_guide.params.is_empty());
}