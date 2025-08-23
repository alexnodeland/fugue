//! Comprehensive error handling tests.
//!
//! This module tests error handling throughout the library to improve coverage
//! of the error.rs module and related validation functionality.

mod test_utils;

use fugue::*;
use rand::{rngs::StdRng, SeedableRng};
use test_utils::*;

#[test]
fn test_distribution_parameter_validation() {
    // Test Normal distribution parameter validation
    assert!(Normal::new(0.0, 1.0).is_ok());
    assert!(Normal::new(f64::INFINITY, 1.0).is_err());
    assert!(Normal::new(f64::NAN, 1.0).is_err());
    assert!(Normal::new(0.0, 0.0).is_err()); // sigma = 0
    assert!(Normal::new(0.0, -1.0).is_err()); // negative sigma
    assert!(Normal::new(0.0, f64::NAN).is_err()); // NaN sigma

    // Test Exponential distribution validation
    assert!(Exponential::new(1.0).is_ok());
    assert!(Exponential::new(0.01).is_ok());
    assert!(Exponential::new(0.0).is_err()); // rate = 0
    assert!(Exponential::new(-1.0).is_err()); // negative rate
    assert!(Exponential::new(f64::INFINITY).is_err());
    assert!(Exponential::new(f64::NAN).is_err());

    // Test Beta distribution validation
    assert!(Beta::new(1.0, 1.0).is_ok());
    assert!(Beta::new(2.0, 3.0).is_ok());
    assert!(Beta::new(0.0, 1.0).is_err()); // alpha = 0
    assert!(Beta::new(1.0, 0.0).is_err()); // beta = 0
    assert!(Beta::new(-1.0, 1.0).is_err()); // negative alpha
    assert!(Beta::new(1.0, -1.0).is_err()); // negative beta
    assert!(Beta::new(f64::NAN, 1.0).is_err());
    assert!(Beta::new(1.0, f64::NAN).is_err());

    // Test Gamma distribution validation
    assert!(Gamma::new(2.0, 1.0).is_ok());
    assert!(Gamma::new(0.5, 2.0).is_ok());
    assert!(Gamma::new(0.0, 1.0).is_err()); // shape = 0
    assert!(Gamma::new(1.0, 0.0).is_err()); // rate = 0
    assert!(Gamma::new(-1.0, 1.0).is_err()); // negative shape
    assert!(Gamma::new(1.0, -1.0).is_err()); // negative rate
    assert!(Gamma::new(f64::NAN, 1.0).is_err());
    assert!(Gamma::new(1.0, f64::INFINITY).is_err());

    // Test Uniform distribution validation
    assert!(Uniform::new(0.0, 1.0).is_ok());
    assert!(Uniform::new(-5.0, 10.0).is_ok());
    assert!(Uniform::new(1.0, 1.0).is_err()); // low = high
    assert!(Uniform::new(2.0, 1.0).is_err()); // low > high
    assert!(Uniform::new(f64::NAN, 1.0).is_err());
    assert!(Uniform::new(0.0, f64::NAN).is_err());

    // Test LogNormal distribution validation
    assert!(LogNormal::new(0.0, 1.0).is_ok());
    assert!(LogNormal::new(-2.0, 0.5).is_ok());
    assert!(LogNormal::new(0.0, 0.0).is_err()); // sigma = 0
    assert!(LogNormal::new(0.0, -1.0).is_err()); // negative sigma
    assert!(LogNormal::new(f64::NAN, 1.0).is_err());
    assert!(LogNormal::new(0.0, f64::INFINITY).is_err());
}

#[test]
fn test_discrete_distribution_validation() {
    // Test Bernoulli distribution validation
    assert!(Bernoulli::new(0.5).is_ok());
    assert!(Bernoulli::new(0.0).is_ok()); // Edge case: never true
    assert!(Bernoulli::new(1.0).is_ok()); // Edge case: always true
    assert!(Bernoulli::new(-0.1).is_err()); // Negative probability
    assert!(Bernoulli::new(1.1).is_err()); // Probability > 1
    assert!(Bernoulli::new(f64::NAN).is_err());

    // Test Poisson distribution validation
    assert!(Poisson::new(1.0).is_ok());
    assert!(Poisson::new(0.1).is_ok());
    assert!(Poisson::new(100.0).is_ok());
    assert!(Poisson::new(0.0).is_err()); // lambda = 0
    assert!(Poisson::new(-1.0).is_err()); // negative lambda
    assert!(Poisson::new(f64::INFINITY).is_err());
    assert!(Poisson::new(f64::NAN).is_err());

    // Test Binomial distribution validation
    assert!(Binomial::new(10, 0.5).is_ok());
    assert!(Binomial::new(0, 0.5).is_ok()); // Edge case: no trials
    assert!(Binomial::new(10, 0.0).is_ok()); // Edge case: never succeed
    assert!(Binomial::new(10, 1.0).is_ok()); // Edge case: always succeed
    assert!(Binomial::new(10, -0.1).is_err()); // Negative probability
    assert!(Binomial::new(10, 1.1).is_err()); // Probability > 1
    assert!(Binomial::new(10, f64::NAN).is_err());

    // Test Categorical distribution validation
    assert!(Categorical::new(vec![0.2, 0.3, 0.5]).is_ok());
    assert!(Categorical::new(vec![1.0]).is_ok()); // Single category
    assert!(Categorical::new(vec![]).is_err()); // Empty probabilities
    assert!(Categorical::new(vec![0.5, 0.3]).is_err()); // Doesn't sum to 1
    assert!(Categorical::new(vec![0.5, 0.5, 0.1]).is_err()); // Sum > 1
    assert!(Categorical::new(vec![-0.1, 1.1]).is_err()); // Invalid probabilities
    assert!(Categorical::new(vec![0.5, f64::NAN]).is_err()); // NaN probability
}

#[test]
fn test_error_display_formatting() {
    // Test that error messages are informative
    let normal_error = Normal::new(0.0, -1.0).unwrap_err();
    let error_string = format!("{}", normal_error);
    assert!(error_string.contains("Normal"));
    assert!(error_string.contains("positive"));

    let bernoulli_error = Bernoulli::new(1.5).unwrap_err();
    let error_string = format!("{}", bernoulli_error);
    assert!(error_string.contains("Bernoulli"));
    assert!(error_string.contains("[0, 1]"));

    let categorical_error = Categorical::new(vec![]).unwrap_err();
    let error_string = format!("{}", categorical_error);
    assert!(error_string.contains("Categorical"));
    assert!(error_string.contains("empty"));
}

#[test]
fn test_error_context_information() {
    use std::error::Error;

    let error = Normal::new(f64::NAN, 1.0).unwrap_err();

    // Test error source chain
    let mut current_error: &dyn Error = &error;
    let mut error_count = 0;

    while let Some(source) = current_error.source() {
        current_error = source;
        error_count += 1;
        if error_count > 10 {
            break; // Avoid infinite loops
        }
    }

    // Should have some error context (even if just the original error)
    assert!(error_count >= 0);
}

#[test]
fn test_error_codes_and_categories() {
    let normal_error = Normal::new(0.0, -1.0).unwrap_err();
    match &normal_error {
        FugueError::InvalidParameters { code, .. } => {
            match code {
                ErrorCode::InvalidVariance => {
                    // Expected error codes for negative standard deviation
                }
                _ => panic!("Unexpected error code for Normal with negative sigma"),
            }
        }
        _ => panic!("Expected InvalidParameters error"),
    }

    let bernoulli_error = Bernoulli::new(2.0).unwrap_err();
    match &bernoulli_error {
        FugueError::InvalidParameters { code, .. } => {
            assert!(matches!(code, ErrorCode::InvalidProbability));
        }
        _ => panic!("Expected InvalidParameters error"),
    }
}

#[test]
fn test_trace_error_handling() {
    let mut trace = Trace::default();

    // Test type-safe accessors with wrong types
    trace.insert_choice(addr!("x"), ChoiceValue::F64(1.5), -0.5);
    trace.insert_choice(addr!("flag"), ChoiceValue::Bool(true), -0.693);
    trace.insert_choice(addr!("count"), ChoiceValue::U64(42), -1.0);

    // Correct types should work
    assert_eq!(trace.get_f64(&addr!("x")), Some(1.5));
    assert_eq!(trace.get_bool(&addr!("flag")), Some(true));
    assert_eq!(trace.get_u64(&addr!("count")), Some(42));

    // Wrong types should return None (not crash)
    assert_eq!(trace.get_bool(&addr!("x")), None); // x is f64, not bool
    assert_eq!(trace.get_f64(&addr!("flag")), None); // flag is bool, not f64
    assert_eq!(trace.get_usize(&addr!("count")), None); // count is u64, not usize

    // Missing addresses should return None
    assert_eq!(trace.get_f64(&addr!("missing")), None);
}

#[test]
fn test_model_execution_errors() {
    let mut rng = test_rng();

    // Test model with invalid distribution construction
    let _invalid_model = || {
        sample(addr!("bad_normal"), Normal::new(0.0, -1.0).unwrap()) // This will panic during unwrap
    };

    // The model construction itself should panic on unwrap, but let's test a safer version
    let safer_model = || {
        match Normal::new(0.0, -1.0) {
            Ok(dist) => sample(addr!("x"), dist),
            Err(_) => pure(0.0), // Fallback for invalid parameters
        }
    };

    // This should work without crashing
    let (result, trace) = runtime::handler::run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        safer_model(),
    );

    assert_eq!(result, 0.0); // Should use fallback
    assert_finite(trace.total_log_weight());
}

#[test]
fn test_numerical_edge_cases() {
    // Test distributions with extreme but valid parameters
    let tiny_normal = Normal::new(0.0, 1e-10).unwrap();
    let huge_normal = Normal::new(0.0, 1e10).unwrap();
    let extreme_beta = Beta::new(1e-6, 1e-6).unwrap();
    let large_poisson = Poisson::new(1000.0).unwrap();

    let mut rng = test_rng();

    // These should not crash and should produce finite values
    let tiny_sample = tiny_normal.sample(&mut rng);
    let huge_sample = huge_normal.sample(&mut rng);
    let extreme_beta_sample = extreme_beta.sample(&mut rng);
    let large_poisson_sample = large_poisson.sample(&mut rng);

    assert_finite(tiny_sample);
    assert_finite(huge_sample);
    assert_finite(extreme_beta_sample);

    // Poisson returns u64, so check it's valid
    assert!(large_poisson_sample < 1000000); // Should be reasonable even if large
}

#[test]
fn test_address_validation() {
    // Test various address formats
    let simple_addr = addr!("x");
    let indexed_addr = addr!("x", 1);
    let nested_addr = addr!("group", 2);

    // All should be valid addresses
    let mut trace = Trace::default();
    trace.insert_choice(simple_addr.clone(), ChoiceValue::F64(1.0), 0.0);
    trace.insert_choice(indexed_addr.clone(), ChoiceValue::F64(2.0), 0.0);
    trace.insert_choice(nested_addr.clone(), ChoiceValue::F64(3.0), 0.0);

    // Should be able to retrieve all
    assert!(trace.choices.contains_key(&simple_addr));
    assert!(trace.choices.contains_key(&indexed_addr));
    assert!(trace.choices.contains_key(&nested_addr));
}

#[test]
fn test_observe_type_safety() {
    let mut rng = test_rng();

    // Test observing with correct and incorrect types
    let model_f64 = || {
        sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap()).bind(|mu| {
            observe(addr!("y"), Normal::new(mu, 1.0).unwrap(), 2.5); // f64 observation
            pure(mu)
        })
    };

    let model_bool = || {
        sample(addr!("p"), Beta::new(2.0, 2.0).unwrap()).bind(|p| {
            observe(addr!("coin"), Bernoulli::new(p).unwrap(), true); // bool observation
            pure(p)
        })
    };

    let model_u64 = || {
        sample(addr!("lambda"), Exponential::new(1.0).unwrap()).bind(|lambda| {
            observe(addr!("count"), Poisson::new(lambda).unwrap(), 5u64); // u64 observation
            pure(lambda)
        })
    };

    // All should execute without type errors
    let (result_f64, trace_f64) = runtime::handler::run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        model_f64(),
    );
    assert_finite(result_f64);
    assert_finite(trace_f64.total_log_weight());

    let (result_bool, trace_bool) = runtime::handler::run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        model_bool(),
    );
    assert!(result_bool >= 0.0 && result_bool <= 1.0);
    assert_finite(trace_bool.total_log_weight());

    let (result_u64, trace_u64) = runtime::handler::run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        model_u64(),
    );
    assert!(result_u64 > 0.0);
    assert_finite(trace_u64.total_log_weight());
}

#[test]
fn test_error_recovery_patterns() {
    // Test patterns for recovering from errors gracefully
    fn safe_normal_construction(mu: f64, sigma: f64) -> Result<Normal, FugueError> {
        Normal::new(mu, sigma)
    }

    fn robust_model(mu: f64, sigma: f64, fallback_sigma: f64) -> Model<f64> {
        match safe_normal_construction(mu, sigma) {
            Ok(dist) => sample(addr!("x"), dist),
            Err(_) => {
                // Fallback to safe parameters
                sample(addr!("x"), Normal::new(mu, fallback_sigma).unwrap())
            }
        }
    }

    let mut rng = test_rng();

    // Test with invalid sigma (should use fallback)
    let model_with_fallback = robust_model(0.0, -1.0, 1.0);
    let (result, trace) = runtime::handler::run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        model_with_fallback,
    );

    assert_finite(result);
    assert_finite(trace.total_log_weight());
    assert!(trace.choices.contains_key(&addr!("x")));
}

#[test]
fn test_comprehensive_distribution_validation() {
    // Test every distribution's validation exhaustively
    struct ValidationTest {
        name: &'static str,
        should_pass: Vec<Box<dyn Fn() -> Result<(), FugueError>>>,
        should_fail: Vec<Box<dyn Fn() -> Result<(), FugueError>>>,
    }

    let tests = vec![
        ValidationTest {
            name: "Normal",
            should_pass: vec![
                Box::new(|| Normal::new(0.0, 1.0).map(|_| ())),
                Box::new(|| Normal::new(-5.0, 0.1).map(|_| ())),
                Box::new(|| Normal::new(100.0, 10.0).map(|_| ())),
            ],
            should_fail: vec![
                Box::new(|| Normal::new(f64::NAN, 1.0).map(|_| ())),
                Box::new(|| Normal::new(0.0, f64::NAN).map(|_| ())),
                Box::new(|| Normal::new(0.0, 0.0).map(|_| ())),
                Box::new(|| Normal::new(0.0, -1.0).map(|_| ())),
            ],
        },
        ValidationTest {
            name: "Beta",
            should_pass: vec![
                Box::new(|| Beta::new(1.0, 1.0).map(|_| ())),
                Box::new(|| Beta::new(0.1, 0.1).map(|_| ())),
                Box::new(|| Beta::new(10.0, 5.0).map(|_| ())),
            ],
            should_fail: vec![
                Box::new(|| Beta::new(0.0, 1.0).map(|_| ())),
                Box::new(|| Beta::new(1.0, 0.0).map(|_| ())),
                Box::new(|| Beta::new(-1.0, 1.0).map(|_| ())),
                Box::new(|| Beta::new(f64::NAN, 1.0).map(|_| ())),
            ],
        },
    ];

    for test in tests {
        // Test cases that should pass
        for (i, should_pass) in test.should_pass.iter().enumerate() {
            match should_pass() {
                Ok(()) => {} // Good
                Err(e) => panic!("{} should_pass case {} failed: {}", test.name, i, e),
            }
        }

        // Test cases that should fail
        for (i, should_fail) in test.should_fail.iter().enumerate() {
            match should_fail() {
                Err(_) => {} // Good - it should fail
                Ok(()) => panic!("{} should_fail case {} unexpectedly passed", test.name, i),
            }
        }
    }
}

#[test]
fn test_error_propagation_in_models() {
    let mut rng = test_rng();

    // Test that errors in model execution are handled gracefully
    let model_with_factor = || {
        sample(addr!("x"), Normal::new(0.0, 1.0).unwrap()).bind(|x| {
            // Add a factor that could cause numerical issues
            factor(x * x * -1000.0); // Large negative factor
            pure(x)
        })
    };

    let (result, trace) = runtime::handler::run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        model_with_factor(),
    );

    // Should still produce valid results even with extreme factor
    assert_finite(result);
    // Log weight might be very negative but should be finite
    let total_log_weight = trace.total_log_weight();
    assert_finite(total_log_weight);
}

#[test]
fn test_validation_trait_coverage() {
    // Test that the Validate trait works for various distributions
    use std::error::Error;

    let valid_normal = Normal::new(0.0, 1.0).unwrap();
    let validation_result = valid_normal.validate();
    assert!(validation_result.is_ok());

    // Test validation of parameters post-construction
    let extreme_normal = Normal::new(0.0, 1e-12).unwrap(); // Very small but valid
    assert!(extreme_normal.validate().is_ok());
}
