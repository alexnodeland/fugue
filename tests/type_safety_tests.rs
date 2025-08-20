use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

#[test]
fn test_safe_distribution_constructors() {
    // Valid constructors should succeed
    let normal = Normal::new(0.0, 1.0);
    assert!(normal.is_ok());

    let bernoulli = Bernoulli::new(0.5);
    assert!(bernoulli.is_ok());

    let poisson = Poisson::new(3.0);
    assert!(poisson.is_ok());

    let categorical = Categorical::new(vec![0.3, 0.5, 0.2]);
    assert!(categorical.is_ok());

    // Invalid constructors should fail
    assert!(Normal::new(0.0, -1.0).is_err()); // Negative sigma
    assert!(Normal::new(f64::NAN, 1.0).is_err()); // NaN mu
    assert!(Bernoulli::new(-0.1).is_err()); // Negative probability
    assert!(Bernoulli::new(1.5).is_err()); // Probability > 1
    assert!(Poisson::new(-1.0).is_err()); // Negative lambda
    assert!(Categorical::new(vec![]).is_err()); // Empty probabilities
    assert!(Categorical::new(vec![0.3, 0.3, 0.3]).is_err()); // Doesn't sum to 1
}

#[test]
fn test_type_safe_trace_accessors() {
    let mut trace = Trace::default();

    // Insert different types
    trace.insert_choice(addr!("x"), ChoiceValue::F64(1.5), -0.5);
    trace.insert_choice(addr!("coin"), ChoiceValue::Bool(true), -0.693);
    trace.insert_choice(addr!("count"), ChoiceValue::U64(7), -2.1);
    trace.insert_choice(addr!("choice"), ChoiceValue::Usize(2), -1.6);

    // Type-safe accessors should work correctly
    assert_eq!(trace.get_f64(&addr!("x")), Some(1.5));
    assert_eq!(trace.get_bool(&addr!("coin")), Some(true));
    assert_eq!(trace.get_u64(&addr!("count")), Some(7));
    assert_eq!(trace.get_usize(&addr!("choice")), Some(2));

    // Wrong type accessors should return None
    assert_eq!(trace.get_bool(&addr!("x")), None); // x is f64, not bool
    assert_eq!(trace.get_f64(&addr!("coin")), None); // coin is bool, not f64

    // Missing addresses should return None
    assert_eq!(trace.get_f64(&addr!("missing")), None);

    // Result-based accessors for error handling
    assert!(trace.get_f64_result(&addr!("x")).is_ok());
    assert!(trace.get_bool_result(&addr!("x")).is_err()); // Type mismatch
    assert!(trace.get_f64_result(&addr!("missing")).is_err()); // Missing
}

#[test]
fn test_safe_handlers() {
    let mut rng = StdRng::seed_from_u64(42);

    // Create base trace with intentional type mismatch
    let mut base_trace = Trace::default();
    base_trace.insert_choice(addr!("test"), ChoiceValue::Bool(true), -0.5);

    // Safe replay handler should handle type mismatch gracefully
    let safe_handler = SafeReplayHandler {
        rng: &mut rng,
        base: base_trace.clone(),
        trace: Trace::default(),
        warn_on_mismatch: false, // Don't pollute test output
    };

    let model = sample(addr!("test"), Normal::new(0.0, 1.0).unwrap());
    let (value, _) = runtime::handler::run(safe_handler, model);

    // Should not panic, should return a valid f64
    assert!(value.is_finite());

    // Safe score handler should also handle gracefully
    let safe_scorer = SafeScoreGivenTrace {
        base: base_trace,
        trace: Trace::default(),
        warn_on_error: false,
    };

    let model2 = sample(addr!("test"), Normal::new(0.0, 1.0).unwrap());
    let (_, score_trace) = runtime::handler::run(safe_scorer, model2);

    // Should have negative infinity log-weight due to type mismatch
    assert_eq!(score_trace.log_prior, f64::NEG_INFINITY);
}

#[test]
fn test_type_safe_diagnostics() {
    // Create traces with different value types
    let mut traces = Vec::new();
    for i in 0..10 {
        let mut trace = Trace::default();
        trace.insert_choice(addr!("f64_val"), ChoiceValue::F64(i as f64), -0.5);
        trace.insert_choice(addr!("bool_val"), ChoiceValue::Bool(i % 2 == 0), -0.693);
        trace.insert_choice(addr!("u64_val"), ChoiceValue::U64(i), -2.0);
        trace.insert_choice(
            addr!("usize_val"),
            ChoiceValue::Usize((i % 3) as usize),
            -1.5,
        );
        traces.push(trace);
    }

    // Type-safe extraction should work correctly
    let f64_values = extract_f64_values(&traces, &addr!("f64_val"));
    let bool_values = extract_bool_values(&traces, &addr!("bool_val"));
    let u64_values = extract_u64_values(&traces, &addr!("u64_val"));
    let usize_values = extract_usize_values(&traces, &addr!("usize_val"));

    assert_eq!(f64_values.len(), 10);
    assert_eq!(bool_values.len(), 10);
    assert_eq!(u64_values.len(), 10);
    assert_eq!(usize_values.len(), 10);

    // Should extract correct values
    assert_eq!(f64_values[0], 0.0);
    assert_eq!(f64_values[9], 9.0);
    assert_eq!(bool_values[0], true); // 0 % 2 == 0
    assert_eq!(bool_values[1], false); // 1 % 2 != 0
    assert_eq!(u64_values[5], 5);
    assert_eq!(usize_values[0], 0); // 0 % 3 = 0
    assert_eq!(usize_values[4], 1); // 4 % 3 = 1

    // Type-safe diagnostics for f64 values
    let chains = vec![traces.clone()];
    let summary = summarize_f64_parameter(&chains, &addr!("f64_val"));

    assert!(summary.mean.is_finite());
    assert!(summary.std.is_finite());
    assert!(!summary.quantiles.is_empty());
}

#[test]
fn test_natural_return_types() {
    let mut rng = StdRng::seed_from_u64(42);

    // Test that distributions return their natural types
    let bernoulli = Bernoulli::new(0.5).unwrap();
    let flip: bool = bernoulli.sample(&mut rng); // Should be bool
    assert!(flip == true || flip == false);

    let poisson = Poisson::new(3.0).unwrap();
    let count: u64 = poisson.sample(&mut rng); // Should be u64
    assert!(count < 1000); // Reasonable upper bound

    let categorical = Categorical::new(vec![0.4, 0.6]).unwrap();
    let choice: usize = categorical.sample(&mut rng); // Should be usize
    assert!(choice == 0 || choice == 1);

    let normal = Normal::new(0.0, 1.0).unwrap();
    let value: f64 = normal.sample(&mut rng); // Should be f64
    assert!(value.is_finite());
}

#[test]
fn test_type_safe_model_composition() {
    let model = prob! {
        // Each sample returns its natural type
        let coin <- sample(addr!("coin"), Bernoulli::new(0.5).unwrap());  // bool
        let count <- sample(addr!("count"), Poisson::new(2.0).unwrap());  // u64
        let choice <- sample(addr!("choice"), Categorical::new(vec![0.3, 0.7]).unwrap()); // usize
        let value <- sample(addr!("value"), Normal::new(0.0, 1.0).unwrap()); // f64

        // Natural usage with correct types
        let result = if coin {
            format!("Heads: {} events, choice {}, value {:.2}", count, choice, value)
        } else {
            format!("Tails: {} events, choice {}, value {:.2}", count, choice, value)
        };

        pure(result)
    };

    let mut rng = StdRng::seed_from_u64(42);
    let (result, trace) = runtime::handler::run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        model,
    );

    // Should produce a valid result string
    assert!(!result.is_empty());
    assert!(result.contains("events"));

    // Trace should contain all values with correct types
    assert!(trace.get_bool(&addr!("coin")).is_some());
    assert!(trace.get_u64(&addr!("count")).is_some());
    assert!(trace.get_usize(&addr!("choice")).is_some());
    assert!(trace.get_f64(&addr!("value")).is_some());
}

#[test]
fn test_safe_constructor_validation() {
    // Test comprehensive validation for all distribution types

    // Normal validation
    assert!(Normal::new(0.0, 1.0).is_ok());
    assert!(Normal::new(f64::INFINITY, 1.0).is_err());
    assert!(Normal::new(0.0, f64::NEG_INFINITY).is_err());
    assert!(Normal::new(0.0, 0.0).is_err());

    // Bernoulli validation
    assert!(Bernoulli::new(0.0).is_ok());
    assert!(Bernoulli::new(1.0).is_ok());
    assert!(Bernoulli::new(0.5).is_ok());
    assert!(Bernoulli::new(-0.1).is_err());
    assert!(Bernoulli::new(1.1).is_err());
    assert!(Bernoulli::new(f64::NAN).is_err());

    // Poisson validation
    assert!(Poisson::new(1.0).is_ok());
    assert!(Poisson::new(0.1).is_ok());
    assert!(Poisson::new(0.0).is_err());
    assert!(Poisson::new(-1.0).is_err());
    assert!(Poisson::new(f64::NAN).is_err());

    // Categorical validation
    assert!(Categorical::new(vec![1.0]).is_ok());
    assert!(Categorical::new(vec![0.5, 0.5]).is_ok());
    assert!(Categorical::new(vec![0.2, 0.3, 0.5]).is_ok());
    assert!(Categorical::new(vec![]).is_err());
    assert!(Categorical::new(vec![0.5, 0.6]).is_err()); // Sum > 1
    assert!(Categorical::new(vec![0.3, 0.3]).is_err()); // Sum < 1
    assert!(Categorical::new(vec![-0.1, 1.1]).is_err()); // Negative prob
    assert!(Categorical::new(vec![f64::NAN, 0.5]).is_err()); // NaN prob

    // Categorical uniform constructor
    assert!(Categorical::uniform(1).is_ok());
    assert!(Categorical::uniform(5).is_ok());
    assert!(Categorical::uniform(0).is_err());

    let uniform_cat = Categorical::uniform(3).unwrap();
    assert_eq!(uniform_cat.len(), 3);
    assert!((uniform_cat.probs()[0] - 1.0 / 3.0).abs() < 1e-10);
}
