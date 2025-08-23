//! Advanced tests for runtime interpreters.
//!
//! This module tests advanced runtime interpreter functionality which currently has low coverage (32.76%).

mod test_utils;

use fugue::runtime::interpreters::*;
use fugue::*;
use rand::{rngs::StdRng, SeedableRng};
use test_utils::*;

#[test]
fn test_replay_handler_complete_replay() {
    let mut rng = seeded_rng(123);

    let model = || {
        sample(addr!("x"), Normal::new(0.0, 1.0).unwrap()).bind(|x| {
            sample(addr!("y"), Normal::new(x, 0.5).unwrap()).bind(move |y| {
                observe(addr!("obs"), Normal::new(y, 0.1).unwrap(), 2.0);
                pure((x, y))
            })
        })
    };

    // First, generate a complete trace
    let (original_result, base_trace) = runtime::handler::run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        model(),
    );

    // Now replay with the same trace
    let (replay_result, replay_trace) = runtime::handler::run(
        ReplayHandler {
            rng: &mut rng,
            base: base_trace.clone(),
            trace: Trace::default(),
        },
        model(),
    );

    // Results should be identical
    assert_eq!(replay_result, original_result);

    // Traces should have same choices
    assert_eq!(
        replay_trace.get_f64(&addr!("x")),
        base_trace.get_f64(&addr!("x"))
    );
    assert_eq!(
        replay_trace.get_f64(&addr!("y")),
        base_trace.get_f64(&addr!("y"))
    );
    assert_eq!(
        replay_trace.get_f64(&addr!("obs")),
        base_trace.get_f64(&addr!("obs"))
    );
}

#[test]
fn test_replay_handler_partial_replay() {
    let mut rng = seeded_rng(456);

    let model = || {
        sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
            .bind(|x| sample(addr!("y"), Normal::new(x, 0.5).unwrap()).bind(move |y| pure((x, y))))
    };

    // Create a trace with only one choice
    let mut partial_trace = Trace::default();
    partial_trace.insert_choice(addr!("x"), ChoiceValue::F64(1.5), -0.5);

    // Replay should use the existing choice and sample the missing one
    let (result, replay_trace) = runtime::handler::run(
        ReplayHandler {
            rng: &mut rng,
            base: partial_trace.clone(),
            trace: Trace::default(),
        },
        model(),
    );

    let (x, y) = result;

    // x should match the base trace
    assert_eq!(x, 1.5);
    assert_eq!(replay_trace.get_f64(&addr!("x")), Some(1.5));

    // y should be sampled fresh
    assert!(replay_trace.get_f64(&addr!("y")).is_some());
    assert_finite(y);
}

#[test]
fn test_replay_handler_different_types() {
    let mut rng = seeded_rng(789);

    let model = || {
        sample(addr!("continuous"), Normal::new(0.0, 1.0).unwrap()).bind(|x| {
            sample(addr!("discrete"), Bernoulli::new(0.5).unwrap()).bind(move |b| {
                sample(addr!("count"), Poisson::new(5.0).unwrap()).bind(move |n| {
                    sample(
                        addr!("category"),
                        Categorical::new(vec![0.2, 0.3, 0.5]).unwrap(),
                    )
                    .bind(move |c| pure((x, b, n, c)))
                })
            })
        })
    };

    // Create a base trace with all types
    let mut base_trace = Trace::default();
    base_trace.insert_choice(addr!("continuous"), ChoiceValue::F64(2.5), -1.2);
    base_trace.insert_choice(addr!("discrete"), ChoiceValue::Bool(true), -0.693);
    base_trace.insert_choice(addr!("count"), ChoiceValue::U64(7), -1.5);
    base_trace.insert_choice(addr!("category"), ChoiceValue::Usize(1), -1.1);

    let (result, replay_trace) = runtime::handler::run(
        ReplayHandler {
            rng: &mut rng,
            base: base_trace.clone(),
            trace: Trace::default(),
        },
        model(),
    );

    let (x, b, n, c) = result;

    // All values should match base trace
    assert_eq!(x, 2.5);
    assert_eq!(b, true);
    assert_eq!(n, 7);
    assert_eq!(c, 1);

    // Check trace consistency
    assert_eq!(replay_trace.get_f64(&addr!("continuous")), Some(2.5));
    assert_eq!(replay_trace.get_bool(&addr!("discrete")), Some(true));
    assert_eq!(replay_trace.get_u64(&addr!("count")), Some(7));
    assert_eq!(replay_trace.get_usize(&addr!("category")), Some(1));
}

#[test]
fn test_score_given_trace_complete() {
    let mut rng = seeded_rng(999);

    let model = || {
        sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap()).bind(|mu| {
            observe(addr!("y"), Normal::new(mu, 0.5).unwrap(), 1.8);
            pure(mu)
        })
    };

    // Generate a complete trace first
    let (_, base_trace) = runtime::handler::run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        model(),
    );

    let original_mu = base_trace.get_f64(&addr!("mu")).unwrap();

    // Score the same trace
    let (score_result, score_trace) = runtime::handler::run(
        ScoreGivenTrace {
            base: base_trace.clone(),
            trace: Trace::default(),
        },
        model(),
    );

    // Result should match original
    assert_eq!(score_result, original_mu);

    // Log probabilities should be computed
    assert_finite(score_trace.log_prior);
    assert_finite(score_trace.log_likelihood);
    assert_finite(score_trace.total_log_weight());
}

#[test]
fn test_score_given_trace_with_factors() {
    let model = || {
        sample(addr!("x"), Normal::new(0.0, 1.0).unwrap()).bind(|x| {
            factor(-x * x / 2.0); // Additional Gaussian-like penalty
            observe(addr!("y"), Normal::new(x, 0.5).unwrap(), 1.0);
            pure(x)
        })
    };

    // Create a trace manually
    let mut base_trace = Trace::default();
    base_trace.insert_choice(addr!("x"), ChoiceValue::F64(0.5), -0.125);
    base_trace.insert_choice(addr!("y"), ChoiceValue::F64(1.0), -0.5);

    let (result, score_trace) = runtime::handler::run(
        ScoreGivenTrace {
            base: base_trace.clone(),
            trace: Trace::default(),
        },
        model(),
    );

    assert_eq!(result, 0.5);

    // Should have computed all components
    assert_finite(score_trace.log_prior);
    assert_finite(score_trace.log_likelihood);
    assert_finite(score_trace.log_factors);

    // Factor should be finite and include both penalties
    assert_finite(score_trace.log_factors);

    // Total should include all components
    let expected_total =
        score_trace.log_prior + score_trace.log_likelihood + score_trace.log_factors;
    assert_approx_eq(
        score_trace.total_log_weight(),
        expected_total,
        TEST_TOLERANCE,
    );
}

#[test]
fn test_safe_score_given_trace_missing_address() {
    let model = || {
        sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
            .bind(|x| sample(addr!("y"), Normal::new(x, 0.5).unwrap()).bind(move |y| pure((x, y))))
    };

    // Create incomplete trace (missing "y")
    let mut incomplete_trace = Trace::default();
    incomplete_trace.insert_choice(addr!("x"), ChoiceValue::F64(1.0), -0.5);

    let (result, score_trace) = runtime::handler::run(
        SafeScoreGivenTrace {
            base: incomplete_trace,
            trace: Trace::default(),
            warn_on_error: false, // Don't print warnings in test
        },
        model(),
    );

    let (x, y) = result;

    // x should be correct, y should be dummy value
    assert_eq!(x, 1.0);
    assert_eq!(y, 0.0); // Dummy value for missing address

    // Trace should have -infinity log weight due to missing address
    assert_eq!(score_trace.total_log_weight(), f64::NEG_INFINITY);
}

#[test]
fn test_safe_score_given_trace_type_mismatch() {
    let model = || {
        sample(addr!("x"), Normal::new(0.0, 1.0).unwrap()) // Expects f64
    };

    // Create trace with wrong type
    let mut wrong_type_trace = Trace::default();
    wrong_type_trace.insert_choice(addr!("x"), ChoiceValue::Bool(true), -0.693); // Wrong type

    let (result, score_trace) = runtime::handler::run(
        SafeScoreGivenTrace {
            base: wrong_type_trace,
            trace: Trace::default(),
            warn_on_error: false,
        },
        model(),
    );

    // Should return dummy value
    assert_eq!(result, 0.0);

    // Should have -infinity log weight due to type mismatch
    assert_eq!(score_trace.total_log_weight(), f64::NEG_INFINITY);
}

#[test]
fn test_safe_score_given_trace_with_warnings() {
    let model = || sample(addr!("missing"), Normal::new(0.0, 1.0).unwrap());

    let empty_trace = Trace::default();

    // This would print warnings if warn_on_error is true
    let (_result, score_trace) = runtime::handler::run(
        SafeScoreGivenTrace {
            base: empty_trace,
            trace: Trace::default(),
            warn_on_error: true, // Enable warnings
        },
        model(),
    );

    // Should still complete with -infinity weight
    assert_eq!(score_trace.total_log_weight(), f64::NEG_INFINITY);
}

#[test]
fn test_prior_handler_all_types() {
    let mut rng = seeded_rng(1234);

    let model = || {
        sample(addr!("f64_val"), Normal::new(0.0, 1.0).unwrap()).bind(|f| {
            sample(addr!("bool_val"), Bernoulli::new(0.7).unwrap()).bind(move |b| {
                sample(addr!("u64_val"), Poisson::new(3.0).unwrap()).bind(move |u| {
                    sample(
                        addr!("usize_val"),
                        Categorical::new(vec![0.3, 0.4, 0.3]).unwrap(),
                    )
                    .bind(move |us| pure((f, b, u, us)))
                })
            })
        })
    };

    let (result, trace) = runtime::handler::run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        model(),
    );

    let (f_val, b_val, u_val, us_val) = result;

    // Check all types are present and valid
    assert_finite(f_val);
    assert!(us_val < 3); // Categorical should be 0, 1, or 2

    // Check trace contains all choices
    assert_eq!(trace.get_f64(&addr!("f64_val")), Some(f_val));
    assert_eq!(trace.get_bool(&addr!("bool_val")), Some(b_val));
    assert_eq!(trace.get_u64(&addr!("u64_val")), Some(u_val));
    assert_eq!(trace.get_usize(&addr!("usize_val")), Some(us_val));

    // Check log probabilities are accumulated
    assert_finite(trace.log_prior);
    assert!(trace.log_prior < 0.0); // Log probabilities should be negative
}

#[test]
fn test_prior_handler_with_observations() {
    let mut rng = seeded_rng(5678);

    let model = || {
        sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap()).bind(|mu| {
            observe(addr!("y1"), Normal::new(mu, 0.5).unwrap(), 1.5);
            observe(addr!("y2"), Normal::new(mu, 0.5).unwrap(), 1.8);
            observe(addr!("y3"), Normal::new(mu, 0.5).unwrap(), 1.2);
            pure(mu)
        })
    };

    let (result, trace) = runtime::handler::run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        model(),
    );

    assert_finite(result);

    // Should have sampled mu
    assert_eq!(trace.get_f64(&addr!("mu")), Some(result));

    // Observations contribute to log_likelihood but aren't stored as choices
    // Check that likelihood was computed (should be negative for valid observations)
    assert_finite(trace.log_likelihood);

    // Should have both prior and likelihood
    assert_finite(trace.log_prior);
    assert_finite(trace.log_likelihood);
    assert!(trace.log_prior < 0.0);
    // log_likelihood should be finite (may be positive or negative depending on observations)
    assert_finite(trace.log_likelihood);
}

#[test]
fn test_handler_with_nested_addressing() {
    let mut rng = seeded_rng(9999);

    let model = || {
        (0..3).fold(pure(Vec::new()), |acc, i| {
            acc.bind(move |mut vec| {
                sample(addr!("item", i), Normal::new(i as f64, 1.0).unwrap()).map(move |val| {
                    vec.push(val);
                    vec
                })
            })
        })
    };

    let (result, trace) = runtime::handler::run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        model(),
    );

    assert_eq!(result.len(), 3);

    // Check all nested addresses are present
    for i in 0..3 {
        assert!(trace.choices.contains_key(&addr!("item", i)));
        assert_eq!(trace.get_f64(&addr!("item", i)), Some(result[i]));
    }

    // All values should be finite
    for &val in &result {
        assert_finite(val);
    }
}

#[test]
fn test_replay_handler_with_factors() {
    let mut rng = seeded_rng(1111);

    let model = || {
        sample(addr!("x"), Normal::new(0.0, 1.0).unwrap()).bind(|x| {
            factor(x * x * -0.5); // Quadratic penalty
            factor(x.abs() * -0.1); // Absolute penalty
            pure(x)
        })
    };

    // Generate base trace
    let (original_result, base_trace) = runtime::handler::run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        model(),
    );

    // Replay the trace
    let (replay_result, replay_trace) = runtime::handler::run(
        ReplayHandler {
            rng: &mut rng,
            base: base_trace.clone(),
            trace: Trace::default(),
        },
        model(),
    );

    // Results should match
    assert_eq!(replay_result, original_result);

    // Factors should be recomputed identically
    assert_approx_eq(
        replay_trace.log_factors,
        base_trace.log_factors,
        TEST_TOLERANCE,
    );

    // Both factor contributions should be finite
    assert_finite(replay_trace.log_factors);
}

#[test]
fn test_interpreter_memory_and_performance() {
    let mut rng = seeded_rng(2222);

    // Create a model with many variables to test memory efficiency
    let model = || {
        (0..100).fold(pure(0.0), |acc, i| {
            acc.bind(move |sum| {
                sample(addr!("var", i), Normal::new(0.0, 1.0).unwrap()).map(move |val| sum + val)
            })
        })
    };

    let start_time = std::time::Instant::now();

    let (result, trace) = runtime::handler::run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        model(),
    );

    let elapsed = start_time.elapsed();

    // Should complete in reasonable time
    assert!(elapsed < std::time::Duration::from_secs(1));

    // Should have all 100 variables
    assert_eq!(trace.choices.len(), 100);

    // Result should be finite (sum of normals)
    assert_finite(result);

    // Replay should also be efficient
    let start_time = std::time::Instant::now();

    let (replay_result, _) = runtime::handler::run(
        ReplayHandler {
            rng: &mut rng,
            base: trace,
            trace: Trace::default(),
        },
        model(),
    );

    let replay_elapsed = start_time.elapsed();

    assert!(replay_elapsed < std::time::Duration::from_secs(1));
    assert_eq!(replay_result, result);
}

#[test]
fn test_handler_error_recovery() {
    let mut rng = seeded_rng(3333);

    // Test with extreme values that could cause numerical issues
    let model = || {
        sample(addr!("x"), Normal::new(0.0, 1e10).unwrap()).bind(|x| {
            // Very large variance
            observe(addr!("y"), Normal::new(x, 1e-10).unwrap(), 0.0); // Very small variance
            pure(x)
        })
    };

    let (result, trace) = runtime::handler::run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        model(),
    );

    // Should handle extreme values gracefully
    assert_finite(result);
    assert_finite(trace.total_log_weight());

    // Replay should also work
    let (replay_result, replay_trace) = runtime::handler::run(
        ReplayHandler {
            rng: &mut rng,
            base: trace,
            trace: Trace::default(),
        },
        model(),
    );

    assert_eq!(replay_result, result);
    assert_finite(replay_trace.total_log_weight());
}
