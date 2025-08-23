//! Tests for runtime interpreters and handlers.
//!
//! This module tests the runtime interpreter functionality which currently has low coverage.

mod test_utils;

use fugue::*;
use rand::{rngs::StdRng, SeedableRng};
use test_utils::*;

#[test]
fn test_prior_handler_basic() {
    let mut rng = test_rng();

    let model = models::gaussian_mean(2.0);

    let (result, trace) = runtime::handler::run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        model,
    );

    assert_finite(result);
    assert!(trace.choices.contains_key(&addr!("mu")));
    assert_finite(trace.log_prior);
    assert_finite(trace.log_likelihood);
}

#[test]
fn test_replay_handler_deterministic() {
    let mut rng = test_rng();

    let model = || models::gaussian_mean(1.0);

    // Run once to get a base trace
    let (_, base_trace) = runtime::handler::run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        model(),
    );

    let base_mu = base_trace.get_f64(&addr!("mu")).unwrap();

    // Replay should give identical results
    let (result1, trace1) = runtime::handler::run(
        ReplayHandler {
            rng: &mut rng,
            base: base_trace.clone(),
            trace: Trace::default(),
        },
        model(),
    );

    let (result2, trace2) = runtime::handler::run(
        ReplayHandler {
            rng: &mut rng,
            base: base_trace.clone(),
            trace: Trace::default(),
        },
        model(),
    );

    // Results should be identical (deterministic replay)
    assert_eq!(result1, result2);
    assert_eq!(result1, base_mu);

    // Traces should have same choices
    assert_eq!(trace1.get_f64(&addr!("mu")), trace2.get_f64(&addr!("mu")));
    assert_eq!(trace1.get_f64(&addr!("mu")), Some(base_mu));
}

#[test]
fn test_score_given_trace() {
    let mut rng = test_rng();

    let model = || models::gaussian_mean(0.5);

    // Generate a trace from prior
    let (_, base_trace) = runtime::handler::run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        model(),
    );

    // Score the same model given the trace
    let (result, scored_trace) = runtime::handler::run(
        ScoreGivenTrace {
            base: base_trace.clone(),
            trace: Trace::default(),
        },
        model(),
    );

    // Result should match the original mu value
    let original_mu = base_trace.get_f64(&addr!("mu")).unwrap();
    assert_eq!(result, original_mu);

    // Scored trace should have same choices but updated log weights
    assert_eq!(scored_trace.get_f64(&addr!("mu")), Some(original_mu));
    assert_finite(scored_trace.total_log_weight());
}

#[test]
fn test_handler_composition() {
    let mut rng = seeded_rng(123);

    let model = || models::gaussian_mean(3.0);

    // Chain: Prior -> Replay -> Score
    let (_, trace1) = runtime::handler::run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        model(),
    );

    let (_, trace2) = runtime::handler::run(
        ReplayHandler {
            rng: &mut rng,
            base: trace1.clone(),
            trace: Trace::default(),
        },
        model(),
    );

    let (_, trace3) = runtime::handler::run(
        ScoreGivenTrace {
            base: trace2.clone(),
            trace: Trace::default(),
        },
        model(),
    );

    // All should have same mu value
    let mu1 = trace1.get_f64(&addr!("mu")).unwrap();
    let mu2 = trace2.get_f64(&addr!("mu")).unwrap();
    let mu3 = trace3.get_f64(&addr!("mu")).unwrap();

    assert_eq!(mu1, mu2);
    assert_eq!(mu2, mu3);

    // All traces should be valid
    assert_finite(trace1.total_log_weight());
    assert_finite(trace2.total_log_weight());
    assert_finite(trace3.total_log_weight());
}

#[test]
fn test_handler_with_mixed_types() {
    let mut rng = seeded_rng(456);

    let (result, trace) = runtime::handler::run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        models::mixed_model(),
    );

    let (continuous, discrete) = result;

    // Check types are preserved
    assert_finite(continuous);
    assert_eq!(trace.get_f64(&addr!("continuous")), Some(continuous));
    assert_eq!(trace.get_bool(&addr!("discrete")), Some(discrete));

    // Test replay with mixed types
    let (result2, trace2) = runtime::handler::run(
        ReplayHandler {
            rng: &mut rng,
            base: trace.clone(),
            trace: Trace::default(),
        },
        models::mixed_model(),
    );

    // Should get identical results
    assert_eq!(result2, result);
    assert_eq!(trace2.get_f64(&addr!("continuous")), Some(continuous));
    assert_eq!(trace2.get_bool(&addr!("discrete")), Some(discrete));
}

#[test]
fn test_handler_with_factors() {
    let mut rng = seeded_rng(789);

    let model_with_factor = || {
        sample(addr!("x"), Normal::new(0.0, 1.0).unwrap()).bind(|x| {
            factor(-x * x / 2.0); // Gaussian-like factor
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

    assert_finite(result);
    assert_finite(trace.log_factors);
    assert_finite(trace.log_factors); // Factor should be finite

    // Test replay preserves factors
    let (result2, trace2) = runtime::handler::run(
        ReplayHandler {
            rng: &mut rng,
            base: trace.clone(),
            trace: Trace::default(),
        },
        model_with_factor(),
    );

    assert_eq!(result2, result);
    assert_eq!(trace2.log_factors, trace.log_factors);
}

#[test]
fn test_handler_error_resilience() {
    let mut rng = seeded_rng(999);

    // Model that could have numerical issues
    let challenging_model = || {
        sample(addr!("x"), Normal::new(0.0, 1.0).unwrap()).bind(|x| {
            // Very tight observation that could cause numerical issues
            observe(addr!("y"), Normal::new(x, 1e-6).unwrap(), 100.0);
            pure(x)
        })
    };

    let (result, trace) = runtime::handler::run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        challenging_model(),
    );

    // Should still produce valid results
    assert_finite(result);
    assert_finite(trace.total_log_weight());

    // Replay should work too
    let (result2, _) = runtime::handler::run(
        ReplayHandler {
            rng: &mut rng,
            base: trace.clone(),
            trace: Trace::default(),
        },
        challenging_model(),
    );

    assert_eq!(result2, result);
}

#[test]
fn test_trace_building_patterns() {
    let mut rng = seeded_rng(1111);

    // Test incremental trace building
    let model_part1 = || sample(addr!("a"), Normal::new(0.0, 1.0).unwrap());
    let model_part2 = |a: f64| sample(addr!("b"), Normal::new(a, 0.5).unwrap());
    let model_part3 = |a: f64, b: f64| {
        observe(addr!("obs"), Normal::new(b, 0.1).unwrap(), 2.0);
        pure((a, b))
    };

    // Build trace incrementally
    let (a, trace1) = runtime::handler::run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        model_part1(),
    );

    let (b, trace2) = runtime::handler::run(
        PriorHandler {
            rng: &mut rng,
            trace: trace1.clone(),
        },
        model_part2(a),
    );

    let (result, final_trace) = runtime::handler::run(
        PriorHandler {
            rng: &mut rng,
            trace: trace2.clone(),
        },
        model_part3(a, b),
    );

    assert_eq!(result, (a, b));
    assert!(final_trace.choices.contains_key(&addr!("a")));
    assert!(final_trace.choices.contains_key(&addr!("b")));
    assert_finite(final_trace.total_log_weight());
}

#[test]
fn test_handler_memory_efficiency() {
    let mut rng = seeded_rng(2222);

    // Test with many variables to check memory usage
    let large_model = || {
        let mut models = Vec::new();
        for i in 0..100 {
            models.push(sample(addr!("x", i), Normal::new(0.0, 1.0).unwrap()));
        }
        sequence_vec(models)
    };

    let (results, trace) = runtime::handler::run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        large_model(),
    );

    assert_eq!(results.len(), 100);
    assert_eq!(trace.choices.len(), 100);

    // All results should be finite
    assert!(results.iter().all(|&x| x.is_finite()));

    // Test replay works with large traces
    let (results2, _) = runtime::handler::run(
        ReplayHandler {
            rng: &mut rng,
            base: trace.clone(),
            trace: Trace::default(),
        },
        large_model(),
    );

    assert_eq!(results2, results);
}

#[test]
fn test_trace_manipulation() {
    let mut trace = Trace::default();

    // Test direct trace manipulation
    trace.insert_choice(addr!("x"), ChoiceValue::F64(1.5), -0.5);
    trace.insert_choice(addr!("y"), ChoiceValue::Bool(true), -0.693);
    trace.log_factors = -1.2;

    // Test that total log weight combines everything
    let expected_total = trace.log_prior + trace.log_likelihood + trace.log_factors;
    assert_approx_eq(trace.total_log_weight(), expected_total, TEST_TOLERANCE);

    // Test trace cloning and modification
    let mut trace2 = trace.clone();
    trace2.insert_choice(addr!("z"), ChoiceValue::U64(42), -2.0);

    // Original trace should be unchanged
    assert!(trace.choices.contains_key(&addr!("x")));
    assert!(!trace.choices.contains_key(&addr!("z")));

    // New trace should have additional choice
    assert!(trace2.choices.contains_key(&addr!("z")));
    assert_eq!(trace2.get_u64(&addr!("z")), Some(42));
}

#[test]
fn test_interpreter_state_isolation() {
    let mut rng1 = seeded_rng(3333);
    let mut rng2 = seeded_rng(3333); // Same seed

    let model = || models::gaussian_mean(1.0);

    // Run same model with separate handlers
    let (result1, trace1) = runtime::handler::run(
        PriorHandler {
            rng: &mut rng1,
            trace: Trace::default(),
        },
        model(),
    );

    let (result2, trace2) = runtime::handler::run(
        PriorHandler {
            rng: &mut rng2,
            trace: Trace::default(),
        },
        model(),
    );

    // With same seed, should get same results
    assert_eq!(result1, result2);
    assert_eq!(trace1.get_f64(&addr!("mu")), trace2.get_f64(&addr!("mu")));

    // But handlers should be independent (different RNG state after)
    let (result3, _) = runtime::handler::run(
        PriorHandler {
            rng: &mut rng1,
            trace: Trace::default(),
        },
        model(),
    );

    let (result4, _) = runtime::handler::run(
        PriorHandler {
            rng: &mut rng2,
            trace: Trace::default(),
        },
        model(),
    );

    // Should still be same (both RNGs advanced similarly)
    assert_eq!(result3, result4);
}

#[test]
fn test_handler_with_observations() {
    let mut rng = seeded_rng(4444);

    let model_with_obs = || {
        sample(addr!("mu"), Normal::new(0.0, 2.0).unwrap()).bind(|mu| {
            sample(addr!("sigma"), LogNormal::new(0.0, 0.5).unwrap()).bind(move |sigma| {
                observe(addr!("y1"), Normal::new(mu, sigma).unwrap(), 1.5);
                observe(addr!("y2"), Normal::new(mu, sigma).unwrap(), 2.0);
                observe(addr!("y3"), Normal::new(mu, sigma).unwrap(), 1.8);
                pure((mu, sigma))
            })
        })
    };

    let (result, trace) = runtime::handler::run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        model_with_obs(),
    );

    let (mu, sigma) = result;
    assert_finite(mu);
    assert!(sigma > 0.0);

    // Should have sampling choices in trace
    assert!(trace.choices.contains_key(&addr!("mu")));
    assert!(trace.choices.contains_key(&addr!("sigma")));

    // Log likelihood should reflect the observations
    assert_finite(trace.log_likelihood);

    // Test replay preserves observations
    let (result2, trace2) = runtime::handler::run(
        ReplayHandler {
            rng: &mut rng,
            base: trace.clone(),
            trace: Trace::default(),
        },
        model_with_obs(),
    );

    assert_eq!(result2, result);
    assert_eq!(trace2.log_likelihood, trace.log_likelihood);
}

#[test]
fn test_handler_performance_characteristics() {
    let mut rng = seeded_rng(5555);

    let simple_model = || models::gaussian_mean(0.0);

    // Time multiple handler operations
    let start = std::time::Instant::now();

    for _ in 0..100 {
        let (_, trace) = runtime::handler::run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            simple_model(),
        );

        // Replay the trace
        runtime::handler::run(
            ReplayHandler {
                rng: &mut rng,
                base: trace.clone(),
                trace: Trace::default(),
            },
            simple_model(),
        );

        // Score the trace
        runtime::handler::run(
            ScoreGivenTrace {
                base: trace,
                trace: Trace::default(),
            },
            simple_model(),
        );
    }

    let elapsed = start.elapsed();

    // Should complete in reasonable time (this is a performance regression test)
    assert!(elapsed < std::time::Duration::from_secs(1));
}
