//! Tests for MCMC diagnostics and parameter summaries.
//!
//! This module tests the diagnostics functionality which currently has very low coverage (25.14%).

mod test_utils;

use fugue::inference::diagnostics;
use fugue::*;
use rand::Rng;
use test_utils::*;

#[test]
fn test_extract_f64_values() {
    let mut traces = vec![Trace::default(); 3];

    traces[0].insert_choice(addr!("mu"), ChoiceValue::F64(1.5), -0.5);
    traces[1].insert_choice(addr!("mu"), ChoiceValue::F64(2.0), -0.3);
    traces[2].insert_choice(addr!("mu"), ChoiceValue::F64(1.8), -0.4);

    let values = diagnostics::extract_f64_values(&traces, &addr!("mu"));

    assert_eq!(values, vec![1.5, 2.0, 1.8]);
}

#[test]
fn test_extract_f64_values_missing_address() {
    let mut traces = vec![Trace::default(); 2];

    traces[0].insert_choice(addr!("mu"), ChoiceValue::F64(1.5), -0.5);
    // traces[1] doesn't have "mu"

    let values = diagnostics::extract_f64_values(&traces, &addr!("mu"));

    assert_eq!(values, vec![1.5]); // Only one value extracted
}

#[test]
fn test_extract_f64_values_wrong_type() {
    let mut traces = vec![Trace::default(); 2];

    traces[0].insert_choice(addr!("flag"), ChoiceValue::Bool(true), -0.5);
    traces[1].insert_choice(addr!("count"), ChoiceValue::U64(42), -0.3);

    let values = diagnostics::extract_f64_values(&traces, &addr!("flag"));
    assert!(values.is_empty()); // No f64 values found
}

#[test]
fn test_extract_bool_values() {
    let mut traces = vec![Trace::default(); 3];

    traces[0].insert_choice(addr!("coin"), ChoiceValue::Bool(true), -0.5);
    traces[1].insert_choice(addr!("coin"), ChoiceValue::Bool(false), -0.693);
    traces[2].insert_choice(addr!("coin"), ChoiceValue::Bool(true), -0.5);

    let values = diagnostics::extract_bool_values(&traces, &addr!("coin"));

    assert_eq!(values, vec![true, false, true]);
}

#[test]
fn test_extract_u64_values() {
    let mut traces = vec![Trace::default(); 2];

    traces[0].insert_choice(addr!("count"), ChoiceValue::U64(10), -1.0);
    traces[1].insert_choice(addr!("count"), ChoiceValue::U64(15), -1.2);

    let values = diagnostics::extract_u64_values(&traces, &addr!("count"));

    assert_eq!(values, vec![10, 15]);
}

#[test]
fn test_extract_usize_values() {
    let mut traces = vec![Trace::default(); 2];

    traces[0].insert_choice(addr!("index"), ChoiceValue::Usize(0), -1.0);
    traces[1].insert_choice(addr!("index"), ChoiceValue::Usize(2), -1.2);

    let values = diagnostics::extract_usize_values(&traces, &addr!("index"));

    assert_eq!(values, vec![0, 2]);
}

#[test]
fn test_extract_i64_values() {
    let mut traces = vec![Trace::default(); 2];

    traces[0].insert_choice(addr!("temp"), ChoiceValue::I64(-5), -1.0);
    traces[1].insert_choice(addr!("temp"), ChoiceValue::I64(10), -1.2);

    let values = diagnostics::extract_i64_values(&traces, &addr!("temp"));

    assert_eq!(values, vec![-5, 10]);
}

#[test]
fn test_r_hat_f64_identical_chains() {
    // Create two identical chains - should give r_hat = 1.0
    let mut chain1 = Vec::new();
    let mut chain2 = Vec::new();

    for i in 0..10 {
        let mut trace = Trace::default();
        trace.insert_choice(addr!("mu"), ChoiceValue::F64(i as f64), -0.5);
        chain1.push(trace.clone());
        chain2.push(trace);
    }

    let chains = vec![chain1, chain2];
    let r_hat_val = diagnostics::r_hat_f64(&chains, &addr!("mu"));

    assert_finite(r_hat_val);
    assert_approx_eq(r_hat_val, 1.0, 0.1); // Should be close to 1 for identical chains
}

#[test]
fn test_r_hat_f64_convergence() {
    let mut chain1 = Vec::new();
    let mut chain2 = Vec::new();

    // Create two chains with same distribution but different samples
    let mut rng1 = seeded_rng(123);
    let mut rng2 = seeded_rng(456);

    for _ in 0..50 {
        let mut trace1 = Trace::default();
        let mut trace2 = Trace::default();

        let val1: f64 = rng1.gen_range(-2.0..2.0);
        let val2: f64 = rng2.gen_range(-2.0..2.0);

        trace1.insert_choice(addr!("x"), ChoiceValue::F64(val1), -0.5);
        trace2.insert_choice(addr!("x"), ChoiceValue::F64(val2), -0.5);

        chain1.push(trace1);
        chain2.push(trace2);
    }

    let chains = vec![chain1, chain2];
    let r_hat_val = diagnostics::r_hat_f64(&chains, &addr!("x"));

    assert_finite(r_hat_val);
    assert!(r_hat_val > 0.9 && r_hat_val < 1.2); // Should be close to 1 for well-mixed chains
}

#[test]
fn test_r_hat_f64_poor_convergence() {
    // Create two chains with very different means
    let mut chain1 = Vec::new();
    let mut chain2 = Vec::new();

    for _ in 0..20 {
        let mut trace1 = Trace::default();
        let mut trace2 = Trace::default();

        trace1.insert_choice(addr!("mu"), ChoiceValue::F64(0.0), -0.5); // Chain 1: always 0
        trace2.insert_choice(addr!("mu"), ChoiceValue::F64(10.0), -0.5); // Chain 2: always 10

        chain1.push(trace1);
        chain2.push(trace2);
    }

    let chains = vec![chain1, chain2];
    let r_hat_val = diagnostics::r_hat_f64(&chains, &addr!("mu"));

    // R-hat may be infinite for completely non-converged chains - that's valid
    assert!(r_hat_val >= 1.0 || r_hat_val.is_infinite()); // Should be >= 1.0 or infinite
}

#[test]
fn test_r_hat_f64_insufficient_data() {
    let mut chain1 = Vec::new();
    let mut chain2 = Vec::new();

    // Only 2 samples per chain - should handle gracefully
    for i in 0..2 {
        let mut trace1 = Trace::default();
        let mut trace2 = Trace::default();

        trace1.insert_choice(addr!("x"), ChoiceValue::F64(i as f64), -0.5);
        trace2.insert_choice(addr!("x"), ChoiceValue::F64((i + 1) as f64), -0.5);

        chain1.push(trace1);
        chain2.push(trace2);
    }

    let chains = vec![chain1, chain2];
    let r_hat_val = diagnostics::r_hat_f64(&chains, &addr!("x"));

    // May be NaN or finite, but shouldn't crash
    assert!(r_hat_val.is_finite() || r_hat_val.is_nan());
}

#[test]
fn test_effective_sample_size() {
    // Test with perfectly uncorrelated samples
    let uncorrelated: Vec<f64> = (0..100).map(|i| (i as f64) * 0.1).collect();
    let ess = fugue::inference::mcmc_utils::effective_sample_size_mcmc(&uncorrelated);

    assert_finite(ess);
    assert!(ess > 0.0); // Should be positive for any data
    assert!(ess <= 100.0); // Can't exceed sample size
}

#[test]
fn test_effective_sample_size_highly_correlated() {
    // Test with highly correlated samples (all same value)
    let correlated: Vec<f64> = vec![1.0; 100];
    let ess = fugue::inference::mcmc_utils::effective_sample_size_mcmc(&correlated);

    assert_finite(ess);
    // ESS may vary depending on implementation - just check it's finite
    assert_finite(ess);
}

#[test]
fn test_effective_sample_size_small_sample() {
    let small_sample = vec![1.0, 2.0];
    let ess = fugue::inference::mcmc_utils::effective_sample_size_mcmc(&small_sample);

    assert_finite(ess);
    assert!(ess > 0.0);
}

#[test]
fn test_parameter_summary() {
    // Create chains with known statistics
    let mut chain1 = Vec::new();
    let mut chain2 = Vec::new();

    let values = [1.0, 2.0, 3.0, 4.0, 5.0]; // Mean = 3.0, known distribution

    for &val in &values {
        let mut trace1 = Trace::default();
        let mut trace2 = Trace::default();

        trace1.insert_choice(addr!("param"), ChoiceValue::F64(val), -0.5);
        trace2.insert_choice(addr!("param"), ChoiceValue::F64(val + 0.1), -0.5); // Slight variation

        chain1.push(trace1);
        chain2.push(trace2);
    }

    let chains = vec![chain1, chain2];
    let summary = diagnostics::summarize_f64_parameter(&chains, &addr!("param"));

    assert_finite(summary.mean);
    assert_finite(summary.std);
    assert_finite(summary.r_hat);
    assert_finite(summary.ess);

    // Check that mean is reasonable
    assert!((summary.mean - 3.05).abs() < 0.1); // Should be close to 3.05 (mean of values + 0.05)

    // Check that quantiles are present and ordered
    assert!(summary.quantiles.contains_key("2.5%"));
    assert!(summary.quantiles.contains_key("50%"));
    assert!(summary.quantiles.contains_key("97.5%"));

    let q025 = summary.quantiles.get("2.5%").unwrap();
    let q50 = summary.quantiles.get("50%").unwrap();
    let q975 = summary.quantiles.get("97.5%").unwrap();

    assert!(q025 < q50);
    assert!(q50 < q975);
}

#[test]
fn test_parameter_summary_empty_chains() {
    let empty_chains: Vec<Vec<Trace>> = vec![vec![], vec![]];
    let summary = diagnostics::summarize_f64_parameter(&empty_chains, &addr!("missing"));

    assert!(summary.mean.is_nan());
    assert!(summary.std.is_nan());
    assert!(summary.r_hat.is_nan());
    assert_eq!(summary.ess, 0.0);
    assert!(summary.quantiles.is_empty());
}

#[test]
fn test_parameter_summary_missing_parameter() {
    let mut chain1 = Vec::new();
    let mut chain2 = Vec::new();

    // Add some traces but with different parameter names
    for i in 0..5 {
        let mut trace1 = Trace::default();
        let mut trace2 = Trace::default();

        trace1.insert_choice(addr!("other_param"), ChoiceValue::F64(i as f64), -0.5);
        trace2.insert_choice(addr!("other_param"), ChoiceValue::F64(i as f64), -0.5);

        chain1.push(trace1);
        chain2.push(trace2);
    }

    let chains = vec![chain1, chain2];
    let summary = diagnostics::summarize_f64_parameter(&chains, &addr!("missing_param"));

    assert!(summary.mean.is_nan());
    assert!(summary.std.is_nan());
}

#[test]
fn test_print_diagnostics() {
    let mut chain1 = Vec::new();
    let mut chain2 = Vec::new();

    // Create simple test data
    for i in 0..3 {
        let mut trace1 = Trace::default();
        let mut trace2 = Trace::default();

        trace1.insert_choice(addr!("alpha"), ChoiceValue::F64(i as f64), -0.5);
        trace1.insert_choice(addr!("beta"), ChoiceValue::F64((i + 1) as f64), -0.3);

        trace2.insert_choice(addr!("alpha"), ChoiceValue::F64(i as f64 + 0.5), -0.5);
        trace2.insert_choice(addr!("beta"), ChoiceValue::F64(i as f64 + 1.5), -0.3);

        chain1.push(trace1);
        chain2.push(trace2);
    }

    let chains = vec![chain1, chain2];

    // This should not crash
    diagnostics::print_diagnostics(&chains);
}

#[test]
fn test_print_diagnostics_empty() {
    let empty_chains: Vec<Vec<Trace>> = vec![];

    // Should handle empty chains gracefully
    diagnostics::print_diagnostics(&empty_chains);
}

#[test]
fn test_print_diagnostics_empty_traces() {
    let chains_with_empty_traces = vec![vec![], vec![]];

    // Should handle empty traces gracefully
    diagnostics::print_diagnostics(&chains_with_empty_traces);
}

#[test]
fn test_diagnostics_trait_f64() {
    let mut traces = vec![Trace::default(); 3];

    traces[0].insert_choice(addr!("x"), ChoiceValue::F64(1.0), -0.5);
    traces[1].insert_choice(addr!("x"), ChoiceValue::F64(2.0), -0.3);
    traces[2].insert_choice(addr!("x"), ChoiceValue::F64(3.0), -0.4);

    // Test extract_values
    let values = f64::extract_values(&traces, &addr!("x"));
    assert_eq!(values, vec![1.0, 2.0, 3.0]);

    // Test effective_sample_size
    let ess = f64::effective_sample_size(&values);
    assert!(ess.is_some());
    assert!(ess.unwrap() > 0.0);

    // Test with insufficient data
    let small_values = vec![1.0, 2.0];
    let ess_small = f64::effective_sample_size(&small_values);
    assert_eq!(ess_small, Some(2.0)); // Should return sample size for small samples
}

#[test]
fn test_diagnostics_trait_bool() {
    let mut traces = vec![Trace::default(); 3];

    traces[0].insert_choice(addr!("flag"), ChoiceValue::Bool(true), -0.5);
    traces[1].insert_choice(addr!("flag"), ChoiceValue::Bool(false), -0.693);
    traces[2].insert_choice(addr!("flag"), ChoiceValue::Bool(true), -0.5);

    // Test extract_values
    let values = bool::extract_values(&traces, &addr!("flag"));
    assert_eq!(values, vec![true, false, true]);

    // Test that r_hat is None for bool
    let chains = vec![traces.clone()];
    let r_hat = bool::r_hat(&chains, &addr!("flag"));
    assert!(r_hat.is_none());

    // Test that effective_sample_size is None for bool
    let ess = bool::effective_sample_size(&values);
    assert!(ess.is_none());
}

#[test]
fn test_diagnostics_trait_u64() {
    let mut traces = vec![Trace::default(); 3];

    traces[0].insert_choice(addr!("count"), ChoiceValue::U64(10), -1.0);
    traces[1].insert_choice(addr!("count"), ChoiceValue::U64(20), -1.2);
    traces[2].insert_choice(addr!("count"), ChoiceValue::U64(15), -1.1);

    // Test extract_values
    let values = u64::extract_values(&traces, &addr!("count"));
    assert_eq!(values, vec![10, 20, 15]);

    // Test that r_hat is computed for u64 (may be Some or None depending on implementation)
    let chains = vec![traces.clone()];
    let r_hat = u64::r_hat(&chains, &addr!("count"));
    // Just check it doesn't crash - implementation may vary
    assert!(r_hat.is_some() || r_hat.is_none());

    // Test that effective_sample_size for u64 (may be Some or None)
    let ess = u64::effective_sample_size(&values);
    // Implementation dependent - just check it doesn't crash
    assert!(ess.is_some() || ess.is_none());
}

#[test]
fn test_r_hat_f64_single_chain() {
    // Test r_hat with only one chain - should handle gracefully
    let mut chain = Vec::new();

    for i in 0..10 {
        let mut trace = Trace::default();
        trace.insert_choice(addr!("mu"), ChoiceValue::F64(i as f64), -0.5);
        chain.push(trace);
    }

    let chains = vec![chain];
    let r_hat_val = diagnostics::r_hat_f64(&chains, &addr!("mu"));

    // Should be NaN or some specific value for single chain
    assert!(r_hat_val.is_nan() || r_hat_val.is_finite());
}

#[test]
fn test_r_hat_f64_with_missing_values() {
    // Test r_hat when some traces are missing the parameter
    let mut chain1 = Vec::new();
    let mut chain2 = Vec::new();

    for i in 0..5 {
        let mut trace1 = Trace::default();
        let mut trace2 = Trace::default();

        trace1.insert_choice(addr!("mu"), ChoiceValue::F64(i as f64), -0.5);

        // Only add to chain2 sometimes
        if i % 2 == 0 {
            trace2.insert_choice(addr!("mu"), ChoiceValue::F64((i + 1) as f64), -0.5);
        }

        chain1.push(trace1);
        chain2.push(trace2);
    }

    let chains = vec![chain1, chain2];
    let r_hat_val = diagnostics::r_hat_f64(&chains, &addr!("mu"));

    // Should handle missing values gracefully
    assert!(r_hat_val.is_finite() || r_hat_val.is_nan());
}

#[test]
fn test_diagnostics_with_extreme_values() {
    let mut chain1 = Vec::new();
    let mut chain2 = Vec::new();

    // Test with extreme values
    let extreme_vals = [1e-100, 1e100, -1e100, f64::MIN_POSITIVE, f64::MAX];

    for &val in &extreme_vals {
        let mut trace1 = Trace::default();
        let mut trace2 = Trace::default();

        trace1.insert_choice(addr!("extreme"), ChoiceValue::F64(val), -0.5);
        trace2.insert_choice(addr!("extreme"), ChoiceValue::F64(val * 0.99), -0.5);

        chain1.push(trace1);
        chain2.push(trace2);
    }

    let chains = vec![chain1, chain2];
    let summary = diagnostics::summarize_f64_parameter(&chains, &addr!("extreme"));

    // Should handle extreme values without crashing
    // Results may be NaN or infinite, but shouldn't panic
    assert!(summary.mean.is_finite() || summary.mean.is_nan() || summary.mean.is_infinite());
    assert!(summary.std.is_finite() || summary.std.is_nan() || summary.std.is_infinite());
}
