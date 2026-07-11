//! Exact-value tests for the core numerically-stable primitives in
//! `src/core/numerical.rs`: `log_sum_exp`, `normalize_log_probs`, and
//! `log1p_exp`. These underlie importance-weight normalization, ESS, and SMC
//! resampling, so they are checked against hand-computable reference values —
//! including extreme spreads, an all-`-inf` input, and single-element input —
//! not merely for finiteness.
//!
//! Covers finding FG-32.

use fugue::{log1p_exp, log_sum_exp, normalize_log_probs};

const TOL: f64 = 1e-9;

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < TOL,
        "expected {expected}, got {actual} (|Δ| = {})",
        (actual - expected).abs()
    );
}

#[test]
fn fg32_log_sum_exp_exact_values() {
    // 701 + ln(e^-1 + 1 + e^-2) = 701.4076059644444
    close(log_sum_exp(&[700.0, 701.0, 699.0]), 701.4076059644444);

    // Single element: log_sum_exp([x]) == x exactly.
    close(log_sum_exp(&[5.0]), 5.0);
    close(log_sum_exp(&[-123.75]), -123.75);

    // Extreme spread on the high end (max-factoring must avoid overflow):
    // 1001 + ln(1 + e^-1).
    close(log_sum_exp(&[1000.0, 1001.0]), 1001.3132616875182);

    // Extreme spread where the small term underflows: ln(1 + e^-1000) == 0.
    close(log_sum_exp(&[0.0, -1000.0]), 0.0);

    // All -inf -> -inf (degenerate).
    assert_eq!(
        log_sum_exp(&[f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY]),
        f64::NEG_INFINITY
    );
    // Empty slice -> -inf.
    assert_eq!(log_sum_exp(&[]), f64::NEG_INFINITY);

    // A single -inf mixed with finite values is ignored, not poisoning.
    close(log_sum_exp(&[f64::NEG_INFINITY, 0.0]), 0.0);
}

#[test]
fn fg32_normalize_log_probs_exact_ratios() {
    let probs = normalize_log_probs(&[-1.0, -2.0, -3.0]);
    // Exact softmax of (-1,-2,-3).
    close(probs[0], 0.6652409557748219);
    close(probs[1], 0.24472847105479764);
    close(probs[2], 0.09003057317038043);
    // Must sum to exactly 1 (within fp tolerance).
    close(probs.iter().sum::<f64>(), 1.0);
    // The ratio of adjacent entries is exp((-1) - (-2)) = e, not merely ordered.
    close(probs[0] / probs[1], std::f64::consts::E);
    close(probs[1] / probs[2], std::f64::consts::E);
}

#[test]
fn fg32_normalize_log_probs_uniform_input() {
    // Equal log-probs normalize to a uniform vector.
    let probs = normalize_log_probs(&[3.0, 3.0, 3.0, 3.0]);
    for &p in &probs {
        close(p, 0.25);
    }
}

#[test]
fn fg32_log1p_exp_exact_values() {
    // log1p_exp(0) = ln(2).
    close(log1p_exp(0.0), std::f64::consts::LN_2);
    // Mid-range uses ln_1p: ln(1 + e^2).
    close(log1p_exp(2.0), 2.1269280110429727);
    // Large x saturates to x (1 + e^x ~= e^x): true value is x + ln(1+e^-x) ~= x.
    close(log1p_exp(100.0), 100.0);
    // Very negative x: ln(1 + e^x) ~= e^x.
    close(log1p_exp(-50.0), 1.9287498479639178e-22);
    // Deep underflow: ln(1 + e^-1000) == 0.
    close(log1p_exp(-1000.0), 0.0);
}
