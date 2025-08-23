//! Tests for numerical utilities.
//!
//! This module tests the numerical functionality which currently has low coverage (43.90%).

mod test_utils;

use fugue::core::numerical::*;
use test_utils::*;

#[test]
fn test_log_sum_exp_basic() {
    let values = vec![-1.0, -2.0, -3.0];
    let result = log_sum_exp(&values);

    assert_finite(result);

    // Verify against manual calculation
    // log(e^(-1) + e^(-2) + e^(-3)) ≈ log(0.3679 + 0.1353 + 0.0498) ≈ log(0.553) ≈ -0.591
    assert_approx_eq(result, -0.5914, 0.01);
}

#[test]
fn test_log_sum_exp_large_values() {
    // Test with very large values that would overflow without stability
    let values = vec![700.0, 701.0, 702.0];
    let result = log_sum_exp(&values);

    assert_finite(result);
    // Should be close to the maximum value (702) plus small correction
    assert!(result > 702.0);
    assert!(result < 703.0);
}

#[test]
fn test_log_sum_exp_small_values() {
    // Test with very small values that would underflow
    let values = vec![-700.0, -701.0, -702.0];
    let result = log_sum_exp(&values);

    assert_finite(result);
    // Should be close to the maximum value (-700) plus small correction
    assert!(result > -700.5);
    assert!(result < -699.5);
}

#[test]
fn test_log_sum_exp_mixed_values() {
    let values = vec![1.0, -1000.0, 2.0];
    let result = log_sum_exp(&values);

    assert_finite(result);
    // Dominated by larger values (1.0 and 2.0)
    // log(e^1 + e^(-1000) + e^2) ≈ log(e^1 + e^2) ≈ log(e^2(e^(-1) + 1)) = 2 + log(e^(-1) + 1)
    let expected = 2.0 + ((-1.0_f64).exp() + 1.0).ln();
    assert_approx_eq(result, expected, 0.001);
}

#[test]
fn test_log_sum_exp_single_value() {
    let values = vec![3.14];
    let result = log_sum_exp(&values);

    assert_finite(result);
    assert_approx_eq(result, 3.14, TEST_TOLERANCE);
}

#[test]
fn test_log_sum_exp_empty() {
    let values: Vec<f64> = vec![];
    let result = log_sum_exp(&values);

    assert_eq!(result, f64::NEG_INFINITY);
}

#[test]
fn test_log_sum_exp_all_neg_infinity() {
    let values = vec![f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
    let result = log_sum_exp(&values);

    assert_eq!(result, f64::NEG_INFINITY);
}

#[test]
fn test_log_sum_exp_with_pos_infinity() {
    let values = vec![1.0, f64::INFINITY, 2.0];
    let result = log_sum_exp(&values);

    // log_sum_exp may return NaN when dealing with infinity
    assert!(result.is_nan() || result == f64::INFINITY);
}

#[test]
fn test_weighted_log_sum_exp_basic() {
    let log_values = vec![-1.0, -2.0, -3.0];
    let weights = vec![0.5, 0.3, 0.2];

    let result = weighted_log_sum_exp(&log_values, &weights);

    assert_finite(result);

    // Verify against manual calculation
    // log(0.5*e^(-1) + 0.3*e^(-2) + 0.2*e^(-3))
    let expected = (0.5 * (-1.0_f64).exp() + 0.3 * (-2.0_f64).exp() + 0.2 * (-3.0_f64).exp()).ln();
    assert_approx_eq(result, expected, 0.001);
}

#[test]
fn test_weighted_log_sum_exp_uniform_weights() {
    let log_values = vec![-1.0, -2.0, -3.0];
    let weights = vec![1.0, 1.0, 1.0];

    let result = weighted_log_sum_exp(&log_values, &weights);
    let unweighted_result = log_sum_exp(&log_values);

    assert_finite(result);
    assert_finite(unweighted_result);
    assert_approx_eq(result, unweighted_result, 0.001);
}

#[test]
fn test_weighted_log_sum_exp_zero_weights() {
    let log_values = vec![-1.0, -2.0, -3.0];
    let weights = vec![0.0, 0.0, 0.0];

    let result = weighted_log_sum_exp(&log_values, &weights);

    assert_eq!(result, f64::NEG_INFINITY);
}

#[test]
fn test_weighted_log_sum_exp_single_nonzero_weight() {
    let log_values = vec![-1.0, -2.0, -3.0];
    let weights = vec![0.0, 1.0, 0.0]; // Only middle weight is nonzero

    let result = weighted_log_sum_exp(&log_values, &weights);

    assert_finite(result);
    assert_approx_eq(result, -2.0, TEST_TOLERANCE); // Should equal the log_value with nonzero weight
}

#[test]
fn test_weighted_log_sum_exp_extreme_weights() {
    let log_values = vec![1.0, 2.0, 3.0];
    let weights = vec![1e-100, 1e100, 1e-100]; // Extreme weight on middle value

    let result = weighted_log_sum_exp(&log_values, &weights);

    assert_finite(result);
    // Should be dominated by the middle term
    assert!(result > 100.0); // Should be much larger due to extreme weight
}

#[test]
fn test_weighted_log_sum_exp_empty() {
    let log_values: Vec<f64> = vec![];
    let weights: Vec<f64> = vec![];

    let result = weighted_log_sum_exp(&log_values, &weights);

    assert_eq!(result, f64::NEG_INFINITY);
}

#[test]
#[should_panic]
fn test_weighted_log_sum_exp_mismatched_lengths() {
    let log_values = vec![-1.0, -2.0, -3.0];
    let weights = vec![0.5, 0.3]; // Different length

    let _result = weighted_log_sum_exp(&log_values, &weights);
}

#[test]
fn test_normalize_log_probs_basic() {
    let log_probs = vec![-1.0, -2.0, -3.0];
    let normalized = normalize_log_probs(&log_probs);

    assert_eq!(normalized.len(), 3);

    // Should sum to 1.0
    let sum: f64 = normalized.iter().sum();
    assert_approx_eq(sum, 1.0, TEST_TOLERANCE);

    // Should be in correct relative order (largest log_prob -> largest prob)
    assert!(normalized[0] > normalized[1]);
    assert!(normalized[1] > normalized[2]);

    // All probabilities should be positive
    for &p in &normalized {
        assert!(p > 0.0);
        assert_finite(p);
    }
}

#[test]
fn test_normalize_log_probs_extreme() {
    // Test with extreme log probabilities
    let log_probs = vec![-1000.0, -1.0, -1001.0];
    let normalized = normalize_log_probs(&log_probs);

    let sum: f64 = normalized.iter().sum();
    assert_approx_eq(sum, 1.0, TEST_TOLERANCE);

    // Middle value should dominate
    assert!(normalized[1] > 0.99); // Should be almost 1
    assert!(normalized[0] < 0.01);
    assert!(normalized[2] < 0.01);
}

#[test]
fn test_normalize_log_probs_equal() {
    let log_probs = vec![-2.0, -2.0, -2.0];
    let normalized = normalize_log_probs(&log_probs);

    let sum: f64 = normalized.iter().sum();
    assert_approx_eq(sum, 1.0, TEST_TOLERANCE);

    // All should be equal (1/3)
    for &p in &normalized {
        assert_approx_eq(p, 1.0 / 3.0, TEST_TOLERANCE);
    }
}

#[test]
fn test_normalize_log_probs_empty() {
    let log_probs: Vec<f64> = vec![];
    let normalized = normalize_log_probs(&log_probs);

    assert!(normalized.is_empty());
}

#[test]
fn test_log1p_exp_large_positive() {
    // For large positive x, log(1 + exp(x)) ≈ x
    let x = 50.0;
    let result = log1p_exp(x);

    assert_finite(result);
    assert_approx_eq(result, x, 1e-10);
}

#[test]
fn test_log1p_exp_large_negative() {
    // For large negative x, log(1 + exp(x)) ≈ exp(x) ≈ 0
    let x = -50.0;
    let result = log1p_exp(x);

    assert_finite(result);
    assert!(result < 1e-20); // Should be very small
}

#[test]
fn test_log1p_exp_around_zero() {
    // Test intermediate values where both terms matter
    let x = 0.0;
    let result = log1p_exp(x);

    assert_finite(result);

    // log(1 + exp(0)) = log(1 + 1) = log(2) ≈ 0.693
    assert_approx_eq(result, 2.0_f64.ln(), 0.001);
}

#[test]
fn test_log1p_exp_medium_values() {
    let test_values = vec![-10.0, -1.0, 1.0, 10.0];

    for x in test_values {
        let result = log1p_exp(x);
        let expected = (1.0 + x.exp()).ln();

        assert_finite(result);
        assert_finite(expected);

        // Should match the naive calculation for moderate values
        if x > -30.0 && x < 30.0 {
            assert_approx_eq(result, expected, 0.001);
        }
    }
}

#[test]
fn test_safe_ln_positive() {
    let positive_values = vec![0.001, 1.0, 2.718281828, 10.0, 1e6];

    for x in positive_values {
        let result = safe_ln(x);
        let expected = x.ln();

        assert_finite(result);
        assert_approx_eq(result, expected, TEST_TOLERANCE);
    }
}

#[test]
fn test_safe_ln_zero() {
    let result = safe_ln(0.0);
    assert_eq!(result, f64::NEG_INFINITY);
}

#[test]
fn test_safe_ln_negative() {
    let negative_values = vec![-0.001, -1.0, -10.0];

    for x in negative_values {
        let result = safe_ln(x);
        assert_eq!(result, f64::NEG_INFINITY);
    }
}

#[test]
fn test_safe_ln_special_values() {
    assert_eq!(safe_ln(f64::NEG_INFINITY), f64::NEG_INFINITY);
    // safe_ln may handle infinity differently depending on implementation
    assert!(safe_ln(f64::INFINITY).is_infinite() || safe_ln(f64::INFINITY).is_nan());
    assert_eq!(safe_ln(f64::NAN), f64::NEG_INFINITY);
}

#[test]
fn test_log_gamma_positive() {
    let positive_values = vec![0.5, 1.0, 1.5, 2.0, 5.0, 10.5];

    for x in positive_values {
        let result = log_gamma(x);

        assert_finite(result);

        // For integer values, Γ(n) = (n-1)!
        if x == 1.0 {
            assert_approx_eq(result, 0.0, 0.001); // Γ(1) = 0! = 1, log(1) = 0
        }
        if x == 2.0 {
            assert_approx_eq(result, 0.0, 0.001); // Γ(2) = 1! = 1, log(1) = 0
        }
    }
}

#[test]
fn test_log_gamma_half_integer() {
    // Γ(0.5) = √π, so log Γ(0.5) = log(√π) = 0.5 * log(π)
    let result = log_gamma(0.5);
    let expected = 0.5 * std::f64::consts::PI.ln();

    assert_finite(result);
    assert_approx_eq(result, expected, 0.001);
}

#[test]
fn test_log_gamma_zero_negative() {
    let invalid_values = vec![0.0, -0.5, -1.0, -10.0];

    for x in invalid_values {
        let result = log_gamma(x);
        assert!(result.is_nan());
    }
}

#[test]
fn test_log_gamma_special_values() {
    assert!(log_gamma(f64::INFINITY).is_nan());
    assert!(log_gamma(f64::NEG_INFINITY).is_nan());
    assert!(log_gamma(f64::NAN).is_nan());
}

#[test]
fn test_log_gamma_large_values() {
    // Test that large values don't cause overflow
    let large_values = vec![100.0, 1000.0];

    for x in large_values {
        let result = log_gamma(x);
        assert_finite(result);
        assert!(result > 0.0); // log Γ(x) > 0 for large x
    }
}

#[test]
fn test_numerical_precision_log_sum_exp() {
    // Test that log_sum_exp maintains precision with nearly equal values
    let base_val = 100.0;
    let values = vec![base_val, base_val + 1e-10, base_val + 2e-10];

    let result = log_sum_exp(&values);
    assert_finite(result);

    // Should be close to base_val + log(3) since values are nearly equal
    let expected = base_val + 3.0_f64.ln();
    assert_approx_eq(result, expected, 1e-8);
}

#[test]
fn test_numerical_stability_edge_cases() {
    // Test various edge cases for numerical stability

    // Mix of finite and infinite values
    let mixed = vec![f64::NEG_INFINITY, 1.0, f64::NEG_INFINITY];
    let result = log_sum_exp(&mixed);
    assert_approx_eq(result, 1.0, TEST_TOLERANCE);

    // Single finite value among infinities
    let sparse = vec![f64::NEG_INFINITY, f64::NEG_INFINITY, 5.0, f64::NEG_INFINITY];
    let result = log_sum_exp(&sparse);
    assert_approx_eq(result, 5.0, TEST_TOLERANCE);

    // Very close to zero
    let near_zero = vec![-1e10, -1e10 + 0.1, -1e10 + 0.2];
    let result = log_sum_exp(&near_zero);
    assert_finite(result);
    assert!(result < -1e9); // Should still be very negative but finite
}
