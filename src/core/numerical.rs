//!Numerical utilities for stable probabilistic computation.
//!
//! This module provides numerically stable implementations of common operations in probabilistic programming.
//! Proper numerical stability is crucial for reliable inference, especially when dealing with extreme probabilities.

/// Compute log(sum(exp(x_i))) stably by factoring out the max; returns -∞ if all inputs are -∞.
///
/// Example:
/// ```rust
/// # use fugue::core::numerical::log_sum_exp;
/// let xs = [-1.0, -2.0, -3.0];
/// let y = log_sum_exp(&xs);
/// assert!((y - (-0.5914)).abs() < 1e-2);
/// ```
pub fn log_sum_exp(log_values: &[f64]) -> f64 {
    if log_values.is_empty() {
        return f64::NEG_INFINITY;
    }

    // Find maximum value
    let max_val = log_values
        .iter()
        .fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));

    // Handle case where all values are -∞
    if max_val.is_infinite() && max_val < 0.0 {
        return f64::NEG_INFINITY;
    }

    // Compute sum(exp(x_i - max)) stably
    let sum_exp: f64 = log_values.iter().map(|&x| (x - max_val).exp()).sum();

    if sum_exp == 0.0 {
        f64::NEG_INFINITY
    } else {
        max_val + sum_exp.ln()
    }
}

/// Compute log(sum(w_i * exp(x_i))) stably for weighted log-sum-exp.
///
/// This generalizes log_sum_exp to handle weighted sums, commonly needed in importance sampling and particle filtering.
///
/// Example:
/// ```rust
/// # use fugue::core::numerical::weighted_log_sum_exp;
/// let log_values = vec![-1.0, -2.0, -3.0];
/// let weights = vec![0.5, 0.3, 0.2];
/// let result = weighted_log_sum_exp(&log_values, &weights);
/// ```
pub fn weighted_log_sum_exp(log_values: &[f64], weights: &[f64]) -> f64 {
    assert_eq!(log_values.len(), weights.len());

    if log_values.is_empty() {
        return f64::NEG_INFINITY;
    }

    let max_val = log_values
        .iter()
        .fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));

    if max_val.is_infinite() && max_val < 0.0 {
        return f64::NEG_INFINITY;
    }

    let weighted_sum: f64 = log_values
        .iter()
        .zip(weights.iter())
        .map(|(&x, &w)| w * (x - max_val).exp())
        .sum();

    if weighted_sum == 0.0 {
        f64::NEG_INFINITY
    } else {
        max_val + weighted_sum.ln()
    }
}

/// Normalize log-probabilities to linear probabilities stably.
///
/// Example:
/// ```rust
/// # use fugue::core::numerical::normalize_log_probs;
/// let log_probs = vec![-1.0, -2.0, -3.0];
/// let normalized = normalize_log_probs(&log_probs);
/// ```
pub fn normalize_log_probs(log_probs: &[f64]) -> Vec<f64> {
    let log_sum = log_sum_exp(log_probs);
    log_probs.iter().map(|&lp| (lp - log_sum).exp()).collect()
}

/// Compute log(1 + exp(x)) stably to avoid overflow.
///
/// Example:
/// ```rust
/// # use fugue::core::numerical::log1p_exp;
/// let x = -100.0;
/// let y = log1p_exp(x);
/// assert!(y.abs() < 1e-40);
/// ```
pub fn log1p_exp(x: f64) -> f64 {
    if x > 33.3 {
        // For large x, 1 + exp(x) ≈ exp(x), so log(1 + exp(x)) ≈ x
        x
    } else if x > -37.0 {
        // Use built-in log1p for stability
        x.exp().ln_1p()
    } else {
        // For very negative x, exp(x) ≈ 0, so log(1 + exp(x)) ≈ log(1) = 0
        x.exp()
    }
}

/// Safe logarithm that handles edge cases gracefully, returns -∞ for non-positive inputs instead of NaN or panicking.
///
/// Example:
/// ```rust
/// # use fugue::core::numerical::safe_ln;
/// let x = 1.0;
/// let y = safe_ln(x);
/// assert_eq!(y, 0.0);
/// ```
pub fn safe_ln(x: f64) -> f64 {
    if x <= 0.0 || !x.is_finite() {
        f64::NEG_INFINITY
    } else {
        x.ln()
    }
}

/// Numerically stable computation of log(Γ(x)) for gamma function.
///
/// Example:
/// ```rust
/// # use fugue::core::numerical::log_gamma;
/// let x = 1.0;
/// let y = log_gamma(x);
/// assert!((y - 0.0).abs() < 1e-10);
/// ```
pub fn log_gamma(x: f64) -> f64 {
    if x <= 0.0 || !x.is_finite() {
        f64::NAN
    } else {
        libm::lgamma(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_sum_exp_stability() {
        // Test with extreme values
        let large_vals = vec![700.0, 701.0, 699.0];
        let result = log_sum_exp(&large_vals);
        assert!(result.is_finite());

        // Test with small values
        let small_vals = vec![-700.0, -701.0, -699.0];
        let result = log_sum_exp(&small_vals);
        assert!(result.is_finite());

        // Test empty case
        assert_eq!(log_sum_exp(&[]), f64::NEG_INFINITY);

        // Test all -∞
        assert_eq!(
            log_sum_exp(&[f64::NEG_INFINITY, f64::NEG_INFINITY]),
            f64::NEG_INFINITY
        );
    }

    #[test]
    fn test_normalize_log_probs() {
        let log_probs = vec![-1.0, -2.0, -3.0];
        let probs = normalize_log_probs(&log_probs);

        // Should sum to 1.0
        assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-10);

        // Should be in correct ratios
        assert!(probs[0] > probs[1]);
        assert!(probs[1] > probs[2]);
    }

    #[test]
    fn test_log1p_exp_stability() {
        // Test extreme cases
        assert!((log1p_exp(50.0) - 50.0).abs() < 1e-10);
        assert!(log1p_exp(-50.0) < 1e-10);
        assert!(log1p_exp(0.0).abs() < 1.0);
    }

    #[test]
    fn test_weighted_log_sum_exp_more_edges() {
        // Mixed signs and zeros
        let logs = vec![-1000.0, 0.0, -10.0];
        let weights = vec![0.0, 1.0, 0.0];
        let res = weighted_log_sum_exp(&logs, &weights);
        assert!((res - 0.0).abs() < 1e-12);

        let weights2 = vec![0.5, 0.5, 0.0];
        let res2 = weighted_log_sum_exp(&logs, &weights2);
        assert!(res2.is_finite());
    }

    #[test]
    fn test_safe_ln_edges() {
        assert_eq!(safe_ln(-1.0), f64::NEG_INFINITY);
        assert_eq!(safe_ln(f64::INFINITY), f64::NEG_INFINITY);
        assert_eq!(safe_ln(1.0), 0.0);
    }
}
