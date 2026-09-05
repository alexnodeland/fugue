//!Numerical utilities for stable probabilistic computation.
//!
//! This module provides numerically stable implementations of common operations in probabilistic programming.
//! Proper numerical stability is crucial for reliable inference, especially when dealing with extreme probabilities.

/// Map a `NaN` log-weight to `-∞`; every other value passes through unchanged.
///
/// A `NaN` log-density is an *invalid* term, and the only interpretation under
/// which every consumer stays well-defined is "probability zero" (FG-N2):
/// added into an accumulator it would otherwise poison the whole trace weight,
/// inside [`log_sum_exp`] it would make a particle population's normalizer
/// `NaN` (so SMC's ESS becomes non-finite and the tempering ladder silently
/// jumps to `β = 1` with uniform weights), and in a Metropolis ratio it makes
/// every comparison false so the chain can never move. The accumulation points
/// (`factor`, the `on_factor`/`on_observe_*` handler methods) and the log-space
/// reductions all route through this.
///
/// ```rust
/// # use fugue::core::numerical::nan_to_neg_inf;
/// assert_eq!(nan_to_neg_inf(f64::NAN), f64::NEG_INFINITY);
/// assert_eq!(nan_to_neg_inf(-2.5), -2.5);
/// assert_eq!(nan_to_neg_inf(f64::INFINITY), f64::INFINITY);
/// ```
#[inline]
pub fn nan_to_neg_inf(logw: f64) -> f64 {
    if logw.is_nan() {
        f64::NEG_INFINITY
    } else {
        logw
    }
}

/// Compute log(sum(exp(x_i))) stably by factoring out the max; returns -∞ if all inputs are -∞.
///
/// `NaN` inputs are treated as `-∞` (a zero-probability term, FG-N2) rather
/// than propagating, so a single invalid log-weight cannot turn the normalizer
/// of a whole population into `NaN`.
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

    // Compute sum(exp(x_i - max)) stably. `f64::max` already ignores NaN when
    // picking `max_val`; map NaN terms to −∞ here so they contribute exp(−∞) = 0.
    let sum_exp: f64 = log_values
        .iter()
        .map(|&x| (nan_to_neg_inf(x) - max_val).exp())
        .sum();

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
        .map(|(&x, &w)| w * (nan_to_neg_inf(x) - max_val).exp())
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
        // FG-32: exact value, not just finiteness.
        // 701 + ln(e^-1 + 1 + e^-2) = 701.4076059644444
        assert!((log_sum_exp(&[700.0, 701.0, 699.0]) - 701.4076059644444).abs() < 1e-9);

        // Single-element input returns the element exactly.
        assert_eq!(log_sum_exp(&[5.0]), 5.0);

        // Test with small values
        let small_vals = vec![-700.0, -701.0, -699.0];
        let result = log_sum_exp(&small_vals);
        assert!(result.is_finite());

        // Extreme spread: the small term underflows, ln(1 + e^-1000) == 0.
        assert_eq!(log_sum_exp(&[0.0, -1000.0]), 0.0);

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

        // FG-32: exact softmax values and ratios, not merely ordering.
        assert!((probs[0] - 0.6652409557748219).abs() < 1e-9);
        assert!((probs[1] - 0.24472847105479764).abs() < 1e-9);
        assert!((probs[2] - 0.09003057317038043).abs() < 1e-9);
        // Adjacent ratio is exp(-1 - (-2)) = e.
        assert!((probs[0] / probs[1] - std::f64::consts::E).abs() < 1e-9);
    }

    #[test]
    fn test_log1p_exp_stability() {
        // FG-32: exact values at hand-computable points.
        assert!((log1p_exp(0.0) - std::f64::consts::LN_2).abs() < 1e-9); // ln(2)
        assert!((log1p_exp(2.0) - 2.1269280110429727).abs() < 1e-9); // ln(1 + e^2)
        assert!((log1p_exp(50.0) - 50.0).abs() < 1e-10);
        assert!((log1p_exp(-50.0) - 1.9287498479639178e-22).abs() < 1e-31);
        assert_eq!(log1p_exp(-1000.0), 0.0);
    }

    // FG-N2: a NaN term is a zero-probability term, not a poison pill.
    #[test]
    fn test_log_sum_exp_treats_nan_as_neg_inf() {
        assert_eq!(log_sum_exp(&[f64::NAN, 0.0]), 0.0);
        assert_eq!(log_sum_exp(&[0.0, f64::NAN, 0.0]), 2.0_f64.ln());
        assert_eq!(log_sum_exp(&[f64::NAN]), f64::NEG_INFINITY);
        assert_eq!(
            log_sum_exp(&[f64::NAN, f64::NEG_INFINITY]),
            f64::NEG_INFINITY
        );
        assert_eq!(
            weighted_log_sum_exp(&[f64::NAN, 0.0], &[0.5, 0.5]),
            0.5_f64.ln()
        );
        assert_eq!(
            weighted_log_sum_exp(&[f64::NAN, f64::NAN], &[0.5, 0.5]),
            f64::NEG_INFINITY
        );
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
