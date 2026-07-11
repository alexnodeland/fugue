//! Convergence diagnostics and parameter summaries for MCMC chains.
//!
//! This module provides essential tools for analyzing MCMC output and assessing
//! the quality of posterior approximations. Proper diagnostics are crucial for
//! ensuring that MCMC chains have converged and that posterior estimates are reliable.
//!
//! ## Available Diagnostics
//!
//! - **Split-R-hat (Potential Scale Reduction Factor)**: Measures between-chain
//!   vs within-chain variance after splitting each chain in half (Vehtari et al.
//!   2021), so within-chain trends are detected. The classic (1992) statistic is
//!   available via [`classic_r_hat_f64`].
//! - **Effective sample size**: Routed through the single normalized estimator in
//!   [`crate::inference::mcmc_utils`]; summaries use the multi-chain estimator.
//! - **Parameter summaries**: Mean, standard deviation, quantiles for each parameter
//! - **Diagnostic printing**: Formatted output for quick assessment
//!
//! ## Convergence Assessment
//!
//! The split-R-hat statistic compares the variance between (split) chains to the
//! variance within chains. Values close to 1.0 indicate convergence, while values
//! > 1.1 suggest that chains haven't mixed well and more sampling is needed.
//!
//! ## Best Practices
//!
//! 1. Run multiple chains from different starting points
//! 2. Check R-hat for all parameters (should be < 1.1)
//! 3. Examine trace plots for proper mixing
//! 4. Use effective sample size to assess sampling efficiency
//!
//! # Examples
//!
//! ```rust
//! use fugue::*;
//! use rand::rngs::StdRng;
//! use rand::SeedableRng;
//!
//! // Generate MCMC samples (simplified for testing)
//! let model_fn = || sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap())
//!     .bind(|mu| observe(addr!("y"), Normal::new(mu, 0.5).unwrap(), 2.0).map(move |_| mu));
//!
//! let mut all_traces = Vec::new();
//! for chain in 0..2 {  // Just 2 chains for testing
//!     let mut rng = StdRng::seed_from_u64(42 + chain);
//!     let samples = adaptive_mcmc_chain(&mut rng, model_fn, 10, 2);  // Small numbers
//!     all_traces.extend(samples.into_iter().map(|(_, trace)| trace));
//! }
//!
//! // Create separate chains for r_hat calculation
//! let mut chain1 = Vec::new();
//! let mut chain2 = Vec::new();
//! for (i, trace) in all_traces.iter().enumerate() {
//!     if i % 2 == 0 {
//!         chain1.push(trace.clone());
//!     } else {
//!         chain2.push(trace.clone());
//!     }
//! }
//!
//! // Compute diagnostics
//! let r_hat_val = r_hat_f64(&[chain1.clone(), chain2.clone()], &addr!("mu"));
//! let summary = summarize_f64_parameter(&[chain1, chain2], &addr!("mu"));
//!
//! assert!(r_hat_val.is_finite() || r_hat_val.is_nan());
//! assert!(summary.mean.is_finite());
//! ```

use crate::core::address::Address;
use crate::inference::mcmc_utils::{effective_sample_size_mcmc, effective_sample_size_multichain};
use crate::runtime::trace::Trace;
use std::collections::HashMap;

/// Type-safe extraction of f64 values from traces.
///
/// Only extracts values that are actually stored as f64, avoiding lossy conversions.
pub fn extract_f64_values(traces: &[Trace], addr: &Address) -> Vec<f64> {
    traces.iter().filter_map(|t| t.get_f64(addr)).collect()
}

/// Type-safe extraction of bool values from traces.
pub fn extract_bool_values(traces: &[Trace], addr: &Address) -> Vec<bool> {
    traces.iter().filter_map(|t| t.get_bool(addr)).collect()
}

/// Type-safe extraction of u64 values from traces.
pub fn extract_u64_values(traces: &[Trace], addr: &Address) -> Vec<u64> {
    traces.iter().filter_map(|t| t.get_u64(addr)).collect()
}

/// Type-safe extraction of usize values from traces.
pub fn extract_usize_values(traces: &[Trace], addr: &Address) -> Vec<usize> {
    traces.iter().filter_map(|t| t.get_usize(addr)).collect()
}

/// Type-safe extraction of i64 values from traces.
pub fn extract_i64_values(traces: &[Trace], addr: &Address) -> Vec<i64> {
    traces.iter().filter_map(|t| t.get_i64(addr)).collect()
}

/// Trait for type-specific diagnostics.
///
/// This trait enables computing diagnostics for different value types without
/// forcing everything to f64, preserving type safety and avoiding lossy conversions.
pub trait Diagnostics<T> {
    /// Extract values of type T from traces at the given address.
    fn extract_values(traces: &[Trace], addr: &Address) -> Vec<T>;

    /// Compute R-hat for this type (if applicable).
    fn r_hat(chains: &[Vec<Trace>], addr: &Address) -> Option<f64>;

    /// Compute effective sample size (if applicable).
    fn effective_sample_size(values: &[T]) -> Option<f64>;
}

impl Diagnostics<f64> for f64 {
    fn extract_values(traces: &[Trace], addr: &Address) -> Vec<f64> {
        extract_f64_values(traces, addr)
    }

    fn r_hat(chains: &[Vec<Trace>], addr: &Address) -> Option<f64> {
        let r_hat_val = r_hat_f64(chains, addr);
        if r_hat_val.is_finite() {
            Some(r_hat_val)
        } else {
            None
        }
    }

    fn effective_sample_size(values: &[f64]) -> Option<f64> {
        if values.len() < 4 {
            return Some(values.len() as f64);
        }
        Some(effective_sample_size(values))
    }
}

impl Diagnostics<bool> for bool {
    fn extract_values(traces: &[Trace], addr: &Address) -> Vec<bool> {
        extract_bool_values(traces, addr)
    }

    fn r_hat(_chains: &[Vec<Trace>], _addr: &Address) -> Option<f64> {
        // R-hat doesn't make sense for boolean variables
        None
    }

    fn effective_sample_size(_values: &[bool]) -> Option<f64> {
        // ESS computed differently for discrete variables
        None
    }
}

impl Diagnostics<u64> for u64 {
    fn extract_values(traces: &[Trace], addr: &Address) -> Vec<u64> {
        extract_u64_values(traces, addr)
    }

    fn r_hat(chains: &[Vec<Trace>], addr: &Address) -> Option<f64> {
        // For count data, we can convert to f64 for R-hat
        let f64_chains: Vec<Vec<f64>> = chains
            .iter()
            .map(|chain| {
                extract_u64_values(chain, addr)
                    .into_iter()
                    .map(|x| x as f64)
                    .collect()
            })
            .collect();

        if f64_chains.iter().any(|v| v.is_empty()) {
            return None;
        }

        // Report split-R-hat for consistency with the f64 path (FG-36).
        let r_hat_val = split_r_hat_from_f64_chains(&f64_chains);
        if r_hat_val.is_finite() {
            Some(r_hat_val)
        } else {
            None
        }
    }

    fn effective_sample_size(values: &[u64]) -> Option<f64> {
        if values.len() < 4 {
            return Some(values.len() as f64);
        }
        // Convert to f64 for ESS computation
        let f64_values: Vec<f64> = values.iter().map(|&x| x as f64).collect();
        Some(effective_sample_size(&f64_values))
    }
}

impl Diagnostics<usize> for usize {
    fn extract_values(traces: &[Trace], addr: &Address) -> Vec<usize> {
        extract_usize_values(traces, addr)
    }

    fn r_hat(_chains: &[Vec<Trace>], _addr: &Address) -> Option<f64> {
        // R-hat for categorical variables needs special treatment
        None
    }

    fn effective_sample_size(_values: &[usize]) -> Option<f64> {
        // ESS for categorical variables is complex
        None
    }
}

/// Compute the split-R-hat convergence diagnostic for f64 values.
///
/// FG-36: this returns *split*-R-hat (Vehtari et al. 2021), the current best
/// practice: each chain is split in half and the halves are treated as separate
/// chains before applying the Gelman-Rubin formula. Splitting lets the
/// diagnostic detect *within-chain* non-stationarity (e.g. a slow trend) that
/// classic R-hat misses when all chains drift the same way. The classic (1992)
/// statistic remains available via [`classic_r_hat_f64`]; summaries report the
/// split value.
pub fn r_hat_f64(chains: &[Vec<Trace>], addr: &Address) -> f64 {
    let chain_values: Vec<Vec<f64>> = chains
        .iter()
        .map(|chain| extract_f64_values(chain, addr))
        .collect();
    split_r_hat_from_f64_chains(&chain_values)
}

/// Compute the classic (non-split) Gelman & Rubin (1992) R-hat for f64 values.
///
/// Retained so callers who specifically want the 1992 statistic can request it;
/// [`r_hat_f64`] and the parameter summaries use the split variant (FG-36).
pub fn classic_r_hat_f64(chains: &[Vec<Trace>], addr: &Address) -> f64 {
    let chain_values: Vec<Vec<f64>> = chains
        .iter()
        .map(|chain| extract_f64_values(chain, addr))
        .collect();
    r_hat_from_f64_chains(&chain_values)
}

/// Split each chain in half (dropping the middle draw when the length is odd)
/// and return the `2m` half-chains, per Vehtari et al. (2021).
fn split_f64_chains(chain_values: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut out = Vec::with_capacity(chain_values.len() * 2);
    for c in chain_values {
        let half = c.len() / 2;
        if half == 0 {
            // Too short to split; keep as-is so downstream guards handle it.
            out.push(c.clone());
            continue;
        }
        out.push(c[..half].to_vec());
        out.push(c[half..2 * half].to_vec());
    }
    out
}

/// Split-R-hat from pre-extracted f64 chains (FG-36).
fn split_r_hat_from_f64_chains(chain_values: &[Vec<f64>]) -> f64 {
    let split = split_f64_chains(chain_values);
    r_hat_from_f64_chains(&split)
}

/// Helper function to compute R-hat from pre-extracted f64 chains.
fn r_hat_from_f64_chains(chain_values: &[Vec<f64>]) -> f64 {
    if chain_values.len() < 2 {
        return 1.0; // Can't compute R-hat with single chain
    }

    if chain_values.iter().any(|v| v.is_empty()) {
        return f64::NAN; // Missing data
    }

    let m = chain_values.len() as f64; // number of chains
    let n = chain_values[0].len() as f64; // samples per chain

    // Chain means
    let chain_means: Vec<f64> = chain_values
        .iter()
        .map(|values| values.iter().sum::<f64>() / values.len() as f64)
        .collect();

    // Overall mean
    let overall_mean = chain_means.iter().sum::<f64>() / m;

    // Between-chain variance
    let b = n / (m - 1.0)
        * chain_means
            .iter()
            .map(|mean| (mean - overall_mean).powi(2))
            .sum::<f64>();

    // Within-chain variances
    let within_variances: Vec<f64> = chain_values
        .iter()
        .zip(&chain_means)
        .map(|(values, &mean)| values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0))
        .collect();

    let w = within_variances.iter().sum::<f64>() / m;

    // Pooled variance estimate
    let var_plus = ((n - 1.0) / n) * w + (1.0 / n) * b;

    // R-hat
    (var_plus / w).sqrt()
}

/// Compute effective sample size for a single chain.
///
/// FG-01: this used to compute `tau` from *raw* autocovariances (never dividing
/// by the lag-0 variance), which made ESS scale with the parameter's variance
/// instead of being a dimensionless diagnostic — silently wrong by an order of
/// magnitude for any parameter whose variance isn't ~1, and able to report
/// `ESS > n` for variance `< 1`. The buggy estimator has been deleted; this is
/// now a thin wrapper over the single, correct normalized estimator in
/// [`crate::inference::mcmc_utils`], so every ESS path in the crate routes
/// through the same implementation.
pub fn effective_sample_size(values: &[f64]) -> f64 {
    effective_sample_size_mcmc(values)
}

/// Compute summary statistics for a parameter across chains.
#[derive(Debug, Clone)]
pub struct ParameterSummary {
    pub mean: f64,
    pub std: f64,
    pub quantiles: HashMap<String, f64>, // "2.5%", "25%", "50%", "75%", "97.5%"
    pub r_hat: f64,
    pub ess: f64,
}

/// Type-safe parameter summary for f64 values.
pub fn summarize_f64_parameter(chains: &[Vec<Trace>], addr: &Address) -> ParameterSummary {
    // Combine all chains
    let all_values: Vec<f64> = chains
        .iter()
        .flat_map(|chain| extract_f64_values(chain, addr))
        .collect();

    if all_values.is_empty() {
        return ParameterSummary {
            mean: f64::NAN,
            std: f64::NAN,
            quantiles: HashMap::new(),
            r_hat: f64::NAN,
            ess: 0.0,
        };
    }

    // Basic statistics
    let mean = all_values.iter().sum::<f64>() / all_values.len() as f64;
    let variance =
        all_values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (all_values.len() - 1) as f64;
    let std = variance.sqrt();

    // Quantiles
    let mut sorted_values = all_values.clone();
    sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut quantiles = HashMap::new();
    let percentiles = [
        ("2.5%", 0.025),
        ("25%", 0.25),
        ("50%", 0.5),
        ("75%", 0.75),
        ("97.5%", 0.975),
    ];

    for (name, p) in percentiles {
        let idx = ((sorted_values.len() - 1) as f64 * p).round() as usize;
        quantiles.insert(name.to_string(), sorted_values[idx]);
    }

    // Diagnostics. FG-36: report split-R-hat. FG-37: compute ESS across ALL
    // chains (Vehtari et al. 2021 multi-chain estimator), consistent with the
    // pooled mean/std/quantiles above — the previous code used only the first
    // chain, discarding (M-1)/M of the data and mislabeling a per-chain ESS as
    // the parameter's ESS.
    let r_hat_val = r_hat_f64(chains, addr);
    let per_chain_values: Vec<Vec<f64>> = chains
        .iter()
        .map(|chain| extract_f64_values(chain, addr))
        .collect();
    let ess_val = effective_sample_size_multichain(&per_chain_values);

    ParameterSummary {
        mean,
        std,
        quantiles,
        r_hat: r_hat_val,
        ess: ess_val,
    }
}

/// Print diagnostic summary for all parameters.
pub fn print_diagnostics(chains: &[Vec<Trace>]) {
    if chains.is_empty() || chains[0].is_empty() {
        println!("No chains or empty chains provided");
        return;
    }

    // Get all unique addresses
    let mut all_addresses = std::collections::HashSet::new();
    for chain in chains {
        for trace in chain {
            for addr in trace.choices.keys() {
                all_addresses.insert(addr.clone());
            }
        }
    }

    println!("MCMC Diagnostics:");
    println!(
        "{:<15} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "Parameter", "Mean", "Std", "2.5%", "50%", "97.5%", "R-hat", "ESS"
    );
    println!("{}", "-".repeat(80));

    for addr in &all_addresses {
        let summary = summarize_f64_parameter(chains, addr);
        println!(
            "{:<15} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.0}",
            addr.to_string(),
            summary.mean,
            summary.std,
            summary.quantiles.get("2.5%").unwrap_or(&f64::NAN),
            summary.quantiles.get("50%").unwrap_or(&f64::NAN),
            summary.quantiles.get("97.5%").unwrap_or(&f64::NAN),
            summary.r_hat,
            summary.ess
        );
    }

    // Overall convergence assessment
    let all_r_hats: Vec<f64> = all_addresses
        .iter()
        .map(|addr| summarize_f64_parameter(chains, addr).r_hat)
        .filter(|&x| x.is_finite())
        .collect();

    if !all_r_hats.is_empty() {
        let max_r_hat = all_r_hats.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let avg_r_hat = all_r_hats.iter().sum::<f64>() / all_r_hats.len() as f64;

        println!("\nConvergence Assessment:");
        if max_r_hat < 1.01 {
            println!("✓ Excellent convergence (max R-hat = {:.3})", max_r_hat);
        } else if max_r_hat < 1.1 {
            println!("⚠ Good convergence (max R-hat = {:.3})", max_r_hat);
        } else {
            println!(
                "✗ Poor convergence (max R-hat = {:.3}) - consider more samples",
                max_r_hat
            );
        }
        println!("  Average R-hat: {:.3}", avg_r_hat);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::addr;
    use crate::core::distribution::*;
    use crate::core::model::{observe, sample, ModelExt};
    use crate::runtime::handler::run;
    use crate::runtime::interpreters::PriorHandler;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn generate_chain(seed: u64, n: usize) -> Vec<Trace> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut traces = Vec::new();
        for _ in 0..n {
            let (_a, t) = run(
                PriorHandler {
                    rng: &mut rng,
                    trace: Trace::default(),
                },
                sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap())
                    .and_then(|mu| observe(addr!("y"), Normal::new(mu, 0.5).unwrap(), 0.0)),
            );
            traces.push(t);
        }
        traces
    }

    #[test]
    fn extractors_return_expected_values() {
        let chain = generate_chain(1, 5);
        let vals = extract_f64_values(&chain, &addr!("mu"));
        assert_eq!(vals.len(), 5);
        let bools = extract_bool_values(&chain, &addr!("mu"));
        assert!(bools.is_empty());
    }

    #[test]
    fn r_hat_and_summary_compute() {
        let chains = vec![generate_chain(2, 10), generate_chain(3, 10)];
        let r = r_hat_f64(&chains, &addr!("mu"));
        assert!(r.is_finite());
        let summary = summarize_f64_parameter(&chains, &addr!("mu"));
        assert!(summary.mean.is_finite());
        assert!(summary.std.is_finite());
    }

    #[test]
    fn diagnostics_trait_for_other_types() {
        let chains = vec![generate_chain(4, 5), generate_chain(5, 5)];
        // u64 r_hat via conversion
        let r_u64 = <u64 as Diagnostics<u64>>::r_hat(&chains, &addr!("mu"));
        // Might be None if chains empty; here should be Some or None acceptable
        let _ = r_u64;

        // usize Diagnostics returns None for r_hat
        let r_usize = <usize as Diagnostics<usize>>::r_hat(&chains, &addr!("mu"));
        assert!(r_usize.is_none());
    }

    #[test]
    fn print_diagnostics_with_multiple_addresses() {
        // Build chains with two addresses
        let mut rng = StdRng::seed_from_u64(6);
        let mut chain = Vec::new();
        for _ in 0..5 {
            let (_a, mut t) = run(
                PriorHandler {
                    rng: &mut rng,
                    trace: Trace::default(),
                },
                sample(addr!("a1"), Normal::new(0.0, 1.0).unwrap())
                    .and_then(|x| observe(addr!("obs"), Normal::new(x, 1.0).unwrap(), 0.0)),
            );
            // Insert second address manually
            t.insert_choice(
                addr!("a2"),
                crate::runtime::trace::ChoiceValue::F64(1.0),
                -0.5,
            );
            chain.push(t);
        }
        let chains = vec![chain.clone(), chain];
        print_diagnostics(&chains);
    }
}
