//! Convergence diagnostics and parameter summaries for MCMC chains.
//!
//! This module provides essential tools for analyzing MCMC output and assessing
//! the quality of posterior approximations. Proper diagnostics are crucial for
//! ensuring that MCMC chains have converged and that posterior estimates are reliable.
//!
//! ## Available Diagnostics
//!
//! - **R-hat (Potential Scale Reduction Factor)**: Measures between-chain vs within-chain variance
//! - **Parameter summaries**: Mean, standard deviation, quantiles for each parameter
//! - **Diagnostic printing**: Formatted output for quick assessment
//!
//! ## Convergence Assessment
//!
//! The R-hat statistic compares the variance between multiple chains to the variance
//! within chains. Values close to 1.0 indicate convergence, while values > 1.1
//! suggest that chains haven't mixed well and more sampling is needed.
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

        let r_hat_val = r_hat_from_f64_chains(&f64_chains);
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

/// Compute R-hat convergence diagnostic for f64 values.
pub fn r_hat_f64(chains: &[Vec<Trace>], addr: &Address) -> f64 {
    let chain_values: Vec<Vec<f64>> = chains
        .iter()
        .map(|chain| extract_f64_values(chain, addr))
        .collect();
    r_hat_from_f64_chains(&chain_values)
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
pub fn effective_sample_size(values: &[f64]) -> f64 {
    if values.len() < 4 {
        return values.len() as f64;
    }

    let n = values.len();
    let mean = values.iter().sum::<f64>() / n as f64;

    // Compute autocorrelations
    let mut autocorrs = Vec::new();
    let max_lag = (n / 4).min(200); // Reasonable maximum lag

    for lag in 0..max_lag {
        if lag >= n - 1 {
            break;
        }

        let mut num = 0.0;
        let mut count = 0;

        for i in 0..(n - lag) {
            num += (values[i] - mean) * (values[i + lag] - mean);
            count += 1;
        }

        if count > 0 {
            autocorrs.push(num / count as f64);
        } else {
            break;
        }
    }

    if autocorrs.is_empty() {
        return n as f64;
    }

    // Find first negative autocorrelation or use all
    let mut _sum_autocorr = autocorrs[0]; // lag 0 = variance
    let mut tau = 1.0;

    for (lag, &rho) in autocorrs.iter().enumerate().skip(1) {
        if rho <= 0.0 {
            break;
        }
        _sum_autocorr += 2.0 * rho;
        tau = 1.0 + 2.0 * autocorrs[1..=lag].iter().sum::<f64>();

        // Automatic windowing condition
        if lag as f64 >= 6.0 * tau {
            break;
        }
    }

    n as f64 / tau
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

    // Diagnostics
    let r_hat_val = r_hat_f64(chains, addr);
    let ess_val = if !chains.is_empty() {
        effective_sample_size(&extract_f64_values(&chains[0], addr))
    } else {
        0.0
    };

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
