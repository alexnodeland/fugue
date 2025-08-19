//! Convergence diagnostics and model checking tools.
//!
//! Provides essential diagnostics for validating MCMC convergence and
//! posterior quality, including R-hat, effective sample size, and trace analysis.

use crate::core::address::Address;
use crate::runtime::trace::{ChoiceValue, Trace};
use std::collections::HashMap;

/// Extract scalar values from traces for a specific address.
pub fn extract_values(traces: &[Trace], addr: &Address) -> Vec<f64> {
    traces
        .iter()
        .filter_map(|t| t.choices.get(addr))
        .filter_map(|choice| match choice.value {
            ChoiceValue::F64(v) => Some(v),
            ChoiceValue::I64(v) => Some(v as f64),
            ChoiceValue::Bool(v) => Some(if v { 1.0 } else { 0.0 }),
        })
        .collect()
}

/// Compute R-hat convergence diagnostic for multiple chains.
pub fn r_hat(chains: &[Vec<Trace>], addr: &Address) -> f64 {
    if chains.len() < 2 {
        return 1.0; // Can't compute R-hat with single chain
    }

    let chain_values: Vec<Vec<f64>> = chains
        .iter()
        .map(|chain| extract_values(chain, addr))
        .collect();

    if chain_values.iter().any(|v| v.is_empty()) {
        return f64::NAN; // Missing data
    }

    let m = chains.len() as f64; // number of chains
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

pub fn summarize_parameter(chains: &[Vec<Trace>], addr: &Address) -> ParameterSummary {
    // Combine all chains
    let all_values: Vec<f64> = chains
        .iter()
        .flat_map(|chain| extract_values(chain, addr))
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
    let r_hat_val = r_hat(chains, addr);
    let ess_val = if !chains.is_empty() {
        effective_sample_size(&extract_values(&chains[0], addr))
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
        let summary = summarize_parameter(chains, &addr);
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
        .map(|addr| summarize_parameter(chains, addr).r_hat)
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
