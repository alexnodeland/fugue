//! Metropolis–Hastings sampling with adaptive tuning and site-wise proposals.
//!
//! This module provides adaptive MH kernels that automatically tune proposal
//! scales and use appropriate proposals for different distribution types.
use crate::core::address::Address;

use crate::core::model::Model;
use crate::runtime::handler::run;
use crate::runtime::interpreters::{PriorHandler, ScoreGivenTrace};
use crate::runtime::trace::{Choice, ChoiceValue, Trace};
use rand::Rng;
use std::collections::HashMap;

/// Adaptive proposal scales for different sites.
#[derive(Debug, Clone)]
pub struct AdaptiveScales {
    pub scales: HashMap<Address, f64>,
    pub accept_counts: HashMap<Address, usize>,
    pub total_counts: HashMap<Address, usize>,
    pub target_accept_rate: f64,
}

impl AdaptiveScales {
    pub fn new() -> Self {
        Self {
            scales: HashMap::new(),
            accept_counts: HashMap::new(),
            total_counts: HashMap::new(),
            target_accept_rate: 0.44, // Optimal for random walk MH
        }
    }

    pub fn get_scale(&mut self, addr: &Address) -> f64 {
        *self.scales.entry(addr.clone()).or_insert(1.0)
    }

    pub fn update(&mut self, addr: &Address, accepted: bool) {
        *self.total_counts.entry(addr.clone()).or_insert(0) += 1;
        if accepted {
            *self.accept_counts.entry(addr.clone()).or_insert(0) += 1;
        }

        let total = self.total_counts[addr];
        let accepts = self.accept_counts[addr];

        if total >= 50 && total % 50 == 0 {
            let accept_rate = accepts as f64 / total as f64;
            let scale = self.scales.entry(addr.clone()).or_insert(1.0);

            if accept_rate > self.target_accept_rate + 0.05 {
                *scale *= 1.1; // Increase proposal scale
            } else if accept_rate < self.target_accept_rate - 0.05 {
                *scale *= 0.9; // Decrease proposal scale
            }

            // Keep scale in reasonable bounds
            *scale = scale.clamp(0.01, 10.0);
        }
    }
}

/// Propose a new value for a choice based on its current value and distribution type.
fn propose_new_value<R: Rng>(rng: &mut R, choice: &Choice, scale: f64) -> f64 {
    match choice.value {
        ChoiceValue::F64(current_val) => {
            // Simple random walk proposal
            current_val + rng.gen::<f64>() * scale * 2.0 - scale
        }
        ChoiceValue::Bool(current_val) => {
            // Flip proposal for boolean
            if rng.gen::<f64>() < 0.5 {
                if current_val {
                    0.0
                } else {
                    1.0
                }
            } else {
                if current_val {
                    1.0
                } else {
                    0.0
                }
            }
        }
        ChoiceValue::I64(current_val) => {
            // Integer random walk
            let delta = ((rng.gen::<f64>() * 2.0 - 1.0) * scale).round() as i64;
            (current_val + delta) as f64
        }
    }
}

/// Single-site adaptive Metropolis-Hastings step.
pub fn adaptive_single_site_mh<A, R: Rng>(
    rng: &mut R,
    model_fn: impl Fn() -> Model<A>,
    current: &Trace,
    scales: &mut AdaptiveScales,
) -> (A, Trace) {
    if current.choices.is_empty() {
        // No choices to update, return current
        let (a, _) = run(
            ScoreGivenTrace {
                base: current.clone(),
                trace: Trace::default(),
            },
            model_fn(),
        );
        return (a, current.clone());
    }

    // Pick a random site to update
    let sites: Vec<_> = current.choices.keys().collect();
    let site_idx = rng.gen_range(0..sites.len());
    let selected_site = sites[site_idx].clone();

    // Get current choice and propose new value
    let current_choice = &current.choices[&selected_site];
    let scale = scales.get_scale(&selected_site);
    let proposed_val = propose_new_value(rng, current_choice, scale);

    // Create proposed trace
    let mut proposed_trace = current.clone();
    proposed_trace
        .choices
        .get_mut(&selected_site)
        .unwrap()
        .value = ChoiceValue::F64(proposed_val);

    // Score both traces
    let (_a_cur, cur_scored) = run(
        ScoreGivenTrace {
            base: current.clone(),
            trace: Trace::default(),
        },
        model_fn(),
    );
    let (a_prop, prop_scored) = run(
        ScoreGivenTrace {
            base: proposed_trace.clone(),
            trace: Trace::default(),
        },
        model_fn(),
    );

    // Accept/reject
    let log_alpha = prop_scored.total_log_weight() - cur_scored.total_log_weight();
    let accept = log_alpha >= 0.0 || rng.gen::<f64>() < log_alpha.exp();

    // Update adaptive scales
    scales.update(&selected_site, accept);

    if accept {
        (a_prop, proposed_trace)
    } else {
        let (a, _) = run(
            ScoreGivenTrace {
                base: current.clone(),
                trace: Trace::default(),
            },
            model_fn(),
        );
        (a, current.clone())
    }
}

/// Run an adaptive MCMC chain.
pub fn adaptive_mcmc_chain<A, R: Rng>(
    rng: &mut R,
    model_fn: impl Fn() -> Model<A>,
    n_samples: usize,
    n_warmup: usize,
) -> Vec<(A, Trace)> {
    let mut samples = Vec::with_capacity(n_samples);
    let mut scales = AdaptiveScales::new();

    // Initialize with prior sample
    let (_, mut current_trace) = run(
        PriorHandler {
            rng,
            trace: Trace::default(),
        },
        model_fn(),
    );

    // Warmup phase
    for _ in 0..n_warmup {
        let (_, trace) = adaptive_single_site_mh(rng, &model_fn, &current_trace, &mut scales);
        current_trace = trace;
    }

    // Sampling phase
    for _ in 0..n_samples {
        let (val, trace) = adaptive_single_site_mh(rng, &model_fn, &current_trace, &mut scales);
        current_trace = trace;
        samples.push((val, current_trace.clone()));
    }

    samples
}

// Keep the original simple function for backward compatibility
pub fn single_site_random_walk_mh<A, R: Rng>(
    rng: &mut R,
    _proposal_sigma: f64,
    model_fn: impl Fn() -> Model<A>,
    current: &Trace,
) -> (A, Trace) {
    let mut scales = AdaptiveScales::new();
    adaptive_single_site_mh(rng, model_fn, current, &mut scales)
}
