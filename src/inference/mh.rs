//! Metropolis-Hastings MCMC with adaptive tuning and single-site updates.
//!
//! This module implements the Metropolis-Hastings algorithm, a fundamental MCMC method
//! for sampling from posterior distributions. The implementation features:
//!
//! - **Adaptive scaling**: Automatically tunes proposal step sizes to achieve target acceptance rates
//! - **Single-site updates**: Updates one random variable at a time for better mixing
//! - **Random-walk proposals**: Uses Gaussian perturbations centered on current values
//! - **Acceptance rate monitoring**: Tracks and optimizes per-site acceptance rates
//!
//! ## Algorithm Overview
//!
//! The Metropolis-Hastings algorithm generates correlated samples from the posterior by:
//! 1. Proposing a new state by modifying the current state
//! 2. Computing the acceptance probability using the ratio of posterior densities
//! 3. Accepting or rejecting the proposal based on this probability
//! 4. Repeating to generate a Markov chain that converges to the posterior
//!
//! ## Adaptive Tuning
//!
//! Good MCMC performance requires well-tuned proposal distributions. This implementation
//! automatically adapts proposal scales to achieve approximately 44% acceptance rate
//! (optimal for random-walk Metropolis on continuous distributions).
//!
//! # Examples
//!
//! ```rust
//! use fugue::*;
//! use rand::rngs::StdRng;
//! use rand::SeedableRng;
//!
//! // Define a simple Bayesian model
//! let model_fn = || {
//!     sample(addr!("mu"), Normal { mu: 0.0, sigma: 2.0 })
//!         .bind(|mu| observe(addr!("y"), Normal { mu, sigma: 1.0 }, 2.5))
//! };
//!
//! // Run adaptive MCMC (small numbers for testing)
//! let mut rng = StdRng::seed_from_u64(42);
//! let samples = adaptive_mcmc_chain(
//!     &mut rng,
//!     model_fn,
//!     50,  // Number of samples (small for test)
//!     10,  // Burn-in period
//! );
//!
//! // Extract parameter estimates
//! let mu_samples: Vec<f64> = samples.iter()
//!     .filter_map(|(_, trace)| trace.choices.get(&addr!("mu")))
//!     .filter_map(|choice| match choice.value {
//!         ChoiceValue::F64(mu) => Some(mu),
//!         _ => None,
//!     })
//!     .collect();
//!
//! assert!(!mu_samples.is_empty());
//! ```
use crate::core::address::Address;

use crate::core::model::Model;
use crate::runtime::handler::run;
use crate::runtime::interpreters::{PriorHandler, ScoreGivenTrace};
use crate::runtime::trace::{Choice, ChoiceValue, Trace};
use rand::Rng;
use std::collections::HashMap;

/// Adaptive proposal scaling system for MCMC sites.
///
/// This struct tracks acceptance rates for each random variable site and automatically
/// adjusts proposal step sizes to maintain optimal acceptance rates. The adaptive
/// mechanism helps achieve good MCMC mixing without manual tuning.
///
/// ## Adaptation Strategy
///
/// - **Target rate**: 44% acceptance (optimal for random-walk Metropolis)
/// - **Update frequency**: Every 50 proposals per site
/// - **Scale adjustment**: Multiplicative updates based on acceptance rate
/// - **Per-site tracking**: Each address gets independent tuning
///
/// # Fields
///
/// * `scales` - Current proposal scale for each site
/// * `accept_counts` - Number of accepted proposals per site
/// * `total_counts` - Total number of proposals per site
/// * `target_accept_rate` - Desired acceptance rate (default: 0.44)
///
/// # Examples
///
/// ```rust
/// use fugue::*;
///
/// let mut scales = AdaptiveScales::new();
/// let addr = addr!("mu");
///
/// // Get current scale (starts at 1.0)
/// let scale = scales.get_scale(&addr);
/// assert_eq!(scale, 1.0);
///
/// // Update with acceptance outcome
/// scales.update(&addr, true);  // Accepted
/// scales.update(&addr, false); // Rejected
/// ```
#[derive(Debug, Clone)]
pub struct AdaptiveScales {
    /// Current proposal scale for each site.
    pub scales: HashMap<Address, f64>,
    /// Number of accepted proposals per site.
    pub accept_counts: HashMap<Address, usize>,
    /// Total number of proposals per site.
    pub total_counts: HashMap<Address, usize>,
    /// Target acceptance rate for adaptation.
    pub target_accept_rate: f64,
}

impl AdaptiveScales {
    /// Create a new adaptive scaling system with default settings.
    ///
    /// Initializes empty tracking maps and sets the target acceptance rate to 0.44,
    /// which is optimal for random-walk Metropolis on continuous distributions.
    pub fn new() -> Self {
        Self {
            scales: HashMap::new(),
            accept_counts: HashMap::new(),
            total_counts: HashMap::new(),
            target_accept_rate: 0.44, // Optimal for random walk MH
        }
    }

    /// Get the current proposal scale for a site, initializing to 1.0 if new.
    ///
    /// # Arguments
    ///
    /// * `addr` - Address of the site to get scale for
    ///
    /// # Returns
    ///
    /// Current scale factor for proposals at this site.
    pub fn get_scale(&mut self, addr: &Address) -> f64 {
        *self.scales.entry(addr.clone()).or_insert(1.0)
    }

    /// Update acceptance statistics and potentially adjust the proposal scale.
    ///
    /// Records the outcome of a proposal and periodically adjusts the scale
    /// based on the running acceptance rate. Updates occur every 50 proposals.
    ///
    /// # Arguments
    ///
    /// * `addr` - Address of the site that was updated
    /// * `accepted` - Whether the proposal was accepted
    pub fn update(&mut self, addr: &Address, accepted: bool) {
        *self.total_counts.entry(addr.clone()).or_insert(0) += 1;
        if accepted {
            *self.accept_counts.entry(addr.clone()).or_insert(0) += 1;
        }

        let total = *self.total_counts.get(addr).unwrap_or(&0);
        let accepts = *self.accept_counts.get(addr).unwrap_or(&0);

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

/// Perform a single adaptive Metropolis-Hastings update step.
///
/// This function implements a single iteration of the MH algorithm with adaptive
/// proposal scaling. It randomly selects one site to update, proposes a new value,
/// and accepts or rejects based on the Metropolis-Hastings acceptance criterion.
///
/// # Algorithm
///
/// 1. Randomly select a site from the current trace
/// 2. Propose a new value using adaptive scaling
/// 3. Score both current and proposed traces
/// 4. Accept with probability min(1, exp(log_prob_new - log_prob_old))
/// 5. Update adaptive scales based on acceptance outcome
///
/// # Arguments
///
/// * `rng` - Random number generator
/// * `model_fn` - Function that creates the model
/// * `current` - Current trace (state of the Markov chain)
/// * `scales` - Adaptive scaling system (modified in-place)
///
/// # Returns
///
/// Tuple of (model_result, new_trace) after the MH step.
///
/// # Examples
///
/// ```rust
/// use fugue::*;
/// use rand::rngs::StdRng;
/// use rand::SeedableRng;
///
/// // Set up initial state with simple model
/// let model_fn = || sample(addr!("mu"), Normal { mu: 0.0, sigma: 1.0 });
///
/// let mut rng = StdRng::seed_from_u64(42);
/// let (_, initial_trace) = runtime::handler::run(
///     PriorHandler { rng: &mut rng, trace: Trace::default() },
///     model_fn()
/// );
///
/// // Perform one MH step
/// let mut scales = AdaptiveScales::new();
/// let (result, new_trace) = adaptive_single_site_mh(
///     &mut rng,
///     model_fn,
///     &initial_trace,
///     &mut scales,
/// );
/// assert!(new_trace.choices.len() > 0);
/// ```
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

/// Run an adaptive MCMC chain with automatic proposal tuning.
///
/// This is the main entry point for running Metropolis-Hastings MCMC on a
/// probabilistic model. It automatically handles initialization, warmup/burn-in,
/// and adaptive tuning of proposal scales to achieve good mixing.
///
/// # Algorithm
///
/// 1. Initialize chain with a prior sample
/// 2. Run warmup period, discarding samples but adapting scales
/// 3. Collect samples with the tuned proposal scales
/// 4. Return the post-warmup samples
///
/// # Arguments
///
/// * `rng` - Random number generator
/// * `model_fn` - Function that creates the model (should be the same each time)
/// * `n_samples` - Number of post-warmup samples to collect
/// * `n_warmup` - Number of warmup/burn-in iterations (not returned)
///
/// # Returns
///
/// Vector of (model_result, trace) pairs from the post-warmup sampling.
///
/// # Examples
///
/// ```rust
/// use fugue::*;
/// use rand::rngs::StdRng;
/// use rand::SeedableRng;
///
/// // Very simple model for testing
/// let model_fn = || {
///     sample(addr!("mu"), Normal { mu: 0.0, sigma: 1.0 })
/// };
///
/// let mut rng = StdRng::seed_from_u64(42);
/// let samples = adaptive_mcmc_chain(
///     &mut rng,
///     model_fn,
///     5, // samples (very small for test)
///     1, // warmup
/// );
///
/// // Extract mu estimates
/// let mu_values: Vec<f64> = samples.iter()
///     .filter_map(|(result, _)| Some(*result))
///     .collect();
/// assert!(!mu_values.is_empty());
/// ```
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
