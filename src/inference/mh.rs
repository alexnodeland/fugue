//! Metropolis-Hastings MCMC with adaptive tuning and single-site updates.
//!
//! This module implements the Metropolis-Hastings algorithm, a fundamental MCMC method
//! for sampling from posterior distributions. The implementation features:
//!
//! - **Adaptive scaling**: Automatically tunes proposal step sizes to achieve target acceptance rates
//! - **Single-site updates**: Updates one random variable at a time for better mixing
//! - **Type-safe proposals**: Preserves original types (bool, u64, usize, etc.) during proposals
//! - **Type-aware proposals**: Uses ProposalStrategy traits based on value types
//!
//! ## Constraint-Aware Proposals
//!
//! The implementation now uses **constraint-aware proposals** that automatically detect
//! and respect parameter constraints based on address names and value ranges:
//!
//! - **Positive parameters** (sigma, scale, rate, etc.) → Log-space proposals (maintains positivity)
//! - **Probability parameters** (p, prob, beta in [0,1]) → Reflection proposals (maintains bounds)
//! - **Unconstrained parameters** (mu, intercept, etc.) → Gaussian proposals (standard)
//!
//! This automatic constraint detection significantly improves MCMC performance and prevents
//! common issues like negative standard deviations or out-of-bounds probability values.
//!
//! For custom distributions requiring specialized proposals (logit-transform for Beta,
//! circular proposals for von Mises, etc.), consider implementing custom ProposalStrategy
//! implementations or contributing distribution-aware extensions.
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
//!     sample(addr!("mu"), Normal::new(0.0, 2.0).unwrap())
//!         .bind(|mu| observe(addr!("y"), Normal::new(mu, 1.0).unwrap(), 2.5))
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
use crate::core::model::Model;
use crate::inference::mcmc_utils::DiminishingAdaptation;
// All proposal logic is now integrated in this module
use crate::runtime::handler::run;
use crate::runtime::interpreters::{PriorHandler, ScoreGivenTrace};
use crate::runtime::trace::{Choice, ChoiceValue, Trace};
use rand::{Rng, RngCore};

/// Trait for distribution-aware proposal strategies.
///
/// This enables more intelligent proposals that take advantage of the distribution
/// structure rather than using generic random walks.
pub trait ProposalStrategy<T> {
    /// Generate a proposal given the current value and scale.
    fn propose(&self, current: T, scale: f64, rng: &mut dyn RngCore) -> T;

    /// Compute the log probability of proposing `to` given `from` (for asymmetric proposals).
    fn log_proposal_prob(&self, from: T, to: T, scale: f64) -> f64 {
        let _ = (from, to, scale);
        0.0 // Default: symmetric proposal
    }
}

/// Gaussian random walk proposal for continuous distributions.
pub struct GaussianWalkProposal;

impl ProposalStrategy<f64> for GaussianWalkProposal {
    fn propose(&self, current: f64, scale: f64, rng: &mut dyn RngCore) -> f64 {
        // Use Box-Muller for better numerical stability
        let u1: f64 = rng.gen::<f64>().max(1e-10); // Avoid log(0)
        let u2: f64 = rng.gen();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        current + scale * z
    }
}

/// Log-space random walk proposal for positive-constrained continuous distributions.
///
/// This proposal strategy works in log-space to maintain positivity constraints.
/// It's appropriate for parameters that must be positive (e.g., standard deviations,
/// rates, scales from Gamma, Exponential, LogNormal distributions).
pub struct LogSpaceWalkProposal;

impl ProposalStrategy<f64> for LogSpaceWalkProposal {
    fn propose(&self, current: f64, scale: f64, rng: &mut dyn RngCore) -> f64 {
        if current <= 0.0 {
            // If current value is non-positive, return a small positive value
            return 1e-6;
        }

        // Work in log-space to maintain positivity
        let log_current = current.ln();

        // Use Box-Muller for Gaussian proposal in log-space
        let u1: f64 = rng.gen::<f64>().max(1e-10);
        let u2: f64 = rng.gen();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();

        let log_proposed = log_current + scale * z;
        log_proposed.exp().max(1e-10) // Ensure minimum positive value
    }
}

/// Reflection-based proposal for bounded continuous distributions.
///
/// This proposal strategy reflects off the boundaries to maintain constraints
/// for distributions with finite support (e.g., Beta distribution on [0,1],
/// Uniform distribution on [a,b]).
pub struct ReflectionWalkProposal {
    /// Lower bound (inclusive)
    pub lower_bound: f64,
    /// Upper bound (inclusive)  
    pub upper_bound: f64,
}

impl ProposalStrategy<f64> for ReflectionWalkProposal {
    fn propose(&self, current: f64, scale: f64, rng: &mut dyn RngCore) -> f64 {
        // Generate Gaussian proposal
        let u1: f64 = rng.gen::<f64>().max(1e-10);
        let u2: f64 = rng.gen();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();

        let mut proposed = current + scale * z;

        // Reflect off boundaries until within bounds
        let range = self.upper_bound - self.lower_bound;
        if range <= 0.0 {
            return current; // Invalid bounds, return current
        }

        while proposed < self.lower_bound || proposed > self.upper_bound {
            if proposed < self.lower_bound {
                proposed = 2.0 * self.lower_bound - proposed;
            }
            if proposed > self.upper_bound {
                proposed = 2.0 * self.upper_bound - proposed;
            }
        }

        proposed.clamp(self.lower_bound, self.upper_bound)
    }
}

/// Flip proposal for boolean distributions.
pub struct FlipProposal;

impl ProposalStrategy<bool> for FlipProposal {
    fn propose(&self, current: bool, _scale: f64, rng: &mut dyn RngCore) -> bool {
        // For Bernoulli, always propose the opposite value for good mixing
        if rng.gen::<f64>() < 0.5 {
            !current
        } else {
            current
        }
    }
}

/// Discrete random walk proposal for count distributions.
pub struct DiscreteWalkProposal;

impl ProposalStrategy<u64> for DiscreteWalkProposal {
    fn propose(&self, current: u64, scale: f64, rng: &mut dyn RngCore) -> u64 {
        let u1: f64 = rng.gen::<f64>().max(1e-10);
        let u2: f64 = rng.gen();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        let delta = (scale * z).round() as i64;
        (current as i64 + delta).max(0) as u64
    }
}

/// Uniform proposal for categorical distributions.
///
/// This proposal strategy is distribution-aware and uses the actual size
/// of the categorical distribution when available.
pub struct UniformCategoricalProposal {
    /// Number of categories in the distribution.
    pub n_categories: Option<usize>,
}

impl ProposalStrategy<usize> for UniformCategoricalProposal {
    fn propose(&self, current: usize, _scale: f64, rng: &mut dyn RngCore) -> usize {
        match self.n_categories {
            Some(n) => rng.gen_range(0..n),
            None => {
                // Fallback heuristic if we don't know the true size
                let max_val = (current + 5).max(10);
                rng.gen_range(0..max_val)
            }
        }
    }
}

/// Unified proposal system using ProposalStrategy traits.
///
/// This function uses the appropriate ProposalStrategy for each type,
/// ensuring type safety and allowing for constraint-aware proposals.
///
/// For f64 values, it applies heuristics to detect likely constraints:
/// - If current value > 0 and seems like a scale/rate parameter, use log-space proposal
/// - Otherwise use standard Gaussian proposal
fn propose_using_strategies<R: RngCore>(rng: &mut R, choice: &Choice, scale: f64) -> ChoiceValue {
    match choice.value {
        ChoiceValue::F64(current_val) => {
            // Heuristic: if current value is positive and the address suggests a scale/rate parameter,
            // use log-space proposal to maintain positivity
            let addr_str = choice.addr.0.to_lowercase();
            let looks_like_scale_param = addr_str.contains("sigma")
                || addr_str.contains("scale")
                || addr_str.contains("rate")
                || addr_str.contains("lambda")
                || addr_str.contains("tau")
                || addr_str.contains("precision")
                || addr_str.contains("nu");

            let strategy: Box<dyn ProposalStrategy<f64>> =
                if current_val > 0.0 && looks_like_scale_param {
                    Box::new(LogSpaceWalkProposal)
                } else if (0.0..=1.0).contains(&current_val)
                    && (addr_str.contains("prob")
                        || addr_str.contains("p")
                        || addr_str.contains("beta"))
                {
                    // Likely a probability parameter - use reflection on [0,1]
                    Box::new(ReflectionWalkProposal {
                        lower_bound: 0.0,
                        upper_bound: 1.0,
                    })
                } else {
                    Box::new(GaussianWalkProposal)
                };

            let proposed = strategy.propose(current_val, scale, rng);
            ChoiceValue::F64(proposed)
        }
        ChoiceValue::Bool(current_val) => {
            let strategy = FlipProposal;
            let proposed = strategy.propose(current_val, scale, rng);
            ChoiceValue::Bool(proposed)
        }
        ChoiceValue::U64(current_val) => {
            let strategy = DiscreteWalkProposal;
            let proposed = strategy.propose(current_val, scale, rng);
            ChoiceValue::U64(proposed)
        }
        ChoiceValue::I64(current_val) => {
            // Convert to u64, propose, then convert back with proper bounds
            let as_u64 = current_val.max(0) as u64;
            let strategy = DiscreteWalkProposal;
            let proposed_u64 = strategy.propose(as_u64, scale, rng);
            // Convert back to i64, handling potential overflow
            let proposed = proposed_u64.min(i64::MAX as u64) as i64;
            // Apply the original sign pattern if current_val was negative
            let final_proposed = if current_val < 0 && proposed > 0 && rng.gen::<bool>() {
                -proposed
            } else {
                proposed
            };
            ChoiceValue::I64(final_proposed)
        }
        ChoiceValue::Usize(current_val) => {
            // Use uniform categorical proposal with reasonable heuristic
            let strategy = UniformCategoricalProposal {
                n_categories: None, // Will use heuristic
            };
            let proposed = strategy.propose(current_val, scale, rng);
            ChoiceValue::Usize(proposed)
        }
    }
}

/// Perform a single adaptive Metropolis-Hastings update step.
///
/// This function implements a single iteration of the MH algorithm with proper
/// diminishing adaptation that preserves ergodicity. It randomly selects one site
/// to update, proposes a new value using adaptive scaling, and accepts or rejects
/// based on the Metropolis-Hastings criterion.
///
/// # Algorithm
///
/// 1. Randomly select a site from the current trace
/// 2. Propose a new value using diminishing adaptive scaling
/// 3. Score both current and proposed traces with numerical stability
/// 4. Accept with probability min(1, exp(log_prob_new - log_prob_old))
/// 5. Update adaptive scales using diminishing step sizes
///
/// # Arguments
///
/// * `rng` - Random number generator
/// * `model_fn` - Function that creates the model
/// * `current` - Current trace (state of the Markov chain)
/// * `adaptation` - Diminishing adaptation system (modified in-place)
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
/// let model_fn = || sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap());
///
/// let mut rng = StdRng::seed_from_u64(42);
/// let (_, initial_trace) = runtime::handler::run(
///     PriorHandler { rng: &mut rng, trace: Trace::default() },
///     model_fn()
/// );
///
/// // Perform one MH step
/// let mut adaptation = DiminishingAdaptation::new(0.44, 0.7);
/// let (result, new_trace) = adaptive_single_site_mh(
///     &mut rng,
///     model_fn,
///     &initial_trace,
///     &mut adaptation,
/// );
/// assert!(new_trace.choices.len() > 0);
/// ```
pub fn adaptive_single_site_mh<A, R: Rng>(
    rng: &mut R,
    model_fn: impl Fn() -> Model<A>,
    current: &Trace,
    adaptation: &mut DiminishingAdaptation,
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

    // Get current choice and propose new value using ProposalStrategy traits
    let current_choice = &current.choices[&selected_site];
    let scale = adaptation.get_scale(&selected_site);
    let proposed_value = propose_using_strategies(rng, current_choice, scale);

    // Create proposed trace - preserving type safety
    let mut proposed_trace = current.clone();
    proposed_trace
        .choices
        .get_mut(&selected_site)
        .unwrap()
        .value = proposed_value;

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

    // Update adaptation
    adaptation.update(&selected_site, accept);

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
///     sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap())
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
    let mut adaptation = DiminishingAdaptation::new(0.44, 0.7);

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
        let (_, trace) = adaptive_single_site_mh(rng, &model_fn, &current_trace, &mut adaptation);
        current_trace = trace;
    }

    // Sampling phase
    for _ in 0..n_samples {
        let (val, trace) = adaptive_single_site_mh(rng, &model_fn, &current_trace, &mut adaptation);
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
    let mut adaptation = DiminishingAdaptation::new(0.44, 0.7);
    adaptive_single_site_mh(rng, model_fn, current, &mut adaptation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::addr;
    use crate::core::distribution::*;
    use crate::core::model::{observe, sample, ModelExt};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn gaussian_walk_proposal_produces_variation() {
        let mut rng = StdRng::seed_from_u64(11);
        let strat = GaussianWalkProposal;
        let x0 = 0.0;
        let x1 = strat.propose(x0, 1.0, &mut rng);
        // With probability 1 it's not guaranteed to change, but very likely; ensure finiteness
        assert!(x1.is_finite());
    }

    #[test]
    fn log_space_proposal_maintains_positivity() {
        let mut rng = StdRng::seed_from_u64(42);
        let strat = LogSpaceWalkProposal;

        // Test with various positive values
        for &current in &[0.1, 1.0, 10.0, 100.0] {
            for _ in 0..20 {
                let proposed = strat.propose(current, 0.5, &mut rng);
                assert!(
                    proposed > 0.0,
                    "LogSpaceWalk proposed negative value: {} -> {}",
                    current,
                    proposed
                );
                assert!(
                    proposed.is_finite(),
                    "LogSpaceWalk proposed non-finite value: {}",
                    proposed
                );
            }
        }

        // Test with edge case: non-positive input
        let proposed = strat.propose(-1.0, 0.5, &mut rng);
        assert!(
            proposed > 0.0,
            "LogSpaceWalk should return positive value for negative input"
        );
    }

    #[test]
    fn reflection_proposal_respects_bounds() {
        let mut rng = StdRng::seed_from_u64(43);
        let strat = ReflectionWalkProposal {
            lower_bound: 0.0,
            upper_bound: 1.0,
        };

        // Test with values in [0,1] range
        for &current in &[0.1, 0.5, 0.9] {
            for _ in 0..20 {
                let proposed = strat.propose(current, 0.3, &mut rng);
                assert!(
                    (0.0..=1.0).contains(&proposed),
                    "ReflectionWalk violated bounds: {} -> {}",
                    current,
                    proposed
                );
                assert!(
                    proposed.is_finite(),
                    "ReflectionWalk proposed non-finite value: {}",
                    proposed
                );
            }
        }
    }

    #[test]
    fn constraint_aware_proposals_work() {
        let mut rng = StdRng::seed_from_u64(44);

        // Test sigma parameter (should use log-space)
        let sigma_choice = Choice {
            addr: crate::addr!("sigma"),
            value: ChoiceValue::F64(2.0),
            logp: -1.0,
        };

        for _ in 0..10 {
            let proposed = propose_using_strategies(&mut rng, &sigma_choice, 0.5);
            if let ChoiceValue::F64(val) = proposed {
                assert!(val > 0.0, "Sigma proposal should be positive: {}", val);
            } else {
                panic!("Expected F64 value");
            }
        }

        // Test regular parameter (should use standard Gaussian)
        let mu_choice = Choice {
            addr: crate::addr!("mu"),
            value: ChoiceValue::F64(0.0),
            logp: -0.5,
        };

        let proposed = propose_using_strategies(&mut rng, &mu_choice, 1.0);
        if let ChoiceValue::F64(val) = proposed {
            assert!(val.is_finite(), "Mu proposal should be finite: {}", val);
            // Note: mu can be negative, so we don't check positivity
        } else {
            panic!("Expected F64 value");
        }
    }

    #[test]
    fn discrete_and_flip_proposals_preserve_types() {
        let mut rng = StdRng::seed_from_u64(12);
        let d = DiscreteWalkProposal;
        let u = d.propose(5u64, 1.0, &mut rng);
        // Note: u is u64, so this comparison is always true, but kept for documentation
        let _ = u; // Just verify it's a valid u64
        let f = FlipProposal;
        let b = f.propose(true, 1.0, &mut rng);
        let _ = b; // Just checking that we got a valid bool
    }

    #[test]
    fn adaptive_chain_runs_and_returns_samples() {
        let model_fn = || {
            sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap()).and_then(|mu| {
                observe(addr!("y"), Normal::new(mu, 1.0).unwrap(), 0.5).map(move |_| mu)
            })
        };
        let mut rng = StdRng::seed_from_u64(13);
        let samples = adaptive_mcmc_chain(&mut rng, model_fn, 5, 2);
        assert_eq!(samples.len(), 5);
        // Ensure types are preserved in trace
        for (_val, t) in &samples {
            assert!(t.get_f64(&addr!("mu")).is_some());
        }
    }
}
