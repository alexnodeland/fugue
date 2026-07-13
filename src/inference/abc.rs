//! Approximate Bayesian Computation (ABC) - likelihood-free inference methods.
//!
//! ABC methods enable Bayesian inference for models where the likelihood function
//! is intractable or unavailable, but forward simulation from the model is possible.
//! Instead of computing likelihoods directly, ABC compares simulated data to observed
//! data using distance functions and accepts samples that produce "similar" outcomes.
//!
//! ## Method Overview
//!
//! ABC algorithms follow this general pattern:
//! 1. Sample parameters from the prior distribution
//! 2. Simulate data using the model with those parameters
//! 3. Compare simulated data to observed data using a distance function
//! 4. Accept samples where the distance is below a threshold ε
//!
//! As ε → 0, the ABC posterior approaches the true posterior distribution.
//!
//! ## Available Methods
//!
//! - [`abc_rejection`]: Basic rejection ABC
//! - [`abc_smc`]: Sequential Monte Carlo ABC for improved efficiency
//! - [`abc_scalar_summary`]: ABC with scalar summary statistics
//!
//! ## Distance Functions
//!
//! The quality of ABC inference depends heavily on the choice of distance function:
//! - [`EuclideanDistance`]: L2 norm for continuous data vectors
//! - [`ManhattanDistance`]: L1 norm for robust distance computation
//! - Custom distance functions via the [`DistanceFunction`] trait
//!
//! # Examples
//!
//! ```rust
//! use fugue::*;
//! use rand::rngs::StdRng;
//! use rand::SeedableRng;
//! use rand::Rng;
//!
//! // Simple ABC example for illustration
//! let mut rng = StdRng::seed_from_u64(42);
//! let observed_data = vec![2.0];
//!
//! let samples = abc_scalar_summary(
//!     &mut rng,
//!     || sample(addr!("mu"), Normal::new(0.0, 2.0).unwrap()),
//!     |trace| {
//!         if let Some(choice) = trace.choices.get(&addr!("mu")) {
//!             if let ChoiceValue::F64(mu) = choice.value {
//!                 mu
//!             } else { 0.0 }
//!         } else { 0.0 }
//!     },
//!     2.0, // observed summary
//!     0.5, // tolerance
//!     10   // max samples
//! );
//!
//! assert!(!samples.is_empty());
//! ```

use crate::core::address::Address;
use crate::core::distribution::{Distribution, Normal};
use crate::core::model::Model;
use crate::core::numerical::log_sum_exp;
use crate::runtime::handler::run;
use crate::runtime::interpreters::{PriorHandler, ScoreGivenTrace};
use crate::runtime::trace::{ChoiceValue, Trace};
use rand::Rng;

/// Trait for computing distances between observed and simulated data.
///
/// Distance functions are crucial for ABC methods as they determine how
/// "similarity" between datasets is measured. The choice of distance function
/// significantly affects the quality of ABC approximations.
///
/// # Type Parameter
///
/// * `T` - Type of data being compared (e.g., `Vec<f64>`, scalar values)
///
/// # Examples
///
/// ```rust
/// use fugue::*;
///
/// // Use built-in Euclidean distance
/// let euclidean = EuclideanDistance;
/// let dist = euclidean.distance(&vec![1.0, 2.0], &vec![1.1, 2.1]);
///
/// // Implement custom distance function
/// struct ScalarDistance;
/// impl DistanceFunction<f64> for ScalarDistance {
///     fn distance(&self, observed: &f64, simulated: &f64) -> f64 {
///         (observed - simulated).abs()
///     }
/// }
/// ```
pub trait DistanceFunction<T> {
    /// Compute the distance between observed and simulated data.
    ///
    /// # Arguments
    ///
    /// * `observed` - The actual observed data
    /// * `simulated` - Data simulated from the model
    ///
    /// # Returns
    ///
    /// A non-negative distance value. Smaller values indicate greater similarity.
    fn distance(&self, observed: &T, simulated: &T) -> f64;
}

/// Euclidean (L2) distance function for vector data.
///
/// Computes the standard Euclidean distance between two vectors:
/// √(Σ(xᵢ - yᵢ)²)
///
/// This is appropriate for continuous data where the magnitude of differences
/// matters and the data dimensions have similar scales.
///
/// # Examples
///
/// ```rust
/// use fugue::*;
///
/// let euclidean = EuclideanDistance;
/// let observed = vec![1.0, 2.0, 3.0];
/// let simulated = vec![1.1, 2.1, 2.9];
/// let distance = euclidean.distance(&observed, &simulated);
/// assert!((distance - 0.173).abs() < 0.01); // ≈ 0.173
/// ```
pub struct EuclideanDistance;

impl DistanceFunction<Vec<f64>> for EuclideanDistance {
    fn distance(&self, observed: &Vec<f64>, simulated: &Vec<f64>) -> f64 {
        if observed.len() != simulated.len() {
            return f64::INFINITY;
        }

        observed
            .iter()
            .zip(simulated.iter())
            .map(|(&o, &s)| (o - s).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

/// Manhattan (L1) distance function for vector data.
///
/// Computes the Manhattan distance between two vectors:
/// Σ|xᵢ - yᵢ|
///
/// This distance is more robust to outliers than Euclidean distance and is
/// appropriate when you want to treat each dimension independently.
///
/// # Examples
///
/// ```rust
/// use fugue::inference::abc::{ManhattanDistance, DistanceFunction};
///
/// let manhattan = ManhattanDistance;
/// let observed = vec![1.0, 2.0, 3.0];
/// let simulated = vec![1.5, 1.5, 3.5];
/// let distance = manhattan.distance(&observed, &simulated);
/// assert!((distance - 1.5).abs() < 0.001); // |1.0-1.5| + |2.0-1.5| + |3.0-3.5| = 0.5 + 0.5 + 0.5 = 1.5
/// ```
pub struct ManhattanDistance;

impl DistanceFunction<Vec<f64>> for ManhattanDistance {
    fn distance(&self, observed: &Vec<f64>, simulated: &Vec<f64>) -> f64 {
        if observed.len() != simulated.len() {
            return f64::INFINITY;
        }

        observed
            .iter()
            .zip(simulated.iter())
            .map(|(&o, &s)| (o - s).abs())
            .sum::<f64>()
    }
}

/// Summary statistics distance.
pub struct SummaryStatsDistance {
    pub weights: Vec<f64>,
}

impl SummaryStatsDistance {
    pub fn new(weights: Vec<f64>) -> Self {
        Self { weights }
    }

    fn compute_stats(data: &[f64]) -> Vec<f64> {
        if data.is_empty() {
            return vec![0.0, 0.0, 0.0];
        }

        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
        let std = variance.sqrt();

        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if sorted.len().is_multiple_of(2) {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };

        vec![mean, std, median]
    }
}

impl DistanceFunction<Vec<f64>> for SummaryStatsDistance {
    fn distance(&self, observed: &Vec<f64>, simulated: &Vec<f64>) -> f64 {
        let obs_stats = Self::compute_stats(observed);
        let sim_stats = Self::compute_stats(simulated);

        obs_stats
            .iter()
            .zip(sim_stats.iter())
            .zip(&self.weights)
            .map(|((&o, &s), &w)| w * (o - s).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

/// Basic ABC rejection sampling algorithm.
///
/// The simplest ABC method: repeatedly sample from the prior, simulate data,
/// and accept samples where the distance to observed data is below a tolerance.
/// This method is straightforward but can be inefficient for small tolerances.
///
/// # Algorithm
///
/// 1. Sample parameters from the prior using `model_fn()`
/// 2. Simulate data using `simulator(trace)`
/// 3. Compute distance between simulated and observed data
/// 4. Accept if distance ≤ tolerance
/// 5. Repeat until `max_samples` accepted or too many attempts
///
/// # Arguments
///
/// * `rng` - Random number generator
/// * `model_fn` - Function that creates a model instance (contains priors)
/// * `simulator` - Function that simulates data given a trace of parameter values
/// * `observed_data` - The actual observed data to match
/// * `distance_fn` - Function for measuring similarity between datasets
/// * `tolerance` - Maximum allowed distance for acceptance
/// * `max_samples` - Maximum number of samples to accept
///
/// # Returns
///
/// Vector of accepted traces (parameter samples that produced similar data).
///
/// # Examples
///
/// ```rust
/// use fugue::*;
/// use rand::rngs::StdRng;
/// use rand::SeedableRng;
///
/// // Simple ABC rejection example
/// let mut rng = StdRng::seed_from_u64(42);
/// let observed_data = vec![2.0];
///
/// let samples = abc_scalar_summary(
///     &mut rng,
///     || sample(addr!("mu"), Normal::new(0.0, 2.0).unwrap()),
///     |trace| {
///         if let Some(choice) = trace.choices.get(&addr!("mu")) {
///             if let ChoiceValue::F64(mu) = choice.value {
///                 mu
///             } else { 0.0 }
///         } else { 0.0 }
///     },
///     2.0, // observed summary
///     0.5, // tolerance
///     5    // max samples (small for test)
/// );
/// assert!(!samples.is_empty());
/// ```
pub fn abc_rejection<A, T, R: Rng>(
    rng: &mut R,
    model_fn: impl Fn() -> Model<A>,
    simulator: impl Fn(&Trace) -> T,
    observed_data: &T,
    distance_fn: &dyn DistanceFunction<T>,
    tolerance: f64,
    max_samples: usize,
) -> Vec<Trace> {
    let mut accepted = Vec::new();
    let mut attempts = 0;

    while accepted.len() < max_samples && attempts < max_samples * 100 {
        // Sample from prior
        let (_a, trace) = run(
            PriorHandler {
                rng,
                trace: Trace::default(),
            },
            model_fn(),
        );

        // Simulate data
        let simulated_data = simulator(&trace);

        // Check distance
        let dist = distance_fn.distance(observed_data, &simulated_data);

        if dist <= tolerance {
            accepted.push(trace);
        }

        attempts += 1;
    }

    if accepted.is_empty() {
        eprintln!(
            "Warning: No samples accepted in ABC. Consider increasing tolerance or max_samples."
        );
    }

    accepted
}

/// Sequential Monte Carlo ABC with adaptive tolerance scheduling.
///
/// An importance-weighted ABC-SMC (Beaumont 2009 / Toni et al. 2009) that
/// iteratively reduces the tolerance, giving better posterior approximations than
/// rejection ABC at stringent tolerances. See [`abc_smc`] (equally-weighted
/// population) and [`abc_smc_weighted`] (weighted population with typed errors).
///
/// # Algorithm
///
/// 1. Start with the initial tolerance and generate a population using rejection ABC.
/// 2. For each subsequent tolerance level:
///    - draw a base particle from the previous population proportional to its
///      importance weight,
///    - perturb its continuous coordinates with a Gaussian kernel scaled by the
///      weighted sample variance,
///    - reject out-of-support proposals and accept those within the new tolerance,
///    - weight each accepted particle by `pi(theta) / sum_j w_j K(theta | theta_j)`.
/// 3. Final particles approximate the posterior at the strictest tolerance.
///
/// # Arguments
///
/// * `rng` - Random number generator
/// * `model_fn` - Function that creates a model instance
/// * `simulator` - Function that simulates data given a trace
/// * `observed_data` - The observed data to match
/// * `distance_fn` - Distance function for comparing datasets
/// * `config` - Initial tolerance, decreasing tolerance schedule, population size
///
/// # Returns
///
/// Vector of traces from the final SMC population.
///
/// # Examples
///
/// ```rust
/// use fugue::{inference::abc::ABCSMCConfig, *};
/// use rand::rngs::StdRng;
/// use rand::SeedableRng;
///
/// // Simple SMC-ABC example with small numbers for testing
/// let observed = vec![2.0];
/// let mut rng = StdRng::seed_from_u64(42);
///
/// let samples = abc_smc(
///     &mut rng,
///     || sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap()),
///     |trace| {
///         if let Some(choice) = trace.choices.get(&addr!("mu")) {
///             if let ChoiceValue::F64(mu) = choice.value {
///                 vec![mu]
///             } else { vec![0.0] }
///         } else { vec![0.0] }
///     },
///     &observed,
///     &EuclideanDistance,
///     ABCSMCConfig {
///         initial_tolerance: 1.0,
///         tolerance_schedule: vec![0.5],
///         particles_per_round: 5,
///     },
/// );
/// assert!(!samples.is_empty());
/// ```
/// Configuration for ABC-SMC algorithm.
#[derive(Debug, Clone)]
pub struct ABCSMCConfig {
    /// Initial tolerance for distance threshold
    pub initial_tolerance: f64,
    /// Schedule of decreasing tolerances across rounds  
    pub tolerance_schedule: Vec<f64>,
    /// Number of particles to generate per round
    pub particles_per_round: usize,
}

/// Default per-stage attempt budget as a multiple of the population size,
/// mirroring the `max_samples * 100` bound used by [`abc_rejection`].
pub const ABC_SMC_DEFAULT_ATTEMPT_FACTOR: usize = 100;

/// Errors that can occur during a bounded ABC-SMC run (finding FG-34).
#[derive(Debug, Clone, PartialEq)]
pub enum ABCError {
    /// The initial rejection round accepted zero particles within its attempt
    /// budget, so there is nothing to perturb. Previously this panicked in
    /// `rng.gen_range(0..0)`.
    EmptyInitialPopulation {
        /// The initial tolerance that admitted no samples.
        tolerance: f64,
        /// Number of prior draws attempted before giving up.
        attempts: usize,
    },
    /// A tolerance stage could not be filled within its attempt budget.
    /// Previously the inner loop had no cap and could spin forever.
    StageExhausted {
        /// The tolerance level that could not be reached.
        tolerance: f64,
        /// Number of particles accepted before the budget was exhausted.
        accepted: usize,
        /// Number of particles requested for the stage.
        requested: usize,
        /// Attempt budget that was exhausted.
        attempts: usize,
    },
}

impl std::fmt::Display for ABCError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ABCError::EmptyInitialPopulation {
                tolerance,
                attempts,
            } => write!(
                f,
                "ABC-SMC initial population is empty: no draw fell within tolerance {tolerance} in {attempts} attempts"
            ),
            ABCError::StageExhausted {
                tolerance,
                accepted,
                requested,
                attempts,
            } => write!(
                f,
                "ABC-SMC stage at tolerance {tolerance} exhausted its budget of {attempts} attempts with only {accepted}/{requested} particles accepted"
            ),
        }
    }
}

impl std::error::Error for ABCError {}

/// A weighted ABC-SMC particle: a parameter trace with its importance weight.
#[derive(Debug, Clone)]
pub struct ABCParticle {
    /// The accepted parameter trace.
    pub trace: Trace,
    /// Normalized importance weight within the population.
    pub weight: f64,
}

/// Result of a correct ABC-SMC run: a weighted posterior population.
#[derive(Debug, Clone)]
pub struct ABCSMCResult {
    /// The final weighted particle population (weights sum to 1).
    pub particles: Vec<ABCParticle>,
    /// The tolerance level of the final population.
    pub final_tolerance: f64,
}

impl ABCSMCResult {
    /// Weighted posterior mean of the f64 value at `addr`, if present.
    pub fn weighted_mean(&self, addr: &Address) -> Option<f64> {
        let mut num = 0.0;
        let mut den = 0.0;
        for p in &self.particles {
            let v = p.trace.get_f64(addr)?;
            num += p.weight * v;
            den += p.weight;
        }
        if den > 0.0 {
            Some(num / den)
        } else {
            None
        }
    }
}

/// Sequential Monte Carlo ABC (Beaumont 2009 / Toni et al. 2009).
///
/// This is the correct, importance-weighted ABC-SMC. It fixes finding FG-09:
/// each population after the first is generated by
///
/// 1. drawing a base particle from the previous population *proportional to its
///    importance weight*,
/// 2. perturbing its continuous coordinates with a Gaussian kernel whose
///    per-component bandwidth is `sqrt(2 · weighted-variance)` of the previous
///    population (Beaumont et al. 2009),
/// 3. rejecting proposals with zero prior density (out of support), and
/// 4. accepting proposals within the new tolerance, then weighting each accepted
///    particle by `w_i ∝ π(θ_i) / Σ_j w_j K(θ_i | θ_j)` — the prior/kernel
///    correction that the previous single-site prior-replacement heuristic
///    omitted.
///
/// Each stage is bounded by `max_attempts_per_stage` attempts (finding FG-34);
/// an empty initial population and an exhausted stage are reported as typed
/// [`ABCError`]s instead of panicking / looping forever.
///
/// The perturbation kernel acts on the model's continuous (f64) sites; discrete
/// sites are carried through from the base particle unchanged, so the importance
/// correction is exact for continuous parameters.
///
/// # Returns
///
/// An [`ABCSMCResult`] with the weighted posterior population at the final
/// tolerance, or an [`ABCError`] if a stage could not be completed.
pub fn abc_smc_weighted<A, T, R: Rng>(
    rng: &mut R,
    model_fn: impl Fn() -> Model<A>,
    simulator: impl Fn(&Trace) -> T,
    observed_data: &T,
    distance_fn: &dyn DistanceFunction<T>,
    config: ABCSMCConfig,
    max_attempts_per_stage: usize,
) -> Result<ABCSMCResult, ABCError> {
    let n = config.particles_per_round;

    // ----- Population 0: bounded rejection ABC at the initial tolerance -----
    let mut current: Vec<ABCParticle> = Vec::with_capacity(n);
    let mut attempts = 0usize;
    while current.len() < n && attempts < max_attempts_per_stage {
        attempts += 1;
        let (_a, trace) = run(
            PriorHandler {
                rng,
                trace: Trace::default(),
            },
            model_fn(),
        );
        let dist = distance_fn.distance(observed_data, &simulator(&trace));
        if dist <= config.initial_tolerance {
            current.push(ABCParticle { trace, weight: 0.0 });
        }
    }
    if current.is_empty() {
        return Err(ABCError::EmptyInitialPopulation {
            tolerance: config.initial_tolerance,
            attempts,
        });
    }
    // Population 0 carries uniform weights.
    let uniform = 1.0 / current.len() as f64;
    for p in &mut current {
        p.weight = uniform;
    }
    let mut current_tolerance = config.initial_tolerance;

    // Continuous coordinate addresses shared by the population.
    let coord_addrs = f64_addresses(&current[0].trace);

    // ----- Sequential rounds with decreasing tolerance -----
    for &new_tolerance in &config.tolerance_schedule {
        if new_tolerance >= current_tolerance {
            continue; // Skip non-decreasing tolerances.
        }

        // Kernel bandwidth per continuous component: sqrt(2 * weighted variance).
        let kernel_std = kernel_bandwidths(&current, &coord_addrs);
        let prev_coords: Vec<Vec<f64>> = current
            .iter()
            .map(|p| coords_of(&p.trace, &coord_addrs))
            .collect();
        let prev_weights: Vec<f64> = current.iter().map(|p| p.weight).collect();

        let mut next: Vec<ABCParticle> = Vec::with_capacity(n);
        let mut log_weights: Vec<f64> = Vec::with_capacity(n);
        let mut stage_attempts = 0usize;

        while next.len() < n && stage_attempts < max_attempts_per_stage {
            stage_attempts += 1;

            // (1) Draw a base particle proportional to its importance weight.
            let j = sample_index(rng, &prev_weights);
            let mut proposed = current[j].trace.clone();

            // (2) Perturb continuous coordinates with the Gaussian kernel.
            for (c, addr) in coord_addrs.iter().enumerate() {
                if let Some(v) = proposed.get_f64(addr) {
                    let z = Normal::new(0.0, 1.0).unwrap().sample(rng);
                    let new_v = v + kernel_std[c] * z;
                    if let Some(choice) = proposed.choices.get_mut(addr) {
                        choice.value = ChoiceValue::F64(new_v);
                    }
                }
            }

            // (3) Reject proposals with zero prior density (out of support).
            let log_prior = score_log_prior(&model_fn, &proposed);
            if !log_prior.is_finite() {
                continue;
            }

            // (4) Accept within tolerance.
            let dist = distance_fn.distance(observed_data, &simulator(&proposed));
            if dist > new_tolerance {
                continue;
            }

            // Importance weight: log w = log π(θ) - log Σ_j w_j K(θ | θ_j).
            let prop_coords = coords_of(&proposed, &coord_addrs);
            let log_denom =
                kernel_mixture_log_density(&prop_coords, &prev_coords, &prev_weights, &kernel_std);
            log_weights.push(log_prior - log_denom);
            next.push(ABCParticle {
                trace: proposed,
                weight: 0.0,
            });
        }

        if next.is_empty() || next.len() < n {
            return Err(ABCError::StageExhausted {
                tolerance: new_tolerance,
                accepted: next.len(),
                requested: n,
                attempts: max_attempts_per_stage,
            });
        }

        // Normalize the importance weights (stable log-sum-exp).
        let log_norm = log_sum_exp(&log_weights);
        for (p, &lw) in next.iter_mut().zip(&log_weights) {
            p.weight = if log_norm.is_finite() {
                (lw - log_norm).exp()
            } else {
                1.0 / n as f64
            };
        }

        current = next;
        current_tolerance = new_tolerance;
    }

    Ok(ABCSMCResult {
        particles: current,
        final_tolerance: current_tolerance,
    })
}

/// Sequential Monte Carlo ABC returning an equally-weighted trace population.
///
/// This is the correct ABC-SMC of [`abc_smc_weighted`] (fixing finding FG-09),
/// wrapped for the common case: it runs the weighted algorithm with the default
/// per-stage attempt budget (`ABC_SMC_DEFAULT_ATTEMPT_FACTOR * particles_per_round`,
/// finding FG-34) and then resamples the final weighted population down to an
/// equally-weighted set of traces, so the returned traces can be summarized
/// directly (e.g. by an unweighted posterior mean).
///
/// Unlike the previous implementation, it never panics on an empty initial
/// population and never loops forever: on any [`ABCError`] it emits a warning and
/// returns an empty vector. Use [`abc_smc_weighted`] for the weighted population,
/// a configurable attempt budget, and typed error handling.
///
/// # Examples
///
/// ```rust
/// use fugue::{inference::abc::ABCSMCConfig, *};
/// use rand::rngs::StdRng;
/// use rand::SeedableRng;
///
/// let observed = vec![2.0];
/// let mut rng = StdRng::seed_from_u64(42);
///
/// let samples = abc_smc(
///     &mut rng,
///     || sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap()),
///     |trace| {
///         if let Some(choice) = trace.choices.get(&addr!("mu")) {
///             if let ChoiceValue::F64(mu) = choice.value {
///                 vec![mu]
///             } else { vec![0.0] }
///         } else { vec![0.0] }
///     },
///     &observed,
///     &EuclideanDistance,
///     ABCSMCConfig {
///         initial_tolerance: 1.0,
///         tolerance_schedule: vec![0.5],
///         particles_per_round: 20,
///     },
/// );
/// assert!(!samples.is_empty());
/// ```
pub fn abc_smc<A, T, R: Rng>(
    rng: &mut R,
    model_fn: impl Fn() -> Model<A>,
    simulator: impl Fn(&Trace) -> T,
    observed_data: &T,
    distance_fn: &dyn DistanceFunction<T>,
    config: ABCSMCConfig,
) -> Vec<Trace> {
    let n = config.particles_per_round;
    let max_attempts = ABC_SMC_DEFAULT_ATTEMPT_FACTOR.saturating_mul(n.max(1));
    match abc_smc_weighted(
        rng,
        model_fn,
        simulator,
        observed_data,
        distance_fn,
        config,
        max_attempts,
    ) {
        Ok(result) => {
            // Resample the weighted population to an equally-weighted trace set,
            // so the returned traces are a valid unweighted posterior sample.
            let weights: Vec<f64> = result.particles.iter().map(|p| p.weight).collect();
            (0..result.particles.len())
                .map(|_| result.particles[sample_index(rng, &weights)].trace.clone())
                .collect()
        }
        Err(e) => {
            eprintln!("Warning: ABC-SMC did not complete: {e}. Returning empty population.");
            Vec::new()
        }
    }
}

/// Ordered list of continuous (f64) sample-site addresses in a trace.
fn f64_addresses(trace: &Trace) -> Vec<Address> {
    trace
        .choices
        .iter()
        .filter(|(_, c)| matches!(c.value, ChoiceValue::F64(_)))
        .map(|(a, _)| a.clone())
        .collect()
}

/// Extract the f64 coordinate vector of a trace at the given addresses.
fn coords_of(trace: &Trace, addrs: &[Address]) -> Vec<f64> {
    addrs
        .iter()
        .map(|a| trace.get_f64(a).unwrap_or(0.0))
        .collect()
}

/// Per-component kernel bandwidth `sqrt(2 · weighted variance)` of the population
/// (Beaumont et al. 2009). Falls back to a small positive value for degenerate
/// (zero-variance) components so the kernel never collapses to a point mass.
fn kernel_bandwidths(population: &[ABCParticle], addrs: &[Address]) -> Vec<f64> {
    let mut std = vec![0.0; addrs.len()];
    let total_w: f64 = population.iter().map(|p| p.weight).sum();
    if total_w <= 0.0 {
        return vec![1e-3; addrs.len()];
    }
    for (c, addr) in addrs.iter().enumerate() {
        let mut mean = 0.0;
        for p in population {
            mean += p.weight * p.trace.get_f64(addr).unwrap_or(0.0);
        }
        mean /= total_w;
        let mut var = 0.0;
        for p in population {
            let d = p.trace.get_f64(addr).unwrap_or(0.0) - mean;
            var += p.weight * d * d;
        }
        var /= total_w;
        let bw = (2.0 * var).sqrt();
        std[c] = if bw > 1e-12 { bw } else { 1e-3 };
    }
    std
}

/// log Σ_j w_j K(x | θ_j) for a component-wise Gaussian kernel with std `kernel_std`.
fn kernel_mixture_log_density(
    x: &[f64],
    centers: &[Vec<f64>],
    weights: &[f64],
    kernel_std: &[f64],
) -> f64 {
    let terms: Vec<f64> = centers
        .iter()
        .zip(weights)
        .map(|(center, &w)| w.ln() + gaussian_log_density(x, center, kernel_std))
        .collect();
    log_sum_exp(&terms)
}

/// Component-wise Gaussian log density Σ_c log N(x_c; mean_c, std_c).
fn gaussian_log_density(x: &[f64], mean: &[f64], std: &[f64]) -> f64 {
    let mut lp = 0.0;
    for ((&xi, &mi), &si) in x.iter().zip(mean).zip(std) {
        let s = si.max(1e-12);
        let z = (xi - mi) / s;
        lp += -0.5 * z * z - s.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln();
    }
    lp
}

/// Score a trace under the model and return its log prior density log π(θ).
///
/// Returns `-inf` when any perturbed value falls outside its support.
fn score_log_prior<A>(model_fn: &impl Fn() -> Model<A>, trace: &Trace) -> f64 {
    let (_a, scored) = run(
        ScoreGivenTrace {
            base: trace.clone(),
            trace: Trace::default(),
        },
        model_fn(),
    );
    scored.log_prior
}

/// Sample an index in `0..weights.len()` proportional to `weights`.
fn sample_index<R: Rng>(rng: &mut R, weights: &[f64]) -> usize {
    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return rng.gen_range(0..weights.len());
    }
    let u = rng.gen::<f64>() * total;
    let mut cum = 0.0;
    for (i, &w) in weights.iter().enumerate() {
        cum += w;
        if u <= cum {
            return i;
        }
    }
    weights.len() - 1
}

/// ABC rejection sampling using scalar summary statistics.
///
/// A convenience function for ABC when both observed and simulated data can be
/// reduced to scalar summary statistics. This is often more efficient than
/// comparing full datasets and can focus inference on specific aspects of the data.
///
/// This function is equivalent to `abc_rejection` but operates on scalar summaries
/// instead of vector data, making it easier to use for simple cases.
///
/// # Arguments
///
/// * `rng` - Random number generator
/// * `model_fn` - Function that creates a model instance
/// * `simulator` - Function that computes a scalar summary from a trace
/// * `observed_summary` - Scalar summary of the observed data
/// * `tolerance` - Maximum allowed absolute difference for acceptance
/// * `max_samples` - Maximum number of samples to accept
///
/// # Returns
///
/// Vector of accepted traces that produced summaries within tolerance.
///
/// # Examples
///
/// ```rust
/// use fugue::*;
/// use rand::rngs::StdRng;
/// use rand::SeedableRng;
///
/// // ABC for estimating mean when we only observe sample mean
/// let observed_mean = 2.0;
/// let mut rng = StdRng::seed_from_u64(42);
///
/// let samples = abc_scalar_summary(
///     &mut rng,
///     || sample(addr!("mu"), Normal::new(0.0, 2.0).unwrap()),
///     |trace| {
///         // Extract mu parameter and return it as summary
///         if let Some(choice) = trace.choices.get(&addr!("mu")) {
///             if let ChoiceValue::F64(mu) = choice.value {
///                 mu // The summary statistic is just the parameter
///             } else { 0.0 }
///         } else { 0.0 }
///     },
///     observed_mean,
///     0.5, // tolerance (larger for easier acceptance)
///     5,   // max samples (small for test)
/// );
/// assert!(!samples.is_empty());
/// ```
pub fn abc_scalar_summary<A, R: Rng>(
    rng: &mut R,
    model_fn: impl Fn() -> Model<A>,
    simulator: impl Fn(&Trace) -> f64,
    observed_summary: f64,
    tolerance: f64,
    max_samples: usize,
) -> Vec<Trace> {
    abc_rejection(
        rng,
        model_fn,
        |trace| vec![simulator(trace)],
        &vec![observed_summary],
        &EuclideanDistance,
        tolerance,
        max_samples,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::addr;
    use crate::core::distribution::*;
    use crate::core::model::sample;

    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn distance_functions_work() {
        let eu = EuclideanDistance;
        let man = ManhattanDistance;
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.1, 2.1, 2.9];
        let d_eu = eu.distance(&a, &b);
        let d_man = man.distance(&a, &b);
        assert!(d_eu > 0.0);
        assert!(d_man > 0.0);
        // Euclidean should be <= Manhattan for same vectors
        assert!(d_eu <= d_man + 1e-12);
    }

    #[test]
    fn abc_scalar_summary_accepts_with_large_tolerance() {
        let mut rng = StdRng::seed_from_u64(42);
        let samples = abc_scalar_summary(
            &mut rng,
            || sample(addr!("mu"), Normal::new(0.0, 2.0).unwrap()),
            |trace| trace.get_f64(&addr!("mu")).unwrap_or(0.0),
            0.0,  // observed summary
            10.0, // large tolerance to ensure acceptance
            3,
        );
        assert!(!samples.is_empty());
    }

    #[test]
    fn abc_rejection_can_return_empty_with_tight_tolerance() {
        let mut rng = StdRng::seed_from_u64(43);
        let observed = vec![1000.0]; // far from prior mean 0
        let res = abc_rejection(
            &mut rng,
            || sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap()),
            |trace| vec![trace.get_f64(&addr!("mu")).unwrap_or(0.0)],
            &observed,
            &EuclideanDistance,
            1e-6, // extremely tight
            3,
        );
        assert!(res.is_empty());
    }

    #[test]
    fn abc_smc_respects_tolerance_schedule() {
        let mut rng = StdRng::seed_from_u64(44);
        let observed = vec![0.0];
        let config = ABCSMCConfig {
            initial_tolerance: 2.0,
            tolerance_schedule: vec![1.0, 0.5],
            particles_per_round: 4,
        };
        let res = abc_smc(
            &mut rng,
            || sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap()),
            |trace| vec![trace.get_f64(&addr!("mu")).unwrap_or(0.0)],
            &observed,
            &EuclideanDistance,
            config,
        );
        assert_eq!(res.len(), 4);
    }
}
