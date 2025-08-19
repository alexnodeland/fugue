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
//!     || sample(addr!("mu"), Normal { mu: 0.0, sigma: 2.0 }),
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

use crate::core::model::Model;
use crate::runtime::handler::run;
use crate::runtime::interpreters::PriorHandler;
use crate::runtime::trace::Trace;
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
        let median = if sorted.len() % 2 == 0 {
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
///     || sample(addr!("mu"), Normal { mu: 0.0, sigma: 2.0 }),
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
/// An advanced ABC method that uses Sequential Monte Carlo to iteratively
/// reduce the tolerance, leading to better approximations of the posterior.
/// SMC-ABC is more efficient than rejection ABC for stringent tolerances.
///
/// # Algorithm
///
/// 1. Start with initial tolerance and generate particles using rejection ABC
/// 2. For each subsequent tolerance level:
///    - Resample particles from the previous population
///    - Perturb parameters using MCMC moves
///    - Re-simulate and check new tolerance
/// 3. Final particles approximate the posterior at the strictest tolerance
///
/// # Arguments
///
/// * `rng` - Random number generator
/// * `model_fn` - Function that creates a model instance
/// * `simulator` - Function that simulates data given a trace
/// * `observed_data` - The observed data to match
/// * `distance_fn` - Distance function for comparing datasets
/// * `initial_tolerance` - Starting tolerance (should be relatively large)
/// * `tolerance_schedule` - Decreasing sequence of tolerances to use
/// * `particles_per_round` - Number of particles to maintain in each round
///
/// # Returns
///
/// Vector of traces from the final SMC population.
///
/// # Examples
///
/// ```rust
/// use fugue::*;
/// use rand::rngs::StdRng;
/// use rand::SeedableRng;
///
/// // Simple SMC-ABC example with small numbers for testing
/// let observed = vec![2.0];
/// let mut rng = StdRng::seed_from_u64(42);
///
/// let samples = abc_smc(
///     &mut rng,
///     || sample(addr!("mu"), Normal { mu: 0.0, sigma: 1.0 }),
///     |trace| {
///         if let Some(choice) = trace.choices.get(&addr!("mu")) {
///             if let ChoiceValue::F64(mu) = choice.value {
///                 vec![mu]
///             } else { vec![0.0] }
///         } else { vec![0.0] }
///     },
///     &observed,
///     &EuclideanDistance,
///     1.0,           // initial tolerance
///     &[0.5],        // tolerance schedule (single step)
///     5,             // particles per round (small for test)
/// );
/// assert!(!samples.is_empty());
/// ```
pub fn abc_smc<A, T, R: Rng>(
    rng: &mut R,
    model_fn: impl Fn() -> Model<A>,
    simulator: impl Fn(&Trace) -> T,
    observed_data: &T,
    distance_fn: &dyn DistanceFunction<T>,
    initial_tolerance: f64,
    tolerance_schedule: &[f64],
    particles_per_round: usize,
) -> Vec<Trace> {
    let mut current_particles;
    let mut current_tolerance = initial_tolerance;

    // Initial round: ABC rejection
    current_particles = abc_rejection(
        rng,
        &model_fn,
        &simulator,
        observed_data,
        distance_fn,
        current_tolerance,
        particles_per_round,
    );

    // Sequential rounds with decreasing tolerance
    for &new_tolerance in tolerance_schedule {
        if new_tolerance >= current_tolerance {
            continue; // Skip if tolerance doesn't decrease
        }

        let mut new_particles = Vec::new();

        while new_particles.len() < particles_per_round {
            // Sample a particle to perturb
            let base_idx = rng.gen_range(0..current_particles.len());
            let base_trace = &current_particles[base_idx];

            // Simple perturbation: resample one site
            let mut perturbed_trace = base_trace.clone();
            if !perturbed_trace.choices.is_empty() {
                let sites: Vec<_> = perturbed_trace.choices.keys().cloned().collect();
                let site_idx = rng.gen_range(0..sites.len());
                let selected_site = &sites[site_idx];

                // Resample this site from prior (simple perturbation)
                let (_a, fresh_trace) = run(
                    PriorHandler {
                        rng,
                        trace: Trace::default(),
                    },
                    model_fn(),
                );

                if let Some(fresh_choice) = fresh_trace.choices.get(selected_site) {
                    perturbed_trace
                        .choices
                        .insert(selected_site.clone(), fresh_choice.clone());
                }
            }

            // Check if perturbed trace meets new tolerance
            let simulated_data = simulator(&perturbed_trace);
            let dist = distance_fn.distance(observed_data, &simulated_data);

            if dist <= new_tolerance {
                new_particles.push(perturbed_trace);
            }
        }

        current_particles = new_particles;
        current_tolerance = new_tolerance;

        println!(
            "ABC SMC: tolerance = {:.4}, accepted = {}",
            current_tolerance,
            current_particles.len()
        );
    }

    current_particles
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
///     || sample(addr!("mu"), Normal { mu: 0.0, sigma: 2.0 }),
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
