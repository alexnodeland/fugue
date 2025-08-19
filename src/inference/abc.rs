//! Approximate Bayesian Computation (ABC) methods.
//!
//! Likelihood-free inference methods for models where the likelihood
//! is intractable but simulation is possible.

use crate::core::model::Model;
use crate::runtime::handler::run;
use crate::runtime::interpreters::PriorHandler;
use crate::runtime::trace::Trace;
use rand::Rng;

/// Distance function for comparing observed and simulated data.
pub trait DistanceFunction<T> {
    fn distance(&self, observed: &T, simulated: &T) -> f64;
}

/// Euclidean distance for vectors.
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

/// Manhattan distance for vectors.
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

/// ABC rejection sampling.
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

/// Sequential ABC with adaptive tolerance.
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

/// Simple ABC for scalar summary statistics.
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
