//! Sequential Monte Carlo (SMC) with particle filtering and resampling.
//!
//! This module implements Sequential Monte Carlo methods, also known as particle filters.
//! SMC maintains a population of weighted particles (traces) and uses resampling to
//! focus computational effort on high-probability regions of the posterior.
//!
//! ## Key Features
//!
//! - **Multiple resampling methods**: Multinomial, Systematic, Stratified
//! - **Effective Sample Size (ESS) monitoring**: Automatic resampling triggers
//! - **Rejuvenation**: Optional MCMC moves to maintain particle diversity
//! - **Adaptive resampling**: Resample only when ESS drops below threshold
//!
//! ## Algorithm Overview
//!
//! SMC works by maintaining a population of particles, each representing a possible
//! state (parameter configuration) with an associated weight:
//!
//! 1. **Initialize**: Start with particles from the prior
//! 2. **Weight**: Compute importance weights based on likelihood
//! 3. **Resample**: When weights become uneven, resample to maintain diversity
//! 4. **Rejuvenate**: Optionally apply MCMC moves to particles
//! 5. **Repeat**: Continue until convergence or max iterations
//!
//! ## When to Use SMC
//!
//! SMC is particularly effective for:
//! - Models with many observations that can be processed sequentially
//! - High-dimensional parameter spaces where MCMC mixes poorly
//! - Real-time inference where new data arrives continuously
//! - Situations where you need multiple diverse posterior samples
//!
//! # Examples
//!
//! ```rust
//! use fugue::*;
//! use rand::rngs::StdRng;
//! use rand::SeedableRng;
//!
//! // Define a simple model
//! let model_fn = || {
//!     sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap())
//!         .bind(|mu| {
//!             observe(addr!("y"), Normal::new(mu, 0.5).unwrap(), 2.0)
//!                 .map(move |_| mu)
//!         })
//! };
//!
//! // Run SMC (small numbers for testing)
//! let mut rng = StdRng::seed_from_u64(42);
//! let config = SMCConfig::default();
//! let particles = adaptive_smc(&mut rng, 10, model_fn, config);
//!
//! // Analyze results
//! let ess = effective_sample_size(&particles);
//! assert!(ess > 0.0);
//! ```
use crate::core::model::Model;
use crate::inference::mcmc_utils::DiminishingAdaptation;
use crate::inference::mh::adaptive_single_site_mh;
use crate::runtime::handler::run;
use crate::runtime::interpreters::PriorHandler;
use crate::runtime::trace::Trace;
use rand::Rng;

/// A weighted particle in the SMC population.
///
/// Each particle represents a possible state (parameter configuration) with
/// associated weights that reflect its probability relative to other particles.
/// The weight decomposition into linear and log space enables numerical stability.
///
/// # Fields
///
/// * `trace` - Execution trace containing parameter values and log-probabilities
/// * `weight` - Normalized linear weight (used for resampling)
/// * `log_weight` - Log-space weight (for numerical stability)
///
/// # Examples
///
/// ```rust
/// use fugue::*;
///
/// // Particles are typically created by SMC algorithms
/// let particle = Particle {
///     trace: Trace::default(),
///     weight: 0.25,           // 25% of total weight
///     log_weight: -1.386,     // ln(0.25)
/// };
///
/// println!("Particle weight: {:.3}", particle.weight);
/// ```
#[derive(Clone, Debug)]
pub struct Particle {
    /// Execution trace containing parameter values and log-probabilities.
    pub trace: Trace,
    /// Normalized linear weight (used for resampling).
    pub weight: f64,
    /// Log-space weight (for numerical stability).
    pub log_weight: f64,
}

/// Resampling algorithms for particle filters.
///
/// Different resampling methods offer trade-offs between computational efficiency,
/// variance reduction, and implementation complexity. All methods aim to replace
/// low-weight particles with copies of high-weight particles.
///
/// # Variants
///
/// * `Multinomial` - Simple multinomial resampling (high variance)
/// * `Systematic` - Low-variance systematic resampling (recommended)
/// * `Stratified` - Stratified resampling (balanced variance/complexity)
///
/// # Examples
///
/// ```rust
/// use fugue::*;
///
/// // Configure SMC with different resampling methods
/// let config_systematic = SMCConfig {
///     resampling_method: ResamplingMethod::Systematic,
///     ..Default::default()
/// };
///
/// let config_multinomial = SMCConfig {
///     resampling_method: ResamplingMethod::Multinomial,
///     ..Default::default()
/// };
/// ```
#[derive(Clone, Copy, Debug)]
pub enum ResamplingMethod {
    /// Simple multinomial resampling with replacement.
    Multinomial,
    /// Low-variance systematic resampling (recommended).
    Systematic,
    /// Stratified resampling with balanced variance.
    Stratified,
}

/// Configuration options for Sequential Monte Carlo.
///
/// This struct controls various aspects of the SMC algorithm, allowing fine-tuning
/// of performance and accuracy trade-offs.
///
/// # Fields
///
/// * `resampling_method` - Algorithm used for particle resampling
/// * `ess_threshold` - ESS threshold that triggers resampling (as fraction of N)
/// * `rejuvenation_steps` - Number of MCMC moves after resampling to increase diversity
///
/// # Examples
///
/// ```rust
/// use fugue::*;
///
/// // Conservative configuration (less resampling, more rejuvenation)
/// let conservative_config = SMCConfig {
///     resampling_method: ResamplingMethod::Systematic,
///     ess_threshold: 0.2,  // Resample when ESS < 20% of particles
///     rejuvenation_steps: 5, // 5 MCMC moves after resampling
/// };
///
/// // Aggressive configuration (frequent resampling, no rejuvenation)
/// let aggressive_config = SMCConfig {
///     resampling_method: ResamplingMethod::Systematic,
///     ess_threshold: 0.8,  // Resample when ESS < 80% of particles
///     rejuvenation_steps: 0, // No rejuvenation
/// };
/// ```
pub struct SMCConfig {
    /// Algorithm used for particle resampling.
    pub resampling_method: ResamplingMethod,
    /// ESS threshold that triggers resampling (as fraction of particle count).
    pub ess_threshold: f64,
    /// Number of MCMC moves after resampling to increase diversity.
    pub rejuvenation_steps: usize,
}

impl Default for SMCConfig {
    fn default() -> Self {
        Self {
            resampling_method: ResamplingMethod::Systematic,
            ess_threshold: 0.5,
            rejuvenation_steps: 0,
        }
    }
}

/// Compute the effective sample size (ESS) of a particle population.
///
/// ESS measures how many "effective" independent samples the weighted particle
/// population represents. It ranges from 1 (all weight on one particle) to N
/// (uniform weights). Low ESS indicates weight degeneracy and triggers resampling.
///
/// **Formula:** ESS = 1 / Σᵢ(wᵢ²) where wᵢ are normalized weights.
///
/// # Arguments
///
/// * `particles` - Population of weighted particles
///
/// # Returns
///
/// Effective sample size (1.0 ≤ ESS ≤ N where N = particles.len()).
///
/// # Examples
///
/// ```rust
/// use fugue::*;
///
/// // Uniform weights -> high ESS
/// let uniform_particles = vec![
///     Particle { trace: Trace::default(), weight: 0.25, log_weight: -1.386 },
///     Particle { trace: Trace::default(), weight: 0.25, log_weight: -1.386 },
///     Particle { trace: Trace::default(), weight: 0.25, log_weight: -1.386 },
///     Particle { trace: Trace::default(), weight: 0.25, log_weight: -1.386 },
/// ];
/// let ess = effective_sample_size(&uniform_particles);
/// assert!((ess - 4.0).abs() < 0.01); // ESS ≈ 4 (perfect)
///
/// // Degenerate weights -> low ESS
/// let degenerate_particles = vec![
///     Particle { trace: Trace::default(), weight: 0.99, log_weight: -0.01 },
///     Particle { trace: Trace::default(), weight: 0.01, log_weight: -4.605 },
/// ];
/// let ess = effective_sample_size(&degenerate_particles);
/// assert!(ess < 1.1); // ESS ≈ 1 (very poor)
/// ```
pub fn effective_sample_size(particles: &[Particle]) -> f64 {
    let sum_sq: f64 = particles.iter().map(|p| p.weight * p.weight).sum();
    1.0 / sum_sq
}

/// Systematic resampling.
pub fn systematic_resample<R: Rng>(rng: &mut R, particles: &[Particle]) -> Vec<usize> {
    let n = particles.len();
    let mut indices = Vec::with_capacity(n);
    let u = rng.gen::<f64>() / n as f64;

    let mut cum_weight = 0.0;
    let mut i = 0;

    for j in 0..n {
        let threshold = u + j as f64 / n as f64;
        while cum_weight < threshold && i < n {
            cum_weight += particles[i].weight;
            i += 1;
        }
        indices.push((i - 1).min(n - 1));
    }
    indices
}

/// Stratified resampling.
pub fn stratified_resample<R: Rng>(rng: &mut R, particles: &[Particle]) -> Vec<usize> {
    let n = particles.len();
    let mut indices = Vec::with_capacity(n);

    let mut cum_weight = 0.0;
    let mut i = 0;

    for j in 0..n {
        let u = rng.gen::<f64>();
        let threshold = (j as f64 + u) / n as f64;
        while cum_weight < threshold && i < n {
            cum_weight += particles[i].weight;
            i += 1;
        }
        indices.push((i - 1).min(n - 1));
    }
    indices
}

/// Multinomial resampling.
pub fn multinomial_resample<R: Rng>(rng: &mut R, particles: &[Particle]) -> Vec<usize> {
    let n = particles.len();
    let mut indices = Vec::with_capacity(n);

    for _ in 0..n {
        let u = rng.gen::<f64>();
        let mut cum_weight = 0.0;
        let mut selected = n - 1;

        for (i, p) in particles.iter().enumerate() {
            cum_weight += p.weight;
            if u <= cum_weight {
                selected = i;
                break;
            }
        }
        indices.push(selected);
    }
    indices
}

/// Resample particles based on weights.
pub fn resample_particles<R: Rng>(
    rng: &mut R,
    particles: &[Particle],
    method: ResamplingMethod,
) -> Vec<Particle> {
    let indices = match method {
        ResamplingMethod::Multinomial => multinomial_resample(rng, particles),
        ResamplingMethod::Systematic => systematic_resample(rng, particles),
        ResamplingMethod::Stratified => stratified_resample(rng, particles),
    };

    let n = particles.len();
    let uniform_weight = 1.0 / n as f64;

    indices
        .into_iter()
        .map(|i| {
            let mut p = particles[i].clone();
            p.weight = uniform_weight;
            p.log_weight = uniform_weight.ln();
            p
        })
        .collect()
}

/// Run adaptive Sequential Monte Carlo with resampling and rejuvenation.
///
/// This is the main SMC algorithm that maintains a population of weighted particles
/// and adaptively resamples when the effective sample size drops below a threshold.
/// Optional rejuvenation steps help maintain particle diversity after resampling.
///
/// # Algorithm
///
/// 1. Initialize particles by sampling from the prior
/// 2. Compute weights and effective sample size
/// 3. If ESS < threshold × N: resample particles
/// 4. Apply rejuvenation moves (MCMC) if configured
/// 5. Return final particle population
///
/// # Arguments
///
/// * `rng` - Random number generator
/// * `num_particles` - Size of particle population to maintain
/// * `model_fn` - Function that creates the model
/// * `config` - SMC configuration (resampling method, thresholds, etc.)
///
/// # Returns
///
/// Final population of weighted particles representing the posterior.
///
/// # Examples
///
/// ```rust
/// use fugue::*;
/// use rand::rngs::StdRng;
/// use rand::SeedableRng;
///
/// // Simple model for testing
/// let model_fn = || {
///     sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap())
///         .bind(|mu| {
///             observe(addr!("y"), Normal::new(mu, 0.5).unwrap(), 1.8)
///                 .map(move |_| mu)
///         })
/// };
///
/// // Run SMC with small numbers for testing
/// let mut rng = StdRng::seed_from_u64(42);
/// let config = SMCConfig {
///     resampling_method: ResamplingMethod::Systematic,
///     ess_threshold: 0.5,
///     rejuvenation_steps: 1,
/// };
///
/// let particles = adaptive_smc(&mut rng, 5, model_fn, config);
///
/// // Analyze posterior
/// let mu_estimates: Vec<f64> = particles.iter()
///     .filter_map(|p| p.trace.choices.get(&addr!("mu")))
///     .filter_map(|choice| match choice.value {
///         ChoiceValue::F64(mu) => Some(mu),
///         _ => None,
///     })
///     .collect();
///
/// assert!(!mu_estimates.is_empty());
/// ```
pub fn adaptive_smc<A, R: Rng>(
    rng: &mut R,
    num_particles: usize,
    model_fn: impl Fn() -> Model<A>,
    config: SMCConfig,
) -> Vec<Particle> {
    let mut particles = smc_prior_particles(rng, num_particles, &model_fn);

    // Check if resampling is needed
    let ess = effective_sample_size(&particles);
    let ess_ratio = ess / num_particles as f64;

    if ess_ratio < config.ess_threshold {
        // Resample
        particles = resample_particles(rng, &particles, config.resampling_method);

        // Optional rejuvenation with MCMC
        if config.rejuvenation_steps > 0 {
            let mut adaptation = DiminishingAdaptation::new(0.44, 0.7);
            for particle in &mut particles {
                for _ in 0..config.rejuvenation_steps {
                    let (_, new_trace) =
                        adaptive_single_site_mh(rng, &model_fn, &particle.trace, &mut adaptation);
                    particle.trace = new_trace;
                    particle.log_weight = particle.trace.total_log_weight();
                }
            }

            // Renormalize after rejuvenation
            normalize_particles(&mut particles);
        }
    }

    particles
}

/// Normalize particle weights using numerically stable log-sum-exp.
///
/// This function properly handles extreme log-weights without underflow or overflow,
/// which is critical for reliable SMC performance.
pub fn normalize_particles(particles: &mut [Particle]) {
    use crate::core::numerical::log_sum_exp;

    if particles.is_empty() {
        return;
    }

    // Collect log weights
    let log_weights: Vec<f64> = particles.iter().map(|p| p.log_weight).collect();

    // Compute log normalizing constant stably
    let log_norm = log_sum_exp(&log_weights);

    // Handle degenerate case where all weights are -∞
    if log_norm.is_infinite() && log_norm < 0.0 {
        let n = particles.len();
        for p in particles {
            p.weight = 1.0 / n as f64; // Uniform weights as fallback
        }
        return;
    }

    // Normalize weights stably
    for (p, &log_w) in particles.iter_mut().zip(&log_weights) {
        p.weight = (log_w - log_norm).exp();
    }

    // Ensure weights sum to 1.0 (handle small numerical errors)
    let weight_sum: f64 = particles.iter().map(|p| p.weight).sum();
    if weight_sum > 0.0 {
        for p in particles {
            p.weight /= weight_sum;
        }
    }
}

pub fn smc_prior_particles<A, R: Rng>(
    rng: &mut R,
    num_particles: usize,
    model_fn: impl Fn() -> Model<A>,
) -> Vec<Particle> {
    let mut particles = Vec::with_capacity(num_particles);
    for _ in 0..num_particles {
        let (_a, t) = run(
            PriorHandler {
                rng,
                trace: Trace::default(),
            },
            model_fn(),
        );
        particles.push(Particle {
            trace: t.clone(),
            weight: 0.0, // Will be set by normalization
            log_weight: t.total_log_weight(),
        });
    }
    normalize_particles(&mut particles);
    particles
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
    fn ess_and_resampling_behave() {
        // Construct 4 particles with uneven weights
        let particles = vec![
            Particle { trace: Trace::default(), weight: 0.7, log_weight: (0.7f64).ln() },
            Particle { trace: Trace::default(), weight: 0.2, log_weight: (0.2f64).ln() },
            Particle { trace: Trace::default(), weight: 0.09, log_weight: (0.09f64).ln() },
            Particle { trace: Trace::default(), weight: 0.01, log_weight: (0.01f64).ln() },
        ];
        let ess_val = effective_sample_size(&particles);
        assert!(ess_val < particles.len() as f64);

        // Resampling indices should be valid and length preserved
        let mut rng = StdRng::seed_from_u64(1);
        let idx_m = multinomial_resample(&mut rng, &particles);
        assert_eq!(idx_m.len(), particles.len());

        let idx_s = systematic_resample(&mut rng, &particles);
        assert_eq!(idx_s.len(), particles.len());

        let idx_t = stratified_resample(&mut rng, &particles);
        assert_eq!(idx_t.len(), particles.len());

        // Resample and check normalized uniform weights
        let resampled = resample_particles(&mut rng, &particles, ResamplingMethod::Systematic);
        let sum_w: f64 = resampled.iter().map(|p| p.weight).sum();
        assert!((sum_w - 1.0).abs() < 1e-12);
        for p in &resampled { assert!((p.weight - 0.25).abs() < 1e-12); }
    }

    #[test]
    fn normalize_particles_handles_neg_inf() {
        let mut particles = vec![
            Particle { trace: Trace::default(), weight: 0.0, log_weight: f64::NEG_INFINITY },
            Particle { trace: Trace::default(), weight: 0.0, log_weight: f64::NEG_INFINITY },
        ];
        normalize_particles(&mut particles);
        // Fallback to uniform
        assert!((particles[0].weight - 0.5).abs() < 1e-12);
        assert!((particles[1].weight - 0.5).abs() < 1e-12);
    }

    #[test]
    fn adaptive_smc_runs_with_small_config() {
        let model_fn = || {
            sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap())
                .and_then(|mu| observe(addr!("y"), Normal::new(mu, 1.0).unwrap(), 0.5).map(move |_| mu))
        };
        let mut rng = StdRng::seed_from_u64(2);
        let config = SMCConfig { resampling_method: ResamplingMethod::Systematic, ess_threshold: 0.5, rejuvenation_steps: 1 };
        let particles = adaptive_smc(&mut rng, 5, model_fn, config);
        assert_eq!(particles.len(), 5);
        // Weights normalized
        let sum_w: f64 = particles.iter().map(|p| p.weight).sum();
        assert!((sum_w - 1.0).abs() < 1e-9);
    }
}

