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
use crate::core::address::Address;
use crate::core::distribution::{Distribution, Normal};
use crate::core::model::Model;
use crate::core::numerical::log_sum_exp;
use crate::inference::mcmc_utils::DiminishingAdaptation;
use crate::runtime::handler::run;
use crate::runtime::interpreters::{PriorHandler, ScoreGivenTrace};
use crate::runtime::trace::{ChoiceValue, Trace};
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

/// Systematic resampling: return the resampled indices for a particle population.
pub fn systematic_resample<R: Rng>(rng: &mut R, particles: &[Particle]) -> Vec<usize> {
    systematic_indices(rng, &particle_weights(particles))
}

/// Stratified resampling: return the resampled indices for a particle population.
pub fn stratified_resample<R: Rng>(rng: &mut R, particles: &[Particle]) -> Vec<usize> {
    stratified_indices(rng, &particle_weights(particles))
}

/// Multinomial resampling: return the resampled indices for a particle population.
pub fn multinomial_resample<R: Rng>(rng: &mut R, particles: &[Particle]) -> Vec<usize> {
    multinomial_indices(rng, &particle_weights(particles))
}

fn particle_weights(particles: &[Particle]) -> Vec<f64> {
    particles.iter().map(|p| p.weight).collect()
}

/// Systematic resampling on a normalized weight vector.
fn systematic_indices<R: Rng>(rng: &mut R, weights: &[f64]) -> Vec<usize> {
    let n = weights.len();
    let mut indices = Vec::with_capacity(n);
    let u = rng.gen::<f64>() / n as f64;

    let mut cum_weight = 0.0;
    let mut i = 0;

    for j in 0..n {
        let threshold = u + j as f64 / n as f64;
        while cum_weight < threshold && i < n {
            cum_weight += weights[i];
            i += 1;
        }
        indices.push((i - 1).min(n - 1));
    }
    indices
}

/// Stratified resampling on a normalized weight vector.
fn stratified_indices<R: Rng>(rng: &mut R, weights: &[f64]) -> Vec<usize> {
    let n = weights.len();
    let mut indices = Vec::with_capacity(n);

    let mut cum_weight = 0.0;
    let mut i = 0;

    for j in 0..n {
        let u = rng.gen::<f64>();
        let threshold = (j as f64 + u) / n as f64;
        while cum_weight < threshold && i < n {
            cum_weight += weights[i];
            i += 1;
        }
        indices.push((i - 1).min(n - 1));
    }
    indices
}

/// Multinomial resampling on a normalized weight vector.
fn multinomial_indices<R: Rng>(rng: &mut R, weights: &[f64]) -> Vec<usize> {
    let n = weights.len();
    let mut indices = Vec::with_capacity(n);

    for _ in 0..n {
        let u = rng.gen::<f64>();
        let mut cum_weight = 0.0;
        let mut selected = n - 1;

        for (i, &w) in weights.iter().enumerate() {
            cum_weight += w;
            if u <= cum_weight {
                selected = i;
                break;
            }
        }
        indices.push(selected);
    }
    indices
}

/// Resample indices from a normalized weight vector using the chosen method.
fn resample_indices<R: Rng>(rng: &mut R, weights: &[f64], method: ResamplingMethod) -> Vec<usize> {
    match method {
        ResamplingMethod::Multinomial => multinomial_indices(rng, weights),
        ResamplingMethod::Systematic => systematic_indices(rng, weights),
        ResamplingMethod::Stratified => stratified_indices(rng, weights),
    }
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

/// Result of a likelihood-tempered Sequential Monte Carlo run.
///
/// In addition to the final weighted particle population, this carries the
/// unbiased log marginal-likelihood (log-evidence) estimate accumulated across
/// the tempering ladder — the key deliverable that motivates SMC over plain
/// MCMC for model comparison (see finding FG-58).
///
/// `SMCResult` dereferences to `Vec<Particle>`, so the population can be used
/// directly with slice/iterator methods and with [`effective_sample_size`].
#[derive(Clone, Debug)]
pub struct SMCResult {
    /// Final weighted particle population approximating the posterior (β = 1).
    pub particles: Vec<Particle>,
    /// Unbiased estimate of the log marginal likelihood log p(y).
    pub log_evidence: f64,
}

impl std::ops::Deref for SMCResult {
    type Target = Vec<Particle>;
    fn deref(&self) -> &Self::Target {
        &self.particles
    }
}

/// The log incremental target factor of a particle: log p(y | θ) = log_likelihood + log_factors.
///
/// Under likelihood tempering the sequence of targets is
/// π_β(θ) ∝ p(θ) · p(y | θ)^β, so the base prior draw contributes p(θ) and the
/// tempered reweighting uses only this likelihood term. This is also the correct
/// (prior-cancelled) importance weight of finding FG-03.
fn particle_log_likelihood(trace: &Trace) -> f64 {
    trace.log_likelihood + trace.log_factors
}

/// Run genuine likelihood-tempered Sequential Monte Carlo.
///
/// This targets the sequence of tempered distributions
/// π_β(θ) ∝ p(θ) · p(y | θ)^β for β increasing 0 → 1, so π_0 is the prior and
/// π_1 is the posterior. It performs:
///
/// 1. **Initialization** — draw `num_particles` particles from the prior (β = 0),
///    with uniform weights.
/// 2. **Adaptive tempering** — pick the next β by bisection so the reweighted ESS
///    hits `ess_threshold · N` (Jasra et al. 2011); reweight by the incremental
///    factor exp((β' − β)·log p(y | θ)).
/// 3. **Evidence accumulation** — add the log-mean incremental weight of each step
///    to an unbiased log-evidence accumulator (finding FG-58).
/// 4. **Resample + rejuvenate** — when `rejuvenation_steps > 0`, systematically
///    resample and apply π_β-invariant MH moves after each intermediate step to
///    restore particle diversity.
///
/// The terminal β = 1 step returns the *weighted* particles (no terminal
/// resample, per finding FG-43): resampling as the final operation would discard
/// information and inflate Monte Carlo variance.
///
/// # Arguments
///
/// * `rng` - Random number generator
/// * `num_particles` - Size of particle population to maintain
/// * `model_fn` - Function that creates the model
/// * `config` - SMC configuration (resampling method, ESS threshold, rejuvenation)
///
/// # Returns
///
/// An [`SMCResult`] with the final weighted particles and the log-evidence estimate.
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
/// let result = adaptive_smc(&mut rng, 5, model_fn, config);
/// assert!(result.log_evidence.is_finite());
///
/// // Analyze posterior
/// let mu_estimates: Vec<f64> = result.iter()
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
) -> SMCResult {
    let n = num_particles;
    if n == 0 {
        return SMCResult {
            particles: Vec::new(),
            log_evidence: 0.0,
        };
    }

    // Step 1: draw the initial population from the prior (β = 0, uniform weights).
    let mut particles = smc_prior_particles(rng, n, &model_fn);
    let mut logliks: Vec<f64> = particles
        .iter()
        .map(|p| particle_log_likelihood(&p.trace))
        .collect();
    // Normalized log-weights (invariant: sum of exp equals 1). Uniform at β = 0.
    let mut log_w = vec![-(n as f64).ln(); n];

    let mut beta = 0.0_f64;
    let mut log_evidence = 0.0_f64;
    // Target ESS for the adaptive β schedule.
    let target_ess = (config.ess_threshold * n as f64).clamp(1.0, n as f64);
    let mut adaptation = DiminishingAdaptation::new(0.44, 0.7);

    if config.rejuvenation_steps == 0 {
        // Without a rejuvenation move the particle positions never change, so a
        // multi-step temper and a single 0→1 jump give identical weighted
        // populations. Resampling here would only add variance (FG-43), so we do
        // a single pure importance-sampling reweight: log Ẑ = log-mean-likelihood
        // and weights ∝ exp(loglik). This is also the FG-03 prior-cancelled weight.
        let combined: Vec<f64> = logliks.iter().map(|ll| -(n as f64).ln() + ll).collect();
        log_evidence = log_sum_exp(&combined);
        beta = 1.0;
        log_w = combined;
    } else {
        // Genuine likelihood-tempered SMC. Because we resample (restart from
        // uniform weights) at every intermediate step, each `next_beta` search
        // begins from ESS = N > target and is guaranteed to make progress toward
        // β = 1. A hard cap on the number of steps is a final safety net.
        const MAX_STEPS: usize = 10_000;
        let mut steps = 0;
        while beta < 1.0 {
            steps += 1;
            let mut beta_new = next_beta(beta, &log_w, &logliks, target_ess);
            if steps >= MAX_STEPS {
                beta_new = 1.0;
            }
            let d_beta = beta_new - beta;

            // Reweight by the incremental likelihood factor and accumulate
            // evidence. Since `log_w` is uniform at the start of every step, this
            // step's contribution is the log-mean incremental weight (FG-58).
            let combined: Vec<f64> = log_w
                .iter()
                .zip(&logliks)
                .map(|(lw, ll)| lw + d_beta * ll)
                .collect();
            let log_norm = log_sum_exp(&combined);
            log_evidence += log_norm;

            if log_norm.is_finite() {
                for (lw, c) in log_w.iter_mut().zip(&combined) {
                    *lw = c - log_norm;
                }
            } else {
                for lw in log_w.iter_mut() {
                    *lw = -(n as f64).ln();
                }
            }
            beta = beta_new;

            // Resample + rejuvenate at intermediate steps only. The terminal
            // β = 1 step returns the weighted particles (no terminal resample,
            // FG-43).
            if beta < 1.0 {
                let weights: Vec<f64> = log_w.iter().map(|lw| lw.exp()).collect();
                let indices = resample_indices(rng, &weights, config.resampling_method);
                particles = indices.iter().map(|&i| particles[i].clone()).collect();
                for lw in log_w.iter_mut() {
                    *lw = -(n as f64).ln();
                }

                // π_β-invariant MH rejuvenation. Weights stay uniform (FG-13): an
                // invariant move does not change them, so we do NOT reweight here.
                for particle in particles.iter_mut() {
                    for _ in 0..config.rejuvenation_steps {
                        particle.trace = tempered_single_site_mh(
                            rng,
                            &model_fn,
                            &particle.trace,
                            beta,
                            &mut adaptation,
                        );
                    }
                }
                logliks = particles
                    .iter()
                    .map(|p| particle_log_likelihood(&p.trace))
                    .collect();
            }
        }
    }
    let _ = beta;

    // Attach the final normalized weights to the particles.
    let log_norm = log_sum_exp(&log_w);
    for (p, &lw) in particles.iter_mut().zip(&log_w) {
        if log_norm.is_finite() {
            let normalized = lw - log_norm;
            p.log_weight = normalized;
            p.weight = normalized.exp();
        } else {
            p.log_weight = -(n as f64).ln();
            p.weight = 1.0 / n as f64;
        }
    }

    SMCResult {
        particles,
        log_evidence,
    }
}

/// Choose the next inverse-temperature β' ∈ (β, 1] by ESS bisection.
///
/// Finds the smallest β' such that reweighting the current (normalized) weights
/// by exp((β' − β)·loglik) drops the ESS to `target_ess`. If reaching β' = 1
/// already keeps ESS ≥ `target_ess`, the ladder terminates at 1.
fn next_beta(beta: f64, log_w: &[f64], logliks: &[f64], target_ess: f64) -> f64 {
    let ess_at = |b: f64| -> f64 {
        let lv: Vec<f64> = log_w
            .iter()
            .zip(logliks)
            .map(|(lw, ll)| lw + (b - beta) * ll)
            .collect();
        let lse1 = log_sum_exp(&lv);
        let lv2: Vec<f64> = lv.iter().map(|x| 2.0 * x).collect();
        let lse2 = log_sum_exp(&lv2);
        if !lse1.is_finite() || !lse2.is_finite() {
            return log_w.len() as f64;
        }
        (2.0 * lse1 - lse2).exp()
    };

    // If a full jump to β = 1 keeps ESS above target, we are done.
    if ess_at(1.0) >= target_ess {
        return 1.0;
    }

    // Bisection: ess_at is decreasing in b; find the crossing with target_ess.
    let mut lo = beta;
    let mut hi = 1.0;
    for _ in 0..64 {
        let mid = 0.5 * (lo + hi);
        if ess_at(mid) < target_ess {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    // `hi` is on the low-ESS side, so ESS(hi) ≤ target. Guarantee strict progress.
    hi.max(beta + 1e-9).min(1.0)
}

/// A single π_β-invariant single-site Metropolis-Hastings rejuvenation move.
///
/// Perturbs one randomly chosen continuous (f64) site with a symmetric Gaussian
/// random walk and accepts against the tempered target π_β(θ) ∝ p(θ)·p(y|θ)^β.
/// Because the proposal is symmetric there is no Hastings correction. The move is
/// invariant for π_β, so applying it to a resampled (uniform-weight) population
/// leaves the weights uniform.
fn tempered_single_site_mh<A, R: Rng>(
    rng: &mut R,
    model_fn: &impl Fn() -> Model<A>,
    current: &Trace,
    beta: f64,
    adaptation: &mut DiminishingAdaptation,
) -> Trace {
    // Collect continuous sites eligible for a Gaussian random-walk perturbation.
    let f64_sites: Vec<Address> = current
        .choices
        .iter()
        .filter(|(_, c)| matches!(c.value, ChoiceValue::F64(_)))
        .map(|(a, _)| a.clone())
        .collect();
    if f64_sites.is_empty() {
        // Nothing to move; doing nothing is trivially π_β-invariant.
        return current.clone();
    }

    let site = f64_sites[rng.gen_range(0..f64_sites.len())].clone();
    let scale = adaptation.get_scale(&site);
    let cur_val = current.choices[&site].value.as_f64().unwrap();

    // Symmetric Gaussian random walk on the selected coordinate.
    let z = Normal::new(0.0, 1.0).unwrap().sample(rng);
    let prop_val = cur_val + scale * z;

    let mut proposed = current.clone();
    proposed.choices.get_mut(&site).unwrap().value = ChoiceValue::F64(prop_val);

    // Score current and proposed traces under the model.
    let (_, cur_scored) = run(
        ScoreGivenTrace {
            base: current.clone(),
            trace: Trace::default(),
        },
        model_fn(),
    );
    let (_, prop_scored) = run(
        ScoreGivenTrace {
            base: proposed,
            trace: Trace::default(),
        },
        model_fn(),
    );

    // Tempered acceptance: Δlog_prior + β·Δloglik (symmetric proposal ⇒ no Hastings).
    let log_alpha = (prop_scored.log_prior - cur_scored.log_prior)
        + beta * (particle_log_likelihood(&prop_scored) - particle_log_likelihood(&cur_scored));
    let accept = log_alpha >= 0.0 || rng.gen::<f64>() < log_alpha.exp();
    adaptation.update(&site, accept);

    if accept {
        prop_scored
    } else {
        cur_scored
    }
}

/// Apply π_β-invariant MH rejuvenation moves to a particle population in place.
///
/// This is the rejuvenation primitive used by [`adaptive_smc`]. It updates each
/// particle's trace with `rejuvenation_steps` single-site MH moves that leave the
/// tempered target π_β invariant. Crucially it does **not** touch particle weights:
/// after resampling the weights are uniform, and an invariant MH move keeps them
/// uniform — reweighting here would re-introduce the prior-squaring bias of
/// findings FG-03/FG-13.
pub fn rejuvenate_particles<A, R: Rng>(
    rng: &mut R,
    particles: &mut [Particle],
    model_fn: impl Fn() -> Model<A>,
    beta: f64,
    rejuvenation_steps: usize,
) {
    let mut adaptation = DiminishingAdaptation::new(0.44, 0.7);
    for particle in particles.iter_mut() {
        for _ in 0..rejuvenation_steps {
            particle.trace =
                tempered_single_site_mh(rng, &model_fn, &particle.trace, beta, &mut adaptation);
        }
        // FG-13: weights are intentionally left unchanged.
    }
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

/// Draw an importance-weighted particle population from the prior.
///
/// Each particle is a full model execution sampled from the prior (β = 0). Its
/// unnormalized log-weight is the log-likelihood only — `log_likelihood +
/// log_factors` — because the proposal (the prior) exactly cancels the prior
/// factor of the target: with q(θ) = p(θ) and target ∝ p(θ)·p(y|θ), the
/// self-normalized importance weight is p(y|θ), not p(θ)·p(y|θ). Including the
/// log-prior term double-counts (squares) the prior and biases every posterior
/// estimate — this is finding FG-03.
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
        // FG-03: prior-proposed weight is the likelihood factor only (the prior
        // cancels against the proposal). FG-59: compute the weight from a borrow,
        // then move `t` into the particle instead of cloning the whole trace.
        let log_weight = particle_log_likelihood(&t);
        particles.push(Particle {
            trace: t,
            weight: 0.0, // Will be set by normalization
            log_weight,
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
            Particle {
                trace: Trace::default(),
                weight: 0.7,
                log_weight: (0.7f64).ln(),
            },
            Particle {
                trace: Trace::default(),
                weight: 0.2,
                log_weight: (0.2f64).ln(),
            },
            Particle {
                trace: Trace::default(),
                weight: 0.09,
                log_weight: (0.09f64).ln(),
            },
            Particle {
                trace: Trace::default(),
                weight: 0.01,
                log_weight: (0.01f64).ln(),
            },
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
        for p in &resampled {
            assert!((p.weight - 0.25).abs() < 1e-12);
        }
    }

    #[test]
    fn normalize_particles_handles_neg_inf() {
        let mut particles = vec![
            Particle {
                trace: Trace::default(),
                weight: 0.0,
                log_weight: f64::NEG_INFINITY,
            },
            Particle {
                trace: Trace::default(),
                weight: 0.0,
                log_weight: f64::NEG_INFINITY,
            },
        ];
        normalize_particles(&mut particles);
        // Fallback to uniform
        assert!((particles[0].weight - 0.5).abs() < 1e-12);
        assert!((particles[1].weight - 0.5).abs() < 1e-12);
    }

    #[test]
    fn adaptive_smc_runs_with_small_config() {
        let model_fn = || {
            sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap()).and_then(|mu| {
                observe(addr!("y"), Normal::new(mu, 1.0).unwrap(), 0.5).map(move |_| mu)
            })
        };
        let mut rng = StdRng::seed_from_u64(2);
        let config = SMCConfig {
            resampling_method: ResamplingMethod::Systematic,
            ess_threshold: 0.5,
            rejuvenation_steps: 1,
        };
        let particles = adaptive_smc(&mut rng, 5, model_fn, config);
        assert_eq!(particles.len(), 5);
        // Weights normalized
        let sum_w: f64 = particles.iter().map(|p| p.weight).sum();
        assert!((sum_w - 1.0).abs() < 1e-9);
    }
}
