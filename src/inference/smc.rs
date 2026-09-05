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
use crate::core::model::Model;
use crate::core::numerical::log_sum_exp;
use crate::inference::mcmc_utils::DiminishingAdaptation;
use crate::inference::mh::{mh_accept, propose_and_score, SiteProposal};
use crate::runtime::handler::run;
use crate::runtime::interpreters::{PriorHandler, ScoreGivenTrace};
use crate::runtime::trace::Trace;
use rand::Rng;
use std::collections::HashMap;

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

/// A population-coupled MCMC move applied to the whole particle slice between
/// SMC tempering steps.
///
/// Unlike per-particle rejuvenation ([`rejuvenate_particles`]), a population
/// kernel may *couple* particles — e.g. a crossover move that swaps a block of
/// choices between two parent traces. It is invoked by
/// [`adaptive_smc_with_kernel`] immediately after resampling and per-particle
/// rejuvenation, on a uniform-weight population that is (approximately)
/// distributed according to the current tempered target π_β.
///
/// # Invariance contract (MUST hold, or SMC estimates are biased)
///
/// For a single model execution, `π_β(θ) ∝ p(θ) · p(y|θ)^β`. A kernel that
/// couples the pair (i, j) MUST leave the **product target**
/// `π_β(θ_i) · π_β(θ_j)` invariant (and analogously for any k-tuple it
/// couples). Concretely the implementation MUST:
///
/// * **(W)** never write `particle.weight` or `particle.log_weight` — after
///   resampling the weights are uniform and an invariant move keeps them
///   uniform (findings FG-03 / FG-13). Reweighting here re-introduces the
///   prior-squaring bias FG-03 fixed.
/// * **(T)** mutate only `particle.trace`, and only to a value obtained by a
///   Metropolis accept/reject whose target is the product of the coupled
///   particles' tempered densities.
/// * **(S)** re-score every trace it writes under `model_fn` (via
///   [`ScoreGivenTrace`] or
///   [`score_given_trace_reconciled`](crate::runtime::interpreters::score_given_trace_reconciled))
///   so the three log accumulators are valid — direct choice surgery does NOT
///   update them (see [`Trace::insert_choice`]).
/// * **(E)** not read or mutate the SMC log-evidence accumulator (it has no
///   access to it) — an invariant move contributes no incremental weight
///   (FG-58).
///
/// # Correctness of the built-in crossover move
///
/// Between tempering steps the uniform-weight population is approximately
/// i.i.d. from π_β, so a coupled pair (θ_i, θ_j) is distributed as
/// `π_β ⊗ π_β`. [`CrossoverKernel`] draws an address mask S from a
/// value-independent, pair-symmetric distribution and deterministically swaps
/// the values on S. This map is an **involution** (swapping S back recovers
/// the parents) with identical forward/reverse mask distributions, so
/// `q(child|parent) = q(parent|child)` and the Hastings correction is 1. The
/// Metropolis acceptance
/// `α = min(1, [π_β(θ_i')·π_β(θ_j')] / [π_β(θ_i)·π_β(θ_j)])` is therefore a
/// valid Metropolis move on `π_β ⊗ π_β`, hence product-invariant; applied to a
/// π_β population it preserves each particle's marginal and keeps the uniform
/// weights correct (W). Being invariant it injects zero incremental weight, so
/// the log-evidence accumulator is untouched (E). The re-score (S) makes
/// off-support swaps (`guard` / `factor(-∞)`) reject via a `-∞` density —
/// support-respecting truncation.
///
/// The kernel is object-safe: `rng` is `&mut dyn RngCore` (rand's blanket
/// `impl<R: RngCore + ?Sized> Rng for R` supplies the sampling methods), and
/// the model constructor is passed as `&dyn Fn`.
pub trait PopulationKernel<A> {
    /// Apply one population sweep in place. `beta` is the current tempering
    /// exponent; `model_fn` reconstructs the single-execution model whose
    /// tempered density defines the (product) target.
    ///
    /// The particles handed in carry **uniform** weights (`weight = 1/n`,
    /// `log_weight = -ln n`, FG-N7): every sweep runs right after a resample.
    /// Contract (W) asks the kernel to leave them so.
    fn sweep(
        &mut self,
        rng: &mut dyn rand::RngCore,
        particles: &mut [Particle],
        model_fn: &dyn Fn() -> Model<A>,
        beta: f64,
    );

    /// `true` iff `sweep` is a no-op, so the driver may skip the tempering
    /// ladder entirely when there is also no per-particle rejuvenation (the
    /// FG-43 single-reweight shortcut). Defaults to `false`; only
    /// [`NoKernel`] overrides it. A kernel that moves particles must not
    /// return `true`: with `rejuvenation_steps == 0` the shortcut would
    /// otherwise silently skip it (FG-N3).
    fn is_identity(&self) -> bool {
        false
    }
}

/// The identity population kernel: does nothing. [`adaptive_smc`] is defined
/// as `adaptive_smc_with_kernel(.., &mut NoKernel)`.
pub struct NoKernel;

impl<A> PopulationKernel<A> for NoKernel {
    fn sweep(
        &mut self,
        _: &mut dyn rand::RngCore,
        _: &mut [Particle],
        _: &dyn Fn() -> Model<A>,
        _: f64,
    ) {
    }

    fn is_identity(&self) -> bool {
        true
    }
}

/// A population crossover kernel: repeatedly picks two distinct particles at
/// random, proposes a child pair by swapping the block of choices at the
/// addresses chosen by `mask`, and accepts the swap with the product-target
/// Metropolis ratio (see the correctness argument on [`PopulationKernel`]).
///
/// # v1 scope: fixed-structure models
///
/// The swap is exact for models whose address set is identical across
/// executions (bit-string / permutation / real-vector genomes and other
/// fixed-structure models): the swap is dimension-preserving and the
/// [`ScoreGivenTrace`] re-score is exact. For variable-dimension models a
/// swapped block can leave a partner with an incomplete assignment; such
/// crossovers need a custom kernel built on
/// [`score_given_trace_reconciled`](crate::runtime::interpreters::score_given_trace_reconciled)
/// carrying the RJMCMC dimension bookkeeping.
pub struct CrossoverKernel {
    /// Number of (pair, swap) proposals per sweep.
    pub n_pairs: usize,
    /// Chooses the address set S to swap. Given the two parent traces it
    /// returns the addresses whose values are exchanged between the pair. MUST
    /// be value-independent (depend only on the address structure) and
    /// symmetric in its two trace arguments for the move to be a symmetric
    /// involution (Hastings ratio 1).
    #[allow(clippy::type_complexity)]
    pub mask: Box<dyn Fn(&Trace, &Trace, &mut dyn rand::RngCore) -> Vec<Address>>,
}

/// Build two children by exchanging the choices at `swap` between `a` and `b`.
/// Pure choice surgery: the children's accumulators are NOT valid until
/// re-scored (contract S of [`PopulationKernel`]).
fn swap_block(a: &Trace, b: &Trace, swap: &[Address]) -> (Trace, Trace) {
    let mut ca = a.clone();
    let mut cb = b.clone();
    for addr in swap {
        let from_a = ca.choices.remove(addr);
        let from_b = cb.choices.remove(addr);
        if let Some(c) = from_b {
            ca.choices.insert(addr.clone(), c);
        }
        if let Some(c) = from_a {
            cb.choices.insert(addr.clone(), c);
        }
    }
    (ca, cb)
}

impl<A> PopulationKernel<A> for CrossoverKernel {
    fn sweep(
        &mut self,
        rng: &mut dyn rand::RngCore,
        particles: &mut [Particle],
        model_fn: &dyn Fn() -> Model<A>,
        beta: f64,
    ) {
        let n = particles.len();
        if n < 2 {
            return;
        }
        for _ in 0..self.n_pairs {
            let i = rng.gen_range(0..n);
            let mut j = rng.gen_range(0..n - 1);
            if j >= i {
                j += 1; // distinct partner
            }
            let s = (self.mask)(&particles[i].trace, &particles[j].trace, rng);
            if s.is_empty() {
                continue;
            }

            // Build child traces by choice surgery, then RE-SCORE (contract S).
            let (ti, tj) = swap_block(&particles[i].trace, &particles[j].trace, &s);
            let (_ai, ci) = run(
                ScoreGivenTrace {
                    base: ti,
                    trace: Trace::default(),
                },
                model_fn(),
            );
            let (_aj, cj) = run(
                ScoreGivenTrace {
                    base: tj,
                    trace: Trace::default(),
                },
                model_fn(),
            );

            // Tempered log-density of a single execution:
            //   log π_β(θ) = log_prior + β·(log_likelihood + log_factors).
            let logd = |t: &Trace| t.log_prior + beta * (t.log_likelihood + t.log_factors);
            let log_alpha =
                (logd(&ci) + logd(&cj)) - (logd(&particles[i].trace) + logd(&particles[j].trace));
            if log_alpha >= 0.0 || rng.gen::<f64>() < log_alpha.exp() {
                particles[i].trace = ci; // (T): only traces move;
                particles[j].trace = cj; // (W): weights untouched.
            }
        }
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
    adaptive_smc_with_kernel(rng, num_particles, model_fn, config, &mut NoKernel)
}

/// Likelihood-tempered SMC with a population-coupled kernel applied between
/// tempering steps.
///
/// Identical to [`adaptive_smc`] except that after each intermediate step's
/// resample + per-particle rejuvenation (and before the likelihood refresh),
/// `kernel.sweep(..)` is invoked on the whole particle slice at the current β.
/// This is the hook for population-coupled MCMC moves — e.g. crossover between
/// particle pairs ([`CrossoverKernel`]) — which per-particle rejuvenation
/// cannot express. With [`NoKernel`] this is exactly [`adaptive_smc`].
///
/// The kernel runs only at **intermediate** tempering steps: the terminal
/// β = 1 step returns the weighted particles without resampling or moves
/// (FG-43), so the kernel never touches the returned weighted population. See
/// [`PopulationKernel`] for the invariance contract the kernel must satisfy.
///
/// The FG-43 shortcut — a single prior-importance reweight with no tempering
/// ladder — is taken only when there is **nothing that moves particles**:
/// `rejuvenation_steps == 0` *and* [`PopulationKernel::is_identity`]. A
/// non-identity kernel with `rejuvenation_steps == 0` therefore still runs
/// the full ladder and is swept at every intermediate step (FG-N3; it used to
/// be silently ignored). Particles enter each sweep with uniform
/// `weight`/`log_weight` (FG-N7).
pub fn adaptive_smc_with_kernel<A, R, K>(
    rng: &mut R,
    num_particles: usize,
    model_fn: impl Fn() -> Model<A>,
    config: SMCConfig,
    kernel: &mut K,
) -> SMCResult
where
    R: Rng,
    K: PopulationKernel<A>,
{
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

    if config.rejuvenation_steps == 0 && kernel.is_identity() {
        // Without any move — no per-particle rejuvenation AND an identity
        // population kernel (FG-N3) — the particle positions never change, so a
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
                // Resampled clones carry uniform weights (FG-N7): the kernel
                // sweep below sees `weight = 1/n`, `log_weight = -ln n`, not
                // the stale importance weights of the particles they were
                // copied from.
                let log_uniform = -(n as f64).ln();
                particles = indices
                    .iter()
                    .map(|&i| {
                        let mut p = particles[i].clone();
                        p.weight = 1.0 / n as f64;
                        p.log_weight = log_uniform;
                        p
                    })
                    .collect();
                for lw in log_w.iter_mut() {
                    *lw = log_uniform;
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
                // Population-coupled kernel sweep (π_β⊗…⊗π_β-invariant, weights
                // untouched — see the PopulationKernel contract). Runs before the
                // likelihood refresh below so moved traces are picked up.
                kernel.sweep(
                    rng as &mut dyn rand::RngCore,
                    &mut particles,
                    &model_fn,
                    beta,
                );
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
/// Picks the target uniformly over **all** of the trace's sites and dispatches
/// the proposal by value type through the same typed machinery as
/// [`adaptive_single_site_mh`](crate::inference::mh::adaptive_single_site_mh):
/// Gaussian/log-space/reflected walks for `F64`, a deterministic flip for
/// `Bool`, a reflected discrete walk for `U64`, prior-resample for `Usize`
/// categoricals, and an integer walk for `I64`. (The previous implementation
/// collected only `F64` sites, so populations of pure Bool/Usize/U64 traces —
/// e.g. bit-string or permutation genomes — never moved during rejuvenation.)
///
/// Acceptance is the tempered trans-dimensional ratio
///
/// ```text
/// log α = Δlog_prior + β·Δloglik + (log q_rev − log q_fwd) + dim_term
/// ```
///
/// against π_β(θ) ∝ p(θ)·p(y|θ)^β. Births/deaths of prior-proposed structure
/// carry their RJMCMC corrections in `log q_fwd`/`log q_rev` (the prior is
/// untempered in π_β, so the usual prior-cancellation still holds), and
/// `dim_term = ln|sites(current)| − ln|sites(proposed)|` (FG-20/FG-21; 0 for
/// fixed-structure models). The move is invariant for π_β, so applying it to a
/// resampled (uniform-weight) population leaves the weights uniform.
fn tempered_single_site_mh<A, R: Rng>(
    rng: &mut R,
    model_fn: &impl Fn() -> Model<A>,
    current: &Trace,
    beta: f64,
    adaptation: &mut DiminishingAdaptation,
) -> Trace {
    // Score the current state FIRST (FG-N6). The site list, the proposal base
    // and every reverse-move (death) density are read from `cur_scored`, never
    // from the caller's per-choice `logp` — a particle whose trace was assembled
    // with `insert_choice(.., 0.0)` would otherwise zero the death term and
    // over-accept structure-shrinking moves. The re-score also refreshes the
    // accumulators of the trace returned on rejection (FG-40). `ScoreGivenTrace`
    // consumes no randomness, so the RNG stream is unchanged by the reorder.
    let (_, cur_scored) = run(
        ScoreGivenTrace {
            base: current.clone(),
            trace: Trace::default(),
        },
        model_fn(),
    );

    if cur_scored.choices.is_empty() {
        // Nothing to move; doing nothing is trivially π_β-invariant.
        return cur_scored;
    }

    let sites: Vec<Address> = cur_scored.choices.keys().cloned().collect();
    let target = sites[rng.gen_range(0..sites.len())].clone();
    let scale = adaptation.get_scale(&target);

    let overrides: HashMap<Address, SiteProposal> = HashMap::new();
    let (_a_prop, prop_trace, _prop_lw, lqf, lqr, _structure_changed) =
        propose_and_score(rng, model_fn, &cur_scored, &target, scale, &overrides);

    let dim_term = (sites.len() as f64).ln() - (prop_trace.choices.len() as f64).ln();
    let logd = |t: &Trace| t.log_prior + beta * particle_log_likelihood(t);
    let log_alpha = (prop_trace.log_prior - cur_scored.log_prior)
        + beta * (particle_log_likelihood(&prop_trace) - particle_log_likelihood(&cur_scored))
        + (lqr - lqf)
        + dim_term;
    let accept = mh_accept(rng, log_alpha, logd(&cur_scored), logd(&prop_trace));
    adaptation.update(&target, accept);

    if accept {
        prop_trace
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

/// Recover the model return value (e.g. a decoded genome) from a particle's
/// trace by replaying the model against it.
///
/// Uses [`ScoreGivenTrace`], which requires the trace to be a complete
/// assignment for `model_fn` — always true for particles produced by
/// [`adaptive_smc`] / [`adaptive_smc_with_kernel`] with the same model. Costs
/// one model execution. `Particle` deliberately does not cache the return
/// value: storing it would force `A: Clone` through every resample/rejuvenation
/// path and break the move-not-clone particle construction (FG-59); decode is
/// deterministic given the trace, so nothing is lost.
///
/// For traces of uncertain provenance (where a site may be missing or
/// type-mismatched), use [`try_decode_particle`] instead — `ScoreGivenTrace`
/// panics on an incomplete assignment.
pub fn decode_particle<A>(particle: &Particle, model_fn: impl Fn() -> Model<A>) -> A {
    let (a, _) = run(
        ScoreGivenTrace {
            base: particle.trace.clone(),
            trace: Trace::default(),
        },
        model_fn(),
    );
    a
}

/// Fallible sibling of [`decode_particle`] for traces of uncertain provenance,
/// backed by [`SafeScoreGivenTrace`](crate::runtime::interpreters::SafeScoreGivenTrace):
/// a missing or type-mismatched site returns `Err` instead of panicking.
///
/// The failure signal is the safe scorer's `-∞` `log_prior` sentinel: a trace
/// that IS a complete, in-support assignment for `model_fn` always scores a
/// finite log-prior, so a non-finite one means the trace does not decode under
/// this model.
pub fn try_decode_particle<A>(
    particle: &Particle,
    model_fn: impl Fn() -> Model<A>,
) -> crate::error::FugueResult<A> {
    let (a, scored) = run(
        crate::runtime::interpreters::SafeScoreGivenTrace {
            base: particle.trace.clone(),
            trace: Trace::default(),
            warn_on_error: false,
        },
        model_fn(),
    );
    if scored.log_prior.is_finite() {
        Ok(a)
    } else {
        Err(crate::error::FugueError::trace_error(
            "try_decode_particle",
            None,
            "particle trace is not a complete in-support assignment for this model",
            crate::error::ErrorCode::TraceAddressNotFound,
        ))
    }
}

/// Decode a whole population, pairing each decoded value with its normalized
/// weight — the shape posterior readouts (weighted mean / argmax) consume.
pub fn decode_particles<A>(
    particles: &[Particle],
    model_fn: impl Fn() -> Model<A>,
) -> Vec<(A, f64)> {
    particles
        .iter()
        .map(|p| (decode_particle(p, &model_fn), p.weight))
        .collect()
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

    /// Regression (EA-as-PPL F1): rejuvenation must move non-F64 sites. The
    /// previous kernel collected only `ChoiceValue::F64` sites and returned
    /// `current.clone()` otherwise, so a population of pure-Bool traces (a
    /// bit-string genome) was frozen forever.
    #[test]
    fn test_smc_rejuvenation_moves_bitstring() {
        let n_bits = 4usize;
        let model_fn = move || {
            let bits: Vec<Model<bool>> = (0..n_bits)
                .map(|i| sample(addr!("bit", i), Bernoulli::new(0.5).unwrap()))
                .collect();
            crate::core::model::sequence_vec(bits).bind(|bs| {
                let k = bs.iter().filter(|&&b| b).count() as f64;
                crate::core::model::factor(k).map(move |_| bs)
            })
        };

        // Direct movement check: a population cloned from ONE prior draw must
        // diversify under rejuvenation (before the fix: bit-identical forever).
        let mut rng = StdRng::seed_from_u64(11);
        let seed_particles = smc_prior_particles(&mut rng, 1, model_fn);
        let mut particles: Vec<Particle> = (0..20).map(|_| seed_particles[0].clone()).collect();
        rejuvenate_particles(&mut rng, &mut particles, model_fn, 1.0, 5);
        let moved = particles.iter().any(|p| {
            (0..n_bits).any(|i| {
                p.trace.get_bool(&addr!("bit", i))
                    != seed_particles[0].trace.get_bool(&addr!("bit", i))
            })
        });
        assert!(
            moved,
            "Bool-only population did not move under rejuvenation"
        );

        // Analytic marginal check: per-bit posterior p(1) = e/(1+e) ≈ 0.7311.
        let config = SMCConfig {
            resampling_method: ResamplingMethod::Systematic,
            ess_threshold: 0.5,
            rejuvenation_steps: 2,
        };
        let result = adaptive_smc(&mut rng, 400, model_fn, config);
        let p1 = std::f64::consts::E / (1.0 + std::f64::consts::E);
        for i in 0..n_bits {
            let mean: f64 = result
                .iter()
                .map(|p| {
                    let b = p.trace.get_bool(&addr!("bit", i)).unwrap();
                    p.weight * if b { 1.0 } else { 0.0 }
                })
                .sum();
            assert!(
                (mean - p1).abs() < 0.09,
                "bit {} posterior mean {} vs analytic {}",
                i,
                mean,
                p1
            );
        }
    }

    /// Helper: two independent Normal(0,1) sites each observed with sd 1.
    /// Posterior per site: Normal(y_i/2, 1/2).
    fn two_site_model(y0: f64, y1: f64) -> impl Fn() -> Model<(f64, f64)> + Clone {
        move || {
            sample(addr!("x", 0), Normal::new(0.0, 1.0).unwrap()).and_then(move |x0| {
                sample(addr!("x", 1), Normal::new(0.0, 1.0).unwrap()).and_then(move |x1| {
                    observe(addr!("y", 0), Normal::new(x0, 1.0).unwrap(), y0).and_then(move |_| {
                        observe(addr!("y", 1), Normal::new(x1, 1.0).unwrap(), y1)
                            .map(move |_| (x0, x1))
                    })
                })
            })
        }
    }

    /// A value-independent, pair-symmetric mask: each of the two sites is
    /// included in the swap independently with probability 1/2.
    #[allow(clippy::type_complexity)]
    fn random_site_mask() -> Box<dyn Fn(&Trace, &Trace, &mut dyn rand::RngCore) -> Vec<Address>> {
        Box::new(|a: &Trace, _b: &Trace, rng: &mut dyn rand::RngCore| {
            a.choices
                .keys()
                .filter(|_| rng.gen::<bool>())
                .cloned()
                .collect()
        })
    }

    /// EA-as-PPL F4: the crossover kernel is π⊗π-invariant — a population
    /// initialized FROM the analytic posterior stays posterior-distributed
    /// under repeated sweeps. (Crossover only exchanges values between
    /// particles, so this — not prior-to-posterior transport — is the correct
    /// invariance check.)
    #[test]
    fn test_crossover_product_invariance() {
        let (y0, y1) = (1.0, -0.5);
        let model_fn = two_site_model(y0, y1);
        let mut rng = StdRng::seed_from_u64(77);

        // Initialize each particle exactly from the product posterior.
        let post0 = Normal::new(y0 / 2.0, (0.5f64).sqrt()).unwrap();
        let post1 = Normal::new(y1 / 2.0, (0.5f64).sqrt()).unwrap();
        let n = 300;
        let mut particles: Vec<Particle> = (0..n)
            .map(|_| {
                let mut base = Trace::default();
                base.insert_choice(
                    addr!("x", 0),
                    crate::runtime::trace::ChoiceValue::F64(post0.sample(&mut rng)),
                    0.0,
                );
                base.insert_choice(
                    addr!("x", 1),
                    crate::runtime::trace::ChoiceValue::F64(post1.sample(&mut rng)),
                    0.0,
                );
                let (_, scored) = run(
                    ScoreGivenTrace {
                        base,
                        trace: Trace::default(),
                    },
                    model_fn(),
                );
                Particle {
                    trace: scored,
                    weight: 1.0 / n as f64,
                    log_weight: -(n as f64).ln(),
                }
            })
            .collect();

        let mut kernel = CrossoverKernel {
            n_pairs: 150,
            mask: random_site_mask(),
        };
        for _ in 0..40 {
            PopulationKernel::<(f64, f64)>::sweep(
                &mut kernel,
                &mut rng,
                &mut particles,
                &model_fn,
                1.0,
            );
        }

        for (i, target_mean) in [(0usize, y0 / 2.0), (1usize, y1 / 2.0)] {
            let xs: Vec<f64> = particles
                .iter()
                .map(|p| p.trace.get_f64(&addr!("x", i)).unwrap())
                .collect();
            let mean: f64 = xs.iter().sum::<f64>() / xs.len() as f64;
            let var: f64 = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / xs.len() as f64;
            assert!(
                (mean - target_mean).abs() < 0.15,
                "site {} marginal mean {} drifted from posterior {}",
                i,
                mean,
                target_mean
            );
            assert!(
                (var - 0.5).abs() < 0.15,
                "site {} marginal var {} drifted from posterior 0.5",
                i,
                var
            );
        }
    }

    /// EA-as-PPL F4 contract (W): a sweep never touches particle weights.
    #[test]
    fn test_crossover_preserves_uniform_weights() {
        let model_fn = two_site_model(1.0, -0.5);
        let mut rng = StdRng::seed_from_u64(88);
        let mut particles = smc_prior_particles(&mut rng, 30, &model_fn);
        let before: Vec<(f64, f64)> = particles.iter().map(|p| (p.weight, p.log_weight)).collect();

        let mut kernel = CrossoverKernel {
            n_pairs: 60,
            mask: random_site_mask(),
        };
        PopulationKernel::<(f64, f64)>::sweep(
            &mut kernel,
            &mut rng,
            &mut particles,
            &model_fn,
            0.7,
        );

        let after: Vec<(f64, f64)> = particles.iter().map(|p| (p.weight, p.log_weight)).collect();
        assert_eq!(before, after, "crossover sweep modified particle weights");
    }

    /// EA-as-PPL F4 contract (E) / FG-58: an invariant kernel must not shift
    /// the log-evidence estimate. Both runs are compared to the analytic
    /// marginal likelihood of the conjugate model.
    #[test]
    fn test_crossover_evidence_noncorruption() {
        let (y0, y1) = (1.0, -0.5);
        let model_fn = two_site_model(y0, y1);
        // Analytic: y_i ~ N(0, sqrt(1² + 1²)) independently.
        let marg = Normal::new(0.0, (2.0f64).sqrt()).unwrap();
        let analytic = marg.log_prob(&y0) + marg.log_prob(&y1);

        let config = || SMCConfig {
            resampling_method: ResamplingMethod::Systematic,
            ess_threshold: 0.7,
            rejuvenation_steps: 2,
        };
        let mut rng = StdRng::seed_from_u64(99);
        let plain = adaptive_smc(&mut rng, 600, &model_fn, config());
        let mut kernel = CrossoverKernel {
            n_pairs: 300,
            mask: random_site_mask(),
        };
        let crossed = adaptive_smc_with_kernel(&mut rng, 600, &model_fn, config(), &mut kernel);

        assert!(
            (plain.log_evidence - analytic).abs() < 0.25,
            "NoKernel evidence {} vs analytic {}",
            plain.log_evidence,
            analytic
        );
        assert!(
            (crossed.log_evidence - analytic).abs() < 0.25,
            "CrossoverKernel evidence {} vs analytic {}",
            crossed.log_evidence,
            analytic
        );
    }

    /// EA-as-PPL F4: swaps that leave the target's support must be rejected.
    /// The constraint couples the two sites (x0 + x1 ≤ 1), so a crossover swap
    /// CAN violate it — the re-scored `-∞` density must reject the move.
    #[test]
    fn test_crossover_support_truncation() {
        let model_fn = || {
            sample(addr!("x", 0), Normal::new(0.0, 1.0).unwrap()).and_then(|x0| {
                sample(addr!("x", 1), Normal::new(0.0, 1.0).unwrap()).and_then(move |x1| {
                    crate::core::model::guard(x0 + x1 <= 1.0).map(move |_| (x0, x1))
                })
            })
        };
        let mut rng = StdRng::seed_from_u64(111);

        // Build a valid population by rejection from the prior.
        let n = 60;
        let mut particles = Vec::with_capacity(n);
        while particles.len() < n {
            let (_, t) = run(
                PriorHandler {
                    rng: &mut rng,
                    trace: Trace::default(),
                },
                model_fn(),
            );
            if t.total_log_weight().is_finite() {
                particles.push(Particle {
                    trace: t,
                    weight: 1.0 / n as f64,
                    log_weight: -(n as f64).ln(),
                });
            }
        }

        let mut kernel = CrossoverKernel {
            n_pairs: 120,
            // Swap only site 0 — guaranteed to threaten the joint constraint.
            mask: Box::new(|_: &Trace, _: &Trace, _: &mut dyn rand::RngCore| vec![addr!("x", 0)]),
        };
        for _ in 0..30 {
            PopulationKernel::<(f64, f64)>::sweep(
                &mut kernel,
                &mut rng,
                &mut particles,
                &model_fn,
                1.0,
            );
            for p in &particles {
                let x0 = p.trace.get_f64(&addr!("x", 0)).unwrap();
                let x1 = p.trace.get_f64(&addr!("x", 1)).unwrap();
                assert!(
                    x0 + x1 <= 1.0 + 1e-12,
                    "accepted crossover left the truncated support: {} + {} > 1",
                    x0,
                    x1
                );
            }
        }
    }

    /// EA-as-PPL F5: decode returns exactly the value implied by the trace,
    /// decoded weights sum to 1, and the fallible variant rejects a foreign
    /// trace.
    #[test]
    fn test_decode_fidelity() {
        let model_fn = || {
            sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap()).and_then(|mu| {
                observe(addr!("y"), Normal::new(mu, 1.0).unwrap(), 0.5).map(move |_| mu)
            })
        };
        let mut rng = StdRng::seed_from_u64(123);
        let particles = smc_prior_particles(&mut rng, 20, model_fn);
        for p in &particles {
            let decoded = decode_particle(p, model_fn);
            assert_eq!(decoded, p.trace.get_f64(&addr!("mu")).unwrap());
            assert_eq!(try_decode_particle(p, model_fn).unwrap(), decoded);
        }
        let decoded = decode_particles(&particles, model_fn);
        let total: f64 = decoded.iter().map(|(_, w)| w).sum();
        assert!((total - 1.0).abs() < 1e-9);

        // A trace missing the model's site must fail the fallible decode.
        let foreign = Particle {
            trace: Trace::default(),
            weight: 1.0,
            log_weight: 0.0,
        };
        assert!(try_decode_particle(&foreign, model_fn).is_err());
    }

    /// EA-as-PPL F5 + EV-16 end-to-end: the fugue-evo conjugate "fitness as
    /// likelihood" target — prior N(0, 2²), factor −½(x−3)² — reproduced
    /// through `adaptive_smc_with_kernel` + `decode_particles`: posterior mean
    /// 2.4 ± 0.15, variance 0.8 ± 0.2. This is the readout path fugue-evo's
    /// rebuilt EvolutionarySMC uses in place of cached genome/fitness fields.
    #[test]
    fn test_decode_weighted_mean() {
        let model_fn = || {
            sample(addr!("gene", 0), Normal::new(0.0, 2.0).unwrap()).and_then(|x| {
                crate::core::model::factor(-0.5 * (x - 3.0) * (x - 3.0)).map(move |_| x)
            })
        };
        let mut rng = StdRng::seed_from_u64(2024);
        let config = SMCConfig {
            resampling_method: ResamplingMethod::Systematic,
            ess_threshold: 0.7,
            rejuvenation_steps: 3,
        };
        let mut kernel = CrossoverKernel {
            n_pairs: 200,
            mask: Box::new(|_: &Trace, _: &Trace, _: &mut dyn rand::RngCore| {
                vec![addr!("gene", 0)]
            }),
        };
        let result = adaptive_smc_with_kernel(&mut rng, 800, model_fn, config, &mut kernel);

        let decoded = decode_particles(&result, model_fn);
        let mean: f64 = decoded.iter().map(|(x, w)| x * w).sum();
        let var: f64 = decoded.iter().map(|(x, w)| w * (x - mean).powi(2)).sum();
        assert!(
            (mean - 2.4).abs() < 0.15,
            "EV-16 posterior mean {} vs analytic 2.4",
            mean
        );
        assert!(
            (var - 0.8).abs() < 0.2,
            "EV-16 posterior variance {} vs analytic 0.8",
            var
        );
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
