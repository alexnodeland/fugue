//! Sequential Monte Carlo with resampling and effective sample size monitoring.
//!
//! Provides a complete SMC implementation with multiple resampling algorithms,
//! ESS monitoring, and rejuvenation steps.
use crate::core::model::Model;
use crate::inference::mh::{adaptive_single_site_mh, AdaptiveScales};
use crate::runtime::handler::run;
use crate::runtime::interpreters::PriorHandler;
use crate::runtime::trace::Trace;
use rand::Rng;

#[derive(Clone, Debug)]
pub struct Particle {
    pub trace: Trace,
    pub weight: f64,
    pub log_weight: f64,
}

#[derive(Clone, Copy, Debug)]
pub enum ResamplingMethod {
    Multinomial,
    Systematic,
    Stratified,
}

pub struct SMCConfig {
    pub resampling_method: ResamplingMethod,
    pub ess_threshold: f64,
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

/// Calculate effective sample size.
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

/// SMC with adaptive resampling and optional rejuvenation.
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
            let mut scales = AdaptiveScales::new();
            for particle in &mut particles {
                for _ in 0..config.rejuvenation_steps {
                    let (_, new_trace) =
                        adaptive_single_site_mh(rng, &model_fn, &particle.trace, &mut scales);
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

/// Normalize particle weights.
pub fn normalize_particles(particles: &mut [Particle]) {
    let max_w = particles
        .iter()
        .map(|p| p.log_weight)
        .fold(f64::NEG_INFINITY, f64::max);
    let sum: f64 = particles.iter().map(|p| (p.log_weight - max_w).exp()).sum();

    for p in particles {
        p.weight = (p.log_weight - max_w).exp() / sum;
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
