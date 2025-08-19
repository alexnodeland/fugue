//! Sequential Monte Carlo (particle filtering) scaffolding.
//!
//! Provides a simple prior particle generator and log-weight normalization. A full
//! SMC implementation would include resampling and model sequencing.
use rand::Rng;
use crate::runtime::trace::Trace;
use crate::runtime::interpreters::PriorHandler;
use crate::runtime::handler::run;
use crate::core::model::Model;

pub struct Particle { pub trace: Trace, pub weight: f64 }

pub fn smc_prior_particles<A, R: Rng>(rng: &mut R, num_particles: usize, model_fn: impl Fn() -> Model<A>) -> Vec<Particle> {
  let mut particles = Vec::with_capacity(num_particles);
  for _ in 0..num_particles {
    let (_a, t) = run(PriorHandler{rng, trace: Trace::default()}, model_fn());
    particles.push(Particle{ trace: t.clone(), weight: t.total_log_weight() });
  }
  // Normalize weights (log-space naive)
  let max_w = particles.iter().map(|p| p.weight).fold(f64::NEG_INFINITY, f64::max);
  let sum = particles.iter().map(|p| (p.weight - max_w).exp()).sum::<f64>();
  for p in &mut particles { p.weight = ((p.weight - max_w).exp()) / sum; }
  particles
}
