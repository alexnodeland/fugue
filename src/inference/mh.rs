//! Metropolis–Hastings sampling primitives.
//!
//! This module provides a minimal MH transition that proposes a fresh prior trace
//! and accepts/rejects based on the score difference. It is intended as a simple
//! placeholder and should be replaced with site-wise proposals.
use rand::Rng;
use crate::runtime::trace::Trace;
use crate::runtime::interpreters::{ScoreGivenTrace, PriorHandler};
use crate::runtime::handler::run;
use crate::core::model::Model;

pub fn single_site_random_walk_mh<A, R: Rng>(rng: &mut R, _proposal_sigma: f64, model_fn: impl Fn() -> Model<A>, current: &Trace) -> (A, Trace) {
  // Propose new trace using prior with a different seed path; simplistic: draw a fresh prior trace
  // In a more advanced system, we'd only perturb one site and reuse others.
  let (_a_prop, prop_trace) = run(PriorHandler{rng, trace: Trace::default()}, model_fn());
  // Score both traces under the same model
  let (_a_cur, cur_scored) = run(ScoreGivenTrace{base: current.clone(), trace: Trace::default()}, model_fn());
  let (a_prop2, prop_scored) = run(ScoreGivenTrace{base: prop_trace.clone(), trace: Trace::default()}, model_fn());
  let log_alpha = prop_scored.total_log_weight() - cur_scored.total_log_weight();
  let accept = log_alpha >= 0.0 || rng.gen::<f64>() < (log_alpha).exp();
  if accept { (a_prop2, prop_trace) } else { (_a_cur, current.clone()) }
}
