//! Variational inference scaffolding.
//!
//! Contains a very basic Monte Carlo ELBO estimator using the prior as proposal.
//! Replace with a structured variational family and reparameterized gradients.
use crate::core::model::Model;
use crate::runtime::handler::run;
use crate::runtime::interpreters::{PriorHandler, ScoreGivenTrace};
use crate::runtime::trace::Trace;
use rand::Rng;

// Very basic Monte Carlo ELBO estimator using prior as proposal (not practical, but a placeholder)
pub fn estimate_elbo<A, R: Rng>(
    rng: &mut R,
    model_fn: impl Fn() -> Model<A>,
    num_samples: usize,
) -> f64 {
    let mut total = 0.0;
    for _ in 0..num_samples {
        let (_a, prior_t) = run(
            PriorHandler {
                rng,
                trace: Trace::default(),
            },
            model_fn(),
        );
        let (_a2, scored) = run(
            ScoreGivenTrace {
                base: prior_t.clone(),
                trace: Trace::default(),
            },
            model_fn(),
        );
        total += scored.total_log_weight();
    }
    total / (num_samples as f64)
}
