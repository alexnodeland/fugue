//! Variational inference with mean-field guides and ELBO optimization.
//!
//! Implements structured variational families and gradient-based optimization
//! for approximate posterior inference.
use crate::core::address::Address;
use crate::core::distribution::*;
use crate::core::model::Model;
use crate::runtime::handler::run;
use crate::runtime::interpreters::{PriorHandler, ScoreGivenTrace};
use crate::runtime::trace::{Choice, ChoiceValue, Trace};
use rand::Rng;
use std::collections::HashMap;

/// Variational parameter for a single site.
#[derive(Clone, Debug)]
pub enum VariationalParam {
    Normal { mu: f64, log_sigma: f64 },
    LogNormal { mu: f64, log_sigma: f64 },
    Beta { log_alpha: f64, log_beta: f64 },
}

impl VariationalParam {
    /// Sample from the variational distribution.
    pub fn sample<R: Rng>(&self, rng: &mut R) -> f64 {
        match self {
            VariationalParam::Normal { mu, log_sigma } => {
                let sigma = log_sigma.exp();
                Normal { mu: *mu, sigma }.sample(rng)
            }
            VariationalParam::LogNormal { mu, log_sigma } => {
                let sigma = log_sigma.exp();
                LogNormal { mu: *mu, sigma }.sample(rng)
            }
            VariationalParam::Beta {
                log_alpha,
                log_beta,
            } => {
                let alpha = log_alpha.exp();
                let beta = log_beta.exp();
                Beta { alpha, beta }.sample(rng)
            }
        }
    }

    /// Log probability under the variational distribution.
    pub fn log_prob(&self, x: f64) -> f64 {
        match self {
            VariationalParam::Normal { mu, log_sigma } => {
                let sigma = log_sigma.exp();
                Normal { mu: *mu, sigma }.log_prob(x)
            }
            VariationalParam::LogNormal { mu, log_sigma } => {
                let sigma = log_sigma.exp();
                LogNormal { mu: *mu, sigma }.log_prob(x)
            }
            VariationalParam::Beta {
                log_alpha,
                log_beta,
            } => {
                let alpha = log_alpha.exp();
                let beta = log_beta.exp();
                Beta { alpha, beta }.log_prob(x)
            }
        }
    }
}

/// Mean-field variational family.
#[derive(Clone, Debug)]
pub struct MeanFieldGuide {
    pub params: HashMap<Address, VariationalParam>,
}

impl MeanFieldGuide {
    pub fn new() -> Self {
        Self {
            params: HashMap::new(),
        }
    }

    /// Initialize guide from a prior trace.
    pub fn from_trace(trace: &Trace) -> Self {
        let mut guide = Self::new();

        for (addr, choice) in &trace.choices {
            let param = match choice.value {
                ChoiceValue::F64(val) => {
                    if val > 0.0 {
                        // Use LogNormal for positive values
                        VariationalParam::LogNormal {
                            mu: val.ln(),
                            log_sigma: 0.0_f64.ln(),
                        }
                    } else {
                        // Use Normal for real values
                        VariationalParam::Normal {
                            mu: val,
                            log_sigma: 1.0_f64.ln(),
                        }
                    }
                }
                ChoiceValue::Bool(_) => {
                    // Use Beta(1,1) = Uniform for boolean (as continuous relaxation)
                    VariationalParam::Beta {
                        log_alpha: 1.0_f64.ln(),
                        log_beta: 1.0_f64.ln(),
                    }
                }
                ChoiceValue::I64(val) => {
                    // Use Normal for integers (continuous relaxation)
                    VariationalParam::Normal {
                        mu: val as f64,
                        log_sigma: 1.0_f64.ln(),
                    }
                }
            };
            guide.params.insert(addr.clone(), param);
        }
        guide
    }

    /// Sample a trace from the guide.
    pub fn sample_trace<R: Rng>(&self, rng: &mut R) -> Trace {
        let mut trace = Trace::default();

        for (addr, param) in &self.params {
            let value = param.sample(rng);
            let log_prob = param.log_prob(value);

            trace.choices.insert(
                addr.clone(),
                Choice {
                    addr: addr.clone(),
                    value: ChoiceValue::F64(value),
                    logp: log_prob,
                },
            );
            trace.log_prior += log_prob;
        }
        trace
    }
}

/// ELBO estimation using a variational guide.
pub fn elbo_with_guide<A, R: Rng>(
    rng: &mut R,
    model_fn: impl Fn() -> Model<A>,
    guide: &MeanFieldGuide,
    num_samples: usize,
) -> f64 {
    let mut total_elbo = 0.0;

    for _ in 0..num_samples {
        let guide_trace = guide.sample_trace(rng);
        let (_a, model_trace) = run(
            ScoreGivenTrace {
                base: guide_trace.clone(),
                trace: Trace::default(),
            },
            model_fn(),
        );

        // ELBO = E_q[log p(x,z) - log q(z)]
        let log_joint = model_trace.total_log_weight();
        let log_guide = guide_trace.log_prior;
        total_elbo += log_joint - log_guide;
    }

    total_elbo / num_samples as f64
}

/// Simple VI optimization using coordinate ascent.
pub fn optimize_meanfield_vi<A, R: Rng>(
    rng: &mut R,
    model_fn: impl Fn() -> Model<A>,
    initial_guide: MeanFieldGuide,
    n_iterations: usize,
    n_samples_per_iter: usize,
    learning_rate: f64,
) -> MeanFieldGuide {
    let mut guide = initial_guide;

    for iter in 0..n_iterations {
        let current_elbo = elbo_with_guide(rng, &model_fn, &guide, n_samples_per_iter);

        // Simple gradient ascent (placeholder - would use automatic differentiation in practice)
        let guide_clone = guide.clone();
        for (_addr, param) in &mut guide.params {
            match param {
                VariationalParam::Normal { mu, log_sigma: _ } => {
                    // Finite difference gradients (very basic)
                    let eps = 0.01;
                    let mut guide_plus = guide_clone.clone();
                    if let Some(VariationalParam::Normal { mu: mu_plus, .. }) =
                        guide_plus.params.get_mut(_addr)
                    {
                        *mu_plus += eps;
                    }
                    let elbo_plus = elbo_with_guide(rng, &model_fn, &guide_plus, 10);
                    let grad_mu = (elbo_plus - current_elbo) / eps;

                    *mu += learning_rate * grad_mu;
                }
                VariationalParam::LogNormal { mu, log_sigma: _ } => {
                    // Similar finite difference for LogNormal parameters
                    let eps = 0.01;
                    let mut guide_plus = guide_clone.clone();
                    if let Some(VariationalParam::LogNormal { mu: mu_plus, .. }) =
                        guide_plus.params.get_mut(_addr)
                    {
                        *mu_plus += eps;
                    }
                    let elbo_plus = elbo_with_guide(rng, &model_fn, &guide_plus, 10);
                    let grad_mu = (elbo_plus - current_elbo) / eps;

                    *mu += learning_rate * grad_mu;
                }
                VariationalParam::Beta {
                    log_alpha,
                    log_beta: _,
                } => {
                    // Basic update for Beta parameters
                    let eps = 0.01;
                    let mut guide_plus = guide_clone.clone();
                    if let Some(VariationalParam::Beta {
                        log_alpha: alpha_plus,
                        ..
                    }) = guide_plus.params.get_mut(_addr)
                    {
                        *alpha_plus += eps;
                    }
                    let elbo_plus = elbo_with_guide(rng, &model_fn, &guide_plus, 10);
                    let grad_alpha = (elbo_plus - current_elbo) / eps;

                    *log_alpha += learning_rate * grad_alpha;
                }
            }
        }

        if iter % 100 == 0 {
            println!("VI Iteration {}: ELBO = {:.4}", iter, current_elbo);
        }
    }

    guide
}

// Keep the original simple function for backward compatibility
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
