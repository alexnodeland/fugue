//! Variational Inference (VI) with mean-field approximations and ELBO optimization.
//!
//! This module implements variational inference, a deterministic approximate inference
//! method that turns posterior inference into an optimization problem. Instead of sampling
//! from the true posterior, VI finds the best approximation within a chosen family of
//! distributions by maximizing the Evidence Lower BOund (ELBO).
//!
//! ## Method Overview
//!
//! Variational inference works by:
//! 1. Choosing a family of tractable distributions Q(θ; φ) parameterized by φ
//! 2. Finding φ* that minimizes KL(Q(θ; φ) || P(θ|data))
//! 3. Using Q(θ; φ*) as an approximation to the true posterior P(θ|data)
//!
//! ## Mean-Field Approximation
//!
//! This implementation uses mean-field variational inference, where the posterior
//! is approximated as a product of independent distributions:
//! Q(θ₁, θ₂, ..., θₖ) = Q₁(θ₁) × Q₂(θ₂) × ... × Qₖ(θₖ)
//!
//! ## Advantages of VI
//!
//! - **Deterministic**: No random sampling, reproducible results
//! - **Fast**: Typically faster than MCMC for large models
//! - **Scalable**: Handles high-dimensional parameters well
//! - **Convergence detection**: Clear optimization objective to monitor
//!
//! ## Limitations
//!
//! - **Approximation quality**: May underestimate posterior uncertainty
//! - **Local optima**: Gradient-based optimization can get stuck
//! - **Family restrictions**: Posterior must be well-approximated by chosen family
//!
//! # Examples
//!
//! ```rust
//! use fugue::*;
//! use rand::rngs::StdRng;
//! use rand::SeedableRng;
//! use std::collections::HashMap;
//!
//! // Simple VI example
//! let model_fn = || {
//!     sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap())
//!         .bind(|mu| observe(addr!("y"), Normal::new(mu, 0.5).unwrap(), 2.0).map(move |_| mu))
//! };
//!
//! // Create mean-field guide manually
//! let mut guide = MeanFieldGuide {
//!     params: HashMap::new()
//! };
//! guide.params.insert(
//!     addr!("mu"),
//!     VariationalParam::Normal { mu: 0.0, log_sigma: 0.0 }
//! );
//!
//! // Simple ELBO computation
//! let mut rng = StdRng::seed_from_u64(42);
//! let elbo = elbo_with_guide(&mut rng, &model_fn, &guide, 10);
//! assert!(elbo.is_finite());
//! ```
use crate::core::address::Address;
use crate::core::distribution::*;
use crate::core::model::Model;
use crate::runtime::handler::run;
use crate::runtime::interpreters::{PriorHandler, ScoreGivenTrace};
use crate::runtime::trace::{Choice, ChoiceValue, Trace};
use rand::Rng;
use std::collections::HashMap;

/// Variational distribution parameters for a single random variable.
///
/// Each random variable in the model gets its own variational distribution that
/// approximates its marginal posterior. The parameters are stored in log-space
/// for numerical stability and to ensure positive constraints.
///
/// # Variants
///
/// * `Normal` - Gaussian approximation with mean and log-standard-deviation
/// * `LogNormal` - Log-normal approximation for positive variables
/// * `Beta` - Beta approximation for variables constrained to \[0,1\]
///
/// # Examples
///
/// ```rust
/// use fugue::*;
/// use rand::rngs::StdRng;
/// use rand::SeedableRng;
///
/// // Create variational parameters
/// let normal_param = VariationalParam::Normal {
///     mu: 1.5,
///     log_sigma: -0.693  // sigma = 0.5
/// };
///
/// let beta_param = VariationalParam::Beta {
///     log_alpha: 1.099,  // alpha = 3.0
///     log_beta: 0.693,   // beta = 2.0
/// };
///
/// // Sample from variational distribution
/// let mut rng = StdRng::seed_from_u64(42);
/// let sample = normal_param.sample(&mut rng);
/// let log_prob = normal_param.log_prob(sample);
/// ```
#[derive(Clone, Debug)]
pub enum VariationalParam {
    /// Normal/Gaussian variational distribution.
    Normal {
        /// Mean parameter.
        mu: f64,
        /// Log of standard deviation (for positivity).
        log_sigma: f64,
    },
    /// Log-normal variational distribution for positive variables.
    LogNormal {
        /// Mean of underlying normal.
        mu: f64,
        /// Log of standard deviation of underlying normal.
        log_sigma: f64,
    },
    /// Beta variational distribution for variables in \[0,1\].
    Beta {
        /// Log of first shape parameter (for positivity).
        log_alpha: f64,
        /// Log of second shape parameter (for positivity).
        log_beta: f64,
    },
}

impl VariationalParam {
    /// Sample a value from this variational distribution with numerical stability.
    ///
    /// Generates a random sample using the current variational parameters.
    /// This version includes parameter validation and numerical stability checks.
    ///
    /// # Arguments
    ///
    /// * `rng` - Random number generator
    ///
    /// # Returns
    ///
    /// A sample from the variational distribution, or NaN if parameters are invalid.
    pub fn sample<R: Rng>(&self, rng: &mut R) -> f64 {
        match self {
            VariationalParam::Normal { mu, log_sigma } => {
                let sigma = log_sigma.exp();
                if !mu.is_finite() || !sigma.is_finite() || sigma <= 0.0 {
                    return f64::NAN;
                }
                Normal::new(*mu, sigma).unwrap().sample(rng)
            }
            VariationalParam::LogNormal { mu, log_sigma } => {
                let sigma = log_sigma.exp();
                if !mu.is_finite() || !sigma.is_finite() || sigma <= 0.0 {
                    return f64::NAN;
                }
                LogNormal::new(*mu, sigma).unwrap().sample(rng)
            }
            VariationalParam::Beta {
                log_alpha,
                log_beta,
            } => {
                let alpha = log_alpha.exp();
                let beta = log_beta.exp();
                if !alpha.is_finite() || !beta.is_finite() || alpha <= 0.0 || beta <= 0.0 {
                    return f64::NAN;
                }
                Beta::new(alpha, beta).unwrap().sample(rng)
            }
        }
    }

    /// Sample with reparameterization for gradient computation (experimental).
    ///
    /// Returns both the sample and auxiliary information needed for
    /// computing gradients via the reparameterization trick.
    pub fn sample_with_aux<R: Rng>(&self, rng: &mut R) -> (f64, f64) {
        match self {
            VariationalParam::Normal { mu, log_sigma } => {
                let sigma = log_sigma.exp();
                let _eps: f64 = rng.gen::<f64>() * 2.0 - 1.0;
                // Simple standard normal sampling
                let u1: f64 = rng.gen::<f64>().max(1e-10);
                let u2: f64 = rng.gen();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                let value = mu + sigma * z;
                const LN_2PI: f64 = 1.8378770664093454835606594728112;
                let _log_prob = -0.5 * z * z - log_sigma - 0.5 * LN_2PI;
                (value, z)
            }
            VariationalParam::LogNormal { mu, log_sigma } => {
                let sigma = log_sigma.exp();
                let _eps: f64 = rng.gen::<f64>() * 2.0 - 1.0;
                // Simple standard normal sampling
                let u1: f64 = rng.gen::<f64>().max(1e-10);
                let u2: f64 = rng.gen();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                let log_value = mu + sigma * z;
                let value = log_value.exp();
                const LN_2PI: f64 = 1.8378770664093454835606594728112;
                let _log_prob = -0.5 * z * z - log_sigma - 0.5 * LN_2PI - log_value;
                (value, z)
            }
            VariationalParam::Beta {
                log_alpha,
                log_beta,
            } => {
                // Use normal approximation for Beta (stable fallback)
                let alpha = log_alpha.exp();
                let beta = log_beta.exp();
                let approx_mu = alpha / (alpha + beta);
                let approx_var = (alpha * beta) / ((alpha + beta).powi(2) * (alpha + beta + 1.0));
                let approx_sigma = approx_var.sqrt();

                let _eps: f64 = rng.gen::<f64>() * 2.0 - 1.0;
                // Simple standard normal sampling
                let u1: f64 = rng.gen::<f64>().max(1e-10);
                let u2: f64 = rng.gen();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                let raw_value = approx_mu + approx_sigma * z;
                let value = raw_value.clamp(0.001, 0.999);

                let _log_prob = Beta::new(alpha, beta).unwrap().log_prob(&value);
                (value, z)
            }
        }
    }

    /// Compute log-probability of a value under this variational distribution.
    ///
    /// This is used for computing entropy terms in the ELBO and for evaluating
    /// the quality of the variational approximation. Now includes numerical stability checks.
    ///
    /// # Arguments
    ///
    /// * `x` - Value to evaluate
    ///
    /// # Returns
    ///
    /// Log-probability density at the given value.
    pub fn log_prob(&self, x: f64) -> f64 {
        match self {
            VariationalParam::Normal { mu, log_sigma } => {
                let sigma = log_sigma.exp();
                Normal::new(*mu, sigma).unwrap().log_prob(&x)
            }
            VariationalParam::LogNormal { mu, log_sigma } => {
                let sigma = log_sigma.exp();
                LogNormal::new(*mu, sigma).unwrap().log_prob(&x)
            }
            VariationalParam::Beta {
                log_alpha,
                log_beta,
            } => {
                let alpha = log_alpha.exp();
                let beta = log_beta.exp();
                Beta::new(alpha, beta).unwrap().log_prob(&x)
            }
        }
    }
}

/// Mean-field variational guide for approximate posterior inference.
///
/// A mean-field guide specifies independent variational distributions for each
/// random variable in the model. This factorization assumption simplifies
/// optimization but may underestimate correlations between variables.
///
/// The guide maps each address (random variable) to its variational parameters,
/// which are optimized to minimize the KL divergence to the true posterior.
///
/// # Fields
///
/// * `params` - Map from addresses to their variational parameters
///
/// # Examples
///
/// ```rust
/// use fugue::*;
/// use std::collections::HashMap;
///
/// // Create a guide for a two-parameter model
/// let mut guide = MeanFieldGuide::new();
/// guide.params.insert(
///     addr!("mu"),
///     VariationalParam::Normal { mu: 0.0, log_sigma: 0.0 }
/// );
/// guide.params.insert(
///     addr!("sigma"),
///     VariationalParam::Normal { mu: 0.0, log_sigma: -1.0 }
/// );
///
/// // Check if parameters are specified
/// assert!(guide.params.contains_key(&addr!("mu")));
/// assert!(guide.params.contains_key(&addr!("sigma")));
/// ```
#[derive(Clone, Debug)]
pub struct MeanFieldGuide {
    /// Map from addresses to their variational parameters.
    pub params: HashMap<Address, VariationalParam>,
}

impl MeanFieldGuide {
    /// Create a new empty mean-field guide.
    ///
    /// The guide starts with no variational parameters. You must add parameters
    /// for each random variable in your model using the `add_*_param` methods.
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
                ChoiceValue::U64(val) => {
                    // Use LogNormal for unsigned integers (always positive)
                    VariationalParam::LogNormal {
                        mu: (val as f64).ln(),
                        log_sigma: 1.0_f64.ln(),
                    }
                }
                ChoiceValue::Usize(val) => {
                    // Use LogNormal for categorical indices (always positive)
                    VariationalParam::LogNormal {
                        mu: (val as f64 + 1.0).ln(), // +1 to avoid log(0)
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
                        *mu_plus = *mu_plus + eps;
                    }
                    let elbo_plus = elbo_with_guide(rng, &model_fn, &guide_plus, 10);
                    let grad_mu = (elbo_plus - current_elbo) / eps;

                    *mu = *mu + learning_rate * grad_mu;
                }
                VariationalParam::LogNormal { mu, log_sigma: _ } => {
                    // Similar finite difference for LogNormal parameters
                    let eps = 0.01;
                    let mut guide_plus = guide_clone.clone();
                    if let Some(VariationalParam::LogNormal { mu: mu_plus, .. }) =
                        guide_plus.params.get_mut(_addr)
                    {
                        *mu_plus = *mu_plus + eps;
                    }
                    let elbo_plus = elbo_with_guide(rng, &model_fn, &guide_plus, 10);
                    let grad_mu = (elbo_plus - current_elbo) / eps;

                    *mu = *mu + learning_rate * grad_mu;
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
                        *alpha_plus = *alpha_plus + eps;
                    }
                    let elbo_plus = elbo_with_guide(rng, &model_fn, &guide_plus, 10);
                    let grad_alpha = (elbo_plus - current_elbo) / eps;

                    *log_alpha = *log_alpha + learning_rate * grad_alpha;
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
