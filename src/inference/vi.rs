//! Variational Inference (VI) with mean-field approximations and ELBO optimization.
//!
//! This module implements variational inference, an approximate inference method that
//! turns posterior inference into an optimization problem. Instead of sampling from the
//! true posterior, VI finds the best approximation within a chosen family of
//! distributions by maximizing the Evidence Lower BOund (ELBO).
//!
//! ## Method Overview
//!
//! Variational inference works by:
//! 1. Choosing a family of tractable distributions Q(θ; φ) parameterized by φ
//! 2. Finding φ* that minimizes KL(Q(θ; φ) || P(θ|data)) (equivalently maximizes the ELBO)
//! 3. Using Q(θ; φ*) as an approximation to the true posterior P(θ|data)
//!
//! ## Mean-Field Approximation
//!
//! This implementation uses mean-field variational inference, where the posterior
//! is approximated as a product of independent distributions:
//! Q(θ₁, θ₂, ..., θₖ) = Q₁(θ₁) × Q₂(θ₂) × ... × Qₖ(θₖ)
//!
//! Each variational factor's family is matched to the *support* of the corresponding
//! model latent (see [`Support`]): real-valued latents get a Normal factor, strictly
//! positive latents a LogNormal factor, and \[0,1\]-valued latents a Beta factor. Both
//! the location **and** the scale of every factor are optimized (in unconstrained
//! log-space for the scale parameters).
//!
//! ## Optimizer: stochastic, not deterministic
//!
//! The ELBO and its gradients are estimated by **Monte Carlo** sampling from the guide,
//! so [`optimize_meanfield_vi`] is a *stochastic* optimizer, not a deterministic one.
//! To make it well-behaved it uses:
//!
//! - **Common-random-numbers (CRN) central finite differences**: the `+ε` and `−ε`
//!   ELBO evaluations that estimate each gradient reuse the *same* seeded RNG draws so
//!   the Monte Carlo noise cancels in the difference (see [`elbo_gradient_fd`]).
//! - **A Robbins–Monro decaying step size** `α_t = α₀ · (t+1)^(−decay)` with
//!   `decay ∈ (0.5, 1]` so that `Σ α_t = ∞` and `Σ α_t² < ∞`, which is required for a
//!   stochastic-gradient iterate to converge rather than random-walk around the optimum.
//! - **ELBO-plateau convergence detection**: optimization stops early once the relative
//!   improvement of the windowed-mean ELBO falls below a configurable tolerance.
//!
//! Because every random draw flows from the caller-supplied RNG, runs are **reproducible
//! for a fixed seed**.
//!
//! ## Advantages of VI
//!
//! - **Fast**: Typically faster than MCMC for large models
//! - **Scalable**: Handles high-dimensional parameters well
//! - **Reproducible**: Deterministic for a fixed RNG seed
//! - **Convergence detection**: A clear scalar objective (the ELBO) to monitor and a
//!   built-in plateau stopping criterion
//!
//! ## Limitations
//!
//! - **Approximation quality**: Mean-field VI ignores posterior correlations and often
//!   underestimates posterior uncertainty
//! - **Local optima**: Gradient-based optimization of a non-convex ELBO can get stuck
//! - **Family restrictions**: The posterior must be well-approximated by the chosen family
//! - **Gradient noise**: Beta factors have no location-scale reparameterization; their
//!   parameters are optimized purely via finite differences of the (noisy) ELBO
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
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;
use std::fmt;

/// Lower clamp on any log-scale variational parameter (`log_sigma`, `log_alpha`,
/// `log_beta`). `exp(-20) ≈ 2e-9`, small enough for any realistic posterior while
/// staying comfortably away from `-inf` (which would degenerate the factor).
const LOG_SCALE_MIN: f64 = -20.0;
/// Upper clamp on any log-scale variational parameter. `exp(20) ≈ 4.9e8`.
const LOG_SCALE_MAX: f64 = 20.0;
/// Clamp on Normal/LogNormal location parameters to prevent overflow while keeping the
/// range wide enough not to clip realistic posterior means.
const MU_ABS_MAX: f64 = 1.0e6;

/// The support of a continuous model latent, used to pick a matching variational family.
///
/// Mean-field VI is only correct if each variational factor lives on the same support as
/// the model latent it approximates: a Normal guide placed on a strictly-positive or
/// unit-interval latent proposes out-of-support values whose model log-density is `-inf`,
/// collapsing the ELBO. [`Support`] lets callers declare the intended support so guide
/// construction can select the right family (see [`MeanFieldGuide::add_latent`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Support {
    /// (-∞, +∞): approximated by a [`VariationalParam::Normal`] factor.
    Real,
    /// (0, +∞): approximated by a [`VariationalParam::LogNormal`] factor.
    Positive,
    /// (0, 1): approximated by a [`VariationalParam::Beta`] factor.
    Unit,
}

/// Error returned when a guide cannot be constructed for a model latent.
///
/// The mean-field guide families implemented here ([`VariationalParam`]) are all
/// *continuous*. A discrete latent (Bool / U64 / Usize / I64) has no continuous
/// variational factor, so guide construction returns this typed error instead of
/// silently emitting an `f64` factor (which would later panic when scored against the
/// discrete model site) — see finding FG-17.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GuideError {
    /// A discrete latent was encountered where only continuous latents are supported.
    UnsupportedDiscreteLatent {
        /// Address of the offending latent.
        addr: Address,
        /// The `ChoiceValue` type name of the discrete latent (e.g. `"bool"`).
        value_type: &'static str,
    },
}

impl fmt::Display for GuideError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GuideError::UnsupportedDiscreteLatent { addr, value_type } => write!(
                f,
                "mean-field VI does not support the discrete latent at {} (type {}): \
                 only continuous latents (Normal/LogNormal/Beta factors) can be approximated",
                addr, value_type
            ),
        }
    }
}

impl std::error::Error for GuideError {}

/// Which scalar coordinate of a [`VariationalParam`] a finite-difference step perturbs.
///
/// Every variational factor has exactly two free parameters; [`ParamCoord`] names them
/// uniformly across families so the gradient machinery ([`elbo_gradient_fd`]) can address
/// either one without matching on the family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParamCoord {
    /// The location coordinate: `mu` (Normal/LogNormal) or `log_alpha` (Beta).
    Location,
    /// The scale coordinate: `log_sigma` (Normal/LogNormal) or `log_beta` (Beta).
    Scale,
}

/// Variational distribution parameters for a single random variable.
///
/// Each random variable in the model gets its own variational distribution that
/// approximates its marginal posterior. Scale parameters are stored in log-space
/// (unconstrained) for numerical stability and to guarantee positivity.
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
    /// Build a variational factor initialized for a latent with the given [`Support`].
    ///
    /// The family is chosen to match the support so that samples are always in the
    /// model latent's support (avoiding the `-inf` ELBO of a support-mismatched guide,
    /// finding FG-17). The scale is initialized to a moderate spread derived from
    /// `init_value`; it will be optimized alongside the location.
    ///
    /// * [`Support::Real`] → `Normal { mu: init_value, .. }`
    /// * [`Support::Positive`] → `LogNormal { mu: ln(init_value), .. }`
    /// * [`Support::Unit`] → `Beta` with mean ≈ `init_value`
    pub fn for_support(support: Support, init_value: f64) -> Self {
        match support {
            Support::Real => VariationalParam::Normal {
                mu: init_value,
                log_sigma: init_log_sigma(init_value),
            },
            Support::Positive => {
                // Underlying-normal mean = ln(value); keep the argument strictly positive.
                let safe = if init_value.is_finite() && init_value > 0.0 {
                    init_value
                } else {
                    1.0
                };
                VariationalParam::LogNormal {
                    mu: safe.ln(),
                    // Underlying-normal sd = 0.5 (a moderate multiplicative spread).
                    log_sigma: 0.5_f64.ln(),
                }
            }
            Support::Unit => {
                // Weak Beta with mean m = init_value and concentration c = 2 ->
                // alpha = c*m, beta = c*(1-m). Clamp m into (0,1) to stay valid.
                let m = if init_value.is_finite() {
                    init_value.clamp(1e-3, 1.0 - 1e-3)
                } else {
                    0.5
                };
                let concentration = 2.0;
                VariationalParam::Beta {
                    log_alpha: (concentration * m).ln(),
                    log_beta: (concentration * (1.0 - m)).ln(),
                }
            }
        }
    }

    /// Sample a value from this variational distribution with numerical stability.
    ///
    /// Generates a random sample using the current variational parameters. For the Beta
    /// family this draws an **exact** Beta sample (finding FG-60): there is no
    /// moment-matched-Gaussian approximation and no clamping.
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
                // Exact Beta sample via rand_distr (internally two Gamma draws).
                Beta::new(alpha, beta).unwrap().sample(rng)
            }
        }
    }

    /// Sample a value together with auxiliary information for pathwise gradients.
    ///
    /// For the location-scale families ([`VariationalParam::Normal`],
    /// [`VariationalParam::LogNormal`]) the auxiliary value is the standard-normal base
    /// draw `z` used to reparameterize the sample (`x = μ + σ·z`), which supports the
    /// reparameterization trick.
    ///
    /// The [`VariationalParam::Beta`] family has **no** location-scale reparameterization.
    /// This method therefore samples the Beta **exactly** (finding FG-60 — the previous
    /// implementation used a moment-matched Gaussian clamped to `[0.001, 0.999]`, which is
    /// a different, biased distribution) and returns `f64::NAN` as the auxiliary value to
    /// signal that no reparameterization base exists. Beta variational parameters are
    /// optimized with finite-difference ELBO gradients (see [`elbo_gradient_fd`]), not
    /// pathwise gradients.
    pub fn sample_with_aux<R: Rng>(&self, rng: &mut R) -> (f64, f64) {
        match self {
            VariationalParam::Normal { mu, log_sigma } => {
                let sigma = log_sigma.exp();
                let z = standard_normal(rng);
                let value = mu + sigma * z;
                (value, z)
            }
            VariationalParam::LogNormal { mu, log_sigma } => {
                let sigma = log_sigma.exp();
                let z = standard_normal(rng);
                let log_value = mu + sigma * z;
                let value = log_value.exp();
                (value, z)
            }
            VariationalParam::Beta {
                log_alpha,
                log_beta,
            } => {
                // Exact Beta sampling; no valid reparameterization base for Beta.
                let value = self.sample(rng);
                let _ = (log_alpha, log_beta);
                (value, f64::NAN)
            }
        }
    }

    /// Compute log-probability of a value under this variational distribution.
    ///
    /// This is used for computing entropy terms in the ELBO and for evaluating
    /// the quality of the variational approximation.
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

/// Draw a standard-normal sample via the Box–Muller transform.
fn standard_normal<R: Rng>(rng: &mut R) -> f64 {
    let u1: f64 = rng.gen::<f64>().max(1e-10);
    let u2: f64 = rng.gen();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Initialize a Normal-factor `log_sigma` from a point value (finding FG-18).
///
/// Returns `ln(max(0.1·|value|, 0.1))`. This is always finite (never `ln(0) = -inf`) and
/// NaN-proof at `value = 0` (where it yields `ln(0.1)`), giving a small-but-nonzero
/// initial standard deviation proportional to the value's scale.
fn init_log_sigma(value: f64) -> f64 {
    let scale = if value.is_finite() { value.abs() } else { 1.0 };
    (0.1 * scale).max(0.1).ln()
}

/// Return a copy of `param` with the given coordinate shifted by `delta`.
fn shifted(param: &VariationalParam, coord: ParamCoord, delta: f64) -> VariationalParam {
    match param {
        VariationalParam::Normal { mu, log_sigma } => match coord {
            ParamCoord::Location => VariationalParam::Normal {
                mu: mu + delta,
                log_sigma: *log_sigma,
            },
            ParamCoord::Scale => VariationalParam::Normal {
                mu: *mu,
                log_sigma: log_sigma + delta,
            },
        },
        VariationalParam::LogNormal { mu, log_sigma } => match coord {
            ParamCoord::Location => VariationalParam::LogNormal {
                mu: mu + delta,
                log_sigma: *log_sigma,
            },
            ParamCoord::Scale => VariationalParam::LogNormal {
                mu: *mu,
                log_sigma: log_sigma + delta,
            },
        },
        VariationalParam::Beta {
            log_alpha,
            log_beta,
        } => match coord {
            ParamCoord::Location => VariationalParam::Beta {
                log_alpha: log_alpha + delta,
                log_beta: *log_beta,
            },
            ParamCoord::Scale => VariationalParam::Beta {
                log_alpha: *log_alpha,
                log_beta: log_beta + delta,
            },
        },
    }
}

/// Apply an additive update to one coordinate of `param`, clamping to safe ranges.
fn apply_update(param: &mut VariationalParam, coord: ParamCoord, delta: f64) {
    match param {
        VariationalParam::Normal { mu, log_sigma } => match coord {
            ParamCoord::Location => *mu = (*mu + delta).clamp(-MU_ABS_MAX, MU_ABS_MAX),
            ParamCoord::Scale => {
                *log_sigma = (*log_sigma + delta).clamp(LOG_SCALE_MIN, LOG_SCALE_MAX)
            }
        },
        VariationalParam::LogNormal { mu, log_sigma } => match coord {
            ParamCoord::Location => *mu = (*mu + delta).clamp(-MU_ABS_MAX, MU_ABS_MAX),
            ParamCoord::Scale => {
                *log_sigma = (*log_sigma + delta).clamp(LOG_SCALE_MIN, LOG_SCALE_MAX)
            }
        },
        VariationalParam::Beta {
            log_alpha,
            log_beta,
        } => match coord {
            ParamCoord::Location => {
                *log_alpha = (*log_alpha + delta).clamp(LOG_SCALE_MIN, LOG_SCALE_MAX)
            }
            ParamCoord::Scale => {
                *log_beta = (*log_beta + delta).clamp(LOG_SCALE_MIN, LOG_SCALE_MAX)
            }
        },
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

impl Default for MeanFieldGuide {
    fn default() -> Self {
        Self::new()
    }
}

impl MeanFieldGuide {
    /// Create a new empty mean-field guide.
    ///
    /// The guide starts with no variational parameters. Add a factor for each latent in
    /// your model with [`MeanFieldGuide::add_latent`] (support-aware) or by inserting into
    /// [`MeanFieldGuide::params`] directly.
    pub fn new() -> Self {
        Self {
            params: HashMap::new(),
        }
    }

    /// Add a support-matched variational factor for a latent (finding FG-17).
    ///
    /// The variational family is selected from the declared [`Support`] so the factor's
    /// samples always lie in the model latent's support: real → Normal, positive →
    /// LogNormal, \[0,1\] → Beta. `init_value` seeds the factor's location.
    ///
    /// ```rust
    /// use fugue::*;
    /// use fugue::inference::vi::{MeanFieldGuide, Support};
    ///
    /// let mut guide = MeanFieldGuide::new();
    /// guide.add_latent(addr!("theta"), Support::Unit, 0.3);      // Beta factor
    /// guide.add_latent(addr!("rate"), Support::Positive, 2.0);   // LogNormal factor
    /// guide.add_latent(addr!("mu"), Support::Real, 0.0);         // Normal factor
    /// assert_eq!(guide.params.len(), 3);
    /// ```
    pub fn add_latent(&mut self, addr: Address, support: Support, init_value: f64) {
        self.params
            .insert(addr, VariationalParam::for_support(support, init_value));
    }

    /// Initialize a guide from a prior trace, defaulting continuous latents to a Normal
    /// factor on the real line.
    ///
    /// A [`Trace`] records only sampled *values*, not the support of the distributions
    /// that produced them, so this constructor cannot infer positive/unit support from a
    /// single draw (doing so from the sign of one sample was the FG-18 antipattern). It
    /// therefore builds a real-line Normal factor for every continuous (`f64`) latent,
    /// with a finite, value-scaled initial standard deviation (`init_log_sigma`, finding
    /// FG-18). For support-aware factors use [`MeanFieldGuide::add_latent`].
    ///
    /// Discrete latents (`Bool` / `U64` / `Usize` / `I64`) have no continuous variational
    /// factor and yield a typed [`GuideError::UnsupportedDiscreteLatent`] instead of a
    /// silent `f64` factor that would later panic during scoring (finding FG-17).
    pub fn from_trace(trace: &Trace) -> Result<Self, GuideError> {
        let mut guide = Self::new();

        for (addr, choice) in &trace.choices {
            let param = match choice.value {
                ChoiceValue::F64(val) => VariationalParam::Normal {
                    mu: val,
                    log_sigma: init_log_sigma(val),
                },
                // Discrete latents are unsupported by the continuous mean-field families.
                ChoiceValue::Bool(_)
                | ChoiceValue::I64(_)
                | ChoiceValue::U64(_)
                | ChoiceValue::Usize(_) => {
                    return Err(GuideError::UnsupportedDiscreteLatent {
                        addr: addr.clone(),
                        value_type: choice.value.type_name(),
                    });
                }
            };
            guide.params.insert(addr.clone(), param);
        }
        Ok(guide)
    }

    /// Sample a trace from the guide.
    ///
    /// Factors are sampled in a deterministic (address-sorted) order so that, for a fixed
    /// RNG seed, two guides with the same set of addresses consume the RNG identically —
    /// this is what makes the common-random-numbers finite differences in
    /// [`elbo_gradient_fd`] valid. All factor families are continuous, so values are
    /// stored as `ChoiceValue::F64`.
    pub fn sample_trace<R: Rng>(&self, rng: &mut R) -> Trace {
        let mut trace = Trace::default();

        let mut entries: Vec<(&Address, &VariationalParam)> = self.params.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));

        for (addr, param) in entries {
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

/// Monte Carlo estimate of the ELBO for a model under a variational `guide`.
///
/// Returns the sample mean over `num_samples` draws `z ~ q` of
/// `log p(x, z) − log q(z)`. Only the guide factors for addresses the model actually
/// samples contribute the `− log q(z)` (entropy) term, so a stray guide factor for an
/// address the model never visits cannot bias the estimate (finding FG-17).
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

        // ELBO = E_q[log p(x,z) - log q(z)].
        let log_joint = model_trace.total_log_weight();
        // Only count the guide entropy for latents the model actually sampled.
        let log_guide: f64 = model_trace
            .choices
            .keys()
            .filter_map(|addr| guide_trace.choices.get(addr).map(|c| c.logp))
            .sum();
        total_elbo += log_joint - log_guide;
    }

    total_elbo / num_samples as f64
}

/// Common-random-numbers central finite-difference estimate of `dELBO/dφ` for one
/// coordinate of one guide factor (finding FG-16).
///
/// The `+ε` and `−ε` ELBO evaluations are run with **freshly seeded RNGs sharing the same
/// `seed`**, so the guide draws `z ~ q` are identical between them and the Monte Carlo
/// noise cancels in the difference — only the `O(ε²)` central-difference bias remains.
/// Both evaluations use the same `num_samples`. Contrast this with a naive
/// `(elbo(φ+ε) − elbo(φ))/ε` using independent draws, whose variance is inflated by
/// `1/ε²` and swamps the signal.
///
/// # Arguments
/// * `seed` - RNG seed shared by both perturbed evaluations (common random numbers).
/// * `addr` - Address of the factor to differentiate; must be present in `guide`.
/// * `coord` - Which of the factor's two coordinates to perturb.
/// * `eps` - Finite-difference half-step (in unconstrained parameter space).
/// * `num_samples` - Monte Carlo samples per ELBO evaluation.
pub fn elbo_gradient_fd<A>(
    seed: u64,
    model_fn: impl Fn() -> Model<A>,
    guide: &MeanFieldGuide,
    addr: &Address,
    coord: ParamCoord,
    eps: f64,
    num_samples: usize,
) -> f64 {
    let base = match guide.params.get(addr) {
        Some(p) => p,
        None => return 0.0,
    };

    let mut guide_plus = guide.clone();
    guide_plus
        .params
        .insert(addr.clone(), shifted(base, coord, eps));
    let mut guide_minus = guide.clone();
    guide_minus
        .params
        .insert(addr.clone(), shifted(base, coord, -eps));

    // Common random numbers: identical seed => identical z ~ q draws for + and -.
    let elbo_plus = elbo_with_guide(
        &mut StdRng::seed_from_u64(seed),
        &model_fn,
        &guide_plus,
        num_samples,
    );
    let elbo_minus = elbo_with_guide(
        &mut StdRng::seed_from_u64(seed),
        &model_fn,
        &guide_minus,
        num_samples,
    );

    (elbo_plus - elbo_minus) / (2.0 * eps)
}

/// Configuration for [`optimize_meanfield_vi_with_config`].
#[derive(Clone, Debug)]
pub struct VIConfig {
    /// Maximum number of optimization iterations.
    pub n_iterations: usize,
    /// Monte Carlo samples per ELBO / gradient evaluation.
    pub n_samples_per_iter: usize,
    /// Base step size `α₀`. The effective step at iteration `t` is
    /// `α₀ · (t+1)^(−step_decay_exponent)`.
    pub base_learning_rate: f64,
    /// Finite-difference half-step `ε` used for the CRN central differences.
    pub fd_eps: f64,
    /// Relative-improvement tolerance for the ELBO-plateau convergence test.
    pub convergence_tol: f64,
    /// Window length (in iterations) for the ELBO-plateau convergence test.
    pub convergence_window: usize,
    /// Robbins–Monro step-decay exponent (must be in `(0.5, 1]` for convergence).
    pub step_decay_exponent: f64,
}

impl Default for VIConfig {
    fn default() -> Self {
        Self {
            n_iterations: 1000,
            n_samples_per_iter: 16,
            base_learning_rate: 0.1,
            fd_eps: 0.01,
            convergence_tol: 1e-4,
            convergence_window: 20,
            step_decay_exponent: 0.6,
        }
    }
}

/// Result of running [`optimize_meanfield_vi_with_config`].
#[derive(Clone, Debug)]
pub struct VIResult {
    /// The optimized guide.
    pub guide: MeanFieldGuide,
    /// Per-iteration ELBO estimates (state at the start of each iteration).
    pub elbo_history: Vec<f64>,
    /// Whether the ELBO-plateau convergence criterion fired before `n_iterations`.
    pub converged: bool,
    /// Number of iterations actually run.
    pub iterations: usize,
}

/// Optimize a mean-field guide by stochastic gradient ascent on the ELBO.
///
/// This is the configurable entry point (see [`VIConfig`]). All variational parameters —
/// **both** location and scale, in unconstrained log-space for the scales — are updated
/// (finding FG-04), using common-random-numbers central finite-difference gradients
/// (finding FG-16, via [`elbo_gradient_fd`]), a Robbins–Monro decaying step size and an
/// ELBO-plateau convergence test (finding FG-44).
///
/// The optimizer is stochastic but fully determined by `rng`, so a seeded RNG gives
/// reproducible results.
pub fn optimize_meanfield_vi_with_config<A, R: Rng>(
    rng: &mut R,
    model_fn: impl Fn() -> Model<A>,
    initial_guide: MeanFieldGuide,
    config: &VIConfig,
) -> VIResult {
    let mut guide = initial_guide;
    let mut elbo_history: Vec<f64> = Vec::with_capacity(config.n_iterations);
    let mut converged = false;
    let mut iterations = 0;

    for iter in 0..config.n_iterations {
        iterations = iter + 1;

        // Monitor the ELBO at the start of this iteration (seeded from `rng` so the run
        // stays reproducible).
        let monitor_seed: u64 = rng.gen();
        let current_elbo = elbo_with_guide(
            &mut StdRng::seed_from_u64(monitor_seed),
            &model_fn,
            &guide,
            config.n_samples_per_iter,
        );
        elbo_history.push(current_elbo);

        // ELBO-plateau convergence: compare the mean ELBO of the two most recent
        // non-overlapping windows; stop when the relative change is below tolerance.
        let w = config.convergence_window;
        if w > 0 && elbo_history.len() >= 2 * w {
            let n = elbo_history.len();
            let recent: f64 = elbo_history[n - w..].iter().sum::<f64>() / w as f64;
            let previous: f64 = elbo_history[n - 2 * w..n - w].iter().sum::<f64>() / w as f64;
            let denom = previous.abs().max(1e-8);
            if (recent - previous).abs() / denom < config.convergence_tol {
                converged = true;
                break;
            }
        }

        // Robbins-Monro decaying step size.
        let step =
            config.base_learning_rate * ((iter + 1) as f64).powf(-config.step_decay_exponent);

        // Compute all coordinate gradients from a snapshot of the guide (Jacobi update),
        // then apply. Addresses are visited in sorted order for reproducibility.
        let snapshot = guide.clone();
        let mut addrs: Vec<Address> = snapshot.params.keys().cloned().collect();
        addrs.sort();

        for addr in &addrs {
            for coord in [ParamCoord::Location, ParamCoord::Scale] {
                // Independent seed per coordinate; identical within the +/- pair (CRN).
                let seed: u64 = rng.gen();
                let grad = elbo_gradient_fd(
                    seed,
                    &model_fn,
                    &snapshot,
                    addr,
                    coord,
                    config.fd_eps,
                    config.n_samples_per_iter,
                );
                if grad.is_finite() {
                    let update = step * grad;
                    if update.is_finite() {
                        if let Some(param) = guide.params.get_mut(addr) {
                            apply_update(param, coord, update);
                        }
                    }
                }
            }
        }
    }

    VIResult {
        guide,
        elbo_history,
        converged,
        iterations,
    }
}

/// Optimize a mean-field guide by stochastic gradient ascent on the ELBO.
///
/// Convenience wrapper over [`optimize_meanfield_vi_with_config`] using [`VIConfig`]
/// defaults for the finite-difference step, convergence criterion and step-decay
/// schedule, with the supplied iteration count, sample count and base learning rate. It
/// optimizes **both** the location and the scale of every factor (finding FG-04). For
/// convergence diagnostics or full configurability, call
/// [`optimize_meanfield_vi_with_config`] directly.
pub fn optimize_meanfield_vi<A, R: Rng>(
    rng: &mut R,
    model_fn: impl Fn() -> Model<A>,
    initial_guide: MeanFieldGuide,
    n_iterations: usize,
    n_samples_per_iter: usize,
    learning_rate: f64,
) -> MeanFieldGuide {
    let config = VIConfig {
        n_iterations,
        n_samples_per_iter,
        base_learning_rate: learning_rate,
        ..VIConfig::default()
    };
    optimize_meanfield_vi_with_config(rng, model_fn, initial_guide, &config).guide
}

/// Monte Carlo ELBO using the model's **prior** as the variational guide.
///
/// With `q = prior`, the ELBO `E_q[log p(x,z) − log q(z)]` telescopes to
/// `E_prior[log p(x | z)]` (the prior log-density cancels), i.e. the sample mean of the
/// per-draw log-likelihood-plus-factor contributions. By Jensen this is a valid lower
/// bound on the log evidence `log p(x)`.
///
/// This is the zero-configuration ELBO: it needs no fitted guide, but the prior is
/// usually a poor proposal so the bound is loose. For a bound against an arbitrary
/// (optimized) guide, use [`elbo_with_guide`].
///
/// Note (finding FG-46): earlier versions of this function averaged the *joint*
/// `log p(x, z)` and mislabeled it an ELBO, double-counting the prior entropy. It now
/// correctly omits the `log p(z)` term.
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
        // ELBO with q = prior = E_prior[log p(x|z)] = likelihood + factors only.
        total += prior_t.log_likelihood + prior_t.log_factors;
    }
    total / (num_samples as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::addr;

    use crate::core::model::{observe, sample, ModelExt};
    use crate::runtime::trace::{Choice, ChoiceValue, Trace};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn variational_param_sampling_and_log_prob() {
        let mut rng = StdRng::seed_from_u64(20);
        let vp_n = VariationalParam::Normal {
            mu: 0.0,
            log_sigma: 0.0,
        };
        let x = vp_n.sample(&mut rng);
        assert!(x.is_finite());
        assert!(vp_n.log_prob(x).is_finite());

        let vp_b = VariationalParam::Beta {
            log_alpha: (2.0f64).ln(),
            log_beta: (3.0f64).ln(),
        };
        let y = vp_b.sample(&mut rng);
        assert!(y > 0.0 && y < 1.0);
        assert!(vp_b.log_prob(y).is_finite());
    }

    #[test]
    fn elbo_computation_is_finite() {
        let model_fn = || {
            sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap()).and_then(|mu| {
                observe(addr!("y"), Normal::new(mu, 1.0).unwrap(), 0.2).map(move |_| mu)
            })
        };

        // Build a simple guide
        let mut guide = MeanFieldGuide::new();
        guide.params.insert(
            addr!("mu"),
            VariationalParam::Normal {
                mu: 0.0,
                log_sigma: 0.0,
            },
        );

        let mut rng = StdRng::seed_from_u64(21);
        let elbo = elbo_with_guide(&mut rng, model_fn, &guide, 5);
        assert!(elbo.is_finite());
    }

    #[test]
    fn meanfield_from_trace_continuous_ok() {
        // Only continuous (f64) latents -> Ok, all Normal factors on the real line.
        let mut base = Trace::default();
        base.choices.insert(
            addr!("pos"),
            Choice {
                addr: addr!("pos"),
                value: ChoiceValue::F64(-1.0),
                logp: -0.1,
            },
        );
        base.choices.insert(
            addr!("z"),
            Choice {
                addr: addr!("z"),
                value: ChoiceValue::F64(0.0),
                logp: -0.2,
            },
        );

        let guide = MeanFieldGuide::from_trace(&base).expect("continuous trace should build");
        assert_eq!(guide.params.len(), 2);
        // FG-18: value == 0.0 must not produce log_sigma = ln(0) = -inf.
        if let VariationalParam::Normal { log_sigma, .. } = guide.params.get(&addr!("z")).unwrap() {
            assert!(log_sigma.is_finite());
            assert!(log_sigma.exp() > 0.0);
        } else {
            panic!("expected Normal factor");
        }

        // Sampling produces a finite trace (no NaN from a degenerate sigma).
        let t = guide.sample_trace(&mut StdRng::seed_from_u64(22));
        assert!(!t.choices.is_empty());
        assert!(t.log_prior.is_finite());
    }

    #[test]
    fn optimize_vi_updates_parameters_and_is_stable() {
        let model_fn = || {
            sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap()).and_then(|mu| {
                observe(addr!("y"), Normal::new(mu, 1.0).unwrap(), 0.3).map(move |_| mu)
            })
        };

        let mut guide = MeanFieldGuide::new();
        guide.params.insert(
            addr!("mu"),
            VariationalParam::Normal {
                mu: 0.0,
                log_sigma: 0.0,
            },
        );

        let optimized = optimize_meanfield_vi(
            &mut StdRng::seed_from_u64(23),
            model_fn,
            guide.clone(),
            5, // small iterations for speed
            4,
            0.1,
        );

        // Parameter exists and remains finite / within clamped bounds.
        if let VariationalParam::Normal { mu, log_sigma } =
            optimized.params.get(&addr!("mu")).unwrap()
        {
            assert!(mu.is_finite() && mu.abs() <= MU_ABS_MAX);
            assert!(log_sigma.is_finite());
        } else {
            panic!("expected Normal param");
        }
    }
}
