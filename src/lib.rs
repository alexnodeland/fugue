//! # Fugue: A Monadic Probabilistic Programming Library
//!
//! Fugue is a tiny, elegant, monadic probabilistic programming library for Rust that enables
//! writing probabilistic programs by composing `Model` values in direct style and running
//! them with pluggable interpreters and inference routines.
//!
//! ## Overview
//!
//! Fugue provides:
//! - **Models**: Monadic composition of probabilistic programs using `Model<T>`
//! - **Distributions**: Common probability distributions with sampling and log-density
//! - **Inference**: Multiple algorithms including ABC, MCMC, SMC, and Variational Inference
//! - **Runtime**: Pluggable handlers and interpreters for executing models
//! - **Traces**: Recording and replaying of probabilistic program execution
//!
//! ## Quick Start
//!
//! ```rust
//! use fugue::*;
//! use rand::rngs::StdRng;
//! use rand::SeedableRng;
//!
//! // Define a simple Bayesian model
//! fn gaussian_mean_model(observation: f64) -> Model<f64> {
//!     // Prior: normal distribution for the mean
//!     sample(addr!("mu"), Normal { mu: 0.0, sigma: 5.0 })
//!         .bind(move |mu| {
//!             // Likelihood: observe data given the mean
//!             observe(addr!("y"), Normal { mu, sigma: 1.0 }, observation)
//!                 .bind(move |_| pure(mu))
//!         })
//! }
//!
//! // Run the model
//! let model = gaussian_mean_model(2.7);
//! let mut rng = StdRng::seed_from_u64(42);
//! let (posterior_mean, trace) = runtime::handler::run(
//!     PriorHandler {
//!         rng: &mut rng,
//!         trace: Trace::default(),
//!     },
//!     model,
//! );
//!
//! println!("Posterior mean: {}", posterior_mean);
//! println!("Log weight: {}", trace.total_log_weight());
//! ```
//!
//! ## Model Composition
//!
//! Models can be composed using monadic operations:
//!
//! ```rust
//! use fugue::*;
//!
//! // Combine multiple random variables
//! let model = sample(addr!("x"), Normal { mu: 0.0, sigma: 1.0 })
//!     .bind(|x| {
//!         sample(addr!("y"), Normal { mu: x, sigma: 0.5 })
//!             .map(move |y| (x, y))
//!     });
//! ```
//!
//! ## Inference Methods
//!
//! Fugue supports several inference algorithms:
//!
//! - **ABC (Approximate Bayesian Computation)**: Likelihood-free inference
//! - **MCMC**: Metropolis-Hastings with adaptive scaling
//! - **SMC (Sequential Monte Carlo)**: Particle filtering with resampling
//! - **VI (Variational Inference)**: Mean-field variational inference
//!
//! See the [`inference`] module for detailed documentation and examples.
//!
//! ## Architecture
//!
//! Fugue is built around several key concepts:
//!
//! - **Models** ([`Model`]): Represent probabilistic computations
//! - **Addresses** ([`Address`]): Identify random variables for conditioning and inference
//! - **Handlers** ([`Handler`]): Interpret model execution (prior sampling, conditioning, etc.)
//! - **Traces** ([`Trace`]): Record execution history for replay and analysis
//!
//! ## Examples
//!
//! See the `examples/` directory for complete working examples including:
//! - Gaussian mean estimation
//! - Mixture models
//! - Exponential hazard models
//! - Conjugate Beta-Binomial models

pub mod core;
pub mod inference;
pub mod macros;
pub mod runtime;

pub use core::address::Address;
// `addr!` macro is exported at the crate root via #[macro_export]
pub use core::distribution::{
    Bernoulli, Beta, Binomial, Categorical, DistributionF64, Exponential, Gamma, LogNormal, Normal,
    Poisson, Uniform,
};
pub use core::model::{
    factor, guard, observe, pure, sample, sequence_vec, traverse_vec, zip, Model, ModelExt,
};
pub use runtime::handler::Handler;
pub use runtime::interpreters::{PriorHandler, ReplayHandler, ScoreGivenTrace};
pub use runtime::trace::{Choice, ChoiceValue, Trace};

// Re-export key inference methods
pub use inference::abc::{
    abc_rejection, abc_scalar_summary, abc_smc, DistanceFunction, EuclideanDistance,
};
pub use inference::diagnostics::{print_diagnostics, r_hat, summarize_parameter, ParameterSummary};
pub use inference::mh::{adaptive_mcmc_chain, adaptive_single_site_mh, AdaptiveScales};
pub use inference::smc::{
    adaptive_smc, effective_sample_size, Particle, ResamplingMethod, SMCConfig,
};
pub use inference::vi::{elbo_with_guide, optimize_meanfield_vi, MeanFieldGuide, VariationalParam};
