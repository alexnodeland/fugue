//! # Fugue: A Production-Ready Monadic Probabilistic Programming Library
//!
//! Fugue is a robust, numerically stable, monadic probabilistic programming library for Rust
//! that enables writing probabilistic programs by composing `Model` values in direct style and
//! running them with pluggable interpreters and state-of-the-art inference routines.
//!
//! ## Overview
//!
//! Fugue provides:
//! - **Models**: Monadic composition of probabilistic programs using `Model<T>`
//! - **Distributions**: Numerically stable probability distributions with validation
//! - **Inference**: Multiple algorithms with theoretical guarantees (ABC, MCMC, SMC, VI)
//! - **Runtime**: Efficient handlers and interpreters with memory optimization
//! - **Traces**: Recording and replaying of probabilistic program execution
//! - **Diagnostics**: Comprehensive convergence assessment and validation tools
//!
//! ## Quick Start
//!
//! ```rust
//! use fugue::*;
//! use rand::rngs::StdRng;
//! use rand::SeedableRng;
//!
//! // Define a Bayesian model with proper error handling
//! fn gaussian_mean_model(observation: f64) -> Result<Model<f64>, FugueError> {
//!     let prior = Normal::new(0.0, 5.0)?; // Validated construction
//!     let likelihood = Normal::new(observation, 1.0)?;
//!     
//!     Ok(sample(addr!("mu"), prior)
//!         .bind(move |mu| {
//!             observe(addr!("y"), Normal { mu, sigma: 1.0 }, observation)
//!                 .bind(move |_| pure(mu))
//!         }))
//! }
//!
//! // Run with improved MCMC
//! let model = gaussian_mean_model(2.7)?;
//! let mut rng = StdRng::seed_from_u64(42);
//! let samples = adaptive_mcmc_chain(
//!     &mut rng,
//!     || gaussian_mean_model(2.7).unwrap(),
//!     1000, // samples
//!     500,  // warmup
//! );
//!
//! // Comprehensive diagnostics
//! let mu_samples: Vec<f64> = samples.iter()
//!     .filter_map(|(_, trace)| trace.choices.get(&addr!("mu")))
//!     .filter_map(|choice| match choice.value {
//!         ChoiceValue::F64(mu) => Some(mu),
//!         _ => None,
//!     })
//!     .collect();
//!
//! let ess = effective_sample_size_mcmc(&mu_samples);
//! let geweke = geweke_diagnostic(&mu_samples);
//! println!("ESS: {:.1}, Geweke: {:.3}", ess, geweke);
//! # Ok::<(), FugueError>(())
//! ```
//!
//! ## Model Composition
//!
//! Models can be composed using monadic operations:
//!
//! ```rust
//! use fugue::*;
//!
//! // Combine multiple random variables with validation
//! let model = sample(addr!("x"), Normal { mu: 0.0, sigma: 1.0 })
//!     .bind(|x| {
//!         sample(addr!("y"), Normal { mu: x, sigma: 0.5 })
//!             .map(move |y| (x, y))
//!     });
//! ```
//!
//! ## Inference Methods
//!
//! Fugue supports several state-of-the-art inference algorithms:
//!
//! - **ABC (Approximate Bayesian Computation)**: Likelihood-free inference with stable distance functions
//! - **MCMC**: Metropolis-Hastings with diminishing adaptation and convergence guarantees
//! - **SMC (Sequential Monte Carlo)**: Particle filtering with proper weight normalization
//! - **VI (Variational Inference)**: Mean-field approximation with reparameterization gradients
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
//! - **Diagnostics**: Assess convergence and validate inference quality
//!
//! ## Production Features
//!
//! - **Numerical Stability**: All operations handle extreme values gracefully
//! - **Memory Efficiency**: Copy-on-write traces and memory pooling
//! - **Error Handling**: Comprehensive error types with context
//! - **Validation**: Statistical tests against known analytical results
//! - **Performance**: Optimized algorithms with theoretical guarantees
//!
//! ## Examples
//!
//! See the `examples/` directory for complete working examples including:
//! - Gaussian mean estimation with comprehensive diagnostics
//! - Mixture models with component selection
//! - Exponential hazard models for survival analysis
//! - Conjugate Beta-Binomial models

pub mod core;
pub mod error;
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
pub use core::numerical::{log1p_exp, log_sum_exp, normalize_log_probs, safe_ln};
pub use error::{FugueError, FugueResult, Validate};
pub use inference::abc::{
    abc_rejection, abc_scalar_summary, abc_smc, DistanceFunction, EuclideanDistance,
};
pub use inference::diagnostics::{print_diagnostics, r_hat, summarize_parameter, ParameterSummary};
pub use inference::mcmc_utils::{
    effective_sample_size_mcmc, geweke_diagnostic, DiminishingAdaptation,
};
pub use inference::mh::{adaptive_mcmc_chain, adaptive_single_site_mh, AdaptiveScales};
pub use inference::smc::{
    adaptive_smc, effective_sample_size, Particle, ResamplingMethod, SMCConfig,
};
pub use inference::validation::{
    ks_test_distribution, test_conjugate_normal_model, ValidationResult,
};
pub use inference::vi::{elbo_with_guide, optimize_meanfield_vi, MeanFieldGuide, VariationalParam};
pub use runtime::memory::{CowTrace, PooledPriorHandler, TraceBuilder, TracePool};
