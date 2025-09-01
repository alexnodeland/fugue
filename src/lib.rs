// Copyright (c) 2025 Alex Nodeland
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0>
// or the MIT license <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your option.
// This file may not be copied, modified, or distributed except according to those terms.

#![doc = include_str!("../README.md")]
// Allow large error types for rich error context
#![allow(clippy::result_large_err)]

pub mod core;
pub mod error;
pub mod inference;
pub mod macros;
pub mod runtime;

pub use core::address::Address;
// `addr!` macro is exported at the crate root via #[macro_export]
pub use core::distribution::{
    Bernoulli, Beta, Binomial, Categorical, Distribution, Exponential, Gamma, LogNormal, Normal,
    Poisson, Uniform,
};
pub use core::model::{
    factor, guard, observe, pure, sample, sample_bool, sample_f64, sample_u64, sample_usize,
    sequence_vec, traverse_vec, zip, Model, ModelExt, SampleType,
};
pub use runtime::handler::Handler;
pub use runtime::interpreters::{
    PriorHandler, ReplayHandler, SafeReplayHandler, SafeScoreGivenTrace, ScoreGivenTrace,
};
pub use runtime::trace::{Choice, ChoiceValue, Trace};

// Re-export key inference methods
pub use core::numerical::{log1p_exp, log_sum_exp, normalize_log_probs, safe_ln};
pub use error::{ErrorCategory, ErrorCode, ErrorContext, FugueError, FugueResult, Validate};
pub use inference::abc::{
    abc_rejection, abc_scalar_summary, abc_smc, DistanceFunction, EuclideanDistance,
};
pub use inference::diagnostics::{
    extract_bool_values, extract_f64_values, extract_i64_values, extract_u64_values,
    extract_usize_values, print_diagnostics, r_hat_f64, summarize_f64_parameter, Diagnostics,
    ParameterSummary,
};
pub use inference::mcmc_utils::{
    effective_sample_size_mcmc, geweke_diagnostic, DiminishingAdaptation,
};
pub use inference::mh::{adaptive_mcmc_chain, adaptive_single_site_mh};
pub use inference::smc::{
    adaptive_smc, effective_sample_size, Particle, ResamplingMethod, SMCConfig,
};
pub use inference::validation::{
    ks_test_distribution, test_conjugate_normal_model, ValidationResult,
};
pub use inference::vi::{elbo_with_guide, optimize_meanfield_vi, MeanFieldGuide, VariationalParam};
pub use runtime::memory::{CowTrace, PooledPriorHandler, TraceBuilder, TracePool};
