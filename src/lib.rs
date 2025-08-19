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
