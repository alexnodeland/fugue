pub mod core;
pub mod inference;
pub mod macros;
pub mod runtime;

pub use core::address::Address;
// `addr!` macro is exported at the crate root via #[macro_export]
pub use core::distribution::{DistributionF64, Exponential, LogNormal, Normal, Uniform, Bernoulli, Categorical, Beta, Gamma, Binomial, Poisson};
pub use core::model::{factor, observe, pure, sample, Model, ModelExt, zip, sequence_vec, traverse_vec, guard};
pub use runtime::handler::Handler;
pub use runtime::interpreters::{PriorHandler, ReplayHandler, ScoreGivenTrace};
pub use runtime::trace::{Choice, ChoiceValue, Trace};
