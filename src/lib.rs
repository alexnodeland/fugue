pub mod core;
pub mod runtime;
pub mod inference;
pub mod macros;

pub use core::address::Address;
// `addr!` macro is exported at the crate root via #[macro_export]
pub use core::distribution::Normal;
pub use core::model::{factor, observe, pure, sample, Model, ModelExt};
pub use runtime::handler::Handler;
pub use runtime::interpreters::{PriorHandler, ReplayHandler, ScoreGivenTrace};
pub use runtime::trace::{ChoiceF64, Trace};
