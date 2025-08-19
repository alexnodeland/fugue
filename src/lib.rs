pub mod core;
pub mod runtime;
pub mod inference;
pub mod macros;

pub use core::address::{addr, Address};
pub use core::distribution::Normal;
pub use core::model::{factor, observe, pure, sample, Model, ModelExt};
pub use runtime::handler::Handler;
pub use runtime::interpreters::{PriorHandler, ReplayHandler, ScoreGivenTrace};
pub use runtime::trace::{ChoiceF64, Trace};
