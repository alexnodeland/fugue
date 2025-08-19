//! Addressing and site naming utilities.
//!
//! Addresses uniquely identify random choices and observation sites within a model.
//! The `addr!` macro provides a concise, stable way to create addresses from human
//! readable names (and optional indices), which helps with tracing and replay.
//!
//! Example:
//!
//! ```rust
//! use monadic_ppl::*;
//! let a = addr!("mu");
//! let b = addr!("mu", 0);
//! assert_ne!(a, b);
//! ```
use std::fmt::{Display, Formatter};
#[derive(Clone, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct Address(pub String);
impl Display for Address {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
#[macro_export]
macro_rules! addr {
    ($name:expr) => {
        $crate::core::address::Address($name.to_string())
    };
    ($name:expr, $i:expr) => {
        $crate::core::address::Address(format!("{}#{}", $name, $i))
    };
}
