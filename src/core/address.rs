//! Addressing and site naming utilities for probabilistic programs.
//!
//! Addresses are crucial for probabilistic programming as they uniquely identify
//! random choices and observation sites within a model. This enables:
//! - **Conditioning**: Observing specific values at named sites
//! - **Inference**: Tracking which random variables to infer
//! - **Replay**: Reproducing exact execution paths from recorded traces
//! - **Debugging**: Understanding model structure and execution flow
//!
//! The `addr!` macro provides a concise, stable way to create addresses from human-readable
//! names with optional indices for handling collections and repeated structures.
//!
//! ## Address Creation
//!
//! ```rust
//! use fugue::*;
//!
//! // Simple named address
//! let mu_addr = addr!("mu");
//!
//! // Indexed address for collections
//! let data_addr = addr!("data", 0);
//!
//! // Addresses are unique
//! assert_ne!(addr!("mu"), addr!("mu", 0));
//! assert_ne!(addr!("x", 1), addr!("x", 2));
//! ```
//!
//! ## Best Practices
//!
//! - Use descriptive names that reflect the semantic meaning
//! - Use indices for repeated structures (loops, arrays, etc.)
//! - Keep address names consistent across model runs for reproducibility
//! - Avoid dynamic address generation in inference loops
use std::fmt::{Display, Formatter};
/// A unique identifier for random variables and observation sites in probabilistic models.
///
/// Addresses serve as stable names for probabilistic choices, enabling conditioning,
/// inference, and replay. They are implemented as wrapped strings with ordering
/// and hashing support for use in collections.
///
/// # Examples
///
/// ```rust
/// use fugue::*;
///
/// // Create addresses using the addr! macro
/// let addr1 = addr!("parameter");
/// let addr2 = addr!("data", 5);
///
/// // Addresses can be compared and used in collections
/// use std::collections::HashMap;
/// let mut map = HashMap::new();
/// map.insert(addr1, 1.0);
/// map.insert(addr2, 2.0);
/// ```
#[derive(Clone, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct Address(pub String);
impl Display for Address {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
/// Create an address for naming random variables and observation sites.
///
/// This macro provides a convenient way to create `Address` instances with
/// human-readable names and optional indices. The macro supports two forms:
///
/// - `addr!("name")` - Simple named address
/// - `addr!("name", index)` - Indexed address using "name#index" format
///
/// # Examples
///
/// ```rust
/// use fugue::*;
///
/// // Simple addresses
/// let mu = addr!("mu");
/// let sigma = addr!("sigma");
///
/// // Indexed addresses for collections
/// let data_0 = addr!("data", 0);
/// let data_1 = addr!("data", 1);
///
/// // Use in models
/// let model = sample(addr!("x"), Normal { mu: 0.0, sigma: 1.0 })
///     .bind(|x| {
///         // Index can be dynamic
///         let i = 42;
///         sample(addr!("y", i), Normal { mu: x, sigma: 0.1 })
///     });
/// ```
#[macro_export]
macro_rules! addr {
    ($name:expr) => {
        $crate::core::address::Address($name.to_string())
    };
    ($name:expr, $i:expr) => {
        $crate::core::address::Address(format!("{}#{}", $name, $i))
    };
}
