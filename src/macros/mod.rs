//! Macros for ergonomic probabilistic programming.
//!
//! Provides `prob!` for do-notation style model composition and `plate!` for
//! replication over ranges.

/// Do-notation style macro for monadic model composition.
///
/// Allows writing probabilistic programs in a more imperative style:
///
/// ```rust
/// use fugue::*;
///
/// let model = prob! {
///     let mu <- sample(addr!("mu"), Normal{mu: 0.0, sigma: 1.0});
///     let sigma <- sample(addr!("sigma"), LogNormal{mu: 0.0, sigma: 1.0});
///     observe(addr!("y"), Normal{mu, sigma}, 2.5);
///     pure((mu, sigma))
/// };
/// ```
#[macro_export]
macro_rules! prob {
    // Simple cases first
    ($e:expr) => { $e };

    // let var <- expr; rest
    (let $var:ident <- $expr:expr; $($rest:tt)*) => {
        $expr.bind(move |$var| prob!($($rest)*))
    };

    // let var = expr; rest
    (let $var:ident = $expr:expr; $($rest:tt)*) => {
        { let $var = $expr; prob!($($rest)*) }
    };

    // expr; rest
    ($expr:expr; $($rest:tt)*) => {
        $expr.bind(move |_| prob!($($rest)*))
    };
}

/// Plate notation for replicating models over ranges.
///
/// ```rust
/// use fugue::*;
///
/// let model = plate!(i in 0..10 => {
///     sample(addr!("x", i), Normal{mu: 0.0, sigma: 1.0})
/// });
/// ```
#[macro_export]
macro_rules! plate {
    ($var:ident in $range:expr => $body:expr) => {
        $crate::core::model::traverse_vec($range.collect::<Vec<_>>(), move |$var| $body)
    };
}

/// Enhanced address macro with scoping support.
#[macro_export]
macro_rules! scoped_addr {
    ($scope:expr, $name:expr) => {
        $crate::core::address::Address(format!("{}::{}", $scope, $name))
    };
    ($scope:expr, $name:expr, $($indices:expr),+) => {
        $crate::core::address::Address(format!("{}::{}#{}", $scope, $name, format!("{}", format_args!($($indices),+))))
    };
}
