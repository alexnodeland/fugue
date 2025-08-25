#![doc = include_str!("../../docs/api/macros/README.md")]

#[doc = include_str!("../../docs/api/macros/prob.md")]
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

#[doc = include_str!("../../docs/api/macros/plate.md")]
#[macro_export]
macro_rules! plate {
    ($var:ident in $range:expr => $body:expr) => {
        $crate::core::model::traverse_vec($range.collect::<Vec<_>>(), move |$var| $body)
    };
}

#[doc = include_str!("../../docs/api/macros/scoped_addr.md")]
#[macro_export]
macro_rules! scoped_addr {
    ($scope:expr, $name:expr) => {
        $crate::core::address::Address(format!("{}::{}", $scope, $name))
    };
    ($scope:expr, $name:expr, $($indices:expr),+) => {
        $crate::core::address::Address(format!("{}::{}#{}", $scope, $name, format!("{}", format_args!($($indices),+))))
    };
}
