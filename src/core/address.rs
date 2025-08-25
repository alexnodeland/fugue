#![doc = include_str!("../../docs/api/core/address/README.md")]
use std::fmt::{Display, Formatter};

#[doc = include_str!("../../docs/api/core/address/address.md")]
#[derive(Clone, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct Address(pub String);
impl Display for Address {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[doc = include_str!("../../docs/api/core/address/addr.md")]
#[macro_export]
macro_rules! addr {
    ($name:expr) => {
        $crate::core::address::Address($name.to_string())
    };
    ($name:expr, $i:expr) => {
        $crate::core::address::Address(format!("{}#{}", $name, $i))
    };
}
