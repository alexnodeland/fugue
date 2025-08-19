use std::fmt::{Display, Formatter};
#[derive(Clone, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct Address(pub String);
impl Display for Address { fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result { write!(f, "{}", self.0) } }
#[macro_export]
macro_rules! addr {
    ($name:expr) => { $crate::core::address::Address($name.to_string()) };
    ($name:expr, $i:expr) => { $crate::core::address::Address(format!("{}#{}", $name, $i)) };
}
