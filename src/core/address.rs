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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeSet, HashSet};

    #[test]
    fn display_formats_inner_string() {
        let a = Address("alpha".to_string());
        assert_eq!(a.to_string(), "alpha");
    }

    #[test]
    fn addr_macro_basic_and_indexed() {
        let a = addr!("x");
        assert_eq!(a.0, "x");

        let b = addr!("x", 3);
        assert_eq!(b.0, "x#3");
    }

    #[test]
    fn equality_hash_and_ordering() {
        let a1 = Address("x".into());
        let a2 = Address("x".into());
        let b = Address("y".into());

        // Eq/Hash
        let mut set = HashSet::new();
        set.insert(a1.clone());
        set.insert(a2.clone());
        set.insert(b.clone());
        assert_eq!(set.len(), 2);

        // Ord/PartialOrd via BTreeSet (lexicographic)
        let mut bset = BTreeSet::new();
        bset.insert(b);
        bset.insert(a1);
        // Expect alphabetical order: "x" comes after "y"? No, "x" < "y"
        let ordered: Vec<String> = bset.into_iter().map(|a| a.0).collect();
        assert_eq!(ordered, vec!["x".to_string(), "y".to_string()]);
    }
}
