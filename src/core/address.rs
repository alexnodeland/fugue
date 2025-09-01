#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/docs/core/address.md"))]
use std::fmt::{Display, Formatter};

/// A unique identifier for random variables and observation sites in probabilistic models.
/// Addresses serve as stable names for probabilistic choices, enabling conditioning, inference, and replay.
/// They are implemented as wrapped strings with ordering and hashing support for use in collections.
///
/// Example:
/// ```rust
/// use fugue::*;
/// // Create addresses using the addr! macro
/// let addr1 = addr!("parameter");
/// let addr2 = addr!("data", 5);
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
/// This macro provides a convenient way to create `Address` instances with human-readable names and optional indices.
/// The macro supports two forms:
///
/// - `addr!("name")` - Simple named address
/// - `addr!("name", index)` - Indexed address using "name#index" format
///
/// Example:
/// ```rust
/// use fugue::*;
/// // Simple addresses
/// let mu = addr!("mu");
/// let sigma = addr!("sigma");
/// // Indexed addresses for collections
/// let data_0 = addr!("data", 0);
/// let data_1 = addr!("data", 1);
/// // Use in models
/// let model = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
///     .bind(|x| {
///         // Index can be dynamic
///         let i = 42;
///         sample(addr!("y", i), Normal::new(x, 0.1).unwrap())
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
