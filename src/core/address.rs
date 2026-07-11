#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/docs/core/address.md"))]
use std::fmt::{Display, Formatter};

/// A unique identifier for random variables and observation sites in probabilistic models.
/// Addresses serve as stable names for probabilistic choices, enabling conditioning, inference, and replay.
/// They are implemented as wrapped strings with ordering and hashing support for use in collections.
///
/// # Index-separator encoding (collision-free)
///
/// Indexed addresses built with `addr!(name, index)` are stored as the string
/// `"{name}#{index}"`, using `'#'` as the separator between the name and its
/// index. To guarantee that two *syntactically distinct* `addr!` calls can never
/// produce the same [`Address`], any literal `'#'` (and any literal `'\'`) that
/// appears **inside** a `name` or `index` segment is escaped when the address is
/// built: `'\' -> "\\"` and `'#' -> "\#"`. The separator itself is the only
/// *unescaped* `'#'` in the stored string.
///
/// This makes the encoding injective, so for example:
///
/// - `addr!("a#1")` stores `"a\#1"` (the literal `'#'` is escaped) — a plain name,
/// - `addr!("a", 1)` stores `"a#1"` (an unescaped separator) — a name with index,
///
/// and the two are therefore **distinct** addresses. Likewise
/// `addr!("a", "b#3")` (`"a#b\#3"`) and `addr!("a#b", 3)` (`"a\#b#3"`) do not
/// collide. Names that contain neither `'#'` nor `'\'` are stored verbatim, so
/// the common case (e.g. `addr!("mu")` -> `"mu"`, `addr!("x", 3)` -> `"x#3"`)
/// is unchanged and `Display` stays human-readable.
///
/// Example:
/// ```rust
/// use fugue::*;
/// // Create addresses using the addr! macro
/// let addr1 = addr!("parameter");
/// let addr2 = addr!("data", 5);
/// // A literal '#' in a name never aliases an indexed address:
/// assert_ne!(addr!("a#1"), addr!("a", 1));
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

/// The reserved separator placed between a name and its index inside an
/// [`Address`] string built by `addr!(name, index)`.
pub const ADDR_INDEX_SEP: char = '#';

/// Escape a single address segment (a name or an index) so that a literal
/// occurrence of the reserved separator [`ADDR_INDEX_SEP`] (`'#'`) can never be
/// confused with the real separator, and so the escape character `'\'` itself is
/// unambiguous.
///
/// The escaping is the standard, injective backslash scheme (`'\' -> "\\"`,
/// `'#' -> "\#"`). Segments that contain neither character are returned verbatim
/// (the common case), so no allocation-visible change occurs for ordinary names.
///
/// This is an implementation detail used by the `addr!` and `scoped_addr!`
/// macros; it is public only so those macros can expand to it.
#[doc(hidden)]
pub fn escape_addr_segment(segment: &str) -> String {
    if segment.contains('\\') || segment.contains('#') {
        let mut out = String::with_capacity(segment.len() + 4);
        for ch in segment.chars() {
            match ch {
                '\\' => out.push_str("\\\\"),
                '#' => out.push_str("\\#"),
                other => out.push(other),
            }
        }
        out
    } else {
        segment.to_string()
    }
}

/// Build the backing string for a plain (unindexed) address, escaping the
/// reserved separator inside the name. Used by `addr!(name)`.
#[doc(hidden)]
pub fn make_name(name: impl Display) -> String {
    escape_addr_segment(&name.to_string())
}

/// Build the backing string for an indexed address `"{name}#{index}"`, escaping
/// the reserved separator inside both segments so the encoding is injective.
/// Used by `addr!(name, index)`.
#[doc(hidden)]
pub fn make_indexed(name: impl Display, index: impl Display) -> String {
    format!(
        "{}{}{}",
        escape_addr_segment(&name.to_string()),
        ADDR_INDEX_SEP,
        escape_addr_segment(&index.to_string())
    )
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
        $crate::core::address::Address($crate::core::address::make_name($name))
    };
    ($name:expr, $i:expr) => {
        $crate::core::address::Address($crate::core::address::make_indexed($name, $i))
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

    // Regression for FG-26 / FG-52: the `addr!` index-separator scheme must be
    // collision-free. A literal '#' inside a name is escaped ("\#"), while the
    // separator between name and index is an unescaped '#', so distinct calls
    // can never alias to the same backing string.
    #[test]
    fn addr_indexed_and_literal_hash_do_not_alias() {
        // The historical footgun: both used to produce "x#3".
        let indexed = addr!("x", 3);
        let literal_hash = addr!("x#3");
        assert_ne!(indexed, literal_hash);
        assert_eq!(indexed.0, "x#3");
        assert_eq!(literal_hash.0, "x\\#3");

        // The auditor's second example: addr!("a", "b#3") vs addr!("a#b", 3).
        let a = addr!("a", "b#3");
        let b = addr!("a#b", 3);
        assert_ne!(a, b);
        assert_eq!(a.0, "a#b\\#3");
        assert_eq!(b.0, "a\\#b#3");

        // The backslash escape character is itself escaped so it cannot forge
        // a separator boundary.
        assert_ne!(addr!("a\\", 1), addr!("a\\#1"));
    }

    // Regression for FG-26 / FG-52: the encoding is injective, so a name/index
    // pair that could previously collide via a shared '#' now stays distinct.
    #[test]
    fn addr_encoding_is_injective_across_hash_placements() {
        // (name = "a#", index = "b") vs (name = "a", index = "#b").
        // Under a naive doubling scheme these both collapse to "a###b"; the
        // backslash scheme keeps them apart.
        let left = addr!("a#", "b");
        let right = addr!("a", "#b");
        assert_ne!(left, right);
        assert_eq!(left.0, "a\\##b");
        assert_eq!(right.0, "a#\\#b");
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
