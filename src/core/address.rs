#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/docs/core/address.md"))]
use std::fmt::{Display, Formatter};
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::sync::Arc;

/// A unique identifier for random variables and observation sites in probabilistic models.
/// Addresses serve as stable names for probabilistic choices, enabling conditioning, inference, and replay.
///
/// # Representation (FG-05)
///
/// `Address` is backed by an `Arc<str>` together with a **precomputed** 64-bit
/// hash of that string. This makes the two operations that dominate inference
/// bookkeeping cheap:
///
/// - **Clone** is an atomic reference-count bump plus a `u64` copy — no heap
///   allocation and no string copy. Concrete handlers (`PriorHandler`,
///   `ScoreGivenTrace`, …) clone every address twice per sample site (once as the
///   `BTreeMap` key, once inside the stored `Choice`), and single-site MH clones
///   the whole trace several times per step, so cheap cloning removes what the
///   audit measured as the per-iteration allocation hot spot.
/// - **Hash** writes the cached `u64` directly instead of re-hashing the string
///   on every `HashMap` probe. Equality still compares the underlying `str`
///   (after a fast hash pre-check), so hash collisions remain correct.
///
/// Ordering (`Ord`/`PartialOrd`) compares the underlying `str` lexicographically,
/// preserving the stable, human-meaningful `BTreeMap` iteration order that traces
/// rely on. `Display` and `Deref<Target = str>` are preserved so downstream code
/// that formatted or string-sliced an address keeps compiling.
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
#[derive(Clone, Debug)]
pub struct Address {
    /// Reference-counted, immutable backing string. Cloning shares this buffer.
    repr: Arc<str>,
    /// Precomputed hash of `repr`, written directly by [`Hash`] so that hashing
    /// an address never re-scans the string.
    hash: u64,
}

/// Compute the cached hash for an address's backing string.
///
/// Uses [`std::collections::hash_map::DefaultHasher`], whose keys are fixed, so
/// the value is deterministic for a given string within and across runs of the
/// same build. The value is only ever compared for equality and fed to another
/// hasher via [`Hasher::write_u64`], so its only requirements are determinism and
/// good dispersion — both of which SipHash satisfies.
#[inline]
fn compute_address_hash(s: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}

impl Address {
    /// Construct an address from any string-like value.
    ///
    /// The backing string is moved into an `Arc<str>` once, and its hash is
    /// computed once, here at construction. All later clones are allocation-free.
    #[inline]
    pub fn new(name: impl Into<Arc<str>>) -> Self {
        let repr: Arc<str> = name.into();
        let hash = compute_address_hash(&repr);
        Address { repr, hash }
    }

    /// Borrow the underlying string slice.
    #[inline]
    pub fn as_str(&self) -> &str {
        &self.repr
    }
}

impl Display for Address {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.repr)
    }
}

impl Deref for Address {
    type Target = str;
    #[inline]
    fn deref(&self) -> &str {
        &self.repr
    }
}

impl Hash for Address {
    /// Write the precomputed hash rather than re-hashing the string on every
    /// `HashMap` probe (FG-05).
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.hash);
    }
}

impl PartialEq for Address {
    /// Equality compares the underlying `str`; the cached hash is used only as a
    /// fast reject so distinct strings that collide in the hash still compare
    /// unequal.
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.hash == other.hash && self.repr == other.repr
    }
}

impl Eq for Address {}

impl PartialOrd for Address {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Address {
    /// Lexicographic ordering on the backing string, preserving stable
    /// `BTreeMap` iteration order.
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.as_str().cmp(other.as_str())
    }
}

impl From<String> for Address {
    #[inline]
    fn from(s: String) -> Self {
        Address::new(s)
    }
}

impl From<&str> for Address {
    #[inline]
    fn from(s: &str) -> Self {
        Address::new(s)
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
        $crate::core::address::Address::new($crate::core::address::make_name($name))
    };
    ($name:expr, $i:expr) => {
        $crate::core::address::Address::new($crate::core::address::make_indexed($name, $i))
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeSet, HashSet};

    #[test]
    fn display_formats_inner_string() {
        let a = Address::new("alpha");
        assert_eq!(a.to_string(), "alpha");
    }

    // Regression for FG-05: an Address caches a hash of its backing string, so
    // the `Hash` impl must agree with `Eq` (equal addresses hash equally) and
    // clones must remain equal and share the backing buffer.
    #[test]
    fn cached_hash_is_consistent_with_eq_and_clone() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        fn h(a: &Address) -> u64 {
            let mut hasher = DefaultHasher::new();
            a.hash(&mut hasher);
            hasher.finish()
        }

        let a = addr!("mu", 7);
        let b = addr!("mu", 7);
        assert_eq!(a, b);
        assert_eq!(h(&a), h(&b), "equal addresses must hash equally");

        // Clone is allocation-free (shares the Arc) and stays equal.
        let c = a.clone();
        assert_eq!(a, c);
        assert!(Arc::ptr_eq(&a.repr, &c.repr));
        assert_eq!(h(&a), h(&c));

        // A different address hashes differently (with overwhelming probability)
        // and, more importantly, compares unequal.
        let d = addr!("mu", 8);
        assert_ne!(a, d);
    }

    #[test]
    fn addr_macro_basic_and_indexed() {
        let a = addr!("x");
        assert_eq!(a.as_str(), "x");

        let b = addr!("x", 3);
        assert_eq!(b.as_str(), "x#3");
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
        assert_eq!(indexed.as_str(), "x#3");
        assert_eq!(literal_hash.as_str(), "x\\#3");

        // The auditor's second example: addr!("a", "b#3") vs addr!("a#b", 3).
        let a = addr!("a", "b#3");
        let b = addr!("a#b", 3);
        assert_ne!(a, b);
        assert_eq!(a.as_str(), "a#b\\#3");
        assert_eq!(b.as_str(), "a\\#b#3");

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
        assert_eq!(left.as_str(), "a\\##b");
        assert_eq!(right.as_str(), "a#\\#b");
    }

    #[test]
    fn equality_hash_and_ordering() {
        let a1 = Address::new("x");
        let a2 = Address::new("x");
        let b = Address::new("y");

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
        let ordered: Vec<String> = bset.into_iter().map(|a| a.as_str().to_string()).collect();
        assert_eq!(ordered, vec!["x".to_string(), "y".to_string()]);
    }
}
