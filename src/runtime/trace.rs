#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/docs/runtime/trace.md"))]

use crate::core::address::Address;
use crate::error::{FugueError, FugueResult};
use std::collections::BTreeMap;

/// Type-safe storage for values from different distribution types.
///
/// ChoiceValue enables traces to store values from any supported distribution
/// while maintaining type safety. Each variant corresponds to a distribution
/// return type, preventing runtime type errors.
///
/// Example:
/// ```rust
/// # use fugue::runtime::trace::ChoiceValue;
///
/// // Different value types from distributions
/// let continuous = ChoiceValue::F64(3.14159);  // Normal, Uniform, etc.
/// let discrete = ChoiceValue::U64(42);         // Poisson, Binomial
/// let categorical = ChoiceValue::Usize(2);     // Categorical selection
/// let binary = ChoiceValue::Bool(true);        // Bernoulli outcome
///
/// // Type-safe extraction
/// assert_eq!(continuous.as_f64(), Some(3.14159));
/// assert_eq!(discrete.as_u64(), Some(42));
/// assert_eq!(binary.as_bool(), Some(true));
///
/// // Type mismatches return None
/// assert_eq!(continuous.as_bool(), None);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub enum ChoiceValue {
    /// Floating-point value (continuous distributions).
    F64(f64),
    /// Signed integer value.
    I64(i64),
    /// Unsigned integer value (Poisson, Binomial counts).
    U64(u64),
    /// Array index value (Categorical choices).
    Usize(usize),
    /// Boolean value (Bernoulli outcomes).
    Bool(bool),
}
impl ChoiceValue {
    /// Try to extract an f64 value, returning None if the type doesn't match.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            ChoiceValue::F64(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to extract a bool value, returning None if the type doesn't match.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            ChoiceValue::Bool(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to extract a u64 value, returning None if the type doesn't match.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            ChoiceValue::U64(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to extract a usize value, returning None if the type doesn't match.
    pub fn as_usize(&self) -> Option<usize> {
        match self {
            ChoiceValue::Usize(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to extract an i64 value, returning None if the type doesn't match.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            ChoiceValue::I64(v) => Some(*v),
            _ => None,
        }
    }

    /// Get the type name as a string for error messages.
    pub fn type_name(&self) -> &'static str {
        match self {
            ChoiceValue::F64(_) => "f64",
            ChoiceValue::Bool(_) => "bool",
            ChoiceValue::U64(_) => "u64",
            ChoiceValue::Usize(_) => "usize",
            ChoiceValue::I64(_) => "i64",
        }
    }
}

/// A single recorded choice made during model execution.
///
/// Each Choice represents a random variable assignment at a specific address,
/// complete with the value chosen and its log-probability. Choices form the
/// building blocks of execution traces.
///
/// Example:
/// ```rust
/// # use fugue::*;
/// # use fugue::runtime::trace::{Choice, ChoiceValue};
///
/// // Choices are typically created by handlers during execution
/// let choice = Choice {
///     addr: addr!("theta"),
///     value: ChoiceValue::F64(1.5),
///     logp: -0.918, // log-probability under generating distribution
/// };
///
/// println!("Choice at {}: {:?} (logp: {:.3})",
///          choice.addr, choice.value, choice.logp);
///
/// // Extract the value with type safety
/// if let Some(val) = choice.value.as_f64() {
///     println!("Theta value: {:.3}", val);
/// }
/// ```
#[derive(Clone, Debug)]
pub struct Choice {
    /// Address where this choice was made.
    pub addr: Address,
    /// Value that was chosen.
    pub value: ChoiceValue,
    /// Log-probability of this value under the generating distribution.
    pub logp: f64,
}

/// Complete execution trace of a probabilistic model.
///
/// A Trace records the complete execution history of a probabilistic model,
/// including all choices made and accumulated log-weights from different sources.
/// This enables replay, scoring, and inference operations.
///
/// Example:
/// ```rust
/// # use fugue::*;
/// # use fugue::runtime::interpreters::PriorHandler;
/// # use rand::rngs::StdRng;
/// # use rand::SeedableRng;
///
/// // Execute a model and examine the trace
/// let model = sample(addr!("theta"), Normal::new(0.0, 1.0).unwrap())
///     .bind(|theta| observe(addr!("y"), Normal::new(theta, 0.5).unwrap(), 1.2)
///         .map(move |_| theta));
///
/// let mut rng = StdRng::seed_from_u64(42);
/// let (result, trace) = runtime::handler::run(
///     PriorHandler { rng: &mut rng, trace: Trace::default() },
///     model
/// );
///
/// // Examine trace components
/// println!("Sampled theta: {:.3}", result);
/// println!("Prior log-weight: {:.3}", trace.log_prior);
/// println!("Likelihood log-weight: {:.3}", trace.log_likelihood);
/// println!("Total log-weight: {:.3}", trace.total_log_weight());
///
/// // Type-safe value access
/// let theta_value = trace.get_f64(&addr!("theta")).unwrap();
/// assert_eq!(theta_value, result);
/// ```
#[derive(Clone, Debug, Default)]
pub struct Trace {
    /// Map from addresses to the choices made at those sites.
    pub choices: BTreeMap<Address, Choice>,
    /// Accumulated log-prior probability from all sampling sites.
    pub log_prior: f64,
    /// Accumulated log-likelihood from all observation sites.
    pub log_likelihood: f64,
    /// Accumulated log-weight from all factor statements.
    pub log_factors: f64,
}

impl Trace {
    /// Compute the total unnormalized log-probability of this execution.
    ///
    /// The total log-weight combines all three components (prior, likelihood, factors)
    /// and represents the unnormalized log-probability of this execution path.
    ///
    /// Example:
    /// ```rust
    /// # use fugue::runtime::trace::Trace;
    ///
    /// let trace = Trace {
    ///     log_prior: -1.5,
    ///     log_likelihood: -2.3,
    ///     log_factors: 0.8,
    ///     ..Default::default()
    /// };
    ///
    /// assert_eq!(trace.total_log_weight(), -3.0);
    /// ```
    pub fn total_log_weight(&self) -> f64 {
        self.log_prior + self.log_likelihood + self.log_factors
    }

    /// Type-safe accessor for f64 values in the trace.
    pub fn get_f64(&self, addr: &Address) -> Option<f64> {
        self.choices.get(addr)?.value.as_f64()
    }

    /// Type-safe accessor for bool values in the trace.
    pub fn get_bool(&self, addr: &Address) -> Option<bool> {
        self.choices.get(addr)?.value.as_bool()
    }

    /// Type-safe accessor for u64 values in the trace.
    pub fn get_u64(&self, addr: &Address) -> Option<u64> {
        self.choices.get(addr)?.value.as_u64()
    }

    /// Type-safe accessor for usize values in the trace.
    pub fn get_usize(&self, addr: &Address) -> Option<usize> {
        self.choices.get(addr)?.value.as_usize()
    }

    /// Type-safe accessor for i64 values in the trace.
    pub fn get_i64(&self, addr: &Address) -> Option<i64> {
        self.choices.get(addr)?.value.as_i64()
    }

    /// Type-safe accessor that returns a Result for better error handling.
    pub fn get_f64_result(&self, addr: &Address) -> FugueResult<f64> {
        let choice = self.choices.get(addr).ok_or_else(|| {
            FugueError::trace_error(
                "get_f64",
                Some(addr.clone()),
                "Address not found in trace",
                crate::error::ErrorCode::TraceAddressNotFound,
            )
        })?;

        choice
            .value
            .as_f64()
            .ok_or_else(|| FugueError::type_mismatch(addr.clone(), "f64", choice.value.type_name()))
    }

    /// Type-safe accessor that returns a Result for better error handling.
    pub fn get_bool_result(&self, addr: &Address) -> FugueResult<bool> {
        let choice = self.choices.get(addr).ok_or_else(|| {
            FugueError::trace_error(
                "get_bool",
                Some(addr.clone()),
                "Address not found in trace",
                crate::error::ErrorCode::TraceAddressNotFound,
            )
        })?;

        choice.value.as_bool().ok_or_else(|| {
            FugueError::type_mismatch(addr.clone(), "bool", choice.value.type_name())
        })
    }

    /// Type-safe accessor that returns a Result for better error handling.
    pub fn get_u64_result(&self, addr: &Address) -> FugueResult<u64> {
        let choice = self.choices.get(addr).ok_or_else(|| {
            FugueError::trace_error(
                "get_u64",
                Some(addr.clone()),
                "Address not found in trace",
                crate::error::ErrorCode::TraceAddressNotFound,
            )
        })?;

        choice
            .value
            .as_u64()
            .ok_or_else(|| FugueError::type_mismatch(addr.clone(), "u64", choice.value.type_name()))
    }

    /// Type-safe accessor that returns a Result for better error handling.
    pub fn get_usize_result(&self, addr: &Address) -> FugueResult<usize> {
        let choice = self.choices.get(addr).ok_or_else(|| {
            FugueError::trace_error(
                "get_usize",
                Some(addr.clone()),
                "Address not found in trace",
                crate::error::ErrorCode::TraceAddressNotFound,
            )
        })?;

        choice.value.as_usize().ok_or_else(|| {
            FugueError::type_mismatch(addr.clone(), "usize", choice.value.type_name())
        })
    }

    /// Type-safe accessor that returns a Result for better error handling.
    pub fn get_i64_result(&self, addr: &Address) -> FugueResult<i64> {
        let choice = self.choices.get(addr).ok_or_else(|| {
            FugueError::trace_error(
                "get_i64",
                Some(addr.clone()),
                "Address not found in trace",
                crate::error::ErrorCode::TraceAddressNotFound,
            )
        })?;

        choice
            .value
            .as_i64()
            .ok_or_else(|| FugueError::type_mismatch(addr.clone(), "i64", choice.value.type_name()))
    }

    /// Insert a typed choice into the trace with type safety.
    ///
    /// This is a convenience method for manually constructing traces. Note that
    /// this method only updates the choices map - it does not modify the
    /// log-weight accumulators (log_prior, log_likelihood, log_factors).
    ///
    /// Example:
    /// ```rust
    /// # use fugue::*;
    /// # use fugue::runtime::trace::{Trace, ChoiceValue};
    ///
    /// let mut trace = Trace::default();
    ///
    /// // Insert different types of choices
    /// trace.insert_choice(addr!("mu"), ChoiceValue::F64(1.5), -0.125);
    /// trace.insert_choice(addr!("success"), ChoiceValue::Bool(true), -0.693);
    /// trace.insert_choice(addr!("count"), ChoiceValue::U64(10), -2.303);
    ///
    /// // Retrieve with type safety
    /// assert_eq!(trace.get_f64(&addr!("mu")), Some(1.5));
    /// assert_eq!(trace.get_bool(&addr!("success")), Some(true));
    /// assert_eq!(trace.get_u64(&addr!("count")), Some(10));
    ///
    /// println!("Trace has {} choices", trace.choices.len());
    /// ```
    pub fn insert_choice(&mut self, addr: Address, value: ChoiceValue, logp: f64) {
        let choice = Choice {
            addr: addr.clone(),
            value,
            logp,
        };
        self.choices.insert(addr, choice);
    }

    /// Collect the addresses under `prefix` (per [`Address::has_prefix`]).
    ///
    /// Addresses sharing a string prefix are lexicographically contiguous, so
    /// this scans only that range of the `BTreeMap` — O(k + log n) where k is
    /// the number of keys sharing the raw string prefix. (The `has_prefix`
    /// matches themselves need not be contiguous — a key like `"a0"` can sort
    /// between `"a#1"` and `"a::b"` — hence the extra filter.)
    fn prefix_addresses(&self, prefix: &str) -> Vec<Address> {
        self.choices
            .range(Address::new(prefix.to_string())..)
            .take_while(|(a, _)| a.as_str().starts_with(prefix))
            .filter(|(a, _)| a.has_prefix(prefix))
            .map(|(a, _)| a.clone())
            .collect()
    }

    /// Return a new trace containing exactly the choices whose address is under
    /// `prefix` (per [`Address::has_prefix`]).
    ///
    /// **The three log accumulators of the result are zeroed and must not be
    /// trusted**: `log_prior`/`log_likelihood`/`log_factors` are flat sums over
    /// the whole execution (observations and factors leave no `choices` entry at
    /// all), so they are not decomposable by address — see [`Self::insert_choice`].
    /// Re-score the result under the relevant model
    /// ([`ScoreGivenTrace`](crate::runtime::interpreters::ScoreGivenTrace) /
    /// [`score_given_trace_reconciled`](crate::runtime::interpreters::score_given_trace_reconciled))
    /// before using its weight. O(k + log n).
    pub fn extract_prefix(&self, prefix: &str) -> Trace {
        let mut out = Trace::default();
        for addr in self.prefix_addresses(prefix) {
            out.choices
                .insert(addr.clone(), self.choices[&addr].clone());
        }
        out
    }

    /// Remove every choice under `prefix` in place (per [`Address::has_prefix`]).
    ///
    /// The log accumulators are left **stale** — they are flat, non-decomposable
    /// sums (see [`Self::extract_prefix`]) — and must be recomputed by
    /// re-scoring under the model. O(k + log n).
    pub fn truncate_prefix(&mut self, prefix: &str) {
        for addr in self.prefix_addresses(prefix) {
            self.choices.remove(&addr);
        }
    }

    /// Replace the block under `prefix` with the choices of `donor` that fall
    /// under `prefix`: removes this trace's `prefix` range, then inserts the
    /// donor's `prefix`-range choices (values and stored `logp`).
    ///
    /// The log accumulators are left **invalid** — the caller MUST re-score the
    /// result under the model before using its weight. A graft that leaves the
    /// trace structurally inconsistent with the model (e.g. an incomplete
    /// assignment) is not detectable here; it surfaces at re-score time, so use
    /// the reconciling re-score
    /// ([`score_given_trace_reconciled`](crate::runtime::interpreters::score_given_trace_reconciled))
    /// when the graft may change model structure. O(k_old + k_donor + log n).
    pub fn graft_prefix(&mut self, prefix: &str, donor: &Trace) {
        self.truncate_prefix(prefix);
        for addr in donor.prefix_addresses(prefix) {
            self.choices
                .insert(addr.clone(), donor.choices[&addr].clone());
        }
    }

    /// Return a copy of this trace with every choice moved to the address `f`
    /// gives for it.
    ///
    /// This is the renaming counterpart of the prefix surgery above:
    /// [`extract_prefix`](Self::extract_prefix),
    /// [`truncate_prefix`](Self::truncate_prefix) and
    /// [`graft_prefix`](Self::graft_prefix) select, drop and replace choices but
    /// keep every address fixed. Like `extract_prefix` it borrows the trace and
    /// returns a new one. Each choice keeps its value and its stored `logp`, and
    /// [`Choice::addr`] is updated with the key.
    ///
    /// **`f` must be injective on the addresses of this trace.** If two
    /// addresses map to the same one the result is an error
    /// ([`ErrorCode::AddressConflict`](crate::error::ErrorCode::AddressConflict),
    /// naming both sources in its context) and no trace is returned: a choice
    /// is never dropped or overwritten silently. `f` is called once per choice,
    /// in address order.
    ///
    /// **The three log accumulators are copied unchanged.** They are flat sums
    /// over the execution that produced the trace, and renaming changes no
    /// value, so they remain exactly the weights of the renamed trace under the
    /// *renamed model* (the same program with its sites moved by `f`). This is
    /// what lets a weighted population be carried along a renaming with its
    /// weights intact. If the target model differs in any other way, re-score
    /// the result as after any other surgery
    /// ([`ScoreGivenTrace`](crate::runtime::interpreters::ScoreGivenTrace) /
    /// [`score_given_trace_reconciled`](crate::runtime::interpreters::score_given_trace_reconciled)).
    /// O(n log n).
    ///
    /// # Errors
    ///
    /// `FugueError::TraceError` with `AddressConflict` if `f` sends two
    /// addresses of this trace to the same address.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use fugue::*;
    /// # use fugue::runtime::trace::ChoiceValue;
    /// let mut trace = Trace::default();
    /// trace.insert_choice(addr!("mu"), ChoiceValue::F64(1.5), -0.5);
    /// trace.insert_choice(addr!("flag"), ChoiceValue::Bool(true), -0.7);
    /// trace.log_prior = -1.2;
    ///
    /// // Move every site under a scope.
    /// let scoped = trace
    ///     .map_addresses(|a| Address::new(format!("v2::{a}")))
    ///     .unwrap();
    /// assert_eq!(scoped.get_f64(&Address::new("v2::mu")), Some(1.5));
    /// assert_eq!(scoped.get_bool(&Address::new("v2::flag")), Some(true));
    /// assert_eq!(scoped.log_prior, -1.2);
    ///
    /// // A map that merges two sites is refused.
    /// let merged = trace.map_addresses(|_| addr!("same"));
    /// assert_eq!(merged.unwrap_err().code(), ErrorCode::AddressConflict);
    /// ```
    pub fn map_addresses(&self, mut f: impl FnMut(&Address) -> Address) -> FugueResult<Trace> {
        let mut out = Trace {
            choices: BTreeMap::new(),
            log_prior: self.log_prior,
            log_likelihood: self.log_likelihood,
            log_factors: self.log_factors,
        };
        // Which source address each target came from, for the collision report.
        let mut sources: BTreeMap<Address, &Address> = BTreeMap::new();
        for (addr, choice) in &self.choices {
            let target = f(addr);
            if let Some(first) = sources.insert(target.clone(), addr) {
                return Err(FugueError::trace_error(
                    "map_addresses",
                    Some(target),
                    "Two addresses map to the same address; a rename must be injective",
                    crate::error::ErrorCode::AddressConflict,
                )
                .with_context("first", first.to_string())
                .with_context("second", addr.to_string()));
            }
            out.choices.insert(
                target.clone(),
                Choice {
                    addr: target,
                    value: choice.value.clone(),
                    logp: choice.logp,
                },
            );
        }
        Ok(out)
    }

    /// Return a copy of this trace with the block under `from` moved under
    /// `to`: every address under `from` (per [`Address::has_prefix`]) has that
    /// leading `from` replaced by `to`, and every other choice is copied
    /// unchanged.
    ///
    /// So `rename_prefix("m/old", "m/new")` sends `m/old` to `m/new`,
    /// `m/old/c/0` to `m/new/c/0` and `m/old#3` to `m/new#3`, and leaves
    /// `m/older` alone. As with the other prefix methods, a caller using a
    /// bespoke separator can pass both prefixes with the separator included
    /// (`"first/"`, `"second/"`).
    ///
    /// **Collisions are an error, never a merge.** If a renamed address lands
    /// on an address that is not itself being moved, the result is
    /// `Err(AddressConflict)` and no choice is dropped. To *replace* an
    /// existing block, call [`truncate_prefix`](Self::truncate_prefix) on `to`
    /// first.
    ///
    /// The log accumulators are copied unchanged, with the meaning given at
    /// [`map_addresses`](Self::map_addresses): they are valid for the model
    /// renamed the same way. O(n log n).
    ///
    /// # Errors
    ///
    /// `FugueError::TraceError` with `AddressConflict` if an address under
    /// `from` would land on an address already present outside `from`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use fugue::*;
    /// # use fugue::runtime::trace::ChoiceValue;
    /// let mut trace = Trace::default();
    /// trace.insert_choice(Address::new("m/old/c/0/mu"), ChoiceValue::F64(0.3), -0.1);
    /// trace.insert_choice(Address::new("m/old/c/1/mu"), ChoiceValue::F64(0.6), -0.2);
    /// trace.insert_choice(Address::new("m/other/mu"), ChoiceValue::F64(0.9), -0.3);
    /// trace.log_likelihood = -4.0;
    ///
    /// let moved = trace.rename_prefix("m/old", "m/new").unwrap();
    /// assert_eq!(moved.get_f64(&Address::new("m/new/c/0/mu")), Some(0.3));
    /// assert_eq!(moved.get_f64(&Address::new("m/new/c/1/mu")), Some(0.6));
    /// assert_eq!(moved.get_f64(&Address::new("m/other/mu")), Some(0.9));
    /// assert!(moved.extract_prefix("m/old").choices.is_empty());
    /// assert_eq!(moved.log_likelihood, -4.0);
    ///
    /// // Landing on an occupied address is refused; nothing is overwritten.
    /// let clash = trace.rename_prefix("m/old/c/0", "m/other");
    /// assert_eq!(clash.unwrap_err().code(), ErrorCode::AddressConflict);
    /// ```
    pub fn rename_prefix(&self, from: &str, to: &str) -> FugueResult<Trace> {
        self.map_addresses(|addr| {
            if addr.has_prefix(from) {
                Address::new(format!("{to}{}", &addr.as_str()[from.len()..]))
            } else {
                addr.clone()
            }
        })
        .map_err(|e| match e {
            FugueError::TraceError {
                address,
                reason,
                code,
                context,
                ..
            } => FugueError::TraceError {
                operation: "rename_prefix".to_string(),
                address,
                reason,
                code,
                context,
            },
            other => other,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::addr;

    #[test]
    fn insert_and_getters_work() {
        let mut t = Trace::default();
        t.insert_choice(addr!("a"), ChoiceValue::F64(1.5), -0.5);
        t.insert_choice(addr!("b"), ChoiceValue::Bool(true), -0.7);
        t.insert_choice(addr!("c"), ChoiceValue::U64(3), -0.2);
        t.insert_choice(addr!("d"), ChoiceValue::Usize(4), -0.3);
        t.insert_choice(addr!("e"), ChoiceValue::I64(-7), -0.1);

        assert_eq!(t.get_f64(&addr!("a")), Some(1.5));
        assert_eq!(t.get_bool(&addr!("b")), Some(true));
        assert_eq!(t.get_u64(&addr!("c")), Some(3));
        assert_eq!(t.get_usize(&addr!("d")), Some(4));
        assert_eq!(t.get_i64(&addr!("e")), Some(-7));

        // Result-based accessors
        assert!(t.get_f64_result(&addr!("a")).is_ok());
        assert!(t.get_bool_result(&addr!("b")).is_ok());
        assert!(t.get_u64_result(&addr!("c")).is_ok());
        assert!(t.get_usize_result(&addr!("d")).is_ok());
        assert!(t.get_i64_result(&addr!("e")).is_ok());

        // Type mismatch
        let err = t.get_f64_result(&addr!("b")).unwrap_err();
        assert!(matches!(err, crate::error::FugueError::TypeMismatch { .. }));
    }

    #[test]
    fn total_log_weight_accumulates() {
        let mut t = Trace::default();
        // insert_choice does not modify log accumulators; set them explicitly
        t.insert_choice(addr!("x"), ChoiceValue::F64(0.0), -1.0);
        t.log_prior = -1.0;
        t.log_likelihood = -2.0;
        t.log_factors = -3.0;
        assert!((t.total_log_weight() - (-6.0)).abs() < 1e-12);
    }

    #[test]
    fn result_accessors_return_errors_for_missing_addresses() {
        let t = Trace::default();
        let e = t.get_f64_result(&addr!("missing")).unwrap_err();
        assert!(matches!(e, crate::error::FugueError::TraceError { .. }));
    }

    /// Boundary regression for prefix surgery: `extract_prefix("a")` must take
    /// the descendants of `a` under the path grammar and exclude both the
    /// sibling `ab/0` and the non-descendant `a0` that sorts inside the raw
    /// string-prefix range.
    #[test]
    fn test_extract_prefix_boundary() {
        let mut t = Trace::default();
        for (name, v) in [
            ("a", 0.0),
            ("a#1", 1.0),
            ("a/0", 2.0),
            ("a/1/x", 3.0),
            ("a::s", 4.0),
            ("a0", 5.0),
            ("ab/0", 6.0),
        ] {
            t.insert_choice(Address::new(name), ChoiceValue::F64(v), -0.5);
        }

        let sub = t.extract_prefix("a");
        let got: Vec<&str> = sub.choices.keys().map(|a| a.as_str()).collect();
        assert_eq!(got, vec!["a", "a#1", "a/0", "a/1/x", "a::s"]);

        // A prefix ending in a separator matches by plain starts_with.
        let slash = t.extract_prefix("a/");
        let got: Vec<&str> = slash.choices.keys().map(|a| a.as_str()).collect();
        assert_eq!(got, vec!["a/0", "a/1/x"]);
    }

    /// `graft_prefix(P, self.extract_prefix(P))` is a no-op on `choices`, and
    /// grafting from a donor replaces exactly the `P` block.
    #[test]
    fn test_graft_round_trip() {
        let mut t = Trace::default();
        t.insert_choice(addr!("gene", 0), ChoiceValue::F64(1.0), -0.1);
        t.insert_choice(addr!("gene", 1), ChoiceValue::F64(2.0), -0.2);
        t.insert_choice(addr!("other"), ChoiceValue::Bool(true), -0.3);

        let before: Vec<(String, ChoiceValue)> = t
            .choices
            .iter()
            .map(|(a, c)| (a.as_str().to_string(), c.value.clone()))
            .collect();
        let block = t.extract_prefix("gene");
        t.graft_prefix("gene", &block);
        let after: Vec<(String, ChoiceValue)> = t
            .choices
            .iter()
            .map(|(a, c)| (a.as_str().to_string(), c.value.clone()))
            .collect();
        assert_eq!(before, after);

        // Graft from a donor with different values on the block.
        let mut donor = Trace::default();
        donor.insert_choice(addr!("gene", 0), ChoiceValue::F64(9.0), -0.9);
        donor.insert_choice(addr!("gene", 1), ChoiceValue::F64(8.0), -0.8);
        donor.insert_choice(addr!("other"), ChoiceValue::Bool(false), -0.7);
        t.graft_prefix("gene", &donor);
        assert_eq!(t.get_f64(&addr!("gene", 0)), Some(9.0));
        assert_eq!(t.get_f64(&addr!("gene", 1)), Some(8.0));
        // Non-block addresses are untouched by the graft.
        assert_eq!(t.get_bool(&addr!("other")), Some(true));
    }

    /// Contract guard: the extracted subtrace's accumulators are zeroed —
    /// callers must re-score before trusting any weight.
    #[test]
    fn test_extract_zeroes_accumulators() {
        let mut t = Trace {
            log_prior: -1.0,
            log_likelihood: -2.0,
            log_factors: -3.0,
            ..Default::default()
        };
        t.insert_choice(addr!("x", 0), ChoiceValue::F64(0.5), -0.5);
        let sub = t.extract_prefix("x");
        assert_eq!(sub.log_prior, 0.0);
        assert_eq!(sub.log_likelihood, 0.0);
        assert_eq!(sub.log_factors, 0.0);
        assert_eq!(sub.choices.len(), 1);
    }

    /// A small deterministic generator for the property tests below (the crate
    /// has no property-testing dependency).
    fn lcg(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *state >> 33
    }

    /// A random trace over a small alphabet of path-like addresses, with all
    /// five value types and non-trivial accumulators.
    fn random_trace(seed: u64) -> Trace {
        let mut s = seed;
        let roots = ["a", "ab", "m/x", "m/y", "s::t", "z"];
        let tails = ["", "#0", "#1", "/0", "/0/mu", "/1/mu", "::k"];
        let mut t = Trace::default();
        for _ in 0..(1 + lcg(&mut s) % 12) {
            let name = format!(
                "{}{}",
                roots[(lcg(&mut s) % roots.len() as u64) as usize],
                tails[(lcg(&mut s) % tails.len() as u64) as usize]
            );
            let v = lcg(&mut s);
            let value = match v % 5 {
                0 => ChoiceValue::F64(v as f64 / 7.0),
                1 => ChoiceValue::Bool(v.is_multiple_of(2)),
                2 => ChoiceValue::U64(v),
                3 => ChoiceValue::Usize(v as usize % 9),
                _ => ChoiceValue::I64(-(v as i64 % 100)),
            };
            t.insert_choice(Address::new(name), value, -(v as f64) / 1e9);
        }
        t.log_prior = -(lcg(&mut s) as f64) / 1e8;
        t.log_likelihood = -(lcg(&mut s) as f64) / 1e8;
        t.log_factors = (lcg(&mut s) as f64) / 1e8;
        t
    }

    /// Everything but the addresses: values and stored `logp` in the given
    /// order, and the three accumulators bit for bit.
    fn payload(t: &Trace) -> (Vec<(ChoiceValue, u64)>, [u64; 3]) {
        (
            t.choices
                .values()
                .map(|c| (c.value.clone(), c.logp.to_bits()))
                .collect(),
            [
                t.log_prior.to_bits(),
                t.log_likelihood.to_bits(),
                t.log_factors.to_bits(),
            ],
        )
    }

    fn assert_keys_match_choice_addrs(t: &Trace) {
        for (k, c) in &t.choices {
            assert_eq!(k, &c.addr, "Choice::addr must follow the key");
        }
    }

    #[test]
    fn map_addresses_identity_is_a_copy() {
        for seed in 0..200 {
            let t = random_trace(seed);
            let same = t.map_addresses(|a| a.clone()).unwrap();
            assert_eq!(
                same.choices.keys().collect::<Vec<_>>(),
                t.choices.keys().collect::<Vec<_>>()
            );
            assert_eq!(payload(&same), payload(&t));
        }
    }

    /// An order-preserving injective map (a common prefix) keeps the payload in
    /// the same order, keeps the accumulators bit for bit, and is undone by the
    /// inverse map.
    #[test]
    fn map_addresses_injective_round_trips_and_preserves_accumulators() {
        for seed in 0..200 {
            let t = random_trace(seed);
            let there = t
                .map_addresses(|a| Address::new(format!("v2/{a}")))
                .unwrap();
            assert_eq!(there.choices.len(), t.choices.len());
            assert_keys_match_choice_addrs(&there);
            assert!(there.choices.keys().all(|a| a.has_prefix("v2")));
            assert_eq!(payload(&there), payload(&t));

            let back = there
                .map_addresses(|a| Address::new(a.as_str().strip_prefix("v2/").unwrap()))
                .unwrap();
            assert_eq!(
                back.choices.keys().collect::<Vec<_>>(),
                t.choices.keys().collect::<Vec<_>>()
            );
            assert_eq!(payload(&back), payload(&t));
            assert_keys_match_choice_addrs(&back);
        }
    }

    /// A map that reverses the order still carries each value to the image of
    /// its own address.
    #[test]
    fn map_addresses_carries_each_choice_to_its_image() {
        let mut t = Trace::default();
        t.insert_choice(addr!("a"), ChoiceValue::F64(1.0), -0.1);
        t.insert_choice(addr!("b"), ChoiceValue::U64(2), -0.2);
        t.insert_choice(addr!("c"), ChoiceValue::Bool(true), -0.3);
        let swapped = t
            .map_addresses(|a| match a.as_str() {
                "a" => addr!("c"),
                "c" => addr!("a"),
                _ => a.clone(),
            })
            .unwrap();
        assert_eq!(swapped.get_f64(&addr!("c")), Some(1.0));
        assert_eq!(swapped.get_bool(&addr!("a")), Some(true));
        assert_eq!(swapped.get_u64(&addr!("b")), Some(2));
        assert_eq!(swapped.choices[&addr!("c")].logp, -0.1);
        assert_keys_match_choice_addrs(&swapped);
    }

    /// `f` is called exactly once per choice, in address order.
    #[test]
    fn map_addresses_calls_f_once_per_choice_in_order() {
        let t = random_trace(7);
        let mut seen = Vec::new();
        t.map_addresses(|a| {
            seen.push(a.clone());
            a.clone()
        })
        .unwrap();
        assert_eq!(seen, t.choices.keys().cloned().collect::<Vec<_>>());
    }

    /// Collision policy: an error naming the target and both sources, and the
    /// input is untouched. A choice is never silently dropped.
    #[test]
    fn map_addresses_collision_is_an_error_not_a_drop() {
        let mut t = Trace::default();
        t.insert_choice(addr!("a"), ChoiceValue::F64(1.0), -0.1);
        t.insert_choice(addr!("b"), ChoiceValue::F64(2.0), -0.2);
        t.insert_choice(addr!("c"), ChoiceValue::F64(3.0), -0.3);

        let err = t
            .map_addresses(|a| {
                if a.as_str() == "c" {
                    addr!("a")
                } else {
                    a.clone()
                }
            })
            .unwrap_err();
        assert_eq!(err.code(), crate::error::ErrorCode::AddressConflict);
        match &err {
            FugueError::TraceError {
                operation, address, ..
            } => {
                assert_eq!(operation, "map_addresses");
                assert_eq!(address.as_ref(), Some(&addr!("a")));
            }
            other => panic!("expected a TraceError, got {other:?}"),
        }
        let ctx = &err.context().context;
        assert!(ctx.contains(&("first".to_string(), "a".to_string())));
        assert!(ctx.contains(&("second".to_string(), "c".to_string())));
        // The source trace is borrowed, so it is intact.
        assert_eq!(t.choices.len(), 3);

        // Property: any non-injective map over any trace with two or more
        // choices is refused.
        for seed in 0..200 {
            let t = random_trace(seed);
            let r = t.map_addresses(|_| addr!("sink"));
            assert_eq!(r.is_err(), t.choices.len() > 1);
        }
    }

    /// `rename_prefix` moves exactly the `has_prefix` block (the same boundary
    /// rule as `extract_prefix`) and nothing else.
    #[test]
    fn rename_prefix_moves_exactly_the_block() {
        let mut t = Trace {
            log_prior: -1.0,
            log_likelihood: -2.0,
            log_factors: -3.0,
            ..Default::default()
        };
        for (name, v) in [
            ("a", 0.0),
            ("a#1", 1.0),
            ("a/0", 2.0),
            ("a/1/x", 3.0),
            ("a::s", 4.0),
            ("a0", 5.0),
            ("ab/0", 6.0),
        ] {
            t.insert_choice(Address::new(name), ChoiceValue::F64(v), -0.5);
        }

        let moved = t.rename_prefix("a", "q/r").unwrap();
        let got: Vec<(&str, f64)> = moved
            .choices
            .iter()
            .map(|(a, c)| (a.as_str(), c.value.as_f64().unwrap()))
            .collect();
        assert_eq!(
            got,
            vec![
                ("a0", 5.0),
                ("ab/0", 6.0),
                ("q/r", 0.0),
                ("q/r#1", 1.0),
                ("q/r/0", 2.0),
                ("q/r/1/x", 3.0),
                ("q/r::s", 4.0),
            ]
        );
        assert_keys_match_choice_addrs(&moved);
        assert_eq!(
            (moved.log_prior, moved.log_likelihood, moved.log_factors),
            (-1.0, -2.0, -3.0)
        );

        // A prefix that ends in a separator moves only what follows it.
        let slash = t.rename_prefix("a/", "b/").unwrap();
        let keys: Vec<&str> = slash.choices.keys().map(|a| a.as_str()).collect();
        assert_eq!(keys, vec!["a", "a#1", "a0", "a::s", "ab/0", "b/0", "b/1/x"]);

        // A prefix that matches nothing is a plain copy.
        let none = t.rename_prefix("nope", "x").unwrap();
        assert_eq!(payload(&none), payload(&t));
    }

    /// Renaming there and back recovers the trace whenever the target prefix is
    /// free, and the moved block is what `extract_prefix` selects.
    #[test]
    fn rename_prefix_round_trips() {
        for seed in 0..300 {
            let t = random_trace(seed);
            for from in ["a", "ab", "m/x", "m", "m/", "s", "s::t", "z"] {
                let to = if from.ends_with('/') {
                    "fresh/"
                } else {
                    "fresh"
                };
                let moved = t.rename_prefix(from, to).unwrap();
                assert_eq!(moved.choices.len(), t.choices.len());
                assert_keys_match_choice_addrs(&moved);
                assert_eq!(
                    moved.extract_prefix(to).choices.len(),
                    t.extract_prefix(from).choices.len()
                );
                assert!(moved.extract_prefix(from).choices.is_empty());
                assert_eq!(payload(&moved).1, payload(&t).1);

                let back = moved.rename_prefix(to, from).unwrap();
                assert_eq!(
                    back.choices.keys().collect::<Vec<_>>(),
                    t.choices.keys().collect::<Vec<_>>()
                );
                assert_eq!(payload(&back), payload(&t));
            }
        }
    }

    /// Landing on an address outside the moved block is an error (reported
    /// under `rename_prefix`), and truncating the target first is the way to
    /// replace a block.
    #[test]
    fn rename_prefix_collision_is_an_error_not_a_merge() {
        let mut t = Trace::default();
        t.insert_choice(Address::new("old/0"), ChoiceValue::F64(1.0), -0.1);
        t.insert_choice(Address::new("old/1"), ChoiceValue::F64(2.0), -0.2);
        t.insert_choice(Address::new("new/1"), ChoiceValue::F64(9.0), -0.9);

        let err = t.rename_prefix("old", "new").unwrap_err();
        assert_eq!(err.code(), crate::error::ErrorCode::AddressConflict);
        match &err {
            FugueError::TraceError {
                operation, address, ..
            } => {
                assert_eq!(operation, "rename_prefix");
                assert_eq!(address.as_ref(), Some(&Address::new("new/1")));
            }
            other => panic!("expected a TraceError, got {other:?}"),
        }

        let mut cleared = t.clone();
        cleared.truncate_prefix("new");
        let moved = cleared.rename_prefix("old", "new").unwrap();
        assert_eq!(moved.get_f64(&Address::new("new/0")), Some(1.0));
        assert_eq!(moved.get_f64(&Address::new("new/1")), Some(2.0));
        assert_eq!(moved.choices.len(), 2);
    }

    /// The reason the accumulators are kept: a prior-run trace renamed along
    /// with its model re-scores to exactly the weights it already carries.
    #[test]
    fn renamed_trace_keeps_its_weights_under_the_renamed_model() {
        use crate::core::distribution::Normal;
        use crate::core::model::{factor, observe, sample, ModelExt};
        use crate::runtime::handler::run;
        use crate::runtime::interpreters::{PriorHandler, ScoreGivenTrace};
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        let model = |root: &'static str| {
            sample(
                Address::new(format!("{root}/mu")),
                Normal::new(0.0, 1.0).unwrap(),
            )
            .bind(move |mu| {
                sample(
                    Address::new(format!("{root}/nu")),
                    Normal::new(mu, 2.0).unwrap(),
                )
                .bind(move |nu| {
                    observe(
                        Address::new(format!("{root}/y")),
                        Normal::new(nu, 0.5).unwrap(),
                        0.3,
                    )
                    .bind(move |_| factor(-0.25 * mu * mu))
                })
            })
        };

        let mut rng = StdRng::seed_from_u64(11);
        let (_, old) = run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            model("old"),
        );
        let renamed = old.rename_prefix("old", "new").unwrap();
        let (_, rescored) = run(
            ScoreGivenTrace {
                base: renamed.clone(),
                trace: Trace::default(),
            },
            model("new"),
        );
        assert_eq!(rescored.log_prior, renamed.log_prior);
        assert_eq!(rescored.log_likelihood, renamed.log_likelihood);
        assert_eq!(rescored.log_factors, renamed.log_factors);
        assert_eq!(rescored.log_prior, old.log_prior);
    }
}
