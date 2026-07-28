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
}
