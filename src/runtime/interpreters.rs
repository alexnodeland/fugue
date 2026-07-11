#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/docs/runtime/interpreters.md"))]

use crate::core::address::Address;
use crate::core::distribution::Distribution;
use crate::core::model::Model;
use crate::error::{ErrorCode, FugueError, FugueResult};
use crate::runtime::handler::{run, Handler};
use crate::runtime::trace::{Choice, ChoiceValue, Trace};

use rand::RngCore;

/// Panic when a sample site reuses an address already recorded in this
/// execution's output trace (FG-47).
///
/// "Fast" handlers (`PriorHandler`, `ReplayHandler`, `ScoreGivenTrace`) treat a
/// duplicate address as a programming error and panic with a precise message —
/// this is the documented fast/safe split. Detection is O(log n) and needs no
/// extra state: the output trace's `choices` map already contains exactly the
/// addresses visited so far in this run, so a hit means the address was sampled
/// twice. Before this check, the second visit silently double-counted its
/// log-prior while dropping the first choice.
#[inline]
fn assert_no_duplicate_sample(trace: &Trace, addr: &Address, handler: &str) {
    if trace.choices.contains_key(addr) {
        panic!(
            "{handler}: address {addr} was sampled twice in one execution \
             (AddressConflict, ErrorCode::{:?}={}). Every sample site must have \
             a unique address.",
            ErrorCode::AddressConflict,
            ErrorCode::AddressConflict as u32
        );
    }
}

/// Build the [`FugueError`] used for a duplicate sample address, so that "safe"
/// handlers and the fallible scoring paths surface a real
/// [`ErrorCode::AddressConflict`] (FG-47) instead of panicking.
fn address_conflict_error(addr: &Address, handler: &str) -> FugueError {
    FugueError::ModelError {
        address: Some(addr.clone()),
        reason: format!("{handler}: address sampled twice in one execution"),
        code: ErrorCode::AddressConflict,
        context: crate::error::ErrorContext::new(),
    }
}

// =============================================================================
// Internal monomorphization macros (FG-54)
//
// Each handler used to hand-copy four (now five, with i64) near-identical
// `on_sample_*` / `on_observe_*` methods. These `macro_rules!` collapse that
// copy-paste: a handler lists the `(method, type, variant)` rows once and the
// macro expands the shared body for every supported value type. The behavior is
// identical to the old hand-written methods, plus the FG-47 duplicate-address
// check and the FG-48 fresh-logp fix, applied uniformly.
// =============================================================================

/// The five value types every handler supports, as `(sample_method,
/// observe_method, rust_type, ChoiceValue variant, type-name literal,
/// Option-getter, Result-getter)` rows. Passed to the per-handler macros so the
/// row list lives in exactly one place.
macro_rules! for_each_value_type {
    ($m:ident) => {
        $m! {
            (on_sample_f64,   on_observe_f64,   f64,   F64,   "f64",   get_f64,   get_f64_result),
            (on_sample_bool,  on_observe_bool,  bool,  Bool,  "bool",  get_bool,  get_bool_result),
            (on_sample_u64,   on_observe_u64,   u64,   U64,   "u64",   get_u64,   get_u64_result),
            (on_sample_usize, on_observe_usize, usize, Usize, "usize", get_usize, get_usize_result),
            (on_sample_i64,   on_observe_i64,   i64,   I64,   "i64",   get_i64,   get_i64_result),
        }
    };
}

/// Generate the five identical `on_observe_*` methods (every handler scores an
/// observation the same way: add its log-density to `log_likelihood`).
macro_rules! impl_observe_methods {
    ($(($sample:ident, $observe:ident, $ty:ty, $variant:ident, $tyname:literal, $get:ident, $get_res:ident)),* $(,)?) => {
        $(
            fn $observe(&mut self, _addr: &Address, dist: &dyn Distribution<$ty>, value: $ty) {
                self.trace.log_likelihood += dist.log_prob(&value);
            }
        )*
    };
}

/// `PriorHandler`: draw a fresh value, score it, record it. Panics on a
/// duplicate address (fast handler).
macro_rules! impl_prior_sample_methods {
    ($(($sample:ident, $observe:ident, $ty:ty, $variant:ident, $tyname:literal, $get:ident, $get_res:ident)),* $(,)?) => {
        $(
            fn $sample(&mut self, addr: &Address, dist: &dyn Distribution<$ty>) -> $ty {
                assert_no_duplicate_sample(&self.trace, addr, "PriorHandler");
                let x = dist.sample(self.rng);
                let lp = dist.log_prob(&x);
                self.trace.log_prior += lp;
                self.trace.choices.insert(
                    addr.clone(),
                    Choice { addr: addr.clone(), value: ChoiceValue::$variant(x), logp: lp },
                );
                x
            }
        )*
    };
}

/// `ReplayHandler`: reuse the base value if present (panic on type mismatch),
/// otherwise sample fresh; always re-score under the current distribution.
/// Panics on a duplicate address (fast handler).
macro_rules! impl_replay_sample_methods {
    ($(($sample:ident, $observe:ident, $ty:ty, $variant:ident, $tyname:literal, $get:ident, $get_res:ident)),* $(,)?) => {
        $(
            fn $sample(&mut self, addr: &Address, dist: &dyn Distribution<$ty>) -> $ty {
                assert_no_duplicate_sample(&self.trace, addr, "ReplayHandler");
                let x = if let Some(c) = self.base.choices.get(addr) {
                    match c.value {
                        ChoiceValue::$variant(v) => v,
                        _ => panic!("expected {} at {}", $tyname, addr),
                    }
                } else {
                    dist.sample(self.rng)
                };
                let lp = dist.log_prob(&x);
                self.trace.log_prior += lp;
                self.trace.choices.insert(
                    addr.clone(),
                    Choice { addr: addr.clone(), value: ChoiceValue::$variant(x), logp: lp },
                );
                x
            }
        )*
    };
}

/// `ScoreGivenTrace`: read the fixed value from the base trace (panic if
/// missing or wrong type), score it under the current distribution, and store a
/// FRESH choice carrying that newly computed logp (FG-48). Panics on a duplicate
/// address (fast handler).
macro_rules! impl_score_sample_methods {
    ($(($sample:ident, $observe:ident, $ty:ty, $variant:ident, $tyname:literal, $get:ident, $get_res:ident)),* $(,)?) => {
        $(
            fn $sample(&mut self, addr: &Address, dist: &dyn Distribution<$ty>) -> $ty {
                assert_no_duplicate_sample(&self.trace, addr, "ScoreGivenTrace");
                let c = self
                    .base
                    .choices
                    .get(addr)
                    .unwrap_or_else(|| panic!("missing value for site {} in base trace", addr));
                let x = match c.value {
                    ChoiceValue::$variant(v) => v,
                    _ => panic!("expected {} at {}", $tyname, addr),
                };
                let lp = dist.log_prob(&x);
                self.trace.log_prior += lp;
                // FG-48: store the freshly computed logp, not the stale base one.
                self.trace.choices.insert(
                    addr.clone(),
                    Choice { addr: addr.clone(), value: ChoiceValue::$variant(x), logp: lp },
                );
                x
            }
        )*
    };
}

/// `SafeReplayHandler`: like `ReplayHandler` but recovers from missing/mismatched
/// base values by sampling fresh (optionally warning). A duplicate address is a
/// programming error even in the safe handler, so it invalidates the trace with
/// `-inf` and warns rather than silently double-counting (FG-47).
macro_rules! impl_safe_replay_sample_methods {
    ($(($sample:ident, $observe:ident, $ty:ty, $variant:ident, $tyname:literal, $get:ident, $get_res:ident)),* $(,)?) => {
        $(
            fn $sample(&mut self, addr: &Address, dist: &dyn Distribution<$ty>) -> $ty {
                if self.trace.choices.contains_key(addr) {
                    if self.warn_on_mismatch {
                        eprintln!("Warning: {}", address_conflict_error(addr, "SafeReplayHandler"));
                    }
                    self.trace.log_prior += f64::NEG_INFINITY;
                    return dist.sample(self.rng);
                }
                let x = match self.base.$get(addr) {
                    Some(v) => v,
                    None => {
                        if self.warn_on_mismatch && self.base.choices.contains_key(addr) {
                            if let Some(choice) = self.base.choices.get(addr) {
                                eprintln!(
                                    "Warning: Type mismatch at {}: expected {}, found {}",
                                    addr, $tyname, choice.value.type_name()
                                );
                            }
                        }
                        dist.sample(self.rng)
                    }
                };
                let lp = dist.log_prob(&x);
                self.trace.log_prior += lp;
                self.trace.choices.insert(
                    addr.clone(),
                    Choice { addr: addr.clone(), value: ChoiceValue::$variant(x), logp: lp },
                );
                x
            }
        )*
    };
}

/// `SafeScoreGivenTrace`: like `ScoreGivenTrace` but returns an invalid (`-inf`)
/// trace instead of panicking on a missing/mismatched address, and stores a
/// FRESH choice with the newly computed logp (FG-48). A duplicate address
/// invalidates the trace (FG-47).
macro_rules! impl_safe_score_sample_methods {
    ($(($sample:ident, $observe:ident, $ty:ty, $variant:ident, $tyname:literal, $get:ident, $get_res:ident)),* $(,)?) => {
        $(
            fn $sample(&mut self, addr: &Address, dist: &dyn Distribution<$ty>) -> $ty {
                if self.trace.choices.contains_key(addr) {
                    if self.warn_on_error {
                        eprintln!("Warning: {}", address_conflict_error(addr, "SafeScoreGivenTrace"));
                    }
                    self.trace.log_prior += f64::NEG_INFINITY;
                    return <$ty as Default>::default();
                }
                match self.base.$get_res(addr) {
                    Ok(x) => {
                        let lp = dist.log_prob(&x);
                        self.trace.log_prior += lp;
                        // FG-48: fresh logp consistent with what we accumulated.
                        self.trace.choices.insert(
                            addr.clone(),
                            Choice { addr: addr.clone(), value: ChoiceValue::$variant(x), logp: lp },
                        );
                        x
                    }
                    Err(e) => {
                        if self.warn_on_error {
                            eprintln!("Warning: Failed to get {} at {}: {}", $tyname, addr, e);
                        }
                        self.trace.log_prior += f64::NEG_INFINITY;
                        <$ty as Default>::default()
                    }
                }
            }
        )*
    };
}

/// `StrictScoreGivenTrace`: the fallible, structure-checking scoring path
/// (FG-20/FG-21). It records the FIRST structural problem into `self.error`
/// instead of panicking: an address absent from the base trace or a type
/// mismatch yields `ErrorCode::UnexpectedModelStructure`; a duplicate address
/// yields `ErrorCode::AddressConflict`. On success it stores a fresh, correctly
/// scored choice.
macro_rules! impl_strict_score_sample_methods {
    ($(($sample:ident, $observe:ident, $ty:ty, $variant:ident, $tyname:literal, $get:ident, $get_res:ident)),* $(,)?) => {
        $(
            fn $sample(&mut self, addr: &Address, dist: &dyn Distribution<$ty>) -> $ty {
                if self.trace.choices.contains_key(addr) {
                    if self.error.is_none() {
                        *self.error = Some(address_conflict_error(addr, "StrictScoreGivenTrace"));
                    }
                    return <$ty as Default>::default();
                }
                match self.base.$get_res(addr) {
                    Ok(x) => {
                        let lp = dist.log_prob(&x);
                        self.trace.log_prior += lp;
                        self.trace.choices.insert(
                            addr.clone(),
                            Choice { addr: addr.clone(), value: ChoiceValue::$variant(x), logp: lp },
                        );
                        x
                    }
                    Err(cause) => {
                        if self.error.is_none() {
                            *self.error = Some(FugueError::ModelError {
                                address: Some(addr.clone()),
                                reason: format!(
                                    "model visited address {} not present (as {}) in the base \
                                     trace; structure varies between the base trace and this model",
                                    addr, $tyname
                                ),
                                code: ErrorCode::UnexpectedModelStructure,
                                context: crate::error::ErrorContext::new().with_cause(cause),
                            });
                        }
                        <$ty as Default>::default()
                    }
                }
            }
        )*
    };
}

/// `ReconcilingScoreGivenTrace`: the reconciling scoring path (FG-20/FG-21).
/// Addresses present in the base trace are replayed and re-scored; NEW addresses
/// (absent, or present with the wrong type) are sampled fresh from the prior and
/// their log-prior accumulated (the RJMCMC-correct treatment of prior-proposed
/// fresh dimensions) and recorded in `fresh`. Vanished addresses are computed
/// after the run by the driver. A duplicate address is still an error.
macro_rules! impl_reconciling_score_sample_methods {
    ($(($sample:ident, $observe:ident, $ty:ty, $variant:ident, $tyname:literal, $get:ident, $get_res:ident)),* $(,)?) => {
        $(
            fn $sample(&mut self, addr: &Address, dist: &dyn Distribution<$ty>) -> $ty {
                if self.trace.choices.contains_key(addr) {
                    if self.error.is_none() {
                        *self.error =
                            Some(address_conflict_error(addr, "ReconcilingScoreGivenTrace"));
                    }
                    return <$ty as Default>::default();
                }
                let x = match self.base.$get(addr) {
                    Some(v) => v,
                    None => {
                        // New (or type-changed) dimension: propose from the prior.
                        self.fresh.push(addr.clone());
                        dist.sample(self.rng)
                    }
                };
                let lp = dist.log_prob(&x);
                self.trace.log_prior += lp;
                self.trace.choices.insert(
                    addr.clone(),
                    Choice { addr: addr.clone(), value: ChoiceValue::$variant(x), logp: lp },
                );
                x
            }
        )*
    };
}

/// Handler for prior sampling - generates fresh random values from distributions.
///
/// This is the foundational interpreter that implements standard "forward sampling"
/// from probabilistic models. It draws fresh values from distributions and accumulates
/// log-probabilities in the trace.
///
/// Example:
/// ```rust
/// # use fugue::*;
/// # use fugue::runtime::interpreters::PriorHandler;
/// # use rand::rngs::StdRng;
/// # use rand::SeedableRng;
///
/// let model = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
///     .bind(|x| observe(addr!("y"), Normal::new(x, 0.5).unwrap(), 1.2)
///         .map(move |_| x));
///
/// let mut rng = StdRng::seed_from_u64(42);
/// let (result, trace) = runtime::handler::run(
///     PriorHandler { rng: &mut rng, trace: Trace::default() },
///     model
/// );
///
/// assert!(result.is_finite());
/// assert!(trace.log_likelihood.is_finite());
/// ```
pub struct PriorHandler<'r, R: RngCore> {
    /// Random number generator for sampling.
    pub rng: &'r mut R,
    /// Trace to accumulate execution history.
    pub trace: Trace,
}
impl<'r, R: RngCore> Handler for PriorHandler<'r, R> {
    for_each_value_type!(impl_prior_sample_methods);
    for_each_value_type!(impl_observe_methods);

    fn on_factor(&mut self, logw: f64) {
        self.trace.log_factors += logw;
    }

    fn finish(self) -> Trace {
        self.trace
    }
}

/// Handler for replaying models with values from an existing trace.
///
/// ReplayHandler replays a model execution using stored trace values. When a sampling
/// site is encountered: if the address exists in the base trace, use that value;
/// if missing, sample fresh. Essential for MCMC where you replay most choices
/// but sample new values at specific sites.
///
/// Example:
/// ```rust
/// # use fugue::*;
/// # use fugue::runtime::interpreters::*;
/// # use rand::rngs::StdRng;
/// # use rand::SeedableRng;
///
/// // Create base trace
/// let mut rng = StdRng::seed_from_u64(42);
/// let model_fn = || sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
/// let (original, base_trace) = runtime::handler::run(
///     PriorHandler { rng: &mut rng, trace: Trace::default() },
///     model_fn()
/// );
///
/// // Replay using base trace values
/// let (replayed, _) = runtime::handler::run(
///     ReplayHandler {
///         rng: &mut rng,
///         base: base_trace,
///         trace: Trace::default()
///     },
///     model_fn()
/// );
///
/// assert_eq!(original, replayed); // Same value replayed
/// ```
pub struct ReplayHandler<'r, R: RngCore> {
    /// Random number generator for sampling at addresses not in base trace.
    pub rng: &'r mut R,
    /// Base trace containing values to replay.
    pub base: Trace,
    /// New trace to accumulate the replay execution.
    pub trace: Trace,
}
impl<'r, R: RngCore> Handler for ReplayHandler<'r, R> {
    for_each_value_type!(impl_replay_sample_methods);
    for_each_value_type!(impl_observe_methods);

    fn on_factor(&mut self, logw: f64) {
        self.trace.log_factors += logw;
    }

    fn finish(self) -> Trace {
        self.trace
    }
}

/// Handler for scoring a model given a complete trace of fixed choices.
///
/// ScoreGivenTrace computes log-probability of a model execution where all random
/// choices are predetermined. No sampling occurs - values are looked up from the
/// base trace and their log-probabilities computed. Essential for MCMC acceptance
/// ratios, importance sampling, and model comparison.
///
/// Example:
/// ```rust
/// # use fugue::*;
/// # use fugue::runtime::interpreters::*;
/// # use rand::rngs::StdRng;
/// # use rand::SeedableRng;
///
/// // Create a complete trace
/// let mut rng = StdRng::seed_from_u64(42);
/// let (_, complete_trace) = runtime::handler::run(
///     PriorHandler { rng: &mut rng, trace: Trace::default() },
///     sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
/// );
///
/// // Score under different model parameters  
/// let (value, score_trace) = runtime::handler::run(
///     ScoreGivenTrace {
///         base: complete_trace,
///         trace: Trace::default()
///     },
///     sample(addr!("x"), Normal::new(1.0, 2.0).unwrap()) // Different parameters
/// );
///
/// assert!(score_trace.total_log_weight().is_finite());
/// ```
pub struct ScoreGivenTrace {
    /// Base trace containing the fixed choices to score.
    pub base: Trace,
    /// New trace to accumulate log-probabilities.
    pub trace: Trace,
}
impl Handler for ScoreGivenTrace {
    for_each_value_type!(impl_score_sample_methods);
    for_each_value_type!(impl_observe_methods);

    fn on_factor(&mut self, logw: f64) {
        self.trace.log_factors += logw;
    }

    fn finish(self) -> Trace {
        self.trace
    }
}

/// Safe version of ReplayHandler that gracefully handles trace inconsistencies.
///
/// SafeReplayHandler replays model execution like ReplayHandler, but handles type
/// mismatches and missing addresses gracefully by logging warnings and sampling
/// fresh values instead of panicking. Essential for production systems where
/// trace consistency cannot be guaranteed.
///
/// Example:
/// ```rust
/// # use fugue::*;
/// # use fugue::runtime::interpreters::*;
/// # use rand::rngs::StdRng;
/// # use rand::SeedableRng;
///
/// // Create trace with potential inconsistencies
/// let mut rng = StdRng::seed_from_u64(42);
/// let (_, base_trace) = runtime::handler::run(
///     PriorHandler { rng: &mut rng, trace: Trace::default() },
///     sample(addr!("x"), Normal::new(0.0, 1.0).unwrap()) // f64 value
/// );
///
/// // Safe replay handles type mismatch gracefully
/// let (result, trace) = runtime::handler::run(
///     SafeReplayHandler {
///         rng: &mut rng,
///         base: base_trace,
///         trace: Trace::default(),
///         warn_on_mismatch: true, // Enable warnings
///     },
///     sample(addr!("x"), Bernoulli::new(0.5).unwrap()) // Expects bool
/// );
///
/// assert!(trace.total_log_weight().is_finite()); // Continues execution
/// ```
pub struct SafeReplayHandler<'r, R: RngCore> {
    /// Random number generator for sampling at addresses not in base trace.
    pub rng: &'r mut R,
    /// Base trace containing values to replay.
    pub base: Trace,
    /// New trace to accumulate the replay execution.
    pub trace: Trace,
    /// Whether to log warnings on type mismatches (useful for debugging).
    pub warn_on_mismatch: bool,
}
impl<'r, R: RngCore> Handler for SafeReplayHandler<'r, R> {
    for_each_value_type!(impl_safe_replay_sample_methods);
    for_each_value_type!(impl_observe_methods);

    fn on_factor(&mut self, logw: f64) {
        self.trace.log_factors += logw;
    }

    fn finish(self) -> Trace {
        self.trace
    }
}

/// Safe version of ScoreGivenTrace that gracefully handles incomplete traces.
///
/// SafeScoreGivenTrace computes log-probability like ScoreGivenTrace, but handles
/// missing addresses or type mismatches by returning negative infinity log-weight
/// instead of panicking. Essential for production inference where trace validity
/// cannot be guaranteed.
///
/// Example:
/// ```rust
/// # use fugue::*;
/// # use fugue::runtime::interpreters::*;
/// # use rand::rngs::StdRng;
/// # use rand::SeedableRng;
///
/// // Create incomplete trace
/// let mut rng = StdRng::seed_from_u64(42);
/// let (_, incomplete_trace) = runtime::handler::run(
///     PriorHandler { rng: &mut rng, trace: Trace::default() },
///     sample(addr!("x"), Normal::new(0.0, 1.0).unwrap()) // Only has "x"
/// );
///
/// // Safe scoring handles missing address gracefully
/// let (_, score_trace) = runtime::handler::run(
///     SafeScoreGivenTrace {
///         base: incomplete_trace,
///         trace: Trace::default(),
///         warn_on_error: true, // Enable warnings
///     },
///     sample(addr!("missing"), Normal::new(0.0, 1.0).unwrap()) // Address not in base
/// );
///
/// assert_eq!(score_trace.total_log_weight(), f64::NEG_INFINITY); // Graceful failure
/// ```
pub struct SafeScoreGivenTrace {
    /// Base trace containing the fixed choices to score.
    pub base: Trace,
    /// New trace to accumulate log-probabilities.
    pub trace: Trace,
    /// Whether to log warnings on missing addresses or type mismatches.
    pub warn_on_error: bool,
}
impl Handler for SafeScoreGivenTrace {
    for_each_value_type!(impl_safe_score_sample_methods);
    for_each_value_type!(impl_observe_methods);

    fn on_factor(&mut self, logw: f64) {
        self.trace.log_factors += logw;
    }

    fn finish(self) -> Trace {
        self.trace
    }
}

// =============================================================================
// Structure-varying (trans-dimensional) scoring paths (FG-20 / FG-21)
// =============================================================================

/// The strict, fallible sibling of [`ScoreGivenTrace`].
///
/// This handler backs [`score_given_trace_strict`]. Instead of panicking when
/// the model's address structure differs from the base trace, it records the
/// first structural problem so the driver can return a [`FugueError`]:
///
/// - visiting an address absent from the base trace (or present with the wrong
///   value type) -> [`ErrorCode::UnexpectedModelStructure`];
/// - visiting the same address twice -> [`ErrorCode::AddressConflict`].
///
/// On success every visited site stores a fresh, correctly scored choice.
pub struct StrictScoreGivenTrace<'e> {
    /// Base trace containing the fixed choices to score.
    pub base: Trace,
    /// New trace to accumulate log-probabilities.
    pub trace: Trace,
    /// First structural error encountered, surfaced by the driver.
    error: &'e mut Option<FugueError>,
}
impl<'e> Handler for StrictScoreGivenTrace<'e> {
    for_each_value_type!(impl_strict_score_sample_methods);
    for_each_value_type!(impl_observe_methods);

    fn on_factor(&mut self, logw: f64) {
        self.trace.log_factors += logw;
    }

    fn finish(self) -> Trace {
        self.trace
    }
}

/// Score `model` against `base` strictly, returning an error rather than
/// panicking when the model's address structure does not match the base trace
/// (FG-20 / FG-21).
///
/// Returns `Ok((value, scored_trace))` when every sample site visited by the
/// model is present in `base` with a matching value type. Returns `Err` with
/// [`ErrorCode::UnexpectedModelStructure`] if the model visits an address absent
/// from `base` (a branch opened by a differing latent value), or
/// [`ErrorCode::AddressConflict`] if the model visits the same address twice.
///
/// This is the mechanism the MCMC layer needs to stop crashing on
/// structure-varying proposals; wiring it into the samplers is a separate work
/// package.
///
/// # Example
///
/// ```rust
/// # use fugue::*;
/// # use fugue::runtime::interpreters::{score_given_trace_strict, PriorHandler};
/// # use rand::rngs::StdRng;
/// # use rand::SeedableRng;
/// let mut rng = StdRng::seed_from_u64(1);
/// let (_, base) = runtime::handler::run(
///     PriorHandler { rng: &mut rng, trace: Trace::default() },
///     sample(addr!("x"), Normal::new(0.0, 1.0).unwrap()),
/// );
///
/// // Same structure: Ok.
/// assert!(score_given_trace_strict(
///     base.clone(),
///     sample(addr!("x"), Normal::new(0.0, 1.0).unwrap()),
/// ).is_ok());
///
/// // Model reaches an address the base trace never recorded: Err.
/// let err = score_given_trace_strict(
///     base,
///     sample(addr!("y"), Normal::new(0.0, 1.0).unwrap()),
/// ).unwrap_err();
/// assert_eq!(err.code(), ErrorCode::UnexpectedModelStructure);
/// ```
pub fn score_given_trace_strict<A>(base: Trace, model: Model<A>) -> FugueResult<(A, Trace)> {
    let mut error: Option<FugueError> = None;
    let handler = StrictScoreGivenTrace {
        base,
        trace: Trace::default(),
        error: &mut error,
    };
    let (a, trace) = run(handler, model);
    match error {
        Some(e) => Err(e),
        None => Ok((a, trace)),
    }
}

/// Report of the structural differences reconciled by
/// [`score_given_trace_reconciled`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ReconcileReport {
    /// Addresses visited by the model that were absent from the base trace (or
    /// present with the wrong value type). Each was proposed fresh from its
    /// prior and its `log_prob` accumulated into `log_prior`.
    pub fresh_addresses: Vec<Address>,
    /// Addresses present in the base trace that the model did NOT visit. Their
    /// contribution should be dropped by the caller (e.g. removed from the
    /// reverse-move density in an RJMCMC step).
    pub vanished_addresses: Vec<Address>,
}

/// The reconciling sibling of [`ScoreGivenTrace`], backing
/// [`score_given_trace_reconciled`].
///
/// Addresses present in the base trace are replayed and re-scored. NEW addresses
/// (absent, or present with a different value type) are sampled fresh from the
/// prior, their `log_prob` accumulated into `log_prior`, and recorded in
/// `fresh`. A duplicate address is still an error.
pub struct ReconcilingScoreGivenTrace<'r, 'f, 'e, R: RngCore> {
    /// RNG used to propose fresh values for new addresses.
    pub rng: &'r mut R,
    /// Base trace containing the fixed choices to replay/score.
    pub base: Trace,
    /// New trace accumulating the reconciled execution.
    pub trace: Trace,
    /// Addresses sampled fresh from the prior (not present in `base`), in
    /// visitation order. Borrowed so the driver can read it after `finish`.
    fresh: &'f mut Vec<Address>,
    /// First duplicate-address error encountered, surfaced by the driver.
    error: &'e mut Option<FugueError>,
}
impl<'r, 'f, 'e, R: RngCore> Handler for ReconcilingScoreGivenTrace<'r, 'f, 'e, R> {
    for_each_value_type!(impl_reconciling_score_sample_methods);
    for_each_value_type!(impl_observe_methods);

    fn on_factor(&mut self, logw: f64) {
        self.trace.log_factors += logw;
    }

    fn finish(self) -> Trace {
        self.trace
    }
}

/// Score `model` against `base`, reconciling a differing address structure
/// instead of panicking (FG-20 / FG-21).
///
/// Addresses shared with `base` are replayed and re-scored under the current
/// model. Addresses the model introduces that are **not** in `base` (new
/// branches) are sampled fresh from their prior and their log-prior accumulated
/// — the RJMCMC-correct treatment of prior-proposed fresh dimensions. Addresses
/// in `base` that the model does **not** visit are reported as
/// [`ReconcileReport::vanished_addresses`] so the caller can drop their
/// contribution.
///
/// Returns `Err` with [`ErrorCode::AddressConflict`] only if the model visits
/// the same address twice.
///
/// # Example
///
/// ```rust
/// # use fugue::*;
/// # use fugue::runtime::interpreters::{score_given_trace_reconciled, PriorHandler};
/// # use rand::rngs::StdRng;
/// # use rand::SeedableRng;
/// let mut rng = StdRng::seed_from_u64(7);
/// let (_, base) = runtime::handler::run(
///     PriorHandler { rng: &mut rng, trace: Trace::default() },
///     sample(addr!("x"), Normal::new(0.0, 1.0).unwrap()),
/// );
///
/// // Model drops "x" and introduces "y": "y" is proposed fresh, "x" vanished.
/// let (_v, trace, report) = score_given_trace_reconciled(
///     base,
///     &mut rng,
///     sample(addr!("y"), Normal::new(0.0, 1.0).unwrap()),
/// ).unwrap();
/// assert_eq!(report.fresh_addresses, vec![addr!("y")]);
/// assert_eq!(report.vanished_addresses, vec![addr!("x")]);
/// assert!(trace.log_prior.is_finite());
/// ```
pub fn score_given_trace_reconciled<A, R: RngCore>(
    base: Trace,
    rng: &mut R,
    model: Model<A>,
) -> FugueResult<(A, Trace, ReconcileReport)> {
    let mut error: Option<FugueError> = None;
    let mut fresh_addresses: Vec<Address> = Vec::new();
    // Snapshot base addresses up front so we can compute vanished ones after the
    // run consumes the handler.
    let base_addresses: Vec<Address> = base.choices.keys().cloned().collect();
    let handler = ReconcilingScoreGivenTrace {
        rng,
        base,
        trace: Trace::default(),
        fresh: &mut fresh_addresses,
        error: &mut error,
    };
    let (a, trace) = run(handler, model);
    if let Some(e) = error {
        return Err(e);
    }
    // Vanished = present in the base trace but not visited by this model run.
    let vanished_addresses: Vec<Address> = base_addresses
        .into_iter()
        .filter(|addr| !trace.choices.contains_key(addr))
        .collect();
    Ok((
        a,
        trace,
        ReconcileReport {
            fresh_addresses,
            vanished_addresses,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::addr;
    use crate::core::distribution::*;
    use crate::core::model::{observe, sample, ModelExt};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn prior_handler_samples_and_accumulates() {
        let mut rng = StdRng::seed_from_u64(7);
        let (_val, trace) = crate::runtime::handler::run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
                .and_then(|x| observe(addr!("y"), Normal::new(x, 1.0).unwrap(), 0.5)),
        );
        assert!(trace.choices.contains_key(&addr!("x")));
        assert!(trace.log_prior.is_finite());
        assert!(trace.log_likelihood.is_finite());
    }

    #[test]
    fn replay_handler_reuses_values() {
        let mut rng = StdRng::seed_from_u64(8);
        let ((), base) = crate::runtime::handler::run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            sample(addr!("x"), Normal::new(0.0, 1.0).unwrap()).map(|_| ()),
        );

        let ((), replayed) = crate::runtime::handler::run(
            ReplayHandler {
                rng: &mut rng,
                base: base.clone(),
                trace: Trace::default(),
            },
            sample(addr!("x"), Normal::new(0.0, 1.0).unwrap()).map(|_| ()),
        );

        let x_base = base.get_f64(&addr!("x")).unwrap();
        let x_replay = replayed.get_f64(&addr!("x")).unwrap();
        assert_eq!(x_base, x_replay);
    }

    #[test]
    fn score_given_trace_scores_fixed_values() {
        let mut rng = StdRng::seed_from_u64(9);
        let (_a, base) = crate::runtime::handler::run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            sample(addr!("x"), Normal::new(0.0, 1.0).unwrap()),
        );

        let (_a2, scored) = crate::runtime::handler::run(
            ScoreGivenTrace {
                base: base.clone(),
                trace: Trace::default(),
            },
            sample(addr!("x"), Normal::new(0.0, 1.0).unwrap()),
        );

        // Should have same value and finite log_prior
        assert_eq!(scored.get_f64(&addr!("x")), base.get_f64(&addr!("x")));
        assert!(scored.log_prior.is_finite());
    }

    #[test]
    fn safe_variants_handle_mismatches() {
        // Build base trace with x as f64, then attempt to replay as bool
        let mut rng = StdRng::seed_from_u64(10);
        let (_a, base) = crate::runtime::handler::run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            sample(addr!("x"), Normal::new(0.0, 1.0).unwrap()),
        );

        // SafeReplayHandler should sample fresh value for bool and continue
        let (_b, t1) = crate::runtime::handler::run(
            SafeReplayHandler {
                rng: &mut rng,
                base: base.clone(),
                trace: Trace::default(),
                warn_on_mismatch: true,
            },
            sample(addr!("x"), Bernoulli::new(0.5).unwrap()),
        );
        assert!(t1.log_prior.is_finite());

        // SafeScoreGivenTrace should mark as invalid by adding -inf
        let (_c, t2) = crate::runtime::handler::run(
            SafeScoreGivenTrace {
                base: base.clone(),
                trace: Trace::default(),
                warn_on_error: true,
            },
            sample(addr!("x"), Bernoulli::new(0.5).unwrap()),
        );
        assert!(t2.log_prior.is_infinite());
    }

    #[test]
    fn handlers_cover_all_types_sample_and_observe() {
        // Model with multiple types
        let model = sample(addr!("f"), Normal::new(0.0, 1.0).unwrap())
            .and_then(|_| sample(addr!("b"), Bernoulli::new(0.6).unwrap()))
            .and_then(|_| sample(addr!("u64"), Poisson::new(3.0).unwrap()))
            .and_then(|_| sample(addr!("usz"), Categorical::new(vec![0.3, 0.7]).unwrap()))
            .and_then(|_| observe(addr!("f_obs"), Normal::new(0.0, 1.0).unwrap(), 0.1))
            .and_then(|_| observe(addr!("b_obs"), Bernoulli::new(0.4).unwrap(), true))
            .and_then(|_| observe(addr!("u64_obs"), Poisson::new(2.0).unwrap(), 1))
            .and_then(|_| {
                observe(
                    addr!("usz_obs"),
                    Categorical::new(vec![0.5, 0.5]).unwrap(),
                    1,
                )
            });

        let (_a, t) = crate::runtime::handler::run(
            PriorHandler {
                rng: &mut StdRng::seed_from_u64(100),
                trace: Trace::default(),
            },
            model,
        );
        assert!(t.get_f64(&addr!("f")).is_some());
        assert!(t.get_bool(&addr!("b")).is_some());
        assert!(t.get_u64(&addr!("u64")).is_some());
        assert!(t.get_usize(&addr!("usz")).is_some());
        assert!(t.log_likelihood.is_finite());

        // Build base and score given trace for all types
        let base = t.clone();
        let (_sv, scored) = crate::runtime::handler::run(
            ScoreGivenTrace {
                base: base.clone(),
                trace: Trace::default(),
            },
            sample(addr!("f"), Normal::new(0.0, 1.0).unwrap())
                .and_then(|_| sample(addr!("b"), Bernoulli::new(0.6).unwrap()))
                .and_then(|_| sample(addr!("u64"), Poisson::new(3.0).unwrap()))
                .and_then(|_| sample(addr!("usz"), Categorical::new(vec![0.3, 0.7]).unwrap())),
        );
        assert!(scored.log_prior.is_finite());

        // Safe replay mismatches for integer/categorical types
        let (_sv2, safe) = crate::runtime::handler::run(
            SafeReplayHandler {
                rng: &mut StdRng::seed_from_u64(101),
                base: base.clone(),
                trace: Trace::default(),
                warn_on_mismatch: true,
            },
            sample(addr!("u64"), Bernoulli::new(0.5).unwrap()),
        );
        assert!(safe.log_prior.is_finite());
    }

    #[test]
    fn safe_score_given_trace_warn_flag_branches() {
        let mut rng = StdRng::seed_from_u64(102);
        let (_a, base) = crate::runtime::handler::run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            sample(addr!("x"), Normal::new(0.0, 1.0).unwrap()),
        );
        // warn_on_error = false
        let (_b, t_false) = crate::runtime::handler::run(
            SafeScoreGivenTrace {
                base: base.clone(),
                trace: Trace::default(),
                warn_on_error: false,
            },
            sample(addr!("x"), Bernoulli::new(0.5).unwrap()),
        );
        assert!(t_false.log_prior.is_infinite());

        // warn_on_error = true
        let (_c, t_true) = crate::runtime::handler::run(
            SafeScoreGivenTrace {
                base: base.clone(),
                trace: Trace::default(),
                warn_on_error: true,
            },
            sample(addr!("x"), Bernoulli::new(0.5).unwrap()),
        );
        assert!(t_true.log_prior.is_infinite());
    }

    #[test]
    #[should_panic]
    fn replay_handler_panics_on_type_mismatch() {
        // Base has f64, replay expects bool -> panic as designed
        let mut rng = StdRng::seed_from_u64(103);
        let (_a, base) = crate::runtime::handler::run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            sample(addr!("x"), Normal::new(0.0, 1.0).unwrap()),
        );
        let (_b, _t) = crate::runtime::handler::run(
            ReplayHandler {
                rng: &mut rng,
                base: base.clone(),
                trace: Trace::default(),
            },
            sample(addr!("x"), Bernoulli::new(0.5).unwrap()),
        );
    }

    #[test]
    fn safe_replay_handler_samples_fresh_for_missing_address() {
        let mut rng = StdRng::seed_from_u64(104);
        // Base trace without address "z"
        let base = Trace::default();
        let (_a, t) = crate::runtime::handler::run(
            SafeReplayHandler {
                rng: &mut rng,
                base,
                trace: Trace::default(),
                warn_on_mismatch: true,
            },
            sample(addr!("z"), Normal::new(0.0, 1.0).unwrap()),
        );
        assert!(t.get_f64(&addr!("z")).is_some());
    }
}
