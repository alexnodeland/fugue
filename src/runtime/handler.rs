//! Generic handler interface and model execution engine.
//!
//! This module provides the core abstraction for interpreting probabilistic models.
//! The `Handler` trait defines how to process the three fundamental effects in
//! probabilistic programming: sampling, observation, and factoring. Different
//! handler implementations enable different execution modes (prior sampling,
//! conditioning, scoring, etc.).
//!
//! ## Handler Pattern
//!
//! Handlers implement the algebraic effects pattern with **full type safety**.
//! Each effect is handled by type-specific methods that match the natural return
//! types of distributions:
//! - `on_sample_f64` for continuous distributions (Normal, Beta, etc.)
//! - `on_sample_bool` for Bernoulli (returns bool directly!)
//! - `on_sample_u64` for count distributions (Poisson, Binomial)
//! - `on_sample_usize` for categorical distributions (safe indexing!)
//!
//! This design enables:
//! - **Type Safety**: Handlers work with natural types, not just f64
//! - **Modularity**: Different handlers for different purposes
//! - **Composability**: Handlers can be combined and extended
//! - **Testability**: Effects can be mocked and controlled
//! - **Performance**: Zero overhead from unnecessary type conversions
//!
//! ## Execution Model
//!
//! The `run` function acts as the interpreter, walking through a `Model` and
//! dispatching effects to the handler. It returns both the model's final value
//! and the accumulated execution trace.
//!
//! # Examples
//!
//! ```rust
//! use fugue::*;
//! use rand::rngs::StdRng;
//! use rand::SeedableRng;
//!
//! // Run type-safe models with prior sampling
//! let normal_model: Model<f64> = sample(addr!("x"), Normal { mu: 0.0, sigma: 1.0 });
//! let coin_model: Model<bool> = sample(addr!("coin"), Bernoulli { p: 0.5 });
//! let count_model: Model<u64> = sample(addr!("events"), Poisson { lambda: 3.0 });
//!
//! let mut rng = StdRng::seed_from_u64(42);
//!
//! // Handler automatically dispatches to correct type-specific method
//! let (value, trace) = runtime::handler::run(
//!     PriorHandler {
//!         rng: &mut rng,
//!         trace: Trace::default(),
//!     },
//!     coin_model, // Handler calls on_sample_bool, returns bool
//! );
//! println!("Coin flip: {}, log-weight: {}", value, trace.total_log_weight());
//! ```
use crate::core::address::Address;
use crate::core::distribution::Distribution;
use crate::core::model::Model;
use crate::runtime::trace::Trace;
/// Trait for handling probabilistic effects during model execution.
///
/// Handlers define the interpretation of the three fundamental effects in probabilistic
/// programming. Different handler implementations enable different execution modes:
/// - Prior sampling (draw fresh random values)
/// - Replay (use values from an existing trace)
/// - Scoring (compute log-probability of a fixed trace)
///
/// # Required Methods
///
/// ## Sampling Methods (Type-Specific)
/// - [`on_sample_f64`](Self::on_sample_f64): Handle f64 sampling (continuous distributions)
/// - [`on_sample_bool`](Self::on_sample_bool): Handle bool sampling (Bernoulli)
/// - [`on_sample_u64`](Self::on_sample_u64): Handle u64 sampling (Poisson, Binomial)  
/// - [`on_sample_usize`](Self::on_sample_usize): Handle usize sampling (Categorical)
///
/// ## Observation Methods (Type-Specific)
/// - [`on_observe_f64`](Self::on_observe_f64): Handle f64 observations
/// - [`on_observe_bool`](Self::on_observe_bool): Handle bool observations
/// - [`on_observe_u64`](Self::on_observe_u64): Handle u64 observations
/// - [`on_observe_usize`](Self::on_observe_usize): Handle usize observations
///
/// ## Other Methods
/// - [`on_factor`](Self::on_factor): Handle arbitrary log-weight contributions
/// - [`finish`](Self::finish): Finalize and return the accumulated trace
///
/// # Examples
///
/// ```rust
/// use fugue::*;
/// use rand::rngs::StdRng;
/// use rand::SeedableRng;
///
/// // Use a built-in handler
/// let mut rng = StdRng::seed_from_u64(42);
/// let handler = PriorHandler {
///     rng: &mut rng,
///     trace: Trace::default(),
/// };
///
/// let model = sample(addr!("x"), Normal { mu: 0.0, sigma: 1.0 });
/// let (result, trace) = runtime::handler::run(handler, model);
/// ```
pub trait Handler {
    /// Handle an f64 sampling operation (continuous distributions).
    fn on_sample_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>) -> f64;

    /// Handle a bool sampling operation (Bernoulli).
    fn on_sample_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>) -> bool;

    /// Handle a u64 sampling operation (Poisson, Binomial).
    fn on_sample_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>) -> u64;

    /// Handle a usize sampling operation (Categorical).
    fn on_sample_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>) -> usize;

    /// Handle an f64 observation operation.
    fn on_observe_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>, value: f64);

    /// Handle a bool observation operation.
    fn on_observe_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>, value: bool);

    /// Handle a u64 observation operation.
    fn on_observe_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>, value: u64);

    /// Handle a usize observation operation.
    fn on_observe_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>, value: usize);

    /// Handle a factor operation.
    ///
    /// This method is called when the model encounters a `factor` operation.
    /// The handler typically adds the log-weight to the trace.
    ///
    /// # Arguments
    ///
    /// * `logw` - Log-weight to add to the model's total weight
    fn on_factor(&mut self, logw: f64);

    /// Finalize the handler and return the accumulated trace.
    ///
    /// This method is called after model execution completes to retrieve
    /// the final trace containing all choices and log-weights.
    fn finish(self) -> Trace
    where
        Self: Sized;
}

/// Execute a probabilistic model using the given handler.
///
/// This is the core execution engine for probabilistic models. It interprets
/// a `Model<A>` by dispatching effects to the provided handler and returns
/// both the model's final result and the accumulated execution trace.
///
/// The execution proceeds by pattern matching on the model structure:
/// - `Pure` values are returned directly
/// - `SampleF64` operations are handled by calling `handler.on_sample_f64`
/// - `SampleBool` operations are handled by calling `handler.on_sample_bool`
/// - `SampleU64` operations are handled by calling `handler.on_sample_u64`
/// - `SampleUsize` operations are handled by calling `handler.on_sample_usize`
/// - `ObserveF64` operations are handled by calling `handler.on_observe_f64`
/// - `ObserveBool` operations are handled by calling `handler.on_observe_bool`
/// - `ObserveU64` operations are handled by calling `handler.on_observe_u64`
/// - `ObserveUsize` operations are handled by calling `handler.on_observe_usize`
/// - `Factor` operations are handled by calling `handler.on_factor`
///
/// # Arguments
///
/// * `h` - Handler that defines how to interpret effects
/// * `m` - Model to execute
///
/// # Returns
///
/// A tuple containing:
/// - The final result of type `A` produced by the model
/// - The execution trace recording all choices and weights
///
/// # Examples
///
/// ```rust
/// use fugue::*;
/// use rand::rngs::StdRng;
/// use rand::SeedableRng;
///
/// // Execute a simple model
/// let model = sample(addr!("x"), Normal { mu: 0.0, sigma: 1.0 })
///     .map(|x| x * 2.0);
///
/// let mut rng = StdRng::seed_from_u64(42);
/// let handler = PriorHandler {
///     rng: &mut rng,
///     trace: Trace::default(),
/// };
///
/// let (result, trace) = runtime::handler::run(handler, model);
/// println!("Result: {}, Log-weight: {}", result, trace.total_log_weight());
/// ```
pub fn run<A>(mut h: impl Handler, m: Model<A>) -> (A, Trace) {
    fn go<A>(h: &mut impl Handler, m: Model<A>) -> A {
        match m {
            Model::Pure(a) => a,
            Model::SampleF64 { addr, dist, k } => {
                let x = h.on_sample_f64(&addr, &*dist);
                go(h, k(x))
            }
            Model::SampleBool { addr, dist, k } => {
                let x = h.on_sample_bool(&addr, &*dist);
                go(h, k(x))
            }
            Model::SampleU64 { addr, dist, k } => {
                let x = h.on_sample_u64(&addr, &*dist);
                go(h, k(x))
            }
            Model::SampleUsize { addr, dist, k } => {
                let x = h.on_sample_usize(&addr, &*dist);
                go(h, k(x))
            }
            Model::ObserveF64 {
                addr,
                dist,
                value,
                k,
            } => {
                h.on_observe_f64(&addr, &*dist, value);
                go(h, k(()))
            }
            Model::ObserveBool {
                addr,
                dist,
                value,
                k,
            } => {
                h.on_observe_bool(&addr, &*dist, value);
                go(h, k(()))
            }
            Model::ObserveU64 {
                addr,
                dist,
                value,
                k,
            } => {
                h.on_observe_u64(&addr, &*dist, value);
                go(h, k(()))
            }
            Model::ObserveUsize {
                addr,
                dist,
                value,
                k,
            } => {
                h.on_observe_usize(&addr, &*dist, value);
                go(h, k(()))
            }
            Model::Factor { logw, k } => {
                h.on_factor(logw);
                go(h, k(()))
            }
        }
    }
    let a = go(&mut h, m);
    let t = h.finish();
    (a, t)
}
