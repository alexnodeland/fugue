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
//! Handlers implement the algebraic effects pattern, where each effect
//! (`sample`, `observe`, `factor`) is handled by a specific method. This design
//! enables:
//! - **Modularity**: Different handlers for different purposes
//! - **Composability**: Handlers can be combined and extended
//! - **Testability**: Effects can be mocked and controlled
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
//! // Run a model with prior sampling
//! let model = sample(addr!("x"), Normal { mu: 0.0, sigma: 1.0 });
//! let mut rng = StdRng::seed_from_u64(42);
//! let (value, trace) = runtime::handler::run(
//!     PriorHandler {
//!         rng: &mut rng,
//!         trace: Trace::default(),
//!     },
//!     model,
//! );
//! println!("Sampled value: {}, log-weight: {}", value, trace.total_log_weight());
//! ```
use crate::core::address::Address;
use crate::core::distribution::DistributionF64;
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
/// - [`on_sample`](Self::on_sample): Handle sampling from a distribution
/// - [`on_observe`](Self::on_observe): Handle conditioning on observed data
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
    /// Handle a sampling operation.
    ///
    /// This method is called when the model encounters a `sample` operation.
    /// The handler decides what value to return and may record the choice in a trace.
    ///
    /// # Arguments
    ///
    /// * `addr` - Address identifying the sampling site
    /// * `dist` - Distribution to sample from
    ///
    /// # Returns
    ///
    /// The value to use for this sampling site.
    fn on_sample(&mut self, addr: &Address, dist: &dyn DistributionF64) -> f64;

    /// Handle an observation operation.
    ///
    /// This method is called when the model encounters an `observe` operation.
    /// The handler typically adds the log-probability of the observation to the trace.
    ///
    /// # Arguments
    ///
    /// * `addr` - Address identifying the observation site
    /// * `dist` - Distribution that generated the observed value
    /// * `value` - The observed value
    fn on_observe(&mut self, addr: &Address, dist: &dyn DistributionF64, value: f64);

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
/// - `SampleF` operations are handled by calling `handler.on_sample`
/// - `ObserveF` operations are handled by calling `handler.on_observe`
/// - `FactorF` operations are handled by calling `handler.on_factor`
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
            Model::SampleF { addr, dist, k } => {
                let x = h.on_sample(&addr, &*dist);
                go(h, k(x))
            }
            Model::ObserveF {
                addr,
                dist,
                value,
                k,
            } => {
                h.on_observe(&addr, &*dist, value);
                go(h, k(()))
            }
            Model::FactorF { logw, k } => {
                h.on_factor(logw);
                go(h, k(()))
            }
        }
    }
    let a = go(&mut h, m);
    let t = h.finish();
    (a, t)
}
