//! Execution traces capturing choices and accumulated log-weights.
//!
//! This module provides data structures for recording the execution history of
//! probabilistic models. Traces capture:
//! - **Choices**: Named random variable assignments with their log-probabilities
//! - **Log-weights**: Accumulated prior, likelihood, and factor contributions
//!
//! Traces enable key capabilities in probabilistic programming:
//! - **Replay**: Re-executing models with the same random choices
//! - **Conditioning**: Computing model probabilities given fixed data
//! - **Inference**: Tracking and updating random variable assignments
//! - **Debugging**: Understanding model execution flow and weights
//!
//! ## Structure
//!
//! A trace consists of:
//! - A map of choices keyed by address
//! - Separate accumulators for prior, likelihood, and factor log-weights
//!
//! The total log-weight combines all three components and represents the
//! unnormalized log-probability of the execution.
//!
//! # Examples
//!
//! ```rust
//! use fugue::*;
//! use rand::rngs::StdRng;
//! use rand::SeedableRng;
//!
//! // Execute a model and examine its trace
//! let model = sample(addr!("mu"), Normal { mu: 0.0, sigma: 1.0 })
//!     .bind(|mu| observe(addr!("y"), Normal { mu, sigma: 0.5 }, 2.0));
//!
//! let mut rng = StdRng::seed_from_u64(42);
//! let (_, trace) = runtime::handler::run(
//!     PriorHandler { rng: &mut rng, trace: Trace::default() },
//!     model,
//! );
//!
//! println!("Prior log-weight: {}", trace.log_prior);
//! println!("Likelihood log-weight: {}", trace.log_likelihood);
//! println!("Total log-weight: {}", trace.total_log_weight());
//!
//! // Access specific choices
//! if let Some(choice) = trace.choices.get(&addr!("mu")) {
//!     println!("Sampled mu: {:?}", choice.value);
//! }
//! ```
use crate::core::address::Address;
use std::collections::BTreeMap;

/// Value stored at a choice site in an execution trace.
///
/// Different types of random variables can be stored in traces, though
/// currently only `f64` values are used by the built-in distributions.
/// Additional variants support future extensions to other value types.
///
/// # Variants
///
/// * `F64` - Floating-point values (most common)
/// * `I64` - Integer values
/// * `Bool` - Boolean values
///
/// # Examples
///
/// ```rust
/// use fugue::*;
///
/// // Most distributions use F64 values
/// let normal_value = ChoiceValue::F64(1.23);
/// let discrete_value = ChoiceValue::F64(3.0); // Categorical/Poisson as f64
///
/// // Future extensions might use other types
/// let integer_value = ChoiceValue::I64(42);
/// let boolean_value = ChoiceValue::Bool(true);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub enum ChoiceValue {
    /// Floating-point value (used by all current distributions).
    F64(f64),
    /// Integer value (for future discrete distributions).
    I64(i64),
    /// Boolean value (for future boolean distributions).
    Bool(bool),
}

/// A recorded choice made during model execution.
///
/// Each choice represents a random variable assignment at a specific address,
/// along with its log-probability under the distribution that generated it.
/// Choices are the building blocks of execution traces.
///
/// # Fields
///
/// * `addr` - Address identifying where this choice was made
/// * `value` - The value that was chosen/assigned
/// * `logp` - Log-probability of this value under the generating distribution
///
/// # Examples
///
/// ```rust
/// use fugue::*;
///
/// // Choices are typically created by handlers during execution
/// let choice = Choice {
///     addr: addr!("x"),
///     value: ChoiceValue::F64(1.5),
///     logp: -0.92, // log-probability under some distribution
/// };
///
/// println!("Choice at {}: {:?} (logp: {})", choice.addr, choice.value, choice.logp);
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
/// A trace records the full execution history of a probabilistic model, including
/// all random choices made and the accumulated log-weights from different sources.
/// Traces are essential for:
///
/// - **Replay**: Re-executing models with the same random choices
/// - **Scoring**: Computing log-probabilities of specific executions
/// - **Inference**: Updating random variables while keeping others fixed
/// - **Debugging**: Understanding model behavior and weight contributions
///
/// ## Log-weight Components
///
/// The total log-weight is decomposed into three components:
/// - **Prior**: Log-probabilities of sampled values under their prior distributions
/// - **Likelihood**: Log-probabilities of observed data given the model
/// - **Factors**: Additional log-weight contributions from factor statements
///
/// # Fields
///
/// * `choices` - Map from addresses to the choices made at those sites
/// * `log_prior` - Accumulated log-prior probability
/// * `log_likelihood` - Accumulated log-likelihood of observations
/// * `log_factors` - Accumulated log-weight from factor statements
///
/// # Examples
///
/// ```rust
/// use fugue::*;
/// use rand::rngs::StdRng;
/// use rand::SeedableRng;
///
/// // Create a model with different weight sources
/// let model = sample(addr!("theta"), Normal { mu: 0.0, sigma: 1.0 })
///     .bind(|theta| {
///         observe(addr!("y"), Normal { mu: theta, sigma: 0.5 }, 1.5)
///             .bind(move |_| factor(-0.1).bind(move |_| pure(theta)))
///     });
///
/// let mut rng = StdRng::seed_from_u64(42);
/// let (theta, trace) = runtime::handler::run(
///     PriorHandler { rng: &mut rng, trace: Trace::default() },
///     model,
/// );
///
/// println!("Sampled theta: {}", theta);
/// println!("Prior contribution: {}", trace.log_prior);
/// println!("Likelihood contribution: {}", trace.log_likelihood);
/// println!("Factor contribution: {}", trace.log_factors);
/// println!("Total log-weight: {}", trace.total_log_weight());
///
/// // Access individual choices
/// if let Some(choice) = trace.choices.get(&addr!("theta")) {
///     println!("Theta choice: {:?}", choice.value);
/// }
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
    /// and represents the unnormalized log-probability of this particular execution
    /// path through the model.
    ///
    /// # Returns
    ///
    /// The sum of log_prior + log_likelihood + log_factors.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fugue::*;
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
}
