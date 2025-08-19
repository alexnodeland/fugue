//! Core model representation: a tiny monadic PPL in direct style.
//!
//! This module provides the core `Model<A>` type and operations for building probabilistic programs.
//! `Model<A>` represents a program that, when interpreted by a runtime handler, yields a value of
//! type `A`. The monadic interface (`pure`, `bind`, `map`) enables compositional probabilistic programs.
//!
//! ## Model Types
//!
//! A `Model<A>` can be one of four types:
//! - **Pure**: Contains a deterministic value
//! - **SampleF**: Samples from a probability distribution
//! - **ObserveF**: Conditions on observed data
//! - **FactorF**: Adds log-weight factors for soft constraints
//!
//! ## Basic Operations
//!
//! ### Creating Models
//!
//! ```rust
//! use fugue::*;
//!
//! // Pure deterministic value
//! let model = pure(42.0);
//!
//! // Sample from a distribution
//! let model = sample(addr!("x"), Normal { mu: 0.0, sigma: 1.0 });
//!
//! // Observe/condition on data
//! let model = observe(addr!("y"), Normal { mu: 0.0, sigma: 1.0 }, 2.5);
//!
//! // Add log-weight factor
//! let model = factor(0.5); // log(exp(0.5)) weight
//! ```
//!
//! ### Composing Models
//!
//! Models can be composed using monadic operations:
//!
//! ```rust
//! use fugue::*;
//!
//! let composed_model = sample(addr!("x"), Normal { mu: 0.0, sigma: 1.0 })
//!     .bind(|x| {
//!         sample(addr!("y"), Normal { mu: x, sigma: 0.5 })
//!             .map(move |y| x + y)
//!     });
//! ```
//!
//! ### Working with Collections
//!
//! ```rust
//! use fugue::*;
//!
//! // Create multiple independent samples
//! let models = vec![
//!     sample(addr!("x", 0), Normal { mu: 0.0, sigma: 1.0 }),
//!     sample(addr!("x", 1), Normal { mu: 0.0, sigma: 1.0 }),
//! ];
//! let combined = sequence_vec(models); // Model<Vec<f64>>
//!
//! // Apply a function to each item
//! let data = vec![1.0, 2.0, 3.0];
//! let model = traverse_vec(data, |x| {
//!     sample(addr!("noise", x as usize), Normal { mu: 0.0, sigma: 0.1 })
//!         .map(move |noise| x + noise)
//! });
//! ```
use crate::core::address::Address;
use crate::core::distribution::{DistributionF64, LogF64};

/// Core model type representing probabilistic computations.
///
/// A `Model<A>` represents a probabilistic program that yields a value of type `A` when executed.
/// Models are built using four fundamental operations:
///
/// - `Pure(a)`: Deterministic computation returning value `a`
/// - `SampleF`: Sample from a probability distribution at a named address
/// - `ObserveF`: Condition on observed data at a named address
/// - `FactorF`: Add a log-weight factor (for soft constraints)
///
/// Models form a monad, allowing compositional construction using `bind`, `map`, and related operations.
///
/// # Examples
///
/// ```rust
/// use fugue::*;
///
/// // Simple deterministic model
/// let model = pure(42.0);
///
/// // Probabilistic model with sampling
/// let model = sample(addr!("x"), Normal { mu: 0.0, sigma: 1.0 });
///
/// // Composed model
/// let model = sample(addr!("x"), Normal { mu: 0.0, sigma: 1.0 })
///     .bind(|x| pure(x * 2.0));
/// ```
pub enum Model<A> {
    /// A deterministic computation yielding a pure value.
    Pure(A),
    /// Sample from a distribution at the given address.
    SampleF {
        /// Unique identifier for this sampling site.
        addr: Address,
        /// Distribution to sample from.
        dist: Box<dyn DistributionF64>,
        /// Continuation function to apply to the sampled value.
        k: Box<dyn FnOnce(f64) -> Model<A> + Send + 'static>,
    },
    /// Observe/condition on a value at the given address.
    ObserveF {
        /// Unique identifier for this observation site.
        addr: Address,
        /// Distribution that generates the observed value.
        dist: Box<dyn DistributionF64>,
        /// The observed value to condition on.
        value: f64,
        /// Continuation function (always receives unit).
        k: Box<dyn FnOnce(()) -> Model<A> + Send + 'static>,
    },
    /// Add a log-weight factor to the model.
    FactorF {
        /// Log-weight to add to the model's total weight.
        logw: LogF64,
        /// Continuation function (always receives unit).
        k: Box<dyn FnOnce(()) -> Model<A> + Send + 'static>,
    },
}
/// Lift a deterministic value into the model monad.
///
/// Creates a `Model` that always returns the given value without any probabilistic behavior.
/// This is the unit operation for the model monad.
///
/// # Arguments
///
/// * `a` - The value to lift into a model
///
/// # Examples
///
/// ```rust
/// use fugue::*;
///
/// let model = pure(42.0);
/// // When executed, this model will always return 42.0
/// ```
pub fn pure<A>(a: A) -> Model<A> {
    Model::Pure(a)
}

/// Create a sampling site that draws a value from a probability distribution.
///
/// This is a fundamental operation for building probabilistic models. Each sampling site
/// must have a unique address to enable conditioning, inference, and replay.
///
/// # Arguments
///
/// * `addr` - Unique address identifying this sampling site
/// * `dist` - Probability distribution to sample from
///
/// # Returns
///
/// A `Model<f64>` that, when executed, samples a value from the given distribution.
///
/// # Examples
///
/// ```rust
/// use fugue::*;
///
/// // Sample from a standard normal distribution
/// let model = sample(addr!("x"), Normal { mu: 0.0, sigma: 1.0 });
///
/// // Sample from a uniform distribution
/// let model = sample(addr!("u"), Uniform { low: 0.0, high: 1.0 });
///
/// // Sample with indexed addresses for multiple samples
/// let model = sample(addr!("data", 0), Normal { mu: 0.0, sigma: 1.0 });
/// ```
pub fn sample(addr: Address, dist: impl DistributionF64 + 'static) -> Model<f64> {
    Model::SampleF {
        addr,
        dist: Box::new(dist),
        k: Box::new(pure),
    }
}

/// Create an observation site that conditions the model on observed data.
///
/// This operation allows incorporating observed data into probabilistic models by conditioning
/// on the fact that a particular random variable took on a specific observed value.
/// The log-probability of the observation under the given distribution contributes to the
/// model's overall log-weight.
///
/// # Arguments
///
/// * `addr` - Unique address identifying this observation site
/// * `dist` - Probability distribution that generated the observed value
/// * `value` - The observed value to condition on
///
/// # Returns
///
/// A `Model<()>` that adds the log-probability of the observation to the model's weight.
///
/// # Examples
///
/// ```rust
/// use fugue::*;
///
/// // Observe that y = 2.5 given a normal distribution
/// let model = observe(addr!("y"), Normal { mu: 1.0, sigma: 0.5 }, 2.5);
///
/// // Condition on multiple observations
/// let model = observe(addr!("y1"), Normal { mu: 0.0, sigma: 1.0 }, 1.2)
///     .bind(|_| observe(addr!("y2"), Normal { mu: 0.0, sigma: 1.0 }, -0.8));
/// ```
pub fn observe(addr: Address, dist: impl DistributionF64 + 'static, value: f64) -> Model<()> {
    Model::ObserveF {
        addr,
        dist: Box::new(dist),
        value,
        k: Box::new(pure),
    }
}

/// Add an unnormalized log-weight factor to the model.
///
/// Factors allow encoding soft constraints or arbitrary log-probability contributions
/// to the model. They are particularly useful for:
/// - Encoding constraints that should be "mostly satisfied"
/// - Adding custom log-likelihood terms
/// - Implementing rejection sampling (using negative infinity)
///
/// # Arguments
///
/// * `logw` - Log-weight to add to the model's total weight
///
/// # Returns
///
/// A `Model<()>` that contributes the given log-weight.
///
/// # Examples
///
/// ```rust
/// use fugue::*;
///
/// // Add positive log-weight (increases probability)
/// let model = factor(1.0); // Adds log(e) = 1.0 to weight
///
/// // Add negative log-weight (decreases probability)
/// let model = factor(-2.0); // Subtracts 2.0 from log-weight
///
/// // Reject/fail (zero probability)
/// let model = factor(f64::NEG_INFINITY);
///
/// // Soft constraint: prefer values near zero
/// let x = 5.0;
/// let soft_constraint = factor(-0.5 * x * x); // Gaussian-like penalty
/// ```
pub fn factor(logw: LogF64) -> Model<()> {
    Model::FactorF {
        logw,
        k: Box::new(pure),
    }
}
/// Monadic operations for composing and transforming models.
///
/// This trait provides the fundamental monadic operations that enable compositional
/// probabilistic programming. All models implement this trait, allowing them to be
/// chained and transformed in a principled way.
///
/// # Core Operations
///
/// - [`bind`](Self::bind): Monadic bind (>>=) - chains dependent computations
/// - [`map`](Self::map): Functor map - transforms the result without adding probabilistic behavior
/// - [`and_then`](Self::and_then): Alias for `bind` for those familiar with Rust's `Option`/`Result`
///
/// # Examples
///
/// ```rust
/// use fugue::*;
///
/// // Using bind for dependent sampling
/// let model = sample(addr!("x"), Normal { mu: 0.0, sigma: 1.0 })
///     .bind(|x| sample(addr!("y"), Normal { mu: x, sigma: 0.5 }));
///
/// // Using map for transformations
/// let model = sample(addr!("x"), Normal { mu: 0.0, sigma: 1.0 })
///     .map(|x| x * 2.0 + 1.0);
///
/// // Chaining multiple operations
/// let model = sample(addr!("x"), Uniform { low: 0.0, high: 1.0 })
///     .bind(|x| {
///         if x > 0.5 {
///             sample(addr!("high"), Normal { mu: 10.0, sigma: 1.0 })
///         } else {
///             sample(addr!("low"), Normal { mu: -10.0, sigma: 1.0 })
///         }
///     })
///     .map(|result| result.abs());
/// ```
pub trait ModelExt<A>: Sized {
    /// Monadic bind operation (>>=).
    ///
    /// Chains two probabilistic computations where the second depends on the result of the first.
    /// This is the fundamental operation for building complex probabilistic models from simpler parts.
    ///
    /// # Arguments
    ///
    /// * `k` - Function that takes the result of this model and returns a new model
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fugue::*;
    ///
    /// // Dependent sampling: y depends on x
    /// let model = sample(addr!("x"), Normal { mu: 0.0, sigma: 1.0 })
    ///     .bind(|x| sample(addr!("y"), Normal { mu: x, sigma: 0.1 }));
    /// ```
    fn bind<B>(self, k: impl FnOnce(A) -> Model<B> + Send + 'static) -> Model<B>;
    
    /// Apply a function to transform the result of this model.
    ///
    /// This is the functor map operation - it transforms the output of a model without
    /// adding any additional probabilistic behavior.
    ///
    /// # Arguments
    ///
    /// * `f` - Function to apply to the model's result
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fugue::*;
    ///
    /// // Transform the sampled value
    /// let model = sample(addr!("x"), Normal { mu: 0.0, sigma: 1.0 })
    ///     .map(|x| x.exp()); // Apply exponential function
    /// ```
    fn map<B>(self, f: impl FnOnce(A) -> B + Send + 'static) -> Model<B> {
        self.bind(|a| pure(f(a)))
    }
    
    /// Alias for `bind` - chains dependent probabilistic computations.
    ///
    /// This method provides a more familiar interface for Rust developers used to
    /// `Option::and_then` and `Result::and_then`.
    ///
    /// # Arguments
    ///
    /// * `k` - Function that takes the result of this model and returns a new model
    fn and_then<B>(self, k: impl FnOnce(A) -> Model<B> + Send + 'static) -> Model<B> {
        self.bind(k)
    }
}

impl<A: 'static> ModelExt<A> for Model<A> {
    fn bind<B>(self, k: impl FnOnce(A) -> Model<B> + Send + 'static) -> Model<B> {
        match self {
            Model::Pure(a) => k(a),
            Model::SampleF { addr, dist, k: k1 } => Model::SampleF {
                addr,
                dist,
                k: Box::new(move |x| k1(x).bind(k)),
            },
            Model::ObserveF {
                addr,
                dist,
                value,
                k: k1,
            } => Model::ObserveF {
                addr,
                dist,
                value,
                k: Box::new(move |u| k1(u).bind(k)),
            },
            Model::FactorF { logw, k: k1 } => Model::FactorF {
                logw,
                k: Box::new(move |u| k1(u).bind(k)),
            },
        }
    }
}
/// Combine two independent models into a model of their paired results.
///
/// This operation runs both models and combines their results into a tuple.
/// The models are executed independently (neither depends on the other's result).
///
/// # Arguments
///
/// * `ma` - First model to execute
/// * `mb` - Second model to execute
///
/// # Returns
///
/// A `Model<(A, B)>` containing the paired results.
///
/// # Examples
///
/// ```rust
/// use fugue::*;
///
/// // Sample two independent random variables
/// let x_model = sample(addr!("x"), Normal { mu: 0.0, sigma: 1.0 });
/// let y_model = sample(addr!("y"), Uniform { low: 0.0, high: 1.0 });
/// let paired = zip(x_model, y_model); // Model<(f64, f64)>
///
/// // Can be used with any model types
/// let mixed = zip(pure(42.0), sample(addr!("z"), Exponential { rate: 1.0 }));
/// ```
pub fn zip<A: Send + 'static, B: Send + 'static>(ma: Model<A>, mb: Model<B>) -> Model<(A, B)> {
    ma.bind(|a| mb.map(move |b| (a, b)))
}

/// Execute a vector of models and collect their results into a single model of a vector.
///
/// This function takes a collection of independent models and runs them all,
/// collecting their results into a vector. This is useful for running multiple
/// similar probabilistic computations.
///
/// # Arguments
///
/// * `models` - Vector of models to execute
///
/// # Returns
///
/// A `Model<Vec<A>>` containing all the results in order.
///
/// # Examples
///
/// ```rust
/// use fugue::*;
///
/// // Create multiple independent samples
/// let models = vec![
///     sample(addr!("x", 0), Normal { mu: 0.0, sigma: 1.0 }),
///     sample(addr!("x", 1), Normal { mu: 1.0, sigma: 1.0 }),
///     sample(addr!("x", 2), Normal { mu: 2.0, sigma: 1.0 }),
/// ];
/// let all_samples = sequence_vec(models); // Model<Vec<f64>>
///
/// // Mix deterministic and probabilistic models
/// let mixed_models = vec![
///     pure(1.0),
///     sample(addr!("random"), Uniform { low: 0.0, high: 1.0 }),
///     pure(3.0),
/// ];
/// let results = sequence_vec(mixed_models);
/// ```
pub fn sequence_vec<A: Send + 'static>(models: Vec<Model<A>>) -> Model<Vec<A>> {
    models.into_iter().fold(pure(Vec::new()), |acc, m| {
        zip(acc, m).map(|(mut v, a)| {
            v.push(a);
            v
        })
    })
}

/// Apply a function that produces models to each item in a vector, collecting the results.
///
/// This is a higher-order function that maps each item in the input vector through a function
/// that produces a model, then sequences all the resulting models into a single model of a vector.
/// This is equivalent to `sequence_vec(items.map(f))` but more convenient.
///
/// # Arguments
///
/// * `items` - Vector of input items to process
/// * `f` - Function that takes an item and produces a model
///
/// # Returns
///
/// A `Model<Vec<A>>` containing all the results in order.
///
/// # Examples
///
/// ```rust
/// use fugue::*;
///
/// // Add noise to each data point
/// let data = vec![1.0, 2.0, 3.0];
/// let noisy_data = traverse_vec(data, |x| {
///     sample(addr!("noise", x as usize), Normal { mu: 0.0, sigma: 0.1 })
///         .map(move |noise| x + noise)
/// });
///
/// // Create observations for each data point
/// let observations = vec![1.2, 2.1, 2.9];
/// let model = traverse_vec(observations, |obs| {
///     observe(addr!("y", obs as usize), Normal { mu: 2.0, sigma: 0.5 }, obs)
/// });
/// ```
pub fn traverse_vec<T, A: Send + 'static>(
    items: Vec<T>,
    f: impl Fn(T) -> Model<A> + Send + Sync + 'static,
) -> Model<Vec<A>> {
    sequence_vec(items.into_iter().map(|t| f(t)).collect())
}

/// Conditional execution: fail with zero probability when predicate is false.
///
/// Guards provide a way to enforce hard constraints in probabilistic models.
/// When the predicate is true, the model continues normally. When false,
/// the model receives negative infinite log-weight, effectively ruling out
/// that execution path.
///
/// # Arguments
///
/// * `pred` - Boolean predicate to check
///
/// # Returns
///
/// A `Model<()>` that either succeeds (pred=true) or fails with zero probability (pred=false).
///
/// # Examples
///
/// ```rust
/// use fugue::*;
///
/// // Ensure a sampled value is positive
/// let model = sample(addr!("x"), Normal { mu: 0.0, sigma: 1.0 })
///     .bind(|x| {
///         guard(x > 0.0).bind(move |_| pure(x))
///     });
///
/// // Multiple constraints
/// let model = sample(addr!("x"), Uniform { low: -2.0, high: 2.0 })
///     .bind(|x| {
///         guard(x > -1.0).bind(move |_|
///             guard(x < 1.0).bind(move |_| pure(x * x))
///         )
///     });
/// ```
pub fn guard(pred: bool) -> Model<()> {
    if pred {
        pure(())
    } else {
        factor(f64::NEG_INFINITY)
    }
}
