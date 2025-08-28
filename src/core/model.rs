#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/docs/api/core/model.md"))]
use crate::core::address::Address;
use crate::core::distribution::{Distribution, LogF64};

/// `Model<A>` represents a probabilistic program that yields a value of type `A` when executed by a handler.
/// Models are built from four variants: `Pure`, `Sample*`, `Observe*`, and `Factor`.
///
/// Example:
/// ```rust
/// # use fugue::*;
/// // Deterministic value
/// let m = pure(42.0);
///
/// // Sample from distribution
/// let s = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
///
/// // Dependent sampling
/// let chain = s.bind(|x| sample(addr!("y"), Normal::new(x, 0.5).unwrap()));
/// ```
pub enum Model<A> {
    /// A deterministic computation yielding a pure value.
    Pure(A),
    /// Sample from an f64 distribution (continuous distributions).
    SampleF64 {
        /// Unique identifier for this sampling site.
        addr: Address,
        /// Distribution to sample from.
        dist: Box<dyn Distribution<f64>>,
        /// Continuation function to apply to the sampled value.
        k: Box<dyn FnOnce(f64) -> Model<A> + Send + 'static>,
    },
    /// Sample from a bool distribution (Bernoulli).
    SampleBool {
        /// Unique identifier for this sampling site.
        addr: Address,
        /// Distribution to sample from.
        dist: Box<dyn Distribution<bool>>,
        /// Continuation function to apply to the sampled value.
        k: Box<dyn FnOnce(bool) -> Model<A> + Send + 'static>,
    },
    /// Sample from a u64 distribution (Poisson, Binomial).
    SampleU64 {
        /// Unique identifier for this sampling site.
        addr: Address,
        /// Distribution to sample from.
        dist: Box<dyn Distribution<u64>>,
        /// Continuation function to apply to the sampled value.
        k: Box<dyn FnOnce(u64) -> Model<A> + Send + 'static>,
    },
    /// Sample from a usize distribution (Categorical).
    SampleUsize {
        /// Unique identifier for this sampling site.
        addr: Address,
        /// Distribution to sample from.
        dist: Box<dyn Distribution<usize>>,
        /// Continuation function to apply to the sampled value.
        k: Box<dyn FnOnce(usize) -> Model<A> + Send + 'static>,
    },
    /// Observe/condition on an f64 value.
    ObserveF64 {
        /// Unique identifier for this observation site.
        addr: Address,
        /// Distribution that generates the observed value.
        dist: Box<dyn Distribution<f64>>,
        /// The observed value to condition on.
        value: f64,
        /// Continuation function (always receives unit).
        k: Box<dyn FnOnce(()) -> Model<A> + Send + 'static>,
    },
    /// Observe/condition on a bool value.
    ObserveBool {
        /// Unique identifier for this observation site.
        addr: Address,
        /// Distribution that generates the observed value.
        dist: Box<dyn Distribution<bool>>,
        /// The observed value to condition on.
        value: bool,
        /// Continuation function (always receives unit).
        k: Box<dyn FnOnce(()) -> Model<A> + Send + 'static>,
    },
    /// Observe/condition on a u64 value.
    ObserveU64 {
        /// Unique identifier for this observation site.
        addr: Address,
        /// Distribution that generates the observed value.
        dist: Box<dyn Distribution<u64>>,
        /// The observed value to condition on.
        value: u64,
        /// Continuation function (always receives unit).
        k: Box<dyn FnOnce(()) -> Model<A> + Send + 'static>,
    },
    /// Observe/condition on a usize value.
    ObserveUsize {
        /// Unique identifier for this observation site.
        addr: Address,
        /// Distribution that generates the observed value.
        dist: Box<dyn Distribution<usize>>,
        /// The observed value to condition on.
        value: usize,
        /// Continuation function (always receives unit).
        k: Box<dyn FnOnce(()) -> Model<A> + Send + 'static>,
    },
    /// Add a log-weight factor to the model.
    Factor {
        /// Log-weight to add to the model's total weight.
        logw: LogF64,
        /// Continuation function (always receives unit).
        k: Box<dyn FnOnce(()) -> Model<A> + Send + 'static>,
    },
}

/// Lift a deterministic value, `a`, into the model monad.
/// Creates a `Model` that always returns the given value, `a`, without any probabilistic behavior.
/// This is the unit operation for the model monad.
///
/// Example:
/// ```rust
/// # use fugue::*;
///
/// let model = pure(42.0);
/// // When executed, this model will always return 42.0
/// ```
pub fn pure<A>(a: A) -> Model<A> {
    Model::Pure(a)
}
/// Sample from an f64 distribution (continuous distributions).
///
/// Example:
/// ```rust
/// # use fugue::*;
///
/// let model = sample_f64(addr!("x"), Normal::new(0.0, 1.0).unwrap());
/// ```
pub fn sample_f64(addr: Address, dist: impl Distribution<f64> + 'static) -> Model<f64> {
    Model::SampleF64 {
        addr,
        dist: Box::new(dist),
        k: Box::new(pure),
    }
}
/// Sample from a bool distribution (Bernoulli).
///
/// Example:
/// ```rust
/// # use fugue::*;
///
/// let model = sample_bool(addr!("coin"), Bernoulli::new(0.5).unwrap());
/// ```
pub fn sample_bool(addr: Address, dist: impl Distribution<bool> + 'static) -> Model<bool> {
    Model::SampleBool {
        addr,
        dist: Box::new(dist),
        k: Box::new(pure),
    }
}
/// Sample from a u64 distribution (Poisson, Binomial).
///
/// Example:
/// ```rust
/// # use fugue::*;
///
/// let model = sample_u64(addr!("count"), Poisson::new(3.0).unwrap());
/// ```
pub fn sample_u64(addr: Address, dist: impl Distribution<u64> + 'static) -> Model<u64> {
    Model::SampleU64 {
        addr,
        dist: Box::new(dist),
        k: Box::new(pure),
    }
}
/// Sample from a usize distribution (Categorical).
///
/// Example:
/// ```rust
/// # use fugue::*;
///
/// let model = sample_usize(addr!("choice"), Categorical::new(vec![0.3, 0.5, 0.2]).unwrap());
/// ```
pub fn sample_usize(addr: Address, dist: impl Distribution<usize> + 'static) -> Model<usize> {
    Model::SampleUsize {
        addr,
        dist: Box::new(dist),
        k: Box::new(pure),
    }
}

/// Sample from a distribution (generic version - chooses the right variant automatically).
// This is the main sampling function that works with any distribution type.
// The return type is inferred from the distribution type.
///
/// Type-specific variants:
/// - `sample_f64` - Sample from f64 distributions (continuous distributions)
/// - `sample_bool` - Sample from bool distributions (Bernoulli)
/// - `sample_u64` - Sample from u64 distributions (Poisson, Binomial)
/// - `sample_usize` - Sample from usize distributions (Categorical)
///
/// Example:
/// ```rust
/// # use fugue::*;
/// // Automatically returns f64 for continuous distributions
/// let normal_sample: Model<f64> = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
/// // Automatically returns bool for Bernoulli
/// let coin_flip: Model<bool> = sample(addr!("coin"), Bernoulli::new(0.5).unwrap());
/// // Automatically returns u64 for Poisson
/// let count: Model<u64> = sample(addr!("count"), Poisson::new(3.0).unwrap());
/// // Automatically returns usize for Categorical
/// let choice: Model<usize> = sample(addr!("choice"),
///     Categorical::new(vec![0.3, 0.5, 0.2]).unwrap());
/// ```
pub fn sample<T>(addr: Address, dist: impl Distribution<T> + 'static) -> Model<T>
where
    T: SampleType,
{
    T::make_sample_model(addr, Box::new(dist))
}

/// Trait for types that can be sampled in Models.
/// This enables automatic dispatch to the right Model variant.
pub trait SampleType: 'static + Send + Sync + Sized {
    fn make_sample_model(addr: Address, dist: Box<dyn Distribution<Self>>) -> Model<Self>;
    fn make_observe_model(
        addr: Address,
        dist: Box<dyn Distribution<Self>>,
        value: Self,
    ) -> Model<()>;
}
impl SampleType for f64 {
    fn make_sample_model(addr: Address, dist: Box<dyn Distribution<f64>>) -> Model<f64> {
        Model::SampleF64 {
            addr,
            dist,
            k: Box::new(pure),
        }
    }
    fn make_observe_model(
        addr: Address,
        dist: Box<dyn Distribution<f64>>,
        value: f64,
    ) -> Model<()> {
        Model::ObserveF64 {
            addr,
            dist,
            value,
            k: Box::new(pure),
        }
    }
}
impl SampleType for bool {
    fn make_sample_model(addr: Address, dist: Box<dyn Distribution<bool>>) -> Model<bool> {
        Model::SampleBool {
            addr,
            dist,
            k: Box::new(pure),
        }
    }
    fn make_observe_model(
        addr: Address,
        dist: Box<dyn Distribution<bool>>,
        value: bool,
    ) -> Model<()> {
        Model::ObserveBool {
            addr,
            dist,
            value,
            k: Box::new(pure),
        }
    }
}
impl SampleType for u64 {
    fn make_sample_model(addr: Address, dist: Box<dyn Distribution<u64>>) -> Model<u64> {
        Model::SampleU64 {
            addr,
            dist,
            k: Box::new(pure),
        }
    }
    fn make_observe_model(
        addr: Address,
        dist: Box<dyn Distribution<u64>>,
        value: u64,
    ) -> Model<()> {
        Model::ObserveU64 {
            addr,
            dist,
            value,
            k: Box::new(pure),
        }
    }
}
impl SampleType for usize {
    fn make_sample_model(addr: Address, dist: Box<dyn Distribution<usize>>) -> Model<usize> {
        Model::SampleUsize {
            addr,
            dist,
            k: Box::new(pure),
        }
    }
    fn make_observe_model(
        addr: Address,
        dist: Box<dyn Distribution<usize>>,
        value: usize,
    ) -> Model<()> {
        Model::ObserveUsize {
            addr,
            dist,
            value,
            k: Box::new(pure),
        }
    }
}

/// Observe a value from a distribution (generic version).
/// This function automatically chooses the right observation variant based on the distribution type and observed value type.
///
/// Example:
/// ```rust
/// use fugue::*;
/// // Observe f64 value from continuous distribution
/// let model = observe(addr!("y"), Normal::new(1.0, 0.5).unwrap(), 2.5);
/// // Observe bool value from Bernoulli
/// let model = observe(addr!("coin"), Bernoulli::new(0.6).unwrap(), true);
/// // Observe u64 count from Poisson
/// let model = observe(addr!("count"), Poisson::new(3.0).unwrap(), 5u64);
/// // Observe usize choice from Categorical
/// let model = observe(addr!("choice"),
///     Categorical::new(vec![0.3, 0.5, 0.2]).unwrap(), 1usize);
/// ```
pub fn observe<T>(addr: Address, dist: impl Distribution<T> + 'static, value: T) -> Model<()>
where
    T: SampleType,
{
    T::make_observe_model(addr, Box::new(dist), value)
}

/// Add an unnormalized log-weight `logw` to the model, returning a `Model<()>`.
///
/// Factors allow encoding soft constraints or arbitrary log-probability contributions to the model.
/// They are particularly useful for:
///
/// - Encoding constraints that should be "mostly satisfied"
/// - Adding custom log-likelihood terms
/// - Implementing rejection sampling (using negative infinity)
///
/// Example:
/// ```rust
/// # use fugue::*;
/// // Add positive log-weight (increases probability)
/// let model = factor(1.0); // Adds log(e) = 1.0 to weight
/// // Add negative log-weight (decreases probability)
/// let model = factor(-2.0); // Subtracts 2.0 from log-weight
/// // Reject/fail (zero probability)
/// let model = factor(f64::NEG_INFINITY);
/// // Soft constraint: prefer values near zero
/// let x = 5.0;
/// let soft_constraint = factor(-0.5 * x * x); // Gaussian-like penalty
/// ```
pub fn factor(logw: LogF64) -> Model<()> {
    Model::Factor {
        logw,
        k: Box::new(pure),
    }
}

/// `ModelExt<A>` provides monadic operations for composing `Model<A>` values.
/// Provides `bind`, `map`, and `and_then` for chaining and transforming probabilistic computations.
///
/// Example:
/// ```rust
/// # use fugue::*;
/// // Transform result with map
/// let transformed = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
/// .map(|x| x * 2.0);
///
/// // Chain dependent computations with bind
/// let dependent = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
/// .bind(|x| sample(addr!("y"), Normal::new(x, 0.5).unwrap()));
/// ```
pub trait ModelExt<A>: Sized {
    /// Monadic bind operation (>>=).
    ///
    /// Chains two probabilistic computations where the second depends on the result of the first.
    /// This is the fundamental operation for building complex probabilistic models from simpler parts.
    /// The function `k` takes the result of this model and returns a new model.
    ///
    /// Example:
    /// ```rust
    /// # use fugue::*;
    /// // Dependent sampling: y depends on x
    /// let model = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
    ///     .bind(|x| sample(addr!("y"), Normal::new(x, 0.1).unwrap()));
    /// ```
    fn bind<B>(self, k: impl FnOnce(A) -> Model<B> + Send + 'static) -> Model<B>;

    /// Apply a function, `f`, to transform the result of this model.
    /// This is the functor map operation - it transforms the output of a model without adding any additional probabilistic behavior.
    ///
    /// Example:
    /// ```rust
    /// # use fugue::*;
    /// // Transform the sampled value
    /// let model = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
    ///     .map(|x| x.exp()); // Apply exponential function
    /// ```
    fn map<B>(self, f: impl FnOnce(A) -> B + Send + 'static) -> Model<B> {
        self.bind(|a| pure(f(a)))
    }

    /// Alias for `bind` - chains dependent probabilistic computations.
    /// This method provides a more familiar interface for Rust developers used to `Option::and_then` and `Result::and_then`.
    ///
    /// Example:
    /// ```rust
    /// # use fugue::*;
    /// // Dependent sampling: y depends on x
    /// let model = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
    ///     .and_then(|x| sample(addr!("y"), Normal::new(x, 0.1).unwrap()));
    /// ```
    fn and_then<B>(self, k: impl FnOnce(A) -> Model<B> + Send + 'static) -> Model<B> {
        self.bind(k)
    }
}
impl<A: 'static> ModelExt<A> for Model<A> {
    fn bind<B>(self, k: impl FnOnce(A) -> Model<B> + Send + 'static) -> Model<B> {
        match self {
            Model::Pure(a) => k(a),
            Model::SampleF64 { addr, dist, k: k1 } => Model::SampleF64 {
                addr,
                dist,
                k: Box::new(move |x| k1(x).bind(k)),
            },
            Model::SampleBool { addr, dist, k: k1 } => Model::SampleBool {
                addr,
                dist,
                k: Box::new(move |x| k1(x).bind(k)),
            },
            Model::SampleU64 { addr, dist, k: k1 } => Model::SampleU64 {
                addr,
                dist,
                k: Box::new(move |x| k1(x).bind(k)),
            },
            Model::SampleUsize { addr, dist, k: k1 } => Model::SampleUsize {
                addr,
                dist,
                k: Box::new(move |x| k1(x).bind(k)),
            },
            Model::ObserveF64 {
                addr,
                dist,
                value,
                k: k1,
            } => Model::ObserveF64 {
                addr,
                dist,
                value,
                k: Box::new(move |()| k1(()).bind(k)),
            },
            Model::ObserveBool {
                addr,
                dist,
                value,
                k: k1,
            } => Model::ObserveBool {
                addr,
                dist,
                value,
                k: Box::new(move |()| k1(()).bind(k)),
            },
            Model::ObserveU64 {
                addr,
                dist,
                value,
                k: k1,
            } => Model::ObserveU64 {
                addr,
                dist,
                value,
                k: Box::new(move |()| k1(()).bind(k)),
            },
            Model::ObserveUsize {
                addr,
                dist,
                value,
                k: k1,
            } => Model::ObserveUsize {
                addr,
                dist,
                value,
                k: Box::new(move |()| k1(()).bind(k)),
            },
            Model::Factor { logw, k: k1 } => Model::Factor {
                logw,
                k: Box::new(move |()| k1(()).bind(k)),
            },
        }
    }
}

/// Combine two independent models, `ma` and `mb`, into a model of their paired results.
/// This operation runs both models and combines their results into a tuple.
/// The models are executed independently (neither depends on the other's result).
///
/// Example:
/// ```rust
/// # use fugue::*;
/// // Sample two independent random variables
/// let x_model = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
/// let y_model = sample(addr!("y"), Uniform::new(0.0, 1.0).unwrap());
/// let paired = zip(x_model, y_model); // Model<(f64, f64)>
/// // Can be used with any model types
/// let mixed = zip(pure(42.0), sample(addr!("z"), Exponential::new(1.0).unwrap()));
/// ```
pub fn zip<A: Send + 'static, B: Send + 'static>(ma: Model<A>, mb: Model<B>) -> Model<(A, B)> {
    ma.bind(|a| mb.map(move |b| (a, b)))
}

/// Execute a vector of models, `models`, and collect their results into a single model of a vector.
/// This function takes a collection of independent models and runs them all, collecting their results into a vector.
/// This is useful for running multiple similar probabilistic computations.
///
/// Example:
/// ```rust
/// use fugue::*;
/// // Create multiple independent samples
/// let models = vec![
///     sample(addr!("x", 0), Normal::new(0.0, 1.0).unwrap()),
///     sample(addr!("x", 1), Normal::new(1.0, 1.0).unwrap()),
///     sample(addr!("x", 2), Normal::new(2.0, 1.0).unwrap()),
/// ];
/// let all_samples = sequence_vec(models); // Model<Vec<f64>>
/// // Mix deterministic and probabilistic models
/// let mixed_models = vec![
///     pure(1.0),
///     sample(addr!("random"), Uniform::new(0.0, 1.0).unwrap()),
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

/// Apply a function, `f`, that produces models to each item in a vector, `items`, collecting the results.
/// This is a higher-order function that maps each item in the input vector through a function that produces a model,
/// then sequences all the resulting models into a single model of a vector.
/// This is equivalent to `sequence_vec(items.map(f))` but more convenient.
///
/// Example:
/// ```rust
/// use fugue::*;
/// // Add noise to each data point
/// let data = vec![1.0, 2.0, 3.0];
/// let noisy_data = traverse_vec(data, |x| {
///     sample(addr!("noise", x as usize), Normal::new(0.0, 0.1).unwrap())
///         .map(move |noise| x + noise)
/// });
/// // Create observations for each data point
/// let observations = vec![1.2, 2.1, 2.9];
/// let model = traverse_vec(observations, |obs| {
///     observe(addr!("y", obs as usize), Normal::new(2.0, 0.5).unwrap(), obs)
/// });
/// ```
pub fn traverse_vec<T, A: Send + 'static>(
    items: Vec<T>,
    f: impl Fn(T) -> Model<A> + Send + Sync + 'static,
) -> Model<Vec<A>> {
    sequence_vec(items.into_iter().map(f).collect())
}

/// Conditional execution: fail with zero probability when predicate is false.
///
/// Guards provide a way to enforce hard constraints in probabilistic models.
/// When the predicate `pred` is true, the model continues normally.
/// When false, the model receives negative infinite log-weight, effectively ruling out that execution path,
/// returning a `Model<()>` that fails with zero probability.
///
/// Example:
/// ```rust
/// # use fugue::*;
/// // Ensure a sampled value is positive
/// let model = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
///     .bind(|x| {
///         guard(x > 0.0).bind(move |_| pure(x))
///     });
/// // Multiple constraints
/// let model = sample(addr!("x"), Uniform::new(-2.0, 2.0).unwrap())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::addr;
    use crate::core::distribution::*;
    use crate::runtime::handler::run;
    use crate::runtime::interpreters::PriorHandler;
    use crate::runtime::trace::Trace;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn pure_and_map_work() {
        let m = pure(2).map(|x| x + 3);
        let (val, t) = run(
            PriorHandler {
                rng: &mut StdRng::seed_from_u64(1),
                trace: Trace::default(),
            },
            m,
        );
        assert_eq!(val, 5);
        assert_eq!(t.choices.len(), 0);
    }

    #[test]
    fn sample_and_observe_sites() {
        let m = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
            .and_then(|x| observe(addr!("y"), Normal::new(x, 1.0).unwrap(), 0.5).map(move |_| x));

        let mut rng = StdRng::seed_from_u64(42);
        let (_val, trace) = run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            m,
        );
        assert!(trace.choices.contains_key(&addr!("x")));
        // Observation contributes to likelihood but not to choices
        assert!((trace.log_likelihood.is_finite()));
    }

    #[test]
    fn factor_and_guard_affect_weight() {
        // factor adds a finite weight
        let m_ok = factor(-1.23);
        let ((), t_ok) = run(
            PriorHandler {
                rng: &mut StdRng::seed_from_u64(2),
                trace: Trace::default(),
            },
            m_ok,
        );
        assert!((t_ok.total_log_weight() + 1.23).abs() < 1e-12);

        // guard(false) adds -inf weight via factor
        let m_bad = guard(false);
        let ((), t_bad) = run(
            PriorHandler {
                rng: &mut StdRng::seed_from_u64(3),
                trace: Trace::default(),
            },
            m_bad,
        );
        assert!(
            t_bad.total_log_weight().is_infinite() && t_bad.total_log_weight().is_sign_negative()
        );
    }

    #[test]
    fn sequence_and_traverse_vec() {
        let models: Vec<Model<i32>> = (0..5).map(|i| pure(i)).collect();
        let seq = sequence_vec(models);
        let (vals, t) = run(
            PriorHandler {
                rng: &mut StdRng::seed_from_u64(4),
                trace: Trace::default(),
            },
            seq,
        );
        assert_eq!(vals, vec![0, 1, 2, 3, 4]);
        assert_eq!(t.choices.len(), 0);

        let trav = traverse_vec(vec![1, 2, 3], |i| pure(i * 2));
        let (v2, _t2) = run(
            PriorHandler {
                rng: &mut StdRng::seed_from_u64(5),
                trace: Trace::default(),
            },
            trav,
        );
        assert_eq!(v2, vec![2, 4, 6]);
    }

    #[test]
    fn zip_and_sequence_empty_and_bind_chaining() {
        // zip
        let m1 = pure(1);
        let m2 = pure(2);
        let (pair, _t) = run(
            PriorHandler {
                rng: &mut StdRng::seed_from_u64(6),
                trace: Trace::default(),
            },
            zip(m1, m2),
        );
        assert_eq!(pair, (1, 2));

        // sequence empty
        let empty: Vec<Model<i32>> = vec![];
        let (vals, _t2) = run(
            PriorHandler {
                rng: &mut StdRng::seed_from_u64(7),
                trace: Trace::default(),
            },
            sequence_vec(empty),
        );
        assert!(vals.is_empty());

        // bind chaining across types
        let model = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
            .bind(|x| pure(x > 0.0))
            .bind(|b| if b { pure(1u64) } else { pure(0u64) });
        let (_val, _t3) = run(
            PriorHandler {
                rng: &mut StdRng::seed_from_u64(8),
                trace: Trace::default(),
            },
            model,
        );
    }
}
