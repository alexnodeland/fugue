#![doc = include_str!("../../docs/api/core/model/README.md")]
use crate::core::address::Address;
use crate::core::distribution::{Distribution, LogF64};

#[doc = include_str!("../../docs/api/core/model/model.md")]
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

#[doc = include_str!("../../docs/api/core/model/pure.md")]
pub fn pure<A>(a: A) -> Model<A> {
    Model::Pure(a)
}
/// Sample from an f64 distribution (continuous distributions).
pub fn sample_f64(addr: Address, dist: impl Distribution<f64> + 'static) -> Model<f64> {
    Model::SampleF64 {
        addr,
        dist: Box::new(dist),
        k: Box::new(pure),
    }
}
/// Sample from a bool distribution (Bernoulli).
pub fn sample_bool(addr: Address, dist: impl Distribution<bool> + 'static) -> Model<bool> {
    Model::SampleBool {
        addr,
        dist: Box::new(dist),
        k: Box::new(pure),
    }
}
/// Sample from a u64 distribution (Poisson, Binomial).
pub fn sample_u64(addr: Address, dist: impl Distribution<u64> + 'static) -> Model<u64> {
    Model::SampleU64 {
        addr,
        dist: Box::new(dist),
        k: Box::new(pure),
    }
}
/// Sample from a usize distribution (Categorical).
pub fn sample_usize(addr: Address, dist: impl Distribution<usize> + 'static) -> Model<usize> {
    Model::SampleUsize {
        addr,
        dist: Box::new(dist),
        k: Box::new(pure),
    }
}

#[doc = include_str!("../../docs/api/core/model/sample.md")]
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

#[doc = include_str!("../../docs/api/core/model/observe.md")]
pub fn observe<T>(addr: Address, dist: impl Distribution<T> + 'static, value: T) -> Model<()>
where
    T: SampleType,
{
    T::make_observe_model(addr, Box::new(dist), value)
}

#[doc = include_str!("../../docs/api/core/model/factor.md")]
pub fn factor(logw: LogF64) -> Model<()> {
    Model::Factor {
        logw,
        k: Box::new(pure),
    }
}

#[doc = include_str!("../../docs/api/core/model/model_ext.md")]
pub trait ModelExt<A>: Sized {
    #[doc = include_str!("../../docs/api/core/model/bind.md")]
    fn bind<B>(self, k: impl FnOnce(A) -> Model<B> + Send + 'static) -> Model<B>;

    #[doc = include_str!("../../docs/api/core/model/map.md")]
    fn map<B>(self, f: impl FnOnce(A) -> B + Send + 'static) -> Model<B> {
        self.bind(|a| pure(f(a)))
    }

    #[doc = include_str!("../../docs/api/core/model/and_then.md")]
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

#[doc = include_str!("../../docs/api/core/model/zip.md")]
pub fn zip<A: Send + 'static, B: Send + 'static>(ma: Model<A>, mb: Model<B>) -> Model<(A, B)> {
    ma.bind(|a| mb.map(move |b| (a, b)))
}

#[doc = include_str!("../../docs/api/core/model/sequence_vec.md")]
pub fn sequence_vec<A: Send + 'static>(models: Vec<Model<A>>) -> Model<Vec<A>> {
    models.into_iter().fold(pure(Vec::new()), |acc, m| {
        zip(acc, m).map(|(mut v, a)| {
            v.push(a);
            v
        })
    })
}

#[doc = include_str!("../../docs/api/core/model/traverse_vec.md")]
pub fn traverse_vec<T, A: Send + 'static>(
    items: Vec<T>,
    f: impl Fn(T) -> Model<A> + Send + Sync + 'static,
) -> Model<Vec<A>> {
    sequence_vec(items.into_iter().map(f).collect())
}

#[doc = include_str!("../../docs/api/core/model/guard.md")]
pub fn guard(pred: bool) -> Model<()> {
    if pred {
        pure(())
    } else {
        factor(f64::NEG_INFINITY)
    }
}
