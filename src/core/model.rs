//! Core model representation: a tiny monadic PPL in direct style.
//!
//! `Model<A>` represents a program that, when interpreted by a runtime handler,
//! yields a value of type `A`. The monadic interface (`pure`, `bind`, `map`)
//! enables compositional probabilistic programs.
use crate::core::address::Address;
use crate::core::distribution::{DistributionF64, LogF64};
pub enum Model<A> {
    Pure(A),
    SampleF {
        addr: Address,
        dist: Box<dyn DistributionF64>,
        k: Box<dyn FnOnce(f64) -> Model<A> + Send + 'static>,
    },
    ObserveF {
        addr: Address,
        dist: Box<dyn DistributionF64>,
        value: f64,
        k: Box<dyn FnOnce(()) -> Model<A> + Send + 'static>,
    },
    FactorF {
        logw: LogF64,
        k: Box<dyn FnOnce(()) -> Model<A> + Send + 'static>,
    },
}
/// Lift a value into the model.
pub fn pure<A>(a: A) -> Model<A> {
    Model::Pure(a)
}
/// Create a sampling site.
pub fn sample(addr: Address, dist: impl DistributionF64 + 'static) -> Model<f64> {
    Model::SampleF {
        addr,
        dist: Box::new(dist),
        k: Box::new(pure),
    }
}
/// Create an observe (condition) site with a fixed value.
pub fn observe(addr: Address, dist: impl DistributionF64 + 'static, value: f64) -> Model<()> {
    Model::ObserveF {
        addr,
        dist: Box::new(dist),
        value,
        k: Box::new(pure),
    }
}
/// Add an unnormalized log-weight factor.
pub fn factor(logw: LogF64) -> Model<()> {
    Model::FactorF {
        logw,
        k: Box::new(pure),
    }
}
/// Monadic operations on models.
pub trait ModelExt<A>: Sized {
    fn bind<B>(self, k: impl FnOnce(A) -> Model<B> + Send + 'static) -> Model<B>;
    fn map<B>(self, f: impl FnOnce(A) -> B + Send + 'static) -> Model<B> {
        self.bind(|a| pure(f(a)))
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
