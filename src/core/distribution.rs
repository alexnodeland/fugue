//! Probability distributions over `f64` with sampling and log-density.
//!
//! This module provides a small set of continuous distributions and a common
//! `DistributionF64` trait so models can be written generically. The trait is
//! dyn-object safe to allow storage in trait objects within the `Model`.
use rand::{Rng, RngCore};
use rand_distr::{
    Distribution as RandDistr, Exp as RDExp, LogNormal as RDLogNormal, Normal as RDNormal,
};
pub type LogF64 = f64;

pub trait DistributionF64: Send + Sync {
    fn sample(&self, rng: &mut dyn RngCore) -> f64;
    fn log_prob(&self, x: f64) -> LogF64;
}

#[derive(Clone, Copy, Debug)]
pub struct Normal {
    pub mu: f64,
    pub sigma: f64,
}
impl DistributionF64 for Normal {
    fn sample(&self, rng: &mut dyn RngCore) -> f64 {
        RDNormal::new(self.mu, self.sigma).unwrap().sample(rng)
    }
    fn log_prob(&self, x: f64) -> LogF64 {
        let z = (x - self.mu) / self.sigma;
        -0.5 * z * z - self.sigma.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Uniform {
    pub low: f64,
    pub high: f64,
}
impl DistributionF64 for Uniform {
    fn sample(&self, rng: &mut dyn RngCore) -> f64 {
        Rng::gen_range(rng, self.low..self.high)
    }
    fn log_prob(&self, x: f64) -> LogF64 {
        if x < self.low || x > self.high {
            f64::NEG_INFINITY
        } else {
            -(self.high - self.low).ln()
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LogNormal {
    pub mu: f64,
    pub sigma: f64,
}
impl DistributionF64 for LogNormal {
    fn sample(&self, rng: &mut dyn RngCore) -> f64 {
        RDLogNormal::new(self.mu, self.sigma).unwrap().sample(rng)
    }
    fn log_prob(&self, x: f64) -> LogF64 {
        if x <= 0.0 {
            return f64::NEG_INFINITY;
        }
        let lx = x.ln();
        let z = (lx - self.mu) / self.sigma;
        -0.5 * z * z - (self.sigma * x).ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Exponential {
    pub rate: f64,
}
impl DistributionF64 for Exponential {
    fn sample(&self, rng: &mut dyn RngCore) -> f64 {
        RDExp::new(self.rate).unwrap().sample(rng)
    }
    fn log_prob(&self, x: f64) -> LogF64 {
        if x < 0.0 {
            f64::NEG_INFINITY
        } else {
            self.rate.ln() - self.rate * x
        }
    }
}
