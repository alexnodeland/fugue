//! Probability distributions over `f64` with sampling and log-density.
//!
//! This module provides a small set of continuous distributions and a common
//! `DistributionF64` trait so models can be written generically. The trait is
//! dyn-object safe to allow storage in trait objects within the `Model`.
use rand::{Rng, RngCore};
use rand_distr::{
    Distribution as RandDistr, Exp as RDExp, LogNormal as RDLogNormal, Normal as RDNormal,
    Bernoulli as RDBernoulli, Beta as RDBeta, Gamma as RDGamma, Binomial as RDBinomial, Poisson as RDPoisson,
};
pub type LogF64 = f64;

pub trait DistributionF64: Send + Sync {
    fn sample(&self, rng: &mut dyn RngCore) -> f64;
    fn log_prob(&self, x: f64) -> LogF64;
    fn clone_box(&self) -> Box<dyn DistributionF64>;
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
    fn clone_box(&self) -> Box<dyn DistributionF64> {
        Box::new(*self)
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
    fn clone_box(&self) -> Box<dyn DistributionF64> {
        Box::new(*self)
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
    fn clone_box(&self) -> Box<dyn DistributionF64> {
        Box::new(*self)
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
    fn clone_box(&self) -> Box<dyn DistributionF64> {
        Box::new(*self)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Bernoulli {
    pub p: f64,
}
impl DistributionF64 for Bernoulli {
    fn sample(&self, rng: &mut dyn RngCore) -> f64 {
        if RDBernoulli::new(self.p).unwrap().sample(rng) { 1.0 } else { 0.0 }
    }
    fn log_prob(&self, x: f64) -> LogF64 {
        if x == 1.0 {
            self.p.ln()
        } else if x == 0.0 {
            (1.0 - self.p).ln()
        } else {
            f64::NEG_INFINITY
        }
    }
    fn clone_box(&self) -> Box<dyn DistributionF64> {
        Box::new(*self)
    }
}

#[derive(Clone, Debug)]
pub struct Categorical {
    pub probs: Vec<f64>,
}
impl DistributionF64 for Categorical {
    fn sample(&self, rng: &mut dyn RngCore) -> f64 {
        let u: f64 = rng.gen();
        let mut cum = 0.0;
        for (i, &p) in self.probs.iter().enumerate() {
            cum += p;
            if u <= cum {
                return i as f64;
            }
        }
        (self.probs.len() - 1) as f64
    }
    fn log_prob(&self, x: f64) -> LogF64 {
        let idx = x as usize;
        if idx < self.probs.len() && (x - idx as f64).abs() < 1e-12 {
            self.probs[idx].ln()
        } else {
            f64::NEG_INFINITY
        }
    }
    fn clone_box(&self) -> Box<dyn DistributionF64> {
        Box::new(self.clone())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Beta {
    pub alpha: f64,
    pub beta: f64,
}
impl DistributionF64 for Beta {
    fn sample(&self, rng: &mut dyn RngCore) -> f64 {
        RDBeta::new(self.alpha, self.beta).unwrap().sample(rng)
    }
    fn log_prob(&self, x: f64) -> LogF64 {
        if x <= 0.0 || x >= 1.0 {
            return f64::NEG_INFINITY;
        }
        // log Beta(x; α, β) = (α-1)ln(x) + (β-1)ln(1-x) - log B(α,β)
        let log_beta_fn = libm::lgamma(self.alpha) + libm::lgamma(self.beta) - libm::lgamma(self.alpha + self.beta);
        (self.alpha - 1.0) * x.ln() + (self.beta - 1.0) * (1.0 - x).ln() - log_beta_fn
    }
    fn clone_box(&self) -> Box<dyn DistributionF64> {
        Box::new(*self)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Gamma {
    pub shape: f64,
    pub rate: f64,
}
impl DistributionF64 for Gamma {
    fn sample(&self, rng: &mut dyn RngCore) -> f64 {
        RDGamma::new(self.shape, 1.0 / self.rate).unwrap().sample(rng)
    }
    fn log_prob(&self, x: f64) -> LogF64 {
        if x <= 0.0 {
            return f64::NEG_INFINITY;
        }
        // log Gamma(x; k, λ) = k*ln(λ) + (k-1)*ln(x) - λ*x - ln Γ(k)
        self.shape * self.rate.ln() + (self.shape - 1.0) * x.ln() - self.rate * x - libm::lgamma(self.shape)
    }
    fn clone_box(&self) -> Box<dyn DistributionF64> {
        Box::new(*self)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Binomial {
    pub n: u64,
    pub p: f64,
}
impl DistributionF64 for Binomial {
    fn sample(&self, rng: &mut dyn RngCore) -> f64 {
        RDBinomial::new(self.n, self.p).unwrap().sample(rng) as f64
    }
    fn log_prob(&self, x: f64) -> LogF64 {
        let k = x as u64;
        if k > self.n || (x - k as f64).abs() > 1e-12 {
            return f64::NEG_INFINITY;
        }
        // log Binomial(k; n, p) = log C(n,k) + k*ln(p) + (n-k)*ln(1-p)
        let log_binom_coeff = libm::lgamma(self.n as f64 + 1.0) - libm::lgamma(k as f64 + 1.0) - libm::lgamma((self.n - k) as f64 + 1.0);
        log_binom_coeff + (k as f64) * self.p.ln() + ((self.n - k) as f64) * (1.0 - self.p).ln()
    }
    fn clone_box(&self) -> Box<dyn DistributionF64> {
        Box::new(*self)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Poisson {
    pub lambda: f64,
}
impl DistributionF64 for Poisson {
    fn sample(&self, rng: &mut dyn RngCore) -> f64 {
        RDPoisson::new(self.lambda).unwrap().sample(rng) as f64
    }
    fn log_prob(&self, x: f64) -> LogF64 {
        let k = x as u64;
        if (x - k as f64).abs() > 1e-12 || x < 0.0 {
            return f64::NEG_INFINITY;
        }
        // log Poisson(k; λ) = k*ln(λ) - λ - ln(k!)
        (k as f64) * self.lambda.ln() - self.lambda - libm::lgamma(k as f64 + 1.0)
    }
    fn clone_box(&self) -> Box<dyn DistributionF64> {
        Box::new(*self)
    }
}
