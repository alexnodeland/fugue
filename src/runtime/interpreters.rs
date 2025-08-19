//! Built-in interpreters: prior sampling, replay, and scoring.
//!
//! These are basic building blocks that accumulate a `Trace` while interpreting
//! `Model`s, and are used by inference algorithms.
use crate::core::address::Address;
use crate::core::distribution::DistributionF64;
use crate::runtime::handler::Handler;
use crate::runtime::trace::{ChoiceF64, Trace};
use rand::RngCore;

pub struct PriorHandler<'r, R: RngCore> {
    pub rng: &'r mut R,
    pub trace: Trace,
}

impl<'r, R: RngCore> Handler for PriorHandler<'r, R> {
    fn on_sample(&mut self, addr: &Address, dist: &dyn DistributionF64) -> f64 {
        let x = dist.sample(self.rng);
        let lp = dist.log_prob(x);
        self.trace.log_prior += lp;
        self.trace.choices.insert(
            addr.clone(),
            ChoiceF64 {
                addr: addr.clone(),
                value: x,
                logp: lp,
            },
        );
        x
    }

    fn on_observe(&mut self, _: &Address, dist: &dyn DistributionF64, value: f64) {
        self.trace.log_likelihood += dist.log_prob(value);
    }

    fn on_factor(&mut self, logw: f64) {
        self.trace.log_factors += logw;
    }

    fn finish(self) -> Trace {
        self.trace
    }
}

pub struct ReplayHandler<'r, R: RngCore> {
    pub rng: &'r mut R,
    pub base: Trace,
    pub trace: Trace,
}

impl<'r, R: RngCore> Handler for ReplayHandler<'r, R> {
    fn on_sample(&mut self, addr: &Address, dist: &dyn DistributionF64) -> f64 {
        let x = if let Some(c) = self.base.choices.get(addr) {
            c.value
        } else {
            dist.sample(self.rng)
        };
        let lp = dist.log_prob(x);
        self.trace.log_prior += lp;
        self.trace.choices.insert(
            addr.clone(),
            ChoiceF64 {
                addr: addr.clone(),
                value: x,
                logp: lp,
            },
        );
        x
    }

    fn on_observe(&mut self, _: &Address, dist: &dyn DistributionF64, value: f64) {
        self.trace.log_likelihood += dist.log_prob(value);
    }

    fn on_factor(&mut self, logw: f64) {
        self.trace.log_factors += logw;
    }

    fn finish(self) -> Trace {
        self.trace
    }
}

pub struct ScoreGivenTrace {
    pub base: Trace,
    pub trace: Trace,
}

impl Handler for ScoreGivenTrace {
    fn on_sample(&mut self, addr: &Address, dist: &dyn DistributionF64) -> f64 {
        let c = self
            .base
            .choices
            .get(addr)
            .unwrap_or_else(|| panic!("missing value for site {} in base trace", addr));
        let lp = dist.log_prob(c.value);
        self.trace.log_prior += lp;
        self.trace.choices.insert(addr.clone(), c.clone());
        c.value
    }

    fn on_observe(&mut self, _: &Address, dist: &dyn DistributionF64, value: f64) {
        self.trace.log_likelihood += dist.log_prob(value);
    }

    fn on_factor(&mut self, logw: f64) {
        self.trace.log_factors += logw;
    }

    fn finish(self) -> Trace {
        self.trace
    }
}
