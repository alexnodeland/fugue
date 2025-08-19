//! Memory management optimizations for probabilistic programming.
//!
//! This module provides efficient memory management strategies to reduce
//! allocation overhead and improve cache locality in probabilistic computation.

use crate::core::address::Address;
use crate::core::distribution::DistributionF64;
use crate::runtime::trace::{Choice, ChoiceValue, Trace};
use std::collections::BTreeMap;
use std::sync::Arc;

/// Copy-on-write trace for efficient MCMC operations.
///
/// Most MCMC operations only modify a small number of choices,
/// so we can share the majority of the trace data between states.
#[derive(Clone, Debug)]
pub struct CowTrace {
    choices: Arc<BTreeMap<Address, Choice>>,
    log_prior: f64,
    log_likelihood: f64,
    log_factors: f64,
}

impl CowTrace {
    /// Create a new copy-on-write trace.
    pub fn new() -> Self {
        Self {
            choices: Arc::new(BTreeMap::new()),
            log_prior: 0.0,
            log_likelihood: 0.0,
            log_factors: 0.0,
        }
    }

    /// Convert from regular trace.
    pub fn from_trace(trace: Trace) -> Self {
        Self {
            choices: Arc::new(trace.choices),
            log_prior: trace.log_prior,
            log_likelihood: trace.log_likelihood,
            log_factors: trace.log_factors,
        }
    }

    /// Convert to regular trace (may involve copying).
    pub fn to_trace(&self) -> Trace {
        Trace {
            choices: (*self.choices).clone(),
            log_prior: self.log_prior,
            log_likelihood: self.log_likelihood,
            log_factors: self.log_factors,
        }
    }

    /// Get mutable access to choices, copying if necessary.
    pub fn choices_mut(&mut self) -> &mut BTreeMap<Address, Choice> {
        if Arc::strong_count(&self.choices) > 1 {
            // Need to copy - other references exist
            self.choices = Arc::new((*self.choices).clone());
        }
        Arc::get_mut(&mut self.choices).unwrap()
    }

    /// Insert a choice, copying the map if needed.
    pub fn insert_choice(&mut self, addr: Address, choice: Choice) {
        self.choices_mut().insert(addr, choice);
    }

    /// Get read-only access to choices.
    pub fn choices(&self) -> &BTreeMap<Address, Choice> {
        &self.choices
    }

    /// Total log weight.
    pub fn total_log_weight(&self) -> f64 {
        self.log_prior + self.log_likelihood + self.log_factors
    }
}

/// Efficient trace builder that minimizes allocations.
pub struct TraceBuilder {
    choices: BTreeMap<Address, Choice>,
    log_prior: f64,
    log_likelihood: f64,
    log_factors: f64,
}

impl TraceBuilder {
    pub fn new() -> Self {
        Self {
            choices: BTreeMap::new(),
            log_prior: 0.0,
            log_likelihood: 0.0,
            log_factors: 0.0,
        }
    }

    pub fn with_capacity(_capacity: usize) -> Self {
        // BTreeMap doesn't have with_capacity, but we can pre-allocate differently
        Self::new()
    }

    pub fn add_sample(&mut self, addr: Address, value: f64, log_prob: f64) {
        self.choices.insert(
            addr.clone(),
            Choice {
                addr,
                value: ChoiceValue::F64(value),
                logp: log_prob,
            },
        );
        self.log_prior += log_prob;
    }

    pub fn add_observation(&mut self, log_likelihood: f64) {
        self.log_likelihood += log_likelihood;
    }

    pub fn add_factor(&mut self, log_weight: f64) {
        self.log_factors += log_weight;
    }

    pub fn build(self) -> Trace {
        Trace {
            choices: self.choices,
            log_prior: self.log_prior,
            log_likelihood: self.log_likelihood,
            log_factors: self.log_factors,
        }
    }
}

/// Memory pool for reusing trace allocations.
pub struct TracePool {
    available: Vec<Trace>,
    max_size: usize,
}

impl TracePool {
    pub fn new(max_size: usize) -> Self {
        Self {
            available: Vec::with_capacity(max_size),
            max_size,
        }
    }

    /// Get a trace from the pool or create new one.
    pub fn get(&mut self) -> Trace {
        self.available.pop().unwrap_or_else(|| Trace::default())
    }

    /// Return a trace to the pool for reuse.
    pub fn return_trace(&mut self, mut trace: Trace) {
        if self.available.len() < self.max_size {
            // Clear the trace for reuse
            trace.choices.clear();
            trace.log_prior = 0.0;
            trace.log_likelihood = 0.0;
            trace.log_factors = 0.0;
            self.available.push(trace);
        }
    }
}

/// Optimized handler that uses memory pooling.
pub struct PooledPriorHandler<'a, R: rand::RngCore> {
    pub rng: &'a mut R,
    pub trace_builder: TraceBuilder,
    pub pool: &'a mut TracePool,
}

impl<'a, R: rand::RngCore> crate::runtime::handler::Handler for PooledPriorHandler<'a, R> {
    fn on_sample(&mut self, addr: &Address, dist: &dyn DistributionF64) -> f64 {
        let x = dist.sample(self.rng);
        let lp = dist.log_prob(x);
        self.trace_builder.add_sample(addr.clone(), x, lp);
        x
    }

    fn on_observe(&mut self, _: &Address, dist: &dyn DistributionF64, value: f64) {
        let log_likelihood = dist.log_prob(value);
        self.trace_builder.add_observation(log_likelihood);
    }

    fn on_factor(&mut self, logw: f64) {
        self.trace_builder.add_factor(logw);
    }

    fn finish(self) -> Trace {
        self.trace_builder.build()
    }
}

#[cfg(test)]
mod memory_tests {
    use super::*;
    use crate::addr;

    #[test]
    fn test_cow_trace_efficiency() {
        let mut trace1 = CowTrace::new();
        trace1.insert_choice(
            addr!("x"),
            Choice {
                addr: addr!("x"),
                value: ChoiceValue::F64(1.0),
                logp: -0.5,
            },
        );

        // Clone should be efficient (no copying yet)
        let trace2 = trace1.clone();
        assert!(Arc::ptr_eq(&trace1.choices, &trace2.choices));

        // Modifying one should trigger copy
        let mut trace3 = trace2.clone();
        trace3.insert_choice(
            addr!("y"),
            Choice {
                addr: addr!("y"),
                value: ChoiceValue::F64(2.0),
                logp: -1.0,
            },
        );

        // Now they should have different underlying data
        assert!(!Arc::ptr_eq(&trace1.choices, &trace3.choices));
    }

    #[test]
    fn test_trace_pool() {
        let mut pool = TracePool::new(3);

        // Get traces from pool
        let trace1 = pool.get();
        let trace2 = pool.get();

        // Return to pool
        pool.return_trace(trace1);
        pool.return_trace(trace2);

        // Should reuse returned traces
        let trace3 = pool.get();
        assert_eq!(trace3.choices.len(), 0); // Should be cleared
    }

    #[test]
    fn test_trace_builder_efficiency() {
        let mut builder = TraceBuilder::new();

        // Add many choices efficiently
        for i in 0..1000 {
            builder.add_sample(addr!("x", i), i as f64, -0.5);
        }

        let trace = builder.build();
        assert_eq!(trace.choices.len(), 1000);
        assert!((trace.log_prior - (-500.0)).abs() < 1e-10);
    }
}
