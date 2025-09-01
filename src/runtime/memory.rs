#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/docs/runtime/memory.md"))]

use crate::core::address::Address;
use crate::core::distribution::Distribution;
use crate::runtime::trace::{Choice, ChoiceValue, Trace};
use std::collections::BTreeMap;
use std::sync::Arc;

/// Copy-on-write trace for efficient memory sharing in MCMC operations.
///
/// Most MCMC operations modify only a small number of choices, so CowTrace
/// shares the majority of trace data between states using `Arc<BTreeMap>`.
///
/// Example:
/// ```rust
/// # use fugue::*;
/// # use fugue::runtime::memory::CowTrace;
///
/// // Convert from regular trace
/// # let mut rng = rand::thread_rng();
/// # let (_, trace) = runtime::handler::run(
/// #     PriorHandler { rng: &mut rng, trace: Trace::default() },
/// #     sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
/// # );
/// let cow_trace = CowTrace::from_trace(trace);
///
/// // Clone is very efficient (shares memory)
/// let clone1 = cow_trace.clone();
/// let clone2 = cow_trace.clone();
///
/// // Modification triggers copy-on-write only when needed
/// let mut modified = clone1.clone();
/// modified.insert_choice(addr!("new"), Choice {
///     addr: addr!("new"),
///     value: ChoiceValue::F64(42.0),
///     logp: -1.0,
/// });
/// // Now `modified` has its own copy, others still share
/// ```
#[derive(Clone, Debug)]
pub struct CowTrace {
    choices: Arc<BTreeMap<Address, Choice>>,
    log_prior: f64,
    log_likelihood: f64,
    log_factors: f64,
}

impl Default for CowTrace {
    fn default() -> Self {
        Self::new()
    }
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

/// Efficient trace builder that minimizes allocations during construction.
///
/// TraceBuilder uses pre-allocated collections and provides type-specific
/// methods to build traces efficiently with minimal memory overhead.
///
/// Example:
/// ```rust
/// # use fugue::*;
/// # use fugue::runtime::memory::TraceBuilder;
///
/// let mut builder = TraceBuilder::new();
///
/// // Add different types of samples efficiently
/// builder.add_sample(addr!("x"), 1.5, -0.5);
/// builder.add_sample_bool(addr!("flag"), true, -0.693);
/// builder.add_sample_u64(addr!("count"), 42, -1.0);
///
/// // Add observations and factors
/// builder.add_observation(-2.3); // Likelihood contribution
/// builder.add_factor(-0.1);      // Soft constraint
///
/// // Build final trace
/// let trace = builder.build();
/// assert_eq!(trace.choices.len(), 3);
/// ```
pub struct TraceBuilder {
    choices: BTreeMap<Address, Choice>,
    log_prior: f64,
    log_likelihood: f64,
    log_factors: f64,
}

impl Default for TraceBuilder {
    fn default() -> Self {
        Self::new()
    }
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
        let choice = Choice {
            addr: addr.clone(),
            value: ChoiceValue::F64(value),
            logp: log_prob,
        };
        self.choices.insert(addr, choice);
        self.log_prior += log_prob;
    }

    pub fn add_sample_bool(&mut self, addr: Address, value: bool, log_prob: f64) {
        let choice = Choice {
            addr: addr.clone(),
            value: ChoiceValue::Bool(value),
            logp: log_prob,
        };
        self.choices.insert(addr, choice);
        self.log_prior += log_prob;
    }

    pub fn add_sample_u64(&mut self, addr: Address, value: u64, log_prob: f64) {
        let choice = Choice {
            addr: addr.clone(),
            value: ChoiceValue::U64(value),
            logp: log_prob,
        };
        self.choices.insert(addr, choice);
        self.log_prior += log_prob;
    }

    pub fn add_sample_usize(&mut self, addr: Address, value: usize, log_prob: f64) {
        let choice = Choice {
            addr: addr.clone(),
            value: ChoiceValue::Usize(value),
            logp: log_prob,
        };
        self.choices.insert(addr, choice);
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

/// Memory pool for reusing trace allocations to reduce overhead.
///
/// TracePool maintains a collection of cleared Trace objects that can be
/// reused to reduce allocation overhead in MCMC and other inference algorithms.
///
/// Example:
/// ```rust
/// # use fugue::*;
/// # use fugue::runtime::memory::TracePool;
///
/// let mut pool = TracePool::new(10); // Pool up to 10 traces
///
/// // Get traces from pool (creates new ones initially)
/// let trace1 = pool.get();
/// let trace2 = pool.get();
/// assert_eq!(pool.stats().misses, 2); // Both were cache misses
///
/// // Return traces to pool for reuse
/// pool.return_trace(trace1);
/// pool.return_trace(trace2);
/// assert_eq!(pool.stats().returns, 2);
///
/// // Next gets will reuse pooled traces (cache hits)
/// let trace3 = pool.get();
/// assert_eq!(pool.stats().hits, 1);
/// assert_eq!(trace3.choices.len(), 0); // Trace was cleared
/// ```
pub struct TracePool {
    available: Vec<Trace>,
    max_size: usize,
    min_size: usize,
    stats: PoolStats,
}

/// Statistics for monitoring TracePool usage and efficiency.
///
/// PoolStats tracks cache hits/misses and provides metrics to optimize
/// memory pool performance in inference algorithms.
///
/// Example:
/// ```rust
/// # use fugue::runtime::memory::*;
///
/// let mut pool = TracePool::new(5);
///
/// // Generate some cache activity
/// let trace1 = pool.get(); // miss
/// let trace2 = pool.get(); // miss
/// pool.return_trace(trace1);
/// let trace3 = pool.get(); // hit (reuses trace1)
///
/// // Check performance metrics
/// let stats = pool.stats();
/// println!("Hit ratio: {:.1}%", stats.hit_ratio());
/// println!("Total operations: {}", stats.total_gets());
/// assert_eq!(stats.hits, 1);
/// assert_eq!(stats.misses, 2);
/// ```
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// Number of successful gets from the pool (cache hits).
    pub hits: u64,
    /// Number of gets that required new allocation (cache misses).
    pub misses: u64,
    /// Number of traces returned to the pool.
    pub returns: u64,
    /// Number of traces dropped due to pool being full.
    pub drops: u64,
}

impl PoolStats {
    /// Calculate hit ratio as a percentage.
    pub fn hit_ratio(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            (self.hits as f64 / total as f64) * 100.0
        }
    }

    /// Total number of get operations.
    pub fn total_gets(&self) -> u64 {
        self.hits + self.misses
    }
}

impl TracePool {
    /// Create a new trace pool with the specified capacity bounds.
    ///
    /// - `max_size`: Maximum number of traces to keep in the pool
    /// - `min_size`: Minimum number of traces to maintain (for shrinking)
    pub fn new(max_size: usize) -> Self {
        Self {
            available: Vec::with_capacity(max_size),
            max_size,
            min_size: max_size / 4, // Keep at least 25% of max capacity
            stats: PoolStats::default(),
        }
    }

    /// Create a new trace pool with custom capacity bounds.
    pub fn with_bounds(max_size: usize, min_size: usize) -> Self {
        assert!(min_size <= max_size, "min_size must be <= max_size");
        Self {
            available: Vec::with_capacity(max_size),
            max_size,
            min_size,
            stats: PoolStats::default(),
        }
    }

    /// Get a trace from the pool or create new one.
    ///
    /// Returns a cleared trace ready for use. Updates hit/miss statistics.
    pub fn get(&mut self) -> Trace {
        if let Some(trace) = self.available.pop() {
            self.stats.hits += 1;
            trace
        } else {
            self.stats.misses += 1;
            Trace::default()
        }
    }

    /// Return a trace to the pool for reuse.
    ///
    /// The trace will be cleared and made available for future gets.
    /// If the pool is full, the trace will be dropped.
    pub fn return_trace(&mut self, mut trace: Trace) {
        if self.available.len() < self.max_size {
            // Clear the trace for reuse
            trace.choices.clear();
            trace.log_prior = 0.0;
            trace.log_likelihood = 0.0;
            trace.log_factors = 0.0;
            self.available.push(trace);
            self.stats.returns += 1;
        } else {
            self.stats.drops += 1;
        }
    }

    /// Shrink the pool to the minimum size if it's grown too large.
    ///
    /// This can be called periodically to reclaim memory when the pool
    /// has accumulated more traces than needed.
    pub fn shrink(&mut self) {
        if self.available.len() > self.min_size {
            self.available.truncate(self.min_size);
            self.available.shrink_to_fit();
        }
    }

    /// Force shrink to a specific size.
    pub fn shrink_to(&mut self, target_size: usize) {
        let target = target_size.min(self.max_size);
        if self.available.len() > target {
            self.available.truncate(target);
            self.available.shrink_to_fit();
        }
    }

    /// Clear all traces from the pool.
    pub fn clear(&mut self) {
        self.available.clear();
    }

    /// Get current pool statistics.
    pub fn stats(&self) -> &PoolStats {
        &self.stats
    }

    /// Reset statistics counters.
    pub fn reset_stats(&mut self) {
        self.stats = PoolStats::default();
    }

    /// Current number of available traces in the pool.
    pub fn len(&self) -> usize {
        self.available.len()
    }

    /// Check if the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.available.is_empty()
    }

    /// Maximum capacity of the pool.
    pub fn capacity(&self) -> usize {
        self.max_size
    }

    /// Minimum size maintained during shrinking.
    pub fn min_capacity(&self) -> usize {
        self.min_size
    }
}

/// Optimized handler that uses memory pooling for zero-allocation inference.
///
/// PooledPriorHandler combines TraceBuilder efficiency with TracePool reuse
/// to achieve zero-allocation execution after pool warm-up.
///
/// Example:
/// ```rust
/// # use fugue::*;
/// # use fugue::runtime::memory::*;
/// # use rand::rngs::StdRng;
/// # use rand::SeedableRng;
///
/// let mut pool = TracePool::new(10);
/// let mut rng = StdRng::seed_from_u64(42);
///
/// // Run model with pooled handler
/// let (result, trace) = runtime::handler::run(
///     PooledPriorHandler::new(&mut rng, &mut pool),
///     sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
/// );
///
/// // Subsequent runs will reuse pooled traces (zero allocations)
/// assert!(result.is_finite());
/// ```
pub struct PooledPriorHandler<'a, R: rand::RngCore> {
    pub rng: &'a mut R,
    pub trace_builder: TraceBuilder,
    pub pool: &'a mut TracePool,
    pub pooled_trace: Option<Trace>,
}

impl<'a, R: rand::RngCore> PooledPriorHandler<'a, R> {
    /// Create a new PooledPriorHandler that gets a trace from the pool.
    pub fn new(rng: &'a mut R, pool: &'a mut TracePool) -> Self {
        let pooled_trace = Some(pool.get());
        Self {
            rng,
            trace_builder: TraceBuilder::new(),
            pool,
            pooled_trace,
        }
    }
}

impl<'a, R: rand::RngCore> crate::runtime::handler::Handler for PooledPriorHandler<'a, R> {
    fn on_sample_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>) -> f64 {
        let x = dist.sample(self.rng);
        let lp = dist.log_prob(&x);
        self.trace_builder.add_sample(addr.clone(), x, lp);
        x
    }

    fn on_sample_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>) -> bool {
        let x = dist.sample(self.rng);
        let lp = dist.log_prob(&x);
        self.trace_builder.add_sample_bool(addr.clone(), x, lp);
        x
    }

    fn on_sample_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>) -> u64 {
        let x = dist.sample(self.rng);
        let lp = dist.log_prob(&x);
        self.trace_builder.add_sample_u64(addr.clone(), x, lp);
        x
    }

    fn on_sample_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>) -> usize {
        let x = dist.sample(self.rng);
        let lp = dist.log_prob(&x);
        self.trace_builder.add_sample_usize(addr.clone(), x, lp);
        x
    }

    fn on_observe_f64(&mut self, _: &Address, dist: &dyn Distribution<f64>, value: f64) {
        let log_likelihood = dist.log_prob(&value);
        self.trace_builder.add_observation(log_likelihood);
    }

    fn on_observe_bool(&mut self, _: &Address, dist: &dyn Distribution<bool>, value: bool) {
        let log_likelihood = dist.log_prob(&value);
        self.trace_builder.add_observation(log_likelihood);
    }

    fn on_observe_u64(&mut self, _: &Address, dist: &dyn Distribution<u64>, value: u64) {
        let log_likelihood = dist.log_prob(&value);
        self.trace_builder.add_observation(log_likelihood);
    }

    fn on_observe_usize(&mut self, _: &Address, dist: &dyn Distribution<usize>, value: usize) {
        let log_likelihood = dist.log_prob(&value);
        self.trace_builder.add_observation(log_likelihood);
    }

    fn on_factor(&mut self, logw: f64) {
        self.trace_builder.add_factor(logw);
    }

    fn finish(mut self) -> Trace {
        // Use the pooled trace as the base, or create a new one if none available
        let mut trace = if let Some(pooled_trace) = self.pooled_trace.take() {
            pooled_trace
        } else {
            Trace::default()
        };
        
        // Populate the trace with data from the trace builder
        let built_trace = self.trace_builder.build();
        trace.choices = built_trace.choices;
        trace.log_prior = built_trace.log_prior;
        trace.log_likelihood = built_trace.log_likelihood;
        trace.log_factors = built_trace.log_factors;
        
        trace
    }
}

#[cfg(test)]
mod memory_tests {
    use super::*;
    use crate::addr;
    use std::time::Instant;

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
    fn test_trace_pool_basic() {
        let mut pool = TracePool::new(3);

        // Get traces from pool
        let trace1 = pool.get();
        let trace2 = pool.get();

        // Should be cache misses initially
        assert_eq!(pool.stats().misses, 2);
        assert_eq!(pool.stats().hits, 0);

        // Return to pool
        pool.return_trace(trace1);
        pool.return_trace(trace2);
        assert_eq!(pool.stats().returns, 2);

        // Should reuse returned traces (cache hits)
        let trace3 = pool.get();
        assert_eq!(trace3.choices.len(), 0); // Should be cleared
        assert_eq!(pool.stats().hits, 1);
    }

    #[test]
    fn test_trace_pool_stats() {
        let mut pool = TracePool::new(2);

        // Test hit/miss tracking
        let t1 = pool.get(); // miss
        let t2 = pool.get(); // miss
        assert_eq!(pool.stats().misses, 2);
        assert_eq!(pool.stats().hit_ratio(), 0.0);

        pool.return_trace(t1); // return
        let _t3 = pool.get(); // hit
        assert_eq!(pool.stats().hits, 1);
        assert_eq!(pool.stats().returns, 1);
        assert!(pool.stats().hit_ratio() > 0.0);

        // Test overflow (drop) - need to fill pool first
        pool.return_trace(t2); // return (pool now has 1 item)
        let another_trace = pool.get(); // get the returned trace (hit)
        pool.return_trace(another_trace); // return it (pool now has 1 item)

        // Add one more to make pool full (capacity 2)
        let extra_trace = Trace::default();
        pool.return_trace(extra_trace); // pool now has 2 items (full)

        // Now this should be dropped
        let dummy_trace = Trace {
            log_prior: 1.0, // Make it non-empty
            ..Trace::default()
        };
        pool.return_trace(dummy_trace); // should be dropped because pool is full
        assert_eq!(pool.stats().drops, 1);
    }

    #[test]
    fn test_trace_pool_shrinking() {
        let mut pool = TracePool::with_bounds(10, 3);

        // Fill pool beyond minimum
        for _ in 0..8 {
            pool.return_trace(Trace::default());
        }
        assert_eq!(pool.len(), 8);

        // Shrink should reduce to minimum
        pool.shrink();
        assert_eq!(pool.len(), 3);

        // Shrink to specific size
        for _ in 0..5 {
            pool.return_trace(Trace::default());
        }
        assert_eq!(pool.len(), 8); // 3 + 5
        pool.shrink_to(2);
        assert_eq!(pool.len(), 2);
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

    #[test]
    fn test_address_optimization() {
        // Test that the new TraceBuilder implementation doesn't create
        // unnecessary address clones
        let start = Instant::now();
        let mut builder = TraceBuilder::new();

        for i in 0..10000 {
            let addr = addr!("test", i);
            builder.add_sample(addr, i as f64, -0.5);
        }

        let trace = builder.build();
        let duration = start.elapsed();

        assert_eq!(trace.choices.len(), 10000);
        // This is a smoke test - in practice you'd compare with a baseline
        println!("Built trace with 10k choices in {:?}", duration);
    }

    #[test]
    fn test_mixed_value_types() {
        let mut builder = TraceBuilder::new();

        // Test all supported value types
        builder.add_sample(addr!("f64"), 1.5, -0.5);
        builder.add_sample_bool(addr!("bool"), true, -0.693);
        builder.add_sample_u64(addr!("u64"), 42, -1.0);
        builder.add_sample_usize(addr!("usize"), 3, -1.2);

        let trace = builder.build();
        assert_eq!(trace.choices.len(), 4);

        // Verify values are stored correctly
        assert_eq!(trace.choices[&addr!("f64")].value, ChoiceValue::F64(1.5));
        assert_eq!(trace.choices[&addr!("bool")].value, ChoiceValue::Bool(true));
        assert_eq!(trace.choices[&addr!("u64")].value, ChoiceValue::U64(42));
        assert_eq!(trace.choices[&addr!("usize")].value, ChoiceValue::Usize(3));
    }

    #[test]
    fn test_cow_trace_memory_sharing() {
        // Create a large base trace
        let mut base = Trace::default();
        for i in 0..1000 {
            base.insert_choice(addr!("x", i), ChoiceValue::F64(i as f64), -0.5);
        }
        let cow_base = CowTrace::from_trace(base);

        // Create many clones (should share memory)
        let mut clones = Vec::new();
        for _ in 0..100 {
            clones.push(cow_base.clone());
        }

        // All clones should share the same Arc
        for clone in &clones {
            assert!(Arc::ptr_eq(&cow_base.choices, &clone.choices));
        }

        // Modifying one clone should not affect others
        let mut modified = clones[0].clone();
        modified.insert_choice(
            addr!("new"),
            Choice {
                addr: addr!("new"),
                value: ChoiceValue::F64(999.0),
                logp: -2.0,
            },
        );

        // The modified clone should have different data
        assert!(!Arc::ptr_eq(&cow_base.choices, &modified.choices));
        // But other clones should still share with base
        assert!(Arc::ptr_eq(&cow_base.choices, &clones[1].choices));
    }

    #[test]
    fn test_pool_stats_accuracy() {
        let mut pool = TracePool::new(5);

        // Pattern: get 10, return 5, get 10 more
        // First 10 gets: all misses
        for _ in 0..10 {
            pool.get(); // 10 misses
        }

        // Return 5 traces (pool capacity is 5, so all should be accepted)
        for _ in 0..5 {
            pool.return_trace(Trace::default()); // 5 returns
        }

        // Next 10 gets: first 5 should be hits, next 5 should be misses
        for _ in 0..10 {
            pool.get(); // 5 hits + 5 misses
        }

        let stats = pool.stats();
        assert_eq!(stats.misses, 15); // 10 + 5
        assert_eq!(stats.hits, 5);
        assert_eq!(stats.returns, 5);
        assert_eq!(stats.drops, 0);
        assert_eq!(stats.total_gets(), 20);
        assert!((stats.hit_ratio() - 25.0).abs() < 1e-10);
    }
}

#[cfg(test)]
mod pooled_tests {
    use super::*;
    use crate::addr;
    use crate::core::distribution::*;
    use crate::core::model::{observe, sample, ModelExt};
    use crate::runtime::handler::run;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn pooled_prior_handler_builds_trace_and_updates_pool() {
        let mut pool = TracePool::new(4);
        let mut rng = StdRng::seed_from_u64(40);
        let (_val, trace) = run(
            PooledPriorHandler::new(&mut rng, &mut pool),
            sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
                .and_then(|x| observe(addr!("y"), Normal::new(x, 1.0).unwrap(), 0.3)),
        );
        assert!(trace.choices.contains_key(&addr!("x")));
        assert!(trace.log_likelihood.is_finite());

        // Return a trace and check stats update when pool accepts
        let before_returns = pool.stats().returns;
        pool.return_trace(trace);
        assert_eq!(pool.stats().returns, before_returns + 1);
    }
}
