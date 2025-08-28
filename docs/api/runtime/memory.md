# Memory Optimization System

## Overview

Fugue's memory optimization system solves a critical performance problem in probabilistic programming: **allocation overhead during inference**. Probabilistic inference algorithms like MCMC and SMC generate thousands or millions of execution traces, creating significant memory pressure and allocation overhead that can dominate runtime performance.

The memory system provides a comprehensive solution through **multiple complementary strategies**:

- **Copy-on-Write Traces**: Share unchanged data between similar traces (crucial for MCMC)
- **Object Pooling**: Reuse trace allocations to eliminate allocation overhead
- **Efficient Construction**: Minimize allocations during trace building
- **Performance Monitoring**: Track and optimize memory usage patterns

This system enables **zero-allocation inference** in performance-critical scenarios while maintaining the simplicity and type safety of the core programming model.

## Usage Examples

### Basic Memory Pooling

```rust
# use fugue::*;
# use fugue::runtime::memory::*;
# use fugue::runtime::interpreters::*;
# use rand::rngs::StdRng;
# use rand::SeedableRng;

// Create a memory pool for trace reuse
let mut pool = TracePool::new(100); // Pool up to 100 traces
let mut rng = StdRng::seed_from_u64(42);

// Define model
let make_model = || {
    sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
        .bind(|x| observe(addr!("y"), Normal::new(x, 0.1).unwrap(), 1.5))
};

// Run inference with pooled handler (zero allocations after warm-up)
for iteration in 0..1000 {
    let (_, trace) = runtime::handler::run(
        PooledPriorHandler {
            rng: &mut rng,
            trace_builder: TraceBuilder::new(),
            pool: &mut pool,
        },
        make_model()
    );
    
    // Return trace to pool for reuse
    pool.return_trace(trace);
    
    // Monitor performance every 100 iterations
    if iteration % 100 == 0 {
        let stats = pool.stats();
        println!("Hit ratio: {:.1}%, Pool size: {}", 
            stats.hit_ratio(), pool.len());
    }
}

// Pool statistics show memory efficiency
let final_stats = pool.stats();
println!("Final hit ratio: {:.1}%", final_stats.hit_ratio());
println!("Total allocations avoided: {}", final_stats.hits);
```

### Copy-on-Write for MCMC

```rust
# use fugue::*;
# use fugue::runtime::memory::*;

// MCMC typically modifies only small portions of traces
// CowTrace shares unchanged data between states

// Start with a base trace from prior sampling
# let mut rng = rand::thread_rng();
# let (_, base_trace) = runtime::handler::run(
#     PriorHandler { rng: &mut rng, trace: Trace::default() },
#     sample(addr!("param"), Normal::new(0.0, 1.0).unwrap())
# );

let base_cow = CowTrace::from_trace(base_trace);

// Create many MCMC states (efficient - shares memory)
let mut mcmc_states = Vec::new();
for chain in 0..10 {
    for step in 0..100 {
        let mut state = base_cow.clone(); // Cheap clone - shares Arc
        
        // Modify only a few addresses (triggers copy-on-write only for changes)
        state.insert_choice(
            addr!("step", step),
            Choice {
                addr: addr!("step", step),
                value: ChoiceValue::F64(step as f64 * 0.1),
                logp: -0.5,
            }
        );
        
        mcmc_states.push(state);
    }
}

println!("Created {} MCMC states with minimal memory overhead", mcmc_states.len());

// Memory usage is much lower than individual traces
// because unchanged portions are shared via Arc
```

### High-Performance Trace Building

```rust
# use fugue::*;
# use fugue::runtime::memory::*;

// TraceBuilder minimizes allocations during trace construction
let mut builder = TraceBuilder::new();

// Efficiently add many choices
for i in 0..10000 {
    builder.add_sample(addr!("param", i), i as f64 * 0.1, -0.5);
    builder.add_sample_bool(addr!("flag", i), i % 2 == 0, -0.693);
    builder.add_sample_u64(addr!("count", i), (i as u64).saturating_mul(2), -1.0);
}

// Add observations and factors
builder.add_observation(-2.5); // Likelihood contribution
builder.add_factor(-0.1);       // Soft constraint

// Build final trace efficiently
let large_trace = builder.build();
assert_eq!(large_trace.choices.len(), 30000);
println!("Built large trace with {} choices", large_trace.choices.len());
```

### Custom Handler with Memory Integration

```rust
# use fugue::*;
# use fugue::runtime::memory::*;
# use rand::RngCore;

/// Custom handler that automatically manages memory pooling
struct OptimizedHandler<'a, R: RngCore> {
    rng: &'a mut R,
    pool: &'a mut TracePool,
    trace_builder: TraceBuilder,
    samples_count: usize,
}

impl<'a, R: RngCore> OptimizedHandler<'a, R> {
    fn new(rng: &'a mut R, pool: &'a mut TracePool) -> Self {
        Self {
            rng,
            pool,
            trace_builder: TraceBuilder::new(),
            samples_count: 0,
        }
    }
}

impl<'a, R: RngCore> Handler for OptimizedHandler<'a, R> {
    fn on_sample_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>) -> f64 {
        let value = dist.sample(self.rng);
        let log_prob = dist.log_prob(&value);
        self.trace_builder.add_sample(addr.clone(), value, log_prob);
        self.samples_count += 1;
        value
    }
    
    // Implement other required methods...
    # fn on_sample_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>) -> bool {
    #     let value = dist.sample(self.rng);
    #     let log_prob = dist.log_prob(&value);
    #     self.trace_builder.add_sample_bool(addr.clone(), value, log_prob);
    #     value
    # }
    # fn on_sample_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>) -> u64 { 0 }
    # fn on_sample_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>) -> usize { 0 }
    # fn on_observe_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>, value: f64) {}
    # fn on_observe_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>, value: bool) {}
    # fn on_observe_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>, value: u64) {}
    # fn on_observe_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>, value: usize) {}
    
    fn on_factor(&mut self, logw: f64) {
        self.trace_builder.add_factor(logw);
    }
    
    fn finish(self) -> Trace {
        println!("Handler processed {} samples", self.samples_count);
        self.trace_builder.build()
    }
}

// Usage example
# let mut pool = TracePool::new(50);
# let mut rng = rand::thread_rng();
# let model = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
let (result, trace) = runtime::handler::run(
    OptimizedHandler::new(&mut rng, &mut pool),
    model
);
pool.return_trace(trace); // Return to pool for reuse
```

## Design & Evolution

### Status

- **Stable**: Core memory optimization types (`CowTrace`, `TracePool`, `TraceBuilder`) are stable since v0.1
- **Performance-Critical**: These optimizations are essential for production inference workloads
- **Integration**: Seamlessly integrates with all handler types and inference algorithms

### Key Design Principles

1. **Zero-Cost Abstraction**: Memory optimizations should not compromise the programming model
2. **Composability**: Memory strategies should work together and with existing handlers
3. **Transparency**: Optimizations should be invisible to model code
4. **Measurability**: Provide metrics to validate and tune memory performance
5. **Incrementally Adoptable**: Teams can adopt optimizations gradually based on performance needs

### Architectural Decisions

#### Copy-on-Write Strategy

- Uses `Arc<BTreeMap>` for sharing unchanged data between traces
- Lazy copying only when traces diverge (perfect for MCMC where most choices unchanged)
- Trades slight access overhead for massive memory savings in typical inference patterns

#### Object Pool Design

- LIFO (stack-based) allocation for better cache locality
- Configurable bounds (min/max) for memory usage control
- Comprehensive statistics for performance monitoring and tuning
- Automatic trace clearing to prevent data leaks between uses

#### Efficient Construction

- `TraceBuilder` uses pre-allocated collections to minimize reallocations
- Type-specific methods avoid boxing/unboxing overhead
- Builder pattern separates construction from final trace immutability

### Invariants

- Pool-returned traces are always completely cleared of previous data
- CowTrace clones share immutable data until mutation occurs
- TraceBuilder maintains internal consistency (log-weights, choice counts)
- Statistics accurately reflect cache performance across all operations

### Proposal Workflow

Memory optimization enhancements follow the standard RFC process:

1. **Performance Analysis**: Demonstrate bottleneck with profiling data
2. **Design Proposal**: RFC with benchmarks showing improvement
3. **Feature Flag Implementation**: New optimizations behind experimental flags
4. **Validation**: A/B testing with real inference workloads
5. **Stabilization**: Graduate to stable API after validation

### Evolution Strategy

- **Backwards Compatible**: New optimizations are opt-in, never breaking existing code
- **Evidence-Based**: All optimizations backed by benchmarks and real-world performance data
- **Incremental**: Focus on highest-impact optimizations first (Pareto principle)

## Error Handling

Memory optimizations must handle several error conditions gracefully:

### Pool Overflow

```rust
# use fugue::*;
# use fugue::runtime::memory::*;

let mut pool = TracePool::new(10); // Small pool for demo

// Pool can handle more returns than capacity
for i in 0..20 {
    let trace = Trace::default();
    pool.return_trace(trace); // Extra traces are dropped, not stored
}

// Check statistics to detect overflow
let stats = pool.stats();
if stats.drops > stats.returns / 10 {
    println!("Warning: Pool overflow, consider increasing capacity");
    println!("Drops: {}, Returns: {}", stats.drops, stats.returns);
}
```

### Memory Pressure Handling

```rust
# use fugue::*;
# use fugue::runtime::memory::*;

let mut pool = TracePool::with_bounds(1000, 100);

// Periodically shrink pool during long-running inference
for epoch in 0..100 {
    // ... run inference ...
    
    if epoch % 10 == 0 {
        pool.shrink(); // Reclaim memory if pool is oversized
        
        let stats = pool.stats();
        if stats.hit_ratio() < 50.0 {
            println!("Warning: Low hit ratio {:.1}%, tune pool size", stats.hit_ratio());
        }
    }
}
```

### Best Practices

- Monitor pool hit ratios - target >80% for good performance
- Size pools based on inference algorithm needs (MCMC: 10-100x, SMC: 100-1000x)
- Use `shrink()` periodically in long-running inference to prevent memory bloat
- Profile memory usage in production to validate optimization effectiveness
- Consider CowTrace for MCMC, Pool for SMC/VI where traces are short-lived

## Integration Notes

### With Inference Algorithms

- **MCMC**: CowTrace ideal for sharing data between proposal states
- **SMC**: TracePool essential for particle generation/resampling
- **VI**: TraceBuilder efficient for gradient estimation with many traces
- **ABC**: Pool + Builder combination for rejection sampling loops

### With Handler System

- `PooledPriorHandler` demonstrates canonical integration pattern
- Custom handlers can use `TraceBuilder` for efficient trace construction
- Memory optimizations compose with all handler types transparently
- Zero-allocation execution possible with proper pool sizing

### Performance Characteristics

- **CowTrace Cloning**: O(1) time, O(1) memory until mutation
- **Pool Operations**: O(1) get/return, O(k) shrink where k = excess capacity
- **TraceBuilder**: O(1) amortized inserts, O(n) final build where n = choices
- **Memory Overhead**: ~8-16 bytes per pooled trace, ~16 bytes per CowTrace reference

### Benchmarking Integration

```rust
# use fugue::runtime::memory::*;
# use std::time::Instant;

// Benchmark memory optimization effectiveness
let mut pool = TracePool::new(100);
let mut total_time = std::time::Duration::ZERO;

for iteration in 0..1000 {
    let start = Instant::now();
    
    // Your inference code here using pool
    let trace = pool.get();
    // ... run model ...
    pool.return_trace(trace);
    
    total_time += start.elapsed();
}

let stats = pool.stats();
println!("Average iteration time: {:?}", total_time / 1000);
println!("Memory efficiency: {:.1}% hit ratio", stats.hit_ratio());
```

## Reference Links

### Core Types

- [`CowTrace`](../memory.rs) - Copy-on-write trace for memory sharing
- [`TracePool`](../memory.rs) - Object pool for trace reuse
- [`TraceBuilder`](../memory.rs) - Efficient trace construction
- [`PoolStats`](../memory.rs) - Performance monitoring
- [`PooledPriorHandler`](../memory.rs) - Memory-optimized handler

### Related Systems

- [`Handler`](../handler.md) - How memory optimizations integrate with execution
- [`Trace`](../trace.md) - The underlying trace representation
- [Inference Algorithms](../../inference/README.md) - Algorithms that benefit from memory optimization

### Performance Guides

- [MCMC Optimization](../../src/how-to/mcmc-performance.md) - CowTrace usage patterns
- [SMC Scaling](../../src/how-to/smc-performance.md) - Pool sizing for particle filters
- [Memory Profiling](../../src/how-to/memory-profiling.md) - Measuring optimization effectiveness

### Examples

- [`memory_pool_basic.rs`](../../../examples/memory_pool_basic.rs) - Basic pool usage
- [`cow_trace_mcmc.rs`](../../../examples/cow_trace_mcmc.rs) - MCMC with copy-on-write
- [`zero_allocation_inference.rs`](../../../examples/zero_allocation_inference.rs) - Performance-optimized inference
- [`memory_profiling_demo.rs`](../../../examples/memory_profiling_demo.rs) - Measuring memory performance
