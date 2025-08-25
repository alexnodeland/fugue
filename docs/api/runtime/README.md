# `runtime` module

## Overview

The runtime module provides the execution engine for probabilistic models through a clean effect handler architecture. It enables different interpretations of the same model (prior sampling, conditioning, scoring) while maintaining type safety and providing efficient trace management with memory optimization.

## Quick Start

```rust
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

// Execute a model with prior sampling
let model = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
let mut rng = StdRng::seed_from_u64(42);
let (value, trace) = run(
    PriorHandler { rng: &mut rng, trace: Trace::default() },
    model
);
println!("Sampled value: {:.3}, Log weight: {:.3}", value, trace.total_log_weight());
```

## Components

### `handler.rs` - Type-Safe Handler Interface

- `Handler` trait: Defines how to interpret model effects with full type safety
- `run` function: Executes a model with a handler, returning value and trace

```rust
pub trait Handler {
    // Type-specific sampling handlers
    fn on_sample_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>) -> f64;
    fn on_sample_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>) -> bool;
    fn on_sample_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>) -> u64;
    fn on_sample_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>) -> usize;

    // Type-specific observation handlers
    fn on_observe_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>, value: f64);
    fn on_observe_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>, value: bool);
    fn on_observe_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>, value: u64);
    fn on_observe_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>, value: usize);

    fn on_factor(&mut self, logw: f64);
    fn finish(self) -> Trace where Self: Sized;
}

let (result, trace) = run(handler, model);
```

### `interpreters.rs` - Built-in Handlers

- `PriorHandler`: Samples from priors, accumulates log-densities
- `ReplayHandler`: Reuses values from a base trace, falls back to sampling
- `ScoreGivenTrace`: Scores a fixed trace under the model

```rust
// Prior sampling
let (value, trace) = run(PriorHandler{rng: &mut rng, trace: Trace::default()}, model);

// Replay with different observations
let (value2, trace2) = run(ReplayHandler{rng: &mut rng, base: trace, trace: Trace::default()}, model2);

// Score existing trace
let (value3, trace3) = run(ScoreGivenTrace{base: trace, trace: Trace::default()}, model);
```

### `trace.rs` - Execution Traces

- `Trace`: Records choices and accumulated log-weights
- `Choice`: Individual random choice with address, value, and log-probability
- `ChoiceValue`: Type-safe value storage - supports `F64`, `Bool`, `U64`, `Usize`, `I64`

```rust
#[derive(Clone, Debug, Default)]
pub struct Trace {
    pub choices: BTreeMap<Address, Choice>,
    pub log_prior: f64,
    pub log_likelihood: f64,
    pub log_factors: f64,
}

impl Trace {
    pub fn total_log_weight(&self) -> f64 {
        self.log_prior + self.log_likelihood + self.log_factors
    }
}
```

### `memory.rs` - Memory Optimization

**Key Types/Functions:**

- `TracePool`: Reusable trace allocation
- `CowTrace`: Copy-on-write trace optimization
- `TraceBuilder`: Efficient trace construction
- `PooledPriorHandler`: Memory-pooled handler

**Example:**

```rust
let pool = TracePool::new();
let handler = PooledPriorHandler::new(&mut rng, &pool);
let (value, trace) = run(handler, model);
```

## Common Patterns

### Multi-Handler Execution Pipeline

Chain handlers for complex inference workflows.

```rust
// 1. Generate base trace from prior
let (_, base_trace) = run(
    PriorHandler { rng: &mut rng, trace: Trace::default() },
    model.clone()
);

// 2. Replay with observations
let (_, conditioned_trace) = run(
    ReplayHandler {
        rng: &mut rng,
        base: base_trace.clone(),
        trace: Trace::default()
    },
    model_with_observations
);

// 3. Score the result
let (_, final_trace) = run(
    ScoreGivenTrace {
        base: conditioned_trace,
        trace: Trace::default()
    },
    model.clone()
);
```

### Memory-Efficient Batch Processing

Use trace pooling for high-throughput scenarios.

```rust
let pool = TracePool::new();
let results: Vec<_> = (0..1000).map(|i| {
    let mut rng = StdRng::seed_from_u64(i);
    let handler = PooledPriorHandler::new(&mut rng, &pool);
    run(handler, model.clone())
}).collect();
```

### Type-Safe Trace Inspection

Extract values with compile-time type checking.

```rust
let trace = execution_trace;

// Type-safe value extraction
let mu: Option<f64> = trace.get_f64(&addr!("mu"));
let is_outlier: Option<bool> = trace.get_bool(&addr!("outlier"));  // Returns bool!
let count: Option<u64> = trace.get_u64(&addr!("events"));  // Returns u64!
let category: Option<usize> = trace.get_usize(&addr!("choice"));  // Returns usize!

// Validation and debugging
trace.validate().expect("Trace should be valid");
println!("Total choices: {}", trace.choices.len());
println!("Log weight breakdown: prior={:.3}, likelihood={:.3}, factors={:.3}",
    trace.log_prior, trace.log_likelihood, trace.log_factors);
```

## Performance Considerations

- **Memory**: Use `TracePool` for repeated allocations, `CowTrace` for read-heavy workloads
- **Computation**: Handlers are zero-cost abstractions with no runtime dispatch overhead
- **Best Practices**:
  - Reuse trace pools across inference runs
  - Use appropriate handler for your use case (don't use `ReplayHandler` if you don't need replay)
  - Validate traces only in debug builds for production performance
  - Consider batch processing with pooled handlers for high-throughput scenarios

## Integration

**Related Modules:**

- [`core`](../core/README.md): Execute `Model<T>` values defined in core
- [`inference`](../inference/README.md): Provide execution infrastructure for all inference algorithms
- [`error`](../error.rs): Handle runtime errors and trace validation failures

**See Also:**

- Main documentation: [API docs](https://docs.rs/fugue)
- Examples: [`examples/trace_manipulation.rs`](../../examples/trace_manipulation.rs)
- Memory benchmarks: [`benches/memory_benchmarks.rs`](../../benches/memory_benchmarks.rs)

## Extension Points

How to extend the runtime system:

1. **Custom Handlers**: Implement specialized execution strategies

   ```rust
   pub struct CustomHandler {
       // Your state here
   }

   impl Handler for CustomHandler {
       fn on_sample_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>) -> f64 {
           // Custom continuous sampling logic
       }

       fn on_sample_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>) -> bool {
           // Custom boolean sampling logic - returns bool directly!
       }

       fn on_sample_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>) -> u64 {
           // Custom counting sampling logic - returns u64 directly!
       }

       fn on_sample_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>) -> usize {
           // Custom categorical sampling logic - returns usize directly!
       }

       // ... other required methods
   }
   ```

2. **Custom Trace Types**: Extend trace functionality for specialized use cases
3. **Memory Optimizations**: Implement custom pooling strategies for specific workloads
4. **Debugging Tools**: Add instrumentation and logging handlers

## Design Principles

- **Effect Handlers**: Clean separation between model definition and execution
- **Trace-based**: All execution produces replayable, scorable traces
- **Type Safety**: Handlers are fully type-safe with compile-time guarantees
- **Composable**: Handlers can be chained and combined for complex workflows
- **Deterministic**: Same trace + same model = same result (reproducible execution)
- **Memory Efficient**: Copy-on-write semantics and pooling minimize allocations
- **Introspectable**: Full visibility into random choices and weights for debugging and analysis
