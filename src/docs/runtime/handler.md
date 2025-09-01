# Handler System

## Overview

Fugue's handler system solves a fundamental problem in probabilistic programming: **how to separate model specification from execution strategy**. The handler architecture enables the same probabilistic model to be executed in radically different ways—prior sampling, trace replay, scoring, MCMC—without changing a single line of model code.

This system implements the **algebraic effects** pattern with full **type safety**, where:

- `Model<A>` describes probabilistic computations as data structures
- `Handler` trait defines interpretation of probabilistic effects
- `run(handler, model)` executes the interpretation

The key innovation is **type-specific effect handling**: instead of forcing all distributions through `f64`, handlers preserve natural types (`bool`, `u64`, `usize`, `f64`) throughout the execution pipeline.

## Usage Examples

### Basic Execution Pattern

```rust
# use fugue::*;
# use fugue::runtime::interpreters::*;
# use rand::rngs::StdRng;
# use rand::SeedableRng;

// Define model once
let model = sample(addr!("coin"), Bernoulli::new(0.6).unwrap())  // Returns bool!
    .bind(|heads| {
        if heads {  // Natural boolean logic
            sample(addr!("reward"), Normal::new(10.0, 2.0).unwrap())
        } else {
            sample(addr!("penalty"), Normal::new(-5.0, 1.0).unwrap())
        }
    });

// Execute with different handlers for different purposes
let mut rng = StdRng::seed_from_u64(42);

// Create the model function to avoid clone issues
let make_model = || {
    sample(addr!("coin"), Bernoulli::new(0.6).unwrap())
        .bind(|heads| {
            if heads {
                sample(addr!("reward"), Normal::new(10.0, 2.0).unwrap())
            } else {
                sample(addr!("penalty"), Normal::new(-5.0, 1.0).unwrap())
            }
        })
};

// 1. Prior sampling - generate random execution
let (value1, trace1) = runtime::handler::run(
    PriorHandler { rng: &mut rng, trace: Trace::default() },
    make_model()
);
println!("Prior sample: {}, coin was: {:?}",
    value1, trace1.choices[&addr!("coin")]);

// 2. Replay - reuse choices from trace1, sample any missing
let (value2, trace2) = runtime::handler::run(
    ReplayHandler {
        rng: &mut rng,
        base: trace1,
        trace: Trace::default()
    },
    make_model()
);

// 3. Scoring - compute log-probability of trace2's choices
let (value3, score_trace) = runtime::handler::run(
    ScoreGivenTrace {
        base: trace2,
        trace: Trace::default()
    },
    make_model()
);
println!("Log-probability: {}", score_trace.total_log_weight());
```

### Type-Safe Effect Handling

The handler system preserves the natural return types of distributions:

```rust
# use fugue::*;
# use fugue::runtime::interpreters::*;
# use rand::rngs::StdRng;
# use rand::SeedableRng;

let type_safe_model = prob! {
    // Each sample returns its natural type
    let is_outlier <- sample(addr!("outlier"), Bernoulli::new(0.1).unwrap());  // → bool
    let component <- sample(addr!("component"), Categorical::uniform(3).unwrap());  // → usize
    let count <- sample(addr!("events"), Poisson::new(3.0).unwrap());  // → u64
    let value <- sample(addr!("value"), Normal::new(0.0, 1.0).unwrap());  // → f64

    // Natural usage - no casting needed!
    let options = vec!["low", "medium", "high"];
    let strategy = options[component];  // Safe indexing with usize

    let multiplier = if is_outlier { 2.0 } else { 1.0 };  // Natural boolean logic
    let adjusted = value * multiplier + count as f64;  // Direct arithmetic

    pure((strategy, adjusted))
};

// Handler automatically dispatches to correct type-specific methods
let mut rng = StdRng::seed_from_u64(123);
let (_result, trace) = runtime::handler::run(
    PriorHandler { rng: &mut rng, trace: Trace::default() },
    type_safe_model
);

// Trace preserves type information
match &trace.choices[&addr!("outlier")].value {
    ChoiceValue::Bool(b) => println!("Outlier flag: {}", b),
    _ => unreachable!(),
}
```

### Custom Handler Implementation

```rust
# use fugue::*;
# use rand::rngs::StdRng;
# use rand::SeedableRng;

/// Handler that logs all sampling operations
struct LoggingHandler<H: Handler> {
    inner: H,
    log: Vec<String>,
}

impl<H: Handler> Handler for LoggingHandler<H> {
    fn on_sample_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>) -> f64 {
        let value = self.inner.on_sample_f64(addr, dist);
        self.log.push(format!("Sampled {} = {:.3}", addr, value));
        value
    }

    fn on_sample_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>) -> bool {
        let value = self.inner.on_sample_bool(addr, dist);
        self.log.push(format!("Sampled {} = {}", addr, value));
        value
    }

    fn on_sample_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>) -> u64 {
        let value = self.inner.on_sample_u64(addr, dist);
        self.log.push(format!("Sampled {} = {}", addr, value));
        value
    }

    fn on_sample_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>) -> usize {
        let value = self.inner.on_sample_usize(addr, dist);
        self.log.push(format!("Sampled {} = {}", addr, value));
        value
    }

    fn on_observe_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>, value: f64) {
        self.log.push(format!("Observed {} = {:.3}", addr, value));
        self.inner.on_observe_f64(addr, dist, value);
    }

    fn on_observe_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>, value: bool) {
        self.log.push(format!("Observed {} = {}", addr, value));
        self.inner.on_observe_bool(addr, dist, value);
    }

    fn on_observe_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>, value: u64) {
        self.log.push(format!("Observed {} = {}", addr, value));
        self.inner.on_observe_u64(addr, dist, value);
    }

    fn on_observe_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>, value: usize) {
        self.log.push(format!("Observed {} = {}", addr, value));
        self.inner.on_observe_usize(addr, dist, value);
    }

    fn on_factor(&mut self, logw: f64) {
        self.log.push(format!("Factor: {:.3}", logw));
        self.inner.on_factor(logw);
    }

    fn finish(self) -> Trace {
        for entry in &self.log {
            println!("{}", entry);
        }
        self.inner.finish()
    }
}

// Example usage
# let mut rng = StdRng::seed_from_u64(42);
# let base_handler = PriorHandler { rng: &mut rng, trace: Trace::default() };
# let logging_handler = LoggingHandler { inner: base_handler, log: Vec::new() };
# let model = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
# let (_result, _trace) = runtime::handler::run(logging_handler, model);
```

## Design & Evolution

### Status

- **Stable**: Core `Handler` trait and `run` function are stable since v0.1
- **Type-safe effects**: The type-specific handler methods are a key architectural decision
- **Performance**: Zero-cost abstraction - compiles to direct function calls

### Key Design Principles

1. **Separation of Concerns**: Models describe _what_ to compute, handlers define _how_ to interpret
2. **Type Preservation**: Each distribution type gets its own handler method to avoid lossy conversions
3. **Effect Isolation**: All side effects (randomness, trace updates) are isolated in handlers
4. **Composability**: Handlers can wrap other handlers for cross-cutting concerns

### Invariants

- Handlers must be deterministic given the same inputs and RNG state
- The `finish()` method is called exactly once at the end of execution
- Type-specific methods must preserve the semantics of the distribution types
- Trace updates must maintain internal consistency (addresses, log-weights)

### Proposal Workflow

Handler extensions follow the standard RFC process:

1. Open a Design Proposal (DP) issue for new handler types
2. Implement behind feature flag for experimental handlers
3. Stabilize based on usage feedback and performance characteristics
4. Document patterns and integration points

### Evolution Strategy

- **Backwards Compatible**: New handler methods can be added without breaking existing implementations
- **Performance**: Handler dispatch is compile-time resolved, enabling aggressive optimization
- **Extensibility**: The trait design supports both simple and complex handler implementations

## Error Handling

Handlers must gracefully handle several error conditions:

### Distribution Parameter Errors

```rust
# use fugue::*;
# struct MyHandler;
# impl Handler for MyHandler {
// Handlers should validate distribution parameters
fn on_sample_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>) -> f64 {
    let value = dist.sample(&mut rand::thread_rng());
    if !value.is_finite() {
        // Log error, return default, or propagate failure
        eprintln!("Invalid sample at {}: {}", addr, value);
        return 0.0; // or handle appropriately
    }
    value
}
#     fn on_sample_bool(&mut self, _: &Address, _: &dyn Distribution<bool>) -> bool { false }
#     fn on_sample_u64(&mut self, _: &Address, _: &dyn Distribution<u64>) -> u64 { 0 }
#     fn on_sample_usize(&mut self, _: &Address, _: &dyn Distribution<usize>) -> usize { 0 }
#     fn on_observe_f64(&mut self, _: &Address, _: &dyn Distribution<f64>, _: f64) {}
#     fn on_observe_bool(&mut self, _: &Address, _: &dyn Distribution<bool>, _: bool) {}
#     fn on_observe_u64(&mut self, _: &Address, _: &dyn Distribution<u64>, _: u64) {}
#     fn on_observe_usize(&mut self, _: &Address, _: &dyn Distribution<usize>, _: usize) {}
#     fn on_factor(&mut self, _: f64) {}
#     fn finish(self) -> Trace { Trace::default() }
# }
```

### Address Collisions

- The same address used twice in a model is a serious error
- ReplayHandler and ScoreGivenTrace must handle missing addresses gracefully
- Consider using `Result<T, HandlerError>` for handlers that can fail

### Trace Consistency

- Log-weights must remain finite during normal operation
- Infinite log-weights (from `guard(false)` or impossible observations) should be handled explicitly
- Memory handlers must manage trace lifecycle correctly

### Best Practices

- Always check `is_finite()` on log-probabilities and samples
- Use defensive programming for address lookups in replay scenarios
- Provide clear error messages that include the problematic address
- Consider timeouts for handlers that might run indefinitely

## Integration Notes

### With Model System

- Handlers work seamlessly with all Model variants and combinators
- The `run` function handles Model continuation passing automatically
- Type dispatch happens at compile time through the `SampleType` trait

### With Inference Algorithms

- **MCMC**: Uses combinations of ReplayHandler and ScoreGivenTrace for proposals and acceptance
- **SMC**: Uses PriorHandler for particle generation and ScoreGivenTrace for reweighting
- **ABC**: Uses PriorHandler with custom distance functions in the handler logic

### With Memory Management

- `PooledPriorHandler` provides zero-allocation execution for performance-critical code
- `CowTrace` enables efficient trace sharing between handlers
- Memory handlers integrate with the standard Handler trait without modification

### Performance Characteristics

- Handler dispatch is zero-cost (resolved at compile time)
- Trace operations are O(log n) for address lookups using BTreeMap
- Memory pooling can eliminate allocation overhead entirely
- Type preservation avoids boxing/unboxing costs

## Reference Links

### Core Types

- [`Handler`](../handler.rs) - Main handler trait definition
- [`run`](../handler.rs) - Model execution function
- [`Trace`](../trace.rs) - Execution trace representation
- [`Model`](../../core/model.md) - Probabilistic model types

### Built-in Handlers

- [`PriorHandler`](../interpreters.rs) - Forward sampling from priors
- [`ReplayHandler`](../interpreters.rs) - Trace replay with fallback
- [`ScoreGivenTrace`](../interpreters.rs) - Fixed trace scoring
- [`PooledPriorHandler`](../memory.rs) - Memory-optimized sampling

### Usage Patterns

- [Custom Handlers Guide](../../src/how-to/custom-handlers.md) - Building new handler types
- [Memory Optimization](../memory.md) - Using pooled handlers for performance
- [MCMC Integration](../../inference/README.md) - How handlers enable inference algorithms

### Examples

- [`handler_basic.rs`](../../../examples/handler_basic.rs) - Basic handler usage
- [`custom_logging_handler.rs`](../../../examples/custom_logging_handler.rs) - Custom handler implementation
- [`trace_replay_patterns.rs`](../../../examples/trace_replay_patterns.rs) - Advanced replay scenarios
