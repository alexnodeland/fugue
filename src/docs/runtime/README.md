# Runtime System: Probabilistic Model Execution Engine

## Overview

The **runtime system** is the **execution heart** of Fugue's probabilistic programming infrastructure. It transforms the declarative `Model<A>` representations from the core module into concrete executions that can be sampled, conditioned, scored, and manipulated.

The runtime solves the fundamental challenge in probabilistic programming: **how to execute the same model description in radically different ways**. A single `Model<A>` can be:

- **Forward sampled** to generate data from priors
- **Conditioned** on observed data to perform inference  
- **Scored** to compute log-probabilities for specific executions
- **Replayed** with modified choices for MCMC proposals
- **Optimized** with memory pooling for high-throughput scenarios

This flexibility is achieved through a **clean effect handler architecture** with four integrated components:

- **[Handler System](handler.md)**: The `Handler` trait and `run` function provide type-safe execution with algebraic effects
- **[Built-in Interpreters](interpreters.md)**: Five foundational handlers (`PriorHandler`, `ReplayHandler`, `ScoreGivenTrace`, etc.)
- **[Trace System](trace.md)**: The foundational data structures (`Trace`, `Choice`, `ChoiceValue`) that record execution history
- **[Memory Optimization](memory.md)**: Efficient allocation strategies (`TracePool`, `CowTrace`, `TraceBuilder`) for production performance

The key architectural insight is the **separation of model description from execution strategy**: models describe *what* should happen, handlers define *how* it happens, and traces record *what actually happened*.

## Usage Examples

### Basic Model Execution

```rust
# use fugue::*;
# use fugue::runtime::interpreters::PriorHandler;
# use rand::rngs::StdRng;
# use rand::SeedableRng;

// Define a Bayesian linear regression model
let linear_model = || {
    sample(addr!("slope"), Normal::new(0.0, 2.0).unwrap())
        .bind(|slope| sample(addr!("intercept"), Normal::new(0.0, 1.0).unwrap())
            .bind(move |intercept| sample(addr!("noise"), Gamma::new(2.0, 1.0).unwrap())
                .bind(move |noise| {
                    // Synthetic observations
                    let x_values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
                    let y_observed = vec![2.1, 4.2, 5.8, 8.1, 10.3];
                    
                    let obs_models = x_values.into_iter().zip(y_observed).enumerate()
                        .map(|(i, (x, y_obs))| {
                            let y_pred = slope * x + intercept;
                            observe(addr!("y", i), Normal::new(y_pred, noise.sqrt()).unwrap(), y_obs)
                        }).collect::<Vec<_>>();
                    
                    sequence_vec(obs_models).map(move |_| (slope, intercept, noise))
                })))
};

// Execute with prior sampling handler
let mut rng = StdRng::seed_from_u64(42);
let (result, trace) = runtime::handler::run(
    PriorHandler { rng: &mut rng, trace: Trace::default() },
    linear_model()
);

let (slope, intercept, noise) = result;
println!("Posterior sample:");
println!("├─ Slope: {:.3}", slope);
println!("├─ Intercept: {:.3}", intercept);
println!("└─ Noise: {:.3}", noise);

println!("\nTrace diagnostics:");
println!("├─ Choices recorded: {}", trace.choices.len());
println!("├─ Prior log-weight: {:.3}", trace.log_prior);
println!("├─ Likelihood log-weight: {:.3}", trace.log_likelihood);
println!("└─ Total log-weight: {:.3}", trace.total_log_weight());
```

## Architecture Components

The runtime system consists of four tightly integrated components, each documented in detail:

### [Handler System](handler.md) - Type-Safe Execution Engine

The foundational abstraction that separates model description from execution strategy through algebraic effects.

**Core Types:**

- `Handler` trait: Type-safe interpretation of model effects with guaranteed return types
- `run` function: Executes any `Model<A>` with any `Handler` implementation

**Key Features:**

- Zero-cost abstractions with compile-time dispatch
- Type-specific methods prevent runtime casting errors
- Composable execution strategies for complex workflows

### [Built-in Interpreters](interpreters.md) - Foundational Execution Modes

Five essential handlers that cover all fundamental probabilistic programming operations.

**Core Interpreters:**

- `PriorHandler`: Forward sampling from prior distributions (the baseline)
- `ReplayHandler`: Deterministic replay with fallback sampling (MCMC proposals)
- `ScoreGivenTrace`: Log-probability computation for fixed traces (importance sampling)

**Safety Variants:**

- `SafeReplayHandler`: Error-resilient replay with graceful type mismatch handling
- `SafeScoreGivenTrace`: Production-safe scoring with invalid trace handling

### [Trace System](trace.md) - Execution History Foundation

The data structures that make probabilistic programming possible by recording execution history.

**Core Types:**

- `Trace`: Complete execution record with decomposed log-weights (prior + likelihood + factors)
- `Choice`: Single random decision with address, value, and log-probability
- `ChoiceValue`: Type-safe value storage for all distribution return types

**Key Capabilities:**

- Enables replay, scoring, and conditioning operations
- Type-safe value access with both Option and Result APIs
- Three-component log-weight decomposition for algorithmic flexibility

### [Memory Optimization](memory.md) - Production Performance Strategies

Advanced allocation strategies for high-throughput probabilistic computing.

**Core Types:**

- `TracePool`: Reusable trace allocation for batch processing
- `CowTrace`: Copy-on-write semantics for efficient trace sharing
- `TraceBuilder`: Optimized trace construction with pre-sized allocations
- `PooledPriorHandler`: Memory-pooled handler for production workloads

**Performance Benefits:**

- Reduces garbage collection pressure in high-frequency sampling
- Enables efficient parallel execution with shared trace data
- Provides detailed allocation statistics for performance monitoring

## Design & Evolution

### Status

- **Stable**: The runtime system has been stable since v0.1 and provides the foundation for all probabilistic programming operations
- **Complete**: All four components (handler, interpreters, trace, memory) provide comprehensive execution capabilities
- **Performance Critical**: Extensively optimized for high-throughput inference workloads
- **Extensible**: Clean abstractions allow custom handlers and optimization strategies

### Architectural Principles

1. **Effect Handler Separation**: Clean separation between model definition (`Model<A>`) and execution strategy (`Handler`)
2. **Trace-Centric Design**: All executions produce replayable, scorable traces that enable advanced inference
3. **Type Safety Throughout**: All value handling is type-safe with compile-time guarantees
4. **Zero-Cost Abstractions**: Handler dispatch and trace operations have no runtime overhead
5. **Memory Conscious**: Copy-on-write semantics and pooling strategies minimize allocation pressure
6. **Composable Architecture**: Handlers can be chained, combined, and extended for complex workflows

### Evolution Strategy

- **Additive Changes Only**: New handler methods, trace fields, and optimization strategies are added without breaking existing code
- **Performance Optimizations**: Internal improvements (pooling, COW) are transparent to user code
- **Extension Points**: Clean abstractions allow library users to add custom functionality
- **Backwards Compatibility**: All v0.1 code continues to work unchanged

## Integration Notes

### With Core Module

The runtime system executes `Model<A>` values defined in the core module:

- **`Model<A>` Execution**: The `run` function interprets model descriptions into concrete executions
- **Address System**: Runtime uses addresses from `core::address` for choice identification
- **Distribution Integration**: Handlers dispatch to distribution methods from `core::distribution`
- **Type Safety Bridge**: Runtime preserves the type safety guarantees established in core

### With Inference Module

The runtime provides execution infrastructure for all inference algorithms:

- **MCMC**: Trace manipulation enables proposal generation and acceptance decisions
- **SMC**: Particle generation through `PriorHandler` and reweighting via `ScoreGivenTrace`
- **Variational Inference**: Trace-based gradient computation for optimization
- **ABC**: Forward simulation capabilities for approximate Bayesian computation

### Performance Characteristics

| Operation | Complexity | Notes |
|---|---|---|
| **Handler Dispatch** | O(1) | Compile-time monomorphization, no virtual calls |
| **Choice Lookup** | O(log n) | BTreeMap lookup by address |
| **Trace Cloning** | O(n) | Optimized with COW strategies |
| **Pool Allocation** | O(1) amortized | Pre-allocated objects reused |
| **Type Access** | O(log n + 1) | Address lookup plus constant-time type extraction |

## Reference Links

### Core Components

- **[Handler System](handler.md)** - Type-safe execution engine with algebraic effects pattern
- **[Built-in Interpreters](interpreters.md)** - Five foundational handlers for all execution modes
- **[Trace System](trace.md)** - Execution history recording with type-safe value access
- **[Memory Optimization](memory.md)** - Efficient allocation strategies for production performance

### Related Modules

- **[Core Module](../core/README.md)** - Model definitions and type system that runtime executes
- **[Inference Module](../inference/README.md)** - Advanced algorithms built on runtime infrastructure
- **[Error Module](../error.rs)** - Comprehensive error handling used throughout runtime

### Implementation Guides

- [Custom Handler Implementation](../../src/how-to/custom-handlers.md) - Building specialized execution strategies
- [Memory Optimization Strategies](../../src/how-to/memory-optimization.md) - High-performance allocation patterns
- [Production Deployment](../../src/how-to/production-deployment.md) - Runtime configuration for production systems
- [Debugging Runtime Issues](../../src/how-to/runtime-debugging.md) - Tools and techniques for runtime analysis

### Examples

- [`trace_manipulation.rs`](../../../examples/trace_manipulation.rs) - Comprehensive trace operations
- [`handler_patterns.rs`](../../../examples/handler_patterns.rs) - Advanced handler usage patterns
- [`memory_optimization.rs`](../../../examples/memory_optimization.rs) - High-performance memory strategies
- [`production_inference.rs`](../../../examples/production_inference.rs) - Production deployment patterns

### Benchmarks

- [`memory_benchmarks.rs`](../../../benches/memory_benchmarks.rs) - Memory allocation performance analysis
- [`handler_benchmarks.rs`](../../../benches/handler_benchmarks.rs) - Handler dispatch and trace operation benchmarks
- [`inference_benchmarks.rs`](../../../benches/inference_benchmarks.rs) - End-to-end inference performance testing
