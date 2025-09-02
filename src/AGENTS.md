# Agent Context: Source Code Directory

## Purpose

The `src/` directory contains the core implementation of the Fugue probabilistic programming library. This includes the fundamental probabilistic programming primitives, runtime system, inference algorithms, and comprehensive error handling.

## Module Architecture

```
src/
├── lib.rs              # Public API exports and crate root
├── core/               # Fundamental PPL abstractions
│   ├── address.rs      # Site addressing and naming
│   ├── distribution.rs # Type-safe probability distributions  
│   ├── model.rs        # Monadic model composition
│   └── numerical.rs    # Numerical stability utilities
├── runtime/            # Execution engine
│   ├── handler.rs      # Handler trait and execution framework
│   ├── interpreters.rs # Built-in model interpreters
│   ├── trace.rs        # Execution history management
│   └── memory.rs       # Memory optimization strategies
├── inference/          # Inference algorithms
│   ├── mcmc/           # Markov Chain Monte Carlo
│   ├── smc/            # Sequential Monte Carlo
│   ├── vi/             # Variational Inference
│   ├── abc/            # Approximate Bayesian Computation
│   └── diagnostics.rs  # Convergence and validation tools
├── error.rs            # Comprehensive error handling
└── macros.rs           # Procedural macros (prob!, plate!, addr!)
```

## Core Design Principles

### Testing-First Development
**Run tests after every code change.** The probabilistic nature of this library means that seemingly minor changes can have significant impacts on correctness and numerical stability.

```bash
# Essential workflow after any source changes
make test

# Complete validation pipeline
make all

# Focus on unit tests during development
cargo test --lib

# Check for performance regressions
make bench
```

### Monadic Architecture
- `Model<T>` as the central abstraction for probabilistic programs
- Pure functional composition through `bind`, `map`, `pure`
- Deferred execution with pluggable interpreters
- Zero-cost abstractions that compile to efficient code

### Type Safety
- Distribution return types match their mathematical domains
- Compile-time prevention of common modeling errors
- Rich error types with contextual information
- Validation at distribution construction time

### Production Readiness
- Numerically stable algorithms (log-space computation)
- Memory optimization (pooling, copy-on-write)
- Comprehensive error handling and recovery
- Performance monitoring and diagnostics

## Module-Specific Context

### `core/` - Fundamental Abstractions

**`address.rs`** - Site Addressing
- Every random choice has a unique, stable address
- Hierarchical naming: `"simple"`, `"indexed#5"`, `"scope::name"`
- Critical for reproducibility and inference targeting
- Address collisions are programming errors, not runtime failures

**`distribution.rs`** - Probability Distributions
- Type-safe distributions with natural return types
- Comprehensive validation at construction time
- Numerically stable implementation of PDF/PMF/CDF
- Error handling for parameter validation

**`model.rs`** - Monadic Model Composition
- `Model<T>` represents probabilistic computations
- Monadic operations: `bind`, `map`, `pure`, `sequence_vec`
- Integration with addressing system for site naming
- Deferred execution enables multiple interpretation strategies

**`numerical.rs`** - Numerical Stability
- Log-space arithmetic: `log_sum_exp`, `log1p_exp`
- Safe logarithm computation with proper error handling
- Probability normalization in log space
- Guard rails against overflow/underflow

### `runtime/` - Execution Engine

**`handler.rs`** - Execution Framework
- `Handler` trait defines interpretation strategy
- `run()` function executes models with given handler
- Type-safe dispatch to handler methods
- Integration point for custom execution strategies

**`interpreters.rs`** - Built-in Handlers
- `PriorHandler`: Forward sampling from priors
- `ReplayHandler`: Deterministic replay with trace
- `ScoreGivenTrace`: Compute log probabilities
- Safe variants with enhanced error checking

**`trace.rs`** - Execution History
- Records all random choices and observations
- Enables replay, scoring, and debugging
- Type-safe value storage and retrieval
- Memory-efficient representation

**`memory.rs`** - Performance Optimization
- `TracePool`: Reusable trace allocation
- `CowTrace`: Copy-on-write semantics
- `PooledHandler`: Memory-pooled execution
- Production-oriented memory management

### `inference/` - Inference Algorithms

**MCMC** (`mcmc/`)
- Metropolis-Hastings with adaptive proposals
- Hamiltonian Monte Carlo implementation
- Convergence diagnostics (R-hat, ESS)
- Multiple chain support for parallel sampling

**SMC** (`smc/`)
- Sequential importance sampling
- Particle filtering for dynamic models
- Adaptive resampling strategies
- Parallel particle processing

**Variational Inference** (`vi/`)
- Mean-field approximations
- Automatic differentiation for gradients
- ELBO optimization strategies
- Convergence monitoring

**ABC** (`abc/`)
- Rejection sampling with distance functions
- SMC-ABC for complex posteriors
- Summary statistic computation
- Distance function composition

**`diagnostics.rs`** - Validation Tools
- Convergence assessment (R-hat, effective sample size)
- Parameter summary statistics
- Trace visualization utilities
- Model validation helpers

### `error.rs` - Error Handling

Comprehensive error taxonomy with rich context:
- `InvalidParameters`: Distribution parameter validation
- `NumericalError`: Overflow, underflow, precision issues
- `ModelError`: Model composition and execution errors
- `InferenceError`: Algorithm-specific failures
- `TraceError`: Trace manipulation problems
- `TypeMismatch`: Type safety violations

### `macros.rs` - Ergonomic Abstractions

**`prob!` Macro** - Do-notation
```rust
prob! {
    let x <- sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
    let y <- sample(addr!("y"), Normal::new(x, 0.5).unwrap());
    pure((x, y))
}
```

**`plate!` Macro** - Vectorized Operations
```rust
plate!(i in 0..n => {
    sample(addr!("x", i), Normal::new(mu, sigma).unwrap())
})
```

**`addr!` Macro** - Address Construction
```rust
addr!("param")           // Simple address
addr!("data", i)         // Indexed address  
scoped_addr!("model", "param") // Scoped address
```

## Development Patterns

### Adding New Distributions
1. **Implement `Distribution<T>` trait**
   ```rust
   impl Distribution<f64> for MyDistribution {
       fn sample<R: Rng>(&self, rng: &mut R) -> f64 { ... }
       fn log_pdf(&self, value: f64) -> f64 { ... }
   }
   ```

2. **Add parameter validation**
   ```rust
   impl MyDistribution {
       pub fn new(param: f64) -> Result<Self, FugueError> {
           if param <= 0.0 {
               return Err(invalid_parameters!(
                   "Parameter must be positive, got {}", param
               ));
           }
           Ok(Self { param })
       }
   }
   ```

3. **Include comprehensive tests**
   - Parameter validation edge cases
   - Statistical properties validation
   - Numerical stability testing

### Implementing Interpreters
1. **Implement `Handler` trait**
   ```rust
   impl<R: Rng> Handler for MyHandler<R> {
       fn sample<T>(&mut self, addr: &Address, dist: &dyn Distribution<T>) -> T {
           // Custom sampling logic
       }
       
       fn observe<T>(&mut self, addr: &Address, dist: &dyn Distribution<T>, value: T) {
           // Custom observation handling
       }
   }
   ```

2. **Consider trace management**
   - How to store/retrieve choices
   - Memory efficiency considerations
   - Error handling strategies

3. **Integration with existing patterns**
   - Composability with other handlers
   - Memory optimization compatibility
   - Debugging and introspection support

### Inference Algorithm Development
1. **Design around existing abstractions**
   - Use `Handler` infrastructure for model execution
   - Leverage `Trace` for execution history
   - Integrate with diagnostics framework

2. **Consider convergence properties**
   - Implement appropriate diagnostics
   - Provide stopping criteria
   - Support parallel execution where applicable

3. **Performance optimization**
   - Memory pooling for high-throughput scenarios
   - Vectorization opportunities
   - Numerical stability considerations

## Code Quality Standards

### Error Handling
- Use `FugueResult<T>` for fallible operations
- Provide rich error context with `ErrorContext`
- Prefer explicit error propagation over panics
- Include error recovery guidance where possible

### Documentation
- Public APIs require comprehensive doc comments
- Include usage examples in documentation
- Document mathematical properties and assumptions
- Provide links to relevant literature

### Testing
- Unit tests for individual functions/methods
- Property-based testing for mathematical properties  
- Integration tests for cross-module interactions
- Benchmark performance-critical paths

### Performance
- Profile before optimizing
- Use appropriate data structures for access patterns
- Consider memory allocation patterns
- Leverage zero-cost abstractions

## Common Pitfalls

### Numerical Stability
```rust
// BAD: Direct probability computation
let prob = p1 * p2 * p3; // Can underflow

// GOOD: Log-space computation  
let log_prob = log_p1 + log_p2 + log_p3;
let prob = log_prob.exp(); // Or keep in log space
```

### Address Management
```rust
// BAD: Non-deterministic addressing
let addr = format!("param_{}", rng.gen::<u64>()); // Random component

// GOOD: Deterministic addressing
let addr = addr!("param", deterministic_index); // Reproducible
```

### Memory Management
```rust
// Consider trace pooling for hot paths
let mut pool = TracePool::new(capacity);
let handler = PooledPriorHandler::new(&mut rng, &mut pool);
```

### Error Propagation
```rust
// Use ? operator for error propagation
fn model_function() -> FugueResult<Model<f64>> {
    let dist = Normal::new(0.0, 1.0)?; // Propagate validation errors
    Ok(sample(addr!("x"), dist))
}
```

## Integration Points

### With External Crates
- **`rand`**: Random number generation, seeding, distribution sampling
- **Standard Library**: Collections, iterators, numerical traits
- **`serde`** (optional): Serialization of traces and parameters

### Performance Profiling
```bash
# Profile specific functionality
cargo flamegraph --test performance_tests

# Memory profiling
valgrind --tool=massif target/debug/examples/large_model

# Benchmark suite
cargo bench --bench inference_benchmarks
```

### Debugging Strategies
- Use `trace.choices()` to inspect execution history
- Enable debug logging for detailed execution traces
- Use safe interpreters in development for enhanced error checking
- Leverage convergence diagnostics for inference debugging

This modular architecture enables independent development of components while maintaining strong integration through well-defined interfaces.