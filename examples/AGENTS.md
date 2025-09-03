# Agent Context: Examples Directory

## Purpose

The `examples/` directory contains comprehensive, real-world examples demonstrating the capabilities and usage patterns of the Fugue probabilistic programming library. These examples serve as both documentation and validation that the library works correctly for practical applications.

## Structure

```text
examples/
├── advanced_distribution_patterns.rs  # Complex distribution composition
├── bayesian_coin_flip.rs              # Basic Bayesian inference
├── building_complex_models.rs         # Model composition patterns
├── classification.rs                  # Bayesian classification models
├── custom_handlers.rs                 # Custom interpreter development
├── debugging_models.rs                # Debugging and introspection
├── hierarchical_models.rs             # Multi-level modeling
├── linear_regression.rs               # Bayesian linear regression
├── mixture_models.rs                  # Gaussian mixture models
├── optimizing_performance.rs          # Performance optimization techniques
├── production_deployment.rs           # Production-ready patterns
├── trace_manipulation.rs              # Trace inspection and modification
├── type_safety.rs                     # Type system demonstrations
└── working_with_distributions.rs      # Distribution library usage
```

## Example Categories

### Learning Examples

**Purpose**: Introduce concepts progressively for new users

- `bayesian_coin_flip.rs` - Basic probabilistic modeling
- `working_with_distributions.rs` - Distribution library overview
- `type_safety.rs` - Type system benefits and safety

### Practical Applications  

**Purpose**: Real-world modeling scenarios

- `linear_regression.rs` - Bayesian linear regression
- `classification.rs` - Probabilistic classification
- `hierarchical_models.rs` - Multi-level data structures
- `mixture_models.rs` - Clustering and density estimation

### Advanced Techniques

**Purpose**: Sophisticated modeling and optimization patterns

- `advanced_distribution_patterns.rs` - Complex distribution composition
- `building_complex_models.rs` - Large-scale model architecture
- `trace_manipulation.rs` - Advanced trace operations
- `custom_handlers.rs` - Extending the interpreter system

### Production Guidance

**Purpose**: Deployment and optimization for real systems

- `production_deployment.rs` - Memory management, error handling
- `optimizing_performance.rs` - Performance tuning strategies
- `debugging_models.rs` - Troubleshooting and diagnostics

## Development Guidelines

### Mandatory Testing Practice

**Always run examples after making changes.** Examples must remain functional as they serve as both documentation and validation.

```bash
# Test all examples after any changes
make test

# Run specific example
cargo run --example bayesian_coin_flip

# Check that all examples compile and run
cargo check --examples

# Run examples with full optimization
cargo run --release --example production_deployment

# Complete validation
make all
```

### Example Quality Standards

#### Code Quality

- **Self-contained**: Each example should be runnable independently
- **Well-documented**: Extensive comments explaining concepts and code
- **Production-ready**: Demonstrate best practices, not just minimal working code
- **Error handling**: Show proper error management patterns
- **Performance-aware**: Include memory and computational considerations

#### Educational Value

- **Progressive complexity**: Start simple, build to advanced concepts
- **Concept focus**: Each example should have a clear learning objective
- **Real-world relevance**: Use realistic data and scenarios where possible
- **Cross-references**: Point to related examples and documentation

#### Technical Requirements

- **Reproducible results**: Use fixed random seeds where appropriate
- **Numerical stability**: Demonstrate stable computational patterns
- **Type safety**: Showcase the benefits of the type system
- **Memory efficiency**: Show memory-conscious programming patterns

## Example Development Patterns

### Basic Example Structure

```rust
//! # Bayesian Coin Flip Example
//! 
//! Demonstrates basic probabilistic modeling with parameter inference.
//! Shows the complete workflow from model definition to posterior analysis.

use fugue::prelude::*;
use rand::{rngs::StdRng, SeedableRng};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Fixed seed for reproducible results
    let mut rng = StdRng::seed_from_u64(42);
    
    // Model definition with clear documentation
    let model = prob! {
        // Prior on coin bias
        let bias <- sample(addr!("bias"), Beta::new(1.0, 1.0)?);
        
        // Observed coin flips
        let _obs <- observe(addr!("flips"), Binomial::new(10, bias)?, 7);
        
        pure(bias)
    };
    
    // Inference with appropriate sample sizes
    let samples = adaptive_mcmc_chain(&mut rng, || model.clone(), 5000, 2000)?;
    
    // Analysis and interpretation
    analyze_posterior_samples(&samples)?;
    
    Ok(())
}

fn analyze_posterior_samples(samples: &[(f64, Trace)]) -> Result<(), Box<dyn std::error::Error>> {
    // Extract parameter values
    let bias_samples: Vec<f64> = samples.iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("bias")))
        .collect();
    
    // Compute summary statistics
    let mean = bias_samples.iter().sum::<f64>() / bias_samples.len() as f64;
    let variance = bias_samples.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / bias_samples.len() as f64;
    
    println!("Posterior mean: {:.3}", mean);
    println!("Posterior variance: {:.3}", variance);
    
    Ok(())
}
```

### Production Example Patterns

```rust
//! # Production Deployment Example
//! 
//! Demonstrates production-ready patterns including error handling,
//! memory management, and performance optimization.

use fugue::prelude::*;
use fugue::runtime::memory::{TracePool, PooledPriorHandler};

fn production_inference_pipeline(
    data: &[f64],
    config: &InferenceConfig,
) -> Result<PosteriorSummary, InferenceError> {
    // Use memory pooling for high-throughput scenarios
    let mut trace_pool = TracePool::new(config.pool_capacity);
    let mut rng = StdRng::seed_from_u64(config.random_seed);
    
    // Model with comprehensive error handling
    let model = create_validated_model(data)?;
    
    // Run inference with pooled memory management
    let handler = PooledPriorHandler::new(&mut rng, &mut trace_pool);
    let samples = run_inference_with_diagnostics(handler, model, config)?;
    
    // Validate results before returning
    validate_convergence(&samples)?;
    
    Ok(compute_posterior_summary(&samples))
}
```

### Advanced Pattern Examples

```rust
//! # Custom Handler Development
//! 
//! Shows how to implement custom interpreters for specialized use cases.

impl<R: Rng> Handler for CustomHandler<R> {
    fn sample<T>(&mut self, addr: &Address, dist: &dyn Distribution<T>) -> T {
        // Custom sampling logic with logging/monitoring
        let value = dist.sample(&mut self.rng);
        self.log_sampling_event(addr, &value);
        value
    }
    
    fn observe<T>(&mut self, addr: &Address, dist: &dyn Distribution<T>, value: T) {
        // Custom observation handling with validation
        let log_likelihood = dist.log_pdf(value);
        if !log_likelihood.is_finite() {
            self.handle_observation_error(addr, log_likelihood);
        }
        self.accumulate_log_likelihood(log_likelihood);
    }
}
```

## Testing Examples

### Validation Requirements

**Every example must be tested.** Examples that don't run correctly mislead users and damage library credibility.

```bash
# Test individual example
cargo run --example bayesian_coin_flip

# Test all examples in sequence
for example in examples/*.rs; do
    name=$(basename "$example" .rs)
    echo "Testing example: $name"
    cargo run --example "$name" || exit 1
done

# Check examples compile with all feature combinations
cargo check --examples --all-features
cargo check --examples --no-default-features
```

### Example Testing Patterns

- **Output validation**: Verify results are reasonable and converged
- **Performance benchmarking**: Ensure examples complete in reasonable time
- **Memory usage**: Monitor memory consumption for large examples
- **Error scenarios**: Test error handling paths where applicable

## Integration with Documentation

### Cross-References

- Examples should be referenced from mdbook documentation
- Use `#include` directives to embed code sections in docs
- Maintain bidirectional links between concepts and examples
- Update documentation when example interfaces change

### Code Extraction for Docs

```markdown
<!-- In documentation -->
```rust,ignore
{{#include ../../../examples/bayesian_coin_flip.rs:model_definition}}
```

This allows documentation to stay synchronized with working code.

## Common Example Pitfalls

### Reproducibility Issues

```rust
// BAD: Non-deterministic examples
let mut rng = StdRng::from_entropy(); // Results vary every run

// GOOD: Fixed seed for reproducible results  
let mut rng = StdRng::seed_from_u64(42); // Consistent results
```

### Poor Error Handling

```rust
// BAD: Panicking on errors
let dist = Normal::new(0.0, -1.0).unwrap(); // Can panic

// GOOD: Proper error propagation
let dist = Normal::new(0.0, sigma)?; // Returns Result
```

### Inadequate Sample Sizes

```rust
// BAD: Too few samples for reliable inference
let samples = mcmc_chain(&mut rng, model, 100, 50); // Unreliable

// GOOD: Adequate samples with burn-in
let samples = mcmc_chain(&mut rng, model, 5000, 2000); // Reliable
```

### Missing Context

```rust
// BAD: Unexplained magic numbers
let prior = Normal::new(0.0, 2.5)?; // Why 2.5?

// GOOD: Documented parameter choices
// Weakly informative prior allowing parameters in [-5, 5] range
let prior = Normal::new(0.0, 2.5)?;
```

## Performance Considerations

### Memory Management

- Use `TracePool` for high-frequency inference
- Consider streaming processing for large datasets
- Monitor memory usage in long-running examples
- Demonstrate memory cleanup patterns

### Computational Efficiency

- Profile examples to identify bottlenecks
- Show both simple and optimized versions where relevant
- Use appropriate inference algorithms for problem scale
- Demonstrate parallel processing where applicable

### Numerical Stability

- Use log-space computations for probability calculations
- Validate parameter ranges and handle edge cases
- Show robust inference practices
- Include convergence diagnostics

## Getting Started with Examples

### For New Users

1. Start with `bayesian_coin_flip.rs` for basic concepts
2. Progress to `working_with_distributions.rs` for distribution usage
3. Try `linear_regression.rs` for practical modeling
4. Explore `type_safety.rs` for understanding the type system

### For Developers

1. Study `custom_handlers.rs` for extension patterns
2. Review `production_deployment.rs` for best practices
3. Examine `optimizing_performance.rs` for efficiency techniques
4. Use `debugging_models.rs` for troubleshooting approaches

### Running All Examples

```bash
# Quick validation that all examples work
make test

# Run examples individually with timing
time cargo run --release --example bayesian_coin_flip

# Monitor resource usage
/usr/bin/time -v cargo run --example hierarchical_models
```

This comprehensive approach ensures examples serve as both effective learning tools and validation of library functionality.
