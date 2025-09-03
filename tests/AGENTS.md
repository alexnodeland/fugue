# Agent Context: Testing Directory

## Purpose

The `tests/` directory contains integration tests that validate the public API and end-to-end workflows of the Fugue probabilistic programming library. These tests complement unit tests within individual modules to ensure overall system correctness.

## Structure

```text
tests/
├── end_to_end_workflows.rs    # Complete PPL workflows (MCMC, SMC, etc.)
├── inference_integration.rs   # Integration between inference and core
├── model_execution.rs         # Model composition and execution patterns
├── public_api_coverage.rs     # Public API completeness validation
└── public_api_validation.rs   # API contract and behavior verification
```

## Testing Philosophy

### Critical Testing Mandate

**Run tests after every change.** Integration tests are the primary validation mechanism for ensuring the probabilistic programming library works correctly end-to-end.

```bash
# Always run after making any changes
make test

# For comprehensive validation
make all

# For coverage analysis
make coverage

# Integration tests only
cargo test --test '*'
```

### Integration Focus

- Test **interactions** between modules, not isolated units
- Validate **public API contracts** and behavior
- Ensure **end-to-end workflows** work correctly
- Verify **cross-module consistency**

### Probabilistic Testing Strategies

1. **Deterministic Testing**: Use fixed seeds for reproducible results
2. **Statistical Testing**: Validate distributional properties over multiple runs
3. **Property-Based Testing**: Check mathematical invariants
4. **Convergence Testing**: Verify inference algorithm behavior

## Test Categories

### End-to-End Workflows (`end_to_end_workflows.rs`)

Complete probabilistic programming workflows from model definition to inference results.

**Testing Patterns:**

```rust
#[test]
fn test_bayesian_regression_workflow() {
    let mut rng = StdRng::seed_from_u64(42);
    
    // Define model
    let model = prob! {
        let beta <- sample(addr!("beta"), Normal::new(0.0, 1.0).unwrap());
        // ... model definition
        pure(beta)
    };
    
    // Run inference
    let samples = adaptive_mcmc_chain(&mut rng, || model.clone(), 1000, 500);
    
    // Validate results
    assert!(samples.len() == 500);
    let beta_samples: Vec<f64> = extract_f64_values(&samples, &addr!("beta"));
    assert!(beta_samples.iter().all(|&x| x.is_finite()));
}
```

### Model Execution (`model_execution.rs`)

Model composition, address management, and interpreter behavior.

**Key Test Areas:**

- Model composition with `bind`, `map`, `pure`
- Address collision detection and scoping
- Interpreter consistency across different handlers
- Memory management and trace manipulation

### Inference Integration (`inference_integration.rs`)

Integration between inference algorithms and core probabilistic programming primitives.

**Testing Focus:**

- MCMC proposal generation and acceptance
- SMC particle resampling and weighting  
- Variational inference gradient computation
- ABC distance function evaluation

### Public API Coverage (`public_api_coverage.rs`)

Systematic validation that all public APIs are exercised and behave correctly.

**Coverage Areas:**

- All public functions and methods
- Error conditions and edge cases
- API stability and backward compatibility
- Documentation example validation

### Public API Validation (`public_api_validation.rs`)

Contract testing for public API behavior and invariants.

**Validation Patterns:**

- Input validation and error handling
- Return value contracts and types
- Side effect verification
- Thread safety where applicable

## Testing Best Practices

### Reproducibility

```rust
use rand::{rngs::StdRng, SeedableRng};

#[test]
fn reproducible_test() {
    let mut rng = StdRng::seed_from_u64(42); // Fixed seed
    // Test implementation with deterministic behavior
}
```

### Statistical Validation

```rust
#[test]
fn test_normal_distribution_properties() {
    let mut rng = StdRng::seed_from_u64(123);
    let samples: Vec<f64> = (0..10000)
        .map(|_| Normal::new(0.0, 1.0).unwrap().sample(&mut rng))
        .collect();
    
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let variance = samples.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / samples.len() as f64;
    
    assert!((mean.abs() < 0.1), "Mean should be close to 0");
    assert!((variance - 1.0).abs() < 0.1, "Variance should be close to 1");
}
```

### Error Condition Testing

```rust
#[test]
fn test_invalid_distribution_parameters() {
    let result = Normal::new(0.0, -1.0); // Invalid negative std dev
    assert!(result.is_err());
    
    if let Err(err) = result {
        assert!(err.is_validation_error());
        assert!(err.to_string().contains("standard deviation"));
    }
}
```

### Convergence Testing

```rust
#[test]
fn test_mcmc_convergence() {
    let mut rng = StdRng::seed_from_u64(789);
    
    let samples = adaptive_mcmc_chain(&mut rng, || simple_model(), 5000, 2000);
    let values: Vec<f64> = extract_f64_values(&samples, &addr!("param"));
    
    let r_hat = r_hat_f64(&[values.clone()]);
    assert!(r_hat < 1.1, "R-hat should indicate convergence");
    
    let ess = effective_sample_size_mcmc(&values);
    assert!(ess > 100.0, "Effective sample size should be reasonable");
}
```

## Integration Test Patterns

### Model Composition Testing

```rust
#[test]
fn test_hierarchical_model_composition() {
    let model = prob! {
        let global_mu <- sample(addr!("global_mu"), Normal::new(0.0, 1.0).unwrap());
        let local_effects <- plate!(i in 0..5 => {
            sample(addr!("local", i), Normal::new(global_mu, 0.1).unwrap())
        });
        pure((global_mu, local_effects))
    };
    
    let mut rng = StdRng::seed_from_u64(456);
    let handler = PriorHandler::new(&mut rng);
    let ((global_mu, locals), trace) = runtime::handler::run(handler, model);
    
    // Validate structure
    assert!(trace.get_f64(&addr!("global_mu")).is_some());
    for i in 0..5 {
        assert!(trace.get_f64(&addr!("local", i)).is_some());
    }
}
```

### Cross-Module Integration

```rust
#[test]
fn test_inference_runtime_integration() {
    // Test that inference algorithms work correctly with runtime interpreters
    let model = bayesian_linear_regression_model(data);
    
    // Test with different handlers
    test_with_prior_handler(&model);
    test_with_replay_handler(&model);
    test_with_scoring_handler(&model);
}
```

## Performance Testing

### Benchmark Integration

```rust
#[test]
fn test_performance_regression() {
    let start = std::time::Instant::now();
    
    let mut rng = StdRng::seed_from_u64(999);
    let _samples = adaptive_mcmc_chain(&mut rng, || large_model(), 1000, 500);
    
    let duration = start.elapsed();
    assert!(duration < std::time::Duration::from_secs(10), 
           "Performance regression detected");
}
```

### Memory Usage Testing

```rust
#[test]
fn test_memory_efficiency() {
    let initial_memory = get_memory_usage();
    
    {
        let mut rng = StdRng::seed_from_u64(111);
        let _samples = large_batch_inference(&mut rng);
        // Memory usage should be bounded during computation
    }
    
    // Memory should be released after scope
    std::thread::sleep(std::time::Duration::from_millis(100));
    let final_memory = get_memory_usage();
    assert!(final_memory <= initial_memory * 1.1, "Memory leak detected");
}
```

## Common Testing Utilities

### Helper Functions

```rust
fn extract_parameter_samples<T>(samples: &[(T, Trace)], addr: &Address) -> Vec<f64> {
    samples.iter()
        .filter_map(|(_, trace)| trace.get_f64(addr))
        .collect()
}

fn assert_convergence(samples: &[f64], target_mean: f64, tolerance: f64) {
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    assert!((mean - target_mean).abs() < tolerance,
           "Sample mean {} not within {} of target {}", mean, tolerance, target_mean);
}
```

### Test Data Generation

```rust
fn generate_test_data(n: usize, true_params: &[f64]) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(42);
    // Generate synthetic data with known ground truth
    (0..n).map(|i| {
        let x = i as f64 / n as f64;
        true_params[0] + true_params[1] * x + Normal::new(0.0, 0.1).unwrap().sample(&mut rng)
    }).collect()
}
```

## Running Tests

### Local Testing

```bash
# Run all integration tests
cargo test --test '*'

# Run specific test file
cargo test --test end_to_end_workflows

# Run with output visible
cargo test --test model_execution -- --nocapture

# Run single test
cargo test --test public_api_validation test_error_handling
```

### Performance Testing

```bash
# Run with release optimizations
cargo test --release --test end_to_end_workflows

# Memory profiling
valgrind --tool=massif cargo test --test memory_tests
```

## Troubleshooting

### Flaky Tests

- Use fixed random seeds for reproducibility
- Increase sample sizes for statistical tests
- Add appropriate tolerances for floating-point comparisons
- Consider statistical significance of test assertions

### Performance Issues

- Profile tests to identify bottlenecks
- Use release builds for performance-sensitive tests
- Consider test parallelization impact
- Monitor memory usage patterns

### Integration Failures

- Verify module interfaces haven't changed
- Check error handling paths
- Validate test data assumptions
- Ensure proper cleanup between tests
