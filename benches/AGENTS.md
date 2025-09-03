# Agent Context: Benchmarks Directory

## Purpose

The `benches/` directory contains performance benchmarks for the Fugue probabilistic programming library. These benchmarks validate performance characteristics, detect regressions, and guide optimization efforts for production deployments.

## Structure

```text
benches/
├── mcmc_benchmarks.rs    # MCMC algorithm performance
└── memory_benchmarks.rs  # Memory usage and allocation patterns
```

## Benchmark Categories

### Algorithm Performance (`mcmc_benchmarks.rs`)

**Purpose**: Measure and track performance of core inference algorithms

- Sampling throughput (samples/second)
- Convergence rates and efficiency
- Scaling behavior with model complexity
- Comparison between different MCMC variants

### Memory Efficiency (`memory_benchmarks.rs`)

**Purpose**: Monitor memory usage patterns and optimization effectiveness

- Trace allocation and pooling performance
- Memory usage scaling with problem size
- Copy-on-write trace efficiency
- Garbage collection impact measurement

## Development Guidelines

### Mandatory Testing Practice

**Always run benchmarks after performance-related changes.** Performance regressions in a probabilistic programming library can make real-world applications infeasible.

```bash
# Run all benchmarks after performance changes
make bench

# Run specific benchmark suite
cargo bench --bench mcmc_benchmarks

# Generate detailed benchmark reports
cargo bench -- --verbose

# Compare against baseline
cargo bench --bench mcmc_benchmarks -- --save-baseline main

# Check for performance regressions
make test && make bench

# Full validation pipeline
make all
```

### Benchmark Development Standards

#### Measurement Reliability

- **Statistical rigor**: Multiple runs with significance testing
- **Stable environment**: Consistent hardware and OS conditions  
- **Realistic workloads**: Benchmarks reflect actual usage patterns
- **Baseline tracking**: Compare against known performance characteristics

#### Code Quality

- **Reproducible**: Fixed random seeds and deterministic setup
- **Well-documented**: Clear purpose and interpretation guidance
- **Parameterized**: Test across relevant problem sizes and configurations
- **Isolated**: Minimize interference between benchmark runs

#### Performance Awareness

- **Meaningful metrics**: Focus on user-relevant performance indicators
- **Scaling analysis**: Understand complexity characteristics
- **Resource monitoring**: Track CPU, memory, and I/O usage
- **Regression detection**: Alert on significant performance degradation

## Benchmark Implementation Patterns

### Basic Benchmark Structure

```rust
//! # MCMC Performance Benchmarks
//! 
//! Measures inference algorithm performance across different model
//! complexities and problem sizes.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use fugue::prelude::*;
use rand::{rngs::StdRng, SeedableRng};

fn benchmark_mcmc_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("mcmc_throughput");
    
    // Test different model complexities
    for &num_params in &[1, 5, 10, 25, 50] {
        group.bench_with_input(
            BenchmarkId::new("adaptive_mcmc", num_params),
            &num_params,
            |b, &num_params| {
                b.iter(|| {
                    let mut rng = StdRng::seed_from_u64(42);
                    let model = create_test_model(num_params);
                    
                    // Benchmark the core sampling loop
                    let samples = black_box(
                        adaptive_mcmc_chain(&mut rng, || model.clone(), 1000, 500)
                    );
                    
                    // Ensure optimizer doesn't eliminate computation
                    black_box(samples)
                });
            },
        );
    }
    
    group.finish();
}

fn create_test_model(num_params: usize) -> Model<Vec<f64>> {
    prob! {
        let params <- sequence_vec((0..num_params).map(|i| {
            sample(addr!("param", i), Normal::new(0.0, 1.0).unwrap())
        }).collect());
        
        // Add some observations to make inference non-trivial
        let _obs <- observe(
            addr!("obs"), 
            Normal::new(params[0], 0.5).unwrap(), 
            1.0
        );
        
        pure(params)
    }
}

criterion_group!(mcmc_benches, benchmark_mcmc_throughput);
criterion_main!(mcmc_benches);
```

### Memory Benchmark Patterns

```rust
//! # Memory Usage Benchmarks
//! 
//! Measures memory allocation patterns and validates memory optimization
//! strategies for high-throughput scenarios.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use fugue::runtime::memory::{TracePool, PooledPriorHandler};

fn benchmark_trace_pooling(c: &mut Criterion) {
    c.bench_function("trace_pooling_vs_allocation", |b| {
        b.iter(|| {
            let mut rng = StdRng::seed_from_u64(42);
            let mut pool = TracePool::new(100);
            
            // Benchmark pooled allocation pattern
            for _ in 0..1000 {
                let handler = PooledPriorHandler::new(&mut rng, &mut pool);
                let model = simple_test_model();
                let result = runtime::handler::run(handler, model);
                black_box(result);
            }
        });
    });
}

fn benchmark_memory_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_scaling");
    
    for &problem_size in &[100, 500, 1000, 5000, 10000] {
        group.bench_with_input(
            BenchmarkId::new("large_model_inference", problem_size),
            &problem_size,
            |b, &problem_size| {
                b.iter(|| {
                    let model = create_large_model(problem_size);
                    let mut rng = StdRng::seed_from_u64(42);
                    
                    // Monitor memory usage during inference
                    let samples = black_box(
                        adaptive_mcmc_chain(&mut rng, || model.clone(), 500, 250)
                    );
                    
                    black_box(samples)
                });
            },
        );
    }
    
    group.finish();
}
```

### Comparative Benchmarks

```rust
//! # Algorithm Comparison Benchmarks
//! 
//! Compares performance characteristics between different inference
//! algorithms and implementation strategies.

fn benchmark_inference_algorithms(c: &mut Criterion) {
    let mut group = c.benchmark_group("inference_comparison");
    
    let test_model = create_comparison_model();
    
    // Benchmark different MCMC variants
    group.bench_function("metropolis_hastings", |b| {
        b.iter(|| {
            let mut rng = StdRng::seed_from_u64(42);
            black_box(metropolis_hastings_chain(&mut rng, || test_model.clone(), 1000))
        });
    });
    
    group.bench_function("adaptive_mcmc", |b| {
        b.iter(|| {
            let mut rng = StdRng::seed_from_u64(42);
            black_box(adaptive_mcmc_chain(&mut rng, || test_model.clone(), 1000, 500))
        });
    });
    
    group.bench_function("hamiltonian_mc", |b| {
        b.iter(|| {
            let mut rng = StdRng::seed_from_u64(42);
            black_box(hamiltonian_mc_chain(&mut rng, || test_model.clone(), 1000))
        });
    });
    
    group.finish();
}
```

## Benchmark Analysis and Interpretation

### Performance Metrics

#### Throughput Measurements

- **Samples per second**: Raw sampling performance
- **Effective samples per second**: Quality-adjusted throughput
- **Memory bandwidth**: Data movement efficiency
- **CPU utilization**: Processor efficiency

#### Scaling Characteristics

- **Linear scaling**: Performance proportional to problem size
- **Sublinear scaling**: Diminishing returns with size increase
- **Memory scaling**: RAM usage growth patterns
- **Parallel efficiency**: Multi-core utilization effectiveness

#### Quality Metrics

- **Convergence speed**: Time to reach target accuracy
- **Sample efficiency**: Information per computational cost
- **Numerical stability**: Precision maintenance under load
- **Error rates**: Frequency of computational failures

### Regression Detection

#### Automated Monitoring

```bash
# Establish performance baseline
cargo bench -- --save-baseline release-1.0

# Compare against baseline after changes
cargo bench -- --baseline release-1.0

# Generate regression report
cargo bench -- --baseline release-1.0 --output-format json > bench_results.json
```

#### Statistical Analysis

- **Significance testing**: Detect meaningful performance changes
- **Confidence intervals**: Quantify measurement uncertainty
- **Trend analysis**: Identify gradual performance degradation
- **Outlier detection**: Handle anomalous benchmark results

### Performance Optimization Workflow

#### Profiling Integration

```bash
# Profile benchmark execution
cargo flamegraph --bench mcmc_benchmarks

# Memory profiling
valgrind --tool=massif --massif-out-file=massif.out \
    cargo bench --bench memory_benchmarks

# Cache analysis
valgrind --tool=cachegrind cargo bench --bench mcmc_benchmarks
```

#### Optimization Validation

1. **Baseline measurement**: Establish current performance
2. **Hypothesis formation**: Identify optimization opportunities  
3. **Implementation**: Apply performance improvements
4. **Validation**: Verify improvements with benchmarks
5. **Regression testing**: Ensure correctness maintained

## Testing and Validation

### Benchmark Reliability

#### Environment Control

- **Consistent hardware**: Same machine for comparable results
- **Thermal stability**: Allow warmup periods, monitor CPU temperature
- **Background processes**: Minimize system interference
- **Power management**: Disable CPU frequency scaling for consistency

#### Statistical Rigor

```rust
// Configure benchmark for statistical reliability
fn configure_benchmark(c: &mut Criterion) {
    c.sample_size(100)           // Sufficient sample size
     .measurement_time(Duration::from_secs(10))  // Adequate measurement time
     .warm_up_time(Duration::from_secs(3))       // Eliminate startup effects
     .significance_level(0.05)   // Statistical significance threshold
     .noise_threshold(0.02);     // Sensitivity to performance changes
}
```

### Continuous Integration

#### Automated Benchmarking

```yaml
# CI benchmark pipeline
benchmark:
  runs-on: benchmark-runner
  steps:
    - uses: actions/checkout@v2
    - name: Run benchmarks
      run: |
        make bench
        cargo bench -- --output-format json > benchmark_results.json
    - name: Check for regressions
      run: |
        python scripts/check_performance_regression.py benchmark_results.json
```

#### Performance Tracking

- **Trend monitoring**: Track performance over time
- **Alert thresholds**: Notify on significant regressions
- **Historical analysis**: Understand long-term performance evolution
- **Release validation**: Ensure performance acceptance before release

## Common Benchmarking Pitfalls

### Measurement Issues

```rust
// BAD: Compiler optimization eliminates work
fn bad_benchmark(b: &mut Bencher) {
    b.iter(|| {
        let result = expensive_computation();
        // Compiler may optimize away unused result
    });
}

// GOOD: Use black_box to prevent optimization
fn good_benchmark(b: &mut Bencher) {
    b.iter(|| {
        let result = expensive_computation();
        black_box(result); // Ensures computation isn't eliminated
    });
}
```

### Statistical Invalidity

```rust
// BAD: Non-deterministic benchmark
fn unreliable_benchmark(b: &mut Bencher) {
    b.iter(|| {
        let mut rng = StdRng::from_entropy(); // Different seed each run
        monte_carlo_simulation(&mut rng)
    });
}

// GOOD: Reproducible benchmark
fn reliable_benchmark(b: &mut Bencher) {
    b.iter(|| {
        let mut rng = StdRng::seed_from_u64(42); // Fixed seed
        monte_carlo_simulation(&mut rng)
    });
}
```

### Unrealistic Workloads

```rust
// BAD: Trivial problem that doesn't reflect real usage
fn toy_benchmark(b: &mut Bencher) {
    b.iter(|| {
        let model = prob! { pure(1.0) }; // Too simple
        simple_sampling(&model)
    });
}

// GOOD: Realistic problem representative of actual use
fn realistic_benchmark(b: &mut Bencher) {
    b.iter(|| {
        let model = bayesian_linear_regression_model(real_data);
        mcmc_inference(&model)
    });
}
```

## Performance Analysis Tools

### Profiling Integration

```bash
# CPU profiling with flame graphs
cargo flamegraph --bench mcmc_benchmarks -- --bench

# Memory profiling
cargo valgrind --bench memory_benchmarks

# Performance counters
perf record cargo bench --bench mcmc_benchmarks
perf report
```

### Benchmark Reporting

```bash
# Generate detailed HTML reports
cargo bench -- --output-format html

# Compare multiple benchmark runs
cargo bench -- --baseline main --save-baseline feature-branch
```

### Custom Metrics

```rust
// Track custom performance metrics
fn benchmark_with_custom_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("custom_metrics");
    
    group.bench_function("inference_with_metrics", |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            
            for _ in 0..iters {
                // Benchmark code here
                let samples = run_inference();
                black_box(samples);
            }
            
            start.elapsed()
        });
    });
}
```

## Running Benchmarks

### Local Development

```bash
# Quick benchmark run
make bench

# Detailed benchmark with baseline
cargo bench -- --save-baseline dev

# Compare performance changes
cargo bench -- --baseline dev

# Profile specific benchmark
cargo flamegraph --bench mcmc_benchmarks -- inference_throughput
```

### Performance Validation

```bash
# Ensure no regressions before release
make all  # Includes benchmarks

# Comprehensive performance analysis
cargo bench --all-targets
cargo bench -- --verbose --output-format json > results.json

# Memory usage analysis
/usr/bin/time -v cargo bench --bench memory_benchmarks
```

This comprehensive benchmarking strategy ensures the Fugue library maintains high performance characteristics suitable for production probabilistic programming applications.
