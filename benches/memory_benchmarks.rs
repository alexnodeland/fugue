//! Benchmarks for memory management optimizations.
//!
//! These benchmarks validate the performance improvements from:
//! - Optimized address handling in TraceBuilder
//! - Enhanced TracePool with statistics and capacity management
//! - Copy-on-write trace operations
//! - End-to-end memory efficiency in MCMC

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use fugue::runtime::memory::{TraceBuilder, TracePool, CowTrace};
use fugue::runtime::trace::{Choice, ChoiceValue, Trace};
use fugue::*;
use rand::{rngs::StdRng, SeedableRng, Rng};
use std::collections::BTreeMap;
use std::hint::black_box as std_black_box;

/// Benchmark TraceBuilder performance with different address patterns.
fn bench_trace_builder(c: &mut Criterion) {
    let mut group = c.benchmark_group("trace_builder");
    
    // Test different numbers of choices
    for &size in &[10, 100, 1000, 5000] {
        group.throughput(Throughput::Elements(size as u64));
        
        // Benchmark sequential address pattern
        group.bench_with_input(
            BenchmarkId::new("sequential_addresses", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let mut builder = TraceBuilder::new();
                    for i in 0..size {
                        let addr = addr!("x", i);
                        builder.add_sample(black_box(addr), black_box(i as f64), black_box(-0.5));
                    }
                    std_black_box(builder.build())
                });
            }
        );

        // Benchmark hierarchical address pattern (more realistic)
        group.bench_with_input(
            BenchmarkId::new("hierarchical_addresses", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let mut builder = TraceBuilder::new();
                    for i in 0..size {
                        let layer = i / 10;
                        let idx = i % 10;
                        let addr = Address(format!("layer#{}/param#{}", layer, idx));
                        builder.add_sample(black_box(addr), black_box(i as f64), black_box(-1.2));
                    }
                    std_black_box(builder.build())
                });
            }
        );

        // Benchmark mixed value types
        group.bench_with_input(
            BenchmarkId::new("mixed_types", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let mut builder = TraceBuilder::new();
                    for i in 0..size {
                        let addr = addr!("mixed", i);
                        match i % 4 {
                            0 => builder.add_sample(black_box(addr), black_box(i as f64), black_box(-0.5)),
                            1 => builder.add_sample_bool(black_box(addr), black_box(i % 2 == 0), black_box(-0.693)),
                            2 => builder.add_sample_u64(black_box(addr), black_box(i as u64), black_box(-1.5)),
                            3 => builder.add_sample_usize(black_box(addr), black_box(i % 3), black_box(-1.1)),
                            _ => unreachable!(),
                        }
                    }
                    std_black_box(builder.build())
                });
            }
        );
    }
    group.finish();
}

/// Benchmark TracePool efficiency under different usage patterns.
fn bench_trace_pool(c: &mut Criterion) {
    let mut group = c.benchmark_group("trace_pool");
    
    // Test different pool sizes
    for &pool_size in &[10, 50, 100, 500] {
        // Benchmark pool hit rate with perfect reuse pattern
        group.bench_with_input(
            BenchmarkId::new("perfect_reuse", pool_size),
            &pool_size,
            |b, &pool_size| {
                b.iter_batched(
                    || TracePool::new(pool_size),
                    |mut pool| {
                        // Fill pool
                        let mut traces = Vec::new();
                        for _ in 0..pool_size.min(20) {
                            let mut trace = pool.get();
                            // Simulate some usage
                            for i in 0..10 {
                                trace.insert_choice(addr!("x", i), ChoiceValue::F64(i as f64), -0.5);
                            }
                            traces.push(trace);
                        }
                        
                        // Return traces to pool
                        for trace in traces {
                            pool.return_trace(black_box(trace));
                        }
                        
                        // Now reuse traces (should all be hits)
                        for _ in 0..pool_size.min(20) {
                            let trace = pool.get();
                            std_black_box(trace);
                        }
                        
                        std_black_box(pool)
                    },
                    criterion::BatchSize::SmallInput
                );
            }
        );

        // Benchmark pool with overflow (realistic pattern)
        group.bench_with_input(
            BenchmarkId::new("with_overflow", pool_size),
            &pool_size,
            |b, &pool_size| {
                b.iter_batched(
                    || TracePool::new(pool_size),
                    |mut pool| {
                        // Generate more traces than pool can hold
                        let num_traces = pool_size * 2;
                        let mut active_traces = Vec::new();
                        
                        for i in 0..num_traces {
                            let mut trace = pool.get();
                            // Simulate trace usage
                            for j in 0..5 {
                                trace.insert_choice(Address(format!("iter#{}/param#{}", i, j)), ChoiceValue::F64(j as f64), -0.5);
                            }
                            active_traces.push(trace);
                            
                            // Periodically return some traces
                            if i % 3 == 0 && !active_traces.is_empty() {
                                let trace = active_traces.remove(0);
                                pool.return_trace(black_box(trace));
                            }
                        }
                        
                        // Return remaining traces
                        for trace in active_traces {
                            pool.return_trace(trace);
                        }
                        
                        std_black_box(pool)
                    },
                    criterion::BatchSize::SmallInput
                );
            }
        );
    }
    group.finish();
}

/// Benchmark CowTrace copy-on-write performance.
fn bench_cow_trace(c: &mut Criterion) {
    let mut group = c.benchmark_group("cow_trace");
    
    // Test different trace sizes
    for &size in &[10, 100, 500, 1000] {
        group.throughput(Throughput::Elements(size as u64));
        
        // Setup base trace
        let mut base_trace = Trace::default();
        for i in 0..size {
            base_trace.insert_choice(addr!("x", i), ChoiceValue::F64(i as f64), -0.5);
            base_trace.log_prior += -0.5;
        }
        let base_cow = CowTrace::from_trace(base_trace);
        
        // Benchmark cloning (should be cheap)
        group.bench_with_input(
            BenchmarkId::new("clone", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let cloned = black_box(base_cow.clone());
                    std_black_box(cloned)
                });
            }
        );
        
        // Benchmark first write (triggers copy)
        group.bench_with_input(
            BenchmarkId::new("first_write", size),
            &size,
            |b, _| {
                b.iter_batched(
                    || base_cow.clone(),
                    |mut cow| {
                        cow.insert_choice(
                            black_box(addr!("new_choice")), 
                            black_box(Choice {
                                addr: addr!("new_choice"),
                                value: ChoiceValue::F64(42.0),
                                logp: -1.0,
                            })
                        );
                        std_black_box(cow)
                    },
                    criterion::BatchSize::SmallInput
                );
            }
        );
        
        // Benchmark subsequent writes (no more copying)
        group.bench_with_input(
            BenchmarkId::new("subsequent_writes", size),
            &size,
            |b, _| {
                b.iter_batched(
                    || {
                        let mut cow = base_cow.clone();
                        // Trigger initial copy
                        cow.insert_choice(addr!("trigger"), Choice {
                            addr: addr!("trigger"),
                            value: ChoiceValue::F64(0.0),
                            logp: 0.0,
                        });
                        cow
                    },
                    |mut cow| {
                        for i in 0..10 {
                            cow.insert_choice(
                                black_box(addr!("write", i)), 
                                black_box(Choice {
                                    addr: addr!("write", i),
                                    value: ChoiceValue::F64(i as f64),
                                    logp: -0.5,
                                })
                            );
                        }
                        std_black_box(cow)
                    },
                    criterion::BatchSize::SmallInput
                );
            }
        );
    }
    group.finish();
}

/// Benchmark end-to-end MCMC memory patterns.
fn bench_mcmc_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("mcmc_memory");
    
    // Simple Gaussian model for testing
    fn gaussian_model() -> Model<f64> {
        sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap())
            .bind(|mu| observe(addr!("y"), Normal::new(mu, 0.5).unwrap(), 1.5).map(move |_| mu))
    }
    
    // Benchmark trace generation patterns
    for &num_samples in &[100, 500, 1000] {
        group.throughput(Throughput::Elements(num_samples as u64));
        
        // Standard trace generation (no pooling)
        group.bench_with_input(
            BenchmarkId::new("standard_traces", num_samples),
            &num_samples,
            |b, &num_samples| {
                b.iter_batched(
                    || StdRng::seed_from_u64(42),
                    |mut rng| {
                        let mut traces = Vec::new();
                        for _ in 0..num_samples {
                            let (_, trace) = runtime::handler::run(
                                PriorHandler { rng: &mut rng, trace: Trace::default() },
                                gaussian_model(),
                            );
                            traces.push(black_box(trace));
                        }
                        std_black_box(traces)
                    },
                    criterion::BatchSize::SmallInput
                );
            }
        );
        
        // Pooled trace generation
        group.bench_with_input(
            BenchmarkId::new("pooled_traces", num_samples),
            &num_samples,
            |b, &num_samples| {
                b.iter_batched(
                    || (StdRng::seed_from_u64(42), TracePool::new(50)),
                    |(mut rng, mut pool)| {
                        let mut traces = Vec::new();
                        for _ in 0..num_samples {
                            let base_trace = pool.get();
                            let (_, trace) = runtime::handler::run(
                                PriorHandler { rng: &mut rng, trace: base_trace },
                                gaussian_model(),
                            );
                            traces.push(trace.clone());
                            pool.return_trace(black_box(trace));
                        }
                        std_black_box((traces, pool))
                    },
                    criterion::BatchSize::SmallInput
                );
            }
        );
        
        // CoW trace simulation (MCMC-like pattern)
        group.bench_with_input(
            BenchmarkId::new("cow_mcmc_pattern", num_samples),
            &num_samples,
            |b, &num_samples| {
                b.iter_batched(
                    || {
                        let mut rng = StdRng::seed_from_u64(42);
                        let (_, initial_trace) = runtime::handler::run(
                            PriorHandler { rng: &mut rng, trace: Trace::default() },
                            gaussian_model(),
                        );
                        (rng, CowTrace::from_trace(initial_trace))
                    },
                    |(mut rng, mut current_cow)| {
                        let mut chain = Vec::new();
                        for _ in 0..num_samples {
                            // Simulate MCMC step: small modification to current state
                            let mut proposal = current_cow.clone();
                            
                            // Modify one choice (simulating MH proposal)
                            let new_mu = current_cow.choices().get(&addr!("mu"))
                                .and_then(|c| c.value.as_f64())
                                .unwrap_or(0.0) + rng.gen::<f64>() * 0.1 - 0.05;
                            
                            proposal.insert_choice(addr!("mu"), Choice {
                                addr: addr!("mu"),
                                value: ChoiceValue::F64(new_mu),
                                logp: Normal::new(0.0, 1.0).unwrap().log_prob(&new_mu),
                            });
                            
                            // Accept/reject (simplified)
                            if rng.gen::<f64>() > 0.5 {
                                current_cow = proposal;
                            }
                            
                            chain.push(black_box(current_cow.total_log_weight()));
                        }
                        std_black_box((chain, current_cow))
                    },
                    criterion::BatchSize::SmallInput
                );
            }
        );
    }
    group.finish();
}

/// Benchmark memory allocation patterns.
fn bench_address_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("address_patterns");
    
    for &depth in &[2, 5, 10] {
        for &width in &[10, 50, 100] {
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("depth_{}_width_{}", depth, width)),
                &(depth, width),
                |b, &(depth, width)| {
                    b.iter(|| {
                        let mut choices = BTreeMap::new();
                        
                        // Generate nested hierarchical addresses
                        for d in 0..depth {
                            for w in 0..width {
                                let addr = Address(format!("root#{}/param#{}", d, w));
                                
                                choices.insert(
                                    black_box(addr.clone()),
                                    black_box(Choice {
                                        addr,
                                        value: ChoiceValue::F64((d * width + w) as f64),
                                        logp: -0.5,
                                    })
                                );
                            }
                        }
                        
                        std_black_box(choices)
                    });
                }
            );
        }
    }
    group.finish();
}

/// Benchmark to compare memory pool statistics tracking overhead.
fn bench_pool_stats(c: &mut Criterion) {
    let mut group = c.benchmark_group("pool_stats");
    
    // Compare pool with and without statistics
    for &operations in &[100, 500, 1000] {
        group.bench_with_input(
            BenchmarkId::new("with_stats", operations),
            &operations,
            |b, &operations| {
                b.iter_batched(
                    || TracePool::new(50),
                    |mut pool| {
                        for _ in 0..operations {
                            let trace = pool.get();
                            pool.return_trace(black_box(trace));
                        }
                        let stats = pool.stats().clone();
                        std_black_box((pool, stats))
                    },
                    criterion::BatchSize::SmallInput
                );
            }
        );
        
        // Simple benchmark without stats tracking (for comparison)
        group.bench_with_input(
            BenchmarkId::new("simple_pool", operations),
            &operations,
            |b, &operations| {
                b.iter_batched(
                    || Vec::<Trace>::with_capacity(50),
                    |mut simple_pool| {
                        for _ in 0..operations {
                            let trace = simple_pool.pop().unwrap_or_else(|| Trace::default());
                            if simple_pool.len() < 50 {
                                simple_pool.push(black_box(trace));
                            }
                        }
                        std_black_box(simple_pool)
                    },
                    criterion::BatchSize::SmallInput
                );
            }
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_trace_builder,
    bench_trace_pool,
    bench_cow_trace,
    bench_mcmc_memory,
    bench_address_patterns,
    bench_pool_stats
);
criterion_main!(benches);
