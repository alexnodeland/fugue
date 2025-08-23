//! Benchmarks for MCMC algorithm optimizations.
//!
//! These benchmarks validate the performance improvements from:
//! - Optimized DiminishingAdaptation with cached log scales
//! - MCMC convergence diagnostics performance
//! - Effective sample size computation efficiency

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use fugue::addr;
use fugue::inference::mcmc_utils::{
    effective_sample_size_mcmc, geweke_diagnostic, DiminishingAdaptation,
};
use std::hint::black_box as std_black_box;

/// Benchmark DiminishingAdaptation performance with the cached log scale optimization.
fn bench_diminishing_adaptation(c: &mut Criterion) {
    let mut group = c.benchmark_group("diminishing_adaptation");

    // Test different numbers of adaptation updates
    for &num_updates in &[1000, 5000, 10000, 50000] {
        group.throughput(Throughput::Elements(num_updates as u64));

        // Benchmark single-site adaptation (most common case)
        group.bench_with_input(
            BenchmarkId::new("single_site", num_updates),
            &num_updates,
            |b, &num_updates| {
                b.iter_batched(
                    || DiminishingAdaptation::new(0.44, 0.7),
                    |mut adapter| {
                        let addr = addr!("param");
                        for i in 0..num_updates {
                            // Realistic acceptance pattern (~50% acceptance)
                            let accept = (i * 7) % 13 < 6; // Pseudo-random but deterministic
                            adapter.update(black_box(&addr), black_box(accept));
                        }
                        std_black_box(adapter)
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );

        // Benchmark multi-site adaptation (realistic MCMC scenario)
        group.bench_with_input(
            BenchmarkId::new("multi_site", num_updates),
            &num_updates,
            |b, &num_updates| {
                b.iter_batched(
                    || DiminishingAdaptation::new(0.234, 0.7), // Optimal scaling target
                    |mut adapter| {
                        let addresses = vec![
                            addr!("param1"),
                            addr!("param2"),
                            addr!("param3"),
                            addr!("param4"),
                            addr!("param5"),
                        ];

                        for i in 0..num_updates {
                            for (j, addr) in addresses.iter().enumerate() {
                                // Different acceptance patterns per parameter
                                let accept = ((i + j) * 11) % 17 < 8;
                                adapter.update(black_box(addr), black_box(accept));
                            }
                        }
                        std_black_box(adapter)
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );

        // Benchmark scale retrieval (common operation)
        group.bench_with_input(
            BenchmarkId::new("scale_retrieval", num_updates),
            &num_updates,
            |b, &num_updates| {
                b.iter_batched(
                    || {
                        let mut adapter = DiminishingAdaptation::new(0.44, 0.7);
                        let addr = addr!("test_param");

                        // Pre-populate with some updates
                        for i in 0..100 {
                            adapter.update(&addr, i % 3 != 0);
                        }

                        (adapter, addr)
                    },
                    |(mut adapter, addr)| {
                        let mut sum = 0.0;
                        for _ in 0..num_updates {
                            let scale = adapter.get_scale(black_box(&addr));
                            sum += black_box(scale);
                        }
                        std_black_box((adapter, sum))
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

/// Benchmark MCMC diagnostics performance.
fn bench_mcmc_diagnostics(c: &mut Criterion) {
    let mut group = c.benchmark_group("mcmc_diagnostics");

    // Test different chain lengths
    for &chain_length in &[100, 500, 1000, 5000] {
        group.throughput(Throughput::Elements(chain_length as u64));

        // Benchmark effective sample size computation
        group.bench_with_input(
            BenchmarkId::new("effective_sample_size", chain_length),
            &chain_length,
            |b, &chain_length| {
                b.iter_batched(
                    || {
                        // Generate realistic MCMC chain with some autocorrelation
                        let mut chain = Vec::with_capacity(chain_length);
                        let mut current = 0.0;
                        for i in 0..chain_length {
                            // AR(1) process with correlation
                            current = 0.8 * current + 0.6 * ((i as f64 * 0.1).sin());
                            chain.push(current);
                        }
                        chain
                    },
                    |chain| {
                        let ess = effective_sample_size_mcmc(black_box(&chain));
                        std_black_box(ess)
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );

        // Benchmark Geweke diagnostic
        group.bench_with_input(
            BenchmarkId::new("geweke_diagnostic", chain_length),
            &chain_length,
            |b, &chain_length| {
                b.iter_batched(
                    || {
                        // Generate chain with trend (non-stationary)
                        let mut chain = Vec::with_capacity(chain_length);
                        for i in 0..chain_length {
                            let trend = (i as f64) / (chain_length as f64) * 2.0; // Linear trend
                            let noise = ((i as f64 * 0.7).sin() + (i as f64 * 1.3).cos()) * 0.5;
                            chain.push(trend + noise);
                        }
                        chain
                    },
                    |chain| {
                        let z_score = geweke_diagnostic(black_box(&chain));
                        std_black_box(z_score)
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

/// Benchmark adaptation statistics collection.
fn bench_adaptation_stats(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaptation_stats");

    // Test different numbers of parameters being adapted
    for &num_params in &[1, 5, 10, 50, 100] {
        group.throughput(Throughput::Elements(num_params as u64));

        group.bench_with_input(
            BenchmarkId::new("get_stats", num_params),
            &num_params,
            |b, &num_params| {
                b.iter_batched(
                    || {
                        let mut adapter = DiminishingAdaptation::new(0.44, 0.7);

                        // Populate with multiple parameters
                        for i in 0..num_params {
                            let addr = addr!("param", i);

                            // Different adaptation histories for each parameter
                            for j in 0..100 {
                                let accept = ((i + j) * 13) % 19 < 10;
                                adapter.update(&addr, accept);
                            }
                        }

                        adapter
                    },
                    |adapter| {
                        let stats = adapter.get_stats();
                        std_black_box(stats)
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

/// Benchmark realistic MCMC adaptation scenario.
fn bench_realistic_mcmc_adaptation(c: &mut Criterion) {
    let mut group = c.benchmark_group("realistic_mcmc");

    // Simulate a realistic MCMC run with adaptation
    group.bench_function("full_adaptation_cycle", |b| {
        b.iter_batched(
            || DiminishingAdaptation::new(0.234, 0.7), // Optimal scaling
            |mut adapter| {
                // Simulate 10-parameter model with 1000 adaptation steps
                let addresses: Vec<_> = (0..10).map(|i| addr!("theta", i)).collect();

                for step in 0..1000 {
                    for (param_idx, addr) in addresses.iter().enumerate() {
                        // Realistic acceptance patterns with parameter-specific rates
                        let base_rate = 0.2 + 0.3 * (param_idx as f64 / 10.0); // 20-50% acceptance
                        let accept = ((step + param_idx) * 17) % 100 < (base_rate * 100.0) as usize;

                        adapter.update(black_box(addr), black_box(accept));

                        // Periodically check scale (realistic usage)
                        if step % 50 == 0 {
                            let _scale = adapter.get_scale(black_box(addr));
                        }
                    }

                    // Check adaptation status periodically
                    if step % 100 == 0 {
                        let _should_continue = adapter.should_continue_adaptation(500);
                    }
                }

                // Final statistics collection
                let _stats = adapter.get_stats();
                std_black_box(adapter)
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_diminishing_adaptation,
    bench_mcmc_diagnostics,
    bench_adaptation_stats,
    bench_realistic_mcmc_adaptation
);
criterion_main!(benches);
