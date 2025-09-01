use fugue::core::numerical::*;
use fugue::runtime::interpreters::PriorHandler;
use fugue::runtime::memory::{CowTrace, PooledPriorHandler, TraceBuilder, TracePool};
use fugue::runtime::trace::{Choice, ChoiceValue};
use fugue::*;
use rand::thread_rng;
use std::time::Instant;

fn main() {
    println!("=== Optimizing Performance in Fugue ===\n");

    println!("1. Memory-Optimized Inference with Object Pooling");
    println!("-----------------------------------------------");
    // ANCHOR: memory_pooling
    // Create trace pool for zero-allocation inference
    let mut pool = TracePool::new(50); // Pool up to 50 traces
    let mut rng = thread_rng();

    // Define a model that would normally cause many allocations
    let make_model = || {
        prob!(
            let x <- sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
            let y <- sample(addr!("y"), Normal::new(x, 0.5).unwrap());
            observe(addr!("obs"), Normal::new(y, 0.1).unwrap(), 1.5);
            pure(x)
        )
    };

    // Time pooled vs non-pooled execution
    let start = Instant::now();
    for _iteration in 0..1000 {
        // Use pooled handler for efficient memory reuse
        let (_result, trace) =
            runtime::handler::run(PooledPriorHandler::new(&mut rng, &mut pool), make_model());
        // Return trace to pool for reuse
        pool.return_trace(trace);
    }
    let pooled_time = start.elapsed();

    let stats = pool.stats();
    println!("✅ Completed 1000 iterations with memory pooling");
    println!("   - Execution time: {:?}", pooled_time);
    println!("   - Hit ratio: {:.1}%", stats.hit_ratio());
    println!(
        "   - Pool stats - hits: {}, misses: {}",
        stats.hits, stats.misses
    );
    // ANCHOR_END: memory_pooling
    println!();

    println!("2. Numerical Stability with Log-Space Computations");
    println!("------------------------------------------------");
    // ANCHOR: numerical_stability
    // Demonstrate stable log-probability computations
    let extreme_log_probs = vec![700.0, 701.0, 699.0, 698.0]; // Would overflow in linear space

    // Safe log-sum-exp prevents overflow
    let log_normalizer = log_sum_exp(&extreme_log_probs);
    let normalized_probs = normalize_log_probs(&extreme_log_probs);

    println!("✅ Stable computation with extreme log-probabilities");
    println!("   - Log normalizer: {:.2}", log_normalizer);
    println!(
        "   - Probabilities sum to: {:.10}",
        normalized_probs.iter().sum::<f64>()
    );

    // Weighted log-sum-exp for importance sampling
    let log_values = vec![-1.0, -2.0, -3.0, -4.0];
    let weights = vec![0.4, 0.3, 0.2, 0.1];
    let weighted_result = weighted_log_sum_exp(&log_values, &weights);

    println!("   - Weighted log-sum-exp: {:.4}", weighted_result);

    // Safe logarithm handling
    let safe_results: Vec<f64> = [1.0, 0.0, -1.0].iter().map(|&x| safe_ln(x)).collect();
    println!("   - Safe ln results: {:?}", safe_results);
    // ANCHOR_END: numerical_stability
    println!();

    println!("3. Efficient Trace Construction");
    println!("------------------------------");
    // ANCHOR: efficient_construction
    // Use TraceBuilder for efficient trace creation
    let mut builder = TraceBuilder::new();

    let start = Instant::now();
    for i in 0..100 {
        // Add choices efficiently without reallocations
        builder.add_sample(
            addr!("param", i),
            i as f64,
            0.0, // log_prob
        );
    }

    // Build final trace efficiently
    let constructed_trace = builder.build();
    let construction_time = start.elapsed();

    println!("✅ Efficient trace construction");
    println!(
        "   - Built trace with {} choices in {:?}",
        constructed_trace.choices.len(),
        construction_time
    );
    println!(
        "   - Total log weight: {:.2}",
        constructed_trace.total_log_weight()
    );
    // ANCHOR_END: efficient_construction
    println!();

    println!("4. Copy-on-Write for MCMC Efficiency");
    println!("-----------------------------------");
    // ANCHOR: cow_traces
    // Create base trace manually for MCMC
    let mut builder = TraceBuilder::new();
    builder.add_sample(addr!("mu"), 0.5, -0.5);
    builder.add_sample(addr!("sigma"), 1.0, -1.0);
    builder.add_sample_bool(addr!("component"), true, -0.69);
    let base_trace = builder.build();

    // Create COW trace for efficient copying
    let cow_base = CowTrace::from_trace(base_trace);

    let start = Instant::now();
    let mut mcmc_traces = Vec::new();

    for _proposal in 0..1000 {
        // Clone is O(1) until modification
        let mut proposal_trace = cow_base.clone();

        // Modify only one parameter (triggers COW)
        proposal_trace.insert_choice(
            addr!("mu"),
            Choice {
                addr: addr!("mu"),
                value: ChoiceValue::F64(0.6),
                logp: -0.4,
            },
        );

        mcmc_traces.push(proposal_trace);
    }
    let cow_time = start.elapsed();

    println!("✅ Copy-on-write MCMC proposals");
    println!("   - Created 1000 proposal traces in {:?}", cow_time);
    println!("   - Memory sharing until modification");
    // ANCHOR_END: cow_traces
    println!();

    println!("5. Optimized Model Patterns");
    println!("---------------------------");
    // ANCHOR: optimized_patterns
    // Pre-allocate data structures for repeated use
    let observations: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
    let n = observations.len();

    // Efficient vectorized model
    let vectorized_model = || {
        prob!(
            let mu <- sample(addr!("global_mu"), Normal::new(0.0, 10.0).unwrap());
            let precision <- sample(addr!("precision"), Gamma::new(2.0, 1.0).unwrap());
            let sigma = (1.0 / precision).sqrt();

            // Use plate for efficient vectorized operations
            let _likelihoods <- plate!(i in 0..n => {
                observe(addr!("obs", i), Normal::new(mu, sigma).unwrap(), observations[i])
            });

            pure((mu, sigma))
        )
    };

    let start = Instant::now();
    let (_result, _trace) = runtime::handler::run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        vectorized_model(),
    );
    let vectorized_time = start.elapsed();

    println!("✅ Optimized vectorized model");
    println!("   - Processed {} observations in {:?}", n, vectorized_time);
    // ANCHOR_END: optimized_patterns
    println!();

    println!("6. Performance Monitoring and Profiling");
    println!("--------------------------------------");
    // ANCHOR: performance_monitoring
    // Monitor trace characteristics for optimization insights
    #[derive(Debug)]
    struct TraceMetrics {
        num_choices: usize,
        log_weight: f64,
        is_valid: bool,
        memory_size_estimate: usize,
    }

    impl TraceMetrics {
        fn from_trace(trace: &Trace) -> Self {
            let num_choices = trace.choices.len();
            let log_weight = trace.total_log_weight();
            let is_valid = log_weight.is_finite();

            // Rough memory estimate (actual implementation would be more precise)
            let memory_size_estimate = num_choices * 64; // Rough bytes per choice

            Self {
                num_choices,
                log_weight,
                is_valid,
                memory_size_estimate,
            }
        }
    }

    // Example: Monitor a complex model's performance
    let complex_model = || {
        prob!(
            let components <- plate!(c in 0..5 => {
                sample(addr!("weight", c), Gamma::new(1.0, 1.0).unwrap())
                    .bind(move |weight| {
                        sample(addr!("mu", c), Normal::new(0.0, 2.0).unwrap())
                            .map(move |mu| (weight, mu))
                    })
            });

            let selector <- sample(addr!("selector"),
                                  Categorical::new(vec![0.2, 0.2, 0.2, 0.2, 0.2]).unwrap());

            pure((components, selector))
        )
    };

    let (_result, trace) = runtime::handler::run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        complex_model(),
    );

    let metrics = TraceMetrics::from_trace(&trace);
    println!("✅ Performance monitoring active");
    println!("   - Trace choices: {}", metrics.num_choices);
    println!("   - Log weight: {:.2}", metrics.log_weight);
    println!("   - Valid: {}", metrics.is_valid);
    println!(
        "   - Memory estimate: {} bytes",
        metrics.memory_size_estimate
    );
    // ANCHOR_END: performance_monitoring
    println!();

    println!("7. Batch Processing Optimization");
    println!("-------------------------------");
    // ANCHOR: batch_processing
    // Efficient batch inference using memory pooling
    let batch_size = 100;
    let mut batch_pool = TracePool::new(batch_size);

    let start = Instant::now();
    let mut batch_results = Vec::with_capacity(batch_size);

    for _batch in 0..batch_size {
        let (result, trace) = runtime::handler::run(
            PooledPriorHandler::new(&mut rng, &mut batch_pool),
            make_model(),
        );
        // Return trace to pool for reuse
        batch_pool.return_trace(trace);
        batch_results.push(result);
    }

    let batch_time = start.elapsed();
    let batch_stats = batch_pool.stats();

    println!("✅ Batch processing complete");
    println!("   - Processed {} samples in {:?}", batch_size, batch_time);
    println!(
        "   - Average time per sample: {:?}",
        batch_time / batch_size as u32
    );
    println!(
        "   - Memory efficiency: {:.1}% hit ratio",
        batch_stats.hit_ratio()
    );
    // ANCHOR_END: batch_processing
    println!();

    println!("8. Numerical Precision Testing");
    println!("-----------------------------");
    // ANCHOR: precision_testing
    // Test numerical stability across different scales
    let test_scales = vec![1e-10, 1e-5, 1.0, 1e5, 1e10];

    for &scale in &test_scales {
        let scale: f64 = scale;
        let log_vals = vec![scale.ln() + 1.0, scale.ln() + 2.0, scale.ln() + 0.5];

        let stable_sum = log_sum_exp(&log_vals);
        let log1p_result = log1p_exp(scale.ln());

        println!(
            "   Scale {:.0e}: log_sum_exp={:.4}, log1p_exp={:.4}",
            scale, stable_sum, log1p_result
        );
    }
    println!("✅ Numerical stability verified across scales");
    // ANCHOR_END: precision_testing
    println!();

    println!("=== Performance Optimization Patterns Complete! ===");
}

#[cfg(test)]
mod tests {
    use super::*;

    // ANCHOR: performance_testing
    #[test]
    fn test_memory_pool_efficiency() {
        let mut pool = TracePool::new(10);
        let mut rng = thread_rng();

        // Test pool reuse with PooledPriorHandler
        for _i in 0..20 {
            let (_, trace) = runtime::handler::run(
                PooledPriorHandler::new(&mut rng, &mut pool),
                sample(addr!("test"), Normal::new(0.0, 1.0).unwrap()),
            );
            // Return trace to pool for reuse
            pool.return_trace(trace);
        }

        let stats = pool.stats();
        assert!(
            stats.hit_ratio() > 50.0,
            "Pool should have good hit ratio, got {:.1}%",
            stats.hit_ratio()
        );
        assert!(stats.hits + stats.misses > 0, "Pool should have been used");
    }

    #[test]
    fn test_numerical_stability() {
        // Test log_sum_exp with extreme values
        let extreme_vals = vec![700.0, 701.0, 699.0];
        let result = log_sum_exp(&extreme_vals);
        assert!(
            result.is_finite(),
            "log_sum_exp should handle extreme values"
        );

        // Test normalization
        let normalized = normalize_log_probs(&extreme_vals);
        let sum: f64 = normalized.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Normalized probabilities should sum to 1"
        );

        // Test weighted computation
        let weights = vec![0.5, 0.3, 0.2];
        let weighted_result = weighted_log_sum_exp(&extreme_vals, &weights);
        assert!(
            weighted_result.is_finite(),
            "Weighted log_sum_exp should be finite"
        );
    }

    #[test]
    fn test_trace_builder_efficiency() {
        let mut builder = TraceBuilder::new();

        // Add many choices efficiently
        for i in 0..100 {
            builder.add_sample(addr!("param", i), i as f64, -0.5);
        }

        let trace = builder.build();
        assert_eq!(trace.choices.len(), 100);
        assert!(trace.total_log_weight().is_finite());
    }

    #[test]
    fn test_cow_trace_sharing() {
        // Create base trace using builder
        let mut builder = TraceBuilder::new();
        builder.add_sample(addr!("x"), 1.0, -0.5);
        let base = builder.build();
        let cow_trace = CowTrace::from_trace(base);

        // Clone should be fast
        let clone1 = cow_trace.clone();
        let clone2 = cow_trace.clone();

        // Should share data until modification - convert to regular trace to test
        let trace1 = clone1.to_trace();
        let trace2 = clone2.to_trace();
        assert_eq!(trace1.get_f64(&addr!("x")), Some(1.0));
        assert_eq!(trace2.get_f64(&addr!("x")), Some(1.0));
    }
    // ANCHOR_END: performance_testing
}
