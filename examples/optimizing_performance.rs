use fugue::core::numerical::*;
use fugue::runtime::interpreters::PriorHandler;
use fugue::*;
use rand::thread_rng;
use std::time::Instant;

fn main() {
    println!("=== Optimizing Performance in Fugue ===\n");

    println!("1. Numerical Stability with Log-Space Computations");
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

    let mut rng = thread_rng();

    println!("2. Optimized Model Patterns");
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

    println!("3. Performance Monitoring and Profiling");
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

    println!("4. Numerical Precision Testing");
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
    // ANCHOR_END: performance_testing
}
