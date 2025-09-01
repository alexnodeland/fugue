use fugue::inference::diagnostics::r_hat_f64;
use fugue::inference::mcmc_utils::effective_sample_size_mcmc;
use fugue::runtime::interpreters::{PriorHandler, SafeReplayHandler, SafeScoreGivenTrace};
use fugue::runtime::trace::{ChoiceValue, Trace};
use fugue::*;
// use fugue::inference::validation::*;
use rand::{thread_rng, SeedableRng};
use std::collections::BTreeMap;

fn main() {
    println!("=== Debugging Probabilistic Models in Fugue ===\n");

    println!("1. Basic Trace Inspection");
    println!("------------------------");
    // ANCHOR: trace_inspection
    // Execute a model and examine its trace structure
    let mut rng = thread_rng();

    let diagnostic_model = || {
        prob!(
            let mu <- sample(addr!("mu"), Normal::new(0.0, 2.0).unwrap());
            let sigma <- sample(addr!("sigma"), Gamma::new(2.0, 1.0).unwrap());
            observe(addr!("obs1"), Normal::new(mu, sigma).unwrap(), 1.5);
            observe(addr!("obs2"), Normal::new(mu, sigma).unwrap(), 1.2);
            factor(if mu.abs() < 3.0 { 0.0 } else { f64::NEG_INFINITY });
            pure((mu, sigma))
        )
    };

    let ((mu_val, sigma_val), trace) = runtime::handler::run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        diagnostic_model(),
    );

    println!("✅ Model execution complete");
    println!("   - Result: mu = {:.3}, sigma = {:.3}", mu_val, sigma_val);
    println!("   - Choices recorded: {}", trace.choices.len());
    println!("   - Prior log-weight: {:.6}", trace.log_prior);
    println!("   - Likelihood log-weight: {:.6}", trace.log_likelihood);
    println!("   - Factor log-weight: {:.6}", trace.log_factors);
    println!("   - Total log-weight: {:.6}", trace.total_log_weight());

    // Per-choice breakdown
    println!("   - Choice breakdown:");
    for (addr, choice) in &trace.choices {
        println!(
            "     {}: {:?} (logp: {:.6})",
            addr, choice.value, choice.logp
        );
    }
    // ANCHOR_END: trace_inspection
    println!();

    println!("2. Type-Safe Value Access and Error Handling");
    println!("-------------------------------------------");
    // ANCHOR: type_safe_access
    // Safe access patterns that handle type mismatches gracefully

    // Option-based access (returns None on mismatch)
    match trace.get_f64(&addr!("mu")) {
        Some(mu) => println!("✅ Retrieved mu = {:.3}", mu),
        None => println!("❌ Failed to get mu as f64"),
    }

    // Result-based access (returns detailed error info)
    match trace.get_f64_result(&addr!("sigma")) {
        Ok(sigma) => println!("✅ Retrieved sigma = {:.3}", sigma),
        Err(e) => println!("❌ Error getting sigma: {}", e),
    }

    // Handle missing addresses
    match trace.get_f64_result(&addr!("missing_param")) {
        Ok(_) => unreachable!(),
        Err(e) => println!("✅ Correctly caught missing address: {}", e),
    }

    // Handle type mismatches
    match trace.get_bool_result(&addr!("mu")) {
        Ok(_) => unreachable!(),
        Err(e) => println!("✅ Correctly caught type mismatch: {}", e),
    }

    // Iterate through all choices for debugging
    println!("   - All choices and their types:");
    for (addr, choice) in &trace.choices {
        let type_info = match &choice.value {
            ChoiceValue::F64(_) => "f64",
            ChoiceValue::Bool(_) => "bool",
            ChoiceValue::U64(_) => "u64",
            ChoiceValue::I64(_) => "i64",
            ChoiceValue::Usize(_) => "usize",
        };
        println!("     {} ({}): {:?}", addr, type_info, choice.value);
    }
    // ANCHOR_END: type_safe_access
    println!();

    println!("3. Model Validation and Testing");
    println!("------------------------------");
    // ANCHOR: model_validation
    // Test a simple conjugate model against analytical solution
    let conjugate_model = || {
        prob!(
            let theta <- sample(addr!("theta"), Beta::new(1.0, 1.0).unwrap());
            observe(addr!("successes"), Binomial::new(10, theta).unwrap(), 7u64);
            pure(theta)
        )
    };

    // Run a few samples to test basic functionality
    let mut theta_samples = Vec::new();
    for _ in 0..20 {
        let (theta, test_trace) = runtime::handler::run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            conjugate_model(),
        );

        // Validate trace structure
        assert!(test_trace.choices.contains_key(&addr!("theta")));
        assert!(
            test_trace.total_log_weight().is_finite(),
            "Trace should have finite log-weight"
        );
        assert!(
            test_trace.log_likelihood.is_finite(),
            "Likelihood should be finite"
        );

        theta_samples.push(theta);
    }

    // Basic statistical checks
    let sample_mean = theta_samples.iter().sum::<f64>() / theta_samples.len() as f64;
    println!("✅ Validation tests passed");
    println!("   - Generated {} samples", theta_samples.len());
    println!(
        "   - Sample mean: {:.3} (expected ~0.7 for Beta-Binomial)",
        sample_mean
    );
    println!("   - All traces had finite log-weights");
    // ANCHOR_END: model_validation
    println!();

    println!("4. Safe vs Strict Handlers for Error Resilience");
    println!("-----------------------------------------------");
    // ANCHOR: safe_handlers
    // Create a trace with known structure for replay testing
    let mut base_trace = Trace::default();
    base_trace.insert_choice(addr!("param"), ChoiceValue::F64(1.5), -0.5);

    let test_model = || sample(addr!("param"), Normal::new(0.0, 1.0).unwrap());

    // Strict replay - will panic on mismatch (commented out for safety)
    // let strict_replay = ReplayHandler { base_trace: &base_trace };
    // let (strict_result, strict_trace) = runtime::handler::run(strict_replay, test_model());

    // Safe replay - handles errors gracefully
    let safe_replay = SafeReplayHandler {
        rng: &mut rng,
        base: base_trace.clone(),
        trace: Trace::default(),
        warn_on_mismatch: true,
    };
    let (safe_result, safe_trace) = runtime::handler::run(safe_replay, test_model());

    println!("✅ Safe replay succeeded");
    println!("   - Result: {:.3}", safe_result);
    println!(
        "   - Retrieved value: {:?}",
        safe_trace.get_f64(&addr!("param"))
    );

    // Test scoring with safe handler
    let safe_score = SafeScoreGivenTrace {
        base: base_trace,
        trace: Trace::default(),
        warn_on_error: false,
    };
    let (_, score_trace) = runtime::handler::run(safe_score, test_model());

    println!(
        "   - Score trace log-weight: {:.3}",
        score_trace.total_log_weight()
    );
    // ANCHOR_END: safe_handlers
    println!();

    println!("5. MCMC Diagnostics and Convergence Checking");
    println!("--------------------------------------------");
    // ANCHOR: mcmc_diagnostics
    // Generate simple MCMC chains for diagnostic testing
    let mcmc_model = || {
        prob!(
            let mu <- sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap());
            observe(addr!("y"), Normal::new(mu, 0.5).unwrap(), 1.0);
            pure(mu)
        )
    };

    // Generate two short chains for R-hat calculation
    let n_samples = 50;
    let n_warmup = 10;

    let mut chain1_samples = Vec::new();
    let mut chain2_samples = Vec::new();

    // Chain 1
    let mut rng1 = rand::rngs::StdRng::seed_from_u64(42);
    let chain1 = adaptive_mcmc_chain(&mut rng1, mcmc_model, n_samples, n_warmup);
    for (_, trace) in &chain1 {
        if let Some(mu) = trace.get_f64(&addr!("mu")) {
            chain1_samples.push(mu);
        }
    }

    // Chain 2
    let mut rng2 = rand::rngs::StdRng::seed_from_u64(123);
    let chain2 = adaptive_mcmc_chain(&mut rng2, mcmc_model, n_samples, n_warmup);
    for (_, trace) in &chain2 {
        if let Some(mu) = trace.get_f64(&addr!("mu")) {
            chain2_samples.push(mu);
        }
    }

    // Compute diagnostics
    if !chain1_samples.is_empty() && !chain2_samples.is_empty() {
        // Extract traces for R-hat calculation
        let chain1_traces: Vec<Trace> = chain1.into_iter().map(|(_, trace)| trace).collect();
        let chain2_traces: Vec<Trace> = chain2.into_iter().map(|(_, trace)| trace).collect();
        let r_hat = r_hat_f64(&[chain1_traces, chain2_traces], &addr!("mu"));
        let ess1 = effective_sample_size_mcmc(&chain1_samples);
        let ess2 = effective_sample_size_mcmc(&chain2_samples);

        println!("✅ MCMC diagnostics computed");
        println!(
            "   - Chain 1: {} samples, ESS = {:.1}",
            chain1_samples.len(),
            ess1
        );
        println!(
            "   - Chain 2: {} samples, ESS = {:.1}",
            chain2_samples.len(),
            ess2
        );
        println!("   - R-hat: {:.4} (< 1.1 indicates convergence)", r_hat);

        if r_hat < 1.1 {
            println!("   - ✅ Chains appear to have converged");
        } else {
            println!("   - ⚠️  Chains may not have converged - run longer");
        }
    }
    // ANCHOR_END: mcmc_diagnostics
    println!();

    println!("6. Debugging Model Structure and Dependencies");
    println!("--------------------------------------------");
    // ANCHOR: model_structure_debugging
    // Create a complex model to demonstrate structure analysis
    let complex_model = || {
        prob!(
            // Hierarchical structure
            let global_scale <- sample(addr!("global_scale"), Gamma::new(2.0, 1.0).unwrap());

            let group_params <- plate!(g in 0..3 => {
                sample(addr!("group_mean", g), Normal::new(0.0, global_scale).unwrap())
                    .bind(move |mean| {
                        sample(addr!("group_precision", g), Gamma::new(2.0, 1.0).unwrap())
                            .map(move |prec| (mean, prec))
                    })
            });

            // Individual observations (simplified to avoid move issues)
            let observations = [1.2, 1.5, 0.8];
            let likelihoods <- plate!(i in 0..observations.len() => {
                // Use fixed parameters for demonstration
                observe(addr!("obs", i), Normal::new(0.0, 1.0).unwrap(), observations[i])
            });

            pure((global_scale, group_params, likelihoods))
        )
    };

    let (_result, complex_trace) = runtime::handler::run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        complex_model(),
    );

    // Analyze model structure
    let mut address_analysis = BTreeMap::new();
    for (addr, choice) in &complex_trace.choices {
        let addr_str = addr.0.clone();
        let category = if addr_str.contains("global") {
            "Global Parameters"
        } else if addr_str.contains("group") {
            "Group Parameters"
        } else if addr_str.contains("obs") {
            "Observations"
        } else {
            "Other"
        };

        address_analysis
            .entry(category)
            .or_insert(Vec::new())
            .push((addr_str, choice.logp));
    }

    println!("✅ Complex model structure analysis");
    println!("   - Total choices: {}", complex_trace.choices.len());
    println!("   - Address structure:");
    for (category, addresses) in address_analysis {
        println!("     {}: {} choices", category, addresses.len());
        for (addr, logp) in addresses.iter().take(3) {
            // Show first 3
            println!("       {} (logp: {:.3})", addr, logp);
        }
        if addresses.len() > 3 {
            println!("       ... and {} more", addresses.len() - 3);
        }
    }
    // ANCHOR_END: model_structure_debugging
    println!();

    println!("7. Performance and Memory Diagnostics");
    println!("------------------------------------");
    // ANCHOR: performance_diagnostics
    use std::time::Instant;

    // Benchmark model execution and trace construction
    let benchmark_model = || {
        prob!(
            let params <- plate!(i in 0..100 => {
                sample(addr!("param", i), Normal::new(0.0, 1.0).unwrap())
            });
            pure(params)
        )
    };

    let start = Instant::now();
    let (_, bench_trace) = runtime::handler::run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        benchmark_model(),
    );
    let execution_time = start.elapsed();

    // Analyze trace characteristics
    let choice_count = bench_trace.choices.len();
    let memory_estimate = choice_count * 64; // Rough estimate
    let log_weight_is_finite = bench_trace.total_log_weight().is_finite();

    println!("✅ Performance diagnostics");
    println!("   - Execution time: {:?}", execution_time);
    println!("   - Choices created: {}", choice_count);
    println!("   - Memory estimate: ~{} bytes", memory_estimate);
    println!("   - Log-weight valid: {}", log_weight_is_finite);

    // Check for potential issues
    if choice_count == 0 {
        println!("   - ⚠️  No choices recorded - possible model issue");
    }
    if !log_weight_is_finite {
        println!("   - ⚠️  Invalid log-weight - check factors and observations");
    }
    if execution_time.as_millis() > 100 {
        println!("   - ⚠️  Slow execution - consider optimization");
    }
    // ANCHOR_END: performance_diagnostics
    println!();

    println!("8. Common Debugging Patterns and Best Practices");
    println!("----------------------------------------------");
    // ANCHOR: debugging_patterns
    // Pattern 1: Systematic model testing
    fn test_model_basic_properties<F, T>(
        model_fn: F,
        expected_choice_count: usize,
        description: &str,
    ) where
        F: Fn() -> Model<T>,
    {
        let mut rng = thread_rng();
        let (_, trace) = runtime::handler::run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            model_fn(),
        );

        println!("Testing {}", description);

        // Basic trace validity
        assert!(
            trace.total_log_weight().is_finite(),
            "Log-weight should be finite"
        );
        assert_eq!(
            trace.choices.len(),
            expected_choice_count,
            "Choice count mismatch"
        );

        // Check for common issues
        if trace.log_prior.is_infinite() {
            println!("  - ⚠️  Infinite prior - check parameter ranges");
        }
        if trace.log_likelihood.is_infinite() {
            println!("  - ⚠️  Infinite likelihood - check observations");
        }
        if trace.log_factors.is_infinite() {
            println!("  - ⚠️  Infinite factors - check constraint satisfaction");
        }

        println!("  - ✅ {} passed basic tests", description);
    }

    // Test simple models
    test_model_basic_properties(
        || sample(addr!("x"), Normal::new(0.0, 1.0).unwrap()),
        1,
        "Simple normal sampling",
    );

    test_model_basic_properties(
        || {
            prob!(
                let x <- sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
                observe(addr!("y"), Normal::new(x, 0.5).unwrap(), 1.0);
                pure(x)
            )
        },
        1,
        "Normal model with observation",
    );

    // Pattern 2: Address collision detection
    fn check_address_collisions(trace: &Trace) -> Vec<String> {
        let mut collisions = Vec::new();
        let addresses: Vec<&str> = trace.choices.keys().map(|addr| addr.0.as_str()).collect();

        for (i, addr1) in addresses.iter().enumerate() {
            for addr2 in addresses.iter().skip(i + 1) {
                if addr1 == addr2 {
                    collisions.push(format!("Duplicate address: {}", addr1));
                }
            }
        }
        collisions
    }

    let test_trace = complex_trace; // Use trace from earlier
    let collisions = check_address_collisions(&test_trace);
    if collisions.is_empty() {
        println!("  - ✅ No address collisions detected");
    } else {
        for collision in collisions {
            println!("  - ⚠️  {}", collision);
        }
    }

    println!("✅ Debugging patterns demonstration complete");
    // ANCHOR_END: debugging_patterns
    println!();

    println!("=== Model Debugging Techniques Demonstrated! ===");
}

#[cfg(test)]
mod tests {
    use super::*;

    // ANCHOR: debugging_tests
    #[test]
    fn test_trace_inspection_patterns() {
        let mut rng = thread_rng();

        let model = prob!(
            let x <- sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
            let y <- sample(addr!("y"), Beta::new(1.0, 1.0).unwrap());
            observe(addr!("obs"), Normal::new(x, 0.1).unwrap(), 1.5);
            pure((x, y))
        );

        let (result, trace) = runtime::handler::run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            model,
        );

        // Basic trace properties
        assert_eq!(trace.choices.len(), 2); // x and y samples
        assert!(trace.total_log_weight().is_finite());
        assert!(trace.log_likelihood.is_finite());

        // Type-safe access
        assert!(trace.get_f64(&addr!("x")).is_some());
        assert!(trace.get_f64(&addr!("y")).is_some());
        assert!(trace.get_bool(&addr!("x")).is_none()); // Type mismatch

        // Result access patterns
        assert!(trace.get_f64_result(&addr!("x")).is_ok());
        assert!(trace.get_f64_result(&addr!("missing")).is_err());
    }

    #[test]
    fn test_safe_vs_strict_handlers() {
        let mut rng = thread_rng();

        // Create base trace
        let mut base_trace = Trace::default();
        base_trace.insert_choice(addr!("param"), ChoiceValue::F64(2.5), -1.0);

        let model = sample(addr!("param"), Normal::new(0.0, 1.0).unwrap());

        // Safe replay should work
        let safe_handler = SafeReplayHandler {
            rng: &mut rng,
            base: base_trace,
            trace: Trace::default(),
            warn_on_mismatch: false,
        };
        let (result, trace) = runtime::handler::run(safe_handler, model);

        assert_eq!(result, 2.5);
        assert_eq!(trace.get_f64(&addr!("param")), Some(2.5));
    }

    #[test]
    fn test_model_structure_analysis() {
        let mut rng = thread_rng();

        let hierarchical_model = || {
            prob!(
                let global <- sample(addr!("global"), Normal::new(0.0, 1.0).unwrap());
                let locals <- plate!(i in 0..3 => {
                    sample(addr!("local", i), Normal::new(global, 0.1).unwrap())
                });
                pure((global, locals))
            )
        };

        let (_, trace) = runtime::handler::run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            hierarchical_model(),
        );

        // Should have global + 3 local parameters
        assert_eq!(trace.choices.len(), 4);

        // Check address structure
        assert!(trace.choices.contains_key(&addr!("global")));
        assert!(trace.choices.contains_key(&addr!("local", 0)));
        assert!(trace.choices.contains_key(&addr!("local", 1)));
        assert!(trace.choices.contains_key(&addr!("local", 2)));
    }

    #[test]
    fn test_performance_diagnostics() {
        use std::time::Instant;
        let mut rng = thread_rng();

        let large_model = || {
            plate!(i in 0..50 => {
                sample(addr!("x", i), Normal::new(0.0, 1.0).unwrap())
            })
        };

        let start = Instant::now();
        let (_, trace) = runtime::handler::run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            large_model(),
        );
        let duration = start.elapsed();

        assert_eq!(trace.choices.len(), 50);
        assert!(trace.total_log_weight().is_finite());

        // Performance should be reasonable
        assert!(duration.as_millis() < 1000, "Model execution too slow");
    }
    // ANCHOR_END: debugging_tests
}
