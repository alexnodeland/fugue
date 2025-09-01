use fugue::inference::diagnostics::{extract_f64_values, r_hat_f64, summarize_f64_parameter};
use fugue::runtime::{
    handler::Handler,
    interpreters::{PriorHandler, ReplayHandler, ScoreGivenTrace},
    memory::{CowTrace, TraceBuilder},
    trace::{Choice, ChoiceValue, Trace},
};
use fugue::*;
use rand::{rngs::StdRng, SeedableRng};

// ANCHOR: basic_trace_inspection
// Demonstrate basic trace inspection and manipulation
fn basic_trace_inspection() {
    println!("=== Basic Trace Inspection ===\n");

    // Define a model with multiple types of choices
    let model = prob!(
        let coin <- sample(addr!("coin"), Bernoulli::new(0.7).unwrap());
        let count <- sample(addr!("count"), Poisson::new(3.0).unwrap());
        let category <- sample(addr!("category"), Categorical::uniform(3).unwrap());
        let measurement <- sample(addr!("measurement"), Normal::new(0.0, 1.0).unwrap());

        // Observation adds to likelihood
        let _obs <- observe(addr!("obs"), Normal::new(measurement, 0.1).unwrap(), 0.5);

        pure((coin, count, category, measurement))
    );

    // Execute and inspect the trace
    let mut rng = StdRng::seed_from_u64(12345);
    let (result, trace) = runtime::handler::run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        model,
    );

    println!("🔍 Trace Inspection:");
    println!("   - Total choices: {}", trace.choices.len());
    println!("   - Prior log-weight: {:.4}", trace.log_prior);
    println!("   - Likelihood log-weight: {:.4}", trace.log_likelihood);
    println!("   - Factor log-weight: {:.4}", trace.log_factors);
    println!("   - Total log-weight: {:.4}", trace.total_log_weight());
    println!();

    println!("📊 Individual Choices:");
    for (addr, choice) in &trace.choices {
        println!(
            "   - {}: {:?} (logp: {:.4})",
            addr, choice.value, choice.logp
        );
    }
    println!();

    println!("🎯 Type-Safe Value Access:");
    println!("   - Coin (bool): {:?}", trace.get_bool(&addr!("coin")));
    println!("   - Count (u64): {:?}", trace.get_u64(&addr!("count")));
    println!(
        "   - Category (usize): {:?}",
        trace.get_usize(&addr!("category"))
    );
    println!(
        "   - Measurement (f64): {:?}",
        trace.get_f64(&addr!("measurement"))
    );

    let (coin, count, category, measurement) = result;
    println!(
        "   - Result: coin={}, count={}, category={}, measurement={:.3}",
        coin, count, category, measurement
    );
    println!();
}
// ANCHOR_END: basic_trace_inspection

// ANCHOR: replay_mechanics
// Demonstrate trace replay mechanics for MCMC
fn replay_mechanics() {
    println!("=== Trace Replay Mechanics ===\n");

    // Define a simple model
    let make_model = || {
        prob!(
            let mu <- sample(addr!("mu"), Normal::new(0.0, 2.0).unwrap());
            let _obs <- observe(addr!("y"), Normal::new(mu, 1.0).unwrap(), 1.5);
            pure(mu)
        )
    };

    // 1. Generate initial trace
    let mut rng = StdRng::seed_from_u64(42);
    let (mu1, trace1) = runtime::handler::run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        make_model(),
    );

    println!("🎲 Original Execution:");
    println!("   - mu = {:.3}", mu1);
    println!("   - Prior logp: {:.3}", trace1.log_prior);
    println!("   - Likelihood logp: {:.3}", trace1.log_likelihood);
    println!("   - Total logp: {:.3}", trace1.total_log_weight());
    println!();

    // 2. Replay with exact same trace
    let mut rng2 = StdRng::seed_from_u64(42); // New RNG for replay
    let (mu2, trace2) = runtime::handler::run(
        ReplayHandler {
            rng: &mut rng2,
            base: trace1.clone(),
            trace: Trace::default(),
        },
        make_model(),
    );

    println!("🔄 Exact Replay:");
    println!("   - mu = {:.3} (should match original)", mu2);
    println!("   - Values match: {}", mu1 == mu2);
    println!(
        "   - Traces match: {}",
        trace1.total_log_weight() == trace2.total_log_weight()
    );
    println!();

    // 3. Modify trace for proposal
    let mut modified_trace = trace1.clone();
    // Modify the mu value (MCMC proposal)
    if let Some(choice) = modified_trace.choices.get_mut(&addr!("mu")) {
        let old_value = choice.value.as_f64().unwrap();
        let new_value = old_value + 0.1; // Small proposal step
        choice.value = ChoiceValue::F64(new_value);

        // Recompute log-probability under the distribution
        let normal_dist = Normal::new(0.0, 2.0).unwrap();
        choice.logp = normal_dist.log_prob(&new_value);

        println!("🔧 Modified Trace (Proposal):");
        println!("   - Old mu: {:.3}", old_value);
        println!("   - New mu: {:.3}", new_value);
        println!(
            "   - Old logp: {:.3}",
            trace1
                .get_f64(&addr!("mu"))
                .map(|v| Normal::new(0.0, 2.0).unwrap().log_prob(&v))
                .unwrap_or(0.0)
        );
        println!("   - New logp: {:.3}", choice.logp);
    }

    // 4. Score the modified trace
    let mut rng3 = StdRng::seed_from_u64(42); // New RNG for proposal
    let (mu3, trace3) = runtime::handler::run(
        ReplayHandler {
            rng: &mut rng3,
            base: modified_trace,
            trace: Trace::default(),
        },
        make_model(),
    );

    println!("   - Proposal result: mu = {:.3}", mu3);
    println!("   - Proposal total logp: {:.3}", trace3.total_log_weight());
    println!(
        "   - Accept/Reject ratio: {:.3}",
        (trace3.total_log_weight() - trace1.total_log_weight()).exp()
    );
    println!();
}
// ANCHOR_END: replay_mechanics

// ANCHOR: custom_handler
// Demonstrate custom handler implementation
struct DebugHandler<R: rand::Rng> {
    rng: R,
    trace: Trace,
    debug_info: Vec<String>,
}

impl<R: rand::Rng> DebugHandler<R> {
    fn new(rng: R) -> Self {
        Self {
            rng,
            trace: Trace::default(),
            debug_info: Vec::new(),
        }
    }
}

impl<R: rand::Rng> Handler for DebugHandler<R> {
    fn on_sample_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>) -> f64 {
        let value = dist.sample(&mut self.rng);
        let logp = dist.log_prob(&value);

        self.debug_info.push(format!(
            "SAMPLE f64 at {}: {} (logp: {:.3})",
            addr, value, logp
        ));

        self.trace.choices.insert(
            addr.clone(),
            Choice {
                addr: addr.clone(),
                value: ChoiceValue::F64(value),
                logp,
            },
        );
        self.trace.log_prior += logp;

        value
    }

    fn on_sample_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>) -> bool {
        let value = dist.sample(&mut self.rng);
        let logp = dist.log_prob(&value);

        self.debug_info.push(format!(
            "SAMPLE bool at {}: {} (logp: {:.3})",
            addr, value, logp
        ));

        self.trace.choices.insert(
            addr.clone(),
            Choice {
                addr: addr.clone(),
                value: ChoiceValue::Bool(value),
                logp,
            },
        );
        self.trace.log_prior += logp;

        value
    }

    fn on_sample_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>) -> u64 {
        let value = dist.sample(&mut self.rng);
        let logp = dist.log_prob(&value);

        self.debug_info.push(format!(
            "SAMPLE u64 at {}: {} (logp: {:.3})",
            addr, value, logp
        ));

        self.trace.choices.insert(
            addr.clone(),
            Choice {
                addr: addr.clone(),
                value: ChoiceValue::U64(value),
                logp,
            },
        );
        self.trace.log_prior += logp;

        value
    }

    fn on_sample_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>) -> usize {
        let value = dist.sample(&mut self.rng);
        let logp = dist.log_prob(&value);

        self.debug_info.push(format!(
            "SAMPLE usize at {}: {} (logp: {:.3})",
            addr, value, logp
        ));

        self.trace.choices.insert(
            addr.clone(),
            Choice {
                addr: addr.clone(),
                value: ChoiceValue::Usize(value),
                logp,
            },
        );
        self.trace.log_prior += logp;

        value
    }

    fn on_observe_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>, value: f64) {
        let logp = dist.log_prob(&value);

        self.debug_info.push(format!(
            "OBSERVE f64 at {}: {} (logp: {:.3})",
            addr, value, logp
        ));

        self.trace.log_likelihood += logp;
    }

    fn on_observe_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>, value: bool) {
        let logp = dist.log_prob(&value);

        self.debug_info.push(format!(
            "OBSERVE bool at {}: {} (logp: {:.3})",
            addr, value, logp
        ));

        self.trace.log_likelihood += logp;
    }

    fn on_observe_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>, value: u64) {
        let logp = dist.log_prob(&value);

        self.debug_info.push(format!(
            "OBSERVE u64 at {}: {} (logp: {:.3})",
            addr, value, logp
        ));

        self.trace.log_likelihood += logp;
    }

    fn on_observe_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>, value: usize) {
        let logp = dist.log_prob(&value);

        self.debug_info.push(format!(
            "OBSERVE usize at {}: {} (logp: {:.3})",
            addr, value, logp
        ));

        self.trace.log_likelihood += logp;
    }

    fn on_factor(&mut self, logw: f64) {
        self.debug_info.push(format!("FACTOR: {:.3}", logw));
        self.trace.log_factors += logw;
    }

    fn finish(self) -> Trace {
        self.trace
    }
}

fn custom_handler_demo() {
    println!("=== Custom Handler Demo ===\n");

    let model = prob!(
        let prior_mu <- sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap());
        let success <- sample(addr!("success"), Bernoulli::new(0.6).unwrap());

        let _obs1 <- observe(addr!("data1"), Normal::new(prior_mu, 0.5).unwrap(), 1.2);
        let _obs2 <- observe(addr!("data2"), Bernoulli::new(if success { 0.8 } else { 0.2 }).unwrap(), true);

        // Add a factor for soft constraints
        let _factor_result <- factor(if prior_mu > 0.0 { 0.1 } else { -0.1 });

        pure((prior_mu, success))
    );

    let rng = StdRng::seed_from_u64(67890);
    let debug_handler = DebugHandler::new(rng);

    let (result, final_trace) = runtime::handler::run(debug_handler, model);

    println!("🔍 Debug Handler Output:");
    println!("   - Result: {:?}", result);
    println!(
        "   - Total log-weight: {:.4}",
        final_trace.total_log_weight()
    );
    println!();

    println!("📝 Execution Log:");
    println!("   - {} operations recorded", final_trace.choices.len());

    println!();
}
// ANCHOR_END: custom_handler

// ANCHOR: trace_scoring
// Demonstrate trace scoring for importance sampling
fn trace_scoring_demo() {
    println!("=== Trace Scoring Demo ===\n");

    let make_model = || {
        prob!(
            let theta <- sample(addr!("theta"), Beta::new(2.0, 2.0).unwrap());

            // Multiple observations
            let _obs1 <- observe(addr!("y1"), Bernoulli::new(theta).unwrap(), true);
            let _obs2 <- observe(addr!("y2"), Bernoulli::new(theta).unwrap(), true);
            let _obs3 <- observe(addr!("y3"), Bernoulli::new(theta).unwrap(), false);

            pure(theta)
        )
    };

    // Generate a trace from the prior
    let mut rng = StdRng::seed_from_u64(111);
    let (theta_val, prior_trace) = runtime::handler::run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        make_model(),
    );

    println!("🎲 Prior Sample:");
    println!("   - theta = {:.3}", theta_val);
    println!("   - Prior logp: {:.3}", prior_trace.log_prior);
    println!("   - Likelihood logp: {:.3}", prior_trace.log_likelihood);
    println!("   - Total logp: {:.3}", prior_trace.total_log_weight());
    println!();

    // Now score this trace under the model (should get same result)
    let (theta_scored, scored_trace) = runtime::handler::run(
        ScoreGivenTrace {
            base: prior_trace.clone(),
            trace: Trace::default(),
        },
        make_model(),
    );

    println!("📊 Scoring Same Trace:");
    println!("   - theta = {:.3} (should match)", theta_scored);
    println!("   - Prior logp: {:.3}", scored_trace.log_prior);
    println!("   - Likelihood logp: {:.3}", scored_trace.log_likelihood);
    println!("   - Total logp: {:.3}", scored_trace.total_log_weight());
    println!(
        "   - Weights match: {}",
        (prior_trace.total_log_weight() - scored_trace.total_log_weight()).abs() < 1e-10
    );
    println!();

    // Create a modified trace for importance sampling
    let mut importance_trace = prior_trace.clone();

    // Change theta to a different value
    if let Some(choice) = importance_trace.choices.get_mut(&addr!("theta")) {
        let new_theta = 0.8; // High success probability
        choice.value = ChoiceValue::F64(new_theta);
        choice.logp = Beta::new(2.0, 2.0).unwrap().log_prob(&new_theta);

        println!("🎯 Importance Sample:");
        println!("   - Modified theta to: {:.3}", new_theta);
        println!("   - New prior logp: {:.3}", choice.logp);
    }

    // Score under original model
    let (theta_is, is_trace) = runtime::handler::run(
        ScoreGivenTrace {
            base: importance_trace,
            trace: Trace::default(),
        },
        make_model(),
    );

    println!("   - IS result: theta = {:.3}", theta_is);
    println!("   - IS total logp: {:.3}", is_trace.total_log_weight());
    println!(
        "   - Importance weight: {:.3}",
        is_trace.total_log_weight() - prior_trace.total_log_weight()
    );
    println!();
}
// ANCHOR_END: trace_scoring

// ANCHOR: memory_optimization
// Demonstrate memory-optimized trace handling
fn memory_optimization_demo() {
    println!("=== Memory Optimization Demo ===\n");

    // Simple model for batch processing
    let make_model = |obs_val: f64| {
        prob!(
            let mu <- sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap());
            let sigma <- sample(addr!("sigma"), Gamma::new(2.0, 0.5).unwrap());
            let _obs <- observe(addr!("y"), Normal::new(mu, sigma).unwrap(), obs_val);
            pure((mu, sigma))
        )
    };

    println!("🏭 Batch Processing with Memory Pool:");

    // Simulate batch inference with trace reuse
    let observations = [1.0, 1.2, 0.8, 1.5, 0.9];
    let mut results = Vec::new();

    // Use copy-on-write traces for efficiency
    let base_trace = CowTrace::new();

    for (i, &obs) in observations.iter().enumerate() {
        let mut rng = StdRng::seed_from_u64(200 + i as u64);
        let handler = PriorHandler {
            rng: &mut rng,
            trace: base_trace.to_trace(), // Convert to regular trace
        };

        let (result, trace) = runtime::handler::run(handler, make_model(obs));
        results.push((result, trace));

        println!(
            "   Sample {}: mu={:.3}, sigma={:.3}, obs={:.1}, logp={:.3}",
            i + 1,
            result.0,
            result.1,
            obs,
            results[i].1.total_log_weight()
        );
    }

    println!();
    println!("📊 Batch Statistics:");
    let mu_mean = results.iter().map(|((mu, _), _)| mu).sum::<f64>() / results.len() as f64;
    let sigma_mean =
        results.iter().map(|((_, sigma), _)| sigma).sum::<f64>() / results.len() as f64;
    let logp_mean = results
        .iter()
        .map(|(_, trace)| trace.total_log_weight())
        .sum::<f64>()
        / results.len() as f64;

    println!("   - Average mu: {:.3}", mu_mean);
    println!("   - Average sigma: {:.3}", sigma_mean);
    println!("   - Average log-probability: {:.3}", logp_mean);
    println!();

    println!("🔧 Trace Builder Demo:");

    // Demonstrate efficient trace building
    let _builder = TraceBuilder::new();
    // Note: TraceBuilder API may not have reserve_choices method
    // This is a conceptual example of memory pre-allocation

    // Manually construct a trace (rarely needed, but shows internals)
    let demo_trace = Trace {
        choices: [
            (
                addr!("param1"),
                Choice {
                    addr: addr!("param1"),
                    value: ChoiceValue::F64(0.5),
                    logp: -1.4,
                },
            ),
            (
                addr!("param2"),
                Choice {
                    addr: addr!("param2"),
                    value: ChoiceValue::Bool(true),
                    logp: -0.7,
                },
            ),
        ]
        .iter()
        .cloned()
        .collect(),
        log_prior: -2.1,
        log_likelihood: -0.5,
        log_factors: 0.0,
    };

    println!(
        "   - Manual trace: {} choices, total logp: {:.3}",
        demo_trace.choices.len(),
        demo_trace.total_log_weight()
    );
    println!();
}
// ANCHOR_END: memory_optimization

// ANCHOR: diagnostic_tools
// Demonstrate diagnostic tools for trace analysis
fn diagnostic_tools_demo() {
    println!("=== Diagnostic Tools Demo ===\n");

    // Generate multiple MCMC-like traces for diagnostics
    let make_model = || {
        prob!(
            let mu <- sample(addr!("mu"), Normal::new(0.0, 2.0).unwrap());
            let precision <- sample(addr!("precision"), Gamma::new(2.0, 1.0).unwrap());

            // Multiple observations
            let _obs1 <- observe(addr!("y1"), Normal::new(mu, 1.0/precision.sqrt()).unwrap(), 1.0);
            let _obs2 <- observe(addr!("y2"), Normal::new(mu, 1.0/precision.sqrt()).unwrap(), 1.2);
            let _obs3 <- observe(addr!("y3"), Normal::new(mu, 1.0/precision.sqrt()).unwrap(), 0.8);

            pure((mu, precision))
        )
    };

    // Simulate two chains
    println!("🔗 Generating MCMC-like traces:");
    let mut chain1 = Vec::new();
    let mut chain2 = Vec::new();

    // Chain 1
    for i in 0..20 {
        let mut rng = StdRng::seed_from_u64(300 + i);
        let (_, trace) = runtime::handler::run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            make_model(),
        );
        chain1.push(trace);
    }

    // Chain 2 (different seed)
    for i in 0..20 {
        let mut rng = StdRng::seed_from_u64(400 + i);
        let (_, trace) = runtime::handler::run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            make_model(),
        );
        chain2.push(trace);
    }

    println!("   - Chain 1: {} samples", chain1.len());
    println!("   - Chain 2: {} samples", chain2.len());
    println!();

    // Extract parameter values
    let mu_values1 = extract_f64_values(&chain1, &addr!("mu"));
    let mu_values2 = extract_f64_values(&chain2, &addr!("mu"));
    let precision_values1 = extract_f64_values(&chain1, &addr!("precision"));
    let _precision_values2 = extract_f64_values(&chain2, &addr!("precision"));

    println!("📈 Parameter Summaries:");
    println!(
        "   - mu chain1: mean={:.3}, min={:.3}, max={:.3}",
        mu_values1.iter().sum::<f64>() / mu_values1.len() as f64,
        mu_values1.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
        mu_values1.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
    );

    println!(
        "   - mu chain2: mean={:.3}, min={:.3}, max={:.3}",
        mu_values2.iter().sum::<f64>() / mu_values2.len() as f64,
        mu_values2.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
        mu_values2.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
    );

    println!(
        "   - precision chain1: mean={:.3}, min={:.3}, max={:.3}",
        precision_values1.iter().sum::<f64>() / precision_values1.len() as f64,
        precision_values1
            .iter()
            .fold(f64::INFINITY, |a, &b| a.min(b)),
        precision_values1
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
    );

    // Compute R-hat (simplified version)
    let chains_mu = vec![chain1.clone(), chain2.clone()];
    let r_hat_mu = r_hat_f64(&chains_mu, &addr!("mu"));
    let r_hat_precision = r_hat_f64(&chains_mu, &addr!("precision"));

    println!();
    println!("🎯 Convergence Diagnostics:");
    println!("   - R-hat for mu: {:.4} (< 1.1 is good)", r_hat_mu);
    println!(
        "   - R-hat for precision: {:.4} (< 1.1 is good)",
        r_hat_precision
    );

    // Parameter summary
    let mu_summary = summarize_f64_parameter(&chains_mu, &addr!("mu"));
    let q5 = mu_summary.quantiles.get("2.5%").unwrap_or(&f64::NAN);
    let q95 = mu_summary.quantiles.get("97.5%").unwrap_or(&f64::NAN);
    println!(
        "   - mu summary: mean={:.3}, std={:.3}, q2.5={:.3}, q97.5={:.3}",
        mu_summary.mean, mu_summary.std, q5, q95
    );

    println!();
    println!("📊 Trace Quality Assessment:");
    let total_logp_chain1: f64 = chain1.iter().map(|t| t.total_log_weight()).sum();
    let total_logp_chain2: f64 = chain2.iter().map(|t| t.total_log_weight()).sum();
    let avg_logp1 = total_logp_chain1 / chain1.len() as f64;
    let avg_logp2 = total_logp_chain2 / chain2.len() as f64;

    println!("   - Chain 1 avg log-probability: {:.3}", avg_logp1);
    println!("   - Chain 2 avg log-probability: {:.3}", avg_logp2);
    println!(
        "   - Chains similar quality: {}",
        (avg_logp1 - avg_logp2).abs() < 0.5
    );
    println!();
}
// ANCHOR_END: diagnostic_tools

// ANCHOR: advanced_debugging
// Demonstrate advanced debugging techniques
fn advanced_debugging_demo() {
    println!("=== Advanced Debugging Techniques ===\n");

    // Model with potential numerical issues
    let _problematic_model = prob!(
        let scale <- sample(addr!("scale"), Exponential::new(1.0).unwrap());

        // This could cause numerical issues if scale is very small
        let precision <- sample(addr!("precision"), Gamma::new(1.0, scale).unwrap());

        let mu <- sample(addr!("mu"), Normal::new(0.0, 1.0 / precision.sqrt()).unwrap());

        // Observation that might conflict
        let _obs <- observe(addr!("y"), Normal::new(mu, 0.01).unwrap(), 10.0);

        pure((scale, precision, mu))
    );

    println!("🚨 Debugging Problematic Model:");

    // Try multiple executions to find issues
    for attempt in 1..=5 {
        let mut rng = StdRng::seed_from_u64(500 + attempt);
        let problematic_model_copy = prob!(
            let scale <- sample(addr!("scale"), Exponential::new(1.0).unwrap());

            // This could cause numerical issues if scale is very small
            let precision <- sample(addr!("precision"), Gamma::new(1.0, scale).unwrap());

            let mu <- sample(addr!("mu"), Normal::new(0.0, 1.0 / precision.sqrt()).unwrap());

            // Observation that might conflict
            let _obs <- observe(addr!("y"), Normal::new(mu, 0.01).unwrap(), 10.0);

            pure((scale, precision, mu))
        );

        let (result, trace) = runtime::handler::run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            problematic_model_copy,
        );

        let (scale, precision, mu) = result;
        let total_logp = trace.total_log_weight();

        println!(
            "   Attempt {}: scale={:.6}, precision={:.6}, mu={:.3}, logp={:.3}",
            attempt, scale, precision, mu, total_logp
        );

        // Check for numerical issues
        if !total_logp.is_finite() {
            println!("     ⚠️ Non-finite log-probability detected!");
        }

        if precision < 1e-6 {
            println!("     ⚠️ Very small precision: {:.8}", precision);
        }

        if mu.abs() > 5.0 {
            println!("     ⚠️ Extreme mu value: {:.3}", mu);
        }

        // Examine individual components
        println!(
            "     Components: prior={:.3}, likelihood={:.3}, factors={:.3}",
            trace.log_prior, trace.log_likelihood, trace.log_factors
        );
    }

    println!();
    println!("🔍 Trace Validation:");

    // Create a trace with known good values for validation
    let validation_trace = Trace {
        choices: [
            (
                addr!("scale"),
                Choice {
                    addr: addr!("scale"),
                    value: ChoiceValue::F64(1.0),
                    logp: Exponential::new(1.0).unwrap().log_prob(&1.0),
                },
            ),
            (
                addr!("precision"),
                Choice {
                    addr: addr!("precision"),
                    value: ChoiceValue::F64(2.0),
                    logp: Gamma::new(1.0, 1.0).unwrap().log_prob(&2.0),
                },
            ),
            (
                addr!("mu"),
                Choice {
                    addr: addr!("mu"),
                    value: ChoiceValue::F64(0.5),
                    logp: Normal::new(0.0, 1.0 / (2.0_f64).sqrt())
                        .unwrap()
                        .log_prob(&0.5),
                },
            ),
        ]
        .iter()
        .cloned()
        .collect(),
        log_prior: 0.0,
        log_likelihood: 0.0,
        log_factors: 0.0,
    };

    // Score this validation trace
    let validation_model = prob!(
        let scale <- sample(addr!("scale"), Exponential::new(1.0).unwrap());
        let precision <- sample(addr!("precision"), Gamma::new(1.0, scale).unwrap());
        let mu <- sample(addr!("mu"), Normal::new(0.0, 1.0 / precision.sqrt()).unwrap());
        let _obs <- observe(addr!("y"), Normal::new(mu, 0.01).unwrap(), 10.0);
        pure((scale, precision, mu))
    );

    let (val_result, val_trace) = runtime::handler::run(
        ScoreGivenTrace {
            base: validation_trace,
            trace: Trace::default(),
        },
        validation_model,
    );

    println!("   - Validation result: {:?}", val_result);
    println!("   - Validation logp: {:.3}", val_trace.total_log_weight());
    println!(
        "   - Validation finite: {}",
        val_trace.total_log_weight().is_finite()
    );
    println!();
}
// ANCHOR_END: advanced_debugging

fn main() {
    println!("🎯 Fugue Trace Manipulation Demonstration");
    println!("=========================================\n");

    basic_trace_inspection();
    replay_mechanics();
    custom_handler_demo();
    trace_scoring_demo();
    memory_optimization_demo();
    diagnostic_tools_demo();
    advanced_debugging_demo();

    println!("🏁 Trace Manipulation Demonstration Complete!");
    println!("\nKey Capabilities:");
    println!("• Complete execution history recording with type safety");
    println!("• Flexible replay mechanics for MCMC and inference");
    println!("• Custom handlers for specialized execution strategies");
    println!("• Trace scoring for importance sampling and model comparison");
    println!("• Memory optimization for production workloads");
    println!("• Comprehensive diagnostic tools for convergence assessment");
    println!("• Advanced debugging techniques for numerical stability");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_basic_operations() {
        let model = prob!(
            let x <- sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
            let y <- sample(addr!("y"), Bernoulli::new(0.5).unwrap());
            pure((x, y))
        );

        let mut rng = StdRng::seed_from_u64(12345);
        let (result, trace) = runtime::handler::run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            model,
        );

        // Verify trace structure
        assert_eq!(trace.choices.len(), 2);
        assert!(trace.choices.contains_key(&addr!("x")));
        assert!(trace.choices.contains_key(&addr!("y")));
        assert!(trace.log_prior != 0.0);
        assert_eq!(trace.log_likelihood, 0.0); // No observations
        assert_eq!(trace.log_factors, 0.0);

        // Verify type-safe access
        let x_val = trace.get_f64(&addr!("x")).unwrap();
        let y_val = trace.get_bool(&addr!("y")).unwrap();
        assert_eq!((x_val, y_val), result);
    }

    #[test]
    fn test_replay_determinism() {
        let make_model = || {
            prob!(
                let mu <- sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap());
                let _obs <- observe(addr!("y"), Normal::new(mu, 0.5).unwrap(), 1.0);
                pure(mu)
            )
        };

        // Generate original trace
        let mut rng = StdRng::seed_from_u64(42);
        let (mu1, trace1) = runtime::handler::run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            make_model(),
        );

        // Replay should give identical results
        let mut rng2 = StdRng::seed_from_u64(42);
        let (mu2, trace2) = runtime::handler::run(
            ReplayHandler {
                rng: &mut rng2,
                base: trace1.clone(),
                trace: Trace::default(),
            },
            make_model(),
        );

        assert_eq!(mu1, mu2);
        assert_eq!(trace1.total_log_weight(), trace2.total_log_weight());
    }

    #[test]
    fn test_trace_scoring() {
        let model = prob!(
            let p <- sample(addr!("p"), Beta::new(1.0, 1.0).unwrap());
            let _obs <- observe(addr!("coin"), Bernoulli::new(p).unwrap(), true);
            pure(p)
        );

        // Create a trace with specific value
        let test_trace = Trace {
            choices: [(
                addr!("p"),
                Choice {
                    addr: addr!("p"),
                    value: ChoiceValue::F64(0.7),
                    logp: Beta::new(1.0, 1.0).unwrap().log_prob(&0.7),
                },
            )]
            .iter()
            .cloned()
            .collect(),
            log_prior: 0.0,
            log_likelihood: 0.0,
            log_factors: 0.0,
        };

        // Score the trace
        let (p_val, scored_trace) = runtime::handler::run(
            ScoreGivenTrace {
                base: test_trace,
                trace: Trace::default(),
            },
            model,
        );

        assert_eq!(p_val, 0.7);
        assert!(scored_trace.log_prior.is_finite()); // Beta(1,1) might have log_prior = 0
        assert!(scored_trace.log_likelihood != 0.0);
        assert!(scored_trace.total_log_weight().is_finite());
    }

    #[test]
    fn test_type_safe_value_extraction() {
        let trace = Trace {
            choices: [
                (
                    addr!("bool_val"),
                    Choice {
                        addr: addr!("bool_val"),
                        value: ChoiceValue::Bool(true),
                        logp: -0.7,
                    },
                ),
                (
                    addr!("u64_val"),
                    Choice {
                        addr: addr!("u64_val"),
                        value: ChoiceValue::U64(42),
                        logp: -2.3,
                    },
                ),
                (
                    addr!("f64_val"),
                    Choice {
                        addr: addr!("f64_val"),
                        value: ChoiceValue::F64(3.14),
                        logp: -1.1,
                    },
                ),
            ]
            .iter()
            .cloned()
            .collect(),
            log_prior: -4.1,
            log_likelihood: 0.0,
            log_factors: 0.0,
        };

        // Test correct type extraction
        assert_eq!(trace.get_bool(&addr!("bool_val")), Some(true));
        assert_eq!(trace.get_u64(&addr!("u64_val")), Some(42));
        assert_eq!(trace.get_f64(&addr!("f64_val")), Some(3.14));

        // Test type mismatches return None
        assert_eq!(trace.get_f64(&addr!("bool_val")), None);
        assert_eq!(trace.get_bool(&addr!("u64_val")), None);
        assert_eq!(trace.get_u64(&addr!("f64_val")), None);

        // Test non-existent addresses
        assert_eq!(trace.get_f64(&addr!("nonexistent")), None);
    }

    #[test]
    fn test_trace_weight_decomposition() {
        let model = prob!(
            let x <- sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
            let _obs <- observe(addr!("y"), Normal::new(x, 0.5).unwrap(), 1.0);
            let _factor_result <- factor(0.5); // Add explicit factor
            pure(x)
        );

        let mut rng = StdRng::seed_from_u64(99);
        let (_result, trace) = runtime::handler::run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            model,
        );

        // Verify weight decomposition
        assert!(trace.log_prior != 0.0); // Should have prior weight
        assert!(trace.log_likelihood != 0.0); // Should have likelihood weight
        assert_eq!(trace.log_factors, 0.5); // Should have factor weight
        assert_eq!(
            trace.total_log_weight(),
            trace.log_prior + trace.log_likelihood + trace.log_factors
        );
    }

    #[test]
    fn test_diagnostic_value_extraction() {
        // Create traces with known values for testing
        let trace1 = Trace {
            choices: [(
                addr!("param"),
                Choice {
                    addr: addr!("param"),
                    value: ChoiceValue::F64(1.0),
                    logp: -0.5,
                },
            )]
            .iter()
            .cloned()
            .collect(),
            ..Default::default()
        };

        let trace2 = Trace {
            choices: [(
                addr!("param"),
                Choice {
                    addr: addr!("param"),
                    value: ChoiceValue::F64(2.0),
                    logp: -0.7,
                },
            )]
            .iter()
            .cloned()
            .collect(),
            ..Default::default()
        };

        let traces = vec![trace1, trace2];
        let values = extract_f64_values(&traces, &addr!("param"));

        assert_eq!(values, vec![1.0, 2.0]);
    }

    #[test]
    fn test_memory_trace_operations() {
        // Test CowTrace basic operations
        let cow_trace = CowTrace::new();

        assert_eq!(cow_trace.choices().len(), 0);
        assert_eq!(cow_trace.total_log_weight(), 0.0);

        // Test conversion to regular trace
        let regular_trace = cow_trace.to_trace();
        assert_eq!(regular_trace.choices.len(), 0);
        assert_eq!(regular_trace.total_log_weight(), 0.0);
    }
}
