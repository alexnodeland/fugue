# Debugging Models

Debugging probabilistic models requires specialized techniques beyond traditional programming. This guide covers systematic approaches to identify, diagnose, and fix issues in your Fugue models.

## Common Model Problems

Before diving into debugging techniques, let's understand common issues:

### 1. Numerical Issues
- Infinite or NaN log-weights
- Numerical underflow/overflow
- Poor conditioning

### 2. Model Specification Errors
- Wrong distributions or parameters
- Missing observations or constraints
- Incorrect conditional logic

### 3. Inference Problems  
- Poor mixing in MCMC
- Numerical instability
- Convergence failures

### 4. Performance Issues
- Slow model execution
- Memory leaks
- Inefficient addressing

## Systematic Debugging Workflow

### Step 1: Isolate the Problem

Start by creating minimal reproducible examples:

```rust
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

// ❌ Complex model that's hard to debug
fn complex_problematic_model(data: &[f64]) -> Model<Vec<f64>> {
    prob! {
        let hyperparams <- plate!(i in 0..5 => {
            sample(addr!("hyper", i), Gamma::new(2.0, 1.0).unwrap())
        });
        
        let params <- plate!(j in 0..data.len() => {
            let hyper_sum: f64 = hyperparams.iter().sum();
            sample(addr!("param", j), Normal::new(hyper_sum, 0.1).unwrap())
        });
        
        for (k, (&param, &obs)) in params.iter().zip(data).enumerate() {
            observe(addr!("obs", k), Normal::new(param * param, 0.01).unwrap(), obs);
        }
        
        pure(params)
    }
}

// ✅ Simplified model for debugging
fn simple_debug_model() -> Model<f64> {
    prob! {
        let x <- sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
        observe(addr!("y"), Normal::new(x * x, 0.01).unwrap(), 4.0);
        pure(x)
    }
}

fn test_isolation() {
    println!("🔍 Problem Isolation");
    println!("==================");
    
    // Test the simplified version first
    let model = simple_debug_model();
    let mut rng = StdRng::seed_from_u64(42);
    
    match std::panic::catch_unwind(|| {
        let (value, trace) = runtime::handler::run(
            runtime::interpreters::PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            model,
        );
        (value, trace.total_log_weight())
    }) {
        Ok((value, log_weight)) => {
            println!("✅ Simple model works: x={:.3}, log_weight={:.4}", value, log_weight);
            if log_weight.is_finite() {
                println!("   Log weight is finite");
            } else {
                println!("⚠️ Log weight is infinite/NaN");
            }
        }
        Err(_) => {
            println!("❌ Simple model panicked");
        }
    }
}
```

### Step 2: Check Model Components

Test each component individually:

```rust
fn test_distributions() {
    println!("\n🧪 Distribution Testing");
    println!("=====================");
    
    // Test distribution construction
    let distributions = vec![
        ("Normal(0,1)", Normal::new(0.0, 1.0)),
        ("Normal(0,-1)", Normal::new(0.0, -1.0)),  // Invalid!
        ("LogNormal(0,1)", LogNormal::new(0.0, 1.0)),
        ("Beta(1,1)", Beta::new(1.0, 1.0)),
        ("Beta(-1,1)", Beta::new(-1.0, 1.0)),  // Invalid!
    ];
    
    for (name, result) in distributions {
        match result {
            Ok(dist) => {
                println!("✅ {}: Created successfully", name);
                
                // Test sampling
                let mut rng = StdRng::seed_from_u64(42);
                let sample = dist.sample(&mut rng);
                println!("   Sample: {:.4}", sample);
                
                // Test log_pdf
                let log_prob = dist.log_pdf(sample);
                if log_prob.is_finite() {
                    println!("   Log prob: {:.4}", log_prob);
                } else {
                    println!("⚠️ Log prob is infinite/NaN: {}", log_prob);
                }
            }
            Err(e) => {
                println!("❌ {}: {}", name, e);
            }
        }
    }
}

fn test_observations() {
    println!("\n👁️ Observation Testing");
    println!("=====================");
    
    let test_cases = vec![
        ("Normal observation in range", Normal::new(0.0, 1.0).unwrap(), 0.5),
        ("Normal observation extreme", Normal::new(0.0, 1.0).unwrap(), 10.0),
        ("Beta observation valid", Beta::new(2.0, 2.0).unwrap(), 0.7),
        ("Beta observation invalid", Beta::new(2.0, 2.0).unwrap(), 1.5),  // Outside [0,1]
    ];
    
    for (description, dist, value) in test_cases {
        let log_prob = dist.log_pdf(value);
        
        if log_prob.is_finite() {
            println!("✅ {}: log_prob = {:.4}", description, log_prob);
        } else {
            println!("⚠️ {}: log_prob = {} (value = {})", description, log_prob, value);
        }
    }
}
```

### Step 3: Trace Analysis for Debugging

Use traces to understand model execution:

```rust
fn debug_with_traces() {
    println!("\n🔬 Trace-Based Debugging");
    println!("========================");
    
    let problematic_model = prob! {
        let sigma <- sample(addr!("sigma"), LogNormal::new(0.0, 2.0).unwrap());
        let mu <- sample(addr!("mu"), Normal::new(0.0, sigma).unwrap());
        
        // This might cause issues if sigma is very large or small
        observe(addr!("obs1"), Normal::new(mu, sigma).unwrap(), 1.0);
        observe(addr!("obs2"), Normal::new(mu, sigma).unwrap(), 100.0);  // Very different scale!
        
        pure((mu, sigma))
    };
    
    let mut rng = StdRng::seed_from_u64(42);
    let ((mu, sigma), trace) = runtime::handler::run(
        runtime::interpreters::PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        problematic_model,
    );
    
    println!("📊 Model Results:");
    println!("   μ = {:.4}, σ = {:.4}", mu, sigma);
    println!("   Prior log-weight: {:.4}", trace.log_prior);
    println!("   Likelihood log-weight: {:.4}", trace.log_likelihood);
    println!("   Total log-weight: {:.4}", trace.total_log_weight());
    
    // Analyze individual choices
    println!("\n🔍 Choice Analysis:");
    for (addr, choice) in &trace.choices {
        println!("   {}: value={:?}, log_prob={:.4}", addr, choice.value, choice.log_prob);
        
        if !choice.log_prob.is_finite() {
            println!("     ⚠️ Infinite/NaN log probability detected!");
        }
        
        if choice.log_prob < -100.0 {
            println!("     ⚠️ Very low probability (< exp(-100))");
        }
    }
    
    // Check for scale mismatches
    if let (Some(obs1_val), Some(obs2_val)) = (trace.choices.get(&addr!("obs1")), trace.choices.get(&addr!("obs2"))) {
        // This is conceptual - observations don't create choices in the trace
        println!("\n📏 Scale Analysis:");
        println!("   Observation 1: 1.0");
        println!("   Observation 2: 100.0");
        println!("   Scale ratio: {:.1}x", 100.0 / 1.0);
        println!("   Model σ: {:.4}", sigma);
        
        if sigma < 0.1 && (1.0 - 100.0).abs() > 10.0 * sigma {
            println!("   ⚠️ Observations have very different scales relative to σ");
        }
    }
}
```

## Advanced Debugging Techniques

### Custom Debug Handler

Create a specialized handler for debugging:

```rust
use std::collections::HashMap;

pub struct DebugHandler<R: rand::Rng> {
    rng: R,
    trace: Trace,
    debug_info: HashMap<Address, DebugEntry>,
    check_numerics: bool,
    log_threshold: f64,
}

#[derive(Debug, Clone)]
struct DebugEntry {
    value: String,
    log_prob: f64,
    is_problematic: bool,
    issues: Vec<String>,
}

impl<R: rand::Rng> DebugHandler<R> {
    pub fn new(rng: R) -> Self {
        Self {
            rng,
            trace: Trace::default(),
            debug_info: HashMap::new(),
            check_numerics: true,
            log_threshold: -50.0,  // Flag very low probabilities
        }
    }
    
    fn check_value(&self, addr: &Address, value: f64, log_prob: f64) -> DebugEntry {
        let mut issues = Vec::new();
        
        if !value.is_finite() {
            issues.push("Value is infinite or NaN".to_string());
        }
        
        if !log_prob.is_finite() {
            issues.push("Log probability is infinite or NaN".to_string());
        } else if log_prob < self.log_threshold {
            issues.push(format!("Very low log probability: {:.2}", log_prob));
        }
        
        if value.abs() > 1e6 {
            issues.push(format!("Very large magnitude: {:.2e}", value));
        }
        
        DebugEntry {
            value: format!("{:.6}", value),
            log_prob,
            is_problematic: !issues.is_empty(),
            issues,
        }
    }
    
    fn report_issues(&self, addr: &Address, entry: &DebugEntry) {
        if entry.is_problematic {
            println!("⚠️ Issues at {}: {}", addr, entry.issues.join(", "));
        }
    }
}

impl<R: rand::Rng> Handler for DebugHandler<R> {
    fn on_sample_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>) -> f64 {
        let value = dist.sample(&mut self.rng);
        let log_prob = dist.log_pdf(value);
        
        let debug_entry = self.check_value(addr, value, log_prob);
        self.report_issues(addr, &debug_entry);
        self.debug_info.insert(addr.clone(), debug_entry);
        
        self.trace.insert_choice(addr.clone(), ChoiceValue::F64(value), log_prob);
        self.trace.log_prior += log_prob;
        
        value
    }
    
    fn on_sample_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>) -> bool {
        let value = dist.sample(&mut self.rng);
        let log_prob = dist.log_pmf(value);
        
        if !log_prob.is_finite() {
            println!("⚠️ Infinite log probability for bool at {}: {}", addr, log_prob);
        }
        
        self.trace.insert_choice(addr.clone(), ChoiceValue::Bool(value), log_prob);
        self.trace.log_prior += log_prob;
        
        value
    }
    
    fn on_sample_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>) -> u64 {
        let value = dist.sample(&mut self.rng);
        let log_prob = dist.log_pmf(value);
        
        if !log_prob.is_finite() {
            println!("⚠️ Infinite log probability for u64 at {}: {}", addr, log_prob);
        }
        
        if value > 1_000_000 {
            println!("⚠️ Very large count at {}: {}", addr, value);
        }
        
        self.trace.insert_choice(addr.clone(), ChoiceValue::U64(value), log_prob);
        self.trace.log_prior += log_prob;
        
        value
    }
    
    fn on_sample_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>) -> usize {
        let value = dist.sample(&mut self.rng);
        let log_prob = dist.log_pmf(value);
        
        if !log_prob.is_finite() {
            println!("⚠️ Infinite log probability for usize at {}: {}", addr, log_prob);
        }
        
        self.trace.insert_choice(addr.clone(), ChoiceValue::Usize(value), log_prob);
        self.trace.log_prior += log_prob;
        
        value
    }
    
    fn on_observe_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>, value: f64) {
        let log_prob = dist.log_pdf(value);
        
        if !log_prob.is_finite() {
            println!("⚠️ Infinite likelihood at observation {}: value={:.4}, log_prob={}", 
                     addr, value, log_prob);
        } else if log_prob < -100.0 {
            println!("⚠️ Very unlikely observation at {}: value={:.4}, log_prob={:.2}", 
                     addr, value, log_prob);
        }
        
        self.trace.log_likelihood += log_prob;
    }
    
    fn on_observe_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>, value: bool) {
        let log_prob = dist.log_pmf(value);
        
        if !log_prob.is_finite() {
            println!("⚠️ Infinite likelihood at bool observation {}: value={}, log_prob={}", 
                     addr, value, log_prob);
        }
        
        self.trace.log_likelihood += log_prob;
    }
    
    fn on_observe_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>, value: u64) {
        let log_prob = dist.log_pmf(value);
        
        if !log_prob.is_finite() {
            println!("⚠️ Infinite likelihood at u64 observation {}: value={}, log_prob={}", 
                     addr, value, log_prob);
        }
        
        self.trace.log_likelihood += log_prob;
    }
    
    fn on_observe_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>, value: usize) {
        let log_prob = dist.log_pmf(value);
        
        if !log_prob.is_finite() {
            println!("⚠️ Infinite likelihood at usize observation {}: value={}, log_prob={}", 
                     addr, value, log_prob);
        }
        
        self.trace.log_likelihood += log_prob;
    }
    
    fn on_factor(&mut self, logw: f64) {
        if !logw.is_finite() {
            println!("⚠️ Infinite factor: {}", logw);
        } else if logw < -100.0 {
            println!("⚠️ Very negative factor: {:.2}", logw);
        }
        
        self.trace.log_factors += logw;
    }
    
    fn finish(self) -> Trace {
        println!("\n🏁 Debug Handler Summary:");
        println!("   Total issues found: {}", 
                 self.debug_info.values().filter(|e| e.is_problematic).count());
        
        let total_log_weight = self.trace.total_log_weight();
        if !total_log_weight.is_finite() {
            println!("   ❌ Final log weight is infinite/NaN: {}", total_log_weight);
        } else if total_log_weight < -1000.0 {
            println!("   ⚠️ Final log weight is very negative: {:.2}", total_log_weight);
        } else {
            println!("   ✅ Final log weight looks reasonable: {:.4}", total_log_weight);
        }
        
        self.trace
    }
}

fn test_debug_handler() {
    println!("\n🐛 Debug Handler Demo");
    println!("====================");
    
    let problematic_model = prob! {
        // This might sample very small or large values
        let scale <- sample(addr!("scale"), LogNormal::new(0.0, 3.0).unwrap());
        
        // This could be problematic if scale is extreme
        let value <- sample(addr!("value"), Normal::new(0.0, scale).unwrap());
        
        // This observation might be very unlikely
        observe(addr!("fixed_obs"), Normal::new(value, 0.001).unwrap(), 1000.0);
        
        pure((scale, value))
    };
    
    let rng = StdRng::seed_from_u64(42);
    let ((scale, value), trace) = runtime::handler::run(
        DebugHandler::new(rng),
        problematic_model,
    );
    
    println!("\n📋 Final Results:");
    println!("   Scale: {:.4}", scale);
    println!("   Value: {:.4}", value);
    println!("   Total log weight: {:.4}", trace.total_log_weight());
}
```

### Model Comparison for Debugging

Compare different model variants to isolate issues:

```rust
fn comparative_debugging() {
    println!("\n🔄 Comparative Debugging");
    println!("=======================");
    
    // Original problematic model
    let original_model = prob! {
        let x <- sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
        let y <- sample(addr!("y"), Normal::new(x, 0.01).unwrap());  // Very tight coupling
        observe(addr!("obs"), Normal::new(y, 0.001).unwrap(), 10.0);  // Unlikely observation
        pure((x, y))
    };
    
    // Simplified version 1: Remove tight coupling
    let simplified_v1 = prob! {
        let x <- sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
        let y <- sample(addr!("y"), Normal::new(x, 1.0).unwrap());    // Looser coupling
        observe(addr!("obs"), Normal::new(y, 0.001).unwrap(), 10.0);
        pure((x, y))
    };
    
    // Simplified version 2: Remove unlikely observation
    let simplified_v2 = prob! {
        let x <- sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
        let y <- sample(addr!("y"), Normal::new(x, 0.01).unwrap());
        observe(addr!("obs"), Normal::new(y, 1.0).unwrap(), 10.0);    // More tolerant observation
        pure((x, y))
    };
    
    let models = vec![
        ("Original", original_model),
        ("Looser coupling", simplified_v1),
        ("Tolerant observation", simplified_v2),
    ];
    
    for (name, model) in models {
        let mut rng = StdRng::seed_from_u64(42);
        let ((x, y), trace) = runtime::handler::run(
            runtime::interpreters::PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            model,
        );
        
        let log_weight = trace.total_log_weight();
        let status = if log_weight.is_finite() && log_weight > -100.0 {
            "✅ OK"
        } else {
            "⚠️ Problematic"
        };
        
        println!("   {}: x={:.3}, y={:.3}, log_weight={:.2} {}", 
                 name, x, y, log_weight, status);
    }
}
```

## Debugging Inference Issues

### MCMC Mixing Problems

```rust
fn debug_mcmc_mixing() {
    println!("\n🔗 MCMC Mixing Diagnostics");
    println!("==========================");
    
    let model = || prob! {
        let x <- sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
        let y <- sample(addr!("y"), Normal::new(x, 0.1).unwrap());  // Highly correlated
        observe(addr!("obs_x"), Normal::new(x, 0.1).unwrap(), 2.0);
        observe(addr!("obs_y"), Normal::new(y, 0.1).unwrap(), 2.1);
        pure((x, y))
    };
    
    let mut rng = StdRng::seed_from_u64(42);
    
    // Run short MCMC chain with detailed tracking
    let mut samples = Vec::new();
    let mut acceptances = Vec::new();
    
    let (mut current_value, mut current_trace) = runtime::handler::run(
        runtime::interpreters::PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        model(),
    );
    
    let n_samples = 100;
    for i in 0..n_samples {
        let (new_value, new_trace) = inference::mh::single_site_random_walk_mh(
            &mut rng,
            0.1,  // Small step size might cause poor mixing
            model,
            &current_trace,
        );
        
        // Check if accepted (simplified)
        let accepted = (new_trace.total_log_weight() - current_trace.total_log_weight()).abs() > 1e-10 ||
                      (new_value.0 - current_value.0).abs() > 1e-10;
        
        if accepted {
            current_value = new_value;
            current_trace = new_trace;
        }
        
        samples.push(current_value);
        acceptances.push(accepted);
        
        if i < 10 {
            println!("   Step {}: x={:.3}, y={:.3}, accepted={}, log_weight={:.4}", 
                     i, current_value.0, current_value.1, accepted, current_trace.total_log_weight());
        }
    }
    
    // Analyze mixing
    let acceptance_rate = acceptances.iter().filter(|&&x| x).count() as f64 / n_samples as f64;
    println!("   Acceptance rate: {:.1}%", acceptance_rate * 100.0);
    
    if acceptance_rate < 0.2 {
        println!("   ⚠️ Low acceptance rate - try larger step size");
    } else if acceptance_rate > 0.8 {
        println!("   ⚠️ High acceptance rate - try smaller step size");
    } else {
        println!("   ✅ Reasonable acceptance rate");
    }
    
    // Check for sticking
    let x_samples: Vec<f64> = samples.iter().map(|(x, _)| *x).collect();
    let unique_x: std::collections::HashSet<_> = x_samples.iter()
        .map(|x| (x * 1000.0) as i32)  // Discretize for uniqueness check
        .collect();
    
    let exploration = unique_x.len() as f64 / n_samples as f64;
    println!("   Exploration ratio: {:.2}", exploration);
    
    if exploration < 0.1 {
        println!("   ⚠️ Poor exploration - chain might be stuck");
    }
}
```

### Numerical Stability Checks

```rust
fn debug_numerical_stability() {
    println!("\n🔢 Numerical Stability Checks");
    println!("=============================");
    
    // Test extreme parameter values
    let extreme_cases = vec![
        ("Very small sigma", Normal::new(0.0, 1e-10)),
        ("Very large sigma", Normal::new(0.0, 1e10)),
        ("Large mu", Normal::new(1e6, 1.0)),
        ("Reasonable", Normal::new(0.0, 1.0)),
    ];
    
    for (description, dist_result) in extreme_cases {
        match dist_result {
            Ok(dist) => {
                let mut rng = StdRng::seed_from_u64(42);
                let sample = dist.sample(&mut rng);
                let log_prob = dist.log_pdf(sample);
                
                println!("   {}: sample={:.2e}, log_prob={:.2e}", description, sample, log_prob);
                
                if !sample.is_finite() || !log_prob.is_finite() {
                    println!("     ⚠️ Numerical instability detected");
                }
            }
            Err(e) => {
                println!("   {}: ❌ {}", description, e);
            }
        }
    }
    
    // Test log-space arithmetic
    println!("\n📊 Log-space Arithmetic:");
    let log_probs = vec![-1000.0, -100.0, -10.0, -1.0, 0.0];
    
    for &lp in &log_probs {
        let prob = lp.exp();
        let back_to_log = prob.ln();
        
        println!("   log_prob={:.1}, exp={:.2e}, ln(exp)={:.1}", lp, prob, back_to_log);
        
        if (lp - back_to_log).abs() > 1e-10 && lp < -700.0 {
            println!("     ⚠️ Precision loss in log-space conversion");
        }
    }
}
```

## Model Validation Techniques

### Posterior Predictive Checks

```rust
fn posterior_predictive_checks() {
    println!("\n🎯 Posterior Predictive Checks");
    println!("==============================");
    
    let observed_data = vec![2.1, 2.3, 1.9, 2.4, 2.0];
    
    let model = |data: &[f64]| prob! {
        let mu <- sample(addr!("mu"), Normal::new(0.0, 2.0).unwrap());
        let sigma <- sample(addr!("sigma"), LogNormal::new(0.0, 0.5).unwrap());
        
        for (i, &obs) in data.iter().enumerate() {
            observe(addr!("obs", i), Normal::new(mu, sigma).unwrap(), obs);
        }
        
        pure((mu, sigma))
    };
    
    // Generate posterior samples
    let mut rng = StdRng::seed_from_u64(42);
    let n_samples = 100;
    let mut posterior_samples = Vec::new();
    
    for i in 0..n_samples {
        let mut sample_rng = StdRng::seed_from_u64(42 + i as u64);
        let ((mu, sigma), _) = runtime::handler::run(
            runtime::interpreters::PriorHandler {
                rng: &mut sample_rng,
                trace: Trace::default(),
            },
            model(&observed_data),
        );
        posterior_samples.push((mu, sigma));
    }
    
    // Generate predictive data
    let mut predictive_datasets = Vec::new();
    for (mu, sigma) in &posterior_samples {
        let mut pred_data = Vec::new();
        for _ in 0..observed_data.len() {
            let pred_value = Normal::new(*mu, *sigma).unwrap().sample(&mut rng);
            pred_data.push(pred_value);
        }
        predictive_datasets.push(pred_data);
    }
    
    // Compute test statistics
    let observed_mean = observed_data.iter().sum::<f64>() / observed_data.len() as f64;
    let observed_std = {
        let variance = observed_data.iter()
            .map(|x| (x - observed_mean).powi(2))
            .sum::<f64>() / (observed_data.len() - 1) as f64;
        variance.sqrt()
    };
    
    println!("   Observed data: {:?}", observed_data);
    println!("   Observed mean: {:.3}", observed_mean);
    println!("   Observed std: {:.3}", observed_std);
    
    // Compare with predictive statistics
    let pred_means: Vec<f64> = predictive_datasets.iter()
        .map(|data| data.iter().sum::<f64>() / data.len() as f64)
        .collect();
    
    let pred_mean_mean = pred_means.iter().sum::<f64>() / pred_means.len() as f64;
    let extreme_means = pred_means.iter()
        .filter(|&&mean| (mean - observed_mean).abs() > (observed_mean - pred_mean_mean).abs())
        .count();
    
    let p_value = extreme_means as f64 / pred_means.len() as f64;
    
    println!("   Predictive mean of means: {:.3}", pred_mean_mean);
    println!("   P-value for mean: {:.3}", p_value);
    
    if p_value < 0.05 || p_value > 0.95 {
        println!("   ⚠️ Observed mean is unusual under the model");
    } else {
        println!("   ✅ Observed mean is consistent with model");
    }
}
```

### Cross-Validation

```rust
fn cross_validation_check() {
    println!("\n🔀 Cross-Validation Check");
    println!("=========================");
    
    let full_data = vec![1.8, 2.1, 2.3, 1.9, 2.4, 2.0, 2.2, 1.7];
    
    let mut log_scores = Vec::new();
    
    // Leave-one-out cross-validation
    for holdout_idx in 0..full_data.len() {
        let training_data: Vec<f64> = full_data.iter()
            .enumerate()
            .filter(|(i, _)| *i != holdout_idx)
            .map(|(_, &x)| x)
            .collect();
        
        let holdout_value = full_data[holdout_idx];
        
        // Fit model on training data
        let model = prob! {
            let mu <- sample(addr!("mu"), Normal::new(0.0, 2.0).unwrap());
            let sigma <- sample(addr!("sigma"), LogNormal::new(0.0, 0.5).unwrap());
            
            for (i, &obs) in training_data.iter().enumerate() {
                observe(addr!("obs", i), Normal::new(mu, sigma).unwrap(), obs);
            }
            
            pure((mu, sigma))
        };
        
        // Sample from posterior
        let mut rng = StdRng::seed_from_u64(42 + holdout_idx as u64);
        let ((mu, sigma), _) = runtime::handler::run(
            runtime::interpreters::PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            model,
        );
        
        // Compute log score for holdout
        let log_score = Normal::new(mu, sigma).unwrap().log_pdf(holdout_value);
        log_scores.push(log_score);
        
        println!("   Holdout {}: value={:.2}, μ={:.3}, σ={:.3}, log_score={:.3}", 
                 holdout_idx, holdout_value, mu, sigma, log_score);
    }
    
    let avg_log_score = log_scores.iter().sum::<f64>() / log_scores.len() as f64;
    println!("   Average log score: {:.3}", avg_log_score);
    
    if avg_log_score < -10.0 {
        println!("   ⚠️ Poor predictive performance");
    } else {
        println!("   ✅ Reasonable predictive performance");
    }
}
```

## Common Debugging Patterns

### Checklist for Model Issues

```rust
fn model_debugging_checklist() {
    println!("\n✅ Model Debugging Checklist");
    println!("============================");
    
    let checks = vec![
        "1. Do all distributions have valid parameters?",
        "2. Are observations within distribution support?",
        "3. Are log-weights finite throughout execution?",
        "4. Do extreme parameter values cause numerical issues?",
        "5. Are there scale mismatches between parameters?",
        "6. Does the model make sense probabilistically?",
        "7. Are addresses unique and stable?",
        "8. Does inference converge reasonably?",
        "9. Are posterior predictive checks reasonable?",
        "10. Does cross-validation show good predictive performance?",
    ];
    
    for check in checks {
        println!("   {}", check);
    }
}
```

### Quick Diagnostic Function

```rust
fn quick_model_diagnostic<M: Clone>(model: M, name: &str) 
where
    M: Fn() -> Model<f64>,
{
    println!("\n🩺 Quick Diagnostic: {}", name);
    println!("{}", "=".repeat(20 + name.len()));
    
    let mut issues = Vec::new();
    
    // Test 1: Basic execution
    let mut rng = StdRng::seed_from_u64(42);
    let execution_result = std::panic::catch_unwind(|| {
        runtime::handler::run(
            runtime::interpreters::PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            model(),
        )
    });
    
    match execution_result {
        Ok((value, trace)) => {
            println!("   ✅ Model executes successfully");
            
            // Test 2: Check log weights
            let log_weight = trace.total_log_weight();
            if !log_weight.is_finite() {
                issues.push("Infinite/NaN log weight".to_string());
            } else if log_weight < -1000.0 {
                issues.push(format!("Very negative log weight: {:.2}", log_weight));
            }
            
            // Test 3: Check returned value
            if !value.is_finite() {
                issues.push("Infinite/NaN return value".to_string());
            }
            
            println!("   Value: {:.4}, Log weight: {:.4}", value, log_weight);
            
        }
        Err(_) => {
            issues.push("Model execution panicked".to_string());
        }
    }
    
    // Test 4: Multiple runs for consistency
    let mut values = Vec::new();
    for seed in 0..5 {
        if let Ok((value, _)) = std::panic::catch_unwind(|| {
            let mut rng = StdRng::seed_from_u64(seed);
            runtime::handler::run(
                runtime::interpreters::PriorHandler {
                    rng: &mut rng,
                    trace: Trace::default(),
                },
                model(),
            )
        }) {
            values.push(value);
        }
    }
    
    if values.len() < 5 {
        issues.push("Inconsistent execution across seeds".to_string());
    } else {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        
        println!("   Multi-run stats: mean={:.3}, range=[{:.3}, {:.3}]", mean, min_val, max_val);
        
        if (max_val - min_val) > 1e6 {
            issues.push("Extremely wide range of values".to_string());
        }
    }
    
    // Summary
    if issues.is_empty() {
        println!("   🎉 No issues detected!");
    } else {
        println!("   ⚠️ Issues found:");
        for issue in issues {
            println!("      - {}", issue);
        }
    }
}
```

## Complete Debugging Example

Here's a complete workflow for debugging a problematic model:

```rust
fn complete_debugging_example() {
    println!("🐛 Complete Debugging Example");
    println!("=============================");
    
    // Problematic model with multiple issues
    let problematic_model = || prob! {
        // Issue 1: Very small sigma can cause numerical problems
        let sigma <- sample(addr!("sigma"), LogNormal::new(-5.0, 0.1).unwrap());
        
        // Issue 2: Extreme correlation
        let x <- sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
        let y <- sample(addr!("y"), Normal::new(x, sigma).unwrap());
        
        // Issue 3: Inconsistent scale in observations
        observe(addr!("obs1"), Normal::new(x, 0.1).unwrap(), 0.1);
        observe(addr!("obs2"), Normal::new(y, 0.1).unwrap(), 1000.0);  // Very different scale!
        
        pure(x)
    };
    
    // Step 1: Quick diagnostic
    quick_model_diagnostic(problematic_model, "Problematic Model");
    
    // Step 2: Detailed debug run
    println!("\n🔍 Detailed Debug Analysis:");
    let rng = StdRng::seed_from_u64(42);
    let (value, trace) = runtime::handler::run(
        DebugHandler::new(rng),
        problematic_model(),
    );
    
    // Step 3: Create fixed version
    let fixed_model = || prob! {
        // Fix 1: More reasonable sigma range
        let sigma <- sample(addr!("sigma"), LogNormal::new(0.0, 0.5).unwrap());
        
        let x <- sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
        let y <- sample(addr!("y"), Normal::new(x, sigma).unwrap());
        
        // Fix 2: Consistent scale in observations
        observe(addr!("obs1"), Normal::new(x, 0.1).unwrap(), 0.1);
        observe(addr!("obs2"), Normal::new(y, 0.1).unwrap(), 0.2);
        
        pure(x)
    };
    
    // Step 4: Compare results
    println!("\n🔄 Before/After Comparison:");
    quick_model_diagnostic(fixed_model, "Fixed Model");
}

fn main() {
    test_isolation();
    test_distributions();
    test_observations();
    debug_with_traces();
    test_debug_handler();
    comparative_debugging();
    debug_mcmc_mixing();
    debug_numerical_stability();
    posterior_predictive_checks();
    cross_validation_check();
    model_debugging_checklist();
    complete_debugging_example();
    
    println!("\n🎯 Debugging tutorial completed!");
}
```

## Key Takeaways

1. **Systematic approach** - Follow a structured debugging workflow
2. **Isolate problems** - Start with simple cases and build up complexity
3. **Use specialized handlers** - Custom handlers provide detailed diagnostic information
4. **Check numerics carefully** - Watch for infinite/NaN values and extreme scales
5. **Validate models** - Use posterior predictive checks and cross-validation
6. **Compare variants** - Test different model formulations to isolate issues
7. **Monitor inference** - Check MCMC mixing and convergence diagnostics

## Next Steps

- **[Linear Regression Tutorial](../tutorials/linear-regression.md)** - Apply debugging to a real model
- **[Custom Handlers](custom-handlers.md)** - Build more sophisticated debugging tools
- **[Trace Manipulation](trace-manipulation.md)** - Advanced trace analysis techniques

---

**Ready for a complex tutorial?** → **[Linear Regression Tutorial](../tutorials/linear-regression.md)**