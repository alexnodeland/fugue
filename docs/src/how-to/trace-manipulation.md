# Trace Manipulation

Understanding and manipulating execution traces is crucial for debugging models, analyzing inference, and implementing advanced algorithms. This guide covers trace structure, manipulation techniques, and practical debugging workflows.

## What Are Traces?

A `Trace` records everything that happens during model execution:
- **Choices**: Random variable assignments with their log-probabilities
- **Log-weights**: Accumulated prior, likelihood, and factor contributions
- **Addresses**: Named locations where random choices were made

Think of traces as "execution recordings" that can be replayed, analyzed, and modified.

```rust
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

// Simple model for demonstration
fn simple_model(observation: f64) -> Model<f64> {
    prob! {
        let mu <- sample(addr!("mu"), Normal::new(0.0, 2.0).unwrap());
        observe(addr!("y"), Normal::new(mu, 1.0).unwrap(), observation);
        pure(mu)
    }
}

fn trace_basics() {
    let model = simple_model(2.5);
    let mut rng = StdRng::seed_from_u64(42);
    
    let (value, trace) = runtime::handler::run(
        runtime::interpreters::PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        model,
    );
    
    println!("🔍 Trace Analysis:");
    println!("  Returned value: {:.4}", value);
    println!("  Prior log-weight: {:.4}", trace.log_prior);
    println!("  Likelihood log-weight: {:.4}", trace.log_likelihood);
    println!("  Factor log-weight: {:.4}", trace.log_factors);
    println!("  Total log-weight: {:.4}", trace.total_log_weight());
    println!("  Number of choices: {}", trace.choices.len());
}
```

## Trace Structure

### Choice Values

Each choice in a trace stores:
- The sampled value
- The log-probability density/mass at that value

```rust
fn examining_choices() {
    let model = prob! {
        let x <- sample(addr!("continuous"), Normal::new(0.0, 1.0).unwrap());
        let coin <- sample(addr!("discrete"), Bernoulli::new(0.7).unwrap());
        let count <- sample(addr!("count"), Poisson::new(3.0).unwrap());
        pure((x, coin, count))
    };
    
    let mut rng = StdRng::seed_from_u64(42);
    let (_, trace) = runtime::handler::run(
        runtime::interpreters::PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        model,
    );
    
    println!("📋 Choice Details:");
    for (addr, choice) in &trace.choices {
        println!("  {}: value={:?}, log_prob={:.4}", 
                 addr, choice.value, choice.log_prob);
    }
}
```

### Log-Weight Components

Traces separate different types of log-weights:

```rust
fn weight_components() {
    let model = prob! {
        let mu <- sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap());  // Prior
        observe(addr!("y"), Normal::new(mu, 0.5).unwrap(), 2.0);       // Likelihood
        factor(-0.5);                                                   // Factor
        pure(mu)
    };
    
    let mut rng = StdRng::seed_from_u64(42);
    let (_, trace) = runtime::handler::run(
        runtime::interpreters::PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        model,
    );
    
    println!("⚖️ Weight Breakdown:");
    println!("  Prior contributions: {:.4}", trace.log_prior);
    println!("  Likelihood contributions: {:.4}", trace.log_likelihood);
    println!("  Factor contributions: {:.4}", trace.log_factors);
    println!("  Total: {:.4}", trace.total_log_weight());
    println!("  Probability ratio: {:.6}", trace.total_log_weight().exp());
}
```

## Trace Manipulation Techniques

### Reading Values from Traces

```rust
fn reading_trace_values() {
    let model = prob! {
        let mu <- sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap());
        let sigma <- sample(addr!("sigma"), LogNormal::new(0.0, 0.5).unwrap());
        let coin <- sample(addr!("coin"), Bernoulli::new(0.6).unwrap());
        let count <- sample(addr!("events"), Poisson::new(4.0).unwrap());
        pure((mu, sigma, coin, count))
    };
    
    let mut rng = StdRng::seed_from_u64(42);
    let (_, trace) = runtime::handler::run(
        runtime::interpreters::PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        model,
    );
    
    // Type-safe value extraction
    if let Some(mu) = trace.get_f64(&addr!("mu")) {
        println!("μ = {:.4}", mu);
    }
    
    if let Some(coin_result) = trace.get_bool(&addr!("coin")) {
        println!("Coin: {}", if coin_result { "Heads" } else { "Tails" });
    }
    
    if let Some(event_count) = trace.get_u64(&addr!("events")) {
        println!("Events: {}", event_count);
    }
    
    // Generic access
    if let Some(choice) = trace.choices.get(&addr!("mu")) {
        println!("Raw choice: {:?}", choice);
    }
}
```

### Manual Trace Construction

```rust
fn manual_trace_construction() {
    let mut trace = Trace::default();
    
    // Insert choices manually
    trace.insert_choice(addr!("mu"), ChoiceValue::F64(1.5), -0.5);
    trace.insert_choice(addr!("sigma"), ChoiceValue::F64(0.8), -0.2);
    trace.insert_choice(addr!("success"), ChoiceValue::Bool(true), -0.6);
    trace.insert_choice(addr!("count"), ChoiceValue::U64(3), -1.2);
    
    // Add weight components
    trace.log_prior += -1.5;
    trace.log_likelihood += -2.3;
    trace.log_factors += -0.8;
    
    println!("🔧 Manual Trace:");
    println!("  Total choices: {}", trace.choices.len());
    println!("  Total log-weight: {:.4}", trace.total_log_weight());
    
    // Use manually constructed trace with replay handler
    let model = prob! {
        let mu <- sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap());
        let sigma <- sample(addr!("sigma"), LogNormal::new(0.0, 0.5).unwrap());
        pure((mu, sigma))
    };
    
    let mut rng = StdRng::seed_from_u64(42);
    let ((mu, sigma), replayed_trace) = runtime::handler::run(
        runtime::interpreters::ReplayHandler {
            rng: &mut rng,
            base: trace,
            trace: Trace::default(),
        },
        model,
    );
    
    println!("  Replayed values: μ={:.4}, σ={:.4}", mu, sigma);
}
```

## Advanced Trace Operations

### Trace Comparison and Difference Analysis

```rust
fn trace_comparison() {
    let model = simple_model(2.0);
    
    // Generate two different traces
    let (_, trace1) = runtime::handler::run(
        runtime::interpreters::PriorHandler {
            rng: &mut StdRng::seed_from_u64(42),
            trace: Trace::default(),
        },
        model.clone(),
    );
    
    let (_, trace2) = runtime::handler::run(
        runtime::interpreters::PriorHandler {
            rng: &mut StdRng::seed_from_u64(123),
            trace: Trace::default(),
        },
        model,
    );
    
    println!("🔄 Trace Comparison:");
    println!("  Trace 1 log-weight: {:.4}", trace1.total_log_weight());
    println!("  Trace 2 log-weight: {:.4}", trace2.total_log_weight());
    
    // Compare specific values
    let mu1 = trace1.get_f64(&addr!("mu")).unwrap();
    let mu2 = trace2.get_f64(&addr!("mu")).unwrap();
    println!("  μ₁ = {:.4}, μ₂ = {:.4}, diff = {:.4}", mu1, mu2, mu1 - mu2);
    
    // Log-weight ratio (for MCMC acceptance)
    let log_ratio = trace2.total_log_weight() - trace1.total_log_weight();
    let acceptance_prob = log_ratio.exp().min(1.0);
    println!("  Acceptance probability: {:.4}", acceptance_prob);
}
```

### Trace Filtering and Transformation

```rust
fn trace_filtering() {
    let model = prob! {
        let params <- plate!(i in 0..5 => {
            sample(addr!("param", i), Normal::new(0.0, 1.0).unwrap())
        });
        
        for (i, &param) in params.iter().enumerate() {
            observe(addr!("obs", i), Normal::new(param, 0.1).unwrap(), i as f64 * 0.5);
        }
        
        pure(params)
    };
    
    let mut rng = StdRng::seed_from_u64(42);
    let (_, trace) = runtime::handler::run(
        runtime::interpreters::PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        model,
    );
    
    // Extract all parameter values
    let param_values: Vec<f64> = (0..5)
        .filter_map(|i| trace.get_f64(&addr!("param", i)))
        .collect();
    
    println!("🔍 Parameter Analysis:");
    println!("  Parameters: {:?}", param_values);
    println!("  Mean: {:.4}", param_values.iter().sum::<f64>() / param_values.len() as f64);
    println!("  Max: {:.4}", param_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)));
    println!("  Min: {:.4}", param_values.iter().fold(f64::INFINITY, |a, &b| a.min(b)));
    
    // Filter choices by type
    let f64_choices: Vec<_> = trace.choices
        .iter()
        .filter(|(_, choice)| matches!(choice.value, ChoiceValue::F64(_)))
        .collect();
    
    println!("  Number of f64 choices: {}", f64_choices.len());
}
```

### Trace Modification and Counterfactuals

```rust
fn trace_modification() {
    let original_obs = 2.0;
    let model = simple_model(original_obs);
    
    // Generate base trace
    let mut rng = StdRng::seed_from_u64(42);
    let (_, mut base_trace) = runtime::handler::run(
        runtime::interpreters::PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        model.clone(),
    );
    
    println!("🔬 Counterfactual Analysis:");
    println!("  Original μ: {:.4}", base_trace.get_f64(&addr!("mu")).unwrap());
    println!("  Original log-weight: {:.4}", base_trace.total_log_weight());
    
    // Modify the mu parameter to a specific value
    let new_mu = 1.5;
    base_trace.insert_choice(
        addr!("mu"), 
        ChoiceValue::F64(new_mu), 
        Normal::new(0.0, 2.0).unwrap().log_pdf(new_mu)
    );
    
    // Re-score the modified trace
    let modified_model = simple_model(original_obs);
    let (_, rescored_trace) = runtime::handler::run(
        runtime::interpreters::ScoreGivenTrace {
            base: base_trace,
            trace: Trace::default(),
        },
        modified_model,
    );
    
    println!("  Modified μ: {:.4}", rescored_trace.get_f64(&addr!("mu")).unwrap());
    println!("  Modified log-weight: {:.4}", rescored_trace.total_log_weight());
    
    // What if we had observed a different value?
    let counterfactual_obs = 5.0;
    let counterfactual_model = simple_model(counterfactual_obs);
    let (_, cf_trace) = runtime::handler::run(
        runtime::interpreters::ReplayHandler {
            rng: &mut rng,
            base: rescored_trace.clone(),
            trace: Trace::default(),
        },
        counterfactual_model,
    );
    
    println!("  Counterfactual obs: {:.4}", counterfactual_obs);
    println!("  Counterfactual log-weight: {:.4}", cf_trace.total_log_weight());
}
```

## Debugging Workflows

### Model Execution Debugging

```rust
fn debug_model_execution() {
    let problematic_model = prob! {
        let x <- sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
        let y <- sample(addr!("y"), Normal::new(x, 0.1).unwrap());
        
        // This might cause issues if y is outside expected range
        if y > 10.0 {
            factor(f64::NEG_INFINITY);  // Reject completely
        }
        
        observe(addr!("z"), Normal::new(y * y, 0.1).unwrap(), 4.0);
        pure((x, y))
    };
    
    println!("🐛 Debugging Model Execution:");
    
    for seed in 0..5 {
        let mut rng = StdRng::seed_from_u64(seed);
        let ((x, y), trace) = runtime::handler::run(
            runtime::interpreters::PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            problematic_model.clone(),
        );
        
        let log_weight = trace.total_log_weight();
        let is_problematic = log_weight.is_infinite() || log_weight.is_nan();
        
        println!("  Seed {}: x={:.3}, y={:.3}, log_weight={:.4} {}", 
                 seed, x, y, log_weight,
                 if is_problematic { "⚠️" } else { "✅" });
        
        if is_problematic {
            println!("    Components: prior={:.4}, likelihood={:.4}, factors={:.4}",
                     trace.log_prior, trace.log_likelihood, trace.log_factors);
        }
    }
}
```

### Inference Quality Assessment

```rust
fn assess_inference_quality() {
    let model = simple_model(2.5);
    
    // Collect many traces
    let mut traces = Vec::new();
    for seed in 0..1000 {
        let mut rng = StdRng::seed_from_u64(seed);
        let (_, trace) = runtime::handler::run(
            runtime::interpreters::PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            model.clone(),
        );
        traces.push(trace);
    }
    
    // Extract parameter values
    let mu_values: Vec<f64> = traces
        .iter()
        .filter_map(|t| t.get_f64(&addr!("mu")))
        .collect();
    
    // Compute diagnostics
    let mean = mu_values.iter().sum::<f64>() / mu_values.len() as f64;
    let variance = mu_values.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / (mu_values.len() - 1) as f64;
    
    println!("📊 Inference Quality Assessment:");
    println!("  Sample size: {}", mu_values.len());
    println!("  Sample mean: {:.4}", mean);
    println!("  Sample std: {:.4}", variance.sqrt());
    
    // Check for outliers
    let median = {
        let mut sorted = mu_values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted[sorted.len() / 2]
    };
    
    let mad = {  // Median absolute deviation
        let mut deviations: Vec<f64> = mu_values.iter()
            .map(|x| (x - median).abs())
            .collect();
        deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());
        deviations[deviations.len() / 2]
    };
    
    println!("  Median: {:.4}", median);
    println!("  MAD: {:.4}", mad);
    
    // Identify potential outliers (> 3 MAD from median)
    let outliers: Vec<f64> = mu_values.iter()
        .filter(|&&x| (x - median).abs() > 3.0 * mad)
        .copied()
        .collect();
    
    if !outliers.is_empty() {
        println!("  ⚠️ Potential outliers: {:?}", outliers);
    } else {
        println!("  ✅ No significant outliers detected");
    }
}
```

### Trace Validation

```rust
fn validate_traces() {
    let model = prob! {
        let mu <- sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap());
        let sigma <- sample(addr!("sigma"), LogNormal::new(0.0, 0.5).unwrap());
        observe(addr!("y"), Normal::new(mu, sigma).unwrap(), 2.0);
        pure((mu, sigma))
    };
    
    let mut rng = StdRng::seed_from_u64(42);
    let (_, trace) = runtime::handler::run(
        runtime::interpreters::PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        model,
    );
    
    println!("✅ Trace Validation:");
    
    // Check for required addresses
    let required_addresses = vec![addr!("mu"), addr!("sigma"), addr!("y")];
    for addr in &required_addresses {
        if trace.choices.contains_key(addr) {
            println!("  ✅ Address '{}' present", addr);
        } else {
            println!("  ❌ Address '{}' missing", addr);
        }
    }
    
    // Check for finite log-weights
    let weights_finite = trace.log_prior.is_finite() && 
                         trace.log_likelihood.is_finite() && 
                         trace.log_factors.is_finite();
    
    if weights_finite {
        println!("  ✅ All log-weights are finite");
    } else {
        println!("  ❌ Some log-weights are infinite/NaN");
        println!("    Prior: {:.4}", trace.log_prior);
        println!("    Likelihood: {:.4}", trace.log_likelihood);
        println!("    Factors: {:.4}", trace.log_factors);
    }
    
    // Check value ranges
    if let Some(sigma) = trace.get_f64(&addr!("sigma")) {
        if sigma > 0.0 {
            println!("  ✅ Sigma is positive: {:.4}", sigma);
        } else {
            println!("  ❌ Sigma is non-positive: {:.4}", sigma);
        }
    }
    
    // Validate trace consistency by re-scoring
    let (_, rescored_trace) = runtime::handler::run(
        runtime::interpreters::ScoreGivenTrace {
            base: trace.clone(),
            trace: Trace::default(),
        },
        model,
    );
    
    let weight_diff = (trace.total_log_weight() - rescored_trace.total_log_weight()).abs();
    if weight_diff < 1e-10 {
        println!("  ✅ Trace is self-consistent");
    } else {
        println!("  ❌ Trace inconsistency: diff={:.2e}", weight_diff);
    }
}
```

## Performance Optimization

### Efficient Trace Operations

```rust
// ❌ Inefficient: Repeated lookups
fn inefficient_trace_access(trace: &Trace) -> f64 {
    let mut sum = 0.0;
    for i in 0..1000 {
        if let Some(value) = trace.get_f64(&addr!("param", i)) {
            sum += value;
        }
    }
    sum
}

// ✅ Efficient: Iterate once
fn efficient_trace_access(trace: &Trace) -> f64 {
    trace.choices
        .iter()
        .filter_map(|(addr, choice)| {
            if addr.0.starts_with("param") {
                choice.value.as_f64()
            } else {
                None
            }
        })
        .sum()
}

// Pre-compute addresses for repeated access
fn precomputed_addresses() -> Vec<Address> {
    (0..1000).map(|i| addr!("param", i)).collect()
}
```

### Memory-Efficient Trace Handling

```rust
fn memory_efficient_trace_processing() {
    let model = plate!(i in 0..1000 => {
        sample(addr!("x", i), Normal::new(0.0, 1.0).unwrap())
    });
    
    let mut rng = StdRng::seed_from_u64(42);
    let (_, trace) = runtime::handler::run(
        runtime::interpreters::PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        model,
    );
    
    // Process trace incrementally instead of storing all values
    let mut sum = 0.0;
    let mut count = 0;
    
    for (addr, choice) in &trace.choices {
        if let ChoiceValue::F64(value) = choice.value {
            sum += value;
            count += 1;
        }
    }
    
    let mean = sum / count as f64;
    println!("🚀 Efficient processing: mean = {:.4} from {} values", mean, count);
    
    // Clear trace if no longer needed (saves memory)
    drop(trace);
}
```

## Real-World Example: MCMC Diagnostics

Here's a complete example showing how to use traces for MCMC diagnostics:

```rust
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn mcmc_diagnostics_example() {
    let observation = 2.5;
    let model = || simple_model(observation);
    
    // Run MCMC chain and collect traces
    let mut rng = StdRng::seed_from_u64(42);
    let mut traces = Vec::new();
    
    // Initial state
    let (mut current_value, mut current_trace) = runtime::handler::run(
        runtime::interpreters::PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        model(),
    );
    
    // MCMC iterations
    let n_samples = 1000;
    let mut n_accepted = 0;
    
    for i in 0..n_samples {
        // Propose new state
        let (new_value, new_trace) = inference::mh::single_site_random_walk_mh(
            &mut rng,
            0.5,  // step size
            model,
            &current_trace,
        );
        
        // Check if proposal was accepted
        let log_ratio = new_trace.total_log_weight() - current_trace.total_log_weight();
        if log_ratio >= 0.0 || rng.gen::<f64>().ln() < log_ratio {
            current_value = new_value;
            current_trace = new_trace;
            n_accepted += 1;
        }
        
        traces.push(current_trace.clone());
        
        if i % 200 == 0 {
            println!("Iteration {}: μ = {:.4}, log_weight = {:.4}", 
                     i, current_value, current_trace.total_log_weight());
        }
    }
    
    // Diagnostics
    let acceptance_rate = n_accepted as f64 / n_samples as f64;
    println!("\n📈 MCMC Diagnostics:");
    println!("  Acceptance rate: {:.1}%", acceptance_rate * 100.0);
    
    // Extract parameter samples
    let mu_samples: Vec<f64> = traces
        .iter()
        .filter_map(|t| t.get_f64(&addr!("mu")))
        .collect();
    
    // Compute effective sample size (simplified)
    let ess = estimate_ess(&mu_samples);
    println!("  Effective sample size: {:.1}", ess);
    
    // Trace plot (conceptual - would need actual plotting library)
    println!("  First 10 μ samples: {:?}", 
             &mu_samples[..10.min(mu_samples.len())]);
    
    // Convergence assessment
    let warmup = mu_samples.len() / 4;
    let first_half_mean = mu_samples[warmup..mu_samples.len()/2].iter().sum::<f64>() / (mu_samples.len()/2 - warmup) as f64;
    let second_half_mean = mu_samples[mu_samples.len()/2..].iter().sum::<f64>() / (mu_samples.len()/2) as f64;
    let mean_diff = (first_half_mean - second_half_mean).abs();
    
    println!("  First half mean: {:.4}", first_half_mean);
    println!("  Second half mean: {:.4}", second_half_mean);
    println!("  |Difference|: {:.4}", mean_diff);
    
    if mean_diff < 0.1 {
        println!("  ✅ Chain appears to have converged");
    } else {
        println!("  ⚠️ Chain may not have converged");
    }
}

// Simplified ESS estimation
fn estimate_ess(samples: &[f64]) -> f64 {
    let n = samples.len() as f64;
    // This is a very simplified ESS estimate
    // Real implementation would compute autocorrelation
    n / (1.0 + 2.0 * 0.5) // Assuming autocorrelation ~ 0.5
}
```

## Key Takeaways

1. **Traces record everything** - Choices, log-weights, and execution history
2. **Type-safe access** - Use `get_f64()`, `get_bool()`, etc. for safe value extraction
3. **Replay and modification** - Traces enable counterfactual analysis and debugging
4. **Debugging workflows** - Use traces to identify model problems and assess inference quality
5. **Performance matters** - Be efficient with trace operations in hot paths
6. **Validation is crucial** - Always check trace consistency and finite log-weights

## Next Steps

- **[Custom Handlers](custom-handlers.md)** - Implement your own trace-aware interpreters
- **[Debugging Models](debugging-models.md)** - Advanced debugging techniques
- **[Mixture Models Tutorial](../tutorials/mixture-models.md)** - See trace manipulation in action

---

**Ready to build custom handlers?** → **[Custom Handlers](custom-handlers.md)**