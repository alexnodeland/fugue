# Trace Manipulation Tutorial

**Level: Advanced** | **Time: 40 minutes**

Master advanced trace operations in Fugue! This tutorial covers the runtime system's powerful trace manipulation capabilities for debugging, counterfactual reasoning, and efficient MCMC implementations.

## Learning Objectives

By the end of this tutorial, you'll understand:

- Trace structure and components (choices, log weights, addresses)
- Deterministic replay with different observations
- Trace scoring for parameter configurations
- Manual trace modification for counterfactual analysis
- How MCMC algorithms use trace manipulation internally
- Debugging techniques for complex models

## What Are Traces?

A **trace** records the execution of a probabilistic program:

```rust
pub struct Trace {
    pub choices: HashMap<Address, Choice>,  // Random choices made
    pub log_prior: f64,                     // Sum of prior log probabilities  
    pub log_likelihood: f64,                // Sum of observation log probabilities
    pub log_factors: f64,                   // Additional scoring factors
}
```

Every choice stores:
```rust
pub struct Choice {
    pub value: ChoiceValue,  // The sampled value
    pub logp: f64,          // Log probability of this choice
}
```

## Basic Trace Operations

**Try it**: Run with `cargo run --example trace_manipulation`

```rust
{{#include ../../../examples/trace_manipulation.rs}}
```

## Understanding Trace Components

Let's break down what happens when you run the example:

### 1. Prior Sampling
```
Prior Trace:
  Choices:
    mu: 1.2345 (logp: -2.3456)
    sigma: 0.8901 (logp: -0.4567)
  Log prior: -2.8023
  Log likelihood: -3.4567
  Total log weight: -6.2590
```

**Key insight**: The trace captures **every random choice** and its probability.

### 2. Replay with Different Observations
```
Replayed Trace:
  Same choices but different likelihood!
  Notice: mu, sigma unchanged but likelihood changed!
```

**Key insight**: Replay uses the **same random choices** but evaluates them against new data.

### 3. Trace Scoring
```
Scored Trace:
  Should match trace1 exactly!
```

**Key insight**: Scoring evaluates a specific parameter configuration without sampling.

## Trace Manipulation Patterns

### Pattern 1: Deterministic Replay

```rust
fn analyze_parameter_sensitivity(model_fn: ModelFn, base_trace: Trace) {
    let test_observations = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    
    for &obs in &test_observations {
        let (result, new_trace) = runtime::handler::run(
            runtime::interpreters::ReplayHandler {
                rng: &mut rng,
                base: base_trace.clone(),
                trace: Trace::default(),
            },
            model_fn(obs)
        );
        
        println!("Observation {}: log weight = {:.3}", obs, new_trace.total_log_weight());
    }
}
```

**Use case**: Test how parameter values perform on different datasets.

### Pattern 2: Counterfactual Analysis

```rust
fn counterfactual_analysis(original_trace: &Trace) {
    let mut modified_trace = original_trace.clone();
    
    // Change a parameter value
    if let Some(choice) = modified_trace.choices.get_mut(&addr!("treatment_effect")) {
        let original_value = match choice.value {
            ChoiceValue::F64(v) => v,
            _ => panic!("Expected f64"),
        };
        
        // "What if the treatment effect was twice as large?"
        let counterfactual_value = original_value * 2.0;
        choice.value = ChoiceValue::F64(counterfactual_value);
        choice.logp = Normal::new(0.0, 1.0).unwrap().log_prob(&counterfactual_value);
        
        println!("Original effect: {:.3}", original_value);
        println!("Counterfactual effect: {:.3}", counterfactual_value);
        
        // Evaluate counterfactual scenario
        let (result, scored_trace) = runtime::handler::run(
            ScoreGivenTrace {
                base: modified_trace,
                trace: Trace::default(),
            },
            model
        );
        
        println!("Counterfactual log weight: {:.3}", scored_trace.total_log_weight());
    }
}
```

**Use case**: "What if" analysis - explore alternative parameter values.

### Pattern 3: Manual Trace Construction

```rust
fn construct_trace_manually() -> Trace {
    let mut trace = Trace::default();
    
    // Add specific parameter values
    trace.insert_choice(
        addr!("slope"), 
        ChoiceValue::F64(2.5),
        Normal::new(0.0, 1.0).unwrap().log_prob(&2.5)
    );
    
    trace.insert_choice(
        addr!("intercept"),
        ChoiceValue::F64(1.0),
        Normal::new(0.0, 10.0).unwrap().log_prob(&1.0)  
    );
    
    trace
}
```

**Use case**: Test specific parameter combinations or initialize MCMC chains.

## Advanced Trace Operations

### Trace Comparison and Diagnostics

```rust
fn compare_traces(trace1: &Trace, trace2: &Trace) {
    println!("Trace Comparison:");
    println!("=================");
    
    // Compare log weights
    let weight_diff = trace1.total_log_weight() - trace2.total_log_weight();
    println!("Log weight difference: {:.4}", weight_diff);
    
    if weight_diff > 0.0 {
        println!("Trace 1 is more probable");
    } else {
        println!("Trace 2 is more probable");
    }
    
    // Compare individual choices
    for (addr, choice1) in &trace1.choices {
        if let Some(choice2) = trace2.choices.get(addr) {
            match (&choice1.value, &choice2.value) {
                (ChoiceValue::F64(v1), ChoiceValue::F64(v2)) => {
                    println!("{}: {:.3} vs {:.3} (diff: {:.3})", addr, v1, v2, v1 - v2);
                },
                _ => println!("{}: different types", addr),
            }
        } else {
            println!("{}: only in trace 1", addr);
        }
    }
}
```

### Trace Validation

```rust
fn validate_trace(trace: &Trace, model_fn: ModelFn) -> bool {
    // Re-score the trace to check consistency
    let (_, rescored_trace) = runtime::handler::run(
        ScoreGivenTrace {
            base: trace.clone(),
            trace: Trace::default(),
        },
        model_fn
    );
    
    let weight_diff = (trace.total_log_weight() - rescored_trace.total_log_weight()).abs();
    let is_valid = weight_diff < 1e-10;
    
    if !is_valid {
        println!("⚠️ Trace validation failed!");
        println!("  Original weight: {:.6}", trace.total_log_weight());
        println!("  Rescored weight: {:.6}", rescored_trace.total_log_weight());
        println!("  Difference: {:.6}", weight_diff);
    }
    
    is_valid
}
```

## Trace-Based Debugging

### Finding Problematic Parameters

```rust
fn debug_low_probability_trace(trace: &Trace) {
    println!("🐛 Debug: Low Probability Trace");
    println!("================================");
    
    // Find choices with very low probabilities
    let mut low_prob_choices = Vec::new();
    
    for (addr, choice) in &trace.choices {
        if choice.logp < -10.0 {  // Very unlikely choice
            low_prob_choices.push((addr, choice.logp));
        }
    }
    
    low_prob_choices.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    
    println!("Choices with low probability:");
    for (addr, logp) in &low_prob_choices {
        println!("  {}: log prob = {:.3}", addr, logp);
    }
    
    // Check component contributions
    println!("\nComponent breakdown:");
    println!("  Log prior: {:.3}", trace.log_prior);
    println!("  Log likelihood: {:.3}", trace.log_likelihood);
    println!("  Log factors: {:.3}", trace.log_factors);
    
    if trace.log_likelihood < -50.0 {
        println!("⚠️ Very poor fit to data - check model specification");
    }
    
    if trace.log_prior < -20.0 {
        println!("⚠️ Parameters far from prior - check prior specification");
    }
}
```

### Model Structure Analysis

```rust
fn analyze_model_structure(traces: &[Trace]) {
    println!("📊 Model Structure Analysis");
    println!("===========================");
    
    // Find all addresses used
    let mut all_addresses = std::collections::HashSet::new();
    for trace in traces {
        for addr in trace.choices.keys() {
            all_addresses.insert(addr.clone());
        }
    }
    
    println!("Parameters in model:");
    for addr in &all_addresses {
        let count = traces.iter()
            .filter(|trace| trace.choices.contains_key(addr))
            .count();
        
        println!("  {}: present in {}/{} traces", addr, count, traces.len());
        
        if count < traces.len() {
            println!("    ⚠️ Not present in all traces - conditional parameter?");
        }
    }
    
    // Analyze parameter correlations
    for addr1 in &all_addresses {
        for addr2 in &all_addresses {
            if addr1 < addr2 {  // Avoid duplicates
                let correlation = compute_correlation(traces, addr1, addr2);
                if correlation.abs() > 0.8 {
                    println!("High correlation between {} and {}: {:.3}", addr1, addr2, correlation);
                }
            }
        }
    }
}
```

## MCMC and Trace Manipulation

Understanding how MCMC uses traces helps you debug inference:

### Metropolis-Hastings Step
```rust
fn mh_step_explained(current_trace: &Trace, model_fn: ModelFn) {
    // 1. Propose new state by modifying trace
    let mut proposed_trace = current_trace.clone();
    
    // Perturb one parameter
    let addr = addr!("mu");
    if let Some(choice) = proposed_trace.choices.get_mut(&addr) {
        if let ChoiceValue::F64(current_value) = choice.value {
            let proposal_std = 0.1;
            let proposed_value = current_value + Normal::new(0.0, proposal_std).unwrap().sample(&mut rng);
            
            choice.value = ChoiceValue::F64(proposed_value);
            choice.logp = Normal::new(0.0, 1.0).unwrap().log_prob(&proposed_value);
        }
    }
    
    // 2. Score proposed state
    let (_, scored_trace) = runtime::handler::run(
        ScoreGivenTrace {
            base: proposed_trace,
            trace: Trace::default(),
        },
        model_fn
    );
    
    // 3. Accept/reject based on log weight difference  
    let log_accept_prob = scored_trace.total_log_weight() - current_trace.total_log_weight();
    let accept_prob = log_accept_prob.exp().min(1.0);
    
    println!("MH Step:");
    println!("  Current log weight: {:.3}", current_trace.total_log_weight());
    println!("  Proposed log weight: {:.3}", scored_trace.total_log_weight());
    println!("  Accept probability: {:.3}", accept_prob);
    
    if rand::random::<f64>() < accept_prob {
        println!("  ✅ Accepted");
    } else {
        println!("  ❌ Rejected");
    }
}
```

## Best Practices

### 1. Always Validate Modified Traces
```rust
// ✅ Good - validate after modification
let mut modified_trace = original_trace.clone();
modify_trace(&mut modified_trace);
assert!(validate_trace(&modified_trace, model));
```

### 2. Use Appropriate Handlers
```rust
// ✅ Replay for same parameter values, different data
let replayed = ReplayHandler { base: trace, .. };

// ✅ Score for evaluating specific configurations
let scored = ScoreGivenTrace { base: trace, .. };

// ✅ Prior for generating fresh samples
let fresh = PriorHandler { .. };
```

### 3. Handle Missing Choices Gracefully
```rust
// ✅ Safe access with fallbacks
let param = trace.get_f64(&addr!("param")).unwrap_or(default_value);

// ❌ Dangerous - will panic if missing
let param = trace.choices[&addr!("param")];
```

## Debugging Workflow

When models behave unexpectedly:

1. **Examine trace structure** - What parameters exist?
2. **Check log weights** - Are they reasonable?  
3. **Compare traces** - How do they differ?
4. **Validate consistency** - Do traces re-score correctly?
5. **Analyze correlations** - Are parameters related as expected?

```rust
fn debug_workflow(problematic_trace: &Trace, model_fn: ModelFn) {
    // Step 1: Structure
    println!("1. Trace structure:");
    for (addr, choice) in &problematic_trace.choices {
        println!("  {}: {:?} (logp: {:.3})", addr, choice.value, choice.logp);
    }
    
    // Step 2: Weights
    println!("\n2. Log weights:");
    debug_low_probability_trace(problematic_trace);
    
    // Step 3: Validation
    println!("\n3. Validation:");
    let is_valid = validate_trace(problematic_trace, model_fn);
    println!("  Valid: {}", is_valid);
}
```

## Applications

Trace manipulation enables:

- **Model debugging** - Understand why inference fails
- **Counterfactual analysis** - Explore "what if" scenarios  
- **Efficient MCMC** - Custom proposal mechanisms
- **Model comparison** - Evaluate specific parameter settings
- **Sensitivity analysis** - Test parameter robustness
- **Initialization** - Start MCMC from good configurations

## Next Steps

Now that you understand trace manipulation:

1. **[Advanced Inference](advanced-inference.md)** - Apply trace techniques to sophisticated models
2. **[Type Safety Features](type-safety-features.md)** - Combine with safe trace operations
3. **[Custom Handlers](../how-to/custom-handlers.md)** - Build your own trace manipulators

## Key Takeaways

- **Traces record everything** - Random choices, probabilities, observations
- **Deterministic replay** - Same choices, different observations
- **Counterfactual reasoning** - Explore alternative parameter values  
- **Debugging tool** - Understand model behavior and failures
- **MCMC foundation** - How inference algorithms work internally

Master trace manipulation to unlock advanced probabilistic programming techniques!

---

**Ready for sophisticated inference methods?** → **[Advanced Inference Tutorial](advanced-inference.md)**
