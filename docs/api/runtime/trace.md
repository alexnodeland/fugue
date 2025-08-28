# Execution Trace System

## Overview

Fugue's trace system is the **foundational data structure** that makes probabilistic programming possible. It solves the central challenge: **how to record, manipulate, and reason about the execution history of probabilistic models**.

Every time a probabilistic model runs, it generates a **trace**—a complete record of all random choices made and log-weights accumulated. This trace is not just a passive record; it's an active data structure that enables:

- **Replay**: Re-executing models with the same random choices (essential for MCMC)
- **Scoring**: Computing log-probabilities of specific execution paths (importance sampling)
- **Conditioning**: Fixing some variables while marginalizing over others
- **Debugging**: Understanding model behavior and weight contributions
- **Inference**: All advanced algorithms build on trace manipulation

The system provides **three core types** that work together:

- **`ChoiceValue`**: Type-safe storage for values from different distribution types
- **`Choice`**: A single recorded decision (address + value + log-probability)  
- **`Trace`**: Complete execution history with accumulated log-weights

The key architectural insight is the **three-component log-weight decomposition**: every trace separates prior probabilities, observation likelihoods, and explicit factors, enabling sophisticated inference algorithms to reason about different sources of probability mass.

## Usage Examples

### Basic Trace Inspection

```rust
# use fugue::*;
# use fugue::runtime::interpreters::PriorHandler;
# use rand::rngs::StdRng;
# use rand::SeedableRng;

// Execute a model and examine its trace structure
let model = sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap())
    .bind(|mu| {
        observe(addr!("y1"), Normal::new(mu, 0.5).unwrap(), 1.2)
            .bind(move |_| observe(addr!("y2"), Normal::new(mu, 0.5).unwrap(), 1.8))
            .bind(move |_| factor(-0.5)) // Manual log-weight adjustment
            .map(move |_| mu)
    });

let mut rng = StdRng::seed_from_u64(42);
let (result, trace) = runtime::handler::run(
    PriorHandler { rng: &mut rng, trace: Trace::default() },
    model
);

// Examine the trace structure
println!("Sampled mu: {:.3}", result);
println!("Number of choices: {}", trace.choices.len());
println!("Prior log-weight: {:.3}", trace.log_prior);
println!("Likelihood log-weight: {:.3}", trace.log_likelihood);  
println!("Factor log-weight: {:.3}", trace.log_factors);
println!("Total log-weight: {:.3}", trace.total_log_weight());

// Access individual choices
if let Some(choice) = trace.choices.get(&addr!("mu")) {
    println!("Mu choice: {:?} (logp: {:.3})", choice.value, choice.logp);
}

// Type-safe value access
let mu_value = trace.get_f64(&addr!("mu")).unwrap();
println!("Retrieved mu: {:.3}", mu_value);
```

### Trace Manipulation for MCMC

```rust
# use fugue::*;
# use fugue::runtime::interpreters::*;
# use rand::rngs::StdRng;
# use rand::{Rng, SeedableRng};

// Create a model for MCMC
let make_model = || {
    sample(addr!("theta"), Normal::new(0.0, 1.0).unwrap())
        .bind(|theta| {
            let observations = vec![1.1, 1.3, 0.9];
            let obs_models = observations.into_iter().enumerate().map(|(i, y)| {
                observe(addr!("obs", i), Normal::new(theta, 0.2).unwrap(), y)
            }).collect::<Vec<_>>();
            sequence_vec(obs_models).map(move |_| theta)
        })
};

let mut rng = StdRng::seed_from_u64(123);

// 1. Generate initial trace
let (_, current_trace) = runtime::handler::run(
    PriorHandler { rng: &mut rng, trace: Trace::default() },
    make_model()
);

// 2. Create MCMC proposal by modifying the trace
let mut proposal_trace = current_trace.clone();
let current_theta = proposal_trace.get_f64(&addr!("theta")).unwrap();
let proposed_theta = current_theta + rng.gen::<f64>() * 0.1 - 0.05; // Random walk

// Update the trace with the proposal
proposal_trace.insert_choice(
    addr!("theta"), 
    ChoiceValue::F64(proposed_theta), 
    Normal::new(0.0, 1.0).unwrap().log_prob(&proposed_theta)
);

// 3. Score both traces under the model
let (_, current_scored) = runtime::handler::run(
    ScoreGivenTrace { base: current_trace.clone(), trace: Trace::default() },
    make_model()
);

let (_, proposal_scored) = runtime::handler::run(
    ScoreGivenTrace { base: proposal_trace.clone(), trace: Trace::default() },
    make_model()
);

// 4. Compute acceptance ratio
let current_weight = current_scored.total_log_weight();
let proposal_weight = proposal_scored.total_log_weight();
let log_alpha = proposal_weight - current_weight;
let alpha = log_alpha.exp().min(1.0);

println!("Current theta: {:.3} (weight: {:.3})", current_theta, current_weight);
println!("Proposed theta: {:.3} (weight: {:.3})", proposed_theta, proposal_weight);
println!("Acceptance probability: {:.3}", alpha);

// Accept/reject based on alpha
let accept = rng.gen::<f64>() < alpha;
let final_trace = if accept { proposal_trace } else { current_trace };
println!("Proposal {}", if accept { "accepted" } else { "rejected" });
```

### Type-Safe Value Access Patterns

```rust
# use fugue::*;
# use fugue::runtime::trace::*;

// Create a trace with different value types
let mut trace = Trace::default();
trace.insert_choice(addr!("continuous"), ChoiceValue::F64(3.14), -0.5);
trace.insert_choice(addr!("discrete"), ChoiceValue::U64(42), -1.2);
trace.insert_choice(addr!("categorical"), ChoiceValue::Usize(2), -0.8);
trace.insert_choice(addr!("binary"), ChoiceValue::Bool(true), -0.6);

// Option-based access (returns None on type mismatch)
assert_eq!(trace.get_f64(&addr!("continuous")), Some(3.14));
assert_eq!(trace.get_u64(&addr!("discrete")), Some(42));
assert_eq!(trace.get_usize(&addr!("categorical")), Some(2));
assert_eq!(trace.get_bool(&addr!("binary")), Some(true));

// Type mismatches return None
assert_eq!(trace.get_bool(&addr!("continuous")), None);

// Result-based access (returns detailed errors)
match trace.get_f64_result(&addr!("continuous")) {
    Ok(val) => println!("Got f64: {}", val),
    Err(e) => println!("Error: {}", e),
}

// Handle missing addresses
match trace.get_f64_result(&addr!("missing")) {
    Ok(_) => unreachable!(),
    Err(e) => println!("Missing address error: {}", e),
}

// Handle type mismatches  
match trace.get_bool_result(&addr!("continuous")) {
    Ok(_) => unreachable!(),
    Err(e) => println!("Type mismatch error: {}", e),
}
```

### Log-Weight Analysis and Debugging

```rust
# use fugue::*;
# use fugue::runtime::interpreters::PriorHandler;
# use rand::rngs::StdRng;
# use rand::SeedableRng;

// Complex model with multiple weight sources
let diagnostic_model = || {
    sample(addr!("prior1"), Normal::new(0.0, 2.0).unwrap())
        .bind(|x1| sample(addr!("prior2"), Normal::new(x1, 1.0).unwrap())
            .bind(move |x2| {
                observe(addr!("obs1"), Normal::new(x2, 0.1).unwrap(), 1.5)
                    .bind(move |_| observe(addr!("obs2"), Normal::new(x2, 0.2).unwrap(), 1.7))
                    .bind(move |_| factor(if x2.abs() < 2.0 { 0.0 } else { f64::NEG_INFINITY }))
                    .map(move |_| (x1, x2))
            }))
};

let mut rng = StdRng::seed_from_u64(456);
let (result, trace) = runtime::handler::run(
    PriorHandler { rng: &mut rng, trace: Trace::default() },
    diagnostic_model()
);

// Detailed weight breakdown
println!("Model result: {:?}", result);
println!("\nTrace diagnostics:");
println!("├─ Choices recorded: {}", trace.choices.len());
println!("├─ Prior log-weight: {:.6}", trace.log_prior);
println!("├─ Likelihood log-weight: {:.6}", trace.log_likelihood);
println!("├─ Factor log-weight: {:.6}", trace.log_factors);
println!("└─ Total log-weight: {:.6}", trace.total_log_weight());

// Per-choice analysis
println!("\nChoice breakdown:");
for (addr, choice) in &trace.choices {
    println!("  {}: {:?} (logp: {:.6})", addr, choice.value, choice.logp);
}

// Validity checks
if trace.total_log_weight().is_finite() {
    println!("\n✓ Trace is valid (finite log-weight)");
} else {
    println!("\n✗ Trace is invalid (infinite log-weight)");
    if trace.log_factors.is_infinite() {
        println!("  └─ Rejection likely due to factor statement");
    }
}
```

### Trace Comparison and Importance Weighting

```rust
# use fugue::*;
# use fugue::runtime::interpreters::*;
# use rand::rngs::StdRng;
# use rand::SeedableRng;

// Compare execution traces from different model parameterizations
let make_target_model = |mu: f64, sigma: f64| move || {
    sample(addr!("x"), Normal::new(mu, sigma).unwrap())
};

let make_proposal_model = |mu: f64, sigma: f64| move || {
    sample(addr!("x"), Normal::new(mu, sigma).unwrap())
};

let mut rng = StdRng::seed_from_u64(789);
let mut importance_weights = Vec::new();

// Generate importance samples
for i in 0..20 {
    // Sample from proposal (broad distribution)
    let (value, proposal_trace) = runtime::handler::run(
        PriorHandler { rng: &mut rng, trace: Trace::default() },
        make_proposal_model(0.0, 2.0)()
    );
    
    // Score under target (narrow distribution)
    let (_, target_trace) = runtime::handler::run(
        ScoreGivenTrace { base: proposal_trace.clone(), trace: Trace::default() },
        make_target_model(1.0, 0.5)()
    );
    
    // Compute importance weight
    let log_weight = target_trace.log_prior - proposal_trace.log_prior;
    importance_weights.push((value, log_weight, proposal_trace.clone(), target_trace));
    
    println!("Sample {}: x={:.3}, log_weight={:.3}", i, value, log_weight);
}

// Analyze importance weights
let max_log_weight = importance_weights.iter()
    .map(|(_, w, _, _)| *w)
    .fold(f64::NEG_INFINITY, f64::max);

let normalized_weights: Vec<f64> = importance_weights.iter()
    .map(|(_, w, _, _)| (w - max_log_weight).exp())
    .collect();

let weight_sum: f64 = normalized_weights.iter().sum();
let effective_sample_size = weight_sum.powi(2) / 
    normalized_weights.iter().map(|w| w.powi(2)).sum::<f64>();

println!("\nImportance sampling diagnostics:");
println!("Effective sample size: {:.1} / {}", effective_sample_size, importance_weights.len());
println!("Weight efficiency: {:.1}%", 100.0 * effective_sample_size / importance_weights.len() as f64);
```

### Advanced Trace Manipulation

```rust
# use fugue::*;
# use fugue::runtime::trace::*;

// Build traces programmatically for testing/debugging
let mut custom_trace = Trace::default();

// Add choices of different types
custom_trace.insert_choice(addr!("mu"), ChoiceValue::F64(1.5), -0.125);
custom_trace.insert_choice(addr!("n_trials"), ChoiceValue::U64(20), -2.996);
custom_trace.insert_choice(addr!("success"), ChoiceValue::Bool(true), -0.693);

// Set log-weight components explicitly
custom_trace.log_prior = -3.814; // Sum of choice log-probabilities
custom_trace.log_likelihood = -5.2; // From observations
custom_trace.log_factors = 0.5; // Manual adjustments

println!("Custom trace total weight: {:.3}", custom_trace.total_log_weight());

// Clone and modify for counterfactual analysis
let mut modified_trace = custom_trace.clone();
modified_trace.insert_choice(addr!("mu"), ChoiceValue::F64(2.0), -0.5);

println!("Original mu: {:?}", custom_trace.get_f64(&addr!("mu")));
println!("Modified mu: {:?}", modified_trace.get_f64(&addr!("mu")));

// Trace merging (for advanced algorithms)
let mut merged_trace = Trace::default();
for (addr, choice) in custom_trace.choices.iter() {
    merged_trace.choices.insert(addr.clone(), choice.clone());
}
merged_trace.log_prior = custom_trace.log_prior;
merged_trace.log_likelihood = custom_trace.log_likelihood;
merged_trace.log_factors = custom_trace.log_factors;

assert_eq!(merged_trace.total_log_weight(), custom_trace.total_log_weight());
```

## Design & Evolution

### Status

- **Stable**: The trace system has been stable since v0.1 and forms the foundation of the runtime
- **Complete**: Supports all value types needed for probabilistic programming
- **Extensible**: New value types can be added to `ChoiceValue` without breaking compatibility
- **Performance Critical**: Optimized for frequent access patterns in inference algorithms

### Key Design Principles

1. **Separation of Concerns**: Traces record execution history; interpreters define execution strategy
2. **Type Safety**: All value access is type-checked, preventing runtime errors from type mismatches
3. **Decomposed Log-Weights**: Prior, likelihood, and factors are tracked separately for algorithmic flexibility
4. **Efficient Access**: BTreeMap provides O(log n) lookups with ordered iteration
5. **Memory Efficiency**: Copy-on-write and pooling strategies (see memory module) optimize allocation patterns

### Architectural Decisions

#### Three-Component Log-Weight Structure

The decision to decompose total log-weight into `log_prior + log_likelihood + log_factors` enables:

- **Importance Sampling**: Compare proposal and target priors separately
- **MCMC**: Compute acceptance ratios using only relevant components  
- **Model Comparison**: Isolate prior vs. likelihood contributions
- **Debugging**: Identify which component is causing numerical issues

#### Type-Safe Value Storage

`ChoiceValue` provides a unified interface for different distribution return types:

- **Runtime Safety**: No unsafe casting between incompatible types
- **Error Clarity**: Type mismatches produce clear, actionable error messages
- **Future Extensibility**: New value types (e.g., vectors, matrices) can be added seamlessly
- **Performance**: Each variant is optimally sized for its contained type

#### Address-Based Choice Organization

Using `BTreeMap<Address, Choice>` provides:

- **Deterministic Iteration**: Consistent ordering across executions
- **Efficient Lookup**: O(log n) access by address
- **Range Queries**: Can iterate over address prefixes (useful for hierarchical models)
- **Memory Locality**: Better cache behavior than hash-based alternatives

### Evolution Strategy

- **Backwards Compatible**: New `ChoiceValue` variants and `Trace` methods are additive
- **Performance Focused**: Internal optimizations (memory pooling, COW) don't change the API
- **Composable**: Traces work seamlessly with all interpreter and memory optimization strategies

## Error Handling

The trace system provides two levels of error handling for different use cases:

### Option-Based Access (Graceful Degradation)

```rust
# use fugue::*;
# use fugue::runtime::trace::*;

let mut trace = Trace::default();
trace.insert_choice(addr!("x"), ChoiceValue::F64(1.5), -0.5);

// Option-based access returns None on errors
match trace.get_f64(&addr!("x")) {
    Some(val) => println!("Found f64: {}", val),
    None => println!("Address missing or type mismatch"),
}

// Type mismatch returns None
assert_eq!(trace.get_bool(&addr!("x")), None);

// Missing address returns None  
assert_eq!(trace.get_f64(&addr!("missing")), None);
```

### Result-Based Access (Detailed Error Information)

```rust
# use fugue::*;
# use fugue::runtime::trace::*;
# let mut trace = Trace::default();
# trace.insert_choice(addr!("x"), ChoiceValue::F64(1.5), -0.5);

// Result-based access provides detailed error information
match trace.get_bool_result(&addr!("x")) {
    Ok(val) => println!("Got bool: {}", val),
    Err(e) => {
        println!("Detailed error: {}", e);
        // Error contains specific information about expected vs. actual types
    }
}

// Missing address error
match trace.get_f64_result(&addr!("missing")) {
    Ok(_) => unreachable!(),
    Err(e) => {
        println!("Address not found: {}", e);
        // Error contains the specific address that was missing
    }
}
```

### Error Handling Best Practices

```rust
# use fugue::*;
# use fugue::runtime::trace::*;

// Production-safe trace access pattern
fn safe_trace_analysis(trace: &Trace) -> Result<String, String> {
    // Use Result-based access for critical operations
    let mu = trace.get_f64_result(&addr!("mu"))
        .map_err(|e| format!("Failed to get mu: {}", e))?;
    
    // Use Option-based access for optional values
    let sigma = trace.get_f64(&addr!("sigma")).unwrap_or(1.0); // Default fallback
    
    // Check trace validity
    if !trace.total_log_weight().is_finite() {
        return Err("Trace has infinite log-weight".to_string());
    }
    
    Ok(format!("Analysis: mu={:.3}, sigma={:.3}, weight={:.3}", 
               mu, sigma, trace.total_log_weight()))
}

# let mut trace = Trace::default();
# trace.insert_choice(addr!("mu"), ChoiceValue::F64(1.5), -0.5);
# trace.log_prior = -0.5;
# println!("{}", safe_trace_analysis(&trace).unwrap());
```

### Integration Error Patterns

```rust
# use fugue::*;
# use fugue::runtime::interpreters::*;
# use rand::rngs::StdRng;
# use rand::SeedableRng;

// Common error pattern: trace/model mismatch
let mut rng = StdRng::seed_from_u64(42);

// Create trace with f64 value
let (_, trace_f64) = runtime::handler::run(
    PriorHandler { rng: &mut rng, trace: Trace::default() },
    sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
);

// Try to use with incompatible model (expects bool)
let bool_model = sample(addr!("x"), Bernoulli::new(0.5).unwrap());

// Safe scoring handles the mismatch gracefully
let (_, safe_result) = runtime::handler::run(
    SafeScoreGivenTrace {
        base: trace_f64,
        trace: Trace::default(),
        warn_on_error: true, // Enable warnings
    },
    bool_model
);

// Check if scoring failed due to type mismatch
if safe_result.total_log_weight().is_infinite() {
    println!("Scoring failed gracefully - type mismatch detected");
}
```

## Integration Notes

### With Handler System

All trace operations integrate seamlessly with the handler system:

- **Handler Trait**: All handlers consume and produce `Trace` objects
- **Type Safety**: Handlers use type-specific methods (`on_sample_f64`, `on_observe_bool`) that map to appropriate `ChoiceValue` variants
- **Log-Weight Accumulation**: Handlers update the three log-weight components (`log_prior`, `log_likelihood`, `log_factors`) according to their interpretation strategy
- **Address Resolution**: Handlers use the trace's address-based storage to implement replay and scoring modes

### With Memory Optimization

The trace system integrates with memory optimization strategies:

- **Copy-on-Write**: `CowTrace` wraps `Trace` to enable efficient sharing in MCMC
- **Memory Pooling**: `TracePool` pre-allocates `Trace` objects to reduce garbage collection pressure
- **TraceBuilder**: Efficient construction of traces with pre-sized allocations

### With Inference Algorithms

| Algorithm | Trace Usage Pattern | Key Operations |
|---|---|---|
| **MCMC** | Clone current trace, modify specific addresses, score proposals | `clone()`, `insert_choice()`, `total_log_weight()` |
| **SMC** | Generate particle traces, reweight based on observations | `PriorHandler` generation, `ScoreGivenTrace` weighting |
| **VI** | Store samples from variational distribution, compute gradients | Type-safe access, log-weight decomposition |
| **ABC** | Generate traces from prior, compare to observed data | `PriorHandler` generation, custom distance functions |

### Performance Characteristics

- **Address Lookup**: O(log n) via `BTreeMap` - efficient for most probabilistic models
- **Choice Insertion**: O(log n) with potential reallocation
- **Trace Cloning**: O(n) but optimized with COW strategies in memory module  
- **Type Access**: O(log n + constant) for address lookup plus O(1) type extraction
- **Log-Weight Computation**: O(1) since components are pre-accumulated

### Production Deployment Patterns

```rust
# use fugue::*;
# use fugue::runtime::trace::*;

// Production inference monitoring
#[derive(Debug)]
struct TraceMetrics {
    num_choices: usize,
    log_weight: f64,
    is_valid: bool,
    type_distribution: std::collections::HashMap<&'static str, usize>,
}

impl TraceMetrics {
    fn from_trace(trace: &Trace) -> Self {
        let num_choices = trace.choices.len();
        let log_weight = trace.total_log_weight();
        let is_valid = log_weight.is_finite();
        
        let mut type_distribution = std::collections::HashMap::new();
        for choice in trace.choices.values() {
            *type_distribution.entry(choice.value.type_name()).or_insert(0) += 1;
        }
        
        Self { num_choices, log_weight, is_valid, type_distribution }
    }
}

// Usage in production monitoring
fn monitor_inference_traces(traces: &[Trace]) {
    let metrics: Vec<TraceMetrics> = traces.iter().map(TraceMetrics::from_trace).collect();
    
    let valid_count = metrics.iter().filter(|m| m.is_valid).count();
    let avg_choices = metrics.iter().map(|m| m.num_choices).sum::<usize>() as f64 / metrics.len() as f64;
    let avg_log_weight = metrics.iter()
        .filter(|m| m.is_valid)
        .map(|m| m.log_weight)
        .sum::<f64>() / valid_count as f64;
    
    println!("Trace diagnostics:");
    println!("├─ Valid traces: {} / {} ({:.1}%)", valid_count, traces.len(), 
             100.0 * valid_count as f64 / traces.len() as f64);
    println!("├─ Avg choices per trace: {:.1}", avg_choices);
    println!("└─ Avg log-weight: {:.3}", avg_log_weight);
}
```

## Reference Links

### Core Types

- [`ChoiceValue`](../trace.rs) - Type-safe storage for values from different distributions
- [`Choice`](../trace.rs) - Single recorded decision with address, value, and log-probability  
- [`Trace`](../trace.rs) - Complete execution history with decomposed log-weights

### Related Systems

- [`Handler`](../handler.md) - How interpreters consume and produce traces
- [`Memory Optimization`](../memory.md) - Efficient trace allocation and sharing strategies
- [`Interpreters`](../interpreters.md) - How different execution modes use traces

### Usage Guides

- [MCMC Implementation](../../src/how-to/mcmc-implementation.md) - Using traces for proposal and scoring
- [Trace Debugging](../../src/how-to/trace-debugging.md) - Analyzing model behavior through traces
- [Production Monitoring](../../src/how-to/production-monitoring.md) - Trace health monitoring patterns

### Examples

- [`trace_manipulation.rs`](../../../examples/trace_manipulation.rs) - Basic trace operations
- [`mcmc_traces.rs`](../../../examples/mcmc_traces.rs) - MCMC with trace manipulation
- [`importance_sampling_traces.rs`](../../../examples/importance_sampling_traces.rs) - Trace-based importance sampling
- [`trace_debugging.rs`](../../../examples/trace_debugging.rs) - Debugging model execution with traces
