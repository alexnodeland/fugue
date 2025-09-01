# Built-in Model Interpreters

## Overview

Fugue's interpreter system solves a fundamental challenge in probabilistic programming: **how to execute the same model in radically different ways**. The same `Model<A>` description can be interpreted for prior sampling, trace replay, likelihood scoring, or error-resilient inference—all without changing a single line of model code.

This system provides five **foundational interpreters** that form the building blocks for all probabilistic inference algorithms:

- **`PriorHandler`**: Forward sampling from prior distributions (the baseline interpretation)
- **`ReplayHandler`**: Deterministic replay using existing trace values (essential for MCMC)  
- **`ScoreGivenTrace`**: Log-probability computation for fixed traces (importance sampling, model comparison)
- **`SafeReplayHandler`**: Error-resilient replay with graceful type mismatch handling
- **`SafeScoreGivenTrace`**: Error-resilient scoring with invalid trace handling

The key insight is the **strict/safe duality**: strict interpreters (Replay/Score) panic on inconsistencies for correctness, while safe variants handle errors gracefully for production robustness.

## Usage Examples

### Prior Sampling: The Foundation

```rust
# use fugue::*;
# use fugue::runtime::interpreters::*;
# use rand::rngs::StdRng;
# use rand::SeedableRng;

// Define model once
let model = sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap())
    .bind(|mu| {
        let observations = vec![1.2, 1.5, 1.1];
        let obs_models = observations.into_iter().enumerate().map(|(i, y)| {
            observe(addr!("y", i), Normal::new(mu, 0.1).unwrap(), y)
        }).collect::<Vec<_>>();
        sequence_vec(obs_models).map(move |_| mu)
    });

// Prior sampling: generate random executions
let mut rng = StdRng::seed_from_u64(42);
let (mu_sample, prior_trace) = runtime::handler::run(
    PriorHandler { 
        rng: &mut rng, 
        trace: Trace::default() 
    },
    model
);

println!("Prior sample: mu = {:.3}", mu_sample);
println!("Log-likelihood: {:.3}", prior_trace.log_likelihood);
println!("Total log-weight: {:.3}", prior_trace.total_log_weight());
```

### MCMC Workflow: Replay + Scoring

```rust
# use fugue::*;
# use fugue::runtime::interpreters::*;
# use rand::rngs::StdRng;
# use rand::SeedableRng;

// Create the model function for reuse
let make_model = || {
    sample(addr!("theta"), Normal::new(0.0, 1.0).unwrap())
        .bind(|theta| {
            observe(addr!("y"), Normal::new(theta, 0.5).unwrap(), 2.1)
                .map(move |_| theta)
        })
};

let mut rng = StdRng::seed_from_u64(123);

// 1. Generate initial state
let (_, current_trace) = runtime::handler::run(
    PriorHandler { rng: &mut rng, trace: Trace::default() },
    make_model()
);

// 2. MCMC step: modify one address, replay others
let mut proposal_trace = current_trace.clone();
// Modify theta (in practice this would be a proper MCMC proposal)
proposal_trace.insert_choice(addr!("theta"), ChoiceValue::F64(0.5), -0.125);

// 3. Score the proposal under the model
let (_, proposal_scored) = runtime::handler::run(
    ScoreGivenTrace { 
        base: proposal_trace, 
        trace: Trace::default() 
    },
    make_model()
);

// 4. Accept/reject based on log-weights (simplified)
let current_weight = current_trace.total_log_weight();
let proposal_weight = proposal_scored.total_log_weight();
let accept_prob = (proposal_weight - current_weight).exp().min(1.0);

println!("Current weight: {:.3}", current_weight);
println!("Proposal weight: {:.3}", proposal_weight);
println!("Accept probability: {:.3}", accept_prob);
```

### Production-Safe Inference

```rust
# use fugue::*;
# use fugue::runtime::interpreters::*;
# use rand::rngs::StdRng;
# use rand::SeedableRng;

// Model that might have trace inconsistencies in production
let robust_model = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
    .bind(|x| sample(addr!("y"), Bernoulli::new(0.5).unwrap()).map(move |y| (x, y)));

let mut rng = StdRng::seed_from_u64(456);

// Create a trace with potential type mismatches
let (_, base_trace) = runtime::handler::run(
    PriorHandler { rng: &mut rng, trace: Trace::default() },
    sample(addr!("x"), Normal::new(0.0, 1.0).unwrap()) // Only has f64, missing bool
);

// Safe replay: handles missing addresses gracefully
let (result, safe_trace) = runtime::handler::run(
    SafeReplayHandler {
        rng: &mut rng,
        base: base_trace.clone(),
        trace: Trace::default(),
        warn_on_mismatch: true, // Log warnings for debugging
    },
    robust_model
);

println!("Safe replay succeeded: {:?}", result);
println!("Trace is valid: {}", safe_trace.total_log_weight().is_finite());

// Safe scoring: returns -∞ instead of panicking on type mismatches
let different_model = sample(addr!("x"), Bernoulli::new(0.3).unwrap()); // Expects bool, trace has f64

let (_, error_trace) = runtime::handler::run(
    SafeScoreGivenTrace {
        base: base_trace,
        trace: Trace::default(),
        warn_on_error: true,
    },
    different_model
);

// Check if scoring failed gracefully
if error_trace.total_log_weight().is_infinite() {
    println!("Trace scoring failed gracefully (returned -∞)");
} else {
    println!("Trace scoring succeeded");
}
```

### Multi-Chain Parallel Sampling

```rust
# use fugue::*;
# use fugue::runtime::interpreters::*;
# use rand::rngs::StdRng;
# use rand::SeedableRng;
# use std::collections::HashMap;

// Define a hierarchical model
let hierarchical_model = || {
    sample(addr!("global_mu"), Normal::new(0.0, 2.0).unwrap())
        .bind(|global_mu| {
            let local_samples: Vec<Model<f64>> = (0..5).map(|i| {
                sample(addr!("local", i), Normal::new(global_mu, 0.5).unwrap())
            }).collect();
            sequence_vec(local_samples).map(move |locals| (global_mu, locals))
        })
};

// Run multiple independent chains
let mut chains = HashMap::new();
for chain_id in 0..4 {
    let mut rng = StdRng::seed_from_u64(100 + chain_id as u64);
    let (result, trace) = runtime::handler::run(
        PriorHandler { rng: &mut rng, trace: Trace::default() },
        hierarchical_model()
    );
    chains.insert(chain_id, (result, trace));
}

// Analyze convergence across chains
for (chain_id, ((global_mu, _locals), trace)) in &chains {
    println!("Chain {}: global_mu = {:.3}, log_weight = {:.3}", 
        chain_id, global_mu, trace.total_log_weight());
}
```

### Importance Sampling with Scoring

```rust
# use fugue::*;
# use fugue::runtime::interpreters::*;
# use rand::rngs::StdRng;
# use rand::SeedableRng;

// Define target and proposal models
let target_model = || sample(addr!("x"), Normal::new(2.0, 1.0).unwrap());
let proposal_model = || sample(addr!("x"), Normal::new(0.0, 2.0).unwrap());

let mut rng = StdRng::seed_from_u64(789);
let mut importance_weights = Vec::new();

// Generate importance samples
for _ in 0..100 {
    // 1. Sample from proposal distribution
    let (value, proposal_trace) = runtime::handler::run(
        PriorHandler { rng: &mut rng, trace: Trace::default() },
        proposal_model()
    );
    
    // 2. Score under target distribution
    let (_, target_trace) = runtime::handler::run(
        ScoreGivenTrace { 
            base: proposal_trace.clone(), 
            trace: Trace::default() 
        },
        target_model()
    );
    
    // 3. Compute importance weight: target_prob / proposal_prob
    let log_weight = target_trace.log_prior - proposal_trace.log_prior;
    importance_weights.push((value, log_weight));
}

// Compute effective sample size and other diagnostics
let max_log_weight = importance_weights.iter().map(|(_, w)| *w).fold(f64::NEG_INFINITY, f64::max);
let normalized_weights: Vec<f64> = importance_weights.iter()
    .map(|(_, w)| (w - max_log_weight).exp())
    .collect();

let weight_sum: f64 = normalized_weights.iter().sum();
let ess = weight_sum.powi(2) / normalized_weights.iter().map(|w| w.powi(2)).sum::<f64>();

println!("Effective sample size: {:.1} / {}", ess, importance_weights.len());
```

## Design & Evolution

### Status

- **Stable**: All five interpreter types are stable since v0.1 and form the foundation of the inference system
- **Complete**: These interpreters cover all fundamental execution modes needed for probabilistic programming
- **Composable**: Interpreters can be combined and extended for complex inference algorithms

### Key Design Principles

1. **Separation of Model and Interpretation**: Models describe computations, interpreters define execution strategy
2. **Type Safety**: All interpreters preserve the type safety guarantees of the distribution system
3. **Error Handling Strategy**: Strict interpreters fail fast for correctness, safe variants handle errors gracefully
4. **Performance**: Zero-cost abstractions with compile-time dispatch through the Handler trait
5. **Completeness**: Cover the three fundamental execution modes (sampling, replay, scoring)

### Interpreter Architecture

#### The Strict/Safe Duality

| Execution Mode | Strict Variant | Safe Variant | Use Case |
|---|---|---|---|
| **Trace Replay** | `ReplayHandler` | `SafeReplayHandler` | MCMC proposals vs. production robustness |
| **Trace Scoring** | `ScoreGivenTrace` | `SafeScoreGivenTrace` | Exact computation vs. error resilience |

#### Error Handling Philosophy

- **Strict Interpreters** (ReplayHandler, ScoreGivenTrace):
  - Panic on missing addresses or type mismatches
  - Guarantee correctness when traces are valid
  - Ideal for algorithm development and testing

- **Safe Interpreters** (SafeReplayHandler, SafeScoreGivenTrace):
  - Handle errors gracefully with fallback behavior
  - Continue execution with warnings/logging
  - Essential for production systems with data inconsistencies

### Architectural Invariants

- All interpreters implement the same `Handler` trait interface
- Type-specific methods preserve distribution return types throughout execution
- Trace accumulation is consistent across all interpreter types
- Safe variants never panic, always return valid traces (potentially with -∞ weights)

### Evolution Strategy

- **Backwards Compatible**: New interpreters can be added without breaking existing code
- **Extensible**: The Handler trait design supports custom interpreter implementations
- **Performance Focused**: Optimizations happen at the handler level, not in model code

## Error Handling

Different interpreters handle errors in fundamentally different ways:

### Strict Interpreter Errors

```rust
# use fugue::*;
# use fugue::runtime::interpreters::*;
# use rand::rngs::StdRng;
# use rand::SeedableRng;

// Create trace with f64 value
let mut rng = StdRng::seed_from_u64(123);
let (_, base_trace) = runtime::handler::run(
    PriorHandler { rng: &mut rng, trace: Trace::default() },
    sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
);

// This will panic - ReplayHandler expects exact type match
// let (_, _) = runtime::handler::run(
//     ReplayHandler { rng: &mut rng, base: base_trace, trace: Trace::default() },
//     sample(addr!("x"), Bernoulli::new(0.5).unwrap()) // Expects bool, trace has f64
// ); // PANICS: "expected bool at x"
```

### Safe Interpreter Error Recovery

```rust
# use fugue::*;
# use fugue::runtime::interpreters::*;
# use rand::rngs::StdRng;
# use rand::SeedableRng;
# let mut rng = StdRng::seed_from_u64(123);
# let (_, base_trace) = runtime::handler::run(
#     PriorHandler { rng: &mut rng, trace: Trace::default() },
#     sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
# );

// Safe replay handles type mismatch gracefully
let (result, safe_trace) = runtime::handler::run(
    SafeReplayHandler {
        rng: &mut rng,
        base: base_trace.clone(),
        trace: Trace::default(),
        warn_on_mismatch: true, // Logs warning but continues
    },
    sample(addr!("x"), Bernoulli::new(0.5).unwrap()) // Type mismatch handled gracefully
);

println!("Safe replay result: {}", result); // Fresh sample from Bernoulli
assert!(safe_trace.total_log_weight().is_finite());

// Safe scoring handles missing/mismatched addresses
let (_, error_trace) = runtime::handler::run(
    SafeScoreGivenTrace {
        base: base_trace,
        trace: Trace::default(),
        warn_on_error: false, // Silent error handling
    },
    sample(addr!("missing"), Normal::new(0.0, 1.0).unwrap()) // Missing address
);

// Returns -∞ instead of panicking
assert_eq!(error_trace.total_log_weight(), f64::NEG_INFINITY);
```

### Best Practices

- **Use strict interpreters** during development and testing for immediate error feedback
- **Use safe interpreters** in production systems where robustness is critical
- **Enable warnings** (`warn_on_mismatch`, `warn_on_error`) during debugging
- **Monitor trace validity** by checking `total_log_weight().is_finite()`
- **Implement fallback strategies** when safe interpreters return invalid traces

## Integration Notes

### With Handler System

All interpreters implement the `Handler` trait and integrate seamlessly:

- Zero-cost dispatch through compile-time trait resolution
- Consistent interface across all execution modes
- Composable with memory optimization systems (pools, COW traces)

### With Inference Algorithms

| Algorithm | Primary Interpreters | Usage Pattern |
|---|---|---|
| **MCMC** | ReplayHandler + ScoreGivenTrace | Replay current state, score proposals |
| **SMC** | PriorHandler + ScoreGivenTrace | Generate particles, reweight importance |
| **VI** | PriorHandler + ScoreGivenTrace | Sample variational params, score gradients |
| **ABC** | PriorHandler | Forward simulate for approximate Bayesian computation |

### Performance Characteristics

- **PriorHandler**: O(1) per sampling operation, fastest for forward simulation
- **ReplayHandler**: O(log n) address lookup, efficient for sparse modifications  
- **ScoreGivenTrace**: O(log n) address lookup, no sampling overhead
- **Safe variants**: Additional O(1) error checking, minimal overhead

### Production Deployment

```rust
# use fugue::*;
# use fugue::runtime::interpreters::*;

// Production inference with error monitoring
struct InferenceRunner {
    error_count: usize,
    total_runs: usize,
}

impl InferenceRunner {
    fn run_safe_inference<M, A>(&mut self, model: M) -> Option<A> 
    where 
        M: Fn() -> Model<A>,
        A: Send + 'static,
    {
        let mut rng = rand::thread_rng();
        let (result, trace) = runtime::handler::run(
            SafeReplayHandler {
                rng: &mut rng,
                base: Trace::default(),
                trace: Trace::default(),
                warn_on_mismatch: true,
            },
            model()
        );
        
        self.total_runs += 1;
        
        if trace.total_log_weight().is_finite() {
            Some(result)
        } else {
            self.error_count += 1;
            if self.error_count % 100 == 0 {
                eprintln!("Warning: {} inference errors out of {} runs ({:.1}%)",
                    self.error_count, self.total_runs,
                    100.0 * self.error_count as f64 / self.total_runs as f64);
            }
            None
        }
    }
}
```

## Reference Links

### Core Types

- [`PriorHandler`](../interpreters.rs) - Forward sampling from prior distributions  
- [`ReplayHandler`](../interpreters.rs) - Trace replay with fallback sampling
- [`ScoreGivenTrace`](../interpreters.rs) - Fixed trace log-probability computation
- [`SafeReplayHandler`](../interpreters.rs) - Error-resilient trace replay
- [`SafeScoreGivenTrace`](../interpreters.rs) - Error-resilient trace scoring

### Related Systems

- [`Handler`](../handler.md) - The trait interface all interpreters implement
- [`Trace`](../trace.md) - The trace representation used by all interpreters  
- [Memory Optimization](../memory.md) - How interpreters integrate with memory pools
- [Inference Algorithms](../../inference/README.md) - How interpreters enable inference

### Usage Guides

- [MCMC Implementation](../../src/how-to/mcmc-implementation.md) - Using replay and scoring interpreters
- [Production Deployment](../../src/how-to/production-inference.md) - Safe interpreter patterns
- [Error Handling Strategies](../../src/how-to/interpreter-error-handling.md) - When to use strict vs safe

### Examples  

- [`interpreter_basics.rs`](../../../examples/interpreter_basics.rs) - Basic usage patterns
- [`mcmc_with_interpreters.rs`](../../../examples/mcmc_with_interpreters.rs) - MCMC implementation
- [`importance_sampling.rs`](../../../examples/importance_sampling.rs) - Using scoring interpreters
- [`production_safe_inference.rs`](../../../examples/production_safe_inference.rs) - Safe interpreter deployment
