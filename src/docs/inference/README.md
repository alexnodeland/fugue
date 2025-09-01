# `inference` module

## Overview

The inference module provides a comprehensive suite of algorithms for posterior inference in probabilistic models. It includes implementations of Markov Chain Monte Carlo (MCMC), Sequential Monte Carlo (SMC), Variational Inference (VI), and Approximate Bayesian Computation (ABC) methods, all designed for production use with comprehensive diagnostics.

## Quick Start

```rust
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

// Define a simple Bayesian model
let model_fn = || {
    sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap())
        .bind(|mu| observe(addr!("y"), Normal::new(mu, 0.5).unwrap(), 2.7).map(move |_| mu))
};

// Run adaptive MCMC
let mut rng = StdRng::seed_from_u64(42);
let samples = adaptive_mcmc_chain(&mut rng, model_fn, 50, 10); // Small numbers for test

// Extract and analyze results
let mu_samples: Vec<f64> = samples.iter()
    .filter_map(|(_, trace)| trace.get_f64(&addr!("mu")))
    .collect();
let ess = effective_sample_size_mcmc(&mu_samples);
println!("Effective Sample Size: {:.1}", ess);
```

## Components

### `mh.rs` - Metropolis-Hastings Sampling

- `single_site_random_walk_mh`: Basic MH transition kernel
- Proposes new traces and accepts/rejects based on score ratios

```rust
use fugue::*;
use fugue::inference::mh::single_site_random_walk_mh;
use rand::rngs::StdRng;
use rand::SeedableRng;

// Setup for the example
let model_fn = || sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap());
let mut rng = StdRng::seed_from_u64(42);
let (_, current_trace) = runtime::handler::run(
    PriorHandler { rng: &mut rng, trace: Trace::default() },
    model_fn()
);

let (new_value, new_trace) = single_site_random_walk_mh(
    &mut rng,
    0.1,                    // proposal standard deviation
    model_fn,               // model factory
    &current_trace          // current state
);
```

### `smc.rs` - Sequential Monte Carlo

- `smc_prior_particles`: Generate weighted particles from the prior
- `Particle`: Represents a trace with associated weight

```rust
use fugue::*;
use fugue::inference::smc::smc_prior_particles;
use rand::rngs::StdRng;
use rand::SeedableRng;

// Setup for the example
let model_fn = || sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap());
let mut rng = StdRng::seed_from_u64(42);

let particles = smc_prior_particles(
    &mut rng,
    10,                     // number of particles (small for test)
    model_fn                // model factory
);

for particle in particles {
    println!("Weight: {:.4}, Trace: {:?}", particle.weight, particle.trace);
}
```

### `vi.rs` - Variational Inference

- `estimate_elbo`: Monte Carlo ELBO estimation using prior as proposal
- Placeholder for more sophisticated variational methods

```rust
use fugue::*;
use fugue::inference::vi::estimate_elbo;
use rand::rngs::StdRng;
use rand::SeedableRng;

// Setup for the example
let model_fn = || sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap())
    .bind(|mu| observe(addr!("y"), Normal::new(mu, 0.5).unwrap(), 1.0).map(move |_| mu));
let mut rng = StdRng::seed_from_u64(42);

let elbo = estimate_elbo(
    &mut rng,
    model_fn,               // model factory
    10                      // number of samples (small for test)
);
```

### `abc.rs` - Approximate Bayesian Computation

**Key Types/Functions:**

- `abc_rejection`: Likelihood-free rejection sampling
- `abc_smc`: Sequential Monte Carlo ABC
- `DistanceFunction`: Trait for defining distance metrics
- `EuclideanDistance`: Built-in Euclidean distance function

**Example:**

```rust
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

// Setup for the example
let mut rng = StdRng::seed_from_u64(42);
let model_fn = || sample(addr!("mu"), Normal::new(0.0, 2.0).unwrap());
let summary_statistic_fn = |trace: &Trace| {
    trace.get_f64(&addr!("mu")).unwrap_or(0.0)
};
let observed_summary = 1.5;
let epsilon = 0.5;
let max_samples = 10;

let samples = abc_scalar_summary(
    &mut rng,
    model_fn,
    summary_statistic_fn,
    observed_summary,
    epsilon,    // tolerance
    max_samples,
);
```

### `diagnostics.rs` - Convergence Assessment

**Key Functions:**

- `r_hat_f64`: Gelman-Rubin convergence diagnostic
- `effective_sample_size_mcmc`: ESS calculation
- `summarize_f64_parameter`: Parameter summary statistics
- `print_diagnostics`: Comprehensive diagnostic reporting

**Example:**

```rust
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

// Setup for the example
let model_fn = || sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap())
    .bind(|mu| observe(addr!("y"), Normal::new(mu, 0.5).unwrap(), 1.0).map(move |_| mu));

// Generate multiple chains
let chains: Vec<Vec<Trace>> = (0..2).map(|chain_id| {
    let mut rng = StdRng::seed_from_u64(42 + chain_id);
    let samples = adaptive_mcmc_chain(&mut rng, &model_fn, 10, 5);
    samples.into_iter().map(|(_, trace)| trace).collect()
}).collect();

let r_hat = r_hat_f64(&chains, &addr!("mu"));
if r_hat.is_finite() {
    println!("R-hat: {:.3}", r_hat);
}
```

### `validation.rs` - Statistical Validation

**Key Functions:**

- `test_conjugate_normal_model`: Validate against analytical solutions
- `ks_test_distribution`: Kolmogorov-Smirnov goodness-of-fit tests
- `ValidationResult`: Structured validation reporting

**Example:**

```rust
use fugue::*;
use fugue::inference::validation::{test_conjugate_normal_model, ConjugateNormalConfig};
use rand::rngs::StdRng;
use rand::SeedableRng;

// Setup for the example
let mut rng = StdRng::seed_from_u64(42);
let config = ConjugateNormalConfig {
    prior_mu: 0.0,
    prior_sigma: 1.0,
    likelihood_sigma: 0.5,
    observation: 1.5,
    n_samples: 20,
    n_warmup: 10,
};

// Simple MCMC sampler that uses fixed parameters
fn simple_mcmc_sampler(rng: &mut StdRng, n_samples: usize, n_warmup: usize) -> Vec<(f64, Trace)> {
    let model_fn = || sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap())
        .bind(|mu| observe(addr!("y"), Normal::new(mu, 0.5).unwrap(), 1.5).map(move |_| mu));
    adaptive_mcmc_chain(rng, model_fn, n_samples, n_warmup)
}

let validation = test_conjugate_normal_model(&mut rng, simple_mcmc_sampler, config);
validation.print_summary();
```

## Common Patterns

### Multi-Chain MCMC with Diagnostics

Run multiple chains and assess convergence using R-hat diagnostics.

```rust
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

// Setup model
let model_fn = || sample(addr!("parameter"), Normal::new(0.0, 1.0).unwrap())
    .bind(|p| observe(addr!("obs"), Normal::new(p, 0.5).unwrap(), 1.0).map(move |_| p));

// Run multiple chains
let chains: Vec<Vec<Trace>> = (0..2).map(|chain_id| {
    let mut rng = StdRng::seed_from_u64(42 + chain_id);
    let samples = adaptive_mcmc_chain(&mut rng, &model_fn, 10, 5); // Small numbers for test
    samples.into_iter().map(|(_, trace)| trace).collect()
}).collect();

let r_hat = r_hat_f64(&chains, &addr!("parameter"));
if r_hat.is_finite() && r_hat > 1.1 {
    eprintln!("Warning: Poor convergence (R-hat = {:.3})", r_hat);
}
```

### Particle Filtering with Resampling

Use SMC for sequential inference with systematic resampling.

```rust
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

// Setup for the example
let model_fn = || sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap())
    .bind(|mu| observe(addr!("y"), Normal::new(mu, 0.5).unwrap(), 1.0).map(move |_| mu));
let mut rng = StdRng::seed_from_u64(42);

let config = SMCConfig {
    resampling_method: ResamplingMethod::Systematic,
    ess_threshold: 0.5,
    rejuvenation_steps: 1,
};

let particles = adaptive_smc(&mut rng, 10, model_fn, config); // Small numbers for test
let ess = effective_sample_size(&particles);
println!("Effective Sample Size: {:.1}", ess);
```

### Variational Inference with Custom Guides

Optimize mean-field variational approximations.

```rust
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::HashMap;

// Setup for the example
let model_fn = || sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap())
    .bind(|mu| observe(addr!("y"), Normal::new(mu, 0.5).unwrap(), 1.0).map(move |_| mu));
let mut rng = StdRng::seed_from_u64(42);

let mut guide = MeanFieldGuide::new();
guide.params.insert(addr!("mu"), VariationalParam::Normal { mu: 0.0, log_sigma: 0.0 });

let result = optimize_meanfield_vi(
    &mut rng, model_fn, guide,
    5,   // max iterations (small for test)
    10,  // samples per iteration
    0.01 // learning rate
);
```

## Performance Considerations

- **Memory**: Use memory-efficient trace handling for long chains
- **Computation**: Adaptive algorithms adjust proposal distributions automatically
- **Best Practices**:
  - Use multiple chains to assess convergence
  - Monitor effective sample size, not just raw sample count
  - Validate against analytical solutions when available
  - Use appropriate warmup periods (typically 10-50% of total samples)

## Integration

**Related Modules:**

- [`core`](../core/README.md): Define models for inference using `Model<T>` and distributions
- [`runtime`](../runtime/README.md): Execute inference using different handlers and trace management
- [`error`](../error.rs): Handle inference-specific errors and validation

**See Also:**

- Main documentation: [API docs](https://docs.rs/fugue)
- Examples: [`examples/improved_gaussian_mean.rs`](../../examples/improved_gaussian_mean.rs), [`examples/gaussian_mixture.rs`](../../examples/gaussian_mixture.rs)

## Current Capabilities

Production-ready implementations include:

- **MCMC**: Adaptive Metropolis-Hastings with diminishing adaptation
- **SMC**: Systematic resampling with particle rejuvenation
- **VI**: Mean-field approximation with reparameterized gradients
- **ABC**: Rejection sampling and SMC variants
- **Diagnostics**: R-hat, ESS, Geweke tests, and validation frameworks

## Extension Points

How to extend the inference module:

1. **Custom Inference Algorithms**: Implement new algorithms following the established patterns

   ```rust
   use fugue::*;
   use rand::Rng;
   
   pub fn custom_sampler<A, F, R: Rng>(
       rng: &mut R,
       model_fn: F,
       n_samples: usize,
   ) -> Vec<(A, Trace)>
   where
       F: Fn() -> Model<A>,
   {
       let mut results = Vec::new();
       for _ in 0..n_samples {
           let (value, trace) = runtime::handler::run(
               PriorHandler { rng, trace: Trace::default() },
               model_fn()
           );
           results.push((value, trace));
       }
       results
   }
   ```

2. **Custom Diagnostics**: Add domain-specific convergence tests

   ```rust
   pub fn custom_diagnostic(samples: &[f64]) -> f64 {
       // Compute sample mean as a simple diagnostic
       if samples.is_empty() {
           return 0.0;
       }
       samples.iter().sum::<f64>() / samples.len() as f64
   }
   ```

3. **Custom Proposal Distributions**: Extend MCMC with new proposal mechanisms
4. **Custom Distance Functions**: Implement new distance metrics for ABC
5. **Custom Variational Families**: Add structured variational approximations beyond mean-field
