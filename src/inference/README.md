# Inference Module

## Overview

The inference module provides a comprehensive suite of algorithms for posterior inference in probabilistic models. It includes implementations of Markov Chain Monte Carlo (MCMC), Sequential Monte Carlo (SMC), Variational Inference (VI), and Approximate Bayesian Computation (ABC) methods, all designed for production use with comprehensive diagnostics.

## Quick Start

```rust
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

// Run adaptive MCMC on a Bayesian model
let model = || gaussian_mean_model(2.7).unwrap();
let mut rng = StdRng::seed_from_u64(42);
let samples = adaptive_mcmc_chain(&mut rng, model, 1000, 500);

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
let (new_value, new_trace) = single_site_random_walk_mh(
    &mut rng,
    0.1,                    // proposal standard deviation
    || model.clone(),       // model factory
    &current_trace          // current state
);
```

### `smc.rs` - Sequential Monte Carlo

- `smc_prior_particles`: Generate weighted particles from the prior
- `Particle`: Represents a trace with associated weight

```rust
let particles = smc_prior_particles(
    &mut rng,
    1000,                   // number of particles
    || model.clone()        // model factory
);

for particle in particles {
    println!("Weight: {:.4}, Trace: {:?}", particle.weight, particle.trace);
}
```

### `vi.rs` - Variational Inference

- `estimate_elbo`: Monte Carlo ELBO estimation using prior as proposal
- Placeholder for more sophisticated variational methods

```rust
let elbo = estimate_elbo(
    &mut rng,
    || model.clone(),       // model factory
    1000                    // number of samples
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
let samples = abc_rejection(
    &mut rng,
    || your_model(),
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
let chains = run_multiple_chains(&mut rng, || model(), 4, 1000, 500);
let r_hat = r_hat_f64(&chains, &addr!("mu"));
assert!(r_hat < 1.1, "Chains have not converged: R-hat = {:.3}", r_hat);
```

### `validation.rs` - Statistical Validation

**Key Functions:**

- `test_conjugate_normal_model`: Validate against analytical solutions
- `ks_test_distribution`: Kolmogorov-Smirnov goodness-of-fit tests
- `ValidationResult`: Structured validation reporting

**Example:**

```rust
let validation = test_conjugate_normal_model(
    &mut rng, mcmc_sampler, prior_mu, prior_sigma,
    likelihood_sigma, observation, n_samples, n_warmup
);
validation.print_summary();
assert!(validation.is_valid());
```

## Common Patterns

### Multi-Chain MCMC with Diagnostics

Run multiple chains and assess convergence using R-hat diagnostics.

```rust
let chains = (0..4).map(|chain_id| {
    let mut rng = StdRng::seed_from_u64(42 + chain_id);
    adaptive_mcmc_chain(&mut rng, || model(), 1000, 500)
}).collect::<Vec<_>>();

let r_hat = r_hat_f64(&chains, &addr!("parameter"));
if r_hat > 1.1 {
    eprintln!("Warning: Poor convergence (R-hat = {:.3})", r_hat);
}
```

### Particle Filtering with Resampling

Use SMC for sequential inference with systematic resampling.

```rust
let config = SMCConfig {
    n_particles: 1000,
    resampling_threshold: 0.5,
    resampling_method: ResamplingMethod::Systematic,
    rejuvenation_steps: 5,
};

let particles = adaptive_smc(&mut rng, 1000, || model(), config);
let ess = effective_sample_size(&particles);
println!("Effective Sample Size: {:.1}", ess);
```

### Variational Inference with Custom Guides

Optimize mean-field variational approximations.

```rust
let mut guide = MeanFieldGuide::new();
guide.add_parameter(addr!("mu"), VariationalNormal::new(0.0, 1.0));
guide.add_parameter(addr!("sigma"), VariationalLogNormal::new(0.0, 1.0));

let result = optimize_meanfield_vi(
    &mut rng, || model(), guide,
    1000,  // max iterations
    0.01,  // learning rate
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
   pub fn custom_sampler<M, F>(
       rng: &mut impl Rng,
       model_fn: F,
       n_samples: usize,
   ) -> Vec<(M::Value, Trace)>
   where
       F: Fn() -> M,
       M: Model,
   {
       // Your algorithm implementation
   }
   ```

2. **Custom Diagnostics**: Add domain-specific convergence tests

   ```rust
   pub fn custom_diagnostic(samples: &[f64]) -> f64 {
       // Your diagnostic computation
   }
   ```

3. **Custom Proposal Distributions**: Extend MCMC with new proposal mechanisms
4. **Custom Distance Functions**: Implement new distance metrics for ABC
5. **Custom Variational Families**: Add structured variational approximations beyond mean-field
