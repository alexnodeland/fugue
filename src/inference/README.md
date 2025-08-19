# Inference Module

The inference module provides algorithms for posterior inference in probabilistic models:

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

## Current Limitations

These are minimal implementations suitable for:
- **Prototyping**: Quick exploration of model behavior
- **Baselines**: Comparing against more sophisticated methods
- **Education**: Understanding basic inference principles

For production use, consider implementing:
- **MH**: Site-wise proposals, adaptive scaling, better mixing
- **SMC**: Resampling, rejuvenation, staged models
- **VI**: Structured variational families, reparameterized gradients

## Usage Examples

### Simple MCMC Chain
```rust
let mut trace = run(PriorHandler{rng: &mut rng, trace: Trace::default()}, model.clone()).1;

for i in 0..1000 {
    let (_, new_trace) = single_site_random_walk_mh(&mut rng, 0.5, || model.clone(), &trace);
    trace = new_trace;
    
    if i % 100 == 0 {
        println!("Iteration {}: log_weight = {:.4}", i, trace.total_log_weight());
    }
}
```

### Importance Sampling
```rust
let particles = smc_prior_particles(&mut rng, 1000, || model.clone());
let weights: Vec<f64> = particles.iter().map(|p| p.weight).collect();
let effective_sample_size = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();
println!("ESS: {:.1}", effective_sample_size);
```

## Extension Points

To add new inference methods:

1. **Implement new handlers** in `runtime/interpreters.rs`
2. **Add algorithm functions** in appropriate inference files  
3. **Export from module** in `inference/mod.rs`
4. **Update library exports** in `lib.rs`

Example custom handler:
```rust
pub struct CustomHandler { /* ... */ }

impl Handler for CustomHandler {
    fn on_sample(&mut self, addr: &Address, dist: &dyn DistributionF64) -> f64 {
        // Custom sampling logic
    }
    // ... other methods
}
```
