# Basic Inference

So far you've learned to build probabilistic models, but models alone don't solve problems - you need **inference** to extract insights from them. This guide introduces Fugue's inference algorithms and shows you how to use them effectively.

## What is Inference?

Inference is the process of computing the **posterior distribution** - what we believe about our model's parameters after seeing data. In Bayesian terms:

```
Posterior ∝ Prior × Likelihood
```

Fugue provides several inference algorithms, each with different trade-offs:

- **Prior Sampling** - Generate samples from the prior (no inference)
- **MCMC** - Markov Chain Monte Carlo for exact posterior sampling
- **SMC** - Sequential Monte Carlo for approximate inference  
- **VI** - Variational Inference for fast approximate inference
- **ABC** - Approximate Bayesian Computation for likelihood-free inference

## Prior Sampling (The Starting Point)

Before doing real inference, let's understand prior sampling - running your model without considering observations:

```rust
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn simple_model(observation: f64) -> Model<f64> {
    prob! {
        let mu <- sample(addr!("mu"), Normal::new(0.0, 2.0).unwrap());
        observe(addr!("y"), Normal::new(mu, 1.0).unwrap(), observation);
        pure(mu)
    }
}

fn prior_sampling_example() {
    let model = simple_model(3.0);  // We observed 3.0
    
    println!("Prior samples (ignoring observation):");
    for i in 0..5 {
        let mut rng = StdRng::seed_from_u64(i);
        let (mu, trace) = runtime::handler::run(
            runtime::interpreters::PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            model.clone(),
        );
        
        println!("  Sample {}: mu = {:.3}, log weight = {:.3}", 
                 i + 1, mu, trace.total_log_weight());
    }
}
```

**Key insight**: Prior sampling ignores observations - it's useful for debugging models and understanding priors, but not for real inference.

## MCMC: The Gold Standard

Markov Chain Monte Carlo (MCMC) is the most commonly used inference method. It generates samples from the exact posterior distribution.

### Basic MCMC with Metropolis-Hastings

```rust
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn mcmc_example() {
    let observation = 2.5;
    let model = || simple_model(observation);
    
    let mut rng = StdRng::seed_from_u64(42);
    
    // Run MCMC chain
    let n_samples = 1000;
    let mut samples = Vec::new();
    
    // Start with a sample from the prior
    let (mut current_value, mut current_trace) = runtime::handler::run(
        runtime::interpreters::PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        model(),
    );
    
    for i in 0..n_samples {
        // Propose new state using single-site MH
        let (new_value, new_trace) = inference::mh::single_site_random_walk_mh(
            &mut rng,
            0.5,  // proposal standard deviation
            model,
            &current_trace,
        );
        
        // Accept or reject (handled automatically by the algorithm)
        current_value = new_value;
        current_trace = new_trace;
        
        samples.push(current_value);
        
        if i % 200 == 0 {
            println!("Iteration {}: mu = {:.3}, log weight = {:.3}", 
                     i, current_value, current_trace.total_log_weight());
        }
    }
    
    // Analyze results
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    println!("\nPosterior mean estimate: {:.3}", mean);
    println!("(True posterior mean ≈ 1.67 for this model)");
}
```

### Adaptive MCMC (Recommended)

For production use, Fugue provides adaptive MCMC that automatically tunes the proposal distribution:

```rust
fn adaptive_mcmc_example() {
    let observation = 2.5;
    let model = || simple_model(observation);
    
    let mut rng = StdRng::seed_from_u64(42);
    
    // Use adaptive MCMC with automatic tuning
    let samples = inference::mcmc::adaptive_mcmc_chain(
        &mut rng,
        model,
        1000,  // n_samples
        500,   // n_warmup (tuning phase)
    );
    
    // Extract parameter values
    let mu_samples: Vec<f64> = samples
        .iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("mu")))
        .collect();
    
    // Basic statistics
    let mean = mu_samples.iter().sum::<f64>() / mu_samples.len() as f64;
    let variance = mu_samples.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / (mu_samples.len() - 1) as f64;
    
    println!("Adaptive MCMC Results:");
    println!("  Posterior mean: {:.3}", mean);
    println!("  Posterior std:  {:.3}", variance.sqrt());
    println!("  Effective samples: {}", mu_samples.len());
    
    // Convergence diagnostics
    let ess = inference::diagnostics::effective_sample_size_mcmc(&mu_samples);
    println!("  Effective sample size: {:.1}", ess);
}
```

## Sequential Monte Carlo (SMC)

SMC uses a population of particles to approximate the posterior. It's especially useful for sequential data and online inference:

```rust
fn smc_example() {
    let observation = 2.5;
    let model = || simple_model(observation);
    
    let mut rng = StdRng::seed_from_u64(42);
    
    // Generate particles from the prior
    let particles = inference::smc::smc_prior_particles(
        &mut rng,
        1000,  // number of particles
        model,
    );
    
    println!("SMC Results:");
    println!("  Number of particles: {}", particles.len());
    
    // Compute weighted statistics
    let total_weight: f64 = particles.iter().map(|p| p.weight).sum();
    let weighted_mean: f64 = particles
        .iter()
        .filter_map(|p| {
            p.trace.get_f64(&addr!("mu")).map(|mu| mu * p.weight)
        })
        .sum::<f64>() / total_weight;
    
    println!("  Weighted posterior mean: {:.3}", weighted_mean);
    
    // Effective sample size
    let weights: Vec<f64> = particles.iter().map(|p| p.weight).collect();
    let ess = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();
    println!("  Effective sample size: {:.1}", ess);
}
```

## Choosing the Right Inference Method

Here's a practical guide for choosing inference methods:

### Use MCMC when:
- ✅ You want exact posterior samples
- ✅ You have a moderate number of parameters (< 100)
- ✅ You can afford longer computation time
- ✅ Model evaluation is fast

```rust
// Best for: Standard Bayesian models
let mcmc_samples = inference::mcmc::adaptive_mcmc_chain(
    &mut rng,
    || your_model(),
    2000,  // samples
    1000,  // warmup
);
```

### Use SMC when:
- ✅ You have sequential/streaming data
- ✅ You need online inference
- ✅ Model has many discrete latent variables
- ✅ You want to visualize the inference process

```rust
// Best for: Time series, online learning
let particles = inference::smc::adaptive_smc(
    &mut rng,
    1000,  // particles
    || your_model(),
    smc_config,
);
```

### Use VI when:
- ✅ You need fast approximate inference
- ✅ You have many parameters (> 100)
- ✅ You can accept approximate results
- ✅ You want predictable runtime

```rust
// Best for: Large models, production systems
let vi_result = inference::vi::mean_field_vi(
    &mut rng,
    || your_model(),
    guide,
    1000,  // iterations
    0.01,  // learning rate
);
```

## Real-World Example: Bayesian Linear Regression

Here's a complete example showing inference on a realistic model:

```rust
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn bayesian_regression(x_data: &[f64], y_data: &[f64]) -> Model<(f64, f64, f64)> {
    prob! {
        // Priors
        let slope <- sample(addr!("slope"), Normal::new(0.0, 1.0).unwrap());
        let intercept <- sample(addr!("intercept"), Normal::new(0.0, 1.0).unwrap());
        let noise <- sample(addr!("noise"), LogNormal::new(0.0, 0.5).unwrap());
        
        // Likelihood
        for (i, (&x, &y)) in x_data.iter().zip(y_data.iter()).enumerate() {
            let y_pred = slope * x + intercept;
            observe(addr!("y", i), Normal::new(y_pred, noise).unwrap(), y);
        }
        
        pure((slope, intercept, noise))
    }
}

fn regression_inference_example() {
    // Generate some synthetic data
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y_data = vec![2.1, 3.9, 6.1, 8.0, 9.9];  // slope ≈ 2, intercept ≈ 0
    
    let model = || bayesian_regression(&x_data, &y_data);
    
    let mut rng = StdRng::seed_from_u64(42);
    
    println!("🔍 Bayesian Linear Regression");
    println!("Data points: {:?}", x_data.iter().zip(&y_data).collect::<Vec<_>>());
    
    // Run adaptive MCMC
    let samples = inference::mcmc::adaptive_mcmc_chain(
        &mut rng,
        model,
        2000,  // samples
        1000,  // warmup
    );
    
    // Extract parameters
    let slopes: Vec<f64> = samples.iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("slope")))
        .collect();
    let intercepts: Vec<f64> = samples.iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("intercept")))
        .collect();
    let noises: Vec<f64> = samples.iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("noise")))
        .collect();
    
    // Compute statistics
    println!("\n📊 Posterior Results:");
    println!("  Slope:     {:.3} ± {:.3}", 
             mean(&slopes), std(&slopes));
    println!("  Intercept: {:.3} ± {:.3}", 
             mean(&intercepts), std(&intercepts));
    println!("  Noise:     {:.3} ± {:.3}", 
             mean(&noises), std(&noises));
    
    // Convergence diagnostics
    println!("\n🔬 Diagnostics:");
    println!("  Slope ESS:     {:.1}", 
             inference::diagnostics::effective_sample_size_mcmc(&slopes));
    println!("  Intercept ESS: {:.1}", 
             inference::diagnostics::effective_sample_size_mcmc(&intercepts));
}

// Helper functions
fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn std(values: &[f64]) -> f64 {
    let m = mean(values);
    let variance = values.iter()
        .map(|x| (x - m).powi(2))
        .sum::<f64>() / (values.len() - 1) as f64;
    variance.sqrt()
}
```

## Debugging Inference Problems

Common issues and solutions:

### Problem: Poor Mixing (MCMC)
**Symptoms**: Samples don't explore the space well, high autocorrelation

**Solutions**:
```rust
// 1. Increase proposal variance
let samples = single_site_random_walk_mh(
    &mut rng,
    1.0,  // Try larger step size
    model,
    &trace,
);

// 2. Use adaptive MCMC (automatically tunes)
let samples = adaptive_mcmc_chain(&mut rng, model, 2000, 1000);

// 3. More warmup
let samples = adaptive_mcmc_chain(&mut rng, model, 2000, 2000);
```

### Problem: Low Effective Sample Size (SMC)
**Symptoms**: Most particles have very low weights

**Solutions**:
```rust
// 1. More particles
let particles = smc_prior_particles(&mut rng, 5000, model);

// 2. Use resampling
let config = SMCConfig {
    resampling_threshold: 0.5,  // Resample when ESS < 50%
    resampling_method: ResamplingMethod::Systematic,
};
```

### Problem: Numerical Instability
**Symptoms**: NaN or infinite log weights

**Solutions**:
```rust
// 1. Check your distributions
let safe_normal = Normal::new(mu, sigma)?;  // Returns Result
let safe_model = sample(addr!("x"), safe_normal);

// 2. Add bounds checking
let bounded_model = prob! {
    let x <- sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
    if x.is_finite() && x.abs() < 10.0 {
        pure(x)
    } else {
        factor(f64::NEG_INFINITY);  // Reject bad values
        pure(0.0)
    }
};
```

## Performance Tips

### 1. Use Efficient Addressing
```rust
// ❌ Inefficient: string concatenation in hot loops
for i in 0..1000 {
    sample(addr!(&format!("x_{}", i)), dist)
}

// ✅ Efficient: pre-computed addresses or simple indexing
for i in 0..1000 {
    sample(addr!("x", i), dist)  // Uses built-in indexing
}
```

### 2. Minimize Model Allocations
```rust
// ❌ Creates new model each time
fn slow_model() -> Model<f64> {
    sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
}

// ✅ Reuse model instances
let fast_model = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
```

### 3. Batch Observations
```rust
// ❌ Many individual observations
for (i, &y) in data.iter().enumerate() {
    observe(addr!("y", i), Normal::new(mu, sigma).unwrap(), y);
}

// ✅ Use vector distributions when available
// (Note: This is conceptual - Fugue doesn't have MultivariateNormal yet)
```

## Next Steps

Now that you understand basic inference:

1. **[Working with Distributions](../how-to/working-with-distributions.md)** - Master all distribution types
2. **[Trace Manipulation](../how-to/trace-manipulation.md)** - Advanced debugging and analysis
3. **[Bayesian Coin Flip Tutorial](../tutorials/bayesian-coin-flip.md)** - Complete worked example

---

**Ready for practical how-to guides?** → **[How-To Guides](../how-to/index.md)**