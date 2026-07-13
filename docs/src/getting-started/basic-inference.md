# Running Inference

```admonish info title="Contents"
<!-- toc -->
```

You now know how to build probabilistic models. But models alone don't give you answers - you need **inference** to extract insights from them. Let's explore Fugue's inference algorithms!

```admonish note
Learning Goals

In 5 minutes, you'll understand:
- What inference is and why you need it
- Fugue's main inference algorithms (MCMC, HMC, SMC, VI, ABC)
- When to use each algorithm
- How to run inference and interpret results

**Time**: ~5 minutes
```

```admonish tip title="Try it live"
Feel the difference between algorithms with your own hands: **[Random Walks in Posterior Space](../explorables/metropolis.md)** (Metropolis-Hastings) and **[Rolling, Not Guessing](../explorables/hmc.md)** (Hamiltonian Monte Carlo) run the same 2D posterior side by side so you can see why gradients help.
```

## What is Inference?

**Inference** is the process of learning about model parameters after seeing data. In Bayesian terms:

$$\text{Posterior} = \frac{\text{Prior} \times \text{Likelihood}}{\text{Evidence}}$$

```mermaid
graph LR
    subgraph "Before Data"
        P["Prior Beliefs<br/>p(theta)"]
    end

    subgraph "Observing Data"
        L["Likelihood<br/>p(y|theta)"]
        D["Data<br/>y₁, y₂, ..."]
    end

    subgraph "After Data"
        Post["Posterior Beliefs<br/>p(theta|y)"]
    end

    P --> Post
    L --> Post
    D --> Post
```

## The Challenge

Most real models don't have analytical solutions. We need **algorithms** to approximate the posterior distribution.

## Fugue's Inference Arsenal

### 1. MCMC (Markov Chain Monte Carlo) 🥇

**Best for**: Most general-purpose Bayesian inference

**How it works**: Generates samples that approximate the posterior distribution

```rust,ignore
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn coin_bias_model(heads: u64, total: u64) -> Model<f64> {
    sample(addr!("bias"), Beta::new(1.0, 1.0).unwrap())  // Prior
        .bind(move |bias| {
            observe(addr!("heads"), Binomial::new(total, bias).unwrap(), heads)  // Likelihood
                .map(move |_| bias)
        })
}

fn main() {
    let mut rng = StdRng::seed_from_u64(42);

    // Run adaptive MCMC
    let samples = inference::mh::adaptive_mcmc_chain(
        &mut rng,
        || coin_bias_model(7, 10),  // 7 heads out of 10 flips
        1000,  // number of samples
        500,   // warmup samples
    );

    // Extract bias estimates
    let bias_samples: Vec<f64> = samples.iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("bias")))
        .collect();

    let mean_bias = bias_samples.iter().sum::<f64>() / bias_samples.len() as f64;
    println!("Estimated bias: {:.3}", mean_bias);

    // Compute 90% credible interval
    let mut sorted = bias_samples.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let lower = sorted[(0.05 * sorted.len() as f64) as usize];
    let upper = sorted[(0.95 * sorted.len() as f64) as usize];
    println!("90% credible interval: [{:.3}, {:.3}]", lower, upper);
}
```

**When to use MCMC:**

- ✅ Want exact posterior samples
- ✅ Moderate number of parameters (< 100)
- ✅ Can afford computation time
- ✅ Model evaluation is reasonably fast

### 2. HMC (Hamiltonian Monte Carlo) 🎢

**Best for**: Continuous, correlated, or higher-dimensional posteriors where single-site MCMC mixes slowly

**How it works**: Treats the negative log-posterior as a landscape and rolls a simulated ball across it using its gradient, moving every continuous parameter together instead of one at a time

```rust,ignore
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn main() {
    let mut rng = StdRng::seed_from_u64(42);

    // Same conjugate model, but HMC moves `bias` using gradient information
    // instead of a random-walk proposal.
    let samples = inference::hmc::hmc_chain(
        &mut rng,
        || coin_bias_model(7, 10),
        1000,               // number of samples
        500,                // warmup iterations (step-size adaptation)
        HMCConfig::default(), // 16 leapfrog steps, target 80% acceptance
    );

    let bias_samples: Vec<f64> = samples.iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("bias")))
        .collect();

    let mean_bias = bias_samples.iter().sum::<f64>() / bias_samples.len() as f64;
    println!("HMC estimated bias: {:.3}", mean_bias);
}
```

Fugue's HMC computes the gradient with central finite differences (models are plain Rust closures, not auto-diff traces), then uses a leapfrog integrator and an exact Metropolis correction against the true log-density — the finite-difference force only affects efficiency, never correctness. Step size is tuned automatically during warmup via dual averaging toward an 80% target acceptance rate (Hoffman & Gelman 2014); discrete sites are held fixed for the duration of the HMC update (Metropolis-within-Gibbs), so compose with `adaptive_mcmc_chain` when a model mixes continuous and discrete latents.

**When to use HMC:**

- ✅ All (or most) latent variables are continuous
- ✅ Parameters are correlated (HMC exploits gradient direction; single-site MH can't)
- ✅ You want fewer iterations to reach the same effective sample size
- ⚠️ Purely discrete models get no benefit — HMC degenerates to prior draws when there are no continuous sites

### 3. SMC (Sequential Monte Carlo) 🎯

**Best for**: Sequential data and online learning

**How it works**: Uses particles to approximate the posterior, good for streaming data

```rust,ignore
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn main() {
    let mut rng = StdRng::seed_from_u64(42);

    // Generate particles from prior
    let particles = inference::smc::smc_prior_particles(
        &mut rng,
        1000,  // number of particles
        || coin_bias_model(7, 10),
    );

    println!("Generated {} particles", particles.len());

    // Compute weighted posterior mean
    let total_weight: f64 = particles.iter().map(|p| p.weight).sum();
    let weighted_mean: f64 = particles.iter()
        .filter_map(|p| {
            p.trace.get_f64(&addr!("bias"))
                .map(|bias| bias * p.weight)
        })
        .sum::<f64>() / total_weight;

    println!("Weighted posterior mean: {:.3}", weighted_mean);

    // Check effective sample size
    let weights: Vec<f64> = particles.iter().map(|p| p.weight).collect();
    let ess = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();
    println!("Effective sample size: {:.1}", ess);
}
```

**When to use SMC:**

- ✅ Sequential/streaming data
- ✅ Online inference needed
- ✅ Many discrete latent variables
- ✅ Want to visualize inference process

```admonish note title="Beyond prior particles"
`smc_prior_particles` above shows the mechanics with plain importance sampling. For a real particle filter — likelihood tempering, adaptive resampling, and rejuvenation — use `inference::smc::adaptive_smc(&mut rng, num_particles, model_fn, SMCConfig::default())`. Its result also carries `log_evidence`: an unbiased estimate of the log marginal likelihood, useful for model comparison. See `examples/smc_inference.rs`.
```

### 4. Variational Inference (VI) ⚡

**Best for**: Fast approximate inference with many parameters

**How it works**: Finds the best approximation within a family of simple distributions

```rust,ignore
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn main() {
    let mut rng = StdRng::seed_from_u64(42);

    // Estimate ELBO (Evidence Lower BOund)
    let elbo = inference::vi::estimate_elbo(
        &mut rng,
        || coin_bias_model(7, 10),
        100,  // number of samples for estimation
    );

    println!("ELBO estimate: {:.3}", elbo);

    // For more sophisticated VI, you'd set up a variational guide
    // and optimize it (see the VI tutorial for details)
}
```

**When to use VI:**

- ✅ Need fast approximate inference
- ✅ Many parameters (> 100)
- ✅ Can accept approximation error
- ✅ Want predictable runtime

```admonish note title="Fitting a guide for real"
`estimate_elbo` above uses the prior itself as the variational guide — a zero-setup bound, but usually loose. For a fitted guide, build a `MeanFieldGuide` (support-matched Normal/LogNormal/Beta factors per latent) and call `inference::vi::optimize_meanfield_vi_with_config`, which optimizes both location and scale via common-random-numbers gradients. See `examples/vi_inference.rs`.
```

### 5. ABC (Approximate Bayesian Computation) 🎲

**Best for**: Models where likelihood is intractable or expensive

**How it works**: Simulation-based inference using distance between simulated and observed data

```rust,ignore
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn main() {
    let mut rng = StdRng::seed_from_u64(42);

    // ABC with summary statistics
    let observed_summary = 0.7; // 7/10 = 0.7 success rate
    let samples = inference::abc::abc_scalar_summary(
        &mut rng,
        || sample(addr!("bias"), Beta::new(1.0, 1.0).unwrap()), // Prior only
        |trace| trace.get_f64(&addr!("bias")).unwrap_or(0.0), // Extract bias
        observed_summary,  // Target summary statistic
        0.1,              // Tolerance
        1000,             // Max samples to try
    );

    println!("ABC accepted {} samples", samples.len());

    if !samples.is_empty() {
        let abc_estimates: Vec<f64> = samples.iter()
            .filter_map(|trace| trace.get_f64(&addr!("bias")))
            .collect();
        let abc_mean = abc_estimates.iter().sum::<f64>() / abc_estimates.len() as f64;
        println!("ABC estimated bias: {:.3}", abc_mean);
    }
}
```

**When to use ABC:**

- ✅ Likelihood is intractable or very expensive
- ✅ Can simulate from the model easily
- ✅ Have good summary statistics
- ✅ Can tolerate approximation error

```admonish note title="ABC-SMC for harder problems"
`abc_scalar_summary` above is plain rejection ABC — simple, but wasteful when the tolerance is tight. `inference::abc::abc_smc_weighted` anneals the tolerance across rounds with importance-weighted particles (replacing the biased prior-replacement heuristic older ABC-SMC implementations used), giving much better acceptance at tight tolerances. See `examples/abc_inference.rs`.
```

## Algorithm Comparison

| Method   | Speed     | Accuracy       | Use Case                                 |
| -------- | --------- | -------------- | ---------------------------------------- |
| **MCMC** | 🐌 Slow   | 🎯 Exact       | General-purpose, exact inference         |
| **HMC**  | 🐌 Slow   | 🎯 Exact       | Continuous, correlated, higher-dimensional posteriors |
| **SMC**  | 🏃 Medium | 🎯 Good        | Sequential data, online learning         |
| **VI**   | 🚀 Fast   | ⚠️ Approximate | Large models, fast approximate inference |
| **ABC**  | 🐌 Slow   | ⚠️ Approximate | Intractable likelihoods                  |

## Practical Inference Workflow

Here's a typical workflow for real inference:

```rust,ignore
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn inference_workflow() {
    let mut rng = StdRng::seed_from_u64(42);

    // 1. Define your model
    let model = || coin_bias_model(17, 25);  // 17 heads out of 25 flips

    // 2. Run inference (adaptive MCMC is often a good default)
    let samples = inference::mh::adaptive_mcmc_chain(
        &mut rng,
        model,
        2000,  // samples
        1000,  // warmup
    );

    // 3. Extract parameter values
    let bias_samples: Vec<f64> = samples.iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("bias")))
        .collect();

    // 4. Compute summary statistics
    let mean = bias_samples.iter().sum::<f64>() / bias_samples.len() as f64;
    let variance = bias_samples.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / (bias_samples.len() - 1) as f64;
    let std_dev = variance.sqrt();

    println!("Posterior Summary:");
    println!("  Mean: {:.3}", mean);
    println!("  Std Dev: {:.3}", std_dev);

    // 5. Check convergence (autocorrelation-based effective sample size)
    let ess = inference::diagnostics::effective_sample_size(&bias_samples);
    println!("  Effective Sample Size: {:.1}", ess);
    // For multiple chains, also check split-R̂ (inference::diagnostics::r_hat_f64)
    // and inference::mcmc_utils::effective_sample_size_multichain — both new in 0.2.0.

    if ess > 100.0 {
        println!("  ✅ Good mixing!");
    } else {
        println!("  ⚠️ Poor mixing - consider more samples");
    }

    // 6. Make predictions
    println!("\nPredictions:");
    println!("  P(bias > 0.5) = {:.2}",
        bias_samples.iter().filter(|&&b| b > 0.5).count() as f64 / bias_samples.len() as f64);
}
```

## Choosing the Right Algorithm

### Decision Tree

```mermaid
graph TD
    A[Need inference?] -->|Yes| B[Real-time/online?]
    A -->|No| Z[Use PriorHandler<br/>for forward sampling]

    B -->|Yes| SMC[SMC]
    B -->|No| C[Likelihood tractable?]

    C -->|No| ABC[ABC]
    C -->|Yes| D[Many parameters?]

    D -->|Yes > 100| VI[Variational Inference]
    D -->|No < 100| E[Need exact samples?]

    E -->|Yes| F[Mostly continuous & correlated?]
    E -->|No| VI2[VI for speed]

    F -->|Yes| HMC[HMC]
    F -->|No, mostly discrete| MCMC[MCMC]
```

### Rules of Thumb

1. **Start with MCMC** for most problems - it's the most general
2. **Reach for HMC** when parameters are continuous and correlated - it mixes far faster than single-site MCMC by using the gradient
3. **Use SMC** if you have sequential/streaming data
4. **Use VI** if you need speed and can accept approximation
5. **Use ABC** only when likelihood is truly intractable

## Key Takeaways

You now know how to extract insights from your models:

✅ **Inference Purpose**: Learn parameters from data using Bayesian updating  
✅ **Algorithm Options**: MCMC, HMC, SMC, VI, ABC each have their strengths  
✅ **Practical Workflow**: Define model → Run inference → Extract parameters → Check diagnostics  
✅ **Algorithm Selection**: Choose based on problem characteristics and requirements

## What's Next?

You've completed Getting Started! 🎉

```admonish tip
Ready for Real Applications?

**Complete Tutorials** - End-to-end projects with real-world applications:
- **[Bayesian Coin Flip](../tutorials/bayesian-coin-flip.md)** - Complete analysis workflow
- **[Linear Regression](../tutorials/linear-regression.md)** - Advanced modeling and diagnostics
- **[Mixture Models](../tutorials/mixture-models.md)** - Latent variable models

**How-To Guides** - Specific techniques and best practices:
- **[Working with Distributions](../how-to/working-with-distributions.md)** - Master all distribution types
- **[Debugging Models](../how-to/debugging-models.md)** - Troubleshoot inference problems
```

---

**Time**: ~5 minutes • **Next**: [Complete Tutorials](../tutorials/README.md)
