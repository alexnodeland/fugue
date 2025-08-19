# Fugue

[![Crates.io](https://img.shields.io/crates/v/fugue.svg)](https://crates.io/crates/fugue)
[![Documentation](https://docs.rs/fugue/badge.svg)](https://docs.rs/fugue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-blue.svg)](https://www.rust-lang.org)

A **production-ready**, **monadic probabilistic programming library** for Rust. Write elegant probabilistic programs by composing `Model` values in direct style; execute them with pluggable interpreters and state-of-the-art inference algorithms.

## ✨ Features

- 🎯 **Monadic PPL**: Compose probabilistic programs using pure functional abstractions
- 🔢 **Rich Distributions**: 10+ built-in probability distributions with validation
- 🎰 **Multiple Inference Methods**: MCMC, SMC, Variational Inference, ABC
- 📊 **Comprehensive Diagnostics**: R-hat convergence, effective sample size, Geweke tests
- 🛡️ **Numerically Stable**: Production-ready numerical algorithms with validation
- 🚀 **Memory Optimized**: Efficient trace handling and memory management
- 🎛️ **Ergonomic Macros**: Do-notation (`prob!`), vectorization (`plate!`), addressing (`addr!`)
- ⚡ **High Performance**: Zero-cost abstractions with pluggable runtime interpreters

## 🚀 Quick Start

Add Fugue to your `Cargo.toml`:

```toml
[dependencies]
fugue = "0.3.0"
```

### Simple Bayesian Linear Regression

```rust
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn bayesian_regression(x_data: &[f64], y_data: &[f64]) -> Model<(f64, f64)> {
    prob! {
        // Priors
        let slope <- sample(addr!("slope"), Normal { mu: 0.0, sigma: 1.0 });
        let intercept <- sample(addr!("intercept"), Normal { mu: 0.0, sigma: 1.0 });
        let noise <- sample(addr!("noise"), LogNormal { mu: 0.0, sigma: 0.5 });
        
        // Likelihood
        for (i, (&x, &y)) in x_data.iter().zip(y_data.iter()).enumerate() {
            let y_pred = slope * x + intercept;
            observe(addr!("y", i), Normal { mu: y_pred, sigma: noise }, y);
        }
        
        pure((slope, intercept))
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y_data = vec![2.1, 3.9, 6.1, 8.0, 9.9];
    
    let mut rng = StdRng::seed_from_u64(42);
    
    // Run adaptive MCMC
    let samples = adaptive_mcmc_chain(
        &mut rng,
        || bayesian_regression(&x_data, &y_data),
        1000,  // samples
        500,   // warmup
    );
    
    // Extract results
    let slopes: Vec<f64> = samples.iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("slope")))
        .collect();
    
    let mean_slope = slopes.iter().sum::<f64>() / slopes.len() as f64;
    println!("Estimated slope: {:.3}", mean_slope);
    
    // Diagnostics
    let ess = effective_sample_size_mcmc(&slopes);
    println!("Effective sample size: {:.1}", ess);
    
    Ok(())
}
```

## 📚 Core Concepts

### Models as First-Class Values

Fugue represents probabilistic programs as `Model<A>` values that can be composed, transformed, and reused:

```rust
use fugue::*;

// Pure deterministic computation
let model1 = pure(42.0);

// Probabilistic sampling
let model2 = sample(addr!("x"), Normal { mu: 0.0, sigma: 1.0 });

// Conditioning on observations
let model3 = observe(addr!("y"), Normal { mu: 0.0, sigma: 1.0 }, 2.5);

// Monadic composition
let composed = model2.bind(|x| {
    sample(addr!("y"), Normal { mu: x, sigma: 0.5 })
        .map(move |y| x + y)
});
```

### Do-Notation with `prob!`

Write probabilistic programs in an imperative style:

```rust
let mixture_model = prob! {
    let z <- sample(addr!("component"), Bernoulli { p: 0.3 });
    let mu = if z > 0.5 { -2.0 } else { 2.0 };
    let x <- sample(addr!("x"), Normal { mu, sigma: 1.0 });
    observe(addr!("y"), Normal { mu: x, sigma: 0.1 }, observed_value);
    pure(x)
};
```

### Vectorized Operations with `plate!`

Efficiently handle collections of random variables:

```rust
// Generate 100 independent samples
let samples = plate!(i in 0..100 => {
    sample(addr!("x", i), Normal { mu: 0.0, sigma: 1.0 })
});

// Hierarchical model with shared parameters
let hierarchical = prob! {
    let global_mu <- sample(addr!("global_mu"), Normal { mu: 0.0, sigma: 1.0 });
    let local_effects <- plate!(i in 0..10 => {
        sample(addr!("local", i), Normal { mu: global_mu, sigma: 0.1 })
    });
    pure((global_mu, local_effects))
};
```

## 🎯 Inference Methods

### Markov Chain Monte Carlo (MCMC)

```rust
// Adaptive Metropolis-Hastings with convergence diagnostics
let samples = adaptive_mcmc_chain(
    &mut rng,
    || your_model(),
    n_samples,
    n_warmup,
);

// Multi-chain diagnostics
let chains = run_multiple_chains(&mut rng, || your_model(), 4, 1000, 500);
let r_hat = r_hat(&chains, &addr!("parameter"));
println!("R-hat convergence diagnostic: {:.4}", r_hat);
```

### Sequential Monte Carlo (SMC)

```rust
let config = SMCConfig {
    n_particles: 1000,
    resampling_threshold: 0.5,
    resampling_method: ResamplingMethod::Systematic,
    rejuvenation_steps: 5,
};

let particles = adaptive_smc(&mut rng, 1000, || your_model(), config);
let ess = effective_sample_size(&particles);
```

### Variational Inference (VI)

```rust
// Mean-field variational approximation
let mut guide = MeanFieldGuide::new();
guide.add_parameter(addr!("mu"), VariationalNormal::new(0.0, 1.0));

let vi_result = mean_field_vi(
    &mut rng,
    || your_model(),
    guide,
    1000,  // max iterations
    0.01,  // learning rate
);
```

### Approximate Bayesian Computation (ABC)

```rust
// Likelihood-free inference
let samples = abc_rejection(
    &mut rng,
    || your_model(),
    summary_statistic_fn,
    observed_summary,
    epsilon,
    max_samples,
);
```

## 📊 Built-in Distributions

| **Distribution** | **Parameters** | **Support** | **Usage** |
|------------------|----------------|-------------|-----------|
| `Normal` | `mu`, `sigma` | ℝ | `Normal { mu: 0.0, sigma: 1.0 }` |
| `LogNormal` | `mu`, `sigma` | ℝ⁺ | `LogNormal { mu: 0.0, sigma: 1.0 }` |
| `Uniform` | `low`, `high` | [low, high] | `Uniform { low: 0.0, high: 1.0 }` |
| `Exponential` | `rate` | ℝ⁺ | `Exponential { rate: 1.0 }` |
| `Beta` | `alpha`, `beta` | [0, 1] | `Beta { alpha: 2.0, beta: 3.0 }` |
| `Gamma` | `shape`, `rate` | ℝ⁺ | `Gamma { shape: 2.0, rate: 1.0 }` |
| `Bernoulli` | `p` | {0, 1} | `Bernoulli { p: 0.3 }` |
| `Binomial` | `n`, `p` | {0, 1, ..., n} | `Binomial { n: 10, p: 0.5 }` |
| `Categorical` | `probs` | {0, 1, ..., k-1} | `Categorical::new(vec![0.2, 0.3, 0.5])` |
| `Poisson` | `rate` | ℕ | `Poisson { rate: 2.0 }` |

All distributions include automatic parameter validation and numerical stability checks.

## 🛠️ Advanced Features

### Custom Interpreters

Implement your own model interpreters:

```rust
struct CustomHandler {
    // Your state here
}

impl<A> Handler<A> for CustomHandler {
    fn handle_sample(&mut self, addr: Address, dist: Box<dyn DistributionF64>) -> f64 {
        // Custom sampling logic
    }
    
    fn handle_observe(&mut self, addr: Address, dist: Box<dyn DistributionF64>, value: f64) {
        // Custom observation handling
    }
    
    fn handle_factor(&mut self, logw: LogF64) {
        // Custom factor accumulation
    }
}
```

### Hierarchical Addressing

Organize complex models with scoped addresses:

```rust
let hierarchical = prob! {
    let global_params <- plate!(layer in 0..3 => {
        sample(scoped_addr!("layer", layer, "weight"), Normal { mu: 0.0, sigma: 1.0 })
    });
    // ... rest of model
    pure(global_params)
};
```

### Memory-Efficient Trace Manipulation

```rust
// Efficient trace operations
let mut trace = Trace::default();
trace.insert_choice(addr!("x"), ChoiceValue::F64(1.5), 0.0);

// Trace validation and debugging
trace.validate()?;
println!("Total log weight: {:.4}", trace.total_log_weight());
```

## 📋 Examples

Explore the [`examples/`](examples/) directory for complete working examples:

- **[`gaussian_mean.rs`](examples/gaussian_mean.rs)** - Basic Bayesian inference for Gaussian mean
- **[`gaussian_mixture.rs`](examples/gaussian_mixture.rs)** - Mixture model with discrete latent variables
- **[`exponential_hazard.rs`](examples/exponential_hazard.rs)** - Survival analysis with exponential distributions
- **[`conjugate_beta_binomial.rs`](examples/conjugate_beta_binomial.rs)** - Beta-Binomial conjugate analysis
- **[`improved_gaussian_mean.rs`](examples/improved_gaussian_mean.rs)** - Advanced MCMC with full diagnostics

Run examples with:

```bash
cargo run --example gaussian_mean -- --obs 2.5 --seed 42
cargo run --example improved_gaussian_mean -- --obs 1.5 --n-samples 2000 --validate
```

## 🧪 Validation & Testing

Fugue includes extensive validation against analytical solutions:

```rust
// Validate MCMC against known posterior
let validation = test_conjugate_normal_model(
    &mut rng,
    mcmc_sampler,
    prior_mu,
    prior_sigma,
    likelihood_sigma,
    observation,
    n_samples,
    n_warmup,
);

validation.print_summary();
assert!(validation.is_valid());
```

## ⚡ Performance

Fugue is designed for production workloads:

- **Zero-cost abstractions**: Monadic composition compiles to efficient code
- **Memory optimization**: Efficient trace representation and garbage collection
- **Numerical stability**: IEEE 754-compliant log-probability arithmetic
- **Scalable inference**: Support for large models with thousands of parameters

Benchmark on your hardware:

```bash
cargo bench
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/alexandernodeland/fugue.git
cd fugue
cargo test
cargo test --doc
cargo run --example gaussian_mean
```

### Running Tests

```bash
# Unit tests
cargo test

# Integration tests  
cargo test --test '*'

# Property-based tests
cargo test property_tests

# Documentation tests
cargo test --doc
```

## 📖 Documentation

- **[API Documentation](https://docs.rs/.../fugue)** - Complete API reference
- **[Examples](examples/)** - Practical usage examples
- **[Core Module Guide](src/core/README.md)** - Deep dive into Model types
- **[Inference Guide](src/inference/README.md)** - Inference algorithms overview

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Fugue draws inspiration from:

- **[Gen.jl](https://github.com/probcomp/Gen.jl)** - General-purpose probabilistic programming in Julia
- **[WebPPL](https://webppl.org/)** - Functional probabilistic programming

## 🔗 Citation

If you use Fugue in your research, please cite:

```bibtex
@software{fugue2024,
  title = {Fugue: Production-Ready Monadic Probabilistic Programming for Rust},
  author = {Alexander Nodeland},
  url = {https://github.com/alexandernodeland/fugue},
  version = {0.3.0},
  year = {2024}
}
```

---

**Built with ❤️ in Rust** | [Website](https://github.com/alexandernodeland/fugue) | [Documentation](https://docs.rs/fugue) | [Crates.io](https://crates.io/crates/fugue)
