# Fugue

[![Crates.io](https://img.shields.io/crates/v/fugue.svg)](https://crates.io/crates/fugue)
[![Documentation](https://docs.rs/fugue/badge.svg)](https://docs.rs/fugue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-blue.svg)](https://www.rust-lang.org)

A **production-ready**, **monadic probabilistic programming library** for Rust. Write elegant probabilistic programs by composing `Model` values in direct style; execute them with pluggable interpreters and state-of-the-art inference algorithms.

## ✨ Features

- 🎯 **Monadic PPL**: Compose probabilistic programs using pure functional abstractions
- 🔢 **Type-Safe Distributions**: 10+ built-in probability distributions with natural return types
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

fn bayesian_regression(x_data: &[f64], y_data: &[f64]) -> FugueResult<Model<(f64, f64)>> {
    Ok(prob! {
        // Priors - using safe constructors
        let slope <- sample(addr!("slope"), Normal::new(0.0, 1.0)?);
        let intercept <- sample(addr!("intercept"), Normal::new(0.0, 1.0)?);
        let noise <- sample(addr!("noise"), LogNormal::new(0.0, 0.5)?);

        // Likelihood
        for (i, (&x, &y)) in x_data.iter().zip(y_data.iter()).enumerate() {
            let y_pred = slope * x + intercept;
            observe(addr!("y", i), Normal::new(y_pred, noise)?, y);
        }

        pure((slope, intercept))
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y_data = vec![2.1, 3.9, 6.1, 8.0, 9.9];

    let mut rng = StdRng::seed_from_u64(42);

    // Run adaptive MCMC
    let samples = adaptive_mcmc_chain(
        &mut rng,
        || bayesian_regression(&x_data, &y_data).unwrap(),
        1000,  // samples
        500,   // warmup
    );

    // Extract results using type-safe accessors
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

## 🎯 Type Safety Revolution

Fugue features a **fully type-safe distribution system** that eliminates common probabilistic programming pitfalls:

### Before (Error-Prone)

```rust
sample(addr!("coin"), Bernoulli { p: 0.5 })
    .bind(|coin_result| {
        // ❌ Error-prone: floating point comparison
        if coin_result == 1.0 {
            pure("heads")
        } else {
            pure("tails")
        }
    })
```

### After (Type-Safe)

```rust
sample(addr!("coin"), Bernoulli { p: 0.5 })
    .bind(|is_heads| {
        // ✅ Natural: direct boolean usage, compiler-enforced
        if is_heads {
            pure("heads")
        } else {
            pure("tails")
        }
    })
```

### 🔥 Key Improvements

- **Bernoulli** → `bool` (no more `== 1.0` comparisons)
- **Poisson/Binomial** → `u64` (natural counting, no casting)
- **Categorical** → `usize` (safe array indexing)
- **Compiler guarantees** type correctness throughout

## 📚 Core Concepts

### Models as First-Class Values

Fugue represents probabilistic programs as `Model<A>` values that can be composed, transformed, and reused:

```rust
use fugue::*;

// Pure deterministic computation
let model1 = pure(42.0);

// Type-safe probabilistic sampling with safe constructors
let normal_sample: Model<f64> = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
let coin_flip: Model<bool> = sample(addr!("coin"), Bernoulli::new(0.5).unwrap());
let event_count: Model<u64> = sample(addr!("count"), Poisson::new(3.0).unwrap());
let category_choice: Model<usize> = sample(addr!("choice"), Categorical::new(
    vec![0.3, 0.5, 0.2]
).unwrap());

// Type-safe observations
let obs1 = observe(addr!("y"), Normal::new(0.0, 1.0).unwrap(), 2.5);       // f64
let obs2 = observe(addr!("success"), Bernoulli::new(0.7).unwrap(), true);   // bool
let obs3 = observe(addr!("events"), Poisson::new(4.0).unwrap(), 7u64);      // u64
let obs4 = observe(addr!("pick"), Categorical::new(vec![0.4, 0.6]).unwrap(), 1usize); // usize

// Monadic composition with type safety
let composed = coin_flip.bind(|is_heads| {
    if is_heads {
        sample(addr!("bonus"), Poisson { lambda: 5.0 })
            .map(|count| format!("Heads! Bonus: {}", count))
    } else {
        pure("Tails!".to_string())
    }
});
```

### Do-Notation with `prob!`

Write probabilistic programs in an imperative style:

```rust
let mixture_model = prob! {
    let z <- sample(addr!("component"), Bernoulli::new(0.3).unwrap());  // Returns bool!
    let mu = if z { -2.0 } else { 2.0 };  // Natural boolean usage
    let x <- sample(addr!("x"), Normal::new(mu, 1.0).unwrap());
    observe(addr!("y"), Normal::new(x, 0.1).unwrap(), observed_value);
    pure(x)
};
```

### Vectorized Operations with `plate!`

Efficiently handle collections of random variables:

```rust
// Generate 100 independent samples
let samples = plate!(i in 0..100 => {
    sample(addr!("x", i), Normal::new(0.0, 1.0).unwrap())
});

// Hierarchical model with shared parameters
let hierarchical = prob! {
    let global_mu <- sample(addr!("global_mu"), Normal::new(0.0, 1.0).unwrap());
    let local_effects <- plate!(i in 0..10 => {
        sample(addr!("local", i), Normal::new(global_mu, 0.1).unwrap())
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

| **Distribution** | **Parameters**  | **Return Type** | **Support**      | **Usage**                                    |
| ---------------- | --------------- | --------------- | ---------------- | -------------------------------------------- |
| `Normal`         | `mu`, `sigma`   | `f64`           | ℝ                | `Normal { mu: 0.0, sigma: 1.0 }`             |
| `LogNormal`      | `mu`, `sigma`   | `f64`           | ℝ⁺               | `LogNormal { mu: 0.0, sigma: 1.0 }`          |
| `Uniform`        | `low`, `high`   | `f64`           | [low, high]      | `Uniform { low: 0.0, high: 1.0 }`            |
| `Exponential`    | `rate`          | `f64`           | ℝ⁺               | `Exponential { rate: 1.0 }`                  |
| `Beta`           | `alpha`, `beta` | `f64`           | [0, 1]           | `Beta { alpha: 2.0, beta: 3.0 }`             |
| `Gamma`          | `shape`, `rate` | `f64`           | ℝ⁺               | `Gamma { shape: 2.0, rate: 1.0 }`            |
| `Bernoulli`      | `p`             | **`bool`**      | {false, true}    | `Bernoulli { p: 0.3 }`                       |
| `Binomial`       | `n`, `p`        | **`u64`**       | {0, 1, ..., n}   | `Binomial { n: 10, p: 0.5 }`                 |
| `Categorical`    | `probs`         | **`usize`**     | {0, 1, ..., k-1} | `Categorical { probs: vec![0.2, 0.3, 0.5] }` |
| `Poisson`        | `lambda`        | **`u64`**       | ℕ                | `Poisson { lambda: 2.0 }`                    |

### 🎯 Type Safety Benefits

- **`Bernoulli`** returns `bool` - no more `if sample == 1.0` comparisons!
- **`Poisson`/`Binomial`** return `u64` - natural counting with no casting needed
- **`Categorical`** returns `usize` - safe array indexing without conversion
- **Continuous distributions** return `f64` as appropriate
- **Compiler guarantees** - type errors caught at compile time

All distributions include automatic parameter validation and numerical stability checks.

## 🛠️ Advanced Features

### Custom Interpreters

Implement your own model interpreters with full type safety:

```rust
struct CustomHandler {
    // Your state here
}

impl Handler for CustomHandler {
    fn on_sample_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>) -> f64 {
        // Handle continuous distributions
    }

    fn on_sample_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>) -> bool {
        // Handle Bernoulli - returns bool directly!
    }

    fn on_sample_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>) -> u64 {
        // Handle Poisson/Binomial - returns counts as u64
    }

    fn on_sample_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>) -> usize {
        // Handle Categorical - returns indices as usize
    }

    fn on_observe_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>, value: f64) {
        // Observe continuous values
    }

    fn on_observe_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>, value: bool) {
        // Observe boolean outcomes
    }

    // ... other methods for u64, usize observations and factors
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
// Efficient trace operations with type-safe values
let mut trace = Trace::default();
trace.insert_choice(addr!("x"), ChoiceValue::F64(1.5), 0.0);       // Continuous
trace.insert_choice(addr!("coin"), ChoiceValue::Bool(true), -0.5);  // Boolean
trace.insert_choice(addr!("count"), ChoiceValue::U64(7), -2.1);     // Count
trace.insert_choice(addr!("choice"), ChoiceValue::Usize(2), -1.6);  // Index

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
- **[`fully_type_safe.rs`](examples/fully_type_safe.rs)** - **NEW**: Demonstrates type-safe distributions
- **[`simple_mixture.rs`](examples/simple_mixture.rs)** - Updated with type-safe boolean usage

Run examples with:

```bash
cargo run --example gaussian_mean -- --obs 2.5 --seed 42
cargo run --example improved_gaussian_mean -- --obs 1.5 --n-samples 2000 --validate
cargo run --example fully_type_safe -- --obs 2.0  # See type-safe distributions in action!
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

We welcome contributions! Please see our [Contributing Guidelines](.github/CONTRIBUTING.md) for details.

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

### 🚀 Getting Started

- **[📚 Complete Documentation](docs/README.md)** - Comprehensive guides and tutorials
- **[⚡ Installation Guide](docs/src/getting-started/installation.md)** - Get up and running quickly
- **[🎯 Your First Model](docs/src/getting-started/your-first-model.md)** - Build your first probabilistic model

### 📚 Learning Resources

- **[🎓 Tutorials](docs/src/tutorials/)** - Step-by-step learning:

  - [Bayesian Coin Flip](docs/src/tutorials/bayesian-coin-flip.md) (Beginner)
  - [Linear Regression](docs/src/tutorials/linear-regression.md) (Intermediate)
  - [Mixture Models](docs/src/tutorials/mixture-models.md) (Intermediate)
  - [Hierarchical Models](docs/src/tutorials/hierarchical-models.md) (Advanced)

- **[🛠️ How-To Guides](docs/src/how-to/)** - Practical solutions:
  - [Working with Distributions](docs/src/how-to/working-with-distributions.md)
  - [Using Macros](docs/src/how-to/using-macros.md) (`prob!`, `plate!`, `addr!`)
  - [Trace Manipulation](docs/src/how-to/trace-manipulation.md)
  - [Custom Handlers](docs/src/how-to/custom-handlers.md)
  - [Debugging Models](docs/src/how-to/debugging-models.md)

### 📖 Reference

- **[API Documentation](https://docs.rs/fugue)** - Complete API reference
- **[Examples](examples/)** - Practical usage examples
- **[Core Module Guide](src/core/README.md)** - Deep dive into Model types
- **[Inference Guide](src/inference/README.md)** - Inference algorithms overview

## 📄 License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## 📄 Contributing

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you shall be dual licensed as above, without any additional terms or conditions.

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
