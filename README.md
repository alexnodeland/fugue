# Fugue

[![Crates.io](https://img.shields.io/crates/v/fugue.svg)](https://crates.io/crates/fugue)
[![Documentation](https://docs.rs/fugue/badge.svg)](https://docs.rs/fugue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![codecov](https://codecov.io/gh/alexnodeland/fugue/graph/badge.svg?token=BDJ5OB6GOB)](https://codecov.io/gh/alexnodeland/fugue)
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

fn bayesian_regression(x_data: &[f64], y_data: &[f64]) -> Model<(f64, f64)> {
    let x_vec = x_data.to_vec(); // Clone to avoid lifetime issues in doctest
    let y_vec = y_data.to_vec(); // Clone to avoid lifetime issues in doctest
    
    prob! {
        // Priors - using safe constructors
        let slope <- sample(addr!("slope"), Normal::new(0.0, 1.0).unwrap());
        let intercept <- sample(addr!("intercept"), Normal::new(0.0, 1.0).unwrap());
        let noise <- sample(addr!("noise"), LogNormal::new(0.0, 0.5).unwrap());

        // Likelihood - handle observations sequentially  
        let _observations <- sequence_vec(x_vec.iter().zip(y_vec.iter()).enumerate().map(|(i, (&x, &y))| {
            let y_pred = slope * x + intercept;
            // Ensure noise is positive for Normal distribution
            let safe_noise = noise.abs().max(1e-6);
            observe(addr!("y", i), Normal::new(y_pred, safe_noise).unwrap(), y)
        }).collect());

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
# use fugue::*;
let _example = sample(addr!("coin"), Bernoulli::new(0.5).unwrap())
    .bind(|coin_result| {
        // ❌ This would be error-prone if this returned f64 instead of bool
        // But Fugue returns bool, so coin_result is naturally a boolean
        if coin_result {
            pure("heads")
        } else {
            pure("tails")
        }
    });
```

### After (Type-Safe)

```rust
# use fugue::*;
let _example = sample(addr!("coin"), Bernoulli::new(0.5).unwrap())
    .bind(|is_heads| {
        // ✅ Natural: direct boolean usage, compiler-enforced
        if is_heads {
            pure("heads")
        } else {
            pure("tails")
        }
    });
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
        sample(addr!("bonus"), Poisson::new(5.0).unwrap())
            .map(|count| format!("Heads! Bonus: {}", count))
    } else {
        pure("Tails!".to_string())
    }
});
```

### Do-Notation with `prob!`

Write probabilistic programs in an imperative style:

```rust
# use fugue::*;
let observed_value = 1.5; // Example observed value
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
# use fugue::*;
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
# use fugue::*;
# use rand::rngs::StdRng;
# use rand::SeedableRng;
# fn your_model() -> Model<f64> { sample(addr!("x"), Normal::new(0.0, 1.0).unwrap()) }
# let mut rng = StdRng::seed_from_u64(42);
# let n_samples = 1000;
# let n_warmup = 500;
// Adaptive Metropolis-Hastings with convergence diagnostics
let samples = adaptive_mcmc_chain(
    &mut rng,
    || your_model(),
    n_samples,
    n_warmup,
);

// Extract parameter values for diagnostics
let parameter_values: Vec<f64> = samples.iter()
    .filter_map(|(_, trace)| trace.get_f64(&addr!("x")))
    .collect();
    
// Compute R-hat for convergence diagnostics (simplified example)
println!("Collected {} samples", parameter_values.len());
```

### Sequential Monte Carlo (SMC)

```rust
# use fugue::*;
# use rand::rngs::StdRng;
# use rand::SeedableRng;
# fn your_model() -> Model<f64> { sample(addr!("x"), Normal::new(0.0, 1.0).unwrap()) }
# let mut rng = StdRng::seed_from_u64(42);
let config = SMCConfig {
    resampling_method: ResamplingMethod::Systematic,
    ess_threshold: 0.5,
    rejuvenation_steps: 5,
};

let particles = adaptive_smc(&mut rng, 1000, || your_model(), config);
let ess = effective_sample_size(&particles);
```

### Variational Inference (VI)

```rust
# use fugue::*;
# use rand::rngs::StdRng;
# use rand::SeedableRng;
# use std::collections::HashMap;
# fn your_model() -> Model<f64> { sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap()) }
# let mut rng = StdRng::seed_from_u64(42);
// Mean-field variational approximation
let mut guide = MeanFieldGuide {
    params: HashMap::new()
};
guide.params.insert(addr!("mu"), VariationalParam::Normal { mu: 0.0, log_sigma: 0.0 });

let optimized_guide = optimize_meanfield_vi(
    &mut rng,
    || your_model(),
    guide,
    1000,  // max iterations
    100,   // samples per iteration
    0.01,  // learning rate
);
```

### Approximate Bayesian Computation (ABC)

```rust
# use fugue::*;
# use rand::rngs::StdRng;
# use rand::SeedableRng;
# fn your_model() -> Model<f64> { sample(addr!("x"), Normal::new(0.0, 1.0).unwrap()) }
# let mut rng = StdRng::seed_from_u64(42);
# let simulator_fn = |trace: &Trace| vec![trace.get_f64(&addr!("x")).unwrap_or(0.0)];
# let observed_data = vec![2.0];
# let distance_fn = &EuclideanDistance;
# let tolerance = 0.1;
# let max_samples = 1000;
// Likelihood-free inference
let samples = abc_rejection(
    &mut rng,
    || your_model(),
    simulator_fn,
    &observed_data,
    distance_fn,
    tolerance,
    max_samples,
);
```

## 📊 Built-in Distributions

| **Distribution** | **Parameters**  | **Return Type** | **Support**      | **Usage**                                    |
| ---------------- | --------------- | --------------- | ---------------- | -------------------------------------------- |
| `Normal`         | `mu`, `sigma`   | `f64`           | ℝ                | `Normal::new(0.0, 1.0).unwrap()`             |
| `LogNormal`      | `mu`, `sigma`   | `f64`           | ℝ⁺               | `LogNormal::new(0.0, 1.0).unwrap()`          |
| `Uniform`        | `low`, `high`   | `f64`           | [low, high]      | `Uniform::new(0.0, 1.0).unwrap()`            |
| `Exponential`    | `rate`          | `f64`           | ℝ⁺               | `Exponential::new(1.0).unwrap()`                  |
| `Beta`           | `alpha`, `beta` | `f64`           | [0, 1]           | `Beta::new(2.0, 3.0).unwrap()`             |
| `Gamma`          | `shape`, `rate` | `f64`           | ℝ⁺               | `Gamma::new(2.0, 1.0).unwrap()`            |
| `Bernoulli`      | `p`             | **`bool`**      | {false, true}    | `Bernoulli::new(0.3).unwrap()`                       |
| `Binomial`       | `n`, `p`        | **`u64`**       | {0, 1, ..., n}   | `Binomial::new(10, 0.5).unwrap()`                 |
| `Categorical`    | `probs`         | **`usize`**     | {0, 1, ..., k-1} | `Categorical::new(vec![0.2, 0.3, 0.5]).unwrap()` |
| `Poisson`        | `lambda`        | **`u64`**       | ℕ                | `Poisson::new(2.0).unwrap()`                    |

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
# use fugue::*;
# use rand::Rng;
struct CustomHandler<R: Rng> {
    rng: R,
    // Your state here
}

impl<R: Rng> Handler for CustomHandler<R> {
    fn on_sample_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>) -> f64 {
        // Handle continuous distributions
        dist.sample(&mut self.rng)
    }

    fn on_sample_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>) -> bool {
        // Handle Bernoulli - returns bool directly!
        dist.sample(&mut self.rng)
    }

    fn on_sample_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>) -> u64 {
        // Handle Poisson/Binomial - returns counts as u64
        dist.sample(&mut self.rng)
    }

    fn on_sample_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>) -> usize {
        // Handle Categorical - returns indices as usize
        dist.sample(&mut self.rng)
    }

    fn on_observe_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>, value: f64) {
        // Observe continuous values
    }

    fn on_observe_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>, value: bool) {
        // Observe boolean outcomes
    }

    fn on_observe_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>, value: u64) {
        // Observe u64 values
    }

    fn on_observe_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>, value: usize) {
        // Observe usize values  
    }

    fn on_factor(&mut self, logw: f64) {
        // Handle factors
    }

    fn finish(self) -> Trace {
        Trace::default()
    }
}
```

### Hierarchical Addressing

Organize complex models with scoped addresses:

```rust
# use fugue::*;
let hierarchical = prob! {
    let global_params <- plate!(layer in 0..3 => {
        sample(scoped_addr!("layer", layer, "weight"), Normal::new(0.0, 1.0).unwrap())
    });
    // ... rest of model
    pure(global_params)
};
```

### Memory-Efficient Trace Manipulation

```rust
# use fugue::*;
// Efficient trace operations with type-safe values
let mut trace = Trace::default();
trace.insert_choice(addr!("x"), ChoiceValue::F64(1.5), 0.0);       // Continuous
trace.insert_choice(addr!("coin"), ChoiceValue::Bool(true), -0.5);  // Boolean
trace.insert_choice(addr!("count"), ChoiceValue::U64(7), -2.1);     // Count
trace.insert_choice(addr!("choice"), ChoiceValue::Usize(2), -1.6);  // Index

// Trace validation and debugging
println!("Total log weight: {:.4}", trace.total_log_weight());
```

## 🧪 Validation & Testing

Fugue includes extensive validation against analytical solutions:

```rust
# use fugue::*;
# use rand::rngs::StdRng;
# use rand::SeedableRng;
# use fugue::inference::validation::ConjugateNormalConfig;
# let mut rng = StdRng::seed_from_u64(42);
# let config = ConjugateNormalConfig {
#     prior_mu: 0.0,
#     prior_sigma: 1.0,
#     likelihood_sigma: 0.5,
#     observation: 2.0,
#     n_samples: 1000,
#     n_warmup: 500,
# };
// Validate MCMC against known posterior  
let prior_mu = config.prior_mu;
let prior_sigma = config.prior_sigma;
let likelihood_sigma = config.likelihood_sigma;
let observation = config.observation;

let validation = test_conjugate_normal_model(
    &mut rng,
    move |rng, n_samples, n_warmup| {
        adaptive_mcmc_chain(rng, move || {
            sample(addr!("mu"), Normal::new(prior_mu, prior_sigma).unwrap())
                .bind(move |mu| {
                    observe(addr!("y"), Normal::new(mu, likelihood_sigma).unwrap(), observation);
                    pure(mu)
                })
        }, n_samples, n_warmup)
    },
    config,
);

println!("Validation complete: {}", validation.is_valid());
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

- **[⚡ Installation Guide](docs/src/getting-started/installation.md)** - Get up and running quickly
- **[🎯 Your First Model](docs/src/getting-started/your-first-model.md)** - Build your first probabilistic model

### 📚 Learning Resources

- **[🎓 Tutorials](docs/src/tutorials/README.md)** - Step-by-step learning:
  - **Foundation Tutorials**:
    - [Bayesian Coin Flip](docs/src/tutorials/foundation/bayesian-coin-flip.md) (Beginner)
    - [Type Safety Features](docs/src/tutorials/foundation/type-safety-features.md) (Beginner)
    - [Trace Manipulation](docs/src/tutorials/foundation/trace-manipulation.md) (Beginner)
  - **Statistical Modeling**:
    - [Linear Regression](docs/src/tutorials/statistical-modeling/linear-regression.md) (Intermediate)
    - [Classification](docs/src/tutorials/statistical-modeling/classification.md) (Intermediate)
    - [Mixture Models](docs/src/tutorials/statistical-modeling/mixture-models.md) (Intermediate)
    - [Hierarchical Models](docs/src/tutorials/statistical-modeling/hierarchical-models.md) (Advanced)
  - **Advanced Applications**:
    - [Time Series & Forecasting](docs/src/tutorials/advanced-applications/time-series-forecasting.md) (Advanced)
    - [Model Comparison & Selection](docs/src/tutorials/advanced-applications/model-comparison-selection.md) (Advanced)
    - [Advanced Inference](docs/src/tutorials/advanced-applications/advanced-inference.md) (Advanced)

- **[🛠️ How-To Guides](docs/src/how-to/README.md)** - Practical solutions:
  - [Working with Distributions](docs/src/how-to/working-with-distributions.md)
  - [Building Complex Models](docs/src/how-to/building-complex-models.md)
  - [Optimizing Performance](docs/src/how-to/optimizing-performance.md)
  - [Debugging Models](docs/src/how-to/debugging-models.md)
  - [Custom Handlers](docs/src/how-to/custom-handlers.md)
  - [Production Deployment](docs/src/how-to/production-deployment.md)

### 📖 Reference

- **[API Documentation](https://docs.rs/fugue)** - Complete API reference

## 📄 License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

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
