# Fugue

[![Crates.io](https://img.shields.io/crates/v/fugue.svg)](https://crates.io/crates/fugue)
[![Documentation](https://docs.rs/fugue/badge.svg)](https://docs.rs/fugue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/alexnodeland/fugue/actions/workflows/ci-develop.yml/badge.svg)](https://github.com/alexnodeland/fugue/actions/workflows/ci-develop.yml)
[![codecov](https://codecov.io/gh/alexnodeland/fugue/branch/develop/graph/badge.svg?token=BDJ5OB6GOB)](https://codecov.io/gh/alexnodeland/fugue)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-blue.svg)](https://www.rust-lang.org)

A **production-ready**, **monadic probabilistic programming library** for Rust. Write elegant probabilistic programs by composing `Model` values in direct style; execute them with pluggable interpreters and state-of-the-art inference algorithms.

> Supported Rust: 1.70+ • Platforms: Linux / macOS / Windows • Crate: [`fugue` on crates.io](https://crates.io/crates/fugue)

## Features

- **Monadic PPL**: Compose probabilistic programs using pure functional abstractions
- **Type-Safe Distributions**: 10+ built-in probability distributions with natural return types  
- **Multiple Inference Methods**: MCMC, SMC, Variational Inference, ABC
- **Comprehensive Diagnostics**: R-hat convergence, effective sample size, validation
- **Production Ready**: Numerically stable algorithms with memory optimization
- **Ergonomic Macros**: Do-notation (`prob!`), vectorization (`plate!`), addressing (`addr!`)

## Why Fugue?

- 🔒 **Type-safe distributions**: natural return types (Bernoulli → `bool`, Poisson/Binomial → `u64`, Categorical → `usize`)
- 🧩 **Direct-style, monadic design**: compose `Model<T>` values with `bind/map` for explicit, readable control flow
- 🔌 **Pluggable interpreters**: prior sampling, replay, scoring, and safe variants for production robustness
- 📊 **Production diagnostics**: R-hat, ESS, validation utilities, and robust error handling
- ⚡ **Performance-minded**: memory pooling, copy-on-write traces, and numerically stable computations

## Installation

```toml
[dependencies]
fugue = "0.3.0"
```

### Quickstart

```bash
cargo add fugue
```

## Example

```rust
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

// Run inference with model defined in closure
let mut rng = StdRng::seed_from_u64(42);
let samples = adaptive_mcmc_chain(&mut rng, || {
    prob! {
        let mu <- sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap());
        observe(addr!("y"), Normal::new(mu, 0.5).unwrap(), 1.2);
        pure(mu)
    }
}, 1000, 500);

let mu_values: Vec<f64> = samples.iter()
    .filter_map(|(_, trace)| trace.get_f64(&addr!("mu")))
    .collect();
```

## Documentation

- **[User Guide](https://alexandernodeland.github.io/fugue/)** - Comprehensive tutorials and examples
- **[API Reference](https://docs.rs/fugue)** - Complete API documentation
- **Examples** - See `examples/` directory

## Community

- **Issues & Bugs**: Use [GitHub Issues](https://github.com/alexandernodeland/fugue/issues)
- **Feature Requests**: Open an issue with the `enhancement` label

## Roadmap

This project is an ongoing exploration of probabilistic programming in Rust. While many pieces are production-leaning, parts may not be 100% complete or correct yet. I’m steadily working toward a more robust implementation and broader feature set.

Planned focus areas:

- Strengthening core correctness and numerical stability
- Expanding distribution and inference coverage
- API refinements and stability guarantees
- Improved documentation, diagnostics, and examples

## Contributing

Contributions welcome! See our [contributing guidelines](.github/CONTRIBUTING.md).

```bash
git clone https://github.com/alexandernodeland/fugue.git
cd fugue && cargo test
```

## License

Licensed under the [MIT License](LICENSE).

---

Built with Rust • Monadic PPL • Type-safe distributions
