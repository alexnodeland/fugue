# Fugue PPL Documentation

Welcome to the Fugue PPL documentation! Fugue is a production-ready, monadic probabilistic programming library for Rust.

## Quick Links

- [Getting Started](./getting-started/README.md) - Installation and first steps
- [How-To Guides](./how-to/README.md) - Practical guides for common tasks
- [Tutorials](./tutorials/README.md) - In-depth learning materials
- [API Reference](./api-reference.md) - Complete API documentation

## About Fugue

Fugue provides:

- **Monadic PPL**: Compose probabilistic programs using pure functional abstractions
- **Type-Safe Distributions**: 10+ built-in probability distributions with natural return types  
- **Multiple Inference Methods**: MCMC, SMC, Variational Inference, ABC
- **Comprehensive Diagnostics**: R-hat convergence, effective sample size, validation
- **Production Ready**: Numerically stable algorithms with memory optimization
- **Ergonomic Macros**: Do-notation (`prob!`), vectorization (`plate!`), addressing (`addr!`)

## Installation

```toml
[dependencies]
fugue-ppl = "0.1.0"
```

For more details, see the [full README on GitHub](https://github.com/alexnodeland/fugue).
