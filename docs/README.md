# Fugue Documentation

Welcome to the Fugue probabilistic programming library documentation! This guide will help you learn Fugue from the ground up, whether you're new to probabilistic programming or coming from other PPL frameworks.

## 📚 Learning Path

### 🚀 Getting Started
Start here if you're new to Fugue or probabilistic programming:

1. **[Installation](getting-started/installation.md)** - Set up Fugue in your Rust project
2. **[Your First Model](getting-started/your-first-model.md)** - Build and run your first probabilistic model
3. **[Understanding Models](getting-started/understanding-models.md)** - Learn about `Model<T>`, composition, and monadic operations
4. **[Basic Inference](getting-started/basic-inference.md)** - Run inference on your models with built-in algorithms

### 🛠️ How-To Guides
Practical guides for specific tasks:

- **[Working with Distributions](how-to/working-with-distributions.md)** - Master Fugue's type-safe distribution system
- **[Using Macros](how-to/using-macros.md)** - Leverage `prob!`, `plate!`, and `addr!` for cleaner code
- **[Trace Manipulation](how-to/trace-manipulation.md)** - Understanding and working with execution traces
- **[Custom Handlers](how-to/custom-handlers.md)** - Implement your own model interpreters
- **[Debugging Models](how-to/debugging-models.md)** - Debug and validate your probabilistic models

### 🎓 Tutorials
Complete worked examples building from simple to complex:

1. **[Bayesian Coin Flip](tutorials/bayesian-coin-flip.md)** - *Beginner* - Learn basic Bayesian inference
2. **[Linear Regression](tutorials/linear-regression.md)** - *Intermediate* - Build a Bayesian linear regression model
3. **[Mixture Models](tutorials/mixture-models.md)** - *Intermediate* - Work with discrete latent variables
4. **[Hierarchical Models](tutorials/hierarchical-models.md)** - *Advanced* - Multi-level modeling with shared parameters

## 🔗 Quick Links

- **[Main README](../README.md)** - Project overview and quick start
- **[Examples](../examples/)** - Complete runnable examples
- **[API Documentation](https://docs.rs/fugue)** - Detailed API reference
- **[Core Module](../src/core/README.md)** - Deep dive into Model types and composition
- **[Inference Module](../src/inference/README.md)** - Inference algorithms overview
- **[Runtime Module](../src/runtime/README.md)** - Handlers and execution

## 🎯 Key Features

**Fugue** is a production-ready, monadic probabilistic programming library for Rust featuring:

- 🎯 **Type-Safe Distributions** - Distributions return natural types (`bool`, `u64`, `usize`, `f64`)
- 🔢 **Monadic Composition** - Build complex models by composing simple `Model<T>` values
- 🎰 **Multiple Inference Methods** - MCMC, SMC, Variational Inference, ABC
- 🎛️ **Ergonomic Macros** - Do-notation style programming with `prob!`
- ⚡ **High Performance** - Zero-cost abstractions with pluggable interpreters

## 💡 Need Help?

- Check the **[How-To Guides](how-to/)** for specific tasks
- Browse **[Complete Examples](../examples/)** for working code
- Read the **[API Documentation](https://docs.rs/fugue)** for detailed reference
- Review **[Module READMEs](../src/)** for implementation details

---

*Ready to get started? Begin with **[Installation](getting-started/installation.md)**!*