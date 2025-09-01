# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-09-XX

### Added

- Initial release of Fugue, a monadic probabilistic programming library for Rust.
- **Core probabilistic programming framework**:
  - `Model<A>` abstraction for composable probabilistic programs.
  - Monadic operations: `bind`, `map`, `and_then`, `pure` for program composition.
  - `sample`, `observe`, `factor`, `guard` primitives for probabilistic modeling.
- **Type-safe distribution system** with natural return types:
  - `Bernoulli` distribution returning `bool` (eliminates `== 1.0` comparisons).
  - `Poisson` and `Binomial` distributions returning `u64` (natural counting).
  - `Categorical` distribution returning `usize` (safe array indexing).
  - Continuous distributions (`Normal`, `Beta`, `Gamma`, etc.) returning `f64`.
  - 10 built-in distributions with parameter validation and numerical stability.
- **Ergonomic macros** for probabilistic programming:
  - `prob!` macro for Haskell-style do-notation.
  - `plate!` macro for vectorized operations over collections.
  - `addr!` and `scoped_addr!` macros for hierarchical addressing.
- **Multiple inference algorithms**:
  - MCMC: Adaptive Metropolis-Hastings with convergence diagnostics.
  - SMC: Sequential Monte Carlo with multiple resampling methods.
  - VI: Mean-field variational inference with ELBO optimization.
  - ABC: Approximate Bayesian Computation with distance functions.
- **Effect handler system**:
  - `Handler` trait for pluggable model interpreters.
  - 5 built-in handlers: `PriorHandler`, `ReplayHandler`, `ScoreGivenTrace`, `SafeReplayHandler`, `SafeScoreGivenTrace`.
  - Type-safe execution preserving distribution return types.
- **Trace system** for execution history:
  - Complete recording of random choices and log-weights.
  - Type-safe value access with `get_f64()`, `get_bool()`, `get_u64()`, `get_usize()`.
  - Three-component log-weight decomposition (prior, likelihood, factors).
- **Memory optimization**:
  - Copy-on-write traces (`CowTrace`) for efficient MCMC proposals.
  - Object pooling (`TracePool`) for zero-allocation inference.
  - Efficient trace construction (`TraceBuilder`).
- **Production features**:
  - Comprehensive error handling with `FugueError` and error codes.
  - Numerically stable algorithms with overflow protection.
  - Convergence diagnostics: R-hat, effective sample size, Geweke tests.
  - Statistical validation against analytical solutions.
- **Documentation and examples**:
  - Comprehensive user guide with 20+ tutorial and how-to pages.
  - Complete API documentation with rustdoc.
  - 14 examples covering foundation concepts, statistical modeling, and advanced patterns.
  - 158+ doctests ensuring example correctness.
- **Testing infrastructure**:
  - 82+ unit tests across all modules.
  - 9+ integration tests for end-to-end workflows.
  - Property-based testing with `proptest`.
  - Continuous integration with format, lint, and test enforcement.

### Changed

- N/A (initial release)

### Deprecated

- N/A (initial release)

### Removed

- N/A (initial release)

### Fixed

- N/A (initial release)

### Security

- N/A (initial release)

[Unreleased]: https://github.com/alexandernodeland/fugue/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/alexandernodeland/fugue/releases/tag/v0.1.0
