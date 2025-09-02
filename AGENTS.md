# Agent Context: Fugue Probabilistic Programming Library

## Project Overview

Fugue is a **production-ready, monadic probabilistic programming library** for Rust. It enables elegant probabilistic program composition through `Model` values executed with pluggable interpreters and state-of-the-art inference algorithms.

### Core Philosophy
- **Monadic Design**: Compose probabilistic programs using pure functional abstractions
- **Type Safety**: Natural return types for distributions (Bernoulli → `bool`, Poisson → `u64`)  
- **Production Ready**: Numerically stable algorithms with comprehensive error handling
- **Performance Focused**: Memory optimization, copy-on-write traces, efficient algorithms

## Architecture Overview

```
fugue/
├── src/           # Core library implementation
│   ├── core/      # Fundamental PPL primitives (Model, Distribution, Address)
│   ├── runtime/   # Execution engine (Handlers, Interpreters, Traces)
│   ├── inference/ # Algorithms (MCMC, SMC, VI, ABC)
│   └── error.rs   # Comprehensive error handling
├── docs/          # User documentation (mdbook)
├── examples/      # Practical usage examples
├── tests/         # Integration and API tests
└── benches/       # Performance benchmarks
```

## Key Concepts

### Model Composition
- `Model<T>`: Core probabilistic program type
- `sample()`: Draw from distributions  
- `observe()`: Condition on data
- `prob!` macro: Do-notation for monadic composition

### Address System
- Every random choice has a unique, stable address
- Format: `"name"`, `"name#index"`, `"scope::name"`, `"scope::name#index"`
- Critical for reproducibility and inference targeting

### Interpreters/Handlers
- `PriorHandler`: Forward sampling from priors
- `ReplayHandler`: Replay with specific trace values
- `ScoreGivenTrace`: Score traces for importance sampling
- Safe variants available for production use

## Development Guidelines

### Mandatory Testing Practice
**Always run tests after making changes.** This is critical for a probabilistic programming library where subtle changes can have significant statistical implications.

```bash
# Always run before committing changes
make all

# For quick feedback during development
make test

# Check test coverage
make coverage
```

### Code Style
- Follow standard Rust conventions (rustfmt, clippy)
- Prefer explicit error handling over panics
- Use type-safe abstractions over raw implementations
- Document public APIs with examples

### Testing Patterns
- Unit tests for individual components
- Integration tests for end-to-end workflows  
- Property-based testing for numerical stability
- Benchmark critical performance paths

### Domain-Specific Considerations
- **Numerical Stability**: Use log-space computations, guard against overflow/underflow
- **Reproducibility**: Ensure deterministic execution given same random seed
- **Memory Management**: Consider trace pooling for high-throughput applications
- **Error Propagation**: Preserve error context through the computation stack

## Common Tasks

### Adding New Distributions
1. Implement `Distribution<T>` trait in `src/core/distribution.rs`
2. Add validation logic and error handling
3. Include comprehensive tests with edge cases
4. Document mathematical properties and use cases

### Implementing Inference Algorithms  
1. Design around existing `Handler` infrastructure
2. Consider trace manipulation patterns
3. Implement convergence diagnostics
4. Provide both basic and production-ready variants

### Performance Optimization
1. Profile with realistic workloads first
2. Consider memory pooling for frequent allocations
3. Use copy-on-write semantics where appropriate
4. Benchmark against baseline implementations

## Dependencies & Ecosystem

### Core Dependencies
- `rand`: Random number generation and distributions
- Standard library only for core functionality

### Development Tools
- `make test`: Run test suite
- `make lint`: Run clippy linter
- `make fmt`: Format code
- `make doc`: Generate and open documentation
- `make mdbook`: Build mdbook documentation
- `make coverage`: Generate coverage report
- `make bench`: Run benchmarks
- `make all`: Run all checks (format, lint, test, coverage)

## Getting Started

```bash
# Build and test
make test

# Run examples
cargo run --example basic_modeling

# Generate documentation  
make docs-all

# Format and lint
make fmt
make lint

# Run all checks
make all
```

## Critical Areas Requiring Domain Knowledge

1. **Probabilistic Programming Theory**: Understanding of measure theory, inference algorithms
2. **Numerical Analysis**: Floating-point precision, log-space computations  
3. **Rust Systems Programming**: Memory management, zero-cost abstractions
4. **Statistical Computing**: Convergence diagnostics, sampling strategies

## Contribution Guidelines

- Changes should preserve API compatibility where possible
- New features require comprehensive tests and documentation
- Performance-critical changes need benchmark validation
- All public APIs must include usage examples

For detailed contribution guidelines, see `.github/CONTRIBUTING.md`.