<div align="center">

# 🎼 Fugue

<img src="assets/fugue-logo.svg" alt="Fugue Logo" width="200" height="200">

**A type-safe, monadic probabilistic programming library for Rust — pre-1.0 and actively developed**

*Write elegant probabilistic programs by composing `Model` values in direct style; execute them with pluggable interpreters and state-of-the-art inference algorithms.*

[![Rust](https://img.shields.io/badge/rust-1.87%2B-blue.svg)](https://www.rust-lang.org)
[![Crates.io](https://img.shields.io/crates/v/fugue-ppl.svg)](https://crates.io/crates/fugue-ppl)
[![Dev Docs](https://docs.rs/fugue-ppl/badge.svg)](https://docs.rs/fugue-ppl)
[![User Docs](https://img.shields.io/badge/guides-fugue.run-blue)](https://fugue.run)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/alexnodeland/fugue/actions/workflows/ci.yml/badge.svg)](https://github.com/alexnodeland/fugue/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/alexnodeland/fugue/branch/main/graph/badge.svg?token=BDJ5OB6GOB)](https://codecov.io/gh/alexnodeland/fugue)
[![Downloads](https://img.shields.io/crates/d/fugue-ppl?logo=rust)](https://crates.io/crates/fugue-ppl)
[![Zotero](https://img.shields.io/badge/Zotero-Fugue-red?style=flat-square&logo=zotero&logoColor=CC2936)](https://www.zotero.org/groups/6138134/fugue/library)
[![Discord](https://img.shields.io/discord/1412802057437712426?logo=discord&label=discord)](https://discord.gg/QAcF7Nwr)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/alexnodeland/fugue)

**Supported Rust:** 1.87+ • **Platforms:** Linux / macOS / Windows • **Crate:** [`fugue-ppl` on crates.io](https://crates.io/crates/fugue-ppl)

</div>

## ✨ Features

- **Monadic PPL**: Compose probabilistic programs using pure functional abstractions
- **Type-Safe Distributions**: 17 built-in probability distributions with natural return types
- **Multiple Inference Methods**: MCMC, HMC, SMC, Variational Inference, ABC
- **Comprehensive Diagnostics**: R-hat convergence, effective sample size, validation
- **Numerically Stable**: Log-space computations throughout for robust probability arithmetic
- **Ergonomic Macros**: Do-notation (`prob!`), vectorization (`plate!`), addressing (`addr!`)

## 🤔 Why Fugue?

- 🔒 **Type-safe distributions**: natural return types (Bernoulli → `bool`, Poisson/Binomial → `u64`, Categorical → `usize`)
- 🧩 **Direct-style, monadic design**: compose `Model<T>` values with `bind/map` for explicit, readable control flow
- 🔌 **Pluggable interpreters**: prior sampling, replay, scoring, and safe variants
- 📊 **Diagnostics**: R-hat, ESS, validation utilities, and a structured error taxonomy (see [`error`](https://docs.rs/fugue-ppl/latest/fugue/error/))
- ⚡ **Performance-minded**: O(1), allocation-free address clones (`Arc<str>` with a cached hash) and numerically stable log-space computations

## 📦 Distributions

Bernoulli, Beta, Binomial, Categorical, Cauchy, ChiSquared, DiscreteUniform, Exponential, Gamma, InverseGamma, Laplace, LogNormal, Normal, Poisson, StudentT, Uniform, Weibull — 17 in total, each with natural return types and validated parameters.

## 🧪 Where Fugue stands today

Fugue is 0.1.x: pre-1.0, actively developed, with no SemVer stability guarantee yet and a single primary maintainer (see Roadmap, below). It's extensively tested — hundreds of unit, integration, and property-based tests, including statistical regression tests against closed-form posteriors — but that's not the same claim as "production-ready." Treat it as a serious, honestly-scoped research-grade PPL: pin an exact version, read the [CHANGELOG](CHANGELOG.md) before upgrading, and expect breaking API changes between 0.1.x releases as the design settles.

## 📦 Installation

```toml
[dependencies]
fugue-ppl = "0.1.0"
```

### Quickstart

```bash
cargo add fugue-ppl
```

## 💡 Example

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

## 📚 Documentation

- **[User Guide](https://fugue.run/)** - Comprehensive tutorials and examples
- **[API Reference](https://docs.rs/fugue-ppl/latest/fugue/)** - Complete API documentation
- **Examples** - See the `examples/` directory, including one runnable example per inference method:
  - `adaptive_mcmc_chain` - most foundation/statistical-modeling examples (e.g. `bayesian_coin_flip.rs`)
  - `hmc_chain` (HMC) - see the [`hmc` module rustdoc](https://docs.rs/fugue-ppl/latest/fugue/inference/hmc/) for a runnable doctest
  - `adaptive_smc` (SMC) - `examples/smc_inference.rs`
  - `abc_smc_weighted` (ABC) - `examples/abc_inference.rs`
  - `optimize_meanfield_vi_with_config` (VI) - `examples/vi_inference.rs`
- **[References](https://www.zotero.org/groups/6138134/fugue/library)** - Zotero library for Fugue

## 🤝 Community

- **Issues & Bugs**: Use [GitHub Issues](https://github.com/alexnodeland/fugue/issues)
- **Feature Requests**: Open an issue with the `enhancement` label
- **Discord**: Join our [Discord server](https://discord.gg/QAcF7Nwr)

## 🗺️ Roadmap

This project is an ongoing exploration of probabilistic programming in Rust. While many pieces are production-leaning, parts may not be 100% complete or correct yet. I’m steadily working toward a more robust implementation and broader feature set.

Planned focus areas:

- Strengthening core correctness and numerical stability
- Expanding distribution and inference coverage
- API refinements and stability guarantees
- Improved documentation, diagnostics, and examples

**API stability / SemVer policy:** Fugue follows [Cargo's pre-1.0 SemVer convention](https://doc.rust-lang.org/cargo/reference/semver.html#change-categories): any `0.x.y -> 0.(x+1).0` bump may contain breaking changes, and `0.x.y -> 0.x.(y+1)` is additive/non-breaking. There is no 1.0 stability commitment yet; always pin an exact version and read the [CHANGELOG](CHANGELOG.md) before upgrading the minor version.

## 🤝 Contributing

Contributions welcome! See our [contributing guidelines](.github/CONTRIBUTING.md).

```bash
git clone https://github.com/alexnodeland/fugue.git
cd fugue && cargo test
```

## 📄 License

Licensed under the [MIT License](LICENSE).

## 🔗 Citation

If you use Fugue in your research, please cite:

```bibtex
@software{fugue2025,
  title = {Fugue: Monadic Probabilistic Programming for Rust},
  author = {Alexander Nodeland},
  url = {https://github.com/alexnodeland/fugue},
  version = {0.1.0},
  year = {2025}
}
```

Or refer to the "Internal" collection in [Zotero](https://www.zotero.org/groups/6138134/fugue/library) to generate a bibliography.
