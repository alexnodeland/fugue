<div align="center">

# 🎼 Fugue User Guide

<img src="fugue-logo.svg" alt="Fugue Logo" width="200" height="200">

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

---

```admonish info title="👋 Welcome"

Check out these resources to get started:

- [Getting Started](./getting-started/README.md) - Installation and first steps
- [How-To Guides](./how-to/README.md) - Practical guides for common tasks
- [Tutorials](./tutorials/README.md) - In-depth learning materials
- [API Reference](./api-reference.md) - Complete API documentation
```

## About Fugue

- 🧩 **Monadic PPL**: Compose probabilistic programs using pure functional abstractions
- 🔒 **Type-Safe Distributions**: 17 built-in probability distributions with natural return types
- 📊 **Multiple Inference Methods**: MCMC, HMC, SMC, Variational Inference, ABC (see [Advanced Inference](./tutorials/advanced-inference/README.md))
- 🔍 **Comprehensive Diagnostics**: R-hat convergence, effective sample size, validation
- ⚡ **Numerically Stable**: Log-space computations and memory optimization throughout
- ✨ **Ergonomic Macros**: Do-notation (`prob!`), vectorization (`plate!`), addressing (`addr!`)

```admonish note title="🧪 Where Fugue stands today"
Fugue is 0.1.x: pre-1.0, actively developed, with no SemVer stability guarantee yet and a single primary maintainer. It's extensively tested (unit, integration, and statistical regression tests against closed-form posteriors), but that's a different claim from "production-ready" — pin an exact version and expect breaking API changes between 0.1.x releases as the design settles.
```

## Installation

```toml
[dependencies]
fugue-ppl = "0.1.0"
```

---

```admonish note title="🔍 More Resources"

- [GitHub Repository](https://github.com/alexnodeland/fugue)
- [API Reference](https://docs.rs/fugue-ppl)
- [Zotero](https://www.zotero.org/groups/6138134/fugue/library)
```
