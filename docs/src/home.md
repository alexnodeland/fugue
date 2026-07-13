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

<div class="fugue-explorable fv-inline" data-viz="dist-strip" data-dist="normal" data-params="0,1" data-caption="This page is alive — most figures in these docs run live. Touch them."></div>

```admonish tip title="🎮 New: Fugue Explorables"
Learn Fugue by **playing**, not just reading. The [Explorables](./explorables/README.md) are interactive, touchable diagrams — drag a slider and watch a posterior re-form, step an interpreter one effect at a time, roll a Hamiltonian trajectory across a density. In the tradition of Bret Victor's explorable explanations and 3Blue1Brown.

- [**Anatomy of a Probabilistic Program**](./explorables/anatomy.md) — the coin-flip Bayes loop, fully touchable
- [**The Model Is a Score**](./explorables/monad.md) — a step-debugger for the `Model` monad
- [**Random Walks in Posterior Space**](./explorables/metropolis.md) — Metropolis–Hastings by hand
- [**Rolling, Not Guessing: HMC**](./explorables/hmc.md) — Hamiltonian Monte Carlo, rolling a ball across the posterior landscape
- [**Particles That Tell Stories**](./explorables/smc.md) — sequential Monte Carlo, resampling made visible
- [**A Field Guide to Distributions**](./explorables/distributions.md) — all 17 distributions, sampled live
```

```admonish info title="👋 Welcome"

Check out these resources to get started:

- [Getting Started](./getting-started/README.md) - Installation and first steps
- [How-To Guides](./how-to/README.md) - Practical guides for common tasks
- [Tutorials](./tutorials/README.md) - In-depth learning materials
- [API Reference](./api-reference.md) - Complete API documentation
```

## About Fugue

- 🧩 **Monadic PPL**: Compose probabilistic programs as pure `Model` values, then interpret them with pluggable handlers — effects are *interpreted*, never performed by the model itself
- 🔒 **Type-Safe Distributions**: 17 built-in distributions with natural return types (`Bernoulli` → `bool`, `Poisson` → `u64`, `Categorical` → `usize`) — the type system tracks what each draw *is*
- 📊 **Multiple Inference Methods**: adaptive Metropolis–Hastings, **Hamiltonian Monte Carlo**, Sequential Monte Carlo (with an unbiased log-evidence estimate), Variational Inference (support-matched guide families), and importance-weighted ABC-SMC (see [Advanced Inference](./tutorials/advanced-inference/README.md))
- 🔍 **Comprehensive Diagnostics**: split-R̂ convergence, autocorrelation-based effective sample size, Geweke, and closed-form validation
- 🌀 **Stack-Safe Interpreter**: a trampolined runtime evaluates models in O(1) stack depth — 100 000-site programs run without overflowing
- ⚡ **Numerically Stable**: log-space computations throughout for robust probability arithmetic
- ✨ **Ergonomic Macros**: Do-notation (`prob!`), vectorization (`plate!`), addressing (`addr!`)

```admonish note title="🧪 Where Fugue stands today"
Fugue is 0.2.x: pre-1.0, actively developed, with no SemVer stability guarantee yet and a single primary maintainer. It's extensively tested (unit, integration, and statistical regression tests against closed-form posteriors), but that's a different claim from "production-ready" — pin an exact version and expect breaking API changes between 0.x releases as the design settles.
```

## Installation

```toml
[dependencies]
fugue-ppl = "0.2.0"
```

---

```admonish note title="🔍 More Resources"

- [GitHub Repository](https://github.com/alexnodeland/fugue)
- [API Reference](https://docs.rs/fugue-ppl)
- [Zotero](https://www.zotero.org/groups/6138134/fugue/library)
```
