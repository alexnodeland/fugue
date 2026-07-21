<div class="fugue-hero">

<div class="fugue-mark"><span>ƒ</span></div>

# Fugue

<p class="fugue-hero-tag">A type-safe, monadic probabilistic programming library for Rust — compose models in direct style, listen backwards with state-of-the-art inference.</p>

<div class="fugue-badge-row">
<a href="https://crates.io/crates/fugue-ppl"><img src="https://img.shields.io/crates/v/fugue-ppl.svg" alt="Crates.io"></a>
<a href="https://docs.rs/fugue-ppl"><img src="https://docs.rs/fugue-ppl/badge.svg" alt="Dev Docs"></a>
<a href="https://github.com/alexnodeland/fugue/actions/workflows/ci.yml"><img src="https://github.com/alexnodeland/fugue/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
<a href="https://codecov.io/gh/alexnodeland/fugue"><img src="https://codecov.io/gh/alexnodeland/fugue/branch/main/graph/badge.svg?token=BDJ5OB6GOB" alt="codecov"></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
<a href="https://discord.gg/QAcF7Nwr"><img src="https://img.shields.io/discord/1412802057437712426?logo=discord&label=discord" alt="Discord"></a>
</div>

</div>

<div class="fugue-explorable fv-inline" data-viz="dist-strip" data-dist="normal" data-params="0,1" data-caption="This page is alive — most figures in these docs run live. Touch them."></div>

<div class="fugue-cards">
<a class="fugue-card" href="./getting-started/index.html"><span class="fugue-card-title">Getting Started</span><span class="fugue-card-desc">Install Fugue and run your first probabilistic model in minutes.</span></a>
<a class="fugue-card" href="./explorables/index.html"><span class="fugue-card-title">Explorables</span><span class="fugue-card-desc">Interactive essays — drag a prior and watch the posterior re-form, roll HMC across a landscape.</span></a>
<a class="fugue-card" href="./playground.html"><span class="fugue-card-title">Playground</span><span class="fugue-card-desc">Write <code>prob!</code> models in the browser and run real inference, compiled to WASM.</span></a>
<a class="fugue-card" href="./how-to/index.html"><span class="fugue-card-title">How-To Guides</span><span class="fugue-card-desc">Task-focused recipes: custom distributions, diagnostics, working with traces.</span></a>
<a class="fugue-card" href="./tutorials/index.html"><span class="fugue-card-title">Tutorials</span><span class="fugue-card-desc">From the coin-flip Bayes loop to advanced inference, in depth.</span></a>
<a class="fugue-card" href="./api-reference.html"><span class="fugue-card-title">API Reference</span><span class="fugue-card-desc">The complete <code>fugue-ppl</code> API, documented on docs.rs.</span></a>
</div>

<a class="fugue-eco" href="https://evo.fugue.run"><span class="fugue-card-title">Fugue Evo <span class="fugue-brand-arrow">↗</span></span><span class="fugue-card-desc">Evolution as Bayesian inference — CMA-ES, NSGA-II, island models, and estimation-of-distribution algorithms on the same foundations, with its own live playground.</span></a>

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

## More Resources

- [GitHub Repository](https://github.com/alexnodeland/fugue)
- [API Reference on docs.rs](https://docs.rs/fugue-ppl)
- [Zotero Library](https://www.zotero.org/groups/6138134/fugue/library)
- [Ask DeepWiki](https://deepwiki.com/alexnodeland/fugue)
