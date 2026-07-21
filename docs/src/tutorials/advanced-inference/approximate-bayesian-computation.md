# Approximate Bayesian Computation

```admonish info title="Contents"
<!-- toc -->
```

Approximate Bayesian Computation (ABC) is what you reach for when the likelihood $p(y \mid \theta)$ is intractable (or you simply don't want to write it down), but you *can* simulate synthetic data from the model. ABC replaces likelihood evaluation with simulate-and-compare: draw $\theta$ from the prior, simulate $y_{\text{sim}}$, and accept $\theta$ if $y_{\text{sim}}$ is close enough to the real observation $y_{\text{obs}}$.

<div class="fugue-explorable fv-inline" data-viz="abc-eps" data-caption="Accepted draws (green) fall inside the shrinking ε band; rejects (coral) fall outside — the accepted cloud is the posterior."></div>

```admonish warning title="This example's likelihood isn't actually intractable"
To keep this tutorial directly comparable to the [SMC](./sequential-monte-carlo.md) and [VI](./variational-inference.md) pages, it reuses the conjugate Normal-Normal model — whose likelihood is very much tractable. ABC's real value is for simulators where no such closed form (or even numerical likelihood) exists at all; using a tractable model here is purely so we have a known target to check the approximation against.
```

## The model: simulate, don't observe

```rust,ignore
{{#include ../../../../examples/abc_inference.rs:model}}
```

Notice the second line uses `sample`, not `observe`: ABC never scores a likelihood, so the model's job is to **forward-simulate** a synthetic observation, not condition on a real one. The address `y_sim` is read back out of the trace after each prior draw.

## Why this converges to the same posterior

Accepting a draw when $|y_{\text{sim}} - y_{\text{obs}}| \le \varepsilon$ and shrinking $\varepsilon \to 0$ converges to conditioning on $y_{\text{sim}} = y_{\text{obs}}$ exactly — the same event a direct Bayesian update on $y \sim \mathcal{N}(\mu, 0.5^2) = 1.5$ conditions on. So in the small-tolerance limit, ABC targets the *same* $\mathcal{N}(1.2, 0.2)$ posterior as the SMC and VI tutorials. At any finite tolerance, though, ABC is only an *approximation* of that target — the price paid for not needing a likelihood at all.

## Running ABC-SMC

A single-shot rejection ABC (`abc_rejection`) at a small tolerance can require an enormous number of prior draws to find even one acceptance. `abc_smc_weighted` fixes this the same way `adaptive_smc` does: a schedule of shrinking tolerances, with each stage's population built by perturbing and re-weighting the previous one (Beaumont et al. 2009 / Toni et al. 2009) rather than restarting from the prior every time.

```rust,ignore
{{#include ../../../../examples/abc_inference.rs:run_abc}}
```

Each stage is bounded by an attempt budget (`200_000` here): if a stage can't fill its particle quota within that budget, `abc_smc_weighted` returns a typed `ABCError` (`EmptyInitialPopulation` or `StageExhausted`) instead of looping forever or panicking on an empty population.

## Reading the result

`abc_smc_weighted` returns an `ABCSMCResult` with a **weighted** particle population (unlike `abc_smc`, its equally-weighted convenience wrapper) plus a `weighted_mean` helper:

```rust,ignore
{{#include ../../../../examples/abc_inference.rs:analyze}}
```

## Checking against ground truth

```rust,ignore
{{#include ../../../../examples/abc_inference.rs:assertions}}
```

Note the wider tolerance band compared to the SMC tutorial's assertions — that gap *is* the ABC trade-off made visible: approximate inference, in exchange for applicability to simulators with no tractable likelihood whatsoever.

Run it yourself:

```bash
cargo run --example abc_inference
```

## Next

- **[Variational Inference](./variational-inference.md)** — same target again, this time fit by optimization.
- Back to **[Sequential Monte Carlo](./sequential-monte-carlo.md)** if you haven't read it yet.
