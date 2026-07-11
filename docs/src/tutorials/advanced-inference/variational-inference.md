# Variational Inference

```admonish info title="Contents"
<!-- toc -->
```

Sampling-based methods (MCMC, HMC, SMC) approximate the posterior with a set of draws. Variational Inference (VI) instead picks a tractable family of distributions $q_\phi$ and **optimizes** $\phi$ so that $q_\phi$ is as close as possible to the true posterior, by maximizing the Evidence Lower BOund (ELBO):

$$\text{ELBO}(\phi) = \mathbb{E}_{z \sim q_\phi}\left[\log p(x, z) - \log q_\phi(z)\right] \le \log p(x)$$

This trades exactness for speed: a converged VI fit is one optimization run, not thousands of MCMC iterations — at the cost of being only as good as the chosen family $q_\phi$ lets it be.

## The model

```rust,ignore
{{#include ../../../../examples/vi_inference.rs:model}}
```

The same conjugate Normal-Normal model as the [SMC](./sequential-monte-carlo.md) and [ABC](./approximate-bayesian-computation.md) tutorials, with `observe` used normally this time — VI, unlike ABC, does need to score the model's log-density.

```admonish success title="This example is a special case: mean-field VI is exact here"
Fugue's `MeanFieldGuide` approximates each latent independently with a matched-support family: `Normal` for real-valued latents, `LogNormal` for positive ones, `Beta` for `[0,1]`-valued ones. Because this model's true posterior is *itself* Gaussian, the `Normal` factor for a `Support::Real` latent can represent it *exactly* — so a converged fit here isn't "a good approximation," it's the right answer, up to optimization noise. That makes this tutorial a clean check that the optimizer works, not just that the guide family is reasonable.
```

## Building a guide and running the optimizer

```rust,ignore
{{#include ../../../../examples/vi_inference.rs:run_vi}}
```

A few things worth calling out:

- **`guide.add_latent(addr, Support::Real, init)`** picks the variational family from the latent's declared support instead of defaulting every latent to an unconstrained Normal — a Normal guide on a strictly-positive or unit-interval latent would propose out-of-support values whose model log-density is $-\infty$, collapsing the ELBO.
- **Both location and scale are optimized.** Each variational factor has two free parameters (e.g. `mu` and `log_sigma` for a Normal factor, the latter unconstrained via a log transform for positivity); `optimize_meanfield_vi_with_config` updates both by gradient ascent, not just the location.
- **Gradients are common-random-numbers finite differences.** Fugue models are plain Rust closures with no autodiff, so the ELBO gradient is estimated by central finite differences with the `+ε`/`-ε` evaluations sharing an RNG seed — this cancels Monte Carlo noise in the difference, leaving only the (much smaller) finite-difference bias.
- **A Robbins-Monro step schedule and an ELBO-plateau convergence test** mean the optimizer can stop before `n_iterations` — check `result.converged` and `result.iterations`.

## Reading the result

```rust,ignore
{{#include ../../../../examples/vi_inference.rs:analyze}}
```

## Checking against ground truth

```rust,ignore
{{#include ../../../../examples/vi_inference.rs:assertions}}
```

Run it yourself:

```bash
cargo run --example vi_inference
```

```admonish tip title="elbo_with_guide for evaluation only"
If you already have a guide (fitted or hand-specified) and just want to evaluate the ELBO without optimizing, use `elbo_with_guide` directly — it's what `optimize_meanfield_vi_with_config` calls internally to monitor progress each iteration.
```

## Next

- Back to **[Sequential Monte Carlo](./sequential-monte-carlo.md)** or **[Approximate Bayesian Computation](./approximate-bayesian-computation.md)** to compare methods on the same target.
- **[Advanced Inference overview](./README.md)** for a summary table of when to reach for each method.
