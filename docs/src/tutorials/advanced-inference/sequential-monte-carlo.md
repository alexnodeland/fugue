# Sequential Monte Carlo

```admonish info title="Contents"
<!-- toc -->
```

Sequential Monte Carlo (SMC) maintains a *population* of weighted particles and moves them through a sequence of intermediate target distributions, resampling and rejuvenating along the way, until the population approximates the posterior. Unlike MCMC's single evolving chain, SMC gives you many (weakly correlated) draws per run *and* an unbiased estimate of the log marginal likelihood — useful for model comparison, which no single-chain MCMC method provides directly.

```admonish tip title="Try it live"
The **[Particles That Tell Stories](../../explorables/smc.md)** explorable animates propagate → weight → resample on a 1D state-space model. Drop the particle count to 10 and watch degeneracy happen before you read a line of the theory below.
```

## The model

```rust,ignore
{{#include ../../../../examples/smc_inference.rs:model}}
```

This is the conjugate Normal-Normal setup introduced in the [section overview](./README.md): prior $\mu \sim \mathcal{N}(0, 1)$, likelihood $y \mid \mu \sim \mathcal{N}(\mu, 0.5^2)$ observed at $y = 1.5$, with closed-form posterior $\mathcal{N}(1.2, 0.2)$.

## How `adaptive_smc` gets there

`adaptive_smc` targets the sequence of **likelihood-tempered** distributions

$$\pi_\beta(\theta) \propto p(\theta) \cdot p(y \mid \theta)^\beta, \qquad \beta: 0 \to 1$$

so $\pi_0$ is the prior (trivial to sample) and $\pi_1$ is the posterior. At each step it:

1. **Adapts** the next $\beta$ by bisection so the reweighted effective sample size (ESS) hits the configured threshold — a big jump when particles agree, a small one when they don't.
2. **Reweights** particles by the incremental likelihood factor and folds the reweighting into a running **log-evidence** estimate.
3. **Resamples** (when not the final step) to discard low-weight particles, then **rejuvenates** with a handful of $\pi_\beta$-invariant Metropolis-Hastings moves to restore diversity lost during resampling.

```rust,ignore
{{#include ../../../../examples/smc_inference.rs:run_smc}}
```

`rejuvenation_steps: 3` is what makes this genuine multi-step tempered SMC rather than a single importance-sampling reweight — with zero rejuvenation steps, particle *positions* never change between resamples, so `adaptive_smc` degenerates to one prior-to-posterior importance-sampling jump (still correct, just less effective for hard posteriors).

## Reading the result

`adaptive_smc` returns an `SMCResult`, which dereferences to `Vec<Particle>` (so slice/iterator methods work directly) and additionally carries `log_evidence`:

```rust,ignore
{{#include ../../../../examples/smc_inference.rs:analyze}}
```

```admonish note title="Weighted, not equal, particles"
Particles in the final population carry non-uniform weights (the terminal step is deliberately *not* resampled — resampling as the last operation would only discard information and inflate variance). Always weight by `p.weight` when summarizing, as above, rather than treating the population as an equally-weighted sample.
```

## Checking against ground truth

```rust,ignore
{{#include ../../../../examples/smc_inference.rs:assertions}}
```

Run it yourself:

```bash
cargo run --example smc_inference
```

```admonish tip title="Effective sample size"
`effective_sample_size(&result)` tells you how many *effectively independent* draws the weighted population represents. It ranges from 1 (all weight on one particle — trust nothing) to `N` (uniform weights — every particle counts). A low final ESS after a run with rejuvenation enabled is a sign to increase `rejuvenation_steps` or `num_particles`.
```

## Next

- **[Approximate Bayesian Computation](./approximate-bayesian-computation.md)** — same target, no likelihood required at all.
- **[Variational Inference](./variational-inference.md)** — same target, fit by optimization instead of sampling.
