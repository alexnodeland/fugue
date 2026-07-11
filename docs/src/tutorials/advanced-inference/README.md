# Advanced Inference

```admonish info title="Contents"
This section demonstrates Fugue's inference methods beyond adaptive MCMC:
- **[Sequential Monte Carlo](./sequential-monte-carlo.md)** - Particle-based inference with an evidence estimate
- **[Approximate Bayesian Computation](./approximate-bayesian-computation.md)** - Likelihood-free inference from forward simulation
- **[Variational Inference](./variational-inference.md)** - Fast, optimization-based posterior approximation
```

Fugue's headline feature list advertises **"Multiple Inference Methods: MCMC, SMC, Variational Inference, ABC."** The [Bayesian Coin Flip](../foundation/bayesian-coin-flip.md) tutorial and most of the [Statistical Modeling](../statistical-modeling/README.md) tutorials use `adaptive_mcmc_chain`. This section covers the other three, each on the same running example so you can compare their posteriors directly.

```admonish note title="A gradient-based fifth option: HMC"
Fugue also ships Hamiltonian Monte Carlo (`hmc_chain`), which mixes far better than single-site MH on correlated continuous posteriors by moving all continuous sites jointly using (finite-difference) gradient information. It isn't covered by its own tutorial page yet, but is fully documented — with a runnable example — in the [`hmc` module rustdoc](https://docs.rs/fugue-ppl/latest/fugue/inference/hmc/) and listed alongside the other methods in the [README](https://github.com/alexnodeland/fugue#-example).
```

## The running example

All three tutorials in this section perform inference on the same conjugate Normal-Normal model, so their results are directly comparable:

$$\mu \sim \mathcal{N}(0, 1), \qquad y \mid \mu \sim \mathcal{N}(\mu, 0.5^2), \qquad y_{\text{obs}} = 1.5$$

Because both the prior and likelihood are Gaussian, the posterior has a closed form (precision-weighted combination of prior and likelihood):

$$\mu \mid y_{\text{obs}} \sim \mathcal{N}(1.2,\ 0.2), \qquad \sigma_{\text{post}} = \sqrt{0.2} \approx 0.4472$$

Having ground truth in hand means each tutorial can check its inference method against an exact number instead of asking you to eyeball a histogram — the same property that makes each method's corresponding example (`examples/smc_inference.rs`, `examples/abc_inference.rs`, `examples/vi_inference.rs`) a genuine regression test rather than a "runs without panicking" demo.

## When to reach for which method

| Method | Function | Good fit when... |
|---|---|---|
| MCMC (adaptive MH) | `adaptive_mcmc_chain` | General-purpose default; works on discrete and continuous sites |
| HMC | `hmc_chain` | Continuous, correlated posteriors where MH mixes slowly |
| SMC | `adaptive_smc` | You also want a log-evidence estimate, or the posterior is multimodal/hard to reach by local moves |
| VI | `optimize_meanfield_vi_with_config` | You need speed over exactness, or a differentiable, tunable approximation |
| ABC | `abc_smc_weighted` | The likelihood is intractable but you can *simulate* from the model |

```admonish tip title="Try it yourself"
cargo run --example smc_inference
cargo run --example abc_inference
cargo run --example vi_inference
```
