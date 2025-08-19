## monadic_ppl

A tiny, elegant, monadic probabilistic programming library for Rust. Write
probabilistic programs by composing `Model` values in direct style; run them
with pluggable interpreters and inference routines.

### Quick start

Add the crate as a dependency, then write a small model:

```rust
use monadic_ppl::*;

fn gaussian_mean(obs:f64)->Model<f64>{
  sample(addr!("mu"), Normal{mu:0.0, sigma:5.0}).bind(move|mu|{
    observe(addr!("y"), Normal{mu, sigma:1.0}, obs).bind(move |_| pure(mu))
  })
}
```

Run the prior interpreter:

```bash
cargo run --example gaussian_mean -- --obs 2.7 --seed 123
```

### Modules

- core: addresses, distributions, and the `Model` monad
- runtime: the `Handler` trait, built-in interpreters, and `Trace`
- inference: placeholder MH/SMC/VI scaffolding
- macros: reserved for future syntactic sugar

### Examples

- gaussian_mean.rs: basic normal mean model with CLI
- gaussian_mixture.rs: simple mixture of two Gaussians
- exponential_hazard.rs: lognormal prior on exponential rate

### Tests

See `tests/layout_tests.rs` for trace behavior (prior and replay).

### Roadmap

- Enrich distribution set; add discrete distributions
- Site-wise MH proposals and adaptive kernels
- Proper SMC with resampling and model staging
- Variational families and reparameterized gradients
