# Playground

Everything on this page runs the **real fugue crate**, compiled to
WebAssembly. The editor below speaks a subset of the `prob!` macro language;
when you press **Run**, your model text is parsed *in Rust*, folded into
actual `Model` combinators, and handed to the same inference kernels this
book documents — adaptive Metropolis–Hastings one transition at a time,
Hamiltonian Monte Carlo with dual-averaging warmup, adaptive tempered SMC
with a log-evidence estimate. The draws you watch stream in are fugue's
draws, not a JavaScript imitation of them.

<div class="fugue-explorable" data-viz="playground" data-seed="11"></div>

## What the editor understands

The playground accepts the statement forms of `prob!`, with data provided as
JSON (an object of named arrays and scalars, or a bare array bound to
`data`):

```text
let p <- sample(addr!("p"), Beta(2.0, 2.0));   // sample a latent
let m = 2.0 * p - 1.0;                          // deterministic let
observe(addr!("y"), Normal(m, 0.8), 1.4);       // condition on data
factor(-0.5 * m * m);                           // add a log-weight
for i in 0..data.len() { ... }                  // plates
if m > 0.0 { ... } else { ... }                 // branches
m = m * 2.0;                                    // reassign a variable
break;                                          // leave a loop early
pure(p)                                         // the model's return value
```

Rust spellings from the docs paste in unchanged — `Normal::new(0.0, 1.0).unwrap()`
parses the same as `Normal(0.0, 1.0)`. Available distributions: `Normal`,
`Uniform`, `LogNormal`, `Exponential`, `Beta`, `Gamma`, `InverseGamma`,
`StudentT`, `Cauchy`, `Laplace`, `Weibull`, `ChiSquared`, `Bernoulli`,
`Binomial`, `Poisson`, `Categorical`, `DiscreteUniform` — the same
constructors, parameterizations, and support checks as the crate, because
they *are* the crate. Expressions know `+ - * /`, the comparisons
`== != < <= > >=`, `&& || !`, `exp`, `ln`, `sqrt`, `abs`, `pow`, `min`,
`max`, `floor`, `sin`, `cos`, `tanh`, array literals `[a, b]`, array
indexing `y[i]`, and `.len()`. Variables are scoped as in Rust, and sites
keep their natural types: a `Bernoulli` draw is a `bool`, a `Poisson` draw
a count.

Two honest limitations: runtime errors — a parameter region that would make
a distribution invalid (say, a negative scale reached through your
arithmetic), an index past the end of an array — score as impossible
(`-inf`) and show a warning rather than stopping the run, exactly how the
samplers treat leaving a support; and the interpreter covers the statement
subset above, not arbitrary Rust — for that, there is `cargo add fugue-ppl`.

## Things to try

1. **Watch R̂ earn its keep.** Run the coin-flip preset with 3 chains: the
   chains start from independent prior draws, disagree, then R̂ falls
   toward 1.00 as their histograms merge — the same split-R̂ computation
   `fugue::inference::diagnostics` runs in production.
2. **Break a model, kindly.** Change a flip observation to `2.0` (a coin
   that landed on its edge?). `Bernoulli` observes a boolean, so anything
   nonzero coerces to `true` — then make the prior `Beta(0.5, 0.5)` and
   watch the posterior bend toward the edges.
3. **Compare kernels on the same posterior.** Run the eight-schools preset
   under MH, then under HMC. Single-site MH updates one coordinate at a
   time and mixes slowly through the funnel; HMC moves all ten coordinates
   jointly. The ESS readout is the receipt.
4. **Ask for evidence.** Switch to SMC on any preset: you get weighted
   posterior particles *and* `log Z`, the marginal likelihood MH and HMC
   never give you.
5. **Make it yours.** Replace the data JSON with your own numbers. The
   model recompiles as you type — the ✓/✗ line is fugue's parser talking.

## How this works

`crates/fugue-wasm` (in the fugue repository) exposes wasm-bindgen entry
points over the crate: compile a model source + data payload, then drive
`step(n)` per animation frame and read draws and diagnostics back as typed
arrays. The model language itself is `fugue::program` (the crate's
`program` feature), so the same programs — as text or as JSON — build the
same models in any Rust host. Sampler state lives in Rust — traces, adaptation state, tuned step
sizes — and every run is seeded, so a seed is a replayable recording here
too. The interactive figures in the explorables chapters use the same
package for their samplers; only their canvas rendering is JavaScript.

## Go deeper

- **Explorables:** [Random Walks in Posterior Space](./explorables/metropolis.md),
  [Rolling, Not Guessing: HMC](./explorables/hmc.md),
  [Particles That Tell Stories](./explorables/smc.md) — the guided tours of
  the kernels this playground exposes.
- **The real thing:** [Installation](./getting-started/installation.md) —
  `cargo add fugue-ppl` and the full `prob!` language, typed traces,
  custom handlers.
- **API:** [docs.rs/fugue-ppl](https://docs.rs/fugue-ppl).
