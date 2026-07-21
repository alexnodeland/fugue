# A Field Guide to Distributions

A distribution is a shape and a rule for drawing from it. The shape is its
density — where values are likely to land. The rule is its sampler — how a
stream of random numbers becomes draws. Every distribution below is one note in
fugue's vocabulary; pick one and hear it.

The blue curve is the exact law. The green bars are samples piling up. Watch
them converge — that is the law of large numbers, live.

<div class="fugue-explorable" data-viz="distributions" data-seed="11"></div>

## Things to try

1. Start on **Normal** and drag σ up to 3 — the bell flattens and spreads, but
   the green samples still trace it.
2. Switch to **Beta** and set α and β both below 1 — the density turns into a
   U, piling mass at the two edges.
3. Open **Cauchy**. Its `mean` and `variance` readouts both read `—`: the tails
   are so heavy that neither integral converges. The samples wander wildly.
4. Pick **Bernoulli** and read the `sample →` badge: it says `bool`. Fugue gives
   you a real boolean, not a float you have to compare against `1.0`.
5. Drag **Uniform**'s `high` below its `low`. The canvas turns red — this is
   exactly the `Err` that `Uniform::new` returns for an invalid interval.
6. On any distribution, drop the **seed** back to a value you already used. The
   green histogram redraws the identical run: a seeded stream is a replayable
   trace.

## What you just saw

For a continuous distribution the blue curve is the **probability density**
<span class="fv-c-prior">f(x)</span>. It is not a probability — it is a
density, so it can exceed 1. What integrates to 1 is the area:

$$\int_{\mathcal{S}} \textcolor{#58A6FF}{f(x)}\, dx = 1$$

For a discrete distribution the blue stems are the **probability mass**
<span class="fv-c-prior">P(x)</span>, one bar per outcome, and the bars sum to 1:

$$\sum_{x \in \mathcal{S}} \textcolor{#58A6FF}{P(x)} = 1$$

The <span class="fv-c-post">green</span> bars are an empirical estimate of that
same law built from samples. As the sample count grows they converge on the
blue — the **law of large numbers**. The <span class="fv-c-hot">coral</span>
line is a single query: it reports $\log \textcolor{#58A6FF}{f(x)}$ at the x you
drag to. Inference works in log-space because a product of thousands of these
densities underflows to zero in ordinary floating point; a sum of their logs
does not.

The markers name three summaries of the shape: the
<span class="fv-c-flow">mean</span> (the balance point), the median (half the
mass on each side), and the mode (the peak). For a skewed law like LogNormal
they sit in different places; for a symmetric one they coincide.

### Fugue's type story

Most PPLs make every draw a `f64`, so a coin flip comes back as `1.0` and you
compare floats; a count comes back as `4.0` and you round it. Fugue draws return
their **natural type**, checked at compile time.

<div class="fugue-explorable fv-inline" data-viz="type-flow" data-caption="Each sampler drops its draw into a slot of its natural type — a bool, a u64 count, a usize index — never a float you have to decode."></div>

```rust
use fugue::*;
use rand::thread_rng;

fn main() {
    let mut rng = thread_rng();

    // Bernoulli -> bool. The coral query line is `log_prob` on one outcome.
    let coin = Bernoulli::new(0.5).unwrap();
    let heads: bool = coin.sample(&mut rng);
    let lp_true: f64 = coin.log_prob(&true); // ln(0.5)
    if heads { /* no `== 1.0`; it's already a bool */ }

    // Poisson -> u64. Counts are counts, not rounded floats.
    let arrivals: u64 = Poisson::new(4.0).unwrap().sample(&mut rng);
    let total_wait = arrivals * 10; // integer arithmetic, no cast

    // Categorical -> usize. Safe to index an array with, by construction.
    let pick: usize = Categorical::new(vec![0.5, 0.3, 0.2]).unwrap().sample(&mut rng);
    let labels = ["a", "b", "c"];
    let chosen = labels[pick];

    // Continuous laws return f64, as expected.
    let x: f64 = Normal::new(0.0, 1.0).unwrap().sample(&mut rng);
    let density: f64 = Normal::new(0.0, 1.0).unwrap().log_prob(&x);

    println!("{heads} {arrivals} {total_wait} {chosen} {x:.3} {density:.3} {lp_true:.3}");
}
```

Every constructor is fallible: `Normal::new`, `Beta::new`, `Uniform::new` and the
rest return `Result`, so an invalid parameter (a negative σ, a `low ≥ high`) is
an `Err` you handle, never a silent `NaN`. That is the red screen in try #5.

Inside a model, a draw is a `sample` at a named address; its distribution rides
along and fugue scores it for you:

```rust
use fugue::*;

// Estimate a coin's bias, then predict the next flip's count over 10 tosses.
let model = prob!(
    let p <- sample(addr!("bias"), Beta::new(2.0, 2.0).unwrap());
    let hits <- sample(addr!("hits"), Binomial::new(10, p).unwrap());
    pure(hits)
);
```

## The full field guide

Every distribution in fugue. The `sample →` column is the natural return type.

### Continuous

| Distribution | Support | Parameters | `sample →` | Reach for it when |
|---|---|---|---|---|
| `Normal` | (−∞, ∞) | `mu`, `sigma` > 0 | `f64` | you want a symmetric bell — noise, error, a CLT limit. |
| `Uniform` | [low, high) | `low` < `high` | `f64` | you want a flat prior on a bounded interval. |
| `LogNormal` | (0, ∞) | `mu`, `sigma` > 0 | `f64` | a positive, right-skewed quantity whose log is Normal. |
| `Exponential` | [0, ∞) | `rate` > 0 | `f64` | the wait until the next memoryless event; mean = 1/rate. |
| `Beta` | [0, 1] | `alpha`, `beta` > 0 | `f64` | a probability about a probability — the coin-bias prior. |
| `Gamma` | (0, ∞) | `shape`, `rate` > 0 | `f64` | positive quantities; **rate**-parameterized, mean = shape/rate. |
| `StudentT` | (−∞, ∞) | `df`, `loc`, `scale` > 0 | `f64` | a heavier-tailed Normal that tolerates outliers. |
| `Cauchy` | (−∞, ∞) | `loc`, `scale` > 0 | `f64` | pathological tails — no mean, no variance. `StudentT(df=1)`. |
| `Laplace` | (−∞, ∞) | `loc`, `scale` > 0 | `f64` | a sharp peak with exponential tails; the L1 / lasso prior. |
| `Weibull` | [0, ∞) | `shape`, `scale` > 0 | `f64` | time-to-failure and survival modeling. |
| `ChiSquared` | (0, ∞) | `k` > 0 | `f64` | sums of squared Normals; = `Gamma(k/2, ½)`. |
| `InverseGamma` | (0, ∞) | `shape`, `rate` > 0 | `f64` | the conjugate prior for a Normal's variance. |

### Discrete

| Distribution | Support | Parameters | `sample →` | Reach for it when |
|---|---|---|---|---|
| `Bernoulli` | {0, 1} | `p` ∈ [0, 1] | `bool` | one yes/no trial — and you want a real `bool`. |
| `Categorical` | {0 … K−1} | `probs` sum to 1 | `usize` | picking one of K labels; index arrays safely. |
| `Binomial` | {0 … n} | `n`, `p` ∈ [0, 1] | `u64` | successes in n independent trials. |
| `Poisson` | {0, 1, 2, …} | `lambda` > 0 | `u64` | rare-event counts; mean = variance = λ. |
| `DiscreteUniform` | {low … high} | `low` ≤ `high` | `i64` | a fair die over an integer range. |

```admonish note title="Gamma is rate-parameterized"
Fugue's `Gamma::new(shape, rate)` uses **rate** (λ), not scale, so the mean is
`shape / rate`. `Exponential`, `InverseGamma`, and `ChiSquared` follow the same
rate convention. If a value looks inverted, check whether you meant scale = 1/rate.
```

## Go deeper

- Tutorial: [Working with Distributions](../how-to/working-with-distributions.md) —
  the same catalog in prose, with modeling patterns.
- API: [`fugue::core::distribution`](https://docs.rs/fugue-ppl/latest/fugue/core/distribution/index.html) —
  every constructor, its constraints, and its `log_prob`.
- Next explorable: [Anatomy of a Probabilistic Program](anatomy.md) — put a
  `Beta` prior and `Bernoulli` data together and watch Bayes multiply them.

---

Next: [Explorables](./README.md)
