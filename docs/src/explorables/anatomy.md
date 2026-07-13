# Anatomy of a Probabilistic Program

A probabilistic program has three moving parts: a **prior** (what you believe
before data), a **likelihood** (how data is explained), and a **posterior**
(what you believe after). Here they are, side by side, for the oldest question
in statistics: is this coin fair? Flip a chip, drag the prior, and watch the
green curve reshape itself in real time — each change leaving a fading ghost of
the belief it just replaced.

<div class="fugue-explorable" data-viz="anatomy" data-seed="11"></div>

You start with a <span class="fv-c-prior">prior</span> belief about the coin's
bias `p`: Beta(α = <span class="fv-scrub" id="fv-anatomy-alpha" data-min="0.5" data-max="20" data-step="0.5" data-value="2">2</span>,
β = <span class="fv-scrub" id="fv-anatomy-beta" data-min="0.5" data-max="20" data-step="0.5" data-value="2">2</span>).
The blue curve is that belief. Each coin chip is one
<span class="fv-c-data">observation</span>; click any chip to flip it between
heads and tails. The <span class="fv-c-post">posterior</span> — the green curve —
is the prior *times* the likelihood, renormalized. Press **Replay** to watch the
flips arrive one at a time and the green curve walk from the prior to the
posterior — Bayesian updating, animated — or press **Step** to take that walk one
flip at a time yourself. Press **Deal** to draw a fresh dozen flips from seed
<span class="fv-scrub" id="fv-anatomy-seed" data-min="1" data-max="99" data-step="1" data-value="11">11</span>;
the same seed always deals the same coins, because in fugue a seeded run is a
replayable recording.

## Things to try

1. Add tails until the <span class="fv-c-post">green</span> curve's peak crosses
   left of `p = 0.5` — that is the moment the data outvotes a "fair coin" prior.
2. Drag **β** up to 20 with only a few flips: a stubborn prior barely moves. Now
   add twenty flips — data eventually wins, no matter the prior.
3. Set **α = 0.5, β = 0.5** (the Jeffreys prior). The blue curve bends up at both
   ends: it says "this coin is probably rigged one way or the other."
4. Turn **show likelihood** on and watch the yellow curve. The green posterior
   always sits *between* blue and yellow — pulled toward whichever is sharper.
5. Press **Replay**: the posterior starts *as* the blue prior, then each flip
   nudges it — heads pull right, tails pull left. The fading green ghosts are the
   beliefs it held along the way. This is Bayesian updating, one datum at a time.
6. Scrub the **seed** while **Deal**ing: every value gives a different but fully
   reproducible dataset. That reproducibility is the whole point of a trace.

## What you just saw

The green curve is not drawn by a formula you have to trust — it is the
literal product of the other two, normalized so its area is one:

$$\textcolor{#56D364}{p(p \mid \mathcal{D})} \;\propto\; \textcolor{#58A6FF}{p(p)}\;\times\;\textcolor{#F2CC60}{p(\mathcal{D} \mid p)}$$

For a coin this product has a closed form. A Beta prior multiplied by
`h` heads and `t` tails of Bernoulli likelihood is again a Beta — the two are
**conjugate** — so the posterior is exact:

$$\underbrace{\textcolor{#58A6FF}{\mathrm{Beta}(\alpha,\beta)}}_{\text{prior}} \times \underbrace{\textcolor{#F2CC60}{p^{\,h}(1-p)^{\,t}}}_{\text{likelihood}} \;\propto\; \underbrace{\textcolor{#56D364}{\mathrm{Beta}(\alpha+h,\ \beta+t)}}_{\text{posterior}}$$

Every heads slides one unit of belief into α; every tails, into β. That is why
**Replay** works: feeding the flips one at a time, using each posterior as the
next prior, lands on exactly the same green curve as folding them in all at
once. Bayesian updating is associative. The posterior mean is the updated ratio,
and the readouts track it live:

$$\mathbb{E}[\textcolor{#56D364}{p \mid \mathcal{D}}] = \frac{\alpha+h}{\alpha+\beta+h+t}$$

The coral marker is the **MAP** — the posterior's most probable bias, its mode.
The shaded band is the **90% credible interval**: the model's honest "I'm 90%
sure the bias is in here." Conjugacy is a lucky gift of the coin; most models
have no such shortcut. That is why fugue exists.

## The fugue code

The widget uses conjugacy because it can. fugue does not need to — it runs
Metropolis–Hastings on the *same* model and lands on the *same* number.

```rust
use fugue::*;
use fugue::inference::mh::adaptive_mcmc_chain;
use rand::{SeedableRng, rngs::StdRng};

// The score: a prior over the bias, then one observe per flip.
fn coin(data: Vec<bool>) -> Model<f64> {
    prob!(
        // PRIOR — the blue curve: belief about the bias p before any flip.
        let p <- sample(addr!("p"), Beta::new(2.0, 2.0).unwrap());

        // LIKELIHOOD — each yellow chip is one `observe`: a flip explained by p.
        let _obs <- plate!(i in 0..data.len() => {
            observe(addr!("flip", i), Bernoulli::new(p).unwrap(), data[i])
        });

        pure(p) // return the inferred bias
    )
}

fn main() {
    // 7 heads, 3 tails — the widget's starting data.
    let data = vec![true, false, true, true, false, true, true, false, true, true];
    let mut rng = StdRng::seed_from_u64(11);

    // Metropolis–Hastings — no conjugacy assumed, adaptive step size.
    let samples = adaptive_mcmc_chain(&mut rng, || coin(data.clone()), 4000, 1000);
    let ps: Vec<f64> = samples
        .iter()
        .filter_map(|(_, t)| t.get_f64(&addr!("p")))
        .collect();
    let mean = ps.iter().sum::<f64>() / ps.len() as f64;

    // Conjugacy says the posterior is Beta(2+7, 2+3) = Beta(9, 5),
    // whose mean is 9/14 ≈ 0.643. MH agrees without ever knowing that.
    println!("MH posterior mean ≈ {mean:.3}  (analytic 9/14 = {:.3})", 9.0 / 14.0);
}
```

`sample` records a choice at an address; `observe` scores the data against the
current bias and folds a log-weight into the trace; `pure` returns a value. The
`prob!` and `plate!` macros are sugar over the `Model` monad — the score is
written once, then *performed* by a handler. That separation is the next
explorable.

## Go deeper

- **Tutorial:** [Bayesian Coin Flip](../tutorials/foundation/bayesian-coin-flip.md)
  walks the same model without the pictures.
- **Next explorable:** [The Model Is a Score](./monad.md) — step through the
  interpreter that actually runs this program.
- **API:** [`Beta`](https://docs.rs/fugue-ppl/latest/fugue/), [`adaptive_mcmc_chain`](https://docs.rs/fugue-ppl/latest/fugue/inference/mh/fn.adaptive_mcmc_chain.html).

---

Next: [The Model Is a Score](./monad.md)
