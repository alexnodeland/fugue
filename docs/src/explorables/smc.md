# Particles That Tell Stories

A hidden thing moves through time. You never see it directly — only noisy
glimpses, one per tick. How do you track it? You send a swarm of guesses forward,
reward the ones that fit each new glimpse, and let the rest die. That swarm is a
**particle filter**, and every guess is a little story about where the truth went.

The score is a state-space model; the particles are performers improvising the
hidden path; each observation is a critic that reweights them. Watch time flow
left to right.

<div class="fugue-explorable" data-viz="smc" data-seed="11"></div>

The <span class="fv-c-data">yellow dots</span> are what you observe. The dashed
ink line is the truth (you never get to see it during real inference). The
<span class="fv-c-prior">blue circles</span> are particles — each radius is that
particle's weight. The particles are the machinery; the
<span class="fv-c-post">green</span> is the answer. At the current time step, the
<span class="fv-c-post">green violin</span> is the **filtering distribution** —
the weighted swarm's full belief about \\( x_t \\) right now, drawn as a
weighted-particle density. The <span class="fv-c-post">green line</span> traces the
filtered mean over time, and the <span class="fv-c-post">green band</span> around
it is that belief's ±1σ spread — how sure the filter is, tick by tick.

## Things to try

1. Press **Step** a few times. Each tick does three things: **propagate** the
   particles forward, **weight** them against the new yellow dot, then
   **resample** if they have grown too uneven.
2. Drag **particles** down to 10 and press **Play**. The cloud collapses onto one
   or two lineages within a few steps — *degeneracy*, the failure mode SMC exists
   to fight.
3. Turn **adaptive resample** off and play. With no resampling, a single particle
   swallows almost all the weight; ESS / N crashes toward 1/N and the green
   estimate goes deaf.
4. Push **obs noise** up. Flatter likelihoods mean gentler reweighting, so ESS
   stays high and lineages survive longer — and watch the <span class="fv-c-post">green
   band</span> fatten: looser data means a less certain filter, and the filtering
   violin spreads to match.
5. Change the **seed** scrub, then reset. A seeded run is a replayable trace: same
   seed, same particles, same story, every time.

## What you just saw

The hidden path is a **random walk**; each observation is a **noisy read** of it:

$$
\textcolor{#58A6FF}{x_t \sim \mathcal{N}(x_{t-1},\, \sigma)}
\qquad
\textcolor{#F2CC60}{y_t \sim \mathcal{N}(x_t,\, \tau)}
$$

The <span class="fv-c-prior">blue</span> line is the **transition** — how a
particle proposes its next position. The <span class="fv-c-data">yellow</span>
line is the **likelihood** — how well that position explains the observation.

**Propagate.** Every particle draws its next state from the transition. On the
canvas this is the drift step: the whole cloud slides one column right.

**Weight.** Each particle's weight is multiplied by how likely the new
observation was under it:

$$
\tilde{w}_t^{(i)} \;\propto\; W_{t-1}^{(i)}\;\textcolor{#F2CC60}{p\!\left(y_t \mid x_t^{(i)}\right)}
$$

Particles near the yellow dot fatten; particles far from it shrink. Normalize and
you have the new weights.

**The filtering distribution.** The weighted swarm *is* an estimate of one
distribution: \\( p(x_t \mid y_{1:t}) \\), your belief about the hidden state given
every observation so far. That is what the <span class="fv-c-post">green violin</span>
draws — a weighted-particle kernel density of \\( \{x_t^{(i)}, W_t^{(i)}\} \\) at the
current column. Its <span class="fv-c-post">mean</span> and ±1σ
<span class="fv-c-post">band</span> are the one-number summaries you actually report:

$$
\hat{\mu}_t = \sum_i W_t^{(i)}\, x_t^{(i)}
\qquad
\hat{\sigma}_t^2 = \sum_i W_t^{(i)}\left(x_t^{(i)} - \hat{\mu}_t\right)^2
$$

**Measure health.** The **effective sample size** counts how many particles are
really doing work:

$$
\mathrm{ESS} \;=\; \frac{1}{\sum_i \left(W_t^{(i)}\right)^2}
$$

All weight on one particle gives ESS = 1; perfectly even weights give ESS = N.
The readout shows ESS / N, <span class="fv-c-post">green</span> above 0.5,
<span class="fv-c-hot">coral</span> below.

**Resample.** When ESS / N falls under the threshold, draw a fresh population by
sampling parents in proportion to weight, then reset weights to uniform. Fat
particles spawn duplicates (the violet fan); starved particles go
<span class="fv-c-hot">extinct</span>. This is the moment of the page: the swarm
forgets its dead ends and concentrates where the evidence is. fugue's default
threshold is exactly this — resample when ESS / N < 0.5.

**Evidence, for free.** Each weighting step also hands you a piece of the marginal
likelihood. Summing the log of each step's mean weight gives an unbiased estimate
of the log-evidence:

$$
\log \textcolor{#56D364}{p(y_{1:T})}
\;=\;
\sum_{t} \log \sum_i W_{t-1}^{(i)}\;\textcolor{#F2CC60}{p\!\left(y_t \mid x_t^{(i)}\right)}
$$

That number — the `log-evidence` readout — is what makes SMC a model-comparison
tool, not merely a sampler.

## The fugue code

fugue expresses the same state-space model as one `Model`. Each latent state
depends on the previous one; each `observe` is a <span class="fv-c-data">yellow
dot</span>.

```rust,ignore
use fugue::*;

const STEP: f64 = 0.7; // latent random-walk step σ
const OBS: f64 = 0.6; // observation noise τ

// x_0 ~ N(0,1); x_t ~ N(x_{t-1}, STEP); observe each y_t ~ N(x_t, OBS).
// Returns the whole latent path.
fn state_space(ys: Vec<f64>) -> Model<Vec<f64>> {
    let y0 = ys[0];
    let mut m: Model<Vec<f64>> = sample(addr!("x", 0), Normal::new(0.0, 1.0).unwrap())
        .bind(move |x0| {
            observe(addr!("y", 0), Normal::new(x0, OBS).unwrap(), y0).map(move |_| vec![x0])
        });
    for t in 1..ys.len() {
        let yt = ys[t];
        m = m.bind(move |xs| {
            let prev = *xs.last().unwrap(); // depend on the previous state
            sample(addr!("x", t), Normal::new(prev, STEP).unwrap()).bind(move |xt| {
                observe(addr!("y", t), Normal::new(xt, OBS).unwrap(), yt) // the yellow dot
                    .map(move |_| {
                        let mut xs = xs;
                        xs.push(xt);
                        xs
                    })
            })
        });
    }
    m
}
```

Run Sequential Monte Carlo over it. The config is the same knobs you just played
with:

```rust,ignore
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn main() {
    let ys: Vec<f64> = vec![/* your observation series */];

    let mut rng = StdRng::seed_from_u64(42);
    let config = SMCConfig {
        resampling_method: ResamplingMethod::Systematic, // fugue's default
        ess_threshold: 0.5,      // resample when ESS / N < 0.5 — the green/coral band
        rejuvenation_steps: 1,   // MCMC moves that restore diversity after a resample
    };

    let result = adaptive_smc(&mut rng, 200, || state_space(ys.clone()), config);

    println!("particles:    {}", result.particles.len());
    println!("ESS:          {:.1}", effective_sample_size(&result)); // the ESS readout
    println!("log-evidence: {:.3}", result.log_evidence); // unbiased log p(y_1:T)
}
```

Every name here is real: `adaptive_smc`, `SMCConfig`, `ResamplingMethod::Systematic`,
`effective_sample_size`, and `SMCResult::log_evidence` all live in
`src/inference/smc.rs`. `SMCResult` dereferences to `Vec<Particle>`, so
`effective_sample_size(&result)` works directly on the returned population.

```admonish note title="Tempering vs. time"
The widget is a textbook **bootstrap filter**: it steps through the sequence of
observations in *time*. fugue's `adaptive_smc` reaches the same posterior along a
different ladder — it **tempers the likelihood**, targeting
\\( \pi_\beta(\theta) \propto p(\theta)\,p(y\mid\theta)^\beta \\) for \\( \beta \\)
climbing 0 → 1. The machinery is identical either way: weight, watch ESS, resample
below a threshold, and accumulate an unbiased log-evidence. What you learned by
dragging sliders is exactly what the library does under the hood.
```

## Go deeper

- Tutorial: [Sequential Monte Carlo](../tutorials/advanced-inference/sequential-monte-carlo.md)
  — the full API, resampling methods, and rejuvenation.
- API: [`adaptive_smc`](https://docs.rs/fugue-ppl/latest/fugue/inference/smc/fn.adaptive_smc.html)
  and [`effective_sample_size`](https://docs.rs/fugue-ppl/latest/fugue/inference/smc/fn.effective_sample_size.html).
- Next explorable: [A Field Guide to Distributions](./distributions.md) — the
  building blocks every model above is made of.

---

Next: [A Field Guide to Distributions](./distributions.md)
