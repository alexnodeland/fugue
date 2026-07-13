# The Model Is a Score

A `Model` in fugue is not a running program. It is a **score** — pure notation
that describes what *could* happen, note by note, but makes no sound on its own.
A **handler** is the performer that reads the score and decides what each note
means: improvise a fresh value, or replay one from a recording.

Below is a real inference problem. Five <span class="fv-c-data">data
points</span> say where the world landed; a <span class="fv-c-prior">prior</span>
says what you believed about the mean μ beforehand; the
<span class="fv-c-post">posterior</span> is the answer — updated live. Grab a
yellow dot and drag it. The green curve follows in real time. The strip below the
picture is the machinery fugue actually steps through to get there.

This run uses seed <span id="monad-seed" class="fv-scrub" data-min="1" data-max="40" data-step="1" data-value="11">11</span>. A seeded run is a replayable trace — the same seed always draws the same μ.

<div class="fugue-explorable" data-viz="monad" data-seed="11"></div>

## Things to try

1. Drag any <span class="fv-c-data">yellow dot</span> to the right. The
   <span class="fv-c-post">green posterior</span> slides after it and the
   posterior mean readout climbs — the data pulled your belief.
2. Press **Step** five times. Watch the <span class="fv-c-prior">SampleF64</span>
   chip fire first (it sets μ), then each <span class="fv-c-data">ObserveF64</span>
   chip light up **and its data dot pulse in the picture** — chip and dot are the
   same event, seen twice.
3. Press **Perform ×200** with the **PriorHandler** active: 200 fresh μ draws
   rain down as a <span class="fv-c-prior">blue cloud</span> that traces the prior
   curve. The handler is improvising.
4. Flip on the **Replay handler** and press **Perform ×200** again: all 200 draws
   stack on one value — a single <span class="fv-c-hot">coral spike</span>. The
   handler is performing a fixed recording, not drawing.
5. Scrub the **seed** in prior mode — the <span class="fv-c-hot">coral μ tick</span>
   jumps to a new draw. Scrub it in replay mode — it does not move. Same score,
   different performer.

## What you just saw

The score is a tiny probabilistic program: sample a mean, then observe five data
points drawn from it.

$$
\textcolor{#58A6FF}{\mu \sim \mathrm{Normal}(0,\,2)}, \qquad
\textcolor{#F2CC60}{y_i \sim \mathrm{Normal}(\mu,\,1)} \quad i = 1,\dots,5.
$$

Because both the prior and the likelihood are Gaussian, the
<span class="fv-c-post">posterior</span> over μ is Gaussian too, in closed form —
this is the exact green curve you were dragging:

$$
\textcolor{#56D364}{\mu \mid y \sim \mathrm{Normal}(\mu_n,\,\sigma_n^2)}, \qquad
\sigma_n^2 = \left(\tfrac{1}{2^2} + \tfrac{n}{1^2}\right)^{-1}, \qquad
\mu_n = \sigma_n^2\left(\tfrac{\textcolor{#58A6FF}{0}}{2^2} + \tfrac{\textcolor{#F2CC60}{\sum_i y_i}}{1^2}\right).
$$

With the five default points (∑yᵢ = 6.0), that is
<span class="fv-c-post">Normal(1.143, 0.436²)</span> — the numbers in the
readout. The score never computes this. The handler does, one node at a time,
accumulating a log-weight:

$$
\underbrace{\textcolor{#56D364}{\log p(\mu, y)}}_{\text{total log-weight}}
= \underbrace{\textcolor{#58A6FF}{\log p(\mu)}}_{\texttt{log\_prior}}
+ \underbrace{\textcolor{#F2CC60}{\textstyle\sum_i \log p(y_i \mid \mu)}}_{\texttt{log\_likelihood}}.
$$

The key idea: **effects are interpreted, not performed.** The score names a
sample site; it does not draw. Whether a value is improvised
(<span class="fv-c-prior">PriorHandler</span>) or replayed
(<span class="fv-c-flow">ReplayHandler</span>) is the handler's decision, made
later. That is what the Perform ×200 button shows: the *same score* becomes a
cloud of guesses or a single fixed spike, depending only on who performs it. This
separation is the machinery every inference algorithm in fugue is built from —
and fugue's MCMC reaches that same green curve *without* knowing the conjugate
formula, by walking this chain over and over.

### Monads, demystified in one paragraph

A `Model` is built from two operations. `bind` is **"and then"**: run this node,
then feed its value to a function that produces the next node. `pure` is
**"done"**: wrap a plain value as a finished model. That is the entire monad —
`and then` and `done`. Each node stores its continuation `k`, the rest of the
score. The score is a linked list of "and then"s ending in a "done", and it does
not play itself.

## The fugue code

Here is the exact score from the widget. The `sample` line is the
<span class="fv-c-prior">SampleF64</span> chip; each `observe` in the fold is an
<span class="fv-c-data">ObserveF64</span> chip lighting its yellow dot;
`.map(move |_| mu)` is the trailing <span class="fv-c-post">Pure(μ)</span>.

```rust
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

// The score: sample the mean, then observe five data points given it.
fn model(data: Vec<f64>) -> Model<f64> {
    sample(addr!("mu"), Normal::new(0.0, 2.0).unwrap()).bind(move |mu| {
        // Fold the observations into the chain: each one is an ObserveF64 node.
        let mut m = pure(mu);
        for (i, y) in data.into_iter().enumerate() {
            m = m.bind(move |mu| {
                observe(addr!("y", i), Normal::new(mu, 1.0).unwrap(), y)
                    .map(move |_| mu)
            });
        }
        m // ends in Pure(mu): the score terminates, returning mu
    })
}

fn main() {
    let data = vec![1.3, 0.7, 2.1, 0.4, 1.5];

    // PriorHandler improvises: it draws a fresh mu from the RNG.
    let mut rng = StdRng::seed_from_u64(11);
    let (mu, recording) = runtime::handler::run(
        PriorHandler { rng: &mut rng, trace: Trace::default() },
        model(data.clone()),
    );
    println!("prior draw: mu = {mu:.3}, total log-weight = {:.3}",
             recording.total_log_weight());

    // ReplayHandler performs that recording: the same mu, no fresh draw.
    let (mu_again, _) = runtime::handler::run(
        ReplayHandler { rng: &mut rng, base: recording, trace: Trace::default() },
        model(data),
    );
    assert_eq!(mu, mu_again); // same score, replayed exactly
}
```

Under the hood, the score is the `Model` enum. Every effect variant carries its
address, its distribution, and a boxed continuation `k` — the rest of the score
(abridged from `src/core/model.rs`):

```rust,ignore
pub enum Model<A> {
    Pure(A),
    SampleF64 {
        addr: Address,
        dist: Box<dyn Distribution<f64>>,
        k: Box<dyn FnOnce(f64) -> Model<A> + Send + 'static>, // "and then"
    },
    ObserveF64 {
        addr: Address,
        dist: Box<dyn Distribution<f64>>,
        value: f64,
        k: Box<dyn FnOnce(()) -> Model<A> + Send + 'static>,
    },
    Factor { logw: f64, k: Box<dyn FnOnce(()) -> Model<A> + Send + 'static> },
    // ... plus SampleBool/U64/Usize/I64 and their Observe twins, one per
    // return type — this is fugue's type-safety story: a Bernoulli site yields
    // a bool, a Poisson site a u64, never an untyped float.
}
```

The handler you toggled is any type that answers those effects. `run` is the
interpreter — a flat trampoline, not recursion (abridged from
`src/runtime/handler.rs`):

```rust,ignore
pub fn run<A>(mut h: impl Handler, m: Model<A>) -> (A, Trace) {
    let mut m = m;
    let a = loop {
        m = match m {
            Model::Pure(a) => break a,                    // "done": return the value
            Model::SampleF64 { addr, dist, k } => {
                let x = h.on_sample_f64(&addr, &*dist);    // ask the handler
                k(x)                                       // advance to the rest
            }
            Model::ObserveF64 { addr, dist, value, k } => {
                h.on_observe_f64(&addr, &*dist, value);    // score the data
                k(())
            }
            Model::Factor { logw, k } => { h.on_factor(logw); k(()) }
            // ... one arm per variant ...
        };
    };
    (a, h.finish())
}
```

Each node hands back its continuation `k(value)` and the loop goes around again.
Nothing recurses, so the interpreter runs in **constant stack depth** — a model
with 100 000 sample sites is interpreted without overflowing the stack.

## Go deeper

- [Understanding Models](../getting-started/understanding-models.md) — the
  conceptual tour of `Model`, `bind`, and `pure`.
- [Custom Handlers](../how-to/custom-handlers.md) — write your own performer.
- [Trace Manipulation](../tutorials/foundation/trace-manipulation.md) — read and
  edit the recordings a handler produces.
- [`Model`](https://docs.rs/fugue-ppl/latest/fugue/enum.Model.html) and
  [`Handler`](https://docs.rs/fugue-ppl/latest/fugue/trait.Handler.html) on
  docs.rs.
- Next explorable: [Random Walks in Posterior Space](./metropolis.md) — now that
  the score has a value and a weight, how do we *listen* our way back to the
  posterior you just watched form?
