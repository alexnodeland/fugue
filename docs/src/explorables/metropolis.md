# Random Walks in Posterior Space

You have data and a line you want to fit through it — but you want the *whole
posterior* over slope and intercept, not a single answer. That posterior has no
formula you can read off. So you walk. Stand at some (slope, intercept), propose a
small random step, and keep it more often when it explains the data better. Do this
long enough and the places you linger *are* the posterior.

That walk is Metropolis-Hastings. The score names a shape — here, the posterior over
two numbers; the walker is a performer who never sees the whole shape at once, only
whether the next fit sounds better or worse than the last.

<div class="fugue-explorable" data-viz="metropolis" data-seed="11"></div>

Two linked spaces. On the **left**, the <span class="fv-c-data">data</span> — twelve
points you can grab and drag. Every <span class="fv-c-post">green line</span> is a
recently accepted fit; the <span class="fv-c-hot">coral line</span> is where each chain
stands now; rejected proposals flash coral-dashed and vanish. On the **right**, the
same walk seen in <span class="fv-c-post">parameter space</span>: a live posterior
heatmap over (slope, intercept), with the <span class="fv-c-hot">coral dot</span>
threading it and <span class="fv-c-prior">blue proposal</span> arrows firing each tick.
A point in the right panel *is* a line in the left panel — watch them move together.

The model is honest Bayesian linear regression:
$y \sim \mathcal{N}(a\,x + b,\ \sigma_{\text{obs}})$ with $\sigma_{\text{obs}}$ fixed at
$0.8$, and priors $a, b \sim \mathcal{N}(0, 2.5)$. The
<span class="fv-c-post">split-R̂</span> and ESS readouts are fugue 0.2.0's real
convergence diagnostics, computed live on the samples accruing on screen.

## Things to try

1. **Press Play, then drag a point far off the line.** The right-hand heatmap morphs
   and the whole chain migrates to the new best fit — same frame. This is the moment:
   the data *is* the posterior, and you are reshaping it with your cursor.
2. **Drag `PROPOSAL σ` down to `0.02`.** Acceptance climbs toward 100% — every tiny
   step is safe — yet the coral dot barely crawls and R̂ stays stubbornly above 1. High
   acceptance is not the goal.
3. **Now drag σ up to `4`.** Steps overshoot the tight posterior ridge and almost every
   proposal is rejected. The chain freezes, twitching in place. Too bold is as stuck as
   too timid.
4. **Find the Goldilocks band** (σ near `0.3`–`0.6`). Acceptance settles around 25–45%,
   the green spaghetti fans tightly around the true line, and R̂ falls toward 1.0.
5. **Set `CHAINS` to 4.** They start from dispersed corners of parameter space. When σ is
   small, watch them stay marooned apart — R̂ stays high because they *disagree*. One
   chain alone could never have told you it was stuck.

## What you just saw

Metropolis-Hastings builds a Markov chain whose stationary distribution *is* the
posterior. Write the parameters as $\textcolor{#FF7B72}{\theta} = (a, b)$. From the
current state you draw a proposal $\textcolor{#58A6FF}{\theta'}$ from a symmetric
Gaussian kernel, $\textcolor{#58A6FF}{\theta'} \sim
\mathcal{N}(\textcolor{#FF7B72}{\theta}, \sigma^2 I)$, and accept it with probability

$$\alpha = \min\!\left(1,\ \frac{\textcolor{#58A6FF}{p(\theta')}\ \textcolor{#F2CC60}{p(\mathcal{D}\mid\theta')}}{\textcolor{#58A6FF}{p(\theta)}\ \textcolor{#F2CC60}{p(\mathcal{D}\mid\theta)}}\right).$$

The ratio is <span class="fv-c-prior">prior</span> times
<span class="fv-c-data">likelihood</span>, new over old — the evidence cancels, which
is the whole trick: you never need the intractable normalizer. Because the proposal is
symmetric, $q(\theta'\mid\theta)=q(\theta\mid\theta')$ drops out too. Here the prior is
$\mathcal{N}(0,2.5)$ on each of $a,b$ and the likelihood is the product of the twelve
$\mathcal{N}(a x_i + b,\ 0.8)$ terms — one per yellow point. Computed in log space,
where the widget lives, acceptance is a subtraction:

$$\log\alpha = \min\!\big(0,\ \log p(\textcolor{#58A6FF}{\theta'}) - \log p(\textcolor{#FF7B72}{\theta})\big).$$

Uphill moves ($\log\alpha = 0$) are always taken; downhill moves are taken with
probability $e^{\log\alpha}$. That occasional downhill step is what lets the chain
explore the full spread of plausible lines instead of collapsing onto the single best
fit — which is exactly the green spaghetti you see fanning around the data.

**Why σ is a dial, not a detail.** The proposal scale trades off two failures. Too small
and consecutive samples are nearly identical — high autocorrelation, so your 2000 draws
carry the information of a handful. Too large and you reject constantly, so the chain
sits still — again few effective draws. The
<span class="fv-c-post">effective sample size</span> (ESS) measures exactly this: how
many *independent* draws your correlated chain is worth.

$$\mathrm{ESS} = \frac{mn}{\hat\tau}, \qquad \hat\tau = 1 + 2\sum_{k\ge 1}\hat\rho_k,$$

where $\hat\rho_k$ is the autocorrelation at lag $k$ and $\hat\tau$ the integrated
autocorrelation time. Fugue estimates $\hat\tau$ with Geyer's initial positive sequence,
pooled across chains (Vehtari et al. 2021).

**Why more than one chain.** A single walker stuck in one corner of parameter space looks
perfectly converged from the inside. <span class="fv-c-post">Split-R̂</span> compares the
variance *between* chains to the variance *within* them, after splitting each chain in
half so a slow within-chain drift can't hide:

$$\hat R = \sqrt{\frac{\widehat{\mathrm{var}}^{+}}{W}}, \qquad
\widehat{\mathrm{var}}^{+} = \frac{n-1}{n}W + \frac{1}{n}B.$$

$W$ is the within-chain variance, $B$ the between-chain variance. When the chains agree,
$B \to 0$ and $\hat R \to 1$. Anything above $1.01$ means they haven't mixed. This is the
same split statistic fugue reports from
[`r_hat_f64`](https://docs.rs/fugue-ppl/latest/fugue/fn.r_hat_f64.html) — the widget
ports its arithmetic verbatim.

## The fugue code

The widget tunes σ by hand. Fugue's `adaptive_mcmc_chain` does it for you, nudging each
site's proposal scale toward the 0.44 acceptance rate you just discovered is healthy —
then you referee convergence with the very diagnostics on screen. This is the same
regression model as the widget: `slope` and `intercept` sampled from priors, one
`observe` per data point.

```rust,ignore
use fugue::*;
use fugue::inference::mh::adaptive_mcmc_chain;
use fugue::inference::diagnostics::r_hat_f64;
use fugue::inference::mcmc_utils::effective_sample_size_mcmc;
use rand::{rngs::StdRng, SeedableRng};

// Bayesian linear regression — the same model the widget samples.
// slope (a) and intercept (b) each get a Normal(0, 2.5) prior; every data
// point contributes one yellow observation y_i ~ Normal(a*x_i + b, 0.8).
fn regression(x: Vec<f64>, y: Vec<f64>) -> impl Fn() -> Model<(f64, f64)> {
    move || {
        let x = x.clone();
        let y = y.clone();
        prob! {
            let a <- sample(addr!("slope"),     Normal::new(0.0, 2.5).unwrap());
            let b <- sample(addr!("intercept"), Normal::new(0.0, 2.5).unwrap());
            // One observe per point — this loop IS the yellow data panel.
            let _obs <- plate!(i in 0..x.len() => {
                observe(addr!("y", i), Normal::new(a * x[i] + b, 0.8).unwrap(), y[i])
            });
            pure((a, b))
        }
    }
}

fn main() {
    // Synthetic data from the true line y = 0.8x - 0.4 + noise, same as the widget.
    let x: Vec<f64> = (0..12).map(|i| -3.0 + 6.0 * i as f64 / 11.0).collect();
    let mut dgen = StdRng::seed_from_u64(11);
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| Normal::new(0.8 * xi - 0.4, 0.8).unwrap().sample(&mut dgen))
        .collect();

    // Reproducibility is a fugue value: a seed *is* the chain. Run four chains
    // and let split-R-hat referee whether they agree.
    let mut chains: Vec<Vec<Trace>> = Vec::new();
    for seed in [11u64, 12, 13, 14] {
        let mut rng = StdRng::seed_from_u64(seed);
        // 2000 kept draws, 500 warmup steps of proposal adaptation.
        let draws = adaptive_mcmc_chain(&mut rng, regression(x.clone(), y.clone()), 2_000, 500);
        chains.push(draws.into_iter().map(|(_, trace)| trace).collect());
    }

    // Split-R-hat (Vehtari et al. 2021) and ESS — 0.2.0's diagnostics, the same
    // numbers the widget shows.
    let rhat = r_hat_f64(&chains, &addr!("slope"));
    let slope_chain0: Vec<f64> = chains[0]
        .iter()
        .filter_map(|t| t.get_f64(&addr!("slope")))
        .collect();
    let ess = effective_sample_size_mcmc(&slope_chain0);

    println!("split-R-hat(slope) = {rhat:.3}   ESS(slope, chain 0) = {ess:.0}");
}
```

`adaptive_mcmc_chain(&mut rng, model_fn, 2_000, 500)` runs single-site Metropolis with a
diminishing-adaptation schedule targeting 0.44 acceptance — the middle of the Goldilocks
band from **Things to try #4**. It returns `Vec<(A, Trace)>`: the returned value and the
full recording behind it. `r_hat_f64` is the coral/green R̂ readout;
`effective_sample_size_mcmc` is the ESS readout.

## Go deeper

- **Next explorable:** [Rolling, Not Guessing: HMC](./hmc.md) — the same regression, the
  same twelve points, but the walker rolls downhill with momentum instead of guessing,
  and covers the posterior in a fraction of the steps.
- **Tutorial:** [Basic Inference](../getting-started/basic-inference.md) — running MCMC
  end to end on a real model.
- **How-to:** [Debugging Models](../how-to/debugging-models.md) — reading R̂ and ESS when
  a chain misbehaves.
- **API:** [`adaptive_mcmc_chain`](https://docs.rs/fugue-ppl/latest/fugue/fn.adaptive_mcmc_chain.html)
  · [`r_hat_f64`](https://docs.rs/fugue-ppl/latest/fugue/fn.r_hat_f64.html)
