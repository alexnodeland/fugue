# Rolling, Not Guessing: Hamiltonian Monte Carlo

Random-walk Metropolis proposes blindly and hopes. Hamiltonian Monte Carlo does
something smarter: it gives the sampler *momentum* and lets it **roll** across the
posterior like a ball on a landscape, following the slope instead of guessing
against it. One good roll crosses ground that a random walk needs hundreds of
timid steps to cover.

Same problem as the [Metropolis explorable](./metropolis.md): fit a straight line
`y = a·x + b` to twelve noisy points. **Left** is data space — the yellow points and
a fan of candidate fit lines. **Right** is parameter space — the posterior over
`(slope, intercept)`, with the sampler rolling through it. A point on the right **is**
a line on the left: when the coral ball moves right, its coral line swings on the left.

<div class="fugue-explorable" data-viz="hmc" data-seed="11"></div>

These are the *same* twelve seeded points as the Metropolis page — run them next to
each other. Every run is a replayable trace: fix the <span id="hmc-seed">11</span>
seed and you get the exact same momenta and trajectories every time.

## Things to try

1. Press **Play**. Watch one violet **leapfrog trajectory** roll across parameter
   space — a single proposal travelling much farther than a random-walk hop — while
   its coral **fit line** swings across the data on the left.
2. **Drag the rightmost yellow point far up.** The posterior heatmap tilts toward
   steeper slopes and the coral ball rolls after it within a few transitions. This
   linked deformation is the whole point of the page.
3. Turn on **MH side-by-side**. Both samplers get the same number of gradient-budget
   evaluations; the random walk's fits are the *dim* green spaghetti, HMC's the bright
   green — HMC fans across the plausible lines noticeably faster.
4. Push **STEP ε** up past `0.25`. Trajectories start glowing coral and the
   **divergence** counter ticks — the integrator has gone unstable and every such
   proposal is rejected.
5. Set **LEAPFROG L** to `1`. HMC collapses toward a random walk — the momentum never
   gets to carry the ball anywhere. Watch the energy strip lurch.

## What you just saw

The target is an honest Bayesian linear regression — no banana, no toy density. Each
observation is Gaussian around the line, with **fixed** noise `σ_obs = 0.8`, and both
parameters get a `Normal(0, 2.5)` prior:

$$\log \pi(\textcolor{#56D364}{a,b}) = \sum_i \textcolor{#F2CC60}{\log \mathcal N\!\big(y_i \mid a x_i + b,\ 0.8\big)} \;+\; \textcolor{#58A6FF}{\log \mathcal N(a \mid 0, 2.5)} \;+\; \textcolor{#58A6FF}{\log \mathcal N(b \mid 0, 2.5)}$$

The <span class="fv-c-data">likelihood</span> (yellow) pulls the line through the
points; the <span class="fv-c-prior">prior</span> (blue) keeps the coefficients
sane; together they make the <span class="fv-c-post">posterior</span> (green) — the
heatmap on the right.

HMC augments the position `q = (a, b)` with a fresh **momentum** `p` drawn from a
Gaussian, then simulates a physical system whose total energy is the **Hamiltonian**:

$$H(q,p) = \textcolor{#56D364}{U(q)} + \textcolor{#BC8CFF}{K(p)}, \qquad
\textcolor{#56D364}{U(q)} = -\log \pi(q), \qquad
\textcolor{#BC8CFF}{K(p)} = \tfrac{1}{2}\,p^\top M^{-1} p$$

The <span class="fv-c-post">potential energy</span> **is** the negative log-posterior
— the green surface the ball rolls on. The <span class="fv-c-flow">kinetic energy</span>
is the momentum you flick it with. Low posterior density means high potential, so the
ball is pulled toward the high-probability valley, exactly where you want samples.

The trajectory is integrated by the **leapfrog** scheme — a half-kick to momentum, a
full drift in position, another half-kick:

$$\textcolor{#BC8CFF}{p_{t+\frac12}} = \textcolor{#BC8CFF}{p_t} + \tfrac{\varepsilon}{2}\,\nabla_{\!q}\log\pi(\textcolor{#56D364}{q_t})$$
$$\textcolor{#56D364}{q_{t+1}} = \textcolor{#56D364}{q_t} + \varepsilon\,M^{-1}\,\textcolor{#BC8CFF}{p_{t+\frac12}}$$
$$\textcolor{#BC8CFF}{p_{t+1}} = \textcolor{#BC8CFF}{p_{t+\frac12}} + \tfrac{\varepsilon}{2}\,\nabla_{\!q}\log\pi(\textcolor{#56D364}{q_{t+1}})$$

Here the gradient `∇ log π` is available in closed form (a Gaussian model), and the
widget uses it exactly — that is why the leapfrog force points straight at the
posterior mode. Leapfrog is **reversible** and **volume-preserving**, so the
Metropolis correction at the end of the trajectory has no Jacobian term. It reduces to
a comparison of total energy at the endpoints:

$$\alpha = \min\!\Big(1,\ \exp\big(H(q,p) - H(q',p')\big)\Big)$$

If the integrator were exact, `H` would be conserved and every proposal accepted. It
is not exact, so a small **energy error** `ΔH` remains — that is the number on the
strip chart. Keep it small (tune `ε`) and acceptance stays high. Let it explode and
the proposal **diverges**: the trajectory shoots off, `ΔH` blows past any sane bound,
and the sample is thrown away. Divergences are not a bug to hide — real samplers count
and report them as a warning that the geometry is too sharp for the current step size.

## The fugue code

Fugue ships HMC as `hmc_chain`. Models are ordinary Rust closures with no autodiff, so
the force `∇ log π` is computed by **central finite differences** — an approximate
*force*, but an exact accept/reject against the true log-density, so the stationary
distribution is exactly the posterior (the finite-difference step only costs
efficiency, never correctness).

```rust,ignore
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

// The widget's model: fit y = a·x + b with fixed observation noise. The green
// heatmap is exactly this posterior over (slope, intercept); the coral ball is
// the current (a, b) rolling toward the values the yellow points support.
fn regression(xs: Vec<f64>, ys: Vec<f64>) -> Model<(f64, f64)> {
    prob! {
        let a <- sample(addr!("slope"),     Normal::new(0.0, 2.5).unwrap()); // blue prior
        let b <- sample(addr!("intercept"), Normal::new(0.0, 2.5).unwrap()); // blue prior
        // one yellow datum per point; sigma_obs fixed at 0.8
        let _obs <- plate!(i in xs.iter().zip(ys.iter()).enumerate() => {
            let (k, (x, y)) = i;
            observe(addr!("y", k), Normal::new(a * x + b, 0.8).unwrap(), *y)
        });
        pure((a, b))
    }
}

fn main() {
    // The same twelve seeded points the widget draws.
    let xs: Vec<f64> = (0..12).map(|i| -3.0 + 6.0 * i as f64 / 11.0).collect();
    let ys: Vec<f64> = xs.iter().map(|&x| 0.8 * x - 0.4).collect(); // + noise in practice

    let mut rng = StdRng::seed_from_u64(11); // a seed is a replayable trace

    // Mirror the widget's sliders: L leapfrog steps, an initial step size eps.
    let config = HMCConfig {
        n_leapfrog: 25,               // the LEAPFROG L slider
        init_step_size: Some(0.08),   // the STEP ε slider (warmup still tunes it
                                      // by dual averaging toward target_accept)
        target_accept: 0.8,           // Hoffman & Gelman's recommended target
        ..HMCConfig::default()        // finite_diff_eps = 1e-5, adapt_mass = false
    };

    // hmc_chain(rng, model_fn, n_samples, n_warmup, config) -> Vec<(A, Trace)>
    let model_fn = move || regression(xs.clone(), ys.clone());
    let samples = hmc_chain(&mut rng, model_fn, 1000, 500, config);

    let slopes: Vec<f64> = samples.iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("slope")))
        .collect();
    let mean = slopes.iter().sum::<f64>() / slopes.len() as f64;
    println!("posterior mean slope ≈ {mean:.3}"); // near the true 0.8
}
```

The same **ESS** you watch converge in the widget is a real diagnostic. Run a few
chains and combine them with the multi-chain estimators (new in 0.2.0):

```rust,ignore
use fugue::*;

// After collecting `chains: Vec<Vec<Trace>>` from several hmc_chain runs:
let slope_series: Vec<Vec<f64>> = chains.iter()
    .map(|c| c.iter().filter_map(|t| t.get_f64(&addr!("slope"))).collect())
    .collect();

let ess = effective_sample_size(&slope_series[0]); // per-chain ESS
let rhat = r_hat_f64(&chains, &addr!("slope"));     // split-R̂: want < 1.01
println!("ESS = {ess:.0},  R̂ = {rhat:.3}");
```

`hmc_chain` holds any discrete sites fixed during the roll (a Metropolis-within-Gibbs
treatment), so compose it with `adaptive_mcmc_chain` when a model mixes continuous and
discrete latents.

## Go deeper

- **Compare by hand:** [Random Walks in Posterior Space](./metropolis.md) — the same
  twelve points, sampled the slow way.
- **Tutorial:** [Basic Inference](../getting-started/basic-inference.md) covers when
  to reach for HMC over MH or SMC.
- **API docs:** [`hmc_chain`](https://docs.rs/fugue-ppl/latest/fugue/inference/hmc/fn.hmc_chain.html)
  and [`HMCConfig`](https://docs.rs/fugue-ppl/latest/fugue/inference/hmc/struct.HMCConfig.html).
- **Next:** [Particles That Tell Stories](./smc.md) — inference that moves through
  time instead of space.
