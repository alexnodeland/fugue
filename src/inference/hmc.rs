//! Hamiltonian Monte Carlo (HMC) over the continuous (`f64`) sites of a trace
//! (FG-31).
//!
//! This is fugue's first gradient-based inference kernel. It targets the
//! unnormalized log-joint of a model — `log_prior + log_likelihood +
//! log_factors`, i.e. [`Trace::total_log_weight`] — as a function of the model's
//! continuous latent sites, and moves all of them jointly by simulating
//! Hamiltonian dynamics. This mixes far better than single-site
//! Metropolis-Hastings on correlated / higher-dimensional continuous posteriors,
//! which is exactly the gap [`crate::inference::mh`] leaves open.
//!
//! # Why finite-difference forces are still EXACT
//!
//! fugue models are ordinary Rust closures with no automatic differentiation, so
//! this kernel computes the force `∇ log π(q)` by **deterministic central finite
//! differences**:
//!
//! ```text
//! ∂/∂q_i log π(q) ≈ (log π(q + h·e_i) − log π(q − h·e_i)) / (2h)
//! ```
//!
//! A natural worry is that an *approximate* gradient makes the sampler
//! *approximate*. It does not. The argument, spelled out:
//!
//! 1. The leapfrog (velocity-Verlet) integrator applied to ANY fixed,
//!    deterministic force field `F(q)` is a smooth, **time-reversible** and
//!    **volume-preserving** (symplectic-form-preserving; its Jacobian has
//!    determinant 1) map on phase space `(q, p)`. Reversibility and volume
//!    preservation are algebraic properties of the leapfrog update equations —
//!    they hold for the finite-difference `F` just as they do for the exact
//!    gradient, because the derivation never assumes `F = ∇log π`.
//! 2. Because the proposal map is deterministic, an involution (composing it with
//!    a momentum flip is its own inverse), and volume-preserving, the
//!    Metropolis–Hastings acceptance ratio collapses to `exp(H(q,p) − H(q',p'))`
//!    with **no Jacobian correction**, where the Hamiltonian
//!    `H(q,p) = −log π(q) + ½ pᵀM⁻¹p` uses the TRUE `log π` (evaluated exactly by
//!    running the model), not the approximate force.
//! 3. Metropolis–Hastings with a proposal that is reversible and volume
//!    preserving leaves the target `π` exactly invariant regardless of how the
//!    proposal was generated. The approximate force only steers the trajectory;
//!    the accept/reject step, driven by the exactly-evaluated `H`, is what
//!    guarantees detailed balance.
//!
//! So the finite-difference approximation costs only **efficiency** (a rougher
//! force yields larger energy errors and hence lower acceptance / shorter usable
//! step sizes), never **correctness**. The stationary distribution is exactly the
//! model posterior. Dual-averaging step-size adaptation (below) then tunes the
//! step size so the energy error — and thus the acceptance rate — stays in the
//! efficient regime.
//!
//! # What the kernel does
//!
//! * **Leapfrog integrator** with a configurable number of steps `L`
//!   ([`HMCConfig::n_leapfrog`]) and an identity mass matrix by default (optional
//!   diagonal mass adaptation via [`HMCConfig::adapt_mass`]).
//! * **Dual-averaging step-size adaptation** to a target acceptance probability
//!   (default 0.8) during warmup, following Hoffman & Gelman (2014) §3.2
//!   (Algorithms 4 & 5). The step size is **frozen** at its dual-averaging
//!   running average `ε̄` once warmup ends, so the sampling phase uses a fixed,
//!   time-homogeneous transition kernel.
//! * **Bounded-support sites** (e.g. a `Gamma`/`LogNormal`/`Beta` latent) are
//!   handled by the target itself: a proposal that leaves the support scores
//!   `log π = −∞`, which makes the trajectory *divergent* and forces rejection.
//!   This is correct but can be inefficient near a hard boundary (the trajectory
//!   is frequently rejected there); reparameterizing to an unconstrained space is
//!   the standard remedy and is left to the user.
//!
//! Only `f64` sites participate in the dynamics. Any discrete sites in the trace
//! are held fixed at their current values for the duration of the HMC update
//! (a Metropolis-within-Gibbs treatment); compose with [`crate::inference::mh`]
//! to also move discrete sites.
//!
//! # Example
//!
//! ```rust
//! use fugue::*;
//! use fugue::inference::hmc::{hmc_chain, HMCConfig};
//! use rand::rngs::StdRng;
//! use rand::SeedableRng;
//!
//! // Conjugate Normal-Normal: prior mu ~ N(0,1), likelihood y ~ N(mu, 1).
//! let model_fn = || {
//!     sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap())
//!         .bind(|mu| observe(addr!("y"), Normal::new(mu, 1.0).unwrap(), 2.0).map(move |_| mu))
//! };
//!
//! let mut rng = StdRng::seed_from_u64(1);
//! let samples = hmc_chain(&mut rng, model_fn, 200, 200, HMCConfig::default());
//! let mus: Vec<f64> = samples.iter().map(|(mu, _)| *mu).collect();
//! let mean = mus.iter().sum::<f64>() / mus.len() as f64;
//! // Posterior mean is 1.0 for this problem; HMC recovers it.
//! assert!((mean - 1.0).abs() < 0.3);
//! ```

use crate::core::address::Address;
use crate::core::model::Model;
use crate::runtime::handler::run;
use crate::runtime::interpreters::{PriorHandler, ScoreGivenTrace};
use crate::runtime::trace::{ChoiceValue, Trace};

use rand::Rng;
use rand_distr::StandardNormal;

/// Configuration for [`hmc_chain`].
#[derive(Clone, Copy, Debug)]
pub struct HMCConfig {
    /// Number of leapfrog steps per proposal (`L`). Longer trajectories
    /// decorrelate faster per iteration but cost more gradient evaluations.
    pub n_leapfrog: usize,
    /// Target Metropolis acceptance probability for dual averaging (Hoffman &
    /// Gelman recommend 0.8 for HMC/NUTS).
    pub target_accept: f64,
    /// Initial leapfrog step size. `None` runs the Hoffman & Gelman (2014)
    /// Algorithm 4 "reasonable initial step size" heuristic.
    pub init_step_size: Option<f64>,
    /// Central finite-difference spacing `h` used for the force evaluation.
    pub finite_diff_eps: f64,
    /// Enable diagonal mass-matrix adaptation from the warmup draws. The inverse
    /// mass is set to the estimated marginal variances at the warmup midpoint and
    /// the step size is re-tuned for the second half of warmup. Default `false`
    /// (identity mass).
    pub adapt_mass: bool,
}

impl Default for HMCConfig {
    fn default() -> Self {
        HMCConfig {
            n_leapfrog: 16,
            target_accept: 0.8,
            init_step_size: None,
            finite_diff_eps: 1e-5,
            adapt_mass: false,
        }
    }
}

/// Dual-averaging step-size adaptation (Hoffman & Gelman 2014, Algorithm 5,
/// §3.2). Drives the running step size so the mean Metropolis acceptance
/// probability converges to `target`.
#[derive(Clone, Debug)]
struct DualAveraging {
    mu: f64,
    log_eps_bar: f64,
    h_bar: f64,
    m: u64,
    gamma: f64,
    t0: f64,
    kappa: f64,
    target: f64,
}

impl DualAveraging {
    fn new(eps0: f64, target: f64) -> Self {
        DualAveraging {
            mu: (10.0 * eps0).ln(),
            log_eps_bar: 0.0,
            h_bar: 0.0,
            m: 0,
            gamma: 0.05,
            t0: 10.0,
            kappa: 0.75,
            target,
        }
    }

    /// Feed the acceptance statistic `alpha` (a probability in `[0, 1]`) for the
    /// completed iteration and return the step size to use next.
    fn update(&mut self, alpha: f64) -> f64 {
        self.m += 1;
        let m = self.m as f64;
        let a = alpha.clamp(0.0, 1.0);
        let frac = 1.0 / (m + self.t0);
        self.h_bar = (1.0 - frac) * self.h_bar + frac * (self.target - a);
        let log_eps = self.mu - (m.sqrt() / self.gamma) * self.h_bar;
        let w = m.powf(-self.kappa);
        self.log_eps_bar = w * log_eps + (1.0 - w) * self.log_eps_bar;
        log_eps.exp()
    }

    /// The frozen (averaged) step size used after warmup.
    fn frozen_step(&self) -> f64 {
        self.log_eps_bar.exp()
    }
}

/// Online per-dimension mean/variance (Welford) for diagonal mass adaptation.
struct Welford {
    n: u64,
    mean: Vec<f64>,
    m2: Vec<f64>,
}

impl Welford {
    fn new(d: usize) -> Self {
        Welford {
            n: 0,
            mean: vec![0.0; d],
            m2: vec![0.0; d],
        }
    }

    fn push(&mut self, x: &[f64]) {
        self.n += 1;
        let n = self.n as f64;
        for ((xi, mean), m2) in x.iter().zip(self.mean.iter_mut()).zip(self.m2.iter_mut()) {
            let delta = xi - *mean;
            *mean += delta / n;
            let delta2 = xi - *mean;
            *m2 += delta * delta2;
        }
    }

    /// Regularized sample variances. Falls back to 1.0 when there is too little
    /// data or a degenerate (zero-variance) coordinate, so the mass matrix stays
    /// positive-definite.
    fn variances(&self) -> Vec<f64> {
        if self.n < 2 {
            return vec![1.0; self.mean.len()];
        }
        let denom = (self.n - 1) as f64;
        self.m2
            .iter()
            .map(|&s| {
                let v = s / denom;
                if v.is_finite() && v > 1e-8 {
                    v
                } else {
                    1.0
                }
            })
            .collect()
    }
}

/// Extract the ordered list of continuous (`f64`) site addresses and their
/// current values from a trace. Iteration order is the trace's `BTreeMap` order,
/// so it is deterministic across calls.
fn positions_from_trace(trace: &Trace) -> (Vec<Address>, Vec<f64>) {
    let mut sites = Vec::new();
    let mut q = Vec::new();
    for (addr, choice) in &trace.choices {
        if let ChoiceValue::F64(v) = choice.value {
            sites.push(addr.clone());
            q.push(v);
        }
    }
    (sites, q)
}

/// Clone `base` and overwrite its continuous site values with `q`. Discrete
/// sites are left untouched (held fixed during the HMC update).
fn trace_with_positions(base: &Trace, sites: &[Address], q: &[f64]) -> Trace {
    let mut t = base.clone();
    for (addr, &val) in sites.iter().zip(q.iter()) {
        if let Some(choice) = t.choices.get_mut(addr) {
            choice.value = ChoiceValue::F64(val);
        }
    }
    t
}

/// Evaluate the unnormalized log-joint `log π(q)` by scoring the model against a
/// trace whose continuous sites are set to `q`. One model execution.
fn log_joint_at<A>(
    model_fn: &impl Fn() -> Model<A>,
    base: &Trace,
    sites: &[Address],
    q: &[f64],
) -> f64 {
    let candidate = trace_with_positions(base, sites, q);
    let (_a, scored) = run(
        ScoreGivenTrace {
            base: candidate,
            trace: Trace::default(),
        },
        model_fn(),
    );
    scored.total_log_weight()
}

/// Score the model at `q` and also return the model result and the freshly
/// scored trace (used to materialize an accepted state). One model execution.
fn score_full<A>(
    model_fn: &impl Fn() -> Model<A>,
    base: &Trace,
    sites: &[Address],
    q: &[f64],
) -> (A, Trace, f64) {
    let candidate = trace_with_positions(base, sites, q);
    let (a, scored) = run(
        ScoreGivenTrace {
            base: candidate,
            trace: Trace::default(),
        },
        model_fn(),
    );
    let lw = scored.total_log_weight();
    (a, scored, lw)
}

/// Central finite-difference force `∇ log π(q)`. Costs `2·d` model executions.
/// Returns `(gradient, all_finite)`; a non-finite component signals the
/// trajectory has left the support (bounded-site boundary) and is divergent.
fn grad_log_joint<A>(
    model_fn: &impl Fn() -> Model<A>,
    base: &Trace,
    sites: &[Address],
    q: &[f64],
    h: f64,
) -> (Vec<f64>, bool) {
    let d = q.len();
    let mut g = vec![0.0; d];
    let mut ok = true;
    let mut qq = q.to_vec();
    for i in 0..d {
        let orig = qq[i];
        qq[i] = orig + h;
        let lp = log_joint_at(model_fn, base, sites, &qq);
        qq[i] = orig - h;
        let lm = log_joint_at(model_fn, base, sites, &qq);
        qq[i] = orig;
        let gi = (lp - lm) / (2.0 * h);
        if !gi.is_finite() {
            ok = false;
        }
        g[i] = gi;
    }
    (g, ok)
}

/// One recorded point of a leapfrog trajectory: the position `q` over the
/// continuous sites (ordered as [`HmcSession::sites`]) and the Hamiltonian
/// `H = -log π(q) + ½ pᵀM⁻¹p` at that point. Produced by
/// [`HmcSession::step_recorded`]; a rising `h` along a trajectory signals
/// integration error, and a divergence truncates the recording at the point
/// the support was left.
#[derive(Clone, Debug)]
pub struct LeapfrogPoint {
    /// Position over the continuous sites.
    pub q: Vec<f64>,
    /// Hamiltonian (potential + kinetic energy) at this point.
    pub h: f64,
}

/// Leapfrog (velocity-Verlet) integration of the Hamiltonian dynamics for `l`
/// steps at step size `eps`. The force is reused between the trailing half-kick
/// of one step and the leading half-kick of the next, so the whole trajectory
/// costs `L + 1` gradient evaluations. Returns the endpoint `(q, p)` and a
/// `divergent` flag set when a force evaluation is non-finite (support left).
/// With `record` attached, each integration point (including the start) is
/// pushed with its Hamiltonian — one extra model execution per point.
#[allow(clippy::too_many_arguments)]
fn leapfrog<A>(
    model_fn: &impl Fn() -> Model<A>,
    base: &Trace,
    sites: &[Address],
    q0: &[f64],
    p0: &[f64],
    eps: f64,
    l: usize,
    h: f64,
    m_inv: &[f64],
    mut record: Option<&mut Vec<LeapfrogPoint>>,
) -> (Vec<f64>, Vec<f64>, bool) {
    let d = q0.len();
    let mut q = q0.to_vec();
    let mut p = p0.to_vec();

    // Hamiltonian at the current (q, p); one extra model execution, so only
    // evaluated when a recorder is attached.
    let record_point = |q: &[f64], p: &[f64], out: &mut Vec<LeapfrogPoint>| {
        let k = 0.5 * (0..d).map(|i| p[i] * p[i] * m_inv[i]).sum::<f64>();
        let lj = log_joint_at(model_fn, base, sites, q);
        out.push(LeapfrogPoint {
            q: q.to_vec(),
            h: -lj + k,
        });
    };
    if let Some(out) = record.as_deref_mut() {
        record_point(&q, &p, out);
    }

    let (mut grad, ok) = grad_log_joint(model_fn, base, sites, &q, h);
    if !ok {
        return (q, p, true);
    }
    for _ in 0..l {
        for i in 0..d {
            p[i] += 0.5 * eps * grad[i];
        }
        for i in 0..d {
            q[i] += eps * m_inv[i] * p[i];
        }
        let (g2, ok2) = grad_log_joint(model_fn, base, sites, &q, h);
        grad = g2;
        if !ok2 {
            return (q, p, true);
        }
        for i in 0..d {
            p[i] += 0.5 * eps * grad[i];
        }
        if let Some(out) = record.as_deref_mut() {
            record_point(&q, &p, out);
        }
    }
    (q, p, false)
}

/// One HMC transition. Returns
/// `(accepted, q_next, endpoint_if_accepted, acceptance_probability, divergent)`.
///
/// `endpoint_if_accepted` carries the model result, freshly-scored trace, and
/// log-joint of the accepted state. `divergent` is set when the trajectory
/// left the support (non-finite force) or landed on a non-finite log-joint;
/// a divergent proposal is always rejected with acceptance probability 0.
type TransitionOut<A> = (bool, Vec<f64>, Option<(A, Trace, f64)>, f64, bool);

#[allow(clippy::too_many_arguments)]
fn hmc_transition<A: Clone, R: Rng>(
    rng: &mut R,
    model_fn: &impl Fn() -> Model<A>,
    base: &Trace,
    sites: &[Address],
    q_cur: &[f64],
    lj_cur: f64,
    eps: f64,
    l: usize,
    h: f64,
    m_inv: &[f64],
    mass_sqrt: &[f64],
    record: Option<&mut Vec<LeapfrogPoint>>,
) -> TransitionOut<A> {
    let d = q_cur.len();

    // Refresh momentum p ~ N(0, M), std_i = sqrt(mass_i) = mass_sqrt[i].
    let p0: Vec<f64> = (0..d)
        .map(|i| {
            let z: f64 = rng.sample(StandardNormal);
            z * mass_sqrt[i]
        })
        .collect();
    let k0 = 0.5 * (0..d).map(|i| p0[i] * p0[i] * m_inv[i]).sum::<f64>();
    let h0 = -lj_cur + k0;

    let (q_new, p_new, divergent) =
        leapfrog(model_fn, base, sites, q_cur, &p0, eps, l, h, m_inv, record);
    if divergent {
        return (false, q_cur.to_vec(), None, 0.0, true);
    }

    let (a_new, t_new, lj_new) = score_full(model_fn, base, sites, &q_new);
    if !lj_new.is_finite() {
        return (false, q_cur.to_vec(), None, 0.0, true);
    }

    let k_new = 0.5 * (0..d).map(|i| p_new[i] * p_new[i] * m_inv[i]).sum::<f64>();
    let h_new = -lj_new + k_new;

    // Exact MH accept using the true Hamiltonian (see module docs).
    let accept_prob = (h0 - h_new).exp().min(1.0);
    let accept = rng.gen::<f64>() < accept_prob;
    if accept {
        (
            true,
            q_new,
            Some((a_new, t_new, lj_new)),
            accept_prob,
            false,
        )
    } else {
        (false, q_cur.to_vec(), None, accept_prob, false)
    }
}

/// Hoffman & Gelman (2014) Algorithm 4: find a reasonable initial step size by
/// doubling/halving `eps` until a single leapfrog step crosses an acceptance
/// probability of 0.5. Uses one freshly-sampled momentum.
#[allow(clippy::too_many_arguments)]
fn find_reasonable_epsilon<A, R: Rng>(
    rng: &mut R,
    model_fn: &impl Fn() -> Model<A>,
    base: &Trace,
    sites: &[Address],
    q: &[f64],
    lj_q: f64,
    h: f64,
    m_inv: &[f64],
    mass_sqrt: &[f64],
) -> f64 {
    let d = q.len();
    let p0: Vec<f64> = (0..d)
        .map(|i| {
            let z: f64 = rng.sample(StandardNormal);
            z * mass_sqrt[i]
        })
        .collect();
    let k0 = 0.5 * (0..d).map(|i| p0[i] * p0[i] * m_inv[i]).sum::<f64>();
    let h0 = -lj_q + k0;

    let log_ratio_at = |eps: f64| -> f64 {
        let (q1, p1, divergent) = leapfrog(model_fn, base, sites, q, &p0, eps, 1, h, m_inv, None);
        if divergent {
            return f64::NEG_INFINITY;
        }
        let lj1 = log_joint_at(model_fn, base, sites, &q1);
        if !lj1.is_finite() {
            return f64::NEG_INFINITY;
        }
        let k1 = 0.5 * (0..d).map(|i| p1[i] * p1[i] * m_inv[i]).sum::<f64>();
        h0 - (-lj1 + k1)
    };

    let mut eps = 1.0_f64;
    let mut lr = log_ratio_at(eps);
    let ln_half = 0.5_f64.ln();
    let ln2 = 2.0_f64.ln();
    // a = +1 if we should grow eps (ratio too high), -1 if we should shrink it.
    let a = if lr > ln_half { 1.0 } else { -1.0 };
    let mut iters = 0u32;
    // while (ratio)^a > 2^{-a}  <=>  a·log_ratio > -a·ln2
    while a * lr > -a * ln2 {
        eps *= 2.0_f64.powf(a);
        lr = log_ratio_at(eps);
        iters += 1;
        if iters > 100 || !(1e-12..=1e12).contains(&eps) {
            break;
        }
        // Growing but already divergent: cannot get an even bigger acceptable eps.
        if a > 0.0 && lr == f64::NEG_INFINITY {
            eps /= 2.0;
            break;
        }
    }
    eps.clamp(1e-6, 1e3)
}

/// Run a Hamiltonian Monte Carlo chain over the continuous sites of `model_fn`.
///
/// The chain is initialized with a prior draw, warmed up for `n_warmup`
/// iterations (adapting the step size by dual averaging to
/// [`HMCConfig::target_accept`], and optionally the diagonal mass matrix), then
/// the step size and mass are **frozen** and `n_samples` samples are collected
/// from the resulting time-homogeneous kernel.
///
/// Each returned pair is `(model_result, trace)` where the trace has correct,
/// freshly-scored log-weight accumulators (so [`Trace::total_log_weight`] is
/// valid on every returned sample).
///
/// If the model has no continuous sites, this degenerates to independent prior
/// draws (HMC has nothing to move) — compose with [`crate::inference::mh`] for
/// discrete-only models.
///
/// # Example
///
/// ```rust
/// use fugue::*;
/// use fugue::inference::hmc::{hmc_chain, HMCConfig};
/// use rand::rngs::StdRng;
/// use rand::SeedableRng;
///
/// let model_fn = || sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
/// let mut rng = StdRng::seed_from_u64(0);
/// let samples = hmc_chain(&mut rng, model_fn, 100, 100, HMCConfig::default());
/// assert_eq!(samples.len(), 100);
/// ```
pub fn hmc_chain<A: Clone, R: Rng>(
    rng: &mut R,
    model_fn: impl Fn() -> Model<A>,
    n_samples: usize,
    n_warmup: usize,
    config: HMCConfig,
) -> Vec<(A, Trace)> {
    let mut session = HmcSession::new(rng, &model_fn, n_warmup, config);
    for _ in 0..n_warmup {
        session.step(rng, &model_fn);
    }
    let mut out = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        session.step(rng, &model_fn);
        out.push((session.result().clone(), session.trace().clone()));
    }
    out
}

/// Metadata for one [`HmcSession`] transition.
#[derive(Clone, Debug)]
pub struct HmcStepInfo {
    /// Whether the proposal was accepted (the session state advanced).
    pub accepted: bool,
    /// Whether the trajectory diverged (non-finite force or log-joint; the
    /// proposal is always rejected).
    pub divergent: bool,
    /// The Metropolis acceptance probability of the proposal (0.0 on
    /// divergence).
    pub accept_prob: f64,
    /// The step size used for this transition.
    pub step_size: f64,
    /// The leapfrog trajectory (start point included), recorded only by
    /// [`HmcSession::step_recorded`]; empty for [`HmcSession::step`]. On a
    /// divergence the recording stops at the last finite point.
    pub trajectory: Vec<LeapfrogPoint>,
}

/// An incremental HMC chain: [`hmc_chain`] exposed one transition at a time.
///
/// Holds everything [`hmc_chain`] tracks between transitions — current
/// position/trace, dual-averaging step-size state, and the (optionally
/// adapted) diagonal mass matrix — so a caller can advance the chain at its
/// own pace (an animation frame, a streaming API, an interleaved multi-chain
/// driver) instead of running to completion. `hmc_chain` itself is a thin
/// wrapper around this type, so a session stepped `n_warmup + n_samples`
/// times visits exactly the states the batch chain returns (given the same
/// RNG stream).
///
/// The model is passed to every call (models are single-use; drivers hold a
/// `Fn() -> Model<A>`), and **must build the same program each time** — same
/// addresses, same structure — as with every other inference driver in this
/// crate.
///
/// Warmup is handled internally: the first `n_warmup` calls adapt the step
/// size (and optionally mass) exactly as [`hmc_chain`] does, then the kernel
/// freezes. [`HmcSession::is_warming_up`] reports which phase the next step
/// runs in.
///
/// # Example
///
/// ```rust
/// use fugue::*;
/// use fugue::inference::hmc::{HmcSession, HMCConfig};
/// use rand::rngs::StdRng;
/// use rand::SeedableRng;
///
/// let model_fn = || sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
/// let mut rng = StdRng::seed_from_u64(0);
/// let mut session = HmcSession::new(&mut rng, &model_fn, 50, HMCConfig::default());
/// for _ in 0..60 {
///     session.step(&mut rng, &model_fn);
/// }
/// let info = session.step_recorded(&mut rng, &model_fn);
/// assert!(!info.trajectory.is_empty()); // leapfrog path with Hamiltonians
/// assert!(session.trace().get_f64(&addr!("x")).is_some());
/// ```
pub struct HmcSession<A> {
    sites: Vec<Address>,
    q: Vec<f64>,
    cur_a: A,
    cur_trace: Trace,
    lj_cur: f64,
    eps: f64,
    frozen_eps: Option<f64>,
    da: DualAveraging,
    m_inv: Vec<f64>,
    mass_sqrt: Vec<f64>,
    welford: Welford,
    mass_adapt_at: Option<usize>,
    n_warmup: usize,
    iter: usize,
    l: usize,
    h: f64,
    target_accept: f64,
}

impl<A: Clone> HmcSession<A> {
    /// Initialize a chain from a prior draw, finding a reasonable initial step
    /// size unless [`HMCConfig::init_step_size`] pins one — identical setup to
    /// [`hmc_chain`].
    pub fn new<R: Rng>(
        rng: &mut R,
        model_fn: &impl Fn() -> Model<A>,
        n_warmup: usize,
        config: HMCConfig,
    ) -> Self {
        let (cur_a, cur_trace) = run(
            PriorHandler {
                rng,
                trace: Trace::default(),
            },
            model_fn(),
        );
        let (sites, q) = positions_from_trace(&cur_trace);
        let d = sites.len();

        let h = config.finite_diff_eps;
        let l = config.n_leapfrog.max(1);
        let m_inv = vec![1.0; d];
        let mass_sqrt = vec![1.0; d];
        let lj_cur = cur_trace.total_log_weight();

        // With no continuous sites every step degenerates to a fresh prior
        // draw; skip the epsilon search (it would divide by d = 0 nowhere,
        // but there is nothing to tune).
        let eps0 = if d == 0 {
            1.0
        } else {
            match config.init_step_size {
                Some(e) => e,
                None => find_reasonable_epsilon(
                    rng, model_fn, &cur_trace, &sites, &q, lj_cur, h, &m_inv, &mass_sqrt,
                ),
            }
        };
        let da = DualAveraging::new(eps0, config.target_accept);
        let welford = Welford::new(d);
        let mass_adapt_at = if config.adapt_mass && n_warmup >= 4 {
            Some(n_warmup / 2)
        } else {
            None
        };

        HmcSession {
            sites,
            q,
            cur_a,
            cur_trace,
            lj_cur,
            eps: eps0,
            frozen_eps: None,
            da,
            m_inv,
            mass_sqrt,
            welford,
            mass_adapt_at,
            n_warmup,
            iter: 0,
            l,
            h,
            target_accept: config.target_accept,
        }
    }

    /// The ordered continuous site addresses the chain moves (the coordinate
    /// order of [`HmcSession::position`] and [`LeapfrogPoint::q`]).
    pub fn sites(&self) -> &[Address] {
        &self.sites
    }

    /// Pin the step size for all subsequent transitions, overriding both
    /// warmup adaptation and the frozen dual-averaging value. Meant for
    /// interactive use (a step-size slider); the kernel stays exact for any
    /// positive `eps`.
    pub fn set_step_size(&mut self, eps: f64) {
        let eps = eps.max(1e-12);
        self.eps = eps;
        self.frozen_eps = Some(eps);
        // Leave warmup: a manually pinned step size must not be re-adapted.
        self.n_warmup = self.n_warmup.min(self.iter);
    }

    /// Change the number of leapfrog steps per proposal for subsequent
    /// transitions.
    pub fn set_n_leapfrog(&mut self, l: usize) {
        self.l = l.max(1);
    }

    /// The current position over the continuous sites.
    pub fn position(&self) -> &[f64] {
        &self.q
    }

    /// The model result at the current state.
    pub fn result(&self) -> &A {
        &self.cur_a
    }

    /// The current state's trace (freshly scored; `total_log_weight` is valid).
    pub fn trace(&self) -> &Trace {
        &self.cur_trace
    }

    /// The step size the next transition will use.
    pub fn step_size(&self) -> f64 {
        if self.iter < self.n_warmup {
            self.eps
        } else {
            self.frozen_or_current()
        }
    }

    /// Whether the next [`step`](HmcSession::step) still adapts (warmup phase).
    pub fn is_warming_up(&self) -> bool {
        self.iter < self.n_warmup
    }

    /// Number of transitions taken so far (warmup included).
    pub fn iterations(&self) -> usize {
        self.iter
    }

    fn frozen_or_current(&self) -> f64 {
        match self.frozen_eps {
            Some(e) => e,
            // Freeze lazily on first read past warmup: the dual-averaging
            // running mean, or the raw eps when there was no warmup —
            // matching hmc_chain's `final_eps`.
            None if self.n_warmup > 0 => self.da.frozen_step(),
            None => self.eps,
        }
    }

    /// Advance the chain by one transition. Equivalent to
    /// [`step_recorded`](HmcSession::step_recorded) without paying for
    /// trajectory recording.
    pub fn step<R: Rng>(&mut self, rng: &mut R, model_fn: &impl Fn() -> Model<A>) -> HmcStepInfo {
        self.advance(rng, model_fn, false)
    }

    /// Advance the chain by one transition, recording the leapfrog trajectory
    /// and the Hamiltonian at each integration point (one extra model
    /// execution per point — meant for visualization and diagnostics, not
    /// hot loops).
    pub fn step_recorded<R: Rng>(
        &mut self,
        rng: &mut R,
        model_fn: &impl Fn() -> Model<A>,
    ) -> HmcStepInfo {
        self.advance(rng, model_fn, true)
    }

    fn advance<R: Rng>(
        &mut self,
        rng: &mut R,
        model_fn: &impl Fn() -> Model<A>,
        record: bool,
    ) -> HmcStepInfo {
        // No continuous sites: independent prior draws, as in hmc_chain.
        if self.sites.is_empty() {
            let (a, t) = run(
                PriorHandler {
                    rng,
                    trace: Trace::default(),
                },
                model_fn(),
            );
            self.cur_a = a;
            self.lj_cur = t.total_log_weight();
            self.cur_trace = t;
            self.iter += 1;
            return HmcStepInfo {
                accepted: true,
                divergent: false,
                accept_prob: 1.0,
                step_size: self.eps,
                trajectory: Vec::new(),
            };
        }

        let warming = self.iter < self.n_warmup;
        let eps = if warming {
            self.eps
        } else {
            let e = self.frozen_or_current();
            self.frozen_eps = Some(e);
            e
        };

        let mut trajectory = Vec::new();
        let recorder = if record { Some(&mut trajectory) } else { None };
        let (accepted, q_new, endpoint, alpha, divergent) = hmc_transition(
            rng,
            model_fn,
            &self.cur_trace,
            &self.sites,
            &self.q,
            self.lj_cur,
            eps,
            self.l,
            self.h,
            &self.m_inv,
            &self.mass_sqrt,
            recorder,
        );
        if accepted {
            let (a, t, lj) = endpoint.unwrap();
            self.cur_a = a;
            self.cur_trace = t;
            self.q = q_new;
            self.lj_cur = lj;
        }

        if warming {
            self.eps = self.da.update(alpha);
            if self.mass_adapt_at.is_some() {
                self.welford.push(&self.q);
            }
            if Some(self.iter + 1) == self.mass_adapt_at {
                // Set M^{-1} to the estimated marginal variances and re-tune
                // eps for the remaining warmup. Any positive diagonal mass
                // keeps the kernel exact (module docs), so this only affects
                // efficiency.
                let vars = self.welford.variances();
                for (i, &v) in vars.iter().enumerate() {
                    self.m_inv[i] = v;
                    self.mass_sqrt[i] = (1.0 / v).sqrt();
                }
                let eps_reset = find_reasonable_epsilon(
                    rng,
                    model_fn,
                    &self.cur_trace,
                    &self.sites,
                    &self.q,
                    self.lj_cur,
                    self.h,
                    &self.m_inv,
                    &self.mass_sqrt,
                );
                self.da = DualAveraging::new(eps_reset, self.target_accept);
                self.eps = eps_reset;
            }
        }
        self.iter += 1;

        HmcStepInfo {
            accepted,
            divergent,
            accept_prob: alpha,
            step_size: eps,
            trajectory,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::addr;
    use crate::core::distribution::{Distribution, Normal};
    use crate::core::model::{observe, sample, ModelExt};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    // FG-31: dual averaging must move the step size toward the value that hits
    // the target acceptance. Feeding acceptances BELOW target should shrink eps;
    // acceptances ABOVE target should grow it.
    #[test]
    fn fg31_dual_averaging_moves_step_size_toward_target() {
        // Persistently too-low acceptance -> eps decreases.
        let mut da = DualAveraging::new(1.0, 0.8);
        for _ in 0..200 {
            let _ = da.update(0.1);
        }
        assert!(
            da.frozen_step() < 1.0,
            "low acceptance should shrink eps, got {}",
            da.frozen_step()
        );

        // Persistently too-high acceptance -> eps increases.
        let mut da = DualAveraging::new(1.0, 0.8);
        for _ in 0..200 {
            let _ = da.update(1.0);
        }
        assert!(
            da.frozen_step() > 1.0,
            "high acceptance should grow eps, got {}",
            da.frozen_step()
        );
    }

    // FG-31: sanity that a single unit-variance normal site is recovered.
    #[test]
    fn fg31_hmc_standard_normal_marginal() {
        let model_fn = || sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
        let mut rng = StdRng::seed_from_u64(7);
        let samples = hmc_chain(&mut rng, model_fn, 2000, 500, HMCConfig::default());
        let xs: Vec<f64> = samples.iter().map(|(x, _)| *x).collect();
        let mean = xs.iter().sum::<f64>() / xs.len() as f64;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / xs.len() as f64;
        assert!(mean.abs() < 0.1, "mean {}", mean);
        assert!((var - 1.0).abs() < 0.15, "var {}", var);
    }

    // FG-31: every returned trace must carry correct (freshly scored) log-weight
    // accumulators (guards against returning stale traces).
    #[test]
    fn fg31_returned_traces_have_fresh_weights() {
        let model_fn = || {
            sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap())
                .bind(|mu| observe(addr!("y"), Normal::new(mu, 1.0).unwrap(), 1.0).map(move |_| mu))
        };
        let mut rng = StdRng::seed_from_u64(11);
        let samples = hmc_chain(&mut rng, model_fn, 50, 50, HMCConfig::default());
        for (mu, t) in &samples {
            // Recompute the expected log joint from the returned mu and check it
            // matches the trace's own accumulators.
            let expected = Normal::new(0.0, 1.0).unwrap().log_prob(mu)
                + Normal::new(*mu, 1.0).unwrap().log_prob(&1.0);
            assert!((t.total_log_weight() - expected).abs() < 1e-9);
        }
    }

    // FG-31: diagonal mass-matrix adaptation path. On a strongly axis-scaled
    // independent target (x ~ N(0,1), y ~ N(0,10)) the adapted diagonal mass
    // rescales each coordinate; the chain must still recover both marginals. This
    // exercises the Welford accumulation + mass reset that identity-mass tests
    // never touch.
    #[test]
    fn fg31_hmc_diagonal_mass_adaptation_axis_scaled() {
        let model_fn = || {
            sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
                .bind(|x| sample(addr!("y"), Normal::new(0.0, 10.0).unwrap()).map(move |y| (x, y)))
        };
        let cfg = HMCConfig {
            adapt_mass: true,
            ..HMCConfig::default()
        };
        let mut rng = StdRng::seed_from_u64(2024);
        let samples = hmc_chain(&mut rng, model_fn, 3000, 1500, cfg);
        let xs: Vec<f64> = samples.iter().map(|((x, _), _)| *x).collect();
        let ys: Vec<f64> = samples.iter().map(|((_, y), _)| *y).collect();
        let mx = xs.iter().sum::<f64>() / xs.len() as f64;
        let my = ys.iter().sum::<f64>() / ys.len() as f64;
        let vx = xs.iter().map(|x| (x - mx).powi(2)).sum::<f64>() / xs.len() as f64;
        let vy = ys.iter().map(|y| (y - my).powi(2)).sum::<f64>() / ys.len() as f64;
        // Loose Monte-Carlo tolerances: the point is that BOTH scales are
        // recovered despite the 10x axis-scale difference.
        assert!(mx.abs() < 0.2, "x mean {mx}");
        assert!(my.abs() < 2.0, "y mean {my}");
        assert!((vx - 1.0).abs() < 0.25, "var(x) {vx}");
        assert!((vy - 100.0).abs() < 25.0, "var(y) {vy}");
    }

    // The incremental session must visit exactly the states the batch chain
    // returns: same seed, same model, same config => identical draws. This
    // pins hmc_chain as a true thin wrapper over HmcSession.
    #[test]
    fn session_matches_batch_chain_exactly() {
        let model_fn = || {
            sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap())
                .bind(|mu| observe(addr!("y"), Normal::new(mu, 1.0).unwrap(), 2.0).map(move |_| mu))
        };
        let cfg = HMCConfig::default();
        let (n_warmup, n_samples) = (50, 40);

        let mut rng1 = StdRng::seed_from_u64(7);
        let batch = hmc_chain(&mut rng1, model_fn, n_samples, n_warmup, cfg);

        let mut rng2 = StdRng::seed_from_u64(7);
        let mut session = HmcSession::new(&mut rng2, &model_fn, n_warmup, cfg);
        for _ in 0..n_warmup {
            assert!(session.is_warming_up());
            session.step(&mut rng2, &model_fn);
        }
        for (batch_mu, batch_trace) in &batch {
            assert!(!session.is_warming_up());
            session.step(&mut rng2, &model_fn);
            assert_eq!(*session.result(), *batch_mu);
            assert_eq!(
                session.trace().total_log_weight(),
                batch_trace.total_log_weight()
            );
        }
    }

    // step_recorded returns the full leapfrog path (L+1 points, start
    // included) with finite Hamiltonians, and recording must not perturb the
    // chain: the extra Hamiltonian evaluations are deterministic re-scores
    // that consume no randomness.
    #[test]
    fn recorded_trajectory_shape_and_rng_neutrality() {
        let model_fn = || {
            sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap())
                .bind(|mu| observe(addr!("y"), Normal::new(mu, 1.0).unwrap(), 0.5).map(move |_| mu))
        };
        let cfg = HMCConfig {
            n_leapfrog: 12,
            init_step_size: Some(0.2),
            ..HMCConfig::default()
        };

        let mut rng1 = StdRng::seed_from_u64(3);
        let mut recorded = HmcSession::new(&mut rng1, &model_fn, 0, cfg);
        let mut rng2 = StdRng::seed_from_u64(3);
        let mut plain = HmcSession::new(&mut rng2, &model_fn, 0, cfg);

        for _ in 0..10 {
            let info = recorded.step_recorded(&mut rng1, &model_fn);
            plain.step(&mut rng2, &model_fn);
            if !info.divergent {
                assert_eq!(info.trajectory.len(), cfg.n_leapfrog + 1);
            }
            for pt in &info.trajectory {
                assert_eq!(pt.q.len(), 1);
                assert!(pt.h.is_finite());
            }
            assert_eq!(recorded.position(), plain.position());
        }
    }
}
