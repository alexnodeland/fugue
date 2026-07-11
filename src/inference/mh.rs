//! Metropolis-Hastings MCMC with adaptive tuning and single-site updates.
//!
//! This module implements the Metropolis-Hastings algorithm, a fundamental MCMC method
//! for sampling from posterior distributions. The implementation features:
//!
//! - **Adaptive scaling**: Automatically tunes proposal step sizes to achieve target acceptance rates
//! - **Single-site updates**: Updates one random variable at a time for better mixing
//! - **Type-safe proposals**: Preserves original types (bool, u64, usize, etc.) during proposals
//! - **Type-aware proposals**: Uses ProposalStrategy traits based on value types
//! - **Correct Hastings corrections**: asymmetric proposals contribute their
//!   `q(x|x') − q(x'|x)` term to the acceptance ratio (FG-02, FG-10)
//!
//! ## Proposal selection (FG-42)
//!
//! Proposal kinds are chosen from the *distribution's actual support*, not from
//! substrings of the address name (the old `sigma`/`scale`/`p`/`beta` heuristics
//! could, e.g., trap an unbounded parameter named `slope` in `[0,1]` and break
//! ergodicity). The rules are:
//!
//! - **`f64`**: default to a symmetric **Gaussian** random walk. Out-of-support
//!   proposals simply receive a `−inf` joint density and are rejected, so
//!   ergodicity is preserved. A site is routed to a **log-space** walk (with the
//!   exact Jacobian/Hastings correction) only when its current value is positive
//!   *and* the site's prior density at a negative probe value is `−inf` — i.e.
//!   the support is genuinely positive. A **reflected** `[a,b]` walk is used only
//!   when explicitly requested per address.
//! - **`usize` (categorical)**: propose by resampling from the site's **prior**
//!   distribution. With `q = prior` the Hastings terms cancel the prior in the
//!   target, so acceptance reduces to the likelihood ratio, and the proposal can
//!   never miss the support (FG-10).
//! - **`u64` (counts)**: a symmetric reflected discrete walk (FG-41).
//! - **`bool`**: a deterministic flip (symmetric).
//!
//! Callers can override the `f64` proposal for any address via
//! [`adaptive_mcmc_chain_with_overrides`] using [`SiteProposal`].
//!
//! ## Algorithm Overview
//!
//! The Metropolis-Hastings algorithm generates correlated samples from the posterior by:
//! 1. Proposing a new state by modifying the current state
//! 2. Computing the acceptance probability using the ratio of posterior densities
//!    plus the proposal (Hastings) correction
//! 3. Accepting or rejecting the proposal based on this probability
//! 4. Repeating to generate a Markov chain that converges to the posterior
//!
//! ## Cost model (FG-11)
//!
//! Lightweight trace-based single-site MCMC is inherently **O(model-size)** per
//! transition: scoring a proposal requires re-executing the whole model to
//! recompute the log-density contributions that depend on the touched site. This
//! implementation removes the *redundant* work (it re-executes the model exactly
//! once per step — see [`adaptive_mcmc_chain`] — caches the current state's
//! score and the site list across iterations, and avoids the extra trace clones),
//! but the per-transition cost still scales with the number of sites. Models with
//! very many latent variables should prefer a gradient-based kernel.
//!
//! ## Adaptive Tuning
//!
//! Good MCMC performance requires well-tuned proposal distributions. This implementation
//! automatically adapts proposal scales during warmup to achieve approximately 44%
//! acceptance rate (optimal for random-walk Metropolis on continuous distributions),
//! then **freezes** the scales for the sampling phase so the recorded draws come from a
//! single fixed transition kernel (FG-57).
//!
//! # Examples
//!
//! ```rust
//! use fugue::*;
//! use rand::rngs::StdRng;
//! use rand::SeedableRng;
//!
//! // Define a simple Bayesian model
//! let model_fn = || {
//!     sample(addr!("mu"), Normal::new(0.0, 2.0).unwrap())
//!         .bind(|mu| observe(addr!("y"), Normal::new(mu, 1.0).unwrap(), 2.5))
//! };
//!
//! // Run adaptive MCMC (small numbers for testing)
//! let mut rng = StdRng::seed_from_u64(42);
//! let samples = adaptive_mcmc_chain(
//!     &mut rng,
//!     model_fn,
//!     50,  // Number of samples (small for test)
//!     10,  // Burn-in period
//! );
//!
//! // Extract parameter estimates
//! let mu_samples: Vec<f64> = samples.iter()
//!     .filter_map(|(_, trace)| trace.choices.get(&addr!("mu")))
//!     .filter_map(|choice| match choice.value {
//!         ChoiceValue::F64(mu) => Some(mu),
//!         _ => None,
//!     })
//!     .collect();
//!
//! assert!(!mu_samples.is_empty());
//! ```
use crate::core::address::Address;
use crate::core::distribution::Distribution;
use crate::core::model::Model;
use crate::inference::mcmc_utils::DiminishingAdaptation;
use crate::runtime::handler::{run, Handler};
use crate::runtime::interpreters::{PriorHandler, ScoreGivenTrace};
use crate::runtime::trace::{Choice, ChoiceValue, Trace};
use rand::{Rng, RngCore};
use std::collections::HashMap;

/// Negative probe value used to detect positive-support `f64` sites (FG-42).
/// A site whose prior density is `−inf` here (and whose current value is
/// positive) is treated as positively constrained and given a log-space walk.
const NEG_SUPPORT_PROBE: f64 = -1.0;

/// Standard-normal draw via Box-Muller (shared by the random-walk proposals).
fn gaussian_z(rng: &mut dyn RngCore) -> f64 {
    let u1: f64 = rng.gen::<f64>().max(1e-10); // avoid ln(0)
    let u2: f64 = rng.gen();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Log-density of `Normal(mean, sd)` at `x` (used for the log-space Jacobian).
fn normal_logpdf(x: f64, mean: f64, sd: f64) -> f64 {
    let z = (x - mean) / sd;
    -0.5 * z * z - sd.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
}

/// User-facing per-address proposal override for `f64` sites (FG-42).
///
/// The samplers pick a sensible proposal automatically from each site's support,
/// but callers can force a specific kind via
/// [`adaptive_mcmc_chain_with_overrides`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SiteProposal {
    /// Symmetric Gaussian random walk (default for unconstrained `f64`).
    Gaussian,
    /// Log-space random walk with the exact Jacobian/Hastings correction, for
    /// positive-support parameters (scales, rates, …). FG-02.
    LogSpace,
    /// Reflected random walk confined to `[lower, upper]` (symmetric).
    Reflect {
        /// Inclusive lower bound.
        lower: f64,
        /// Inclusive upper bound.
        upper: f64,
    },
    /// Independence proposal that resamples the site from its prior. FG-10.
    PriorResample,
}

/// Trait for distribution-aware proposal strategies.
///
/// This enables more intelligent proposals that take advantage of the distribution
/// structure rather than using generic random walks.
pub trait ProposalStrategy<T> {
    /// Generate a proposal given the current value and scale.
    fn propose(&self, current: T, scale: f64, rng: &mut dyn RngCore) -> T;

    /// Log-density `log q(to | from)` of proposing `to` from `from` at the given
    /// `scale`. Defaults to `0` for symmetric proposals (the constant cancels in
    /// the Hastings ratio); asymmetric proposals override it.
    fn log_proposal_prob(&self, from: T, to: T, scale: f64) -> f64 {
        let _ = (from, to, scale);
        0.0 // Default: symmetric proposal
    }
}

/// Gaussian random walk proposal for continuous distributions (symmetric).
pub struct GaussianWalkProposal;

impl ProposalStrategy<f64> for GaussianWalkProposal {
    fn propose(&self, current: f64, scale: f64, rng: &mut dyn RngCore) -> f64 {
        current + scale * gaussian_z(rng)
    }
}

/// Log-space random walk proposal for positive-constrained continuous parameters.
///
/// Proposes `x' = exp(ln x + scale·z)`, which keeps `x'` strictly positive. This
/// map is **asymmetric** in the original space: the induced density is
/// `q(x'|x) = N(ln x'; ln x, scale²) / x'`. Its [`log_proposal_prob`] returns
/// exactly that log-density, so the acceptance ratio picks up the Jacobian term
/// `+(ln x' − ln x)` (FG-02). Omitting it makes the chain target `π(x)/x` instead
/// of `π(x)`.
///
/// [`log_proposal_prob`]: ProposalStrategy::log_proposal_prob
pub struct LogSpaceWalkProposal;

impl ProposalStrategy<f64> for LogSpaceWalkProposal {
    fn propose(&self, current: f64, scale: f64, rng: &mut dyn RngCore) -> f64 {
        if current <= 0.0 {
            // Out of the proposal's domain; nudge to the smallest positive value.
            return f64::MIN_POSITIVE;
        }
        let z = gaussian_z(rng);
        let proposed = (current.ln() + scale * z).exp();
        if proposed.is_finite() {
            proposed.max(f64::MIN_POSITIVE)
        } else {
            // Extreme tail; a huge finite value will score to −inf and reject.
            f64::MAX
        }
    }

    fn log_proposal_prob(&self, from: f64, to: f64, scale: f64) -> f64 {
        if from <= 0.0 || to <= 0.0 {
            return 0.0;
        }
        // q(to|from) = N(ln to; ln from, scale) · |d ln to / d to| = N(...) / to.
        normal_logpdf(to.ln(), from.ln(), scale) - to.ln()
    }
}

/// Reflection-based proposal for bounded continuous distributions (symmetric).
///
/// Reflects a Gaussian step off the boundaries to stay within `[lower, upper]`.
/// Reflection preserves symmetry, so no Hastings correction is needed.
pub struct ReflectionWalkProposal {
    /// Lower bound (inclusive)
    pub lower_bound: f64,
    /// Upper bound (inclusive)
    pub upper_bound: f64,
}

impl ProposalStrategy<f64> for ReflectionWalkProposal {
    fn propose(&self, current: f64, scale: f64, rng: &mut dyn RngCore) -> f64 {
        let mut proposed = current + scale * gaussian_z(rng);

        let range = self.upper_bound - self.lower_bound;
        if range <= 0.0 {
            return current; // Invalid bounds, return current
        }

        // Reflect off boundaries until within bounds.
        while proposed < self.lower_bound || proposed > self.upper_bound {
            if proposed < self.lower_bound {
                proposed = 2.0 * self.lower_bound - proposed;
            }
            if proposed > self.upper_bound {
                proposed = 2.0 * self.upper_bound - proposed;
            }
        }

        proposed.clamp(self.lower_bound, self.upper_bound)
    }
}

/// Flip proposal for boolean distributions (symmetric).
pub struct FlipProposal;

impl ProposalStrategy<bool> for FlipProposal {
    fn propose(&self, current: bool, _scale: f64, _rng: &mut dyn RngCore) -> bool {
        // Deterministic flip: q(!x|x) = q(x|!x) = 1, so the proposal is symmetric
        // and mixes maximally for a single binary site.
        !current
    }
}

/// Discrete random walk proposal for non-negative count distributions.
///
/// Draws `delta = round(scale·z)` from a symmetric integer distribution and
/// reflects at the boundary about `−1/2` (`k → −k−1` when `x + delta < 0`).
///
/// FG-41: plain `|x + delta|` (reflection about `0`) is **not** symmetric at the
/// boundary — `0` is a fixed point of negation, so it has no reflection partner
/// and moves involving state `0` are mis-weighted by a factor of 2
/// (`q(y|0) = 2·q(0|y)`). Reflecting about `−1/2` instead makes the map a clean
/// two-to-one folding with no fixed point, giving an exactly symmetric kernel
/// (`q(a|b) = q(b|a)` everywhere, including at `0`), so no Hastings correction is
/// needed.
pub struct DiscreteWalkProposal;

impl ProposalStrategy<u64> for DiscreteWalkProposal {
    fn propose(&self, current: u64, scale: f64, rng: &mut dyn RngCore) -> u64 {
        let delta = (scale * gaussian_z(rng)).round() as i64;
        let k = current as i64 + delta;
        if k >= 0 {
            k as u64
        } else {
            (-k - 1) as u64 // reflect about −1/2 (symmetric)
        }
    }
}

/// Handler that performs one single-site proposal *inside* a single model run.
///
/// All sites except `target` are replayed from `base` and re-scored under the
/// current model (their densities may change when `target` changes, e.g. in a
/// hierarchical model). The `target` site is proposed according to its value
/// type and support, freshly scored, and its forward/reverse proposal
/// log-densities are written to `log_q_forward` / `log_q_reverse` for the
/// acceptance ratio. Producing a fully, freshly-scored proposal trace in one run
/// is what lets the driver return correct accumulators (FG-40) and avoid the
/// extra current-scoring run (FG-11/FG-12).
///
/// If `target`'s new value opens a branch that requires an address absent from
/// `base`, that address is sampled fresh from its prior (rather than panicking as
/// raw `ScoreGivenTrace` would). Trans-dimensional acceptance corrections
/// (RJMCMC) are out of scope here; for fixed-structure models this path is never
/// taken.
struct SingleSiteProposalHandler<'a, R: RngCore> {
    rng: &'a mut R,
    base: &'a Trace,
    target: &'a Address,
    scale: f64,
    overrides: &'a HashMap<Address, SiteProposal>,
    kind_cache: &'a mut HashMap<Address, SiteProposal>,
    log_q_forward: &'a mut f64,
    log_q_reverse: &'a mut f64,
    trace: Trace,
}

impl<'a, R: RngCore> SingleSiteProposalHandler<'a, R> {
    /// Decide the `f64` proposal kind for the target site (FG-42), caching the
    /// probe result so support detection happens at most once per address.
    fn f64_kind(
        &mut self,
        addr: &Address,
        current: f64,
        dist: &dyn Distribution<f64>,
    ) -> SiteProposal {
        if let Some(&k) = self.overrides.get(addr) {
            return k;
        }
        if let Some(&k) = self.kind_cache.get(addr) {
            return k;
        }
        let kind = if current > 0.0 && !dist.log_prob(&NEG_SUPPORT_PROBE).is_finite() {
            SiteProposal::LogSpace
        } else {
            SiteProposal::Gaussian
        };
        self.kind_cache.insert(addr.clone(), kind);
        kind
    }
}

impl<'a, R: RngCore> Handler for SingleSiteProposalHandler<'a, R> {
    fn on_sample_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>) -> f64 {
        if addr == self.target {
            let current = self
                .base
                .get_f64(addr)
                .unwrap_or_else(|| dist.sample(self.rng));
            let kind = self.f64_kind(addr, current, dist);
            let (proposed, lqf, lqr) = match kind {
                SiteProposal::Gaussian => {
                    let s = GaussianWalkProposal;
                    let p = s.propose(current, self.scale, self.rng);
                    (
                        p,
                        s.log_proposal_prob(current, p, self.scale),
                        s.log_proposal_prob(p, current, self.scale),
                    )
                }
                SiteProposal::LogSpace => {
                    let s = LogSpaceWalkProposal;
                    let p = s.propose(current, self.scale, self.rng);
                    (
                        p,
                        s.log_proposal_prob(current, p, self.scale),
                        s.log_proposal_prob(p, current, self.scale),
                    )
                }
                SiteProposal::Reflect { lower, upper } => {
                    let s = ReflectionWalkProposal {
                        lower_bound: lower,
                        upper_bound: upper,
                    };
                    let p = s.propose(current, self.scale, self.rng);
                    (
                        p,
                        s.log_proposal_prob(current, p, self.scale),
                        s.log_proposal_prob(p, current, self.scale),
                    )
                }
                SiteProposal::PriorResample => {
                    let p = dist.sample(self.rng);
                    (p, dist.log_prob(&p), dist.log_prob(&current))
                }
            };
            *self.log_q_forward = lqf;
            *self.log_q_reverse = lqr;
            let lp = dist.log_prob(&proposed);
            self.trace.log_prior += lp;
            self.trace.choices.insert(
                addr.clone(),
                Choice {
                    addr: addr.clone(),
                    value: ChoiceValue::F64(proposed),
                    logp: lp,
                },
            );
            proposed
        } else {
            let x = self
                .base
                .get_f64(addr)
                .unwrap_or_else(|| dist.sample(self.rng));
            let lp = dist.log_prob(&x);
            self.trace.log_prior += lp;
            self.trace.choices.insert(
                addr.clone(),
                Choice {
                    addr: addr.clone(),
                    value: ChoiceValue::F64(x),
                    logp: lp,
                },
            );
            x
        }
    }

    fn on_sample_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>) -> bool {
        let x = if addr == self.target {
            let current = self
                .base
                .get_bool(addr)
                .unwrap_or_else(|| dist.sample(self.rng));
            // Symmetric deterministic flip.
            *self.log_q_forward = 0.0;
            *self.log_q_reverse = 0.0;
            FlipProposal.propose(current, self.scale, self.rng)
        } else {
            self.base
                .get_bool(addr)
                .unwrap_or_else(|| dist.sample(self.rng))
        };
        let lp = dist.log_prob(&x);
        self.trace.log_prior += lp;
        self.trace.choices.insert(
            addr.clone(),
            Choice {
                addr: addr.clone(),
                value: ChoiceValue::Bool(x),
                logp: lp,
            },
        );
        x
    }

    fn on_sample_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>) -> u64 {
        let x = if addr == self.target {
            let current = self
                .base
                .get_u64(addr)
                .unwrap_or_else(|| dist.sample(self.rng));
            // Symmetric reflected discrete walk (FG-41).
            *self.log_q_forward = 0.0;
            *self.log_q_reverse = 0.0;
            DiscreteWalkProposal.propose(current, self.scale, self.rng)
        } else {
            self.base
                .get_u64(addr)
                .unwrap_or_else(|| dist.sample(self.rng))
        };
        let lp = dist.log_prob(&x);
        self.trace.log_prior += lp;
        self.trace.choices.insert(
            addr.clone(),
            Choice {
                addr: addr.clone(),
                value: ChoiceValue::U64(x),
                logp: lp,
            },
        );
        x
    }

    fn on_sample_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>) -> usize {
        let x = if addr == self.target {
            let current = self
                .base
                .get_usize(addr)
                .unwrap_or_else(|| dist.sample(self.rng));
            // FG-10: resample from the site's prior. With q = prior the Hastings
            // terms cancel the prior in the target, so acceptance reduces to the
            // likelihood ratio and no category can ever be missed.
            let proposed = dist.sample(self.rng);
            *self.log_q_forward = dist.log_prob(&proposed);
            *self.log_q_reverse = dist.log_prob(&current);
            proposed
        } else {
            self.base
                .get_usize(addr)
                .unwrap_or_else(|| dist.sample(self.rng))
        };
        let lp = dist.log_prob(&x);
        self.trace.log_prior += lp;
        self.trace.choices.insert(
            addr.clone(),
            Choice {
                addr: addr.clone(),
                value: ChoiceValue::Usize(x),
                logp: lp,
            },
        );
        x
    }

    fn on_sample_i64(&mut self, addr: &Address, dist: &dyn Distribution<i64>) -> i64 {
        let x = if addr == self.target {
            let current = self
                .base
                .get_i64(addr)
                .unwrap_or_else(|| dist.sample(self.rng));
            // Symmetric integer random walk (no boundary to reflect at).
            *self.log_q_forward = 0.0;
            *self.log_q_reverse = 0.0;
            let delta = (self.scale * gaussian_z(self.rng)).round() as i64;
            current + delta
        } else {
            self.base
                .get_i64(addr)
                .unwrap_or_else(|| dist.sample(self.rng))
        };
        let lp = dist.log_prob(&x);
        self.trace.log_prior += lp;
        self.trace.choices.insert(
            addr.clone(),
            Choice {
                addr: addr.clone(),
                value: ChoiceValue::I64(x),
                logp: lp,
            },
        );
        x
    }

    fn on_observe_f64(&mut self, _addr: &Address, dist: &dyn Distribution<f64>, value: f64) {
        self.trace.log_likelihood += dist.log_prob(&value);
    }
    fn on_observe_bool(&mut self, _addr: &Address, dist: &dyn Distribution<bool>, value: bool) {
        self.trace.log_likelihood += dist.log_prob(&value);
    }
    fn on_observe_u64(&mut self, _addr: &Address, dist: &dyn Distribution<u64>, value: u64) {
        self.trace.log_likelihood += dist.log_prob(&value);
    }
    fn on_observe_usize(&mut self, _addr: &Address, dist: &dyn Distribution<usize>, value: usize) {
        self.trace.log_likelihood += dist.log_prob(&value);
    }
    fn on_observe_i64(&mut self, _addr: &Address, dist: &dyn Distribution<i64>, value: i64) {
        self.trace.log_likelihood += dist.log_prob(&value);
    }

    fn on_factor(&mut self, logw: f64) {
        self.trace.log_factors += logw;
    }

    fn finish(self) -> Trace {
        self.trace
    }
}

/// Propose a new value at `target` and fully score the resulting trace in one
/// model run. Returns `(model_result, proposed_trace, proposed_log_weight,
/// log_q_forward, log_q_reverse)`.
fn propose_and_score<A, F, R>(
    rng: &mut R,
    model_fn: &F,
    current: &Trace,
    target: &Address,
    scale: f64,
    overrides: &HashMap<Address, SiteProposal>,
    kind_cache: &mut HashMap<Address, SiteProposal>,
) -> (A, Trace, f64, f64, f64)
where
    F: Fn() -> Model<A>,
    R: Rng,
{
    let mut lqf = 0.0;
    let mut lqr = 0.0;
    let (a, trace) = run(
        SingleSiteProposalHandler {
            rng,
            base: current,
            target,
            scale,
            overrides,
            kind_cache,
            log_q_forward: &mut lqf,
            log_q_reverse: &mut lqr,
            trace: Trace::default(),
        },
        model_fn(),
    );
    let lw = trace.total_log_weight();
    (a, trace, lw, lqf, lqr)
}

/// One cached single-site MH transition used by the chain driver.
///
/// The current state's log-weight (`current_lw`) and the ordered `sites` list are
/// supplied by the caller and cached across iterations, so this performs exactly
/// one model run (the proposal). Returns `Some((result, trace, log_weight))` on
/// acceptance (a freshly-scored trace, FG-40) and `None` on rejection — the
/// caller keeps its cached current state, so no extra model run happens on
/// rejection (FG-12).
#[allow(clippy::too_many_arguments)]
fn single_site_mh_step<A, F, R>(
    rng: &mut R,
    model_fn: &F,
    current: &Trace,
    current_lw: f64,
    sites: &[Address],
    adaptation: &mut DiminishingAdaptation,
    overrides: &HashMap<Address, SiteProposal>,
    kind_cache: &mut HashMap<Address, SiteProposal>,
    adapt: bool,
) -> Option<(A, Trace, f64)>
where
    F: Fn() -> Model<A>,
    R: Rng,
{
    if sites.is_empty() {
        return None;
    }
    let target = sites[rng.gen_range(0..sites.len())].clone();
    let scale = adaptation.get_scale(&target);

    let (a_prop, prop_trace, prop_lw, lqf, lqr) = propose_and_score(
        rng, model_fn, current, &target, scale, overrides, kind_cache,
    );

    // log α = Δlog-joint + log q(x|x') − log q(x'|x).
    let log_alpha = prop_lw - current_lw + (lqr - lqf);
    let accept = log_alpha >= 0.0 || rng.gen::<f64>() < log_alpha.exp();

    if adapt {
        adaptation.update(&target, accept);
    }

    if accept {
        Some((a_prop, prop_trace, prop_lw))
    } else {
        None
    }
}

/// Perform a single adaptive Metropolis-Hastings update step.
///
/// This function implements a single iteration of the MH algorithm with proper
/// diminishing adaptation that preserves ergodicity. It randomly selects one site
/// to update, proposes a new value using adaptive scaling, and accepts or rejects
/// based on the Metropolis-Hastings criterion (including the proposal/Hastings
/// correction for asymmetric proposals).
///
/// # Algorithm
///
/// 1. Score the current state once (reused on rejection — no redundant third
///    model run, FG-12).
/// 2. Randomly select a site and propose a new value using diminishing adaptive
///    scaling, scoring the proposal in the same run (FG-11).
/// 3. Accept with probability `min(1, exp(log α))` where
///    `log α = Δlog-joint + q(x|x') − q(x'|x)`.
/// 4. Update adaptive scales using diminishing step sizes.
///
/// On acceptance the returned trace is freshly scored, so its
/// `total_log_weight()` is correct (FG-40).
///
/// # Arguments
///
/// * `rng` - Random number generator
/// * `model_fn` - Function that creates the model
/// * `current` - Current trace (state of the Markov chain)
/// * `adaptation` - Diminishing adaptation system (modified in-place)
///
/// # Returns
///
/// Tuple of (model_result, new_trace) after the MH step.
///
/// # Examples
///
/// ```rust
/// use fugue::*;
/// use rand::rngs::StdRng;
/// use rand::SeedableRng;
///
/// // Set up initial state with simple model
/// let model_fn = || sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap());
///
/// let mut rng = StdRng::seed_from_u64(42);
/// let (_, initial_trace) = runtime::handler::run(
///     PriorHandler { rng: &mut rng, trace: Trace::default() },
///     model_fn()
/// );
///
/// // Perform one MH step
/// let mut adaptation = DiminishingAdaptation::new(0.44, 0.7);
/// let (result, new_trace) = adaptive_single_site_mh(
///     &mut rng,
///     model_fn,
///     &initial_trace,
///     &mut adaptation,
/// );
/// assert!(new_trace.choices.len() > 0);
/// ```
pub fn adaptive_single_site_mh<A, R: Rng>(
    rng: &mut R,
    model_fn: impl Fn() -> Model<A>,
    current: &Trace,
    adaptation: &mut DiminishingAdaptation,
) -> (A, Trace) {
    let overrides: HashMap<Address, SiteProposal> = HashMap::new();
    let mut kind_cache: HashMap<Address, SiteProposal> = HashMap::new();

    if current.choices.is_empty() {
        // No latent choices to update; just recover the model result.
        let (a, _) = run(
            ScoreGivenTrace {
                base: current.clone(),
                trace: Trace::default(),
            },
            model_fn(),
        );
        return (a, current.clone());
    }

    // Score the current state once. The model result `a_cur` is reused on
    // rejection instead of re-executing the model a third time (FG-12).
    let (a_cur, cur_scored) = run(
        ScoreGivenTrace {
            base: current.clone(),
            trace: Trace::default(),
        },
        model_fn(),
    );
    let current_lw = cur_scored.total_log_weight();

    let sites: Vec<Address> = current.choices.keys().cloned().collect();
    let target = sites[rng.gen_range(0..sites.len())].clone();
    let scale = adaptation.get_scale(&target);

    let (a_prop, prop_trace, prop_lw, lqf, lqr) = propose_and_score(
        rng,
        &model_fn,
        current,
        &target,
        scale,
        &overrides,
        &mut kind_cache,
    );

    let log_alpha = prop_lw - current_lw + (lqr - lqf);
    let accept = log_alpha >= 0.0 || rng.gen::<f64>() < log_alpha.exp();
    adaptation.update(&target, accept);

    if accept {
        (a_prop, prop_trace)
    } else {
        (a_cur, current.clone())
    }
}

/// Run an adaptive MCMC chain with automatic proposal tuning.
///
/// This is the main entry point for running Metropolis-Hastings MCMC on a
/// probabilistic model. It automatically handles initialization, warmup/burn-in,
/// and adaptive tuning of proposal scales to achieve good mixing.
///
/// # Algorithm
///
/// 1. Initialize the chain with a prior sample (correct, fresh accumulators).
/// 2. Run the warmup period, discarding samples but adapting scales. The current
///    state's score and the site list are cached across iterations, so each step
///    re-executes the model exactly once (FG-11/FG-12).
/// 3. **Freeze** the tuned scales and collect samples from the resulting fixed
///    transition kernel (FG-57).
/// 4. Return the post-warmup samples, each carrying a freshly-scored trace (FG-40).
///
/// Use [`adaptive_mcmc_chain_with_overrides`] to force specific proposals per
/// address.
///
/// # Arguments
///
/// * `rng` - Random number generator
/// * `model_fn` - Function that creates the model (should be the same each time)
/// * `n_samples` - Number of post-warmup samples to collect
/// * `n_warmup` - Number of warmup/burn-in iterations (not returned)
///
/// # Returns
///
/// Vector of (model_result, trace) pairs from the post-warmup sampling.
///
/// # Examples
///
/// ```rust
/// use fugue::*;
/// use rand::rngs::StdRng;
/// use rand::SeedableRng;
///
/// // Very simple model for testing
/// let model_fn = || {
///     sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap())
/// };
///
/// let mut rng = StdRng::seed_from_u64(42);
/// let samples = adaptive_mcmc_chain(
///     &mut rng,
///     model_fn,
///     5, // samples (very small for test)
///     1, // warmup
/// );
///
/// // Extract mu estimates
/// let mu_values: Vec<f64> = samples.iter()
///     .filter_map(|(result, _)| Some(*result))
///     .collect();
/// assert!(!mu_values.is_empty());
/// ```
pub fn adaptive_mcmc_chain<A: Clone, R: Rng>(
    rng: &mut R,
    model_fn: impl Fn() -> Model<A>,
    n_samples: usize,
    n_warmup: usize,
) -> Vec<(A, Trace)> {
    let overrides: HashMap<Address, SiteProposal> = HashMap::new();
    adaptive_mcmc_chain_with_overrides(rng, model_fn, n_samples, n_warmup, &overrides)
}

/// Like [`adaptive_mcmc_chain`], but with explicit per-address `f64` proposal
/// overrides (FG-42).
///
/// Any address present in `overrides` uses the specified [`SiteProposal`] instead
/// of the automatically-detected one. This is the escape hatch for cases the
/// support-based auto-detection cannot infer (e.g. a `[a,b]`-bounded parameter
/// that should use a reflected walk).
pub fn adaptive_mcmc_chain_with_overrides<A: Clone, R: Rng>(
    rng: &mut R,
    model_fn: impl Fn() -> Model<A>,
    n_samples: usize,
    n_warmup: usize,
    overrides: &HashMap<Address, SiteProposal>,
) -> Vec<(A, Trace)> {
    let mut samples = Vec::with_capacity(n_samples);
    let mut adaptation = DiminishingAdaptation::new(0.44, 0.7);
    let mut kind_cache: HashMap<Address, SiteProposal> = HashMap::new();

    // Initialize with a prior sample (fresh, correct accumulators).
    let (mut current_a, mut current_trace) = run(
        PriorHandler {
            rng,
            trace: Trace::default(),
        },
        model_fn(),
    );
    let mut current_lw = current_trace.total_log_weight();

    // FG-11: cache the ordered site list; rebuild only when the address set
    // changes. Single-site MH keeps the model structure fixed, so for the common
    // case this is built once and reused for the whole chain.
    let mut sites: Vec<Address> = current_trace.choices.keys().cloned().collect();

    // Warmup phase: adapt proposal scales.
    for _ in 0..n_warmup {
        if sites.len() != current_trace.choices.len() {
            sites = current_trace.choices.keys().cloned().collect();
        }
        if let Some((a, t, lw)) = single_site_mh_step(
            rng,
            &model_fn,
            &current_trace,
            current_lw,
            &sites,
            &mut adaptation,
            overrides,
            &mut kind_cache,
            true, // adapt during warmup
        ) {
            current_a = a;
            current_trace = t;
            current_lw = lw;
        }
    }

    // Sampling phase: FG-57 freeze adaptation so the recorded draws come from a
    // single fixed transition kernel.
    for _ in 0..n_samples {
        if sites.len() != current_trace.choices.len() {
            sites = current_trace.choices.keys().cloned().collect();
        }
        if let Some((a, t, lw)) = single_site_mh_step(
            rng,
            &model_fn,
            &current_trace,
            current_lw,
            &sites,
            &mut adaptation,
            overrides,
            &mut kind_cache,
            false, // frozen scales during sampling
        ) {
            current_a = a;
            current_trace = t;
            current_lw = lw;
        }
        samples.push((current_a.clone(), current_trace.clone()));
    }

    samples
}

/// Backward-compatible thin wrapper over [`adaptive_single_site_mh`].
pub fn single_site_random_walk_mh<A, R: Rng>(
    rng: &mut R,
    _proposal_sigma: f64,
    model_fn: impl Fn() -> Model<A>,
    current: &Trace,
) -> (A, Trace) {
    let mut adaptation = DiminishingAdaptation::new(0.44, 0.7);
    adaptive_single_site_mh(rng, model_fn, current, &mut adaptation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::addr;
    use crate::core::distribution::*;
    use crate::core::model::{observe, sample, ModelExt};
    use crate::runtime::handler::run;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn gaussian_walk_proposal_produces_variation() {
        let mut rng = StdRng::seed_from_u64(11);
        let strat = GaussianWalkProposal;
        let x1 = strat.propose(0.0, 1.0, &mut rng);
        assert!(x1.is_finite());
    }

    #[test]
    fn log_space_proposal_maintains_positivity() {
        let mut rng = StdRng::seed_from_u64(42);
        let strat = LogSpaceWalkProposal;
        for &current in &[0.1, 1.0, 10.0, 100.0] {
            for _ in 0..20 {
                let proposed = strat.propose(current, 0.5, &mut rng);
                assert!(
                    proposed > 0.0,
                    "LogSpaceWalk proposed non-positive: {current} -> {proposed}"
                );
                assert!(
                    proposed.is_finite(),
                    "LogSpaceWalk proposed non-finite: {proposed}"
                );
            }
        }
    }

    // FG-02: the log-space walk's Jacobian/Hastings correction must equal
    // +(ln x' − ln x). log_proposal_prob returns N(ln·) − ln·, so the net
    // reverse−forward correction is exactly that. Verify numerically.
    #[test]
    fn log_space_jacobian_is_correct() {
        let s = LogSpaceWalkProposal;
        let (x, xp, scale) = (2.0_f64, 3.5_f64, 0.7_f64);
        let fwd = s.log_proposal_prob(x, xp, scale);
        let rev = s.log_proposal_prob(xp, x, scale);
        let net = rev - fwd;
        let expected = xp.ln() - x.ln();
        assert!(
            (net - expected).abs() < 1e-12,
            "net correction {net} != {expected}"
        );
    }

    #[test]
    fn reflection_proposal_respects_bounds() {
        let mut rng = StdRng::seed_from_u64(43);
        let strat = ReflectionWalkProposal {
            lower_bound: 0.0,
            upper_bound: 1.0,
        };
        for &current in &[0.1, 0.5, 0.9] {
            for _ in 0..20 {
                let proposed = strat.propose(current, 0.3, &mut rng);
                assert!(
                    (0.0..=1.0).contains(&proposed),
                    "bounds violated: {current} -> {proposed}"
                );
            }
        }
    }

    #[test]
    fn discrete_and_flip_proposals_preserve_types() {
        let mut rng = StdRng::seed_from_u64(12);
        let u = DiscreteWalkProposal.propose(5u64, 1.0, &mut rng);
        let _ = u;
        let b = FlipProposal.propose(true, 1.0, &mut rng);
        assert!(!b); // deterministic flip
    }

    // FG-41: the reflected discrete walk must be a symmetric kernel, including at
    // the boundary state 0 (where naive |x+δ| is asymmetric by a factor of 2).
    // Estimate q(a→b) and q(b→a) by Monte Carlo and check equality for pairs that
    // straddle the boundary.
    #[test]
    fn discrete_walk_is_symmetric_at_boundary() {
        let mut rng = StdRng::seed_from_u64(2718);
        let s = DiscreteWalkProposal;
        let scale = 1.5;
        let iters = 400_000;
        // Estimate transition probabilities for the pairs (0,1) and (1,0) etc.
        let estimate = |from: u64, to: u64, rng: &mut StdRng| -> f64 {
            let mut hits = 0u64;
            for _ in 0..iters {
                if s.propose(from, scale, rng) == to {
                    hits += 1;
                }
            }
            hits as f64 / iters as f64
        };
        for &(a, b) in &[(0u64, 1u64), (0, 2), (1, 3), (2, 5)] {
            let q_ab = estimate(a, b, &mut rng);
            let q_ba = estimate(b, a, &mut rng);
            // Symmetric: q(a→b) == q(b→a). Tolerance covers MC noise on ~4e5 draws.
            let diff = (q_ab - q_ba).abs();
            assert!(
                diff < 0.004,
                "asymmetry at ({a},{b}): q_ab={q_ab:.4}, q_ba={q_ba:.4}, diff={diff:.4}"
            );
        }
    }

    #[test]
    fn adaptive_chain_runs_and_returns_samples() {
        let model_fn = || {
            sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap()).and_then(|mu| {
                observe(addr!("y"), Normal::new(mu, 1.0).unwrap(), 0.5).map(move |_| mu)
            })
        };
        let mut rng = StdRng::seed_from_u64(13);
        let samples = adaptive_mcmc_chain(&mut rng, model_fn, 5, 2);
        assert_eq!(samples.len(), 5);
        for (_val, t) in &samples {
            assert!(t.get_f64(&addr!("mu")).is_some());
        }
    }

    // FG-40: accepted samples carry freshly-scored accumulators — the returned
    // trace's total_log_weight() must equal a fresh full rescore.
    #[test]
    fn returned_trace_weight_matches_fresh_rescore() {
        let model_fn = || {
            sample(addr!("mu"), Normal::new(0.0, 2.0).unwrap()).and_then(|mu| {
                observe(addr!("y"), Normal::new(mu, 1.0).unwrap(), 1.3).map(move |_| mu)
            })
        };
        let mut rng = StdRng::seed_from_u64(77);
        let samples = adaptive_mcmc_chain(&mut rng, model_fn, 20, 20);
        for (_v, t) in &samples {
            let (_a, fresh) = run(
                ScoreGivenTrace {
                    base: t.clone(),
                    trace: Trace::default(),
                },
                model_fn(),
            );
            assert!(
                (t.total_log_weight() - fresh.total_log_weight()).abs() < 1e-9,
                "stale accumulators: {} vs {}",
                t.total_log_weight(),
                fresh.total_log_weight()
            );
        }
    }

    // FG-11 / FG-12: each transition re-executes the model exactly once. The
    // chain builds the model once for the initial prior draw and once per step;
    // on rejection there is no extra run. Count model_fn invocations.
    #[test]
    fn one_model_run_per_transition() {
        use std::cell::Cell;
        let count = Cell::new(0usize);
        let model_fn = || {
            count.set(count.get() + 1);
            sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap()).and_then(|mu| {
                observe(addr!("y"), Normal::new(mu, 1.0).unwrap(), 0.5).map(move |_| mu)
            })
        };
        let mut rng = StdRng::seed_from_u64(5);
        let n_warmup = 30;
        let n_samples = 40;
        let _ = adaptive_mcmc_chain(&mut rng, model_fn, n_samples, n_warmup);
        // 1 initial prior build + one build per warmup + sampling step.
        assert_eq!(count.get(), 1 + n_warmup + n_samples);
    }

    // FG-57: scales must be frozen during the sampling phase. Drive the internal
    // step with adapt=false and confirm the scale map does not change, while
    // adapt=true does change it.
    #[test]
    fn adaptation_freezes_after_warmup() {
        let model_fn = || {
            sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap()).and_then(|mu| {
                observe(addr!("y"), Normal::new(mu, 1.0).unwrap(), 0.5).map(move |_| mu)
            })
        };
        let mut rng = StdRng::seed_from_u64(99);
        let mut adaptation = DiminishingAdaptation::new(0.44, 0.7);
        let overrides: HashMap<Address, SiteProposal> = HashMap::new();
        let mut kind_cache: HashMap<Address, SiteProposal> = HashMap::new();

        let (_a, mut current) = run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            model_fn(),
        );
        let mut current_lw = current.total_log_weight();
        let sites: Vec<Address> = current.choices.keys().cloned().collect();

        // Warm up with adaptation on.
        for _ in 0..100 {
            if let Some((_a, t, lw)) = single_site_mh_step(
                &mut rng,
                &model_fn,
                &current,
                current_lw,
                &sites,
                &mut adaptation,
                &overrides,
                &mut kind_cache,
                true,
            ) {
                current = t;
                current_lw = lw;
            }
        }
        let scales_before = adaptation.scales.clone();

        // Sampling with adaptation frozen: scales must be untouched.
        for _ in 0..200 {
            if let Some((_a, t, lw)) = single_site_mh_step(
                &mut rng,
                &model_fn,
                &current,
                current_lw,
                &sites,
                &mut adaptation,
                &overrides,
                &mut kind_cache,
                false,
            ) {
                current = t;
                current_lw = lw;
            }
        }
        assert_eq!(
            scales_before, adaptation.scales,
            "scales changed while adaptation was frozen"
        );

        // Sanity: with adaptation on, the scale does move.
        let before = adaptation.get_scale(&sites[0]);
        for _ in 0..100 {
            let _ = single_site_mh_step(
                &mut rng,
                &model_fn,
                &current,
                current_lw,
                &sites,
                &mut adaptation,
                &overrides,
                &mut kind_cache,
                true,
            );
        }
        let after = adaptation.get_scale(&sites[0]);
        assert!(
            (before - after).abs() > 0.0,
            "adaptation did nothing while enabled"
        );
    }
}
