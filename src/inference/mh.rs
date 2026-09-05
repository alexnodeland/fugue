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
//! - **`f64`**: the kind is a function of the site's distribution alone, read
//!   from [`Distribution::support`] (FG-N1): [`Support::Real`] gets a symmetric
//!   **Gaussian** random walk (out-of-support proposals score `−inf` and are
//!   rejected, so this is always correct, if not always efficient);
//!   [`Support::Positive`] gets a **log-space** walk with the exact
//!   Jacobian/Hastings correction (FG-02); [`Support::Bounded`] gets a
//!   Gaussian walk **reflected** at both bounds (symmetric). The previous
//!   heuristic chose log-space from `current > 0` plus a density probe at `−1`,
//!   which confined a chain on e.g. `Uniform(−0.5, 0.5)` to the positive half
//!   forever: a kernel whose shape depends on the current *value* is not
//!   invariant for the target, and the kind was cached per address, so the
//!   whole chain inherited its first state's sign.
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
//! ## Structure-varying (trans-dimensional) models (FG-20 / FG-21)
//!
//! Models whose set of sample addresses depends on a sampled value (e.g.
//! `b ~ Bernoulli; if b { x ~ … }`) are handled without panicking. A proposal
//! that opens a new branch samples the fresh sites from their prior and treats
//! the change as a reversible-jump birth; a proposal that closes a branch treats
//! the vanished sites as a death. Both the fresh/vanished sites' prior densities
//! (as prior-proposal q terms) and the change in the single-site selection
//! probability (`ln|sites(current)| − ln|sites(proposed)|`) enter the acceptance
//! ratio, so the chain leaves the correct trans-dimensional posterior invariant
//! for prior-proposed structure changes rather than silently biasing it. For
//! fixed-structure models every one of these corrections is identically zero, so
//! behavior is unchanged.
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
use crate::core::distribution::{Distribution, Support};
use crate::core::model::Model;
use crate::inference::mcmc_utils::DiminishingAdaptation;
use crate::runtime::handler::{run, Handler};
use crate::runtime::interpreters::{score_given_trace_reconciled, PriorHandler, ScoreGivenTrace};
use crate::runtime::trace::{Choice, ChoiceValue, Trace};
use rand::{Rng, RngCore};
use std::collections::HashMap;

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
/// The samplers pick a proposal automatically from each site's
/// [`Distribution::support`] (see [`proposal_kind_for_support`]), but callers
/// can force a specific kind via [`adaptive_mcmc_chain_with_overrides`] or
/// [`adaptive_single_site_mh_with_overrides`]. An explicit override always wins.
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

/// The automatic `f64` proposal kind for a site whose distribution advertises
/// `support` (FG-N1).
///
/// This is deliberately a function of the *distribution* only — never of the
/// value currently at the site — so the single-site kernel has the same shape
/// in every state and is invariant for the target. `Real` → Gaussian walk,
/// `Positive` → log-space walk (FG-02 Jacobian included), `Bounded` → reflected
/// walk within the bounds.
pub fn proposal_kind_for_support(support: Support) -> SiteProposal {
    match support {
        Support::Real => SiteProposal::Gaussian,
        Support::Positive => SiteProposal::LogSpace,
        Support::Bounded { lower, upper } => SiteProposal::Reflect { lower, upper },
    }
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
/// ## Structure-varying (trans-dimensional) proposals (FG-20 / FG-21)
///
/// If `target`'s new value opens a branch that requires an address absent from
/// `base`, that address is sampled fresh from its prior (rather than panicking as
/// raw `ScoreGivenTrace` would). This is treated as a **reversible-jump birth
/// with the prior as the proposal**: the fresh site's prior log-density is added
/// to `log_q_forward`, so it cancels the same `log_prior` term the site
/// contributes to the proposal's joint and the acceptance ratio reduces to the
/// correct RJMCMC form (Jacobian `= 1`). Symmetrically, an address present in
/// `base` that the proposed structure no longer visits (a **death**) has its
/// prior log-density added to `log_q_reverse` by [`propose_and_score`], canceling
/// its contribution to the current state's joint. Together these make single-site
/// MH leave the correct (trans-dimensional) posterior invariant for models whose
/// fresh sub-structure is sampled from the prior — e.g. `b ~ Bernoulli; if b { x ~
/// … }` — instead of silently biasing it. The sampler never panics on a
/// structure-varying model; it continues with the RJMCMC-corrected ratio.
struct SingleSiteProposalHandler<'a, R: RngCore> {
    rng: &'a mut R,
    base: &'a Trace,
    target: &'a Address,
    scale: f64,
    overrides: &'a HashMap<Address, SiteProposal>,
    log_q_forward: &'a mut f64,
    log_q_reverse: &'a mut f64,
    trace: Trace,
}

impl<'a, R: RngCore> SingleSiteProposalHandler<'a, R> {
    /// Decide the `f64` proposal kind for the target site: an explicit override
    /// wins (FG-42); otherwise the kind is read from the distribution's declared
    /// support (FG-N1). Nothing here looks at the current value, and nothing is
    /// cached across executions — a distribution whose bounds depend on another
    /// site (e.g. `Uniform(0, sigma)`) gets the bounds of *this* execution.
    fn f64_kind(&self, addr: &Address, dist: &dyn Distribution<f64>) -> SiteProposal {
        if let Some(&k) = self.overrides.get(addr) {
            return k;
        }
        proposal_kind_for_support(dist.support())
    }
}

impl<'a, R: RngCore> Handler for SingleSiteProposalHandler<'a, R> {
    fn on_sample_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>) -> f64 {
        if addr == self.target {
            let current = self
                .base
                .get_f64(addr)
                .unwrap_or_else(|| dist.sample(self.rng));
            let kind = self.f64_kind(addr, dist);
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
            // Accumulate (`+=`, not `=`) so a fresh dimension born earlier in the
            // execution order (its prior term already added to `log_q_forward`)
            // is not clobbered by the target's own proposal density.
            *self.log_q_forward += lqf;
            *self.log_q_reverse += lqr;
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
            let (x, born) = match self.base.get_f64(addr) {
                Some(v) => (v, false),
                None => (dist.sample(self.rng), true),
            };
            let lp = dist.log_prob(&x);
            if born {
                // RJMCMC birth from the prior: cancel this fresh site's log_prior.
                *self.log_q_forward += lp;
            }
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
        let mut born = false;
        let x = if addr == self.target {
            let current = self
                .base
                .get_bool(addr)
                .unwrap_or_else(|| dist.sample(self.rng));
            // Symmetric deterministic flip: contributes 0 to both q terms (leave
            // any born/died structural corrections already accumulated intact).
            FlipProposal.propose(current, self.scale, self.rng)
        } else {
            match self.base.get_bool(addr) {
                Some(v) => v,
                None => {
                    born = true;
                    dist.sample(self.rng)
                }
            }
        };
        let lp = dist.log_prob(&x);
        if born {
            // RJMCMC birth from the prior: cancel this fresh site's log_prior.
            *self.log_q_forward += lp;
        }
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
        let mut born = false;
        let x = if addr == self.target {
            let current = self
                .base
                .get_u64(addr)
                .unwrap_or_else(|| dist.sample(self.rng));
            // Symmetric reflected discrete walk (FG-41): contributes 0 to both q
            // terms (leave any born/died structural corrections intact).
            DiscreteWalkProposal.propose(current, self.scale, self.rng)
        } else {
            match self.base.get_u64(addr) {
                Some(v) => v,
                None => {
                    born = true;
                    dist.sample(self.rng)
                }
            }
        };
        let lp = dist.log_prob(&x);
        if born {
            // RJMCMC birth from the prior: cancel this fresh site's log_prior.
            *self.log_q_forward += lp;
        }
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
        let mut born = false;
        let x = if addr == self.target {
            let current = self
                .base
                .get_usize(addr)
                .unwrap_or_else(|| dist.sample(self.rng));
            // FG-10: resample from the site's prior. With q = prior the Hastings
            // terms cancel the prior in the target, so acceptance reduces to the
            // likelihood ratio and no category can ever be missed.
            let proposed = dist.sample(self.rng);
            // `+=` so a born fresh dimension's prior term is preserved.
            *self.log_q_forward += dist.log_prob(&proposed);
            *self.log_q_reverse += dist.log_prob(&current);
            proposed
        } else {
            match self.base.get_usize(addr) {
                Some(v) => v,
                None => {
                    born = true;
                    dist.sample(self.rng)
                }
            }
        };
        let lp = dist.log_prob(&x);
        if born {
            // RJMCMC birth from the prior: cancel this fresh site's log_prior.
            *self.log_q_forward += lp;
        }
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
        let mut born = false;
        let x = if addr == self.target {
            let current = self
                .base
                .get_i64(addr)
                .unwrap_or_else(|| dist.sample(self.rng));
            // Symmetric integer random walk (no boundary to reflect at):
            // contributes 0 to both q terms (leave born/died corrections intact).
            let delta = (self.scale * gaussian_z(self.rng)).round() as i64;
            current + delta
        } else {
            match self.base.get_i64(addr) {
                Some(v) => v,
                None => {
                    born = true;
                    dist.sample(self.rng)
                }
            }
        };
        let lp = dist.log_prob(&x);
        if born {
            // RJMCMC birth from the prior: cancel this fresh site's log_prior.
            *self.log_q_forward += lp;
        }
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
///
/// `log_q_forward` accumulates the target's proposal density plus the prior
/// density of every fresh dimension born by the proposal (the RJMCMC birth term).
/// `log_q_reverse` accumulates the target's reverse density plus the prior
/// density of every dimension that DIED — present in `current` but not visited by
/// the proposed structure — which is the reverse-move birth term for those sites
/// (FG-20 / FG-21). Together they make `log α = Δlog-joint + log q_reverse −
/// log q_forward` the correct trans-dimensional acceptance ratio for
/// prior-proposed structural changes.
///
/// The final `bool` reports whether the proposed trace's address SET differs from
/// `current` (a birth and/or death occurred), so the chain driver can refresh its
/// cached site list even when the site count is unchanged (e.g. a branch that
/// swaps one address for another).
#[allow(clippy::type_complexity)]
pub(crate) fn propose_and_score<A, F, R>(
    rng: &mut R,
    model_fn: &F,
    current: &Trace,
    target: &Address,
    scale: f64,
    overrides: &HashMap<Address, SiteProposal>,
) -> (A, Trace, f64, f64, f64, bool)
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
            log_q_forward: &mut lqf,
            log_q_reverse: &mut lqr,
            trace: Trace::default(),
        },
        model_fn(),
    );
    // Death correction: any address in `current` the proposal no longer visits is
    // a dimension the reverse move would have to birth from its prior. Adding its
    // stored prior log-density (the reverse-birth proposal density) to
    // `log_q_reverse` cancels its contribution to `current`'s joint in the
    // acceptance ratio, completing the RJMCMC dimension-matching (FG-20/FG-21).
    let mut died = 0usize;
    for (addr, choice) in &current.choices {
        if !trace.choices.contains_key(addr) {
            lqr += choice.logp;
            died += 1;
        }
    }
    // born = |proposed| − |current| + died  (|proposed| = |current| − died + born).
    let born = trace.choices.len() + died - current.choices.len();
    let structure_changed = born > 0 || died > 0;
    let lw = trace.total_log_weight();
    (a, trace, lw, lqf, lqr, structure_changed)
}

/// One cached single-site MH transition used by the chain driver.
///
/// The current state's log-weight (`current_lw`) and the ordered `sites` list are
/// supplied by the caller and cached across iterations, so this performs exactly
/// one model run (the proposal). Returns `Some((result, trace, log_weight))` on
/// acceptance (a freshly-scored trace, FG-40) and `None` on rejection — the
/// caller keeps its cached current state, so no extra model run happens on
/// rejection (FG-12).
///
/// On acceptance the returned tuple's final `bool` flags whether the accepted
/// move changed the model's address structure, so the driver can refresh its
/// cached site list (FG-20/FG-21).
#[allow(clippy::too_many_arguments)]
fn single_site_mh_step<A, F, R>(
    rng: &mut R,
    model_fn: &F,
    current: &Trace,
    current_lw: f64,
    sites: &[Address],
    adaptation: &mut DiminishingAdaptation,
    overrides: &HashMap<Address, SiteProposal>,
    adapt: bool,
) -> Option<(A, Trace, f64, bool)>
where
    F: Fn() -> Model<A>,
    R: Rng,
{
    if sites.is_empty() {
        return None;
    }
    let target = sites[rng.gen_range(0..sites.len())].clone();
    let scale = adaptation.get_scale(&target);

    let (a_prop, prop_trace, prop_lw, lqf, lqr, structure_changed) =
        propose_and_score(rng, model_fn, current, &target, scale, overrides);

    // log α = Δlog-joint + log q(x|x') − log q(x'|x) + dimension term.
    //
    // The single-site kernel picks the target uniformly among the *current*
    // sites, so the forward move carries proposal factor 1/|sites(current)| and
    // the reverse carries 1/|sites(proposed)|. For structure-varying proposals
    // these differ, and the term `ln|sites(current)| − ln|sites(proposed)|`
    // completes the RJMCMC dimension matching (FG-20/FG-21). For fixed-structure
    // models the two site counts are equal and the term is exactly 0.
    let dim_term = (sites.len() as f64).ln() - (prop_trace.choices.len() as f64).ln();
    let log_alpha = prop_lw - current_lw + (lqr - lqf) + dim_term;
    let accept = log_alpha >= 0.0 || rng.gen::<f64>() < log_alpha.exp();

    if adapt {
        adaptation.update(&target, accept);
    }

    if accept {
        Some((a_prop, prop_trace, prop_lw, structure_changed))
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
///    `log α = Δlog-joint + q(x|x') − q(x'|x)` plus, for structure-varying
///    proposals, the RJMCMC dimension term (0 for fixed-structure models).
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

    let (a_prop, prop_trace, prop_lw, lqf, lqr, _structure_changed) =
        propose_and_score(rng, &model_fn, current, &target, scale, &overrides);

    // Dimension term for structure-varying proposals (see `single_site_mh_step`);
    // 0 for fixed-structure models.
    let dim_term = (sites.len() as f64).ln() - (prop_trace.choices.len() as f64).ln();
    let log_alpha = prop_lw - current_lw + (lqr - lqf) + dim_term;
    let accept = log_alpha >= 0.0 || rng.gen::<f64>() < log_alpha.exp();
    adaptation.update(&target, accept);

    if accept {
        (a_prop, prop_trace)
    } else {
        (a_cur, current.clone())
    }
}

/// One block-regeneration Metropolis–Hastings transition.
///
/// Deletes the choices at every address in `block` from `current`, replays the
/// model to fill them (fresh draws from the prior, via
/// [`score_given_trace_reconciled`]), and accepts or rejects with the
/// prior-cancelling acceptance ratio below. This generalizes single-site MH
/// from one target address to an arbitrary address set S — the "selective
/// resampling" primitive: proposal = regenerate the sub-trace at S from the
/// prior conditioned on the untouched coordinates.
///
/// `beta` tempers the likelihood (`π_β(θ) ∝ p(θ)·p(y|θ)^β`; pass `1.0` for an
/// untempered posterior move), which makes the move directly usable as an SMC
/// block-rejuvenation kernel. Addresses in `block` absent from `current` are
/// ignored; present-but-mismatched-type entries are treated as fresh by the
/// reconciler. The returned trace is freshly scored (FG-40/FG-48), so
/// `total_log_weight()` is valid.
///
/// # Acceptance ratio
///
/// ```text
/// log α = (log_prior′ − log_prior) + β·(loglik′ − loglik) + log q_rev − log q_fwd
/// log q_fwd = Σ_{a ∈ fresh}                    logp′(a)     (forward births from the prior)
/// log q_rev = Σ_{a ∈ S present in current}     logp(a)
///           + Σ_{a ∈ vanished}                 logp(a)      (death correction, FG-20/FG-21)
/// ```
///
/// For a **fixed address structure** (`fresh = S`, `vanished = ∅`) the prior
/// and proposal terms cancel exactly and this collapses to
/// `log α = β·(loglik′ − loglik)` — the prior-cancellation property that makes
/// block regeneration cheap to accept/reject.
///
/// Unlike the single-site kernels there is **no dimension-selection term**
/// (`ln|sites(current)| − ln|sites(proposed)|`): that term corrects for picking
/// the target uniformly among a state-dependent site set, whereas here the
/// block S is fixed by the caller — the forward and reverse moves regenerate
/// deterministic address sets whose proposal densities are fully accounted by
/// the `log q` terms above.
///
/// If the reconciling replay reports an address conflict (the model visits the
/// same address twice, FG-47), the move is treated as a rejection and the
/// (re-scored) current state is returned.
pub fn block_regeneration_mh<A, R: Rng>(
    rng: &mut R,
    model_fn: impl Fn() -> Model<A>,
    current: &Trace,
    block: &[Address],
    beta: f64,
) -> (A, Trace) {
    // Score the current state once; also the state returned on rejection (FG-40).
    let (a_cur, cur_scored) = run(
        ScoreGivenTrace {
            base: current.clone(),
            trace: Trace::default(),
        },
        model_fn(),
    );

    // Delete the block, then replay: removed addresses the model re-visits are
    // drawn fresh from their prior (reported in `fresh_addresses`); addresses
    // the model no longer visits are `vanished_addresses`.
    let mut base = current.clone();
    for a in block {
        base.choices.remove(a);
    }
    let (a_prop, prop, report) = match score_given_trace_reconciled(base, rng, model_fn()) {
        Ok(t) => t,
        Err(_) => return (a_cur, cur_scored), // reject on address conflict
    };

    let log_q_fwd: f64 = report
        .fresh_addresses
        .iter()
        .filter_map(|a| prop.choices.get(a).map(|c| c.logp))
        .sum();
    // Block ∩ vanished = ∅ (vanished is computed against the block-deleted
    // base), so the two reverse-birth sums never double-count a site.
    let log_q_rev: f64 = block
        .iter()
        .filter_map(|a| current.choices.get(a).map(|c| c.logp))
        .sum::<f64>()
        + report
            .vanished_addresses
            .iter()
            .filter_map(|a| current.choices.get(a).map(|c| c.logp))
            .sum::<f64>();

    let loglik = |t: &Trace| t.log_likelihood + t.log_factors;
    let log_alpha = (prop.log_prior - cur_scored.log_prior)
        + beta * (loglik(&prop) - loglik(&cur_scored))
        + log_q_rev
        - log_q_fwd;

    if log_alpha >= 0.0 || rng.gen::<f64>() < log_alpha.exp() {
        (a_prop, prop)
    } else {
        (a_cur, cur_scored)
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
    adaptive_mcmc_chain_thinned(rng, model_fn, n_samples, n_warmup, 1)
}

/// Like [`adaptive_mcmc_chain`], but retaining only every `thin`-th draw.
///
/// # Why this exists
///
/// [`adaptive_mcmc_chain`] materializes **every** iteration: it pushes an
/// `(A, Trace)` per step into the `Vec` it returns by value. A caller that only
/// wants a thinned subsequence — which is the common case, because
/// autocorrelated single-site draws are usually thinned before use — has no way
/// to say so, and pays peak memory for the whole chain before discarding most
/// of it one line later.
///
/// The cost is not theoretical. A structure-varying model with ~140 sites over
/// a 10 000-step chain holds ~10 000 `Trace` clones of ~140 `BTreeMap` entries
/// live simultaneously, to keep 500 of them. On a 32-bit wasm heap that is a
/// plausible out-of-memory rather than mere waste, and it scales with
/// `n_samples` — so the caller's only lever is to shorten the chain, i.e. to
/// pay in statistics for a memory problem.
///
/// # What is guaranteed
///
/// **The surviving draws are bit-identical to thinning the full chain.** The
/// loop still runs `n_samples` iterations, every transition is still attempted,
/// and the RNG is consumed in exactly the same order and quantity — `thin` gates
/// the `push` and nothing else. So for any `thin`:
///
/// ```text
/// adaptive_mcmc_chain_thinned(seeded_rng(s), f, n, w, thin)
///     == adaptive_mcmc_chain(seeded_rng(s), f, n, w)
///            .into_iter().step_by(thin).collect()
/// ```
///
/// Retained indices are `0, thin, 2·thin, …`, matching
/// [`Iterator::step_by`]. `thin = 0` is treated as `1`.
///
/// This is a memory optimization with no statistical content: thinning a chain
/// discards information and is *not* a way to improve mixing. Use it when the
/// draws were going to be thinned anyway.
pub fn adaptive_mcmc_chain_thinned<A: Clone, R: Rng>(
    rng: &mut R,
    model_fn: impl Fn() -> Model<A>,
    n_samples: usize,
    n_warmup: usize,
    thin: usize,
) -> Vec<(A, Trace)> {
    let overrides: HashMap<Address, SiteProposal> = HashMap::new();
    adaptive_mcmc_chain_with_overrides_thinned(rng, model_fn, n_samples, n_warmup, &overrides, thin)
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
    adaptive_mcmc_chain_with_overrides_thinned(rng, model_fn, n_samples, n_warmup, overrides, 1)
}

/// [`adaptive_mcmc_chain_with_overrides`] with retention thinning — see
/// [`adaptive_mcmc_chain_thinned`] for what `thin` does and does not change.
pub fn adaptive_mcmc_chain_with_overrides_thinned<A: Clone, R: Rng>(
    rng: &mut R,
    model_fn: impl Fn() -> Model<A>,
    n_samples: usize,
    n_warmup: usize,
    overrides: &HashMap<Address, SiteProposal>,
    thin: usize,
) -> Vec<(A, Trace)> {
    // A `thin` of 0 would divide by zero below; it can only mean "keep
    // everything", which is what 1 does.
    let thin = thin.max(1);
    let mut samples = Vec::with_capacity(n_samples.div_ceil(thin));
    let mut adaptation = DiminishingAdaptation::new(0.44, 0.7);

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
    // case this is built once and reused for the whole chain. For structure-
    // varying models the list is refreshed after any accepted move that changed
    // the address set — including swaps that keep the site COUNT constant
    // (FG-20/FG-21).
    let mut sites: Vec<Address> = current_trace.choices.keys().cloned().collect();

    // Warmup phase: adapt proposal scales.
    for _ in 0..n_warmup {
        if let Some((a, t, lw, structure_changed)) = single_site_mh_step(
            rng,
            &model_fn,
            &current_trace,
            current_lw,
            &sites,
            &mut adaptation,
            overrides,
            true, // adapt during warmup
        ) {
            current_a = a;
            current_trace = t;
            current_lw = lw;
            if structure_changed {
                sites = current_trace.choices.keys().cloned().collect();
            }
        }
    }

    // Sampling phase: FG-57 freeze adaptation so the recorded draws come from a
    // single fixed transition kernel.
    for i in 0..n_samples {
        if let Some((a, t, lw, structure_changed)) = single_site_mh_step(
            rng,
            &model_fn,
            &current_trace,
            current_lw,
            &sites,
            &mut adaptation,
            overrides,
            false, // frozen scales during sampling
        ) {
            current_a = a;
            current_trace = t;
            current_lw = lw;
            if structure_changed {
                sites = current_trace.choices.keys().cloned().collect();
            }
        }
        // `thin` gates retention and nothing else — the transition above ran
        // regardless, so the chain's arithmetic and its RNG consumption are
        // identical at every `thin`. That is what makes the retained draws
        // bit-identical to `step_by(thin)` over the unthinned chain, and it is
        // why this must stay *below* the step rather than wrapping it.
        if i % thin == 0 {
            samples.push((current_a.clone(), current_trace.clone()));
        }
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

    /// **Thinning must not change the chain — only what is kept from it.**
    ///
    /// This is the whole contract of `adaptive_mcmc_chain_thinned`, and it is
    /// the reason the parameter can be added without any statistical review:
    /// the transition still runs on every iteration, so the RNG is consumed
    /// identically and the retained draws are the *same draws*, not merely
    /// draws from the same distribution.
    ///
    /// Asserted on the values and on the trace weights, at three strides, over
    /// a model with more than one site so a structural difference would show.
    #[test]
    fn thinning_retains_exactly_the_draws_step_by_would() {
        let model_fn = || {
            sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap()).and_then(|mu| {
                sample(addr!("sigma"), LogNormal::new(0.0, 1.0).unwrap()).and_then(move |s| {
                    observe(addr!("y"), Normal::new(mu, s).unwrap(), 0.7).map(move |_| (mu, s))
                })
            })
        };

        // The unthinned reference, from a fixed seed.
        let full = adaptive_mcmc_chain(&mut StdRng::seed_from_u64(0xF117), model_fn, 200, 50);
        assert_eq!(full.len(), 200);

        for thin in [1usize, 7, 20] {
            let thinned = adaptive_mcmc_chain_thinned(
                &mut StdRng::seed_from_u64(0xF117),
                model_fn,
                200,
                50,
                thin,
            );
            let expected: Vec<_> = full.iter().step_by(thin).collect();
            assert_eq!(
                thinned.len(),
                expected.len(),
                "thin={thin}: kept {} draws, step_by kept {}",
                thinned.len(),
                expected.len()
            );
            for (i, (got, want)) in thinned.iter().zip(&expected).enumerate() {
                assert_eq!(
                    got.0, want.0,
                    "thin={thin}: draw {i} differs in value — the chain itself moved"
                );
                assert_eq!(
                    got.1.total_log_weight(),
                    want.1.total_log_weight(),
                    "thin={thin}: draw {i} differs in trace weight"
                );
            }
        }
    }

    /// `thin = 0` cannot mean "keep nothing" — it is normalized to 1 rather
    /// than dividing by zero.
    #[test]
    fn thinning_by_zero_keeps_everything() {
        let model_fn = || sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap());
        let got = adaptive_mcmc_chain_thinned(&mut StdRng::seed_from_u64(7), model_fn, 32, 8, 0);
        assert_eq!(got.len(), 32);
    }

    /// A stride longer than the chain keeps exactly the first draw, matching
    /// `step_by`'s behaviour rather than returning nothing.
    #[test]
    fn thinning_longer_than_the_chain_keeps_the_first_draw() {
        let model_fn = || sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap());
        let got = adaptive_mcmc_chain_thinned(&mut StdRng::seed_from_u64(7), model_fn, 16, 4, 1000);
        assert_eq!(got.len(), 1);
    }

    /// EA-as-PPL F2: block regeneration over the single latent site of a
    /// Beta-Bernoulli model is an independence sampler from the prior and must
    /// reproduce the closed-form Beta posterior.
    #[test]
    #[allow(clippy::needless_borrows_for_generic_args)] // &model_fn is reused across loop iterations
    fn test_block_regen_beta_bernoulli() {
        use crate::inference::validation::{
            test_conjugate_beta_bernoulli_model, ConjugateBetaBernoulliConfig,
        };
        use crate::runtime::interpreters::PriorHandler;

        let observations = vec![
            true, true, false, true, false, true, true, false, true, true,
        ];
        let obs_for_model = observations.clone();
        let model_fn = move || {
            let obs = obs_for_model.clone();
            sample(addr!("theta"), Beta::new(2.0, 2.0).unwrap()).and_then(move |theta| {
                let mut m = crate::core::model::pure(());
                for (i, &o) in obs.iter().enumerate() {
                    m = m.and_then(move |_| {
                        observe(
                            addr!("obs", i),
                            Bernoulli::new(theta.clamp(1e-9, 1.0 - 1e-9)).unwrap(),
                            o,
                        )
                    });
                }
                m.map(move |_| theta)
            })
        };

        let mcmc_fn = |rng: &mut StdRng, n_samples: usize, n_warmup: usize| {
            let (_, mut current) = run(
                PriorHandler {
                    rng,
                    trace: Trace::default(),
                },
                model_fn(),
            );
            let block = [addr!("theta")];
            let mut samples = Vec::with_capacity(n_samples);
            for it in 0..(n_samples + n_warmup) {
                let (v, t) = block_regeneration_mh(rng, &model_fn, &current, &block, 1.0);
                current = t;
                if it >= n_warmup {
                    samples.push((v, current.clone()));
                }
            }
            samples
        };

        let mut rng = StdRng::seed_from_u64(34);
        let result = test_conjugate_beta_bernoulli_model(
            &mut rng,
            mcmc_fn,
            ConjugateBetaBernoulliConfig {
                prior_alpha: 2.0,
                prior_beta: 2.0,
                observations,
                n_samples: 8000,
                n_warmup: 500,
            },
        );
        assert!(
            result.is_valid(),
            "block-regeneration chain failed conjugate Beta-Bernoulli validation"
        );
    }

    /// EA-as-PPL F2: on a fixed-structure product-Normal model, a single block
    /// move over ALL sites targets the same (analytic) posterior as the
    /// single-site kernel — the prior-cancellation collapse `log α = β·Δloglik`.
    #[test]
    #[allow(clippy::needless_borrows_for_generic_args)] // &model_fn is reused across loop iterations
    fn test_block_vs_sequential_single_site() {
        use crate::runtime::interpreters::PriorHandler;

        // Two independent Normal(0,1) sites, each observed with sd 1:
        // posterior per site is Normal(y/2, 1/2).
        let (y0, y1) = (1.0, -0.5);
        let model_fn = move || {
            sample(addr!("x", 0), Normal::new(0.0, 1.0).unwrap()).and_then(move |x0| {
                sample(addr!("x", 1), Normal::new(0.0, 1.0).unwrap()).and_then(move |x1| {
                    observe(addr!("y", 0), Normal::new(x0, 1.0).unwrap(), y0).and_then(move |_| {
                        observe(addr!("y", 1), Normal::new(x1, 1.0).unwrap(), y1)
                            .map(move |_| (x0, x1))
                    })
                })
            })
        };

        let mut rng = StdRng::seed_from_u64(44);
        let (_, mut current) = run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            model_fn(),
        );
        let block = [addr!("x", 0), addr!("x", 1)];
        let mut xs0 = Vec::new();
        let mut xs1 = Vec::new();
        for it in 0..6000 {
            let (_, t) = block_regeneration_mh(&mut rng, &model_fn, &current, &block, 1.0);
            current = t;
            if it >= 500 {
                xs0.push(current.get_f64(&addr!("x", 0)).unwrap());
                xs1.push(current.get_f64(&addr!("x", 1)).unwrap());
            }
        }
        for (xs, y) in [(&xs0, y0), (&xs1, y1)] {
            let mean: f64 = xs.iter().sum::<f64>() / xs.len() as f64;
            let var: f64 = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / xs.len() as f64;
            assert!(
                (mean - y / 2.0).abs() < 0.08,
                "posterior mean {} vs analytic {}",
                mean,
                y / 2.0
            );
            assert!((var - 0.5).abs() < 0.08, "posterior var {} vs 0.5", var);
        }
    }

    /// EA-as-PPL F2: trans-dimensional block regeneration. A Bernoulli switch
    /// gates an extra Normal site; the block = {switch, extra} move opens and
    /// closes the branch, and the fresh/vanished bookkeeping must reproduce the
    /// analytic posterior over the switch.
    #[test]
    #[allow(clippy::needless_borrows_for_generic_args)] // &model_fn is reused across loop iterations
    fn test_block_regen_transdimensional() {
        use crate::runtime::interpreters::PriorHandler;

        let y = 1.5;
        let p_switch = 0.3;
        let model_fn = move || {
            sample(addr!("b"), Bernoulli::new(p_switch).unwrap()).and_then(move |b| {
                if b {
                    sample(addr!("x"), Normal::new(0.0, 1.0).unwrap()).and_then(move |x| {
                        observe(addr!("y"), Normal::new(x, 1.0).unwrap(), y).map(move |_| b)
                    })
                } else {
                    observe(addr!("y"), Normal::new(0.0, 2.0).unwrap(), y).map(move |_| b)
                }
            })
        };

        // Analytic: p(y|b=1) = N(y; 0, sqrt(2)) (x marginalized), p(y|b=0) = N(y; 0, 2).
        let lik1 = normal_logpdf(y, 0.0, std::f64::consts::SQRT_2).exp();
        let lik0 = normal_logpdf(y, 0.0, 2.0).exp();
        let post_b1 = p_switch * lik1 / (p_switch * lik1 + (1.0 - p_switch) * lik0);

        let mut rng = StdRng::seed_from_u64(55);
        let (_, mut current) = run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            model_fn(),
        );
        let block = [addr!("b"), addr!("x")];
        let mut b_sum = 0.0;
        let mut n = 0.0;
        for it in 0..20000 {
            let (_, t) = block_regeneration_mh(&mut rng, &model_fn, &current, &block, 1.0);
            current = t;
            if it >= 1000 {
                b_sum += if current.get_bool(&addr!("b")).unwrap() {
                    1.0
                } else {
                    0.0
                };
                n += 1.0;
            }
        }
        let b_mean = b_sum / n;
        assert!(
            (b_mean - post_b1).abs() < 0.03,
            "P(b=1) estimate {} vs analytic {}",
            b_mean,
            post_b1
        );
    }

    /// EA-as-PPL F2 (FG-48 style): every state the block-regeneration chain
    /// returns carries accumulators equal to a fresh from-scratch re-score.
    #[test]
    #[allow(clippy::needless_borrows_for_generic_args)] // &model_fn is reused across loop iterations
    fn test_block_regen_fresh_rescore_equality() {
        use crate::runtime::interpreters::PriorHandler;

        let model_fn = || {
            sample(addr!("a"), Normal::new(0.0, 1.0).unwrap()).and_then(|a| {
                sample(addr!("b"), Normal::new(a, 1.0).unwrap()).and_then(move |b| {
                    observe(addr!("y"), Normal::new(a + b, 0.5).unwrap(), 0.7).map(move |_| (a, b))
                })
            })
        };
        let mut rng = StdRng::seed_from_u64(66);
        let (_, mut current) = run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            model_fn(),
        );
        let block = [addr!("a")];
        for _ in 0..50 {
            let (_, t) = block_regeneration_mh(&mut rng, &model_fn, &current, &block, 1.0);
            let (_, rescored) = run(
                ScoreGivenTrace {
                    base: t.clone(),
                    trace: Trace::default(),
                },
                model_fn(),
            );
            assert!(
                (t.total_log_weight() - rescored.total_log_weight()).abs() < 1e-12,
                "returned trace's accumulators diverge from a fresh re-score"
            );
            current = t;
        }
    }

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
            if let Some((_a, t, lw, _sc)) = single_site_mh_step(
                &mut rng,
                &model_fn,
                &current,
                current_lw,
                &sites,
                &mut adaptation,
                &overrides,
                true,
            ) {
                current = t;
                current_lw = lw;
            }
        }
        let scales_before = adaptation.scales.clone();

        // Sampling with adaptation frozen: scales must be untouched.
        for _ in 0..200 {
            if let Some((_a, t, lw, _sc)) = single_site_mh_step(
                &mut rng,
                &model_fn,
                &current,
                current_lw,
                &sites,
                &mut adaptation,
                &overrides,
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
