//! Built-in interpreters for different model execution modes.
//!
//! This module provides three fundamental handlers that form the building blocks
//! for inference algorithms:
//!
//! - [`PriorHandler`]: Samples fresh values from prior distributions
//! - [`ReplayHandler`]: Replays from an existing trace with fallback sampling
//! - [`ScoreGivenTrace`]: Computes log-probability of a fixed trace
//!
//! These handlers accumulate execution traces while interpreting models and are
//! the foundation for more complex inference algorithms like MCMC, SMC, and ABC.
//!
//! ## Usage Patterns
//!
//! ### Prior Sampling
//! Use `PriorHandler` to generate samples from the model's prior distribution:
///
/// ```rust
/// use fugue::*;
/// use rand::rngs::StdRng;
/// use rand::SeedableRng;
///
/// let model = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
/// let mut rng = StdRng::seed_from_u64(42);
/// let (value, trace) = runtime::handler::run(
///     PriorHandler { rng: &mut rng, trace: Trace::default() },
///     model
/// );
/// ```
///
/// ### Trace Replay
/// Use `ReplayHandler` to replay a model with values from an existing trace:
///
/// ```rust
/// use fugue::*;
/// use rand::rngs::StdRng;
/// use rand::SeedableRng;
///
/// # let existing_trace = Trace::default(); // From previous execution
/// let model = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
/// let mut rng = StdRng::seed_from_u64(42);
/// let (value, new_trace) = runtime::handler::run(
///     ReplayHandler {
///         rng: &mut rng,
///         base: existing_trace,
///         trace: Trace::default()
///     },
///     model
/// );
/// ```
///
/// ### Scoring
/// Use `ScoreGivenTrace` to compute the log-probability of a model given fixed choices:
///
/// ```rust
/// use fugue::*;
/// use rand::rngs::StdRng;
/// use rand::SeedableRng;
///
/// // Create a trace with choices first
/// let model_fn = || sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
/// let mut rng = StdRng::seed_from_u64(42);
/// let (_, existing_trace) = runtime::handler::run(
///     PriorHandler { rng: &mut rng, trace: Trace::default() },
///     model_fn()
/// );
///
/// // Now score the same model with the trace
/// let (value, score_trace) = runtime::handler::run(
///     ScoreGivenTrace {
///         base: existing_trace,
///         trace: Trace::default()
///     },
///     model_fn()
/// );
/// assert!(score_trace.total_log_weight().is_finite());
/// ```
use crate::core::address::Address;
use crate::core::distribution::Distribution;
use crate::runtime::handler::Handler;
use crate::runtime::trace::{Choice, ChoiceValue, Trace};

use rand::RngCore;

/// Handler for prior sampling - generates fresh random values from distributions.
///
/// This handler implements the standard "forward sampling" interpretation of probabilistic
/// models. When encountering sampling sites, it draws fresh random values from the
/// specified distributions. Observations contribute their log-probabilities to the likelihood,
/// and factors are accumulated directly.
///
/// This is the most basic handler and is often used as a building block for more
/// sophisticated inference algorithms.
///
/// # Fields
///
/// * `rng` - Random number generator for sampling
/// * `trace` - Trace to accumulate choices and log-weights
///
/// # Examples
///
/// ```rust
/// use fugue::*;
/// use rand::rngs::StdRng;
/// use rand::SeedableRng;
///
/// let model = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
///     .bind(|x| observe(addr!("y"), Normal::new(x, 0.5).unwrap(), 1.2).map(move |_| x));
///
/// let mut rng = StdRng::seed_from_u64(123);
/// let (result, trace) = runtime::handler::run(
///     PriorHandler {
///         rng: &mut rng,
///         trace: Trace::default(),
///     },
///     model,
/// );
///
/// println!("Sampled x: {}", result);
/// println!("Log-likelihood: {}", trace.log_likelihood);
/// assert!(result.is_finite());
/// ```
pub struct PriorHandler<'r, R: RngCore> {
    /// Random number generator for sampling.
    pub rng: &'r mut R,
    /// Trace to accumulate execution history.
    pub trace: Trace,
}

impl<'r, R: RngCore> Handler for PriorHandler<'r, R> {
    fn on_sample_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>) -> f64 {
        let x = dist.sample(self.rng);
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

    fn on_sample_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>) -> bool {
        let x = dist.sample(self.rng);
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
        let x = dist.sample(self.rng);
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
        let x = dist.sample(self.rng);
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

    fn on_observe_f64(&mut self, _: &Address, dist: &dyn Distribution<f64>, value: f64) {
        self.trace.log_likelihood += dist.log_prob(&value);
    }

    fn on_observe_bool(&mut self, _: &Address, dist: &dyn Distribution<bool>, value: bool) {
        self.trace.log_likelihood += dist.log_prob(&value);
    }

    fn on_observe_u64(&mut self, _: &Address, dist: &dyn Distribution<u64>, value: u64) {
        self.trace.log_likelihood += dist.log_prob(&value);
    }

    fn on_observe_usize(&mut self, _: &Address, dist: &dyn Distribution<usize>, value: usize) {
        self.trace.log_likelihood += dist.log_prob(&value);
    }

    fn on_factor(&mut self, logw: f64) {
        self.trace.log_factors += logw;
    }

    fn finish(self) -> Trace {
        self.trace
    }
}

/// Handler for replaying models with values from an existing trace.
///
/// This handler replays a model execution using values stored in a base trace.
/// When a sampling site is encountered:
/// - If the address exists in the base trace, use that value
/// - If the address is missing, sample a fresh value from the distribution
///
/// This is essential for MCMC algorithms where you want to replay most of a trace
/// but sample new values at specific sites that are being updated.
///
/// # Fields
///
/// * `rng` - Random number generator for sampling at missing addresses
/// * `base` - Existing trace containing values to replay
/// * `trace` - New trace to accumulate the replay execution
///
/// # Examples
///
/// ```rust
/// use fugue::*;
/// use rand::rngs::StdRng;
/// use rand::SeedableRng;
///
/// // First, create a base trace
/// let model_fn = || sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
/// let mut rng = StdRng::seed_from_u64(123);
/// let (original_value, base_trace) = runtime::handler::run(
///     PriorHandler { rng: &mut rng, trace: Trace::default() },
///     model_fn()
/// );
///
/// // Now replay the model using the base trace
/// let mut rng2 = StdRng::seed_from_u64(456);
/// let (replayed_value, new_trace) = runtime::handler::run(
///     ReplayHandler {
///         rng: &mut rng2,
///         base: base_trace,
///         trace: Trace::default(),
///     },
///     model_fn(),
/// );
/// // replayed_value will be the same as the original value
/// assert_eq!(original_value, replayed_value);
/// ```
pub struct ReplayHandler<'r, R: RngCore> {
    /// Random number generator for sampling at addresses not in base trace.
    pub rng: &'r mut R,
    /// Base trace containing values to replay.
    pub base: Trace,
    /// New trace to accumulate the replay execution.
    pub trace: Trace,
}

impl<'r, R: RngCore> Handler for ReplayHandler<'r, R> {
    fn on_sample_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>) -> f64 {
        let x = if let Some(c) = self.base.choices.get(addr) {
            match c.value {
                ChoiceValue::F64(v) => v,
                _ => panic!("expected f64 at {}", addr),
            }
        } else {
            dist.sample(self.rng)
        };
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

    fn on_sample_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>) -> bool {
        let x = if let Some(c) = self.base.choices.get(addr) {
            match c.value {
                ChoiceValue::Bool(v) => v,
                _ => panic!("expected bool at {}", addr),
            }
        } else {
            dist.sample(self.rng)
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
        let x = if let Some(c) = self.base.choices.get(addr) {
            match c.value {
                ChoiceValue::U64(v) => v,
                _ => panic!("expected u64 at {}", addr),
            }
        } else {
            dist.sample(self.rng)
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
        let x = if let Some(c) = self.base.choices.get(addr) {
            match c.value {
                ChoiceValue::Usize(v) => v,
                _ => panic!("expected usize at {}", addr),
            }
        } else {
            dist.sample(self.rng)
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

    fn on_observe_f64(&mut self, _: &Address, dist: &dyn Distribution<f64>, value: f64) {
        self.trace.log_likelihood += dist.log_prob(&value);
    }

    fn on_observe_bool(&mut self, _: &Address, dist: &dyn Distribution<bool>, value: bool) {
        self.trace.log_likelihood += dist.log_prob(&value);
    }

    fn on_observe_u64(&mut self, _: &Address, dist: &dyn Distribution<u64>, value: u64) {
        self.trace.log_likelihood += dist.log_prob(&value);
    }

    fn on_observe_usize(&mut self, _: &Address, dist: &dyn Distribution<usize>, value: usize) {
        self.trace.log_likelihood += dist.log_prob(&value);
    }

    fn on_factor(&mut self, logw: f64) {
        self.trace.log_factors += logw;
    }

    fn finish(self) -> Trace {
        self.trace
    }
}

/// Handler for scoring a model given a complete trace of choices.
///
/// This handler computes the log-probability of a model execution where all
/// random choices are fixed by an existing trace. It does not perform any
/// sampling - instead, it looks up values from the base trace and computes
/// their log-probabilities under the current model's distributions.
///
/// This is essential for:
/// - Computing proposal densities in MCMC
/// - Importance weighting in particle filters
/// - Model comparison and Bayes factors
///
/// # Fields
///
/// * `base` - Trace containing the fixed choices to score
/// * `trace` - New trace to accumulate log-probabilities
///
/// # Panics
///
/// Panics if the base trace is missing a value for any sampling site encountered
/// during execution. The base trace must be complete for the model being scored.
///
/// # Examples
///
/// ```rust
/// use fugue::*;
/// use rand::rngs::StdRng;
/// use rand::SeedableRng;
///
/// // Create a trace with some choices
/// let model_fn = || sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
/// let mut rng = StdRng::seed_from_u64(123);
/// let (_, complete_trace) = runtime::handler::run(
///     PriorHandler { rng: &mut rng, trace: Trace::default() },
///     model_fn()
/// );
///
/// // Score the model under different parameters
/// let different_model_fn = || sample(addr!("x"), Normal::new(1.0, 2.0).unwrap());
/// let (value, score_trace) = runtime::handler::run(
///     ScoreGivenTrace {
///         base: complete_trace,
///         trace: Trace::default(),
///     },
///     different_model_fn(),
/// );
///
/// assert!(score_trace.total_log_weight().is_finite());
/// ```
pub struct ScoreGivenTrace {
    /// Base trace containing the fixed choices to score.
    pub base: Trace,
    /// New trace to accumulate log-probabilities.
    pub trace: Trace,
}

/// Safe version of ReplayHandler that uses type-safe trace accessors.
///
/// This handler replays a model execution using values stored in a base trace,
/// but gracefully handles type mismatches by logging warnings and falling back
/// to fresh sampling instead of panicking.
///
/// # Examples
///
/// ```rust
/// use fugue::*;
/// use rand::rngs::StdRng;
/// use rand::SeedableRng;
///
/// let base_trace = Trace::default(); // From previous execution
/// let mut rng = StdRng::seed_from_u64(42);
/// let handler = SafeReplayHandler {
///     rng: &mut rng,
///     base: base_trace,
///     trace: Trace::default(),
///     warn_on_mismatch: true,
/// };
/// ```
pub struct SafeReplayHandler<'r, R: RngCore> {
    /// Random number generator for sampling at addresses not in base trace.
    pub rng: &'r mut R,
    /// Base trace containing values to replay.
    pub base: Trace,
    /// New trace to accumulate the replay execution.
    pub trace: Trace,
    /// Whether to log warnings on type mismatches (useful for debugging).
    pub warn_on_mismatch: bool,
}

impl<'r, R: RngCore> Handler for SafeReplayHandler<'r, R> {
    fn on_sample_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>) -> f64 {
        let x = match self.base.get_f64(addr) {
            Some(v) => v,
            None => {
                if self.warn_on_mismatch && self.base.choices.contains_key(addr) {
                    if let Some(choice) = self.base.choices.get(addr) {
                        eprintln!(
                            "Warning: Type mismatch at {}: expected f64, found {}",
                            addr,
                            choice.value.type_name()
                        );
                    }
                }
                dist.sample(self.rng)
            }
        };
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

    fn on_sample_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>) -> bool {
        let x = match self.base.get_bool(addr) {
            Some(v) => v,
            None => {
                if self.warn_on_mismatch && self.base.choices.contains_key(addr) {
                    if let Some(choice) = self.base.choices.get(addr) {
                        eprintln!(
                            "Warning: Type mismatch at {}: expected bool, found {}",
                            addr,
                            choice.value.type_name()
                        );
                    }
                }
                dist.sample(self.rng)
            }
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
        let x = match self.base.get_u64(addr) {
            Some(v) => v,
            None => {
                if self.warn_on_mismatch && self.base.choices.contains_key(addr) {
                    if let Some(choice) = self.base.choices.get(addr) {
                        eprintln!(
                            "Warning: Type mismatch at {}: expected u64, found {}",
                            addr,
                            choice.value.type_name()
                        );
                    }
                }
                dist.sample(self.rng)
            }
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
        let x = match self.base.get_usize(addr) {
            Some(v) => v,
            None => {
                if self.warn_on_mismatch && self.base.choices.contains_key(addr) {
                    if let Some(choice) = self.base.choices.get(addr) {
                        eprintln!(
                            "Warning: Type mismatch at {}: expected usize, found {}",
                            addr,
                            choice.value.type_name()
                        );
                    }
                }
                dist.sample(self.rng)
            }
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

    fn on_observe_f64(&mut self, _: &Address, dist: &dyn Distribution<f64>, value: f64) {
        self.trace.log_likelihood += dist.log_prob(&value);
    }

    fn on_observe_bool(&mut self, _: &Address, dist: &dyn Distribution<bool>, value: bool) {
        self.trace.log_likelihood += dist.log_prob(&value);
    }

    fn on_observe_u64(&mut self, _: &Address, dist: &dyn Distribution<u64>, value: u64) {
        self.trace.log_likelihood += dist.log_prob(&value);
    }

    fn on_observe_usize(&mut self, _: &Address, dist: &dyn Distribution<usize>, value: usize) {
        self.trace.log_likelihood += dist.log_prob(&value);
    }

    fn on_factor(&mut self, logw: f64) {
        self.trace.log_factors += logw;
    }

    fn finish(self) -> Trace {
        self.trace
    }
}

/// Safe version of ScoreGivenTrace that uses type-safe trace accessors.
///
/// This handler computes log-probability of a model execution but gracefully
/// handles missing addresses or type mismatches by returning negative infinity
/// log-weight instead of panicking.
pub struct SafeScoreGivenTrace {
    /// Base trace containing the fixed choices to score.
    pub base: Trace,
    /// New trace to accumulate log-probabilities.
    pub trace: Trace,
    /// Whether to log warnings on missing addresses or type mismatches.
    pub warn_on_error: bool,
}

impl Handler for SafeScoreGivenTrace {
    fn on_sample_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>) -> f64 {
        match self.base.get_f64_result(addr) {
            Ok(x) => {
                let lp = dist.log_prob(&x);
                self.trace.log_prior += lp;
                if let Some(choice) = self.base.choices.get(addr) {
                    self.trace.choices.insert(addr.clone(), choice.clone());
                }
                x
            }
            Err(e) => {
                if self.warn_on_error {
                    eprintln!("Warning: Failed to get f64 at {}: {}", addr, e);
                }
                // Add negative infinity to make this trace invalid
                self.trace.log_prior += f64::NEG_INFINITY;
                0.0 // Return a dummy value
            }
        }
    }

    fn on_sample_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>) -> bool {
        match self.base.get_bool_result(addr) {
            Ok(x) => {
                let lp = dist.log_prob(&x);
                self.trace.log_prior += lp;
                if let Some(choice) = self.base.choices.get(addr) {
                    self.trace.choices.insert(addr.clone(), choice.clone());
                }
                x
            }
            Err(e) => {
                if self.warn_on_error {
                    eprintln!("Warning: Failed to get bool at {}: {}", addr, e);
                }
                self.trace.log_prior += f64::NEG_INFINITY;
                false
            }
        }
    }

    fn on_sample_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>) -> u64 {
        match self.base.get_u64_result(addr) {
            Ok(x) => {
                let lp = dist.log_prob(&x);
                self.trace.log_prior += lp;
                if let Some(choice) = self.base.choices.get(addr) {
                    self.trace.choices.insert(addr.clone(), choice.clone());
                }
                x
            }
            Err(e) => {
                if self.warn_on_error {
                    eprintln!("Warning: Failed to get u64 at {}: {}", addr, e);
                }
                self.trace.log_prior += f64::NEG_INFINITY;
                0
            }
        }
    }

    fn on_sample_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>) -> usize {
        match self.base.get_usize_result(addr) {
            Ok(x) => {
                let lp = dist.log_prob(&x);
                self.trace.log_prior += lp;
                if let Some(choice) = self.base.choices.get(addr) {
                    self.trace.choices.insert(addr.clone(), choice.clone());
                }
                x
            }
            Err(e) => {
                if self.warn_on_error {
                    eprintln!("Warning: Failed to get usize at {}: {}", addr, e);
                }
                self.trace.log_prior += f64::NEG_INFINITY;
                0
            }
        }
    }

    fn on_observe_f64(&mut self, _: &Address, dist: &dyn Distribution<f64>, value: f64) {
        self.trace.log_likelihood += dist.log_prob(&value);
    }

    fn on_observe_bool(&mut self, _: &Address, dist: &dyn Distribution<bool>, value: bool) {
        self.trace.log_likelihood += dist.log_prob(&value);
    }

    fn on_observe_u64(&mut self, _: &Address, dist: &dyn Distribution<u64>, value: u64) {
        self.trace.log_likelihood += dist.log_prob(&value);
    }

    fn on_observe_usize(&mut self, _: &Address, dist: &dyn Distribution<usize>, value: usize) {
        self.trace.log_likelihood += dist.log_prob(&value);
    }

    fn on_factor(&mut self, logw: f64) {
        self.trace.log_factors += logw;
    }

    fn finish(self) -> Trace {
        self.trace
    }
}

impl Handler for ScoreGivenTrace {
    fn on_sample_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>) -> f64 {
        let c = self
            .base
            .choices
            .get(addr)
            .unwrap_or_else(|| panic!("missing value for site {} in base trace", addr));
        let x = match c.value {
            ChoiceValue::F64(v) => v,
            _ => panic!("expected f64 at {}", addr),
        };
        let lp = dist.log_prob(&x);
        self.trace.log_prior += lp;
        self.trace.choices.insert(addr.clone(), c.clone());
        x
    }

    fn on_sample_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>) -> bool {
        let c = self
            .base
            .choices
            .get(addr)
            .unwrap_or_else(|| panic!("missing value for site {} in base trace", addr));
        let x = match c.value {
            ChoiceValue::Bool(v) => v,
            _ => panic!("expected bool at {}", addr),
        };
        let lp = dist.log_prob(&x);
        self.trace.log_prior += lp;
        self.trace.choices.insert(addr.clone(), c.clone());
        x
    }

    fn on_sample_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>) -> u64 {
        let c = self
            .base
            .choices
            .get(addr)
            .unwrap_or_else(|| panic!("missing value for site {} in base trace", addr));
        let x = match c.value {
            ChoiceValue::U64(v) => v,
            _ => panic!("expected u64 at {}", addr),
        };
        let lp = dist.log_prob(&x);
        self.trace.log_prior += lp;
        self.trace.choices.insert(addr.clone(), c.clone());
        x
    }

    fn on_sample_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>) -> usize {
        let c = self
            .base
            .choices
            .get(addr)
            .unwrap_or_else(|| panic!("missing value for site {} in base trace", addr));
        let x = match c.value {
            ChoiceValue::Usize(v) => v,
            _ => panic!("expected usize at {}", addr),
        };
        let lp = dist.log_prob(&x);
        self.trace.log_prior += lp;
        self.trace.choices.insert(addr.clone(), c.clone());
        x
    }

    fn on_observe_f64(&mut self, _: &Address, dist: &dyn Distribution<f64>, value: f64) {
        self.trace.log_likelihood += dist.log_prob(&value);
    }

    fn on_observe_bool(&mut self, _: &Address, dist: &dyn Distribution<bool>, value: bool) {
        self.trace.log_likelihood += dist.log_prob(&value);
    }

    fn on_observe_u64(&mut self, _: &Address, dist: &dyn Distribution<u64>, value: u64) {
        self.trace.log_likelihood += dist.log_prob(&value);
    }

    fn on_observe_usize(&mut self, _: &Address, dist: &dyn Distribution<usize>, value: usize) {
        self.trace.log_likelihood += dist.log_prob(&value);
    }

    fn on_factor(&mut self, logw: f64) {
        self.trace.log_factors += logw;
    }

    fn finish(self) -> Trace {
        self.trace
    }
}
