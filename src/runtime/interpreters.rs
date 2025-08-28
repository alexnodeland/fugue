#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/docs/api/runtime/interpreters.md"))]

use crate::core::address::Address;
use crate::core::distribution::Distribution;
use crate::runtime::handler::Handler;
use crate::runtime::trace::{Choice, ChoiceValue, Trace};

use rand::RngCore;

/// Handler for prior sampling - generates fresh random values from distributions.
///
/// This is the foundational interpreter that implements standard "forward sampling"
/// from probabilistic models. It draws fresh values from distributions and accumulates
/// log-probabilities in the trace.
///
/// Example:
/// ```rust
/// # use fugue::*;
/// # use fugue::runtime::interpreters::PriorHandler;
/// # use rand::rngs::StdRng;
/// # use rand::SeedableRng;
///
/// let model = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
///     .bind(|x| observe(addr!("y"), Normal::new(x, 0.5).unwrap(), 1.2)
///         .map(move |_| x));
///
/// let mut rng = StdRng::seed_from_u64(42);
/// let (result, trace) = runtime::handler::run(
///     PriorHandler { rng: &mut rng, trace: Trace::default() },
///     model
/// );
///
/// assert!(result.is_finite());
/// assert!(trace.log_likelihood.is_finite());
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
/// ReplayHandler replays a model execution using stored trace values. When a sampling
/// site is encountered: if the address exists in the base trace, use that value;
/// if missing, sample fresh. Essential for MCMC where you replay most choices
/// but sample new values at specific sites.
///
/// Example:
/// ```rust
/// # use fugue::*;
/// # use fugue::runtime::interpreters::*;
/// # use rand::rngs::StdRng;
/// # use rand::SeedableRng;
///
/// // Create base trace
/// let mut rng = StdRng::seed_from_u64(42);
/// let model_fn = || sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
/// let (original, base_trace) = runtime::handler::run(
///     PriorHandler { rng: &mut rng, trace: Trace::default() },
///     model_fn()
/// );
///
/// // Replay using base trace values
/// let (replayed, _) = runtime::handler::run(
///     ReplayHandler {
///         rng: &mut rng,
///         base: base_trace,
///         trace: Trace::default()
///     },
///     model_fn()
/// );
///
/// assert_eq!(original, replayed); // Same value replayed
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

/// Handler for scoring a model given a complete trace of fixed choices.
///
/// ScoreGivenTrace computes log-probability of a model execution where all random
/// choices are predetermined. No sampling occurs - values are looked up from the
/// base trace and their log-probabilities computed. Essential for MCMC acceptance
/// ratios, importance sampling, and model comparison.
///
/// Example:
/// ```rust
/// # use fugue::*;
/// # use fugue::runtime::interpreters::*;
/// # use rand::rngs::StdRng;
/// # use rand::SeedableRng;
///
/// // Create a complete trace
/// let mut rng = StdRng::seed_from_u64(42);
/// let (_, complete_trace) = runtime::handler::run(
///     PriorHandler { rng: &mut rng, trace: Trace::default() },
///     sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
/// );
///
/// // Score under different model parameters  
/// let (value, score_trace) = runtime::handler::run(
///     ScoreGivenTrace {
///         base: complete_trace,
///         trace: Trace::default()
///     },
///     sample(addr!("x"), Normal::new(1.0, 2.0).unwrap()) // Different parameters
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

/// Safe version of ReplayHandler that gracefully handles trace inconsistencies.
///
/// SafeReplayHandler replays model execution like ReplayHandler, but handles type
/// mismatches and missing addresses gracefully by logging warnings and sampling
/// fresh values instead of panicking. Essential for production systems where
/// trace consistency cannot be guaranteed.
///
/// Example:
/// ```rust
/// # use fugue::*;
/// # use fugue::runtime::interpreters::*;
/// # use rand::rngs::StdRng;
/// # use rand::SeedableRng;
///
/// // Create trace with potential inconsistencies
/// let mut rng = StdRng::seed_from_u64(42);
/// let (_, base_trace) = runtime::handler::run(
///     PriorHandler { rng: &mut rng, trace: Trace::default() },
///     sample(addr!("x"), Normal::new(0.0, 1.0).unwrap()) // f64 value
/// );
///
/// // Safe replay handles type mismatch gracefully
/// let (result, trace) = runtime::handler::run(
///     SafeReplayHandler {
///         rng: &mut rng,
///         base: base_trace,
///         trace: Trace::default(),
///         warn_on_mismatch: true, // Enable warnings
///     },
///     sample(addr!("x"), Bernoulli::new(0.5).unwrap()) // Expects bool
/// );
///
/// assert!(trace.total_log_weight().is_finite()); // Continues execution
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

/// Safe version of ScoreGivenTrace that gracefully handles incomplete traces.
///
/// SafeScoreGivenTrace computes log-probability like ScoreGivenTrace, but handles
/// missing addresses or type mismatches by returning negative infinity log-weight
/// instead of panicking. Essential for production inference where trace validity
/// cannot be guaranteed.
///
/// Example:
/// ```rust
/// # use fugue::*;
/// # use fugue::runtime::interpreters::*;
/// # use rand::rngs::StdRng;
/// # use rand::SeedableRng;
///
/// // Create incomplete trace
/// let mut rng = StdRng::seed_from_u64(42);
/// let (_, incomplete_trace) = runtime::handler::run(
///     PriorHandler { rng: &mut rng, trace: Trace::default() },
///     sample(addr!("x"), Normal::new(0.0, 1.0).unwrap()) // Only has "x"
/// );
///
/// // Safe scoring handles missing address gracefully
/// let (_, score_trace) = runtime::handler::run(
///     SafeScoreGivenTrace {
///         base: incomplete_trace,
///         trace: Trace::default(),
///         warn_on_error: true, // Enable warnings
///     },
///     sample(addr!("missing"), Normal::new(0.0, 1.0).unwrap()) // Address not in base
/// );
///
/// assert_eq!(score_trace.total_log_weight(), f64::NEG_INFINITY); // Graceful failure
/// ```
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::addr;
    use crate::core::distribution::*;
    use crate::core::model::{observe, sample, ModelExt};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn prior_handler_samples_and_accumulates() {
        let mut rng = StdRng::seed_from_u64(7);
        let (_val, trace) = crate::runtime::handler::run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
                .and_then(|x| observe(addr!("y"), Normal::new(x, 1.0).unwrap(), 0.5)),
        );
        assert!(trace.choices.contains_key(&addr!("x")));
        assert!(trace.log_prior.is_finite());
        assert!(trace.log_likelihood.is_finite());
    }

    #[test]
    fn replay_handler_reuses_values() {
        let mut rng = StdRng::seed_from_u64(8);
        let ((), base) = crate::runtime::handler::run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            sample(addr!("x"), Normal::new(0.0, 1.0).unwrap()).map(|_| ()),
        );

        let ((), replayed) = crate::runtime::handler::run(
            ReplayHandler {
                rng: &mut rng,
                base: base.clone(),
                trace: Trace::default(),
            },
            sample(addr!("x"), Normal::new(0.0, 1.0).unwrap()).map(|_| ()),
        );

        let x_base = base.get_f64(&addr!("x")).unwrap();
        let x_replay = replayed.get_f64(&addr!("x")).unwrap();
        assert_eq!(x_base, x_replay);
    }

    #[test]
    fn score_given_trace_scores_fixed_values() {
        let mut rng = StdRng::seed_from_u64(9);
        let (_a, base) = crate::runtime::handler::run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            sample(addr!("x"), Normal::new(0.0, 1.0).unwrap()),
        );

        let (_a2, scored) = crate::runtime::handler::run(
            ScoreGivenTrace {
                base: base.clone(),
                trace: Trace::default(),
            },
            sample(addr!("x"), Normal::new(0.0, 1.0).unwrap()),
        );

        // Should have same value and finite log_prior
        assert_eq!(scored.get_f64(&addr!("x")), base.get_f64(&addr!("x")));
        assert!(scored.log_prior.is_finite());
    }

    #[test]
    fn safe_variants_handle_mismatches() {
        // Build base trace with x as f64, then attempt to replay as bool
        let mut rng = StdRng::seed_from_u64(10);
        let (_a, base) = crate::runtime::handler::run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            sample(addr!("x"), Normal::new(0.0, 1.0).unwrap()),
        );

        // SafeReplayHandler should sample fresh value for bool and continue
        let (_b, t1) = crate::runtime::handler::run(
            SafeReplayHandler {
                rng: &mut rng,
                base: base.clone(),
                trace: Trace::default(),
                warn_on_mismatch: true,
            },
            sample(addr!("x"), Bernoulli::new(0.5).unwrap()),
        );
        assert!(t1.log_prior.is_finite());

        // SafeScoreGivenTrace should mark as invalid by adding -inf
        let (_c, t2) = crate::runtime::handler::run(
            SafeScoreGivenTrace {
                base: base.clone(),
                trace: Trace::default(),
                warn_on_error: true,
            },
            sample(addr!("x"), Bernoulli::new(0.5).unwrap()),
        );
        assert!(t2.log_prior.is_infinite());
    }

    #[test]
    fn handlers_cover_all_types_sample_and_observe() {
        // Model with multiple types
        let model = sample(addr!("f"), Normal::new(0.0, 1.0).unwrap())
            .and_then(|_| sample(addr!("b"), Bernoulli::new(0.6).unwrap()))
            .and_then(|_| sample(addr!("u64"), Poisson::new(3.0).unwrap()))
            .and_then(|_| sample(addr!("usz"), Categorical::new(vec![0.3, 0.7]).unwrap()))
            .and_then(|_| observe(addr!("f_obs"), Normal::new(0.0, 1.0).unwrap(), 0.1))
            .and_then(|_| observe(addr!("b_obs"), Bernoulli::new(0.4).unwrap(), true))
            .and_then(|_| observe(addr!("u64_obs"), Poisson::new(2.0).unwrap(), 1))
            .and_then(|_| {
                observe(
                    addr!("usz_obs"),
                    Categorical::new(vec![0.5, 0.5]).unwrap(),
                    1,
                )
            });

        let (_a, t) = crate::runtime::handler::run(
            PriorHandler {
                rng: &mut StdRng::seed_from_u64(100),
                trace: Trace::default(),
            },
            model,
        );
        assert!(t.get_f64(&addr!("f")).is_some());
        assert!(t.get_bool(&addr!("b")).is_some());
        assert!(t.get_u64(&addr!("u64")).is_some());
        assert!(t.get_usize(&addr!("usz")).is_some());
        assert!(t.log_likelihood.is_finite());

        // Build base and score given trace for all types
        let base = t.clone();
        let (_sv, scored) = crate::runtime::handler::run(
            ScoreGivenTrace {
                base: base.clone(),
                trace: Trace::default(),
            },
            sample(addr!("f"), Normal::new(0.0, 1.0).unwrap())
                .and_then(|_| sample(addr!("b"), Bernoulli::new(0.6).unwrap()))
                .and_then(|_| sample(addr!("u64"), Poisson::new(3.0).unwrap()))
                .and_then(|_| sample(addr!("usz"), Categorical::new(vec![0.3, 0.7]).unwrap())),
        );
        assert!(scored.log_prior.is_finite());

        // Safe replay mismatches for integer/categorical types
        let (_sv2, safe) = crate::runtime::handler::run(
            SafeReplayHandler {
                rng: &mut StdRng::seed_from_u64(101),
                base: base.clone(),
                trace: Trace::default(),
                warn_on_mismatch: true,
            },
            sample(addr!("u64"), Bernoulli::new(0.5).unwrap()),
        );
        assert!(safe.log_prior.is_finite());
    }

    #[test]
    fn safe_score_given_trace_warn_flag_branches() {
        let mut rng = StdRng::seed_from_u64(102);
        let (_a, base) = crate::runtime::handler::run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            sample(addr!("x"), Normal::new(0.0, 1.0).unwrap()),
        );
        // warn_on_error = false
        let (_b, t_false) = crate::runtime::handler::run(
            SafeScoreGivenTrace {
                base: base.clone(),
                trace: Trace::default(),
                warn_on_error: false,
            },
            sample(addr!("x"), Bernoulli::new(0.5).unwrap()),
        );
        assert!(t_false.log_prior.is_infinite());

        // warn_on_error = true
        let (_c, t_true) = crate::runtime::handler::run(
            SafeScoreGivenTrace {
                base: base.clone(),
                trace: Trace::default(),
                warn_on_error: true,
            },
            sample(addr!("x"), Bernoulli::new(0.5).unwrap()),
        );
        assert!(t_true.log_prior.is_infinite());
    }

    #[test]
    #[should_panic]
    fn replay_handler_panics_on_type_mismatch() {
        // Base has f64, replay expects bool -> panic as designed
        let mut rng = StdRng::seed_from_u64(103);
        let (_a, base) = crate::runtime::handler::run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            sample(addr!("x"), Normal::new(0.0, 1.0).unwrap()),
        );
        let (_b, _t) = crate::runtime::handler::run(
            ReplayHandler {
                rng: &mut rng,
                base: base.clone(),
                trace: Trace::default(),
            },
            sample(addr!("x"), Bernoulli::new(0.5).unwrap()),
        );
    }

    #[test]
    fn safe_replay_handler_samples_fresh_for_missing_address() {
        let mut rng = StdRng::seed_from_u64(104);
        // Base trace without address "z"
        let base = Trace::default();
        let (_a, t) = crate::runtime::handler::run(
            SafeReplayHandler {
                rng: &mut rng,
                base,
                trace: Trace::default(),
                warn_on_mismatch: true,
            },
            sample(addr!("z"), Normal::new(0.0, 1.0).unwrap()),
        );
        assert!(t.get_f64(&addr!("z")).is_some());
    }
}
