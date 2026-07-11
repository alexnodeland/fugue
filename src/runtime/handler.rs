#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/docs/runtime/handler.md"))]

use crate::core::address::Address;
use crate::core::distribution::Distribution;
use crate::core::model::Model;
use crate::runtime::trace::Trace;

/// Core trait for interpreting probabilistic model effects.
///
/// Handlers define how to interpret the three fundamental effects in probabilistic programming:
/// sampling, observation, and factoring. Different implementations enable different execution modes.
///
/// Example:
/// ```rust
/// # use fugue::*;
/// # use fugue::runtime::interpreters::PriorHandler;
/// # use rand::rngs::StdRng;
/// # use rand::SeedableRng;
///
/// // Use a built-in handler
/// let mut rng = StdRng::seed_from_u64(42);
/// let handler = PriorHandler {
///     rng: &mut rng,
///     trace: Trace::default()
/// };
/// let model = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
/// let (result, trace) = runtime::handler::run(handler, model);
/// ```
pub trait Handler {
    /// Handle an f64 sampling operation (continuous distributions).
    fn on_sample_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>) -> f64;

    /// Handle a bool sampling operation (Bernoulli).
    fn on_sample_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>) -> bool;

    /// Handle a u64 sampling operation (Poisson, Binomial).
    fn on_sample_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>) -> u64;

    /// Handle a usize sampling operation (Categorical).
    fn on_sample_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>) -> usize;

    /// Handle an i64 sampling operation (signed discrete distributions).
    ///
    /// This has a default implementation that panics so that handlers written
    /// before the i64 sample path existed keep compiling unchanged; every
    /// handler shipped in this crate overrides it. A model only reaches this
    /// method if it contains a [`Model::SampleI64`](crate::Model::SampleI64)
    /// node (e.g. a future `DiscreteUniform` distribution).
    fn on_sample_i64(&mut self, addr: &Address, _dist: &dyn Distribution<i64>) -> i64 {
        panic!(
            "handler does not implement on_sample_i64 (i64 sample site at {})",
            addr
        )
    }

    /// Handle an f64 observation operation.
    fn on_observe_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>, value: f64);

    /// Handle a bool observation operation.
    fn on_observe_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>, value: bool);

    /// Handle a u64 observation operation.
    fn on_observe_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>, value: u64);

    /// Handle a usize observation operation.
    fn on_observe_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>, value: usize);

    /// Handle an i64 observation operation.
    ///
    /// Defaults to a panic for the same forward-compatibility reason as
    /// [`Handler::on_sample_i64`]; all in-crate handlers override it.
    fn on_observe_i64(&mut self, addr: &Address, _dist: &dyn Distribution<i64>, _value: i64) {
        panic!(
            "handler does not implement on_observe_i64 (i64 observe site at {})",
            addr
        )
    }

    /// Handle a factor operation.
    ///
    /// This method is called when the model encounters a `factor` operation.
    /// The handler typically adds the log-weight to the trace.
    ///
    /// # Arguments
    ///
    /// * `logw` - Log-weight to add to the model's total weight
    fn on_factor(&mut self, logw: f64);

    /// Finalize the handler and return the accumulated trace.
    ///
    /// This method is called after model execution completes to retrieve
    /// the final trace containing all choices and log-weights.
    fn finish(self) -> Trace
    where
        Self: Sized;
}

/// Execute a probabilistic model using the provided handler.
///
/// This is the core execution engine for probabilistic models. It walks through
/// the model structure and dispatches effects to the handler, returning both
/// the model's final result and the accumulated execution trace.
///
/// Example:
/// ```rust
/// # use fugue::*;
/// # use fugue::runtime::interpreters::PriorHandler;
/// # use rand::rngs::StdRng;
/// # use rand::SeedableRng;
///
/// // Create a simple model
/// let model = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
///     .bind(|x| observe(addr!("y"), Normal::new(x, 0.1).unwrap(), 1.2))
///     .map(|_| "completed");
///
/// let mut rng = StdRng::seed_from_u64(123);
/// let (result, trace) = runtime::handler::run(
///     PriorHandler { rng: &mut rng, trace: Trace::default() },
///     model
/// );
/// assert_eq!(result, "completed");
/// assert!(trace.total_log_weight().is_finite());
/// ```
pub fn run<A>(mut h: impl Handler, m: Model<A>) -> (A, Trace) {
    // Iterative trampoline (FG-19): the model is a CPS-encoded linked list of
    // continuations, so we interpret it in an explicit loop instead of the old
    // `go(h, k(x))` recursion. This keeps interpretation O(1) in stack depth
    // regardless of model length, so deep chains (e.g. `plate!`/`sequence_vec`
    // over thousands of sites, or a 100k-deep sample+bind loop) no longer
    // overflow the stack. Each effectful node advances `m` to its continuation
    // `k(value)` and loops; only `Model::Pure` terminates.
    let mut m = m;
    let a = loop {
        m = match m {
            Model::Pure(a) => break a,
            Model::SampleF64 { addr, dist, k } => {
                let x = h.on_sample_f64(&addr, &*dist);
                k(x)
            }
            Model::SampleBool { addr, dist, k } => {
                let x = h.on_sample_bool(&addr, &*dist);
                k(x)
            }
            Model::SampleU64 { addr, dist, k } => {
                let x = h.on_sample_u64(&addr, &*dist);
                k(x)
            }
            Model::SampleUsize { addr, dist, k } => {
                let x = h.on_sample_usize(&addr, &*dist);
                k(x)
            }
            Model::SampleI64 { addr, dist, k } => {
                let x = h.on_sample_i64(&addr, &*dist);
                k(x)
            }
            Model::ObserveF64 {
                addr,
                dist,
                value,
                k,
            } => {
                h.on_observe_f64(&addr, &*dist, value);
                k(())
            }
            Model::ObserveBool {
                addr,
                dist,
                value,
                k,
            } => {
                h.on_observe_bool(&addr, &*dist, value);
                k(())
            }
            Model::ObserveU64 {
                addr,
                dist,
                value,
                k,
            } => {
                h.on_observe_u64(&addr, &*dist, value);
                k(())
            }
            Model::ObserveUsize {
                addr,
                dist,
                value,
                k,
            } => {
                h.on_observe_usize(&addr, &*dist, value);
                k(())
            }
            Model::ObserveI64 {
                addr,
                dist,
                value,
                k,
            } => {
                h.on_observe_i64(&addr, &*dist, value);
                k(())
            }
            Model::Factor { logw, k } => {
                h.on_factor(logw);
                k(())
            }
        };
    };
    let t = h.finish();
    (a, t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::addr;
    use crate::core::distribution::*;
    use crate::core::model::ModelExt;
    use crate::runtime::interpreters::PriorHandler;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn run_accumulates_logs_for_sample_observe_factor() {
        // Model: sample x ~ Normal(0,1); observe y ~ Normal(x,1) with value 0.5; factor(-1.0)
        let model = crate::core::model::sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
            .and_then(|x| {
                crate::core::model::observe(addr!("y"), Normal::new(x, 1.0).unwrap(), 0.5)
            })
            .and_then(|_| crate::core::model::factor(-1.0));

        let mut rng = StdRng::seed_from_u64(123);
        let (_a, trace) = crate::runtime::handler::run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            model,
        );

        // Should have a sample recorded and finite prior
        assert!(trace.choices.contains_key(&addr!("x")));
        assert!(trace.log_prior.is_finite());
        // Observation contributes to likelihood
        assert!(trace.log_likelihood.is_finite());
        // Factor contributes exact -1.0
        assert!((trace.log_factors + 1.0).abs() < 1e-12);
    }

    // Regression for FG-19: interpretation must be stack-safe. Before the
    // trampoline, `run` recursed once per effectful node (`go(h, k(x))`), so a
    // deep sample+bind chain overflowed the stack. This model is a loop of
    // 100_000 sequential `sample`+`bind` sites (the accumulator is threaded as a
    // plain parameter so each continuation directly yields the next node); it
    // overflows the stack on the pre-fix recursive interpreter and completes in
    // O(1) stack on the trampoline. Runs on a small-stack thread to make the
    // guarantee explicit rather than relying on the test harness's stack size.
    #[test]
    fn interpretation_is_stack_safe_for_deep_models() {
        fn build(i: usize, n: usize, acc: f64) -> Model<f64> {
            if i >= n {
                crate::core::model::pure(acc)
            } else {
                crate::core::model::sample(addr!("x", i), Normal::new(0.0, 1.0).unwrap())
                    .bind(move |x| build(i + 1, n, acc + x))
            }
        }

        // 512 KiB stack: comfortably too small for 100_000 recursive frames,
        // but ample for the constant-stack trampoline.
        let handle = std::thread::Builder::new()
            .stack_size(512 * 1024)
            .spawn(|| {
                let n = 100_000;
                let mut rng = StdRng::seed_from_u64(2024);
                let (sum, trace) = crate::runtime::handler::run(
                    PriorHandler {
                        rng: &mut rng,
                        trace: Trace::default(),
                    },
                    build(0, n, 0.0),
                );
                assert!(sum.is_finite());
                assert_eq!(trace.choices.len(), n);
                assert!(trace.log_prior.is_finite());
            })
            .expect("spawn thread");
        handle
            .join()
            .expect("deep model interpretation overflowed the stack");
    }
}
