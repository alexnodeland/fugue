#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/docs/runtime/async_handler.md"))]

use crate::core::address::Address;
use crate::core::distribution::Distribution;
use crate::core::model::Model;
use crate::runtime::handler::Handler;
use crate::runtime::trace::Trace;

/// Asynchronous counterpart of [`Handler`], for effects resolved by I/O.
///
/// Every method has the name, arguments and meaning of the [`Handler`] method
/// it mirrors, but is an `async fn`, so a handler can resolve a site over the
/// network (a tool call, a remote model, an LLM) and await the answer instead
/// of blocking its thread. [`run_async`] calls the methods in the order
/// [`run`](crate::runtime::handler::run) calls the synchronous ones and awaits
/// each before the model continues. The `addr` and `dist` a method receives
/// are borrowed from the model node, which `run_async` keeps alive until the
/// method's future completes, so both stay usable after an `.await`.
///
/// Any synchronous handler is an `AsyncHandler` through [`FromSync`]. A
/// complete custom handler is in the
/// [module documentation](crate::runtime::async_handler).
///
/// # `Send`
///
/// The methods are declared with `async fn`, so the trait does not bound their
/// futures by `Send`, and generic code over `H: AsyncHandler` cannot require
/// that they are (stable Rust cannot yet name that bound; return-type
/// notation, rust-lang/rust#109417, is unstable). For a concrete handler type
/// the compiler sees the actual futures, because auto traits leak through
/// `async fn`: if the handler is `Send`, none of its methods holds anything
/// else `!Send` (an `Rc`, a `RefCell` borrow) across an `.await`, and the
/// model's value type is `Send`, then `run_async(handler, model)` is `Send`.
/// A handler that also borrows nothing makes it `'static`, so the run can be
/// spawned on a multi-threaded executor. The trait stays usable by handlers
/// that are not `Send`, which run on a single-threaded executor instead.
///
/// Example:
/// ```rust
/// # use fugue::*;
/// # use fugue::runtime::interpreters::PriorHandler;
/// # use rand::rngs::StdRng;
/// # use rand::SeedableRng;
///
/// // Use a built-in handler, through the synchronous adapter
/// let mut rng = StdRng::seed_from_u64(42);
/// let handler = FromSync(PriorHandler {
///     rng: &mut rng,
///     trace: Trace::default(),
/// });
/// let model = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
///
/// // Any executor can drive the future; this one is tokio's.
/// let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
/// let (x, trace) = rt.block_on(run_async(handler, model));
/// assert_eq!(trace.get_f64(&addr!("x")), Some(x));
/// ```
#[allow(async_fn_in_trait)]
pub trait AsyncHandler {
    /// Handle an f64 sampling operation (continuous distributions).
    async fn on_sample_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>) -> f64;

    /// Handle a bool sampling operation (Bernoulli).
    async fn on_sample_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>) -> bool;

    /// Handle a u64 sampling operation (Poisson, Binomial).
    async fn on_sample_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>) -> u64;

    /// Handle a usize sampling operation (Categorical).
    async fn on_sample_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>) -> usize;

    /// Handle an i64 sampling operation (signed discrete distributions).
    ///
    /// Like [`Handler::on_sample_i64`], this defaults to a panic with the same
    /// message,
    /// `handler does not implement on_sample_i64 (i64 sample site at {addr})`,
    /// so a handler only for models without [`Model::SampleI64`] nodes need
    /// not implement it. [`FromSync`] forwards to the wrapped handler's
    /// method.
    async fn on_sample_i64(&mut self, addr: &Address, _dist: &dyn Distribution<i64>) -> i64 {
        panic!(
            "handler does not implement on_sample_i64 (i64 sample site at {})",
            addr
        )
    }

    /// Handle an f64 observation operation.
    async fn on_observe_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>, value: f64);

    /// Handle a bool observation operation.
    async fn on_observe_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>, value: bool);

    /// Handle a u64 observation operation.
    async fn on_observe_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>, value: u64);

    /// Handle a usize observation operation.
    async fn on_observe_usize(
        &mut self,
        addr: &Address,
        dist: &dyn Distribution<usize>,
        value: usize,
    );

    /// Handle an i64 observation operation.
    ///
    /// Like [`Handler::on_observe_i64`], this defaults to a panic with the
    /// same message,
    /// `handler does not implement on_observe_i64 (i64 observe site at {addr})`.
    /// [`FromSync`] forwards to the wrapped handler's method.
    async fn on_observe_i64(&mut self, addr: &Address, _dist: &dyn Distribution<i64>, _value: i64) {
        panic!(
            "handler does not implement on_observe_i64 (i64 observe site at {})",
            addr
        )
    }

    /// Handle a factor operation.
    ///
    /// Called when the model reaches a `factor` statement; the handler
    /// typically adds `logw` to the trace's `log_factors`.
    ///
    /// # Arguments
    ///
    /// * `logw` - Log-weight to add to the model's total weight
    async fn on_factor(&mut self, logw: f64);

    /// Finalize the handler and return the accumulated trace.
    ///
    /// [`run_async`] awaits this once, after the model has returned its
    /// value. It is never called for a run whose future is dropped first.
    async fn finish(self) -> Trace
    where
        Self: Sized;
}

/// A synchronous [`Handler`] used as an [`AsyncHandler`].
///
/// Each method calls the wrapped handler's method of the same name and
/// completes without suspending, so `run_async(FromSync(h), m)` yields exactly
/// what [`run(h, m)`](crate::runtime::handler::run) does: the same value, the
/// same trace, and the same draws from the handler's RNG. Use it to hand the
/// built-in interpreters ([`PriorHandler`](crate::PriorHandler),
/// [`ReplayHandler`](crate::ReplayHandler),
/// [`ScoreGivenTrace`](crate::ScoreGivenTrace) and the others) to code written
/// against `AsyncHandler`.
///
/// Example:
/// ```rust
/// # use fugue::*;
///
/// // Score a recorded trace from inside async code
/// let mut recorded = Trace::default();
/// recorded.insert_choice(addr!("x"), ChoiceValue::F64(0.5), 0.0);
/// let scorer = FromSync(ScoreGivenTrace {
///     base: recorded,
///     trace: Trace::default(),
/// });
/// let model = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
///
/// let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
/// let (x, scored) = rt.block_on(run_async(scorer, model));
/// assert_eq!(x, 0.5);
/// assert_eq!(scored.log_prior, Normal::new(0.0, 1.0).unwrap().log_prob(&0.5));
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct FromSync<H>(pub H);

impl<H: Handler> AsyncHandler for FromSync<H> {
    async fn on_sample_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>) -> f64 {
        self.0.on_sample_f64(addr, dist)
    }

    async fn on_sample_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>) -> bool {
        self.0.on_sample_bool(addr, dist)
    }

    async fn on_sample_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>) -> u64 {
        self.0.on_sample_u64(addr, dist)
    }

    async fn on_sample_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>) -> usize {
        self.0.on_sample_usize(addr, dist)
    }

    async fn on_sample_i64(&mut self, addr: &Address, dist: &dyn Distribution<i64>) -> i64 {
        self.0.on_sample_i64(addr, dist)
    }

    async fn on_observe_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>, value: f64) {
        self.0.on_observe_f64(addr, dist, value)
    }

    async fn on_observe_bool(
        &mut self,
        addr: &Address,
        dist: &dyn Distribution<bool>,
        value: bool,
    ) {
        self.0.on_observe_bool(addr, dist, value)
    }

    async fn on_observe_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>, value: u64) {
        self.0.on_observe_u64(addr, dist, value)
    }

    async fn on_observe_usize(
        &mut self,
        addr: &Address,
        dist: &dyn Distribution<usize>,
        value: usize,
    ) {
        self.0.on_observe_usize(addr, dist, value)
    }

    async fn on_observe_i64(&mut self, addr: &Address, dist: &dyn Distribution<i64>, value: i64) {
        self.0.on_observe_i64(addr, dist, value)
    }

    async fn on_factor(&mut self, logw: f64) {
        self.0.on_factor(logw)
    }

    async fn finish(self) -> Trace {
        self.0.finish()
    }
}

/// Execute a probabilistic model with an asynchronous handler.
///
/// The asynchronous counterpart of [`run`](crate::runtime::handler::run): the
/// same walk over the same model, dispatching each effect to the matching
/// [`AsyncHandler`] method and awaiting it before the continuation runs, then
/// awaiting [`AsyncHandler::finish`]. It returns the model's final value and
/// the accumulated trace.
///
/// - **Sequential**: a run awaits its sites one at a time, in program order,
///   since the next site is not known until the current one has its value.
///   Concurrency comes from running many models at once.
/// - **Stack-safe**: like `run` (FG-19), an explicit loop rather than
///   recursion, so the stack depth of every poll is independent of the number
///   of sites, whether or not the handler suspends.
/// - **Executor-agnostic**: it is a plain future; fugue depends on no async
///   runtime.
/// - **Cancellable**: dropping the future stops the run at the site it is
///   waiting on; the handler is dropped without [`AsyncHandler::finish`].
///
/// The future is `Send` when the handler, its method futures and `A` are; see
/// [`AsyncHandler`].
///
/// Example:
/// ```rust
/// # use fugue::*;
/// # use fugue::runtime::interpreters::PriorHandler;
/// # use rand::rngs::StdRng;
/// # use rand::SeedableRng;
///
/// let model = || {
///     sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
///         .bind(|x| observe(addr!("y"), Normal::new(x, 0.1).unwrap(), 1.2).map(move |_| x))
/// };
///
/// let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
/// let mut rng = StdRng::seed_from_u64(123);
/// let (x, trace) = rt.block_on(run_async(
///     FromSync(PriorHandler { rng: &mut rng, trace: Trace::default() }),
///     model(),
/// ));
///
/// // Exactly what the synchronous interpreter gives from the same seed
/// let mut rng = StdRng::seed_from_u64(123);
/// let (x_sync, trace_sync) = runtime::handler::run(
///     PriorHandler { rng: &mut rng, trace: Trace::default() },
///     model(),
/// );
/// assert_eq!(x, x_sync);
/// assert_eq!(trace.total_log_weight(), trace_sync.total_log_weight());
/// ```
pub async fn run_async<A, H: AsyncHandler>(mut h: H, m: Model<A>) -> (A, Trace) {
    // The same iterative trampoline as `run` (FG-19), awaiting each effect.
    // The model is a CPS-encoded linked list of continuations, so one node is
    // live at a time: its `addr` and `dist` are locals of the match arm,
    // borrowed by the handler's future until that future completes, and only
    // then does the continuation `k(value)` build the next node. Nothing
    // recurses, so neither the stack depth of a poll nor this future's size
    // depends on the model's length. Only `Model::Pure` terminates.
    let mut m = m;
    let a = loop {
        m = match m {
            Model::Pure(a) => break a,
            Model::SampleF64 { addr, dist, k } => {
                let x = h.on_sample_f64(&addr, &*dist).await;
                k(x)
            }
            Model::SampleBool { addr, dist, k } => {
                let x = h.on_sample_bool(&addr, &*dist).await;
                k(x)
            }
            Model::SampleU64 { addr, dist, k } => {
                let x = h.on_sample_u64(&addr, &*dist).await;
                k(x)
            }
            Model::SampleUsize { addr, dist, k } => {
                let x = h.on_sample_usize(&addr, &*dist).await;
                k(x)
            }
            Model::SampleI64 { addr, dist, k } => {
                let x = h.on_sample_i64(&addr, &*dist).await;
                k(x)
            }
            Model::ObserveF64 {
                addr,
                dist,
                value,
                k,
            } => {
                h.on_observe_f64(&addr, &*dist, value).await;
                k(())
            }
            Model::ObserveBool {
                addr,
                dist,
                value,
                k,
            } => {
                h.on_observe_bool(&addr, &*dist, value).await;
                k(())
            }
            Model::ObserveU64 {
                addr,
                dist,
                value,
                k,
            } => {
                h.on_observe_u64(&addr, &*dist, value).await;
                k(())
            }
            Model::ObserveUsize {
                addr,
                dist,
                value,
                k,
            } => {
                h.on_observe_usize(&addr, &*dist, value).await;
                k(())
            }
            Model::ObserveI64 {
                addr,
                dist,
                value,
                k,
            } => {
                h.on_observe_i64(&addr, &*dist, value).await;
                k(())
            }
            Model::Factor { logw, k } => {
                h.on_factor(logw).await;
                k(())
            }
        };
    };
    let t = h.finish().await;
    (a, t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::addr;
    use crate::core::distribution::*;
    use crate::core::model::{factor, observe, pure, sample, ModelExt};
    use crate::runtime::handler::run;
    use crate::runtime::interpreters::{PriorHandler, ReplayHandler, ScoreGivenTrace};
    use rand::rngs::StdRng;
    use rand::{RngCore, SeedableRng};
    use std::future::Future;
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::pin::Pin;
    use std::sync::Arc;
    use std::task::{Context, Poll, Wake, Waker};

    /// A minimal executor: poll the future on this thread, park until woken.
    /// Enough to drive `run_async` with no async runtime at all.
    fn block_on<F: Future>(fut: F) -> F::Output {
        struct ThreadWaker(std::thread::Thread);
        impl Wake for ThreadWaker {
            fn wake(self: Arc<Self>) {
                self.0.unpark();
            }
        }
        let waker = Waker::from(Arc::new(ThreadWaker(std::thread::current())));
        let mut cx = Context::from_waker(&waker);
        let mut fut = std::pin::pin!(fut);
        loop {
            match fut.as_mut().poll(&mut cx) {
                Poll::Ready(out) => return out,
                Poll::Pending => std::thread::park(),
            }
        }
    }

    /// Returns `Pending` once (after waking its task), then `Ready`: a full
    /// suspend-and-resume round trip through the executor.
    struct YieldNow(bool);

    impl Future for YieldNow {
        type Output = ();
        fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
            if self.0 {
                Poll::Ready(())
            } else {
                self.0 = true;
                cx.waker().wake_by_ref();
                Poll::Pending
            }
        }
    }

    /// Suspends once before delegating each effect to the inner handler, so
    /// every site of a run goes through the executor.
    struct Yielding<H>(H);

    impl<H: AsyncHandler> AsyncHandler for Yielding<H> {
        async fn on_sample_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>) -> f64 {
            YieldNow(false).await;
            self.0.on_sample_f64(addr, dist).await
        }
        async fn on_sample_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>) -> bool {
            YieldNow(false).await;
            self.0.on_sample_bool(addr, dist).await
        }
        async fn on_sample_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>) -> u64 {
            YieldNow(false).await;
            self.0.on_sample_u64(addr, dist).await
        }
        async fn on_sample_usize(
            &mut self,
            addr: &Address,
            dist: &dyn Distribution<usize>,
        ) -> usize {
            YieldNow(false).await;
            self.0.on_sample_usize(addr, dist).await
        }
        async fn on_sample_i64(&mut self, addr: &Address, dist: &dyn Distribution<i64>) -> i64 {
            YieldNow(false).await;
            self.0.on_sample_i64(addr, dist).await
        }
        async fn on_observe_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>, v: f64) {
            YieldNow(false).await;
            self.0.on_observe_f64(addr, dist, v).await
        }
        async fn on_observe_bool(
            &mut self,
            addr: &Address,
            dist: &dyn Distribution<bool>,
            v: bool,
        ) {
            YieldNow(false).await;
            self.0.on_observe_bool(addr, dist, v).await
        }
        async fn on_observe_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>, v: u64) {
            YieldNow(false).await;
            self.0.on_observe_u64(addr, dist, v).await
        }
        async fn on_observe_usize(
            &mut self,
            addr: &Address,
            dist: &dyn Distribution<usize>,
            v: usize,
        ) {
            YieldNow(false).await;
            self.0.on_observe_usize(addr, dist, v).await
        }
        async fn on_observe_i64(&mut self, addr: &Address, dist: &dyn Distribution<i64>, v: i64) {
            YieldNow(false).await;
            self.0.on_observe_i64(addr, dist, v).await
        }
        async fn on_factor(&mut self, logw: f64) {
            YieldNow(false).await;
            self.0.on_factor(logw).await
        }
        async fn finish(self) -> Trace {
            YieldNow(false).await;
            self.0.finish().await
        }
    }

    type AllTypes = (f64, bool, u64, usize, i64);

    /// Samples and observes each of the five value types and ends with a
    /// factor; the observations depend on the sampled values, so the
    /// likelihood depends on every site.
    fn all_effects_model() -> Model<AllTypes> {
        crate::prob! {
            let x <- sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
            let b <- sample(addr!("b"), Bernoulli::new(0.3).unwrap());
            let n <- sample(addr!("n"), Poisson::new(4.0).unwrap());
            let c <- sample(addr!("c"), Categorical::new(vec![0.2, 0.5, 0.3]).unwrap());
            let z <- sample(addr!("z"), DiscreteUniform::new(-3, 3).unwrap());
            observe(addr!("obs_f64"), Normal::new(x, 0.5).unwrap(), 0.4);
            observe(addr!("obs_bool"), Bernoulli::new(if b { 0.9 } else { 0.2 }).unwrap(), true);
            observe(addr!("obs_u64"), Poisson::new(n as f64 + 1.0).unwrap(), 3u64);
            observe(addr!("obs_usize"), Categorical::uniform(3).unwrap(), c);
            observe(addr!("obs_i64"), DiscreteUniform::new(z - 1, z + 1).unwrap(), z);
            factor(-0.5 * x * x);
            pure((x, b, n, c, z))
        }
    }

    /// Bit-for-bit equality of two runs: the value, every choice (address,
    /// value, logp) and the three log-weight accumulators.
    fn assert_same_run(expected: &(AllTypes, Trace), actual: &(AllTypes, Trace)) {
        let ((x, b, n, c, z), t) = expected;
        let ((ax, ab, an, ac, az), at) = actual;
        assert_eq!(x.to_bits(), ax.to_bits(), "value x");
        assert_eq!((b, n, c, z), (ab, an, ac, az), "value");
        assert_eq!(t.choices.len(), at.choices.len(), "number of choices");
        for ((addr, choice), (a_addr, a_choice)) in t.choices.iter().zip(&at.choices) {
            assert_eq!(addr, a_addr);
            assert_eq!(choice.addr, a_choice.addr);
            assert_eq!(choice.value, a_choice.value, "value at {addr}");
            assert_eq!(
                choice.logp.to_bits(),
                a_choice.logp.to_bits(),
                "logp at {addr}"
            );
        }
        assert_eq!(t.log_prior.to_bits(), at.log_prior.to_bits(), "log_prior");
        assert_eq!(
            t.log_likelihood.to_bits(),
            at.log_likelihood.to_bits(),
            "log_likelihood"
        );
        assert_eq!(
            t.log_factors.to_bits(),
            at.log_factors.to_bits(),
            "log_factors"
        );
    }

    // #62: `run_async(FromSync(h), m)` is `run(h, m)`, for each fast built-in
    // handler, on a model with every effect: the same value, the same trace
    // and the same RNG consumption. The `Yielding` runs add a suspension at
    // every site, which must not change anything either.
    #[test]
    fn from_sync_matches_run_for_builtin_handlers() {
        // PriorHandler, from the same seed.
        let mut rngs = [0, 1, 2].map(|_| StdRng::seed_from_u64(62));
        let [r_sync, r_async, r_yield] = &mut rngs;
        let prior = |rng| PriorHandler {
            rng,
            trace: Trace::default(),
        };
        let expected = run(prior(r_sync), all_effects_model());
        let from_sync = block_on(run_async(FromSync(prior(r_async)), all_effects_model()));
        let yielding = block_on(run_async(
            Yielding(FromSync(prior(r_yield))),
            all_effects_model(),
        ));
        assert_same_run(&expected, &from_sync);
        assert_same_run(&expected, &yielding);
        let next = rngs.map(|mut rng| rng.next_u64());
        assert!(next[0] == next[1] && next[0] == next[2], "RNG consumption");

        // The model really exercises every effect.
        let prior_trace = expected.1;
        assert_eq!(prior_trace.choices.len(), 5);
        assert!(prior_trace.log_prior.is_finite());
        assert!(prior_trace.log_likelihood.is_finite() && prior_trace.log_likelihood < 0.0);
        assert!(prior_trace.log_factors <= 0.0);

        // ReplayHandler, with one site missing from the base trace, so the
        // run both replays and draws fresh from its RNG.
        let mut base = prior_trace.clone();
        base.choices.remove(&addr!("n"));
        let mut rngs = [0, 1, 2].map(|_| StdRng::seed_from_u64(7));
        let [r_sync, r_async, r_yield] = &mut rngs;
        let replay = |rng| ReplayHandler {
            rng,
            base: base.clone(),
            trace: Trace::default(),
        };
        let expected = run(replay(r_sync), all_effects_model());
        let from_sync = block_on(run_async(FromSync(replay(r_async)), all_effects_model()));
        let yielding = block_on(run_async(
            Yielding(FromSync(replay(r_yield))),
            all_effects_model(),
        ));
        assert_same_run(&expected, &from_sync);
        assert_same_run(&expected, &yielding);
        let next = rngs.map(|mut rng| rng.next_u64());
        assert!(next[0] == next[1] && next[0] == next[2], "RNG consumption");

        // ScoreGivenTrace, on the full prior trace.
        let score = || ScoreGivenTrace {
            base: prior_trace.clone(),
            trace: Trace::default(),
        };
        let expected = run(score(), all_effects_model());
        let from_sync = block_on(run_async(FromSync(score()), all_effects_model()));
        let yielding = block_on(run_async(Yielding(FromSync(score())), all_effects_model()));
        assert_same_run(&expected, &from_sync);
        assert_same_run(&expected, &yielding);
        assert_eq!(
            expected.1.total_log_weight().to_bits(),
            prior_trace.total_log_weight().to_bits()
        );
    }

    // #62, mirroring `interpretation_is_stack_safe_for_deep_models` (FG-19):
    // 100_000 sequential sample+bind sites through `run_async` on a 512 KiB
    // thread, driven by `block_on` on that thread. Once with effects that
    // complete without suspending (the whole run happens inside one poll), and
    // once with a suspension at every site (100_000 round trips through the
    // executor); neither may grow the stack with the model.
    #[test]
    fn run_async_is_stack_safe_for_deep_models() {
        fn build(i: usize, n: usize, acc: f64) -> Model<f64> {
            if i >= n {
                pure(acc)
            } else {
                sample(addr!("x", i), Normal::new(0.0, 1.0).unwrap())
                    .bind(move |x| build(i + 1, n, acc + x))
            }
        }

        let handle = std::thread::Builder::new()
            .stack_size(512 * 1024)
            .spawn(|| {
                let n = 100_000;
                let mut rng = StdRng::seed_from_u64(2024);
                let (sum, trace) = block_on(run_async(
                    FromSync(PriorHandler {
                        rng: &mut rng,
                        trace: Trace::default(),
                    }),
                    build(0, n, 0.0),
                ));
                assert!(sum.is_finite());
                assert_eq!(trace.choices.len(), n);
                assert!(trace.log_prior.is_finite());

                let mut rng = StdRng::seed_from_u64(2024);
                let (sum_yielding, trace_yielding) = block_on(run_async(
                    Yielding(FromSync(PriorHandler {
                        rng: &mut rng,
                        trace: Trace::default(),
                    })),
                    build(0, n, 0.0),
                ));
                assert_eq!(sum_yielding.to_bits(), sum.to_bits());
                assert_eq!(trace_yielding.choices.len(), n);
                assert_eq!(
                    trace_yielding.log_prior.to_bits(),
                    trace.log_prior.to_bits()
                );
            })
            .expect("spawn thread");
        handle
            .join()
            .expect("deep model async interpretation overflowed the stack");
    }

    /// Implements every required method of both handler traits, and neither
    /// trait's i64 methods, so their defaults run. The tests below reach only
    /// i64 sites.
    struct NoI64;

    impl Handler for NoI64 {
        fn on_sample_f64(&mut self, _: &Address, _: &dyn Distribution<f64>) -> f64 {
            unreachable!()
        }
        fn on_sample_bool(&mut self, _: &Address, _: &dyn Distribution<bool>) -> bool {
            unreachable!()
        }
        fn on_sample_u64(&mut self, _: &Address, _: &dyn Distribution<u64>) -> u64 {
            unreachable!()
        }
        fn on_sample_usize(&mut self, _: &Address, _: &dyn Distribution<usize>) -> usize {
            unreachable!()
        }
        fn on_observe_f64(&mut self, _: &Address, _: &dyn Distribution<f64>, _: f64) {
            unreachable!()
        }
        fn on_observe_bool(&mut self, _: &Address, _: &dyn Distribution<bool>, _: bool) {
            unreachable!()
        }
        fn on_observe_u64(&mut self, _: &Address, _: &dyn Distribution<u64>, _: u64) {
            unreachable!()
        }
        fn on_observe_usize(&mut self, _: &Address, _: &dyn Distribution<usize>, _: usize) {
            unreachable!()
        }
        fn on_factor(&mut self, _: f64) {
            unreachable!()
        }
        fn finish(self) -> Trace {
            unreachable!()
        }
    }

    impl AsyncHandler for NoI64 {
        async fn on_sample_f64(&mut self, _: &Address, _: &dyn Distribution<f64>) -> f64 {
            unreachable!()
        }
        async fn on_sample_bool(&mut self, _: &Address, _: &dyn Distribution<bool>) -> bool {
            unreachable!()
        }
        async fn on_sample_u64(&mut self, _: &Address, _: &dyn Distribution<u64>) -> u64 {
            unreachable!()
        }
        async fn on_sample_usize(&mut self, _: &Address, _: &dyn Distribution<usize>) -> usize {
            unreachable!()
        }
        async fn on_observe_f64(&mut self, _: &Address, _: &dyn Distribution<f64>, _: f64) {
            unreachable!()
        }
        async fn on_observe_bool(&mut self, _: &Address, _: &dyn Distribution<bool>, _: bool) {
            unreachable!()
        }
        async fn on_observe_u64(&mut self, _: &Address, _: &dyn Distribution<u64>, _: u64) {
            unreachable!()
        }
        async fn on_observe_usize(&mut self, _: &Address, _: &dyn Distribution<usize>, _: usize) {
            unreachable!()
        }
        async fn on_factor(&mut self, _: f64) {
            unreachable!()
        }
        async fn finish(self) -> Trace {
            unreachable!()
        }
    }

    fn panic_message(f: impl FnOnce()) -> String {
        let payload = catch_unwind(AssertUnwindSafe(f)).expect_err("expected a panic");
        match payload.downcast::<String>() {
            Ok(message) => *message,
            Err(payload) => payload
                .downcast_ref::<&str>()
                .expect("panic payload is a string")
                .to_string(),
        }
    }

    // #62: the i64 methods of `AsyncHandler` default to a panic with the
    // documented message, which is `Handler`'s: a handler that does not
    // implement them fails the same way through `run`, through `run_async`,
    // and through `run_async` over `FromSync`.
    #[test]
    fn i64_defaults_panic_with_the_handler_message() {
        let sample_i64 = || sample(addr!("z"), DiscreteUniform::new(-1, 1).unwrap());
        let documented = "handler does not implement on_sample_i64 (i64 sample site at z)";
        assert_eq!(
            panic_message(|| drop(block_on(run_async(NoI64, sample_i64())))),
            documented
        );
        assert_eq!(panic_message(|| drop(run(NoI64, sample_i64()))), documented);
        assert_eq!(
            panic_message(|| drop(block_on(run_async(FromSync(NoI64), sample_i64())))),
            documented
        );

        let observe_i64 = || observe(addr!("w"), DiscreteUniform::new(-1, 1).unwrap(), 0);
        let documented = "handler does not implement on_observe_i64 (i64 observe site at w)";
        assert_eq!(
            panic_message(|| drop(block_on(run_async(NoI64, observe_i64())))),
            documented
        );
        assert_eq!(
            panic_message(|| drop(run(NoI64, observe_i64()))),
            documented
        );
        assert_eq!(
            panic_message(|| drop(block_on(run_async(FromSync(NoI64), observe_i64())))),
            documented
        );
    }
}
