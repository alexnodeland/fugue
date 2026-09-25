# Async Interpretation

## Overview

[`run`](crate::runtime::handler::run) is a synchronous trampoline: every
[`Handler`](crate::Handler) method returns its value directly. A handler that
resolves a site over the network (a tool call, a model that answers in ~100 ms,
an LLM) then has to block its thread for the length of every call.

This module interprets the same [`Model`](crate::Model) values asynchronously:

- [`AsyncHandler`] mirrors [`Handler`](crate::Handler) method for method
  (`on_sample_*`, `on_observe_*`, `on_factor`, `finish`), each an `async fn`.
- [`run_async`] walks the model exactly as `run` does and awaits each effect
  before the model continues. A model is data, a chain of `Send`
  continuations, so this is a second loop over the same values, not a second
  kind of model.
- [`FromSync`] makes any synchronous handler an `AsyncHandler`, and
  `run_async(FromSync(h), m)` yields exactly what `run(h, m)` does.

fugue depends on no async runtime: `run_async` returns a plain future, and any
executor can drive it (tokio, smol, a hand-written `block_on`). A handler's own
I/O may need a particular one, as tokio's timers do in the example below.

**When to use it**: when a handler's effects are resolved by I/O. A handler that
computes locally, like the built-in interpreters and the inference algorithms
built on them, has nothing to wait for and should keep using `run`.

## Usage Examples

### A handler that asks a remote oracle

```rust
use fugue::*;
use std::time::Duration;

/// Stands in for a network round trip: an HTTP request, a tool call, an LLM.
async fn ask_oracle(addr: &Address) -> f64 {
    tokio::time::sleep(Duration::from_millis(1)).await;
    if addr.as_str() == "temperature" { 21.5 } else { 0.0 }
}

/// Resolves each f64 sample site by asking the oracle, and scores the answer
/// under the site's prior, so the run leaves an ordinary trace to audit.
struct OracleHandler {
    trace: Trace,
}

impl AsyncHandler for OracleHandler {
    async fn on_sample_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>) -> f64 {
        let x = ask_oracle(addr).await;
        // `addr` and `dist` are still borrowed after the `.await`.
        let logp = dist.log_prob(&x);
        self.trace.log_prior += logp;
        self.trace.insert_choice(addr.clone(), ChoiceValue::F64(x), logp);
        x
    }

    async fn on_observe_f64(&mut self, _addr: &Address, dist: &dyn Distribution<f64>, value: f64) {
        self.trace.log_likelihood += dist.log_prob(&value);
    }

    async fn on_factor(&mut self, logw: f64) {
        self.trace.log_factors += logw;
    }

    async fn finish(self) -> Trace {
        self.trace
    }

    // The bool, u64 and usize methods are required too. This model never
    // reaches them, so here they are `unreachable!()` (hidden). The i64
    // methods have panicking defaults.
#     async fn on_sample_bool(&mut self, _: &Address, _: &dyn Distribution<bool>) -> bool { unreachable!() }
#     async fn on_sample_u64(&mut self, _: &Address, _: &dyn Distribution<u64>) -> u64 { unreachable!() }
#     async fn on_sample_usize(&mut self, _: &Address, _: &dyn Distribution<usize>) -> usize { unreachable!() }
#     async fn on_observe_bool(&mut self, _: &Address, _: &dyn Distribution<bool>, _: bool) { unreachable!() }
#     async fn on_observe_u64(&mut self, _: &Address, _: &dyn Distribution<u64>, _: u64) { unreachable!() }
#     async fn on_observe_usize(&mut self, _: &Address, _: &dyn Distribution<usize>, _: usize) { unreachable!() }
}

let model = sample(addr!("temperature"), Normal::new(20.0, 2.0).unwrap())
    .bind(|t| observe(addr!("reading"), Normal::new(t, 0.5).unwrap(), 21.0).map(move |_| t));

// Any executor can drive the future; this one is tokio's.
let rt = tokio::runtime::Builder::new_current_thread()
    .enable_time()
    .build()
    .unwrap();
let (t, trace) = rt.block_on(run_async(OracleHandler { trace: Trace::default() }, model));

assert_eq!(t, 21.5);
assert_eq!(trace.get_f64(&addr!("temperature")), Some(21.5));
assert!(trace.total_log_weight().is_finite());
```

### A synchronous handler in async code

```rust
# use fugue::*;
# use fugue::runtime::interpreters::PriorHandler;
# use rand::rngs::StdRng;
# use rand::SeedableRng;
let mut rng = StdRng::seed_from_u64(7);
let prior = FromSync(PriorHandler { rng: &mut rng, trace: Trace::default() });
let model = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());

let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
let (x, trace) = rt.block_on(run_async(prior, model));
assert_eq!(trace.get_f64(&addr!("x")), Some(x));
```

`FromSync` calls the synchronous methods and never suspends, so the run is the
one `run` would make from the same handler: the same value, trace and RNG
draws.

## `Send`, and spawning runs

`AsyncHandler`'s methods are declared with `async fn`, so the trait does not
bound their futures by `Send`, and generic code over `H: AsyncHandler` cannot
require that they are: stable Rust cannot yet name that bound (return-type
notation, rust-lang/rust#109417, is unstable).

For a concrete handler type the compiler does see the actual futures, because
auto traits leak through `async fn`. If the handler is `Send`, none of its
methods holds anything else `!Send` (an `Rc`, a `RefCell` borrow) across an
`.await`, and the model's value type is `Send`, then
`run_async(handler, model)` is `Send`. If
the handler also borrows nothing (unlike `PriorHandler`, which borrows its
RNG), the future is `'static` as well and can be passed to `tokio::spawn` or
any other multi-threaded executor. `Model<A>` is itself `Send` whenever `A` is:
its continuations are `Send`, and [`Distribution`](crate::Distribution)
requires `Send + Sync`. The tests pin this by spawning runs on tokio's
multi-threaded runtime.

A function generic over the handler cannot spawn its run on a multi-threaded
executor, because it cannot state that the handler's futures are `Send`. Write
such code for the concrete handler types, or drive the runs on a
single-threaded executor (a current-thread runtime, a `LocalSet`), which does
not need `Send`. The same goes for handlers that are not `Send` at all: the
trait does not rule them out, and that is why it uses `async fn` rather than
`-> impl Future + Send`.

## Design Notes

- **One run is sequential.** `run_async` awaits one site at a time, in program
  order, exactly as `run` visits them: the next site is not known until the
  current one has its value. Interpretation stays single-threaded per run: a
  multi-threaded executor may resume a run on another worker after an
  `.await`, but never works on two of its sites at once. Concurrency comes from
  running many models at once, one future per particle, episode or request, so
  that their waits overlap.
- **Stack-safe.** Like `run` since FG-19, `run_async` is an explicit loop, not
  a recursion, so the stack depth of every poll is independent of the number of
  sites, whether or not the handler suspends. A 100 000-site sample-and-bind
  chain runs on a 512 KiB thread either way.
- **Cancellation.** Dropping the future stops the run at the site it is waiting
  on: the rest of the model never runs, the handler is dropped without
  `finish`, and no trace is returned. Whether a request in flight may be
  abandoned is up to the handler.
- **Same semantics otherwise.** The handler invariants of the
  [handler module](crate::runtime::handler) apply unchanged: `finish` is
  awaited exactly once for a run that completes, and the i64 methods default to
  the same panics as `Handler`'s.

## Reference Links

- [`AsyncHandler`], [`run_async`], [`FromSync`]: this module
- [`Handler`](crate::Handler) and [`run`](crate::runtime::handler::run): the
  synchronous interpreter this mirrors
- [`interpreters`](crate::runtime::interpreters): the built-in handlers, usable
  here through `FromSync`
- Tests: equivalence with `run`, stack safety and the i64 defaults in this
  module's unit tests; in `tests/f_async_handler.rs`, a timer-awaiting handler
  spawned on tokio's multi-threaded runtime, and one that is not `Send` on a
  current-thread runtime
