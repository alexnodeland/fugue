# Delegating Handlers

## Overview

A [`Handler`](crate::runtime::handler::Handler) implements a method for every kind of site: `on_sample_*` and `on_observe_*` for `f64`, `bool`, `u64` and `usize` (the `i64` pair has defaults that panic), plus `on_factor` and `finish`. A handler that cares about one kind of site still writes all ten, and a decorator that wraps another handler spends most of them forwarding to it.

[`Delegate`] is that forwarding, written once. It wraps an inner handler and an [`Overrides`] value. Each site goes to the hook of the same name on the `Overrides`, which receives the inner handler as an argument, and every hook defaults to calling the same method on the inner handler. An implementation overrides only the sites it cares about:

- `Delegate::new(inner)` overrides nothing and behaves exactly like `inner`.
- `Delegate::with(inner, overrides)` sends each site through the hooks of `overrides`.

## Usage Examples

### Overriding one kind of site

A decorator that works with any handler implements `Overrides<H>` for every `H: Handler`. This one records the observations it sees and forwards them; every sample site and the factor go straight to the inner handler:

```rust
# use fugue::*;
# use fugue::runtime::interpreters::PriorHandler;
# use rand::rngs::StdRng;
# use rand::SeedableRng;
/// Records each f64 observation, then forwards it.
struct Observations<'a>(&'a mut Vec<(Address, f64)>);

impl<H: Handler> Overrides<H> for Observations<'_> {
    fn on_observe_f64(
        &mut self,
        inner: &mut H,
        addr: &Address,
        dist: &dyn Distribution<f64>,
        value: f64,
    ) {
        self.0.push((addr.clone(), value));
        inner.on_observe_f64(addr, dist, value)
    }
}

let model = prob! {
    let mu <- sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap());
    observe(addr!("y", 0), Normal::new(mu, 1.0).unwrap(), 0.8);
    observe(addr!("y", 1), Normal::new(mu, 1.0).unwrap(), 1.3);
    factor(-0.5);
    pure(mu)
};

let mut seen = Vec::new();
let mut rng = StdRng::seed_from_u64(42);
let handler = Delegate::with(
    PriorHandler { rng: &mut rng, trace: Trace::default() },
    Observations(&mut seen),
);
let (_mu, trace) = runtime::handler::run(handler, model);

assert_eq!(seen, [(addr!("y", 0), 0.8), (addr!("y", 1), 1.3)]);
// The PriorHandler still scored everything it was forwarded.
assert!(trace.log_likelihood.is_finite());
assert_eq!(trace.log_factors, -0.5);
```

`run` consumes the handler, so state a caller wants back lives behind a reference, as `seen` does here, or is reported by a `finish` override.

### Answering a site with a concrete inner handler

An `Overrides` implementation can name one concrete handler instead of every `H: Handler`, and then reach its fields. This one resolves the `usize` decision sites itself, always picking option `0`, and records the choice in the `PriorHandler`'s trace the way `PriorHandler` records a draw. The `PriorHandler` handles every other site:

```rust
# use fugue::*;
# use fugue::runtime::interpreters::PriorHandler;
# use rand::rngs::StdRng;
# use rand::{RngCore, SeedableRng};
struct AlwaysFirst;

impl<R: RngCore> Overrides<PriorHandler<'_, R>> for AlwaysFirst {
    fn on_sample_usize(
        &mut self,
        inner: &mut PriorHandler<'_, R>,
        addr: &Address,
        dist: &dyn Distribution<usize>,
    ) -> usize {
        let logp = dist.log_prob(&0);
        inner.trace.log_prior += logp;
        inner.trace.insert_choice(addr.clone(), ChoiceValue::Usize(0), logp);
        0
    }
}

let model = prob! {
    let arm <- sample(addr!("arm"), Categorical::new(vec![0.25, 0.75]).unwrap());
    let reward <- sample(addr!("reward"), Normal::new(arm as f64, 1.0).unwrap());
    pure((arm, reward))
};

let mut rng = StdRng::seed_from_u64(7);
let handler = Delegate::with(PriorHandler { rng: &mut rng, trace: Trace::default() }, AlwaysFirst);
let ((arm, _reward), trace) = runtime::handler::run(handler, model);

assert_eq!(arm, 0);
assert_eq!(trace.get_usize(&addr!("arm")), Some(0));
assert!((trace.choices[&addr!("arm")].logp - 0.25_f64.ln()).abs() < 1e-12);
assert!(trace.get_f64(&addr!("reward")).is_some()); // drawn by the PriorHandler
```

### Nesting decorators

Delegates nest. In `Delegate<Delegate<H, A>, B>` the hooks of `B` see each site first, with the `Delegate<H, A>` as their inner handler, so a site passes through `B`, then `A`, then reaches `H`. The `finish` hooks run in the same order:

```rust
# use fugue::*;
# use fugue::runtime::interpreters::PriorHandler;
# use rand::rngs::StdRng;
# use rand::SeedableRng;
# use std::cell::RefCell;
/// Logs the f64 samples it sees and the end of the run, under a name.
struct Tap<'a>(&'static str, &'a RefCell<Vec<String>>);

impl<H: Handler> Overrides<H> for Tap<'_> {
    fn on_sample_f64(
        &mut self,
        inner: &mut H,
        addr: &Address,
        dist: &dyn Distribution<f64>,
    ) -> f64 {
        self.1.borrow_mut().push(format!("{}: sample {}", self.0, addr));
        inner.on_sample_f64(addr, dist)
    }

    fn finish(self, inner: H) -> Trace {
        self.1.borrow_mut().push(format!("{}: finish", self.0));
        inner.finish()
    }
}

let log = RefCell::new(Vec::new());
let mut rng = StdRng::seed_from_u64(3);
let prior = PriorHandler { rng: &mut rng, trace: Trace::default() };
let handler = Delegate::with(Delegate::with(prior, Tap("A", &log)), Tap("B", &log));

let (_x, trace) =
    runtime::handler::run(handler, sample(addr!("x"), Normal::new(0.0, 1.0).unwrap()));

assert_eq!(log.into_inner(), ["B: sample x", "A: sample x", "B: finish", "A: finish"]);
assert!(trace.get_f64(&addr!("x")).is_some());
```

## Design

- **The hooks receive the inner handler explicitly.** A hook is a method of the `Overrides` value that takes `inner: &mut H`, not a method of the handler itself. So there is no blanket implementation of `Handler` for some other trait, no ambiguity between a hook and the `Handler` method of the same name, and decorators nest, since a `Delegate` is itself a `Handler` another `Delegate` can wrap.
- **`Handler`'s contract is unchanged.** Default methods on `Handler` would let a handler that forgot a site silently do something at it; the `i64` defaults panic for that reason. With `Delegate`, forwarding is something a caller asks for by naming the inner handler.
- **The `i64` hooks forward too**, so a delegate panics at an `i64` site only if its inner handler does.
- **Zero cost.** `Delegate<H, O>` is a plain struct of the two, with no boxing, and every hook call is statically dispatched. A hook that is not overridden compiles to a direct call on the inner handler.

## Reference Links

- [`Handler`](crate::runtime::handler::Handler) and [`run`](crate::runtime::handler::run): the trait a `Delegate` implements, and the interpreter that drives it
- [`PriorHandler`](crate::runtime::interpreters::PriorHandler), [`ReplayHandler`](crate::runtime::interpreters::ReplayHandler), [`ScoreGivenTrace`](crate::runtime::interpreters::ScoreGivenTrace): built-in handlers to wrap
- [Custom Handlers how-to](https://fugue.run/how-to/custom-handlers.html): the logging decorator written with `Delegate`
- [`custom_handlers.rs`](https://github.com/alexnodeland/fugue/blob/main/examples/custom_handlers.rs): the runnable example behind the how-to
