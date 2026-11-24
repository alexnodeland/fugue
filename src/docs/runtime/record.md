# Per-Site Weight Records

## Overview

A [`Trace`](crate::runtime::trace::Trace) keeps three flat sums: `log_prior`,
`log_likelihood` and `log_factors`. Sample sites also leave a
[`Choice`](crate::runtime::trace::Choice) with its own `logp`, but observations
and factors leave nothing behind except their contribution to a sum, so a
trace cannot say *which* observation or factor a log-weight came from.

This module adds that attribution as an opt-in, without changing the trace:

- [`SiteWeights`] is the record: log-weight per observation address, per
  addressed factor ([`factor_at`](crate::core::model::factor_at)), and one sum
  for anonymous factors ([`factor`](crate::core::model::factor)).
- [`RecordSites`] is a [`Handler`](crate::runtime::handler::Handler) that
  wraps any other handler, fills a `SiteWeights`, and forwards every site
  unchanged. The inner handler's trace is bit for bit what it would have been
  without the wrapper.
- [`run_recorded`] runs a model under a wrapped handler and returns the
  record with the value and the trace.

Nothing is recorded unless a model is run through `RecordSites`: plain runs
and every inference routine pay nothing.

## Usage Examples

### Attributing a likelihood to its sites

```rust
# use fugue::*;
# use fugue::runtime::record::run_recorded;
# use rand::rngs::StdRng;
# use rand::SeedableRng;
let model = sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap()).bind(|mu| {
    observe(addr!("y", 0), Normal::new(mu, 1.0).unwrap(), 0.4)
        .bind(move |_| observe(addr!("y", 1), Normal::new(mu, 1.0).unwrap(), 2.5))
        .bind(move |_| factor_at(addr!("penalty"), -0.5 * mu * mu))
        .bind(|_| factor(-0.1))
});

let mut rng = StdRng::seed_from_u64(7);
let (_, trace, sites) = run_recorded(
    PriorHandler { rng: &mut rng, trace: Trace::default() },
    model,
);

// One entry per observation and per addressed factor.
assert_eq!(sites.observations().len(), 2);
assert!(sites.observation(&addr!("y", 1)).unwrap() < sites.observation(&addr!("y", 0)).unwrap());
assert!(sites.factor(&addr!("penalty")).is_some());
assert_eq!(sites.unaddressed_factors(), -0.1);

// The per-site sums are the trace's flat accumulators.
assert!((sites.log_likelihood() - trace.log_likelihood).abs() < 1e-12);
assert!((sites.log_factors() - trace.log_factors).abs() < 1e-12);
```

### Scoring an existing trace by site

Wrap [`ScoreGivenTrace`](crate::runtime::interpreters::ScoreGivenTrace) to
attribute the likelihood of a trace you already have, such as an SMC particle
or an MCMC state, and sum by address prefix:

```rust
# use fugue::*;
# use fugue::runtime::record::run_recorded;
# use rand::rngs::StdRng;
# use rand::SeedableRng;
let model = || {
    sample(Address::new("rate"), Beta::new(1.0, 1.0).unwrap()).bind(|p| {
        observe(Address::new("m/a/x/0"), Bernoulli::new(p).unwrap(), true)
            .bind(move |_| observe(Address::new("m/a/x/1"), Bernoulli::new(p).unwrap(), true))
            .bind(move |_| observe(Address::new("m/b/x/0"), Bernoulli::new(p).unwrap(), false))
    })
};
let mut rng = StdRng::seed_from_u64(3);
let (_, particle) = runtime::handler::run(
    PriorHandler { rng: &mut rng, trace: Trace::default() },
    model(),
);

let (_, scored, sites) = run_recorded(
    ScoreGivenTrace { base: particle, trace: Trace::default() },
    model(),
);
let a = sites.observations_under("m/a");
let b = sites.observations_under("m/b");
assert!((a + b - scored.log_likelihood).abs() < 1e-12);
```

## Design

- **What is recorded.** At an observation, `dist.log_prob(&value)`, with `NaN`
  read as `-∞`: the number the built-in interpreters add to `log_likelihood`.
  At a factor, its `logw`, with the same rule. So for any handler that scores
  sites the way the built-in ones do, [`SiteWeights::log_likelihood`] and
  [`SiteWeights::log_factors`] equal the trace's `log_likelihood` and
  `log_factors` up to the rounding of summing in a different order (the trace
  adds in program order, the record by address). A custom inner handler that
  scales or drops weights is still recorded at the model's own log-density.
- **Repeated addresses add up.** Observation and factor addresses are not
  checked for uniqueness anywhere in the crate, so a site visited twice at one
  address contributes the sum of both visits.
- **The trace is untouched.** `Trace` is a plain struct that callers build
  with a literal, so a per-site map could not be added to it without breaking
  them. The record lives beside the trace instead.
- **Composition.** `RecordSites` forwards
  [`on_factor_at`](crate::runtime::handler::Handler::on_factor_at) with its
  address, so recorders nest and work inside
  [`Delegate::new`](crate::runtime::delegate::Delegate::new). An
  [`Overrides`](crate::runtime::delegate::Overrides) with hooks of its own
  must forward `on_factor_at` to keep factor addresses for a recorder
  underneath it; one placed on the outside sees everything.
