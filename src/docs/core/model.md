# Model

Fugue’s `Model<A>` is a tiny, direct-style probabilistic programming interface. It favors explicit control flow, strong typing, and composability over magic. This page explains when and how to use `Model`, the patterns it encourages, and the architectural decisions behind it.

## Why `Model` exists

- Express probabilistic programs in plain Rust with first-class composition.
- Keep inference strategies pluggable via handlers/interpreters (e.g., prior runs, scoring, replay, MCMC).
- Make dependencies explicit using addresses and monadic control flow (no hidden globals).

## Mental model

`Model<A>` is a recipe that, when run by a handler, produces a value of type `A` and a trace with choices and weights. You build programs by chaining a small set of primitives:

- `pure(a)` — deterministic value
- `sample(addr, dist)` — draw a random choice at `addr`
- `observe(addr, dist, value)` — condition on observed `value`
- `factor(logw)` — add log-weight (soft constraint)
- `bind/map/and_then` — compose computations

These map to `Model`’s internal variants and are designed to remain stable across interpreters.

## Choosing the right primitive

| Intent | Use | Notes |
| --- | --- | --- |
| Deterministic transformation | `pure`, then `map` | Prefer `map` to keep probabilistic structure minimal |
| Draw a latent variable | `sample(addr, dist)` | Use stable, descriptive addresses |
| Condition on data | `observe(addr, dist, y)` | Leaves no choice in the trace; affects likelihood |
| Soft constraint / score tweak | `factor(logw)` | Use `guard(pred)` for hard constraints |

## Addressing strategy

- Use `addr!("name")` for scalar sites and `addr!("name", i)` for plates/loops.
- Addresses must be stable across runs and data orders; treat them as part of your public model interface.
- Prefer semantic names over indices when possible (e.g., `addr!("user", user_id)`).

## Common patterns

- Dependent sampling (hierarchical): sample parent, then child conditioned on parent via `bind`.
- Branching by data: use `bind` and plain Rust `if`/`match` to choose submodels.
- Collections: use `traverse_vec` to map data to models and collect results, or `sequence_vec` when you already have models.
- Constraints: use `guard(pred)` for hard filters and `factor(logw)` for graded preferences.

### Pattern: dependent sampling

```rust
# use fugue::*;

let model = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
    .bind(|x| sample(addr!("y"), Normal::new(x, 0.5).unwrap()));
```

### Pattern: working over datasets

```rust
# use fugue::*;

let data = vec![1.0, 2.0, 3.0];
let model = traverse_vec(data, |x| {
    sample(addr!("noise", x as usize), Normal::new(0.0, 0.1).unwrap())
        .map(move |e| x + e)
});
```

### Pattern: constraints

```rust
# use fugue::*;

let non_negative = sample(addr!("z"), Normal::new(0.0, 1.0).unwrap())
    .bind(|z| guard(z >= 0.0).map(move |_| z));

let soft_preference = factor(-0.5);
```

## Design notes (architecture)

- Direct style over free monads: keeps code idiomatic, easy to debug, and handler-friendly.
- Addresses, not implicit scope: reproducibility and replay depend on stable addresses.
- Generic `sample/observe` dispatch via `SampleType`: compile-time selection of the right variant without macros or dynamic checks.
- Handlers interpret the same `Model` differently (e.g., `PriorHandler`, replay/score). Your model code remains unchanged.

## Do/Don’t

- Do keep addresses stable and meaningful.
- Do prefer `map` for pure transforms and `bind` only when you need dependent structure.
- Don’t rely on evaluation side effects inside closures; treat models as descriptions, not eager computations.
- Don’t mix observation and sampling at the same address.

## Model variants and composition

### Variants at a glance

| Variant | When to use | Trace effect | Typical helper |
| --- | --- | --- | --- |
| `Pure(A)` | Deterministic values and glue | No new choice | `pure` + `map` |
| `Sample{..}` | Introduce a latent random variable | Records a choice at `addr` | `sample` (or type-specific) |
| `Observe{..}` | Condition on observed data | No choice; updates likelihood | `observe` |
| `Factor{..}` | Soft constraints / custom scores | Adds to total log-weight | `factor`, `guard` |

### Monadic operations (`ModelExt`)

`Model` implements `ModelExt` providing the fundamental monadic operations:

- `bind` (>>=): Chains dependent computations where the next step depends on a previous random result
- `map`: Transforms the result without adding probabilistic behavior (functor map)
- `and_then`: Alias for `bind` for those familiar with Rust's `Option`/`Result`

**Composition guidelines:**

- Use `map` for pure transformations; it keeps structure declarative and optimizable
- Use `bind` (or `and_then`) when the next step depends on a previous random result
- Prefer `zip`/`sequence_vec`/`traverse_vec` over manual loops; they encode intent and reduce boilerplate

### Extended composition examples

```rust
# use fugue::*;
// Using bind for dependent sampling
let dependent = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
    .bind(|x| sample(addr!("y"), Normal::new(x, 0.5).unwrap()));

// Using map for transformations
let transformed = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
    .map(|x| x * 2.0 + 1.0);

// Chaining multiple operations with branching
let branched = sample(addr!("x"), Uniform::new(0.0, 1.0).unwrap())
    .bind(|x| {
        if x > 0.5 {
            sample(addr!("high"), Normal::new(10.0, 1.0).unwrap())
        } else {
            sample(addr!("low"), Normal::new(-10.0, 1.0).unwrap())
        }
    })
    .map(|result| result.abs());
```

## See also

- API: `Model`, `ModelExt::{bind,map,and_then}`, `zip`, `sequence_vec`, `traverse_vec`, `guard`, `factor`
- Runtimes: prior execution, replay, and scoring handlers in `runtime::interpreters`
