# Addressing and Site Naming

Addresses are the stable, human-readable identifiers for random choices and observation sites. They are the backbone of reproducibility and model tooling:

- **Conditioning**: Attach observations to specific sites.
- **Inference targeting**: Select which sites to sample or clamp.
- **Replay & debugging**: Reproduce execution paths and inspect traces.

This page describes the addressing architecture, naming conventions, and recommended patterns for production models.

## Design goals

- **Determinism**: Same model + same inputs → same addresses.
- **Human readability**: Semantic names that make traces understandable.
- **Composability**: Easy to build hierarchical, indexed addresses.
- **Zero footguns**: Minimize collisions and accidental reuse.

## Core concepts

- `Address`: an ordered, hashable wrapper around a `String` used as a site key.
- `addr!(name[, index])`: macro to construct `Address` from a base name and optional index.
- `scoped_addr!(scope, name[, fmt, indices...])`: macro to prepend a scope, for hierarchical models.

## Naming schema and conventions

| Component | Format | Examples | Use when |
|---|---|---|---|
| Simple site | `name` | `"mu"`, `"sigma"` | Single variable with no repetition |
| Indexed site | `name#index` | `"data#0"`, `"weight#12"` | Repeated structures (loops, arrays) |
| Scoped site | `scope::name` | `"encoder::z"` | Submodule or hierarchical context |
| Scoped + indexed | `scope::name#index` | `"layer1::w#3"` | Repeated substructures in a scope |

Guidelines:

- Prefer semantic nouns: `"mu"`, `"theta"`, `"obs"`, `"mixture_weight"`.
- Index with deterministic integers from data/loops, not random values.
- Use scopes to clarify ownership or module (`"encoder::z"`, `"plate::x#i"`).
- Never concatenate floating-point or non-deterministic values into names.

## Quick start

```rust
# use fugue::*;
// Simple named site
let mu = addr!("mu");

// Indexed site for a plate/loop
let x0 = addr!("x", 0);
let x1 = addr!("x", 1);

// Scoped site
let z_enc = scoped_addr!("encoder", "z");

assert_ne!(addr!("mu"), addr!("mu", 0));
assert_ne!(addr!("x", 1), addr!("x", 2));
```

## Patterns

### 1) Single variable sites

```rust
# use fugue::*;
let model = sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap())
    .bind(|mu| sample(addr!("x"), Normal::new(mu, 1.0).unwrap()));
```

Why: Names map directly to conceptual variables and appear as such in traces.

### 2) Plates and collections (indexed)

```rust
# use fugue::*;
let data = vec![0.1, 0.5, -0.2];
let indexed: Vec<(usize, f64)> = data.into_iter().enumerate().collect();
let model = traverse_vec(indexed, move |(i, y)| {
    sample(addr!("x", i), Normal::new(0.0, 1.0).unwrap())
        .bind(move |x_i| observe(addr!("y", i), Normal::new(x_i, 1.0).unwrap(), y))
});
```

Why: Using `addr!("x", i)` and `addr!("y", i)` ensures a 1:1 mapping to data indices.

### 3) Hierarchical/structured models (scopes)

```rust
# use fugue::*;
let z_enc = sample(scoped_addr!("encoder", "z"), Normal::new(0.0, 1.0).unwrap());
let z_dec = sample(scoped_addr!("decoder", "z"), Normal::new(0.0, 1.0).unwrap());
let model = z_enc.bind(|_| z_dec);
```

Why: Scopes communicate ownership and separate similarly named variables.

### 4) Scoped + indexed

```rust
# use fugue::*;
let model = traverse_vec((0..3).collect::<Vec<_>>(), |i| {
    sample(scoped_addr!("layer1", "w", "{}", i), Normal::new(0.0, 1.0).unwrap())
});
```

Why: Combine scopes for hierarchy with indices for repetition.

## Anti-patterns and footguns

- Generating names with randomness or non-deterministic sources.
- Reusing the same address for semantically different variables.
- Using stringified floats or timestamps inside addresses.
- Forgetting to index repeated sites inside loops.

## Integration notes

- Addresses are keys in traces. Collisions overwrite entries; treat collisions as bugs.
- Ordering and hashing are defined to allow `BTreeMap`/`HashMap` usage.
- Keep address construction close to where sampling/observing occurs.

## Reference: macros and helpers

```rust
# use fugue::*;
// Simple
let a = addr!("theta");
// Indexed
let b = addr!("x", 42);
// Scoped
let c = scoped_addr!("block", "z");
// Scoped + indexed
let d = scoped_addr!("chain", "w", "{}", 7);
```

## Status and invariants

- Stable: Addressing scheme and `addr!` are stable.
- Invariants: Deterministic formatting; ordering is lexicographic; hashing matches inner string.

## See also

- `Model` building blocks: `sample`, `observe`, `traverse_vec`, `zip`
- Macros: `prob!`, `plate!`, `scoped_addr!`
