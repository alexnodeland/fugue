# `ReplayHandler` struct

Handler for replaying models with values from an existing trace. This handler replays a model execution using values stored in a base trace. When a sampling site is encountered:

- If the address exists in the base trace, use that value
- If the address is missing, sample a fresh value from the distribution

This is essential for MCMC algorithms where you want to replay most of a trace but sample new values at specific sites that are being updated.

## Fields

- `rng` - Random number generator for sampling at missing addresses
- `base` - Existing trace containing values to replay
- `trace` - New trace to accumulate the replay execution

## Examples

```rust
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
// First, create a base trace
let model_fn = || sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
let mut rng = StdRng::seed_from_u64(123);
let (original_value, base_trace) = runtime::handler::run(
    PriorHandler { rng: &mut rng, trace: Trace::default() },
    model_fn()
);
// Now replay the model using the base trace
let mut rng2 = StdRng::seed_from_u64(456);
let (replayed_value, new_trace) = runtime::handler::run(
    ReplayHandler {
        rng: &mut rng2,
        base: base_trace,
        trace: Trace::default(),
    },
    model_fn(),
);
// replayed_value will be the same as the original value
assert_eq!(original_value, replayed_value);
```
