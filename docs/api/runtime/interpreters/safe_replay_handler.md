# `SafeReplayHandler` struct

Safe version of ReplayHandler that uses type-safe trace accessors.

This handler replays a model execution using values stored in a base trace, but gracefully handles type mismatches by logging warnings and falling back to fresh sampling instead of panicking.

## Examples

```rust
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
let base_trace = Trace::default(); // From previous execution
let mut rng = StdRng::seed_from_u64(42);
let handler = SafeReplayHandler {
    rng: &mut rng,
    base: base_trace,
    trace: Trace::default(),
    warn_on_mismatch: true,
};
```
