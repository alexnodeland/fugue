# Addressing and Site Naming

Addressing and site naming utilities for probabilistic programs. Addresses are crucial for probabilistic programming as they uniquely identify random choices and observation sites within a model. This enables:

- **Conditioning**: Observing specific values at named sites
- **Inference**: Tracking which random variables to infer
- **Replay**: Reproducing exact execution paths from recorded traces
- **Debugging**: Understanding model structure and execution flow

The `addr!` macro provides a concise, stable way to create addresses from human-readable names with optional indices for handling collections and repeated structures.

## Address Creation

```rust
use fugue::*;
// Simple named address
let mu_addr = addr!("mu");
// Indexed address for collections
let data_addr = addr!("data", 0);
// Addresses are unique
assert_ne!(addr!("mu"), addr!("mu", 0));
assert_ne!(addr!("x", 1), addr!("x", 2));
```

## Best Practices

- Use descriptive names that reflect the semantic meaning
- Use indices for repeated structures (loops, arrays, etc.)
- Keep address names consistent across model runs for reproducibility
- Avoid dynamic address generation in inference loops
