# `Choice` struct

A recorded choice made during model execution.

Each choice represents a random variable assignment at a specific address, along with its log-probability under the distribution that generated it. Choices are the building blocks of execution traces.

## Examples

```rust
use fugue::*;

// Choices are typically created by handlers during execution
let choice = Choice {
    addr: addr!("x"),
    value: ChoiceValue::F64(1.5),
    logp: -0.92, // log-probability under some distribution
};

println!("Choice at {}: {:?} (logp: {})", choice.addr, choice.value, choice.logp);
```
