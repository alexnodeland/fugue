# `Address` struct

A unique identifier for random variables and observation sites in probabilistic models. Addresses serve as stable names for probabilistic choices, enabling conditioning, inference, and replay. They are implemented as wrapped strings with ordering and hashing support for use in collections.

## Examples

```rust
use fugue::*;
// Create addresses using the addr! macro
let addr1 = addr!("parameter");
let addr2 = addr!("data", 5);
// Addresses can be compared and used in collections
use std::collections::HashMap;
let mut map = HashMap::new();
map.insert(addr1, 1.0);
map.insert(addr2, 2.0);
```
