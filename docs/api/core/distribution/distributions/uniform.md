# Uniform distribution over a continuous interval

The uniform distribution assigns equal probability density to all values within a specified interval [low, high) and zero probability outside.

**Probability density function:**

```text
f(x) = 1 / (high - low)  for low ≤ x < high
f(x) = 0                 otherwise
```

**Support:** [low, high)

## Fields

- `low` - Lower bound of the distribution (inclusive)
- `high` - Upper bound of the distribution (exclusive)

## Examples

```rust
use fugue::*;
// Unit interval
let unit_uniform = Uniform::new(0.0, 1.0).unwrap();
// Symmetric interval around zero
let symmetric = Uniform::new(-5.0, 5.0).unwrap();
// Use as uninformative prior
let model = sample(addr!("weight"), Uniform::new(0.0, 100.0).unwrap());
```
