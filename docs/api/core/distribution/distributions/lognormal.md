# Log-normal distribution

A continuous distribution where the logarithm of the random variable follows a normal distribution. This distribution is useful for modeling positive-valued quantities that are naturally multiplicative or skewed.

**Relationship to Normal:** If X ~ LogNormal(μ, σ), then ln(X) ~ Normal(μ, σ)
**Probability density function:**

```text
f(x) = (1 / (x * σ√(2π))) * exp(-0.5 * ((ln(x) - μ) / σ)²)  for x > 0
f(x) = 0                                                      for x ≤ 0
```

**Support:** (0, +∞)

## Fields

- `mu` - Mean of the underlying normal distribution
- `sigma` - Standard deviation of the underlying normal distribution (must be positive)

## Examples

```rust
use fugue::*;
// Standard log-normal
let log_normal = LogNormal::new(0.0, 1.0).unwrap();
// Model for positive scale parameters
let model = sample(addr!("scale"), LogNormal::new(0.0, 0.5).unwrap());
// Income distribution (often log-normal)
let income_model = sample(addr!("income"), LogNormal::new(10.0, 0.8).unwrap());
```
