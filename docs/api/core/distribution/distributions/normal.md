# Normal (Gaussian) distribution

The normal distribution is a continuous probability distribution characterized by its mean (μ) and standard deviation (σ). It's one of the most important distributions in statistics and is commonly used as a prior or likelihood in Bayesian models.

**Probability density function:**

```text
f(x) = (1 / (σ√(2π))) * exp(-0.5 * ((x - μ) / σ)²)
```

**Support:** All real numbers (-∞, +∞)

## Fields

- `mu` - Mean of the distribution
- `sigma` - Standard deviation (must be positive)

## Examples

```rust
use fugue::*;
// Standard normal distribution
let std_normal = Normal::new(0.0, 1.0).unwrap();
// Normal prior for a parameter
let model = sample(addr!("theta"), Normal::new(0.0, 2.0).unwrap());
// Normal likelihood for observations
let model = observe(addr!("y"), Normal::new(1.5, 0.5).unwrap(), 2.0);
```
