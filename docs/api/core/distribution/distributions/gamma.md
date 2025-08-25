# Gamma distribution

A continuous probability distribution over positive real numbers, parameterized by shape (k) and rate (λ). The Gamma distribution is commonly used for modeling waiting times, scale parameters, and as a conjugate prior for Poisson distributions.

**Probability density function:**

```text
f(x) = (λ^k / Γ(k)) * x^(k-1) * exp(-λx)  for x > 0
f(x) = 0                                   for x ≤ 0
```

where Γ(k) is the gamma function.

**Support:** (0, +∞)

## Fields

- `shape` - Shape parameter k (must be positive)
- `rate` - Rate parameter λ (must be positive). Note: rate = 1/scale

## Examples

```rust
use fugue::*;
// Exponential is Gamma(1, rate)
let exponential_like = Gamma::new(1.0, 2.0).unwrap();
// Prior for precision (inverse variance)
let precision = sample(addr!("precision"), Gamma::new(2.0, 1.0).unwrap());
// Conjugate prior for Poisson rate
let model = sample(addr!("rate"), Gamma::new(3.0, 2.0).unwrap())
    .bind(|lambda| observe(addr!("count"), Poisson::new(lambda).unwrap(), 5u64));
```
