# Beta distribution on the interval [0, 1]

A continuous distribution over the unit interval, commonly used for modeling probabilities, proportions, and as a conjugate prior for Bernoulli/Binomial distributions. The shape is controlled by two positive parameters α and β.

**Probability density function:**

```text
f(x) = (x^(α-1) * (1-x)^(β-1)) / B(α,β)  for 0 < x < 1
f(x) = 0                                  otherwise
```

where B(α,β) is the beta function.

**Support:** (0, 1)

## Fields

- `alpha` - First shape parameter α (must be positive)
- `beta` - Second shape parameter β (must be positive)

## Examples

```rust
use fugue::*;
// Uniform on [0,1] (alpha=1, beta=1)
let uniform_beta = Beta::new(1.0, 1.0).unwrap();
// Prior for a probability parameter
let prob_prior = sample(addr!("p"), Beta::new(2.0, 5.0).unwrap());
// Conjugate prior for Bernoulli likelihood
let model = sample(addr!("success_rate"), Beta::new(3.0, 7.0).unwrap())
    .bind(|p| observe(addr!("trial"), Bernoulli::new(p).unwrap(), true));
```
