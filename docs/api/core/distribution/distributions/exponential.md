# Exponential distribution

A continuous probability distribution often used to model waiting timesbetween events in a Poisson process. It has a single parameter (rate) andis characterized by the memoryless property**Probability density function:**

```text
f(x) = λ * exp(-λx)  for x ≥ 0f(x) = 0             for x < 0
```

**Support:** [0, +∞)

## Field

- `rate` - Rate parameter λ (must be positive). Higher values = shorter waiting times.
