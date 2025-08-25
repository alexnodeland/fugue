# Type-safe Poisson distribution

**→ returns `u64`**

A discrete probability distribution expressing the probability of a given number of events occurring in a fixed interval of time or space, given that these events occur with a known constant mean rate and independently of each other.

## 🎯 Type Safety Innovation

**Unlike traditional PPLs**, Fugue's Poisson distribution returns **`u64` directly**, providing natural counting semantics without error-prone casting from `f64`.
**Probability mass function:**

```text
P(X = k) = (λ^k * exp(-λ)) / k!  for k ∈ {0, 1, 2, ...}
```

**Support:** {0, 1, 2, ...} (natural non-negative integers!)

## Fields

- `lambda` - Rate parameter λ (must be positive). This is both the mean and variance.

## Examples

```rust
use fugue::*;
// Type-safe count modeling - returns u64 directly!
let count_model: Model<u64> = sample(addr!("events"), Poisson::new(3.5).unwrap());
let analysis = count_model.bind(|count| {
    // count is naturally u64 - can be used directly in match patterns
    let status = match count {
        0 => "No events occurred",
        1 => "Single event occurred",
        n if n > 10 => "High activity period!",
        n => &format!("{} events occurred", n),
    };
    pure(status.to_string())
});
// Hierarchical count modeling with type safety
let hierarchical = sample(addr!("rate"), Gamma::new(2.0, 1.0).unwrap())
    .bind(|lambda| {
        sample(addr!("count"), Poisson::new(lambda).unwrap())
            .bind(|count| {
                // count is naturally u64 - no casting needed!
                let bonus = if count > 5 { count * 2 } else { count };
                pure(bonus)
            })
    });
// Type-safe observation of count data
let obs_model = observe(addr!("observed_count"), Poisson::new(4.0).unwrap(), 7u64);
```
