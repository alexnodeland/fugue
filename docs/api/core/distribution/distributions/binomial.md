# Type-safe Binomial distribution

**→ returns `u64`**

A discrete distribution representing the number of successes in n independent Bernoulli trials, each with success probability p. This distribution models counting processes and is widely used in statistics.

## 🎯 Type Safety Innovation

**Unlike traditional PPLs**, Fugue's Binomial distribution returns **`u64` directly**, providing natural counting semantics for the number of successes without casting.
**Probability mass function:**

```text
P(X = k) = C(n,k) * p^k * (1-p)^(n-k)  for k ∈ {0, 1, ..., n}
```

where C(n,k) is the binomial coefficient "n choose k".

**Support:** {0, 1, ..., n} (natural success counts!)

## Fields

- `n` - Number of trials (must be non-negative)
- `p` - Probability of success on each trial (must be in [0, 1])

## Examples

```rust
use fugue::*;
// Type-safe success counting - returns u64 directly!
let trial_model: Model<u64> = sample(addr!("successes"), Binomial::new(10, 0.5).unwrap());
let analysis = trial_model.bind(|success_count| {
    // success_count is naturally u64 - can be used in arithmetic directly
    let success_rate = success_count as f64 / 10.0;
    let verdict = if success_rate > 0.7 {
        "High success rate!"
    } else if success_rate < 0.3 {
        "Low success rate"
    } else {
        "Moderate success rate"
    };
    pure(verdict.to_string())
});
// Clinical trial with type-safe counting
let clinical_trial = sample(addr!("success_rate"), Beta::new(1.0, 1.0).unwrap())
    .bind(|p| {
        sample(addr!("successes"), Binomial::new(100, p).unwrap())
            .bind(|successes| {
                // successes is naturally u64 - no casting needed!
                let efficacy = successes as f64 / 100.0;
                pure(efficacy)
            })
    });
// Type-safe observation of trial results
let obs_model = observe(addr!("trial_successes"), Binomial::new(20, 0.3).unwrap(), 7u64);
```
