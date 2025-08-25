# Type-safe Bernoulli distribution

**→ returns `bool`**

A discrete distribution representing a single trial with two possible outcomes: success (true) with probability p, or failure (false) with probability 1-p. This is the building block for binomial distributions and binary classification.

## 🎯 Type Safety Innovation

**Unlike traditional PPLs**, Fugue's Bernoulli distribution returns **`bool` directly**, eliminating error-prone floating-point comparisons like `if sample == 1.0`.
**Probability mass function:**

```text
P(X = true) = p
P(X = false) = 1 - p
```

**Support:** {false, true} (natural boolean values!)

## Fields

- `p` - Probability of success (must be in [0, 1])

## Examples

```rust
use fugue::*;
// Type-safe boolean sampling - no more f64 comparisons!
let coin_model: Model<bool> = sample(addr!("coin"), Bernoulli::new(0.5).unwrap());
let decision = coin_model.bind(|heads| {
    if heads {  // ✅ Natural boolean usage!
        pure("Heads - take action!".to_string())
    } else {
        pure("Tails - wait...".to_string())
    }
});
// Mixture component selection with natural boolean logic
let component_model = sample(addr!("component"), Bernoulli::new(0.3).unwrap())
    .bind(|is_component_2| {
        let component_name = if is_component_2 {
            "Component 2"
        } else {
            "Component 1"
        };
        pure(component_name.to_string())
    });
// Type-safe observation of boolean outcomes
let obs_model = observe(addr!("success"), Bernoulli::new(0.8).unwrap(), true);
```
