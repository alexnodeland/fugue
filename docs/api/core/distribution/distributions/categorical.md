# Type-safe Categorical distribution

**→ returns `usize`**

A discrete distribution that represents choosing among k different categories with specified probabilities. The outcome is the index of the chosen category as a `usize`, making it **naturally suitable for safe array indexing**.

## 🎯 Type Safety Innovation

**Unlike traditional PPLs**, Fugue's Categorical distribution returns **`usize` directly**, enabling safe array indexing without error-prone casting from `f64`.
**Probability mass function:**

```text
P(X = i) = probs[i]  for i ∈ {0, 1, ..., k-1}
```

**Support:** {0, 1, ..., k-1} where k = probs.len() (natural array indices!)

## Fields

- `probs` - Vector of probabilities for each category (should sum to 1.0)

## Examples

```rust
use fugue::*;
// Type-safe categorical choice - returns usize directly!
let options = vec!["red", "green", "blue"];
let color_model: Model<usize> = sample(addr!("color"),
    Categorical::new(vec![0.5, 0.3, 0.2]).unwrap());
let result = color_model.bind(move |color_idx| {
    // color_idx is naturally usize - safe for direct array indexing!
    let chosen_color = options[color_idx]; // No casting, no bounds checking needed!
    pure(chosen_color.to_string())
});
// Multi-armed bandit with type-safe action selection
let action_model = sample(addr!("action"),
    Categorical::new(vec![0.4, 0.3, 0.2, 0.1]).unwrap()  // 4 possible actions
).bind(|action_idx| {
    let action_rewards = vec![10.0, 15.0, 5.0, 20.0];
    let reward = action_rewards[action_idx]; // Direct, safe indexing!
    pure(reward)
});
// Type-safe observation of categorical outcomes
let obs_model = observe(addr!("user_choice"),
    Categorical::new(vec![0.2, 0.3, 0.3, 0.2]).unwrap(), 2usize);  // Observed choice was index 2
```
