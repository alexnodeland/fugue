# `Distribution` trait

Generic interface for type-safe probability distributions.

This trait provides the essential operations needed for probabilistic programming with **full type safety**. Unlike traditional PPLs that force all distributions to return `f64`, Fugue's `Distribution<T>` trait is generic over the sample type `T`, enabling:

- **Natural return types**: Each distribution returns its mathematically appropriate type
- **Compile-time safety**: Type errors are caught by the compiler, not at runtime
- **Zero overhead**: No unnecessary type conversions or boxing
- **Intuitive code**: Write code that matches statistical intuition

## Type Safety Benefits

| Distribution | Traditional PPL        | Fugue Type-Safe             |
| ------------ | ---------------------- | --------------------------- |
| Bernoulli    | `f64` (0.0/1.0)        | **`bool`** (true/false)     |
| Poisson      | `f64` (needs casting)  | **`u64`** (natural counts)  |
| Categorical  | `f64` (risky indexing) | **`usize`** (safe indexing) |
| Binomial     | `f64` (needs casting)  | **`u64`** (natural counts)  |
| Normal       | `f64` ✓                | **`f64`** ✓                 |

## Required Methods

- [`sample`](Self::sample): Generate a random sample of type `T`
- [`log_prob`](Self::log_prob): Compute log-probability of a value of type `T`
- [`clone_box`](Self::clone_box): Clone into a boxed trait object

## Examples

```rust
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
let mut rng = StdRng::seed_from_u64(42);
// Continuous distribution returns f64
let normal = Normal::new(0.0, 1.0).unwrap();
let value: f64 = normal.sample(&mut rng);
let log_prob = normal.log_prob(&value);
// Discrete distributions return natural types!
let coin = Bernoulli::new(0.5).unwrap();
let flip: bool = coin.sample(&mut rng);  // bool, not f64!
let coin_prob = coin.log_prob(&flip);
let counter = Poisson::new(3.0).unwrap();
let count: u64 = counter.sample(&mut rng);  // u64, not f64!
let count_prob = counter.log_prob(&count);
let choice = Categorical::new(vec![0.3, 0.5, 0.2]).unwrap();
let idx: usize = choice.sample(&mut rng);  // usize for safe indexing!
let choice_prob = choice.log_prob(&idx);
```
