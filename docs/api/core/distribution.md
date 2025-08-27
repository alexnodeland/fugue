# Distributions

Fugue's distribution system solves a fundamental problem in probabilistic programming: **type safety without sacrificing statistical expressiveness**. This document explains the architectural decisions behind Fugue's type-safe distributions, when to use each distribution, and how they compose with the Model system.

## The Type Safety Problem

Traditional probabilistic programming libraries force all distributions to return `f64`, leading to:

- **Runtime errors**: `array[sample.round() as usize]` can panic
- **Awkward comparisons**: `if bernoulli_sample == 1.0` instead of natural boolean logic
- **Casting overhead**: Converting counts back to integers for arithmetic
- **Semantic confusion**: Is this `f64` a probability, count, or continuous value?

## Fugue's Solution: Natural Return Types

Each distribution returns its **mathematically appropriate type**, enabling:

| Problem            | Traditional PPL        | Fugue Solution | Benefit                                 |
| ------------------ | ---------------------- | -------------- | --------------------------------------- |
| Boolean outcomes   | `f64` (0.0/1.0)        | **`bool`**     | Natural `if` statements, no comparisons |
| Count data         | `f64` (needs casting)  | **`u64`**      | Direct arithmetic, no precision loss    |
| Category selection | `f64` (risky indexing) | **`usize`**    | Safe array indexing, no bounds checking |
| Continuous values  | `f64` ✓                | **`f64`** ✓    | Unchanged, as expected                  |

## Distribution Selection Guide

### When to use each distribution

| Use Case                      | Distribution         | Return Type | Key Benefit                    |
| ----------------------------- | -------------------- | ----------- | ------------------------------ |
| **Binary decisions**          | `Bernoulli`          | `bool`      | Natural branching logic        |
| **Count processes**           | `Poisson`            | `u64`       | Direct arithmetic on counts    |
| **Success counting**          | `Binomial`           | `u64`       | Natural trial counting         |
| **Category selection**        | `Categorical`        | `usize`     | Safe array/vec indexing        |
| **Continuous parameters**     | `Normal`             | `f64`       | Standard continuous modeling   |
| **Positive scales**           | `LogNormal`, `Gamma` | `f64`       | Natural for variance, rates    |
| **Probabilities/proportions** | `Beta`               | `f64`       | Conjugate priors for Bernoulli |
| **Waiting times**             | `Exponential`        | `f64`       | Memoryless processes           |
| **Bounded intervals**         | `Uniform`            | `f64`       | Uninformative priors           |

### Decision flowchart

```text
Is your random variable...
├─ Binary (yes/no, success/failure)? → Bernoulli → bool
├─ A count (0, 1, 2, ...)?
│  ├─ Fixed trials? → Binomial → u64
│  └─ Rate-based events? → Poisson → u64
├─ A category choice? → Categorical → usize
└─ Continuous?
   ├─ Unbounded? → Normal → f64
   ├─ Positive only?
   │  ├─ Multiplicative/skewed? → LogNormal → f64
   │  └─ Rate/scale parameter? → Gamma/Exponential → f64
   ├─ On [0,1]? → Beta → f64
   └─ Bounded interval? → Uniform → f64
```

## Architectural Patterns

### Pattern: Type-Safe Branching

```rust
use fugue::*;

// ✅ Natural boolean logic - no comparisons needed
let strategy = sample(addr!("risky"), Bernoulli::new(0.3).unwrap())
    .bind(|take_risk| {
        if take_risk {  // Direct boolean usage!
            sample(addr!("high_reward"), Normal::new(10.0, 3.0).unwrap())
        } else {
            sample(addr!("safe_reward"), Normal::new(5.0, 1.0).unwrap())
        }
    });
```

### Pattern: Safe Indexing

```rust
# use fugue::*;
# 
// ✅ No casting, no bounds checking needed
let options = vec!["aggressive", "moderate", "conservative"];
let choice = sample(addr!("strategy"), Categorical::uniform(3).unwrap())
    .map(move |idx| options[idx].to_string());  // Direct, safe indexing!
```

### Pattern: Count Arithmetic

```rust
use fugue::*;

// ✅ Direct arithmetic on natural count types
let events = sample(addr!("events"), Poisson::new(4.0).unwrap())
    .bind(|count| {
        let bonus = if count > 5 { count * 2 } else { count };  // Direct u64 arithmetic!
        pure(bonus)
    });
```

### Pattern: Hierarchical Modeling

```rust
use fugue::*;

// Type-safe hierarchical model with natural conjugacy
let model = sample(addr!("success_rate"), Beta::new(2.0, 5.0).unwrap())  // Prior
    .bind(|p| {
        sample(addr!("trials"), Binomial::new(20, p).unwrap())  // Likelihood → u64
            .bind(|successes| {
                let rate = successes as f64 / 20.0;  // Natural conversion when needed
                pure(rate)
            })
    });
```

## Design Principles

### 1. **Type Safety First**

Every distribution returns the type that makes semantic sense for its domain, eliminating a whole class of runtime errors.

### 2. **Zero-Cost Abstractions**

No boxing, no dynamic dispatch for common operations. The type system does the work at compile time.

### 3. **Composability**

Distributions work seamlessly with Fugue's `Model` system and with each other in hierarchical structures.

### 4. **Statistical Correctness**

All implementations use numerically stable algorithms with proper parameter validation.

### 5. **Rust Idioms**

Distributions feel natural in Rust code - no fighting the type system or borrowing rules.

## Integration with Model System

### Dual Usage Pattern

Distributions work both **inside** and **outside** the Model system:

```rust
use fugue::*;
use rand::thread_rng;

// Inside Model system (for probabilistic programs)
let model: Model<bool> = sample(addr!("coin"), Bernoulli::new(0.5).unwrap());

// Outside Model system (for direct statistical computation)
let coin = Bernoulli::new(0.5).unwrap();
let flip: bool = coin.sample(&mut thread_rng());
let prob: f64 = coin.log_prob(&true);
```

### Handler Compatibility

All distributions work with every Fugue handler (prior, replay, MCMC, etc.) without modification - the type safety is preserved throughout the inference pipeline.

## Evolution Strategy

- **Stable API**: The `Distribution<T>` trait and core distributions are considered stable
- **Extensibility**: New distributions follow the same type-safe pattern
- **Backwards Compatibility**: Adding distributions doesn't break existing code
- **Performance**: Optimizations happen at the implementation level, not the interface

## Common Anti-Patterns

❌ **Don't cast unnecessarily**

```rust
# use fugue::*;
# 
// Bad - unnecessary casting
let count = sample(addr!("count"), Poisson::new(3.0).unwrap())
    .map(|c| c as f64);
```

❌ **Don't use f64 distributions for discrete data**

```rust
# use fugue::*;
# 
// Bad - using Normal for binary choice
let choice = sample(addr!("choice"), Normal::new(0.5, 0.1).unwrap())
    .map(|x| x > 0.5);  // Error-prone!
```

✅ **Do use natural types**

```rust
# use fugue::*;
# 
// Good - let the type system help you
let choice = sample(addr!("choice"), Bernoulli::new(0.5).unwrap());
let count = sample(addr!("count"), Poisson::new(3.0).unwrap());
```

## See Also

- **Implementation**: Individual distribution docs for parameter details and mathematical properties
- **Model Integration**: How distributions compose with `sample()` and `observe()` in the Model system
- **Inference**: How type-safe distributions work with MCMC, VI, and other inference algorithms
- **Examples**: Real-world usage patterns in `examples/` directory
