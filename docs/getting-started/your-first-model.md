# Your First Model

Now that Fugue is installed, let's build your first probabilistic model step by step. We'll start simple and gradually add complexity to help you understand the core concepts.

## The Simplest Model: A Constant

Let's start with the simplest possible model - one that always returns the same value:

```rust
use fugue::*;

fn constant_model() -> Model<f64> {
    pure(42.0)
}
```

This model always returns `42.0`. The `pure` function creates a deterministic `Model<f64>` containing a constant value.

## Adding Randomness: Sampling from a Distribution

Now let's add some randomness by sampling from a probability distribution:

```rust
use fugue::*;

fn random_model() -> Model<f64> {
    sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
}
```

This model:
- Samples a value from a standard normal distribution (mean=0, std=1)
- The `addr!("x")` gives this random variable a name
- Returns a `Model<f64>` that produces different values each time it runs

**Key insight**: Notice the `.unwrap()` - Fugue uses safe constructors that validate parameters and return `Result` types.

## Running Your Model

To actually get a value from your model, you need to "run" it with a handler:

```rust
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn main() {
    let model = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
    
    // Create a random number generator
    let mut rng = StdRng::seed_from_u64(42);
    
    // Run the model
    let (value, trace) = runtime::handler::run(
        runtime::interpreters::PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        model,
    );
    
    println!("Sampled value: {:.4}", value);
    println!("Log probability: {:.4}", trace.total_log_weight());
}
```

Running this gives:
```
Sampled value: 1.0175
Log probability: -0.9189
```

## Your First Bayesian Model

Let's create a simple Bayesian model where we estimate the mean of a normal distribution given an observation:

```rust
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn estimate_mean(observation: f64) -> Model<f64> {
    // Prior: we believe the mean is around 0, but we're uncertain
    sample(addr!("mu"), Normal::new(0.0, 2.0).unwrap())
        .bind(move |mu| {
            // Likelihood: observe data given our hypothesis about mu
            observe(addr!("y"), Normal::new(mu, 1.0).unwrap(), observation)
                .map(move |_| mu)  // Return the mean parameter
        })
}

fn main() {
    let observation = 3.0;  // We observed a value of 3.0
    let model = estimate_mean(observation);
    
    let mut rng = StdRng::seed_from_u64(42);
    let (estimated_mu, trace) = runtime::handler::run(
        runtime::interpreters::PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        model,
    );
    
    println!("Observation: {}", observation);
    println!("Estimated mean: {:.4}", estimated_mu);
    println!("Log probability: {:.4}", trace.total_log_weight());
}
```

## Understanding What Happened

In the Bayesian model above:

1. **Prior**: `sample(addr!("mu"), Normal::new(0.0, 2.0).unwrap())` 
   - We start with a prior belief that the mean is around 0 with uncertainty (std=2.0)

2. **Likelihood**: `observe(addr!("y"), Normal::new(mu, 1.0).unwrap(), observation)`
   - We observe data and compute how likely it is given our hypothesis

3. **Bind operation**: `.bind(move |mu| ...)`
   - This lets us use the sampled `mu` value in the rest of the model

4. **Return**: `.map(move |_| mu)`
   - We return the parameter we're interested in estimating

## Type Safety in Action

Fugue's type safety shines with discrete distributions. Let's flip a coin:

```rust
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn coin_flip_model() -> Model<String> {
    sample(addr!("coin"), Bernoulli::new(0.6).unwrap())  // 60% chance of heads
        .map(|is_heads| {
            // Notice: is_heads is a bool, not a float!
            if is_heads {
                "Heads".to_string()
            } else {
                "Tails".to_string()
            }
        })
}

fn main() {
    let model = coin_flip_model();
    
    let mut rng = StdRng::seed_from_u64(42);
    let (result, _trace) = runtime::handler::run(
        runtime::interpreters::PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        model,
    );
    
    println!("Coin flip result: {}", result);
    
    // Let's flip multiple times
    for i in 0..5 {
        let mut rng = StdRng::seed_from_u64(42 + i);
        let (result, _) = runtime::handler::run(
            runtime::interpreters::PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            coin_flip_model(),
        );
        println!("Flip {}: {}", i + 1, result);
    }
}
```

**Key insight**: `Bernoulli` returns `bool` directly - no need to compare with `1.0` like in other PPLs!

## Complete Working Example

Here's a complete, runnable example that demonstrates the concepts:

```rust
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

// A model that estimates bias of a coin given observations
fn coin_bias_model(heads_count: u64, total_flips: u64) -> Model<f64> {
    // Prior: uniform belief about coin bias
    sample(addr!("bias"), Beta::new(1.0, 1.0).unwrap())
        .bind(move |bias| {
            // Likelihood: observed heads given bias
            observe(
                addr!("heads"),
                Binomial::new(total_flips, bias).unwrap(),
                heads_count,
            )
            .map(move |_| bias)
        })
}

fn main() {
    println!("🎲 Estimating Coin Bias");
    println!("======================");
    
    // We observed 7 heads out of 10 flips
    let heads = 7;
    let total = 10;
    
    let model = coin_bias_model(heads, total);
    
    // Sample from the posterior
    for seed in 0..5 {
        let mut rng = StdRng::seed_from_u64(seed);
        let (bias, trace) = runtime::handler::run(
            runtime::interpreters::PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            model.clone(),
        );
        
        println!(
            "Sample {}: bias = {:.3}, log prob = {:.4}",
            seed + 1,
            bias,
            trace.total_log_weight()
        );
    }
    
    println!("\n✨ You've built your first Bayesian model!");
}
```

Save this as `src/main.rs` and run with `cargo run`.

## Key Takeaways

After working through these examples, you should understand:

1. **Models are values**: `Model<T>` represents a probabilistic computation
2. **Safe constructors**: Distributions use `.new().unwrap()` for parameter validation
3. **Type safety**: Distributions return natural types (`bool`, `u64`, `f64`)
4. **Addressing**: `addr!("name")` gives names to random variables
5. **Execution**: Models need handlers to actually run and produce values
6. **Composition**: Use `bind` and `map` to combine simple models into complex ones

## What's Next?

Now that you can build and run basic models:

1. **[Understanding Models](understanding-models.md)** - Deep dive into model composition and monadic operations
2. **[Basic Inference](basic-inference.md)** - Learn about MCMC, SMC, and other inference methods
3. **[Working with Distributions](../how-to/working-with-distributions.md)** - Master all distribution types

---

**Ready to understand models deeply?** → **[Understanding Models](understanding-models.md)**