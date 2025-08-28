# Your First Model

Now that Fugue is installed, let's build your first probabilistic model step by step. We'll start simple and gradually introduce the key concepts.

```admonish note
Learning Goals

In 5 minutes, you'll understand:
- How to create deterministic and probabilistic models
- The role of addresses in probabilistic programming
- How to condition models on observed data
- Fugue's type-safe approach to distributions

**Time**: ~5 minutes
```

## Step 1: The Simplest Model

Let's start with the simplest possible model - one that always returns the same value:

```rust,ignore
use fugue::*;

fn constant_model() -> Model<f64> {
    pure(42.0)
}
```

This model always returns `42.0`. The `pure` function creates a deterministic `Model<f64>`.

**Key insight**: Models are **descriptions** of computations, not the computations themselves.

## Step 2: Adding Randomness

Now let's add some randomness by sampling from a probability distribution:

```rust,ignore
use fugue::*;

fn random_model() -> Model<f64> {
    sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
}
```

```admonish note
New Concepts

- `sample()` - Draw a random value from a distribution
- `addr!("x")` - Give this random choice a unique name/address  
- `Normal::new(0.0, 1.0).unwrap()` - Standard normal distribution (mean=0, std=1)
- `.unwrap()` - Fugue uses safe constructors that validate parameters
```

## Step 3: Running Your Model

To actually get values from your model, you need to "run" it with a **handler**:

```rust,ignore
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn main() {
    let model = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
    
    // Create a seeded random number generator
    let mut rng = StdRng::seed_from_u64(42);
    
    // Run the model with PriorHandler (forward sampling)
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

Running this outputs:

```text
Sampled value: 1.0175
Log probability: -0.9189
```

```admonish tip
Understanding the Output

- **`value`** - The random sample from our distribution
- **`trace`** - Records what happened during execution (choices made, probabilities)
- **`log_probability`** - How likely this particular execution was
```

## Step 4: Type Safety in Action

Fugue's type safety really shines with discrete distributions:

```rust,ignore
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn type_safe_examples() {
    let mut rng = StdRng::seed_from_u64(42);
    
    // Flip a coin - returns bool directly!
    let (is_heads, _) = runtime::handler::run(
        runtime::interpreters::PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        sample(addr!("coin"), Bernoulli::new(0.6).unwrap()),
    );
    
    // Natural boolean usage - no comparisons needed!
    let outcome = if is_heads { "Heads" } else { "Tails" };
    println!("Coin flip: {}", outcome);
    
    // Count events - returns u64 directly!
    let (count, _) = runtime::handler::run(
        runtime::interpreters::PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        sample(addr!("events"), Poisson::new(3.0).unwrap()),
    );
    
    println!("Event count: {} (no casting needed!)", count);
    
    // Choose category - returns usize for safe indexing!
    let colors = vec!["red", "green", "blue", "yellow"];
    let (idx, _) = runtime::handler::run(
        runtime::interpreters::PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        sample(addr!("color"), Categorical::uniform(4).unwrap()),
    );
    
    println!("Chosen color: {}", colors[idx]); // Safe indexing!
}
```

````admonish warning
Contrast with Other PPLs

In most probabilistic programming languages:
```python
# Other PPLs - everything returns float
coin = sample("coin", Bernoulli(0.6))  # Returns 0.0 or 1.0 
if coin == 1.0:  # Need comparison ❌
    ...

count = sample("events", Poisson(3.0))  # Returns float
count_int = int(count)  # Need casting ❌

idx = sample("color", Categorical([...]))  # Returns float  
colors[int(idx)]  # Risky casting and indexing ❌
```

Fugue prevents these errors at compile time! ✅
````

## Step 5: Your First Bayesian Model

Now let's create a simple Bayesian model that learns from data:

```rust,ignore
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn estimate_mean(observation: f64) -> Model<f64> {
    sample(addr!("mu"), Normal::new(0.0, 2.0).unwrap())  // Prior belief
        .bind(move |mu| {
            observe(addr!("y"), Normal::new(mu, 1.0).unwrap(), observation)  // Likelihood
                .map(move |_| mu)  // Return the parameter
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

```admonish note
What Just Happened?

This is a complete Bayesian inference setup:

1. **Prior**: `sample(addr!("mu"), Normal::new(0.0, 2.0).unwrap())`
   - Our initial belief about the mean (uncertain, centered at 0)

2. **Likelihood**: `observe(addr!("y"), Normal::new(mu, 1.0).unwrap(), observation)`  
   - How likely our observation is, given different values of `mu`

3. **Bind**: `.bind(move |mu| ...)`
   - Use the sampled `mu` in the rest of the model

4. **Return**: `.map(move |_| mu)`
   - Return the parameter we want to estimate
```

## Understanding Model Composition

Fugue models compose using two key operations:

### `map` - Transform Values

```rust,ignore
let doubled = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
    .map(|x| x * 2.0);  // Apply function to the result
```

### `bind` - Dependent Computations

```rust,ignore
let dependent = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
    .bind(|x| sample(addr!("y"), Normal::new(x, 0.5).unwrap()));  // y depends on x
```

```admonish tip
Mental Model

- `map` = "transform the output"
- `bind` = "use the output in the next step"

These are the fundamental building blocks for complex probabilistic models!
```

## Key Takeaways

After working through these examples, you should understand:

✅ **Models are values**: `Model<T>` represents a probabilistic computation  
✅ **Safe constructors**: Distributions use `.new().unwrap()` for parameter validation  
✅ **Type safety**: Distributions return natural types (`bool`, `u64`, `f64`)  
✅ **Addressing**: `addr!("name")` gives names to random variables  
✅ **Execution**: Models need handlers to run and produce values  
✅ **Composition**: Use `map` and `bind` to build complex models from simple parts

## What's Next?

You can now build and run basic probabilistic models! 🎉

**Continue your journey:**

```admonish tip
Next Steps

- **[Understanding Models](understanding-models.md)** - Deep dive into model composition and addressing
- **[Running Inference](basic-inference.md)** - Learn about MCMC and other inference methods

**Ready for complete projects?**
- **[Bayesian Coin Flip Tutorial](../tutorials/bayesian-coin-flip.md)** - Your first end-to-end analysis
```

---

**Time**: ~5 minutes • **Next**: [Understanding Models](understanding-models.md)
