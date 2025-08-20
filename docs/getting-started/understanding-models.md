# Understanding Models

In Fugue, everything revolves around the `Model<T>` type. Understanding how models work, compose, and transform is key to becoming productive with probabilistic programming. This guide provides a deep dive into the core concepts.

## What is a Model?

A `Model<T>` represents a probabilistic computation that, when executed, produces a value of type `T`. Think of it as a "recipe" or "program" that can be run multiple times to get different results.

```rust
use fugue::*;

// Model<f64> - produces floating point numbers
let normal_sample: Model<f64> = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());

// Model<bool> - produces boolean values  
let coin_flip: Model<bool> = sample(addr!("coin"), Bernoulli::new(0.5).unwrap());

// Model<u64> - produces counts
let event_count: Model<u64> = sample(addr!("events"), Poisson::new(3.0).unwrap());

// Model<String> - can produce any type!
let message: Model<String> = pure("Hello, probabilistic world!".to_string());
```

**Key insight**: Models are pure values - they don't do anything until you run them with a handler.

## The Four Types of Models

Internally, every `Model<T>` is one of four types:

### 1. Pure - Deterministic Values

```rust
let constant: Model<f64> = pure(42.0);
let computed: Model<f64> = pure(2.0 + 3.0 * 7.0);
```

Pure models always return the same value. They're useful for constants and deterministic computations.

### 2. Sample - Drawing from Distributions

```rust
let normal: Model<f64> = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
let coin: Model<bool> = sample(addr!("fair_coin"), Bernoulli::new(0.5).unwrap());
```

Sample models draw random values from probability distributions.

### 3. Observe - Conditioning on Data

```rust
let observation: Model<()> = observe(
    addr!("measurement"),
    Normal::new(0.0, 1.0).unwrap(),
    2.5  // observed value
);
```

Observe models condition the model on observed data, affecting the overall probability.

### 4. Factor - Soft Constraints

```rust
let soft_constraint: Model<()> = factor(-0.5);  // Reduces probability by exp(-0.5)
```

Factor models add log-weights to encourage or discourage certain outcomes.

## Monadic Composition

Models compose using monadic operations. The three key operations are:

### `pure` - Lift Values into Models

```rust
let value: f64 = 42.0;
let model: Model<f64> = pure(value);
```

### `map` - Transform Model Values

```rust
let original: Model<f64> = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
let transformed: Model<f64> = original.map(|x| x * 2.0 + 1.0);
let absolute: Model<f64> = original.map(|x| x.abs());
let string: Model<String> = original.map(|x| format!("Value: {:.2}", x));
```

`map` lets you transform the value inside a model without changing its probabilistic structure.

### `bind` - Sequential Composition

```rust
let sequential: Model<f64> = 
    sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
        .bind(|x| {
            // Use x to create the next model
            sample(addr!("y"), Normal::new(x, 0.5).unwrap())
        });
```

`bind` (also called `and_then`) lets you use the result of one model to create another model. This is how you build complex, dependent probabilistic programs.

## Practical Composition Examples

### Example 1: Hierarchical Model

```rust
fn hierarchical_model() -> Model<Vec<f64>> {
    // Global parameter
    sample(addr!("global_mean"), Normal::new(0.0, 1.0).unwrap())
        .bind(|global_mean| {
            // Local parameters depend on global
            let local_models: Vec<Model<f64>> = (0..5)
                .map(|i| {
                    sample(
                        addr!("local", i),
                        Normal::new(global_mean, 0.1).unwrap()
                    )
                })
                .collect();
            
            // Collect all local samples
            sequence_vec(local_models)
        })
}
```

### Example 2: Conditional Logic

```rust
fn conditional_model(use_alternative: bool) -> Model<f64> {
    if use_alternative {
        sample(addr!("alt"), Exponential::new(2.0).unwrap())
    } else {
        sample(addr!("std"), Normal::new(0.0, 1.0).unwrap())
    }
}

fn probabilistic_choice() -> Model<f64> {
    sample(addr!("choice"), Bernoulli::new(0.3).unwrap())
        .bind(|use_alt| conditional_model(use_alt))
}
```

### Example 3: Data-Driven Model

```rust
fn regression_model(x_data: &[f64], y_data: &[f64]) -> Model<(f64, f64)> {
    // Parameters
    sample(addr!("slope"), Normal::new(0.0, 1.0).unwrap())
        .bind(|slope| {
            sample(addr!("intercept"), Normal::new(0.0, 1.0).unwrap())
                .bind(move |intercept| {
                    sample(addr!("noise"), LogNormal::new(0.0, 0.5).unwrap())
                        .bind(move |noise| {
                            // Observations
                            let obs_models: Vec<Model<()>> = x_data
                                .iter()
                                .zip(y_data.iter())
                                .enumerate()
                                .map(|(i, (&x, &y))| {
                                    let y_pred = slope * x + intercept;
                                    observe(
                                        addr!("y", i),
                                        Normal::new(y_pred, noise).unwrap(),
                                        y
                                    )
                                })
                                .collect();
                            
                            // Run all observations, return parameters
                            sequence_vec(obs_models)
                                .map(move |_| (slope, intercept))
                        })
                })
        })
}
```

## The `prob!` Macro - Do Notation

Writing nested `bind` calls can get verbose. The `prob!` macro provides imperative-style syntax:

### Without `prob!` (verbose):

```rust
let verbose_model = 
    sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
        .bind(|x| {
            sample(addr!("y"), Normal::new(x, 0.5).unwrap())
                .bind(move |y| {
                    observe(addr!("z"), Normal::new(x + y, 0.1).unwrap(), 2.0)
                        .map(move |_| (x, y))
                })
        });
```

### With `prob!` (clean):

```rust
let clean_model = prob! {
    let x <- sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
    let y <- sample(addr!("y"), Normal::new(x, 0.5).unwrap());
    observe(addr!("z"), Normal::new(x + y, 0.1).unwrap(), 2.0);
    pure((x, y))
};
```

### `prob!` Syntax Rules

- `let var <- model_expr;` - Bind a model result to a variable
- `let var = expr;` - Regular variable binding (deterministic)
- `model_expr;` - Execute a model and ignore its result
- `pure(value)` - Final expression (can be any model)

```rust
let example = prob! {
    let x <- sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
    let doubled = x * 2.0;  // Regular assignment
    let y <- sample(addr!("y"), Normal::new(doubled, 0.1).unwrap());
    
    // Conditional logic
    if y > 0.0 {
        observe(addr!("pos"), Normal::new(y, 0.1).unwrap(), 1.5);
    } else {
        observe(addr!("neg"), Normal::new(-y, 0.1).unwrap(), 0.8);
    }
    
    pure((x, y, doubled))
};
```

## Working with Collections

### The `plate!` Macro

For collections of independent random variables:

```rust
// Generate 10 independent samples
let samples: Model<Vec<f64>> = plate!(i in 0..10 => {
    sample(addr!("x", i), Normal::new(0.0, 1.0).unwrap())
});

// With shared parameters
let hierarchical: Model<Vec<f64>> = prob! {
    let global_mu <- sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap());
    let local_samples <- plate!(i in 0..5 => {
        sample(addr!("local", i), Normal::new(global_mu, 0.1).unwrap())
    });
    pure(local_samples)
};
```

### Manual Collection Handling

For more control, use `sequence_vec` and `traverse_vec`:

```rust
// Sequence: Convert Vec<Model<T>> to Model<Vec<T>>
let models: Vec<Model<f64>> = (0..5)
    .map(|i| sample(addr!("x", i), Normal::new(0.0, 1.0).unwrap()))
    .collect();
let combined: Model<Vec<f64>> = sequence_vec(models);

// Traverse: Apply function to each element
let data = vec![1.0, 2.0, 3.0];
let noisy_data: Model<Vec<f64>> = traverse_vec(data, |x| {
    sample(addr!("noise", x as usize), Normal::new(x, 0.1).unwrap())
});
```

## Advanced Composition Patterns

### Guards and Validation

```rust
fn safe_division_model(denominator: f64) -> Model<Option<f64>> {
    if denominator.abs() < 1e-10 {
        pure(None)
    } else {
        sample(addr!("numerator"), Normal::new(0.0, 1.0).unwrap())
            .map(move |num| Some(num / denominator))
    }
}

// Or use the guard combinator
fn guarded_model() -> Model<f64> {
    sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
        .bind(|x| {
            guard(x > 0.0, move || {
                sample(addr!("y"), LogNormal::new(x.ln(), 0.1).unwrap())
            })
        })
}
```

### Model Factories

```rust
fn make_mixture_component(component_id: usize, weight: f64) -> Model<f64> {
    prob! {
        let should_use <- sample(
            addr!("use_component", component_id),
            Bernoulli::new(weight).unwrap()
        );
        
        if should_use {
            let value <- sample(
                addr!("value", component_id),
                Normal::new(component_id as f64, 1.0).unwrap()
            );
            pure(value)
        } else {
            pure(0.0)  // Not selected
        }
    }
}

fn mixture_model(weights: &[f64]) -> Model<Vec<f64>> {
    let components: Vec<Model<f64>> = weights
        .iter()
        .enumerate()
        .map(|(i, &w)| make_mixture_component(i, w))
        .collect();
    
    sequence_vec(components)
}
```

## Complete Example: Bayesian A/B Test

Here's a complete example showing advanced model composition:

```rust
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn ab_test_model(
    a_successes: u64,
    a_trials: u64,
    b_successes: u64,
    b_trials: u64,
) -> Model<(f64, f64, f64)> {
    prob! {
        // Priors for conversion rates
        let rate_a <- sample(addr!("rate_a"), Beta::new(1.0, 1.0).unwrap());
        let rate_b <- sample(addr!("rate_b"), Beta::new(1.0, 1.0).unwrap());
        
        // Observations
        observe(
            addr!("obs_a"),
            Binomial::new(a_trials, rate_a).unwrap(),
            a_successes
        );
        observe(
            addr!("obs_b"),
            Binomial::new(b_trials, rate_b).unwrap(),
            b_successes
        );
        
        // Derived quantity: difference
        let difference = rate_b - rate_a;
        
        pure((rate_a, rate_b, difference))
    }
}

fn main() {
    // A: 45 successes out of 1000 trials
    // B: 55 successes out of 1000 trials
    let model = ab_test_model(45, 1000, 55, 1000);
    
    let mut rng = StdRng::seed_from_u64(42);
    let ((rate_a, rate_b, diff), trace) = runtime::handler::run(
        runtime::interpreters::PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        model,
    );
    
    println!("Rate A: {:.3}", rate_a);
    println!("Rate B: {:.3}", rate_b);
    println!("Difference (B - A): {:.3}", diff);
    println!("Log probability: {:.4}", trace.total_log_weight());
}
```

## Key Takeaways

Understanding models deeply involves:

1. **Models are composable values** - Pure, functional composition
2. **Four fundamental operations** - pure, sample, observe, factor
3. **Monadic interface** - map, bind, and_then for composition
4. **`prob!` macro** - Imperative syntax for complex models
5. **Collection handling** - plate!, sequence_vec, traverse_vec
6. **Conditional logic** - Models can branch and include guards
7. **Type safety** - The type system prevents many modeling errors

## What's Next?

Now that you understand models:

1. **[Basic Inference](basic-inference.md)** - Learn to run inference on your models
2. **[Using Macros](../how-to/using-macros.md)** - Master `prob!`, `plate!`, and `addr!`
3. **[Working with Distributions](../how-to/working-with-distributions.md)** - Explore all distribution types

---

**Ready to run inference?** → **[Basic Inference](basic-inference.md)**