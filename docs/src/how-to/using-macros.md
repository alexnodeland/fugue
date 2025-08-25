# Using Macros

Fugue provides powerful macros that make probabilistic programming more ergonomic and expressive. This guide covers the three key macros: `prob!` for do-notation style composition, `plate!` for vectorized operations, and `addr!` for addressing random variables.

## The `prob!` Macro - Do Notation

The `prob!` macro transforms imperative-style code into proper monadic composition, making complex probabilistic programs much more readable.

### Basic Syntax

```rust
use fugue::*;

let simple_model = prob! {
    let x <- sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
    let y <- sample(addr!("y"), Normal::new(x, 0.5).unwrap());
    observe(addr!("z"), Normal::new(x + y, 0.1).unwrap(), 2.0);
    pure((x, y))
};
```

**Key syntax rules**:
- `let var <- model_expr;` - Bind the result of a model to a variable
- `let var = expr;` - Regular deterministic variable binding
- `model_expr;` - Execute a model and ignore its result
- `pure(value)` - Final expression (can be any model)

### Without vs. With `prob!`

**Without `prob!` (verbose monadic composition)**:
```rust
let verbose_model = 
    sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
        .bind(|x| {
            sample(addr!("y"), Normal::new(x, 0.5).unwrap())
                .bind(move |y| {
                    observe(addr!("z"), Normal::new(x + y, 0.1).unwrap(), 2.0)
                        .bind(move |_| {
                            factor(if x > 0.0 { 0.0 } else { -1.0 })
                                .map(move |_| (x, y))
                        })
                })
        });
```

**With `prob!` (clean and readable)**:
```rust
let clean_model = prob! {
    let x <- sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
    let y <- sample(addr!("y"), Normal::new(x, 0.5).unwrap());
    observe(addr!("z"), Normal::new(x + y, 0.1).unwrap(), 2.0);
    
    if x > 0.0 {
        factor(0.0);
    } else {
        factor(-1.0);
    }
    
    pure((x, y))
};
```

### Control Flow in `prob!`

The `prob!` macro supports natural control flow:

#### Conditional Logic

```rust
let conditional_model = prob! {
    let coin <- sample(addr!("coin"), Bernoulli::new(0.5).unwrap());
    
    if coin {
        let heads_value <- sample(addr!("heads"), Normal::new(1.0, 0.1).unwrap());
        pure(heads_value)
    } else {
        let tails_value <- sample(addr!("tails"), Normal::new(-1.0, 0.1).unwrap());
        pure(tails_value)
    }
};
```

#### Loops and Iteration

```rust
let iterative_model = prob! {
    let mut sum = 0.0;
    
    for i in 0..5 {
        let value <- sample(addr!("x", i), Normal::new(0.0, 1.0).unwrap());
        sum += value;
    }
    
    pure(sum)
};

// Or using functional style
let functional_model = prob! {
    let values <- plate!(i in 0..5 => {
        sample(addr!("x", i), Normal::new(0.0, 1.0).unwrap())
    });
    
    let sum: f64 = values.iter().sum();
    pure(sum)
};
```

### Complex Example: Hierarchical Model

```rust
fn hierarchical_regression(x_data: &[f64], y_data: &[Vec<f64>]) -> Model<(f64, Vec<f64>)> {
    prob! {
        // Global hyperparameters
        let global_mean <- sample(addr!("global_mean"), Normal::new(0.0, 1.0).unwrap());
        let global_scale <- sample(addr!("global_scale"), LogNormal::new(0.0, 0.5).unwrap());
        
        // Group-specific parameters
        let group_effects <- plate!(g in 0..y_data.len() => {
            sample(addr!("group", g), Normal::new(global_mean, global_scale).unwrap())
        });
        
        // Observations for each group
        for (g, group_y) in y_data.iter().enumerate() {
            let group_effect = group_effects[g];
            
            for (i, &y) in group_y.iter().enumerate() {
                let x = x_data[i];
                let predicted = group_effect * x;
                observe(addr!("y", g, i), Normal::new(predicted, 0.1).unwrap(), y);
            }
        }
        
        pure((global_mean, group_effects))
    }
}
```

## The `plate!` Macro - Vectorized Operations

The `plate!` macro is inspired by plate notation in Bayesian statistics. It replicates a model across a range, creating independent parallel computations.

### Basic Plate Operations

```rust
// Generate 10 independent normal samples
let samples: Model<Vec<f64>> = plate!(i in 0..10 => {
    sample(addr!("x", i), Normal::new(0.0, 1.0).unwrap())
});

// With different parameters for each element
let varying_samples: Model<Vec<f64>> = plate!(i in 0..5 => {
    let mu = i as f64;  // Mean increases with index
    sample(addr!("x", i), Normal::new(mu, 1.0).unwrap())
});
```

### Plate with Shared Parameters

```rust
let shared_parameter_model: Model<Vec<f64>> = prob! {
    // Shared hyperparameter
    let global_mu <- sample(addr!("global_mu"), Normal::new(0.0, 1.0).unwrap());
    let global_sigma <- sample(addr!("global_sigma"), LogNormal::new(0.0, 0.5).unwrap());
    
    // Individual samples sharing the hyperparameters
    let samples <- plate!(i in 0..20 => {
        sample(addr!("x", i), Normal::new(global_mu, global_sigma).unwrap())
    });
    
    pure(samples)
};
```

### Nested Plates

```rust
// Matrix of samples (groups × individuals)
let nested_model: Model<Vec<Vec<f64>>> = prob! {
    let group_means <- plate!(g in 0..3 => {
        sample(addr!("group_mean", g), Normal::new(0.0, 2.0).unwrap())
    });
    
    let all_samples <- plate!(g in 0..3 => {
        let group_mean = group_means[g];
        plate!(i in 0..10 => {
            sample(addr!("sample", g, i), Normal::new(group_mean, 1.0).unwrap())
        })
    });
    
    pure(all_samples)
};
```

### Plate with Observations

```rust
fn regression_plate_model(x_data: &[f64], y_data: &[f64]) -> Model<(f64, f64)> {
    prob! {
        let slope <- sample(addr!("slope"), Normal::new(0.0, 1.0).unwrap());
        let intercept <- sample(addr!("intercept"), Normal::new(0.0, 1.0).unwrap());
        let noise <- sample(addr!("noise"), LogNormal::new(0.0, 0.5).unwrap());
        
        // Observations using plate
        let _observations <- plate!(i in 0..x_data.len() => {
            let x = x_data[i];
            let y = y_data[i];
            let predicted = slope * x + intercept;
            observe(addr!("y", i), Normal::new(predicted, noise).unwrap(), y)
        });
        
        pure((slope, intercept))
    }
}
```

### Alternative Plate Patterns

```rust
// Using iterators directly
let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let iterator_model: Model<Vec<f64>> = plate!(x in data.iter() => {
    let x_val = *x;
    sample(addr!("noise", x_val as usize), Normal::new(x_val, 0.1).unwrap())
});

// Using enumerate for index and value
let enumerated_model: Model<Vec<String>> = plate!((i, x) in data.iter().enumerate() => {
    let noisy_x <- sample(addr!("x", i), Normal::new(*x, 0.1).unwrap());
    pure(format!("Item {}: {:.2}", i, noisy_x))
});
```

## The `addr!` Macro - Addressing Random Variables

The `addr!` macro creates unique addresses for random variables. Proper addressing is crucial for:
- Conditioning on observations
- Trace manipulation and debugging
- Inference algorithm correctness

### Basic Addressing

```rust
// Simple named addresses
let mu_addr = addr!("mu");
let sigma_addr = addr!("sigma");

// Use in models
let model = prob! {
    let mu <- sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap());
    let sigma <- sample(addr!("sigma"), LogNormal::new(0.0, 0.5).unwrap());
    let x <- sample(addr!("x"), Normal::new(mu, sigma).unwrap());
    pure((mu, sigma, x))
};
```

### Indexed Addressing

```rust
// Indexed addresses for collections
let indexed_model = prob! {
    let samples <- plate!(i in 0..10 => {
        sample(addr!("x", i), Normal::new(0.0, 1.0).unwrap())
    });
    pure(samples)
};

// Multiple indices
let matrix_model = prob! {
    let matrix <- plate!(i in 0..3 => {
        plate!(j in 0..4 => {
            sample(addr!("x", i, j), Normal::new(0.0, 1.0).unwrap())
        })
    });
    pure(matrix)
};
```

### Dynamic Addressing

```rust
// Addresses can be computed at runtime
fn dynamic_addressing_model(n: usize) -> Model<Vec<f64>> {
    plate!(i in 0..n => {
        let address_suffix = format!("item_{}", i);
        sample(addr!(&address_suffix), Normal::new(0.0, 1.0).unwrap())
    })
}

// Conditional addressing
let conditional_addresses = prob! {
    let use_alternative <- sample(addr!("use_alt"), Bernoulli::new(0.5).unwrap());
    
    if use_alternative {
        let alt_value <- sample(addr!("alternative"), Normal::new(1.0, 0.1).unwrap());
        pure(alt_value)
    } else {
        let std_value <- sample(addr!("standard"), Normal::new(0.0, 1.0).unwrap());
        pure(std_value)
    }
};
```

### Scoped Addressing

For complex models, you can create hierarchical address scopes:

```rust
// Manual scoping
fn scoped_model() -> Model<(f64, f64)> {
    prob! {
        let prior_mu <- sample(addr!("prior::mu"), Normal::new(0.0, 1.0).unwrap());
        let prior_sigma <- sample(addr!("prior::sigma"), LogNormal::new(0.0, 0.5).unwrap());
        
        let likelihood_x <- sample(addr!("likelihood::x"), Normal::new(prior_mu, prior_sigma).unwrap());
        observe(addr!("likelihood::obs"), Normal::new(likelihood_x, 0.1).unwrap(), 2.5);
        
        pure((prior_mu, likelihood_x))
    }
}

// Using helper functions for scoping
fn make_scoped_addr(scope: &str, name: &str) -> Address {
    addr!(&format!("{}::{}", scope, name))
}

fn hierarchical_scopes() -> Model<Vec<f64>> {
    prob! {
        let global_params <- plate!(level in 0..3 => {
            let scope = format!("level_{}", level);
            sample(make_scoped_addr(&scope, "param"), Normal::new(0.0, 1.0).unwrap())
        });
        pure(global_params)
    }
}
```

## Advanced Macro Patterns

### Combining All Three Macros

```rust
fn complete_example(groups: &[Vec<f64>]) -> Model<(f64, Vec<f64>)> {
    prob! {
        // Global hyperparameters with clear addressing
        let global_mean <- sample(addr!("hyperparams::mean"), Normal::new(0.0, 1.0).unwrap());
        let global_scale <- sample(addr!("hyperparams::scale"), LogNormal::new(0.0, 0.5).unwrap());
        
        // Group-specific parameters using plate
        let group_means <- plate!(g in 0..groups.len() => {
            sample(addr!("groups::mean", g), Normal::new(global_mean, global_scale).unwrap())
        });
        
        // Observations for each group
        for (g, group_data) in groups.iter().enumerate() {
            let group_mean = group_means[g];
            
            // Individual observations within each group  
            let _group_obs <- plate!(i in 0..group_data.len() => {
                let observed_value = group_data[i];
                observe(
                    addr!("observations", g, i), 
                    Normal::new(group_mean, 0.1).unwrap(), 
                    observed_value
                )
            });
        }
        
        pure((global_mean, group_means))
    }
}
```

### Macro Hygiene and Scope

```rust
// Variables in prob! have proper scope
let scoped_example = prob! {
    let x <- sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
    
    {
        let y <- sample(addr!("y"), Normal::new(x, 0.5).unwrap());  // x is accessible
        pure(y)
    }  // y goes out of scope here
};

// Variable capture in closures
fn closure_example(external_param: f64) -> Model<f64> {
    prob! {
        let internal <- sample(addr!("internal"), Normal::new(0.0, 1.0).unwrap());
        let combined = internal + external_param;  // Captures external_param
        pure(combined)
    }
}
```

## Performance Considerations

### Efficient Address Creation

```rust
// ❌ Inefficient: String allocation in hot paths
for i in 0..10000 {
    let addr = addr!(&format!("x_{}", i));  // Allocates each time
}

// ✅ Efficient: Use built-in indexing
for i in 0..10000 {
    let addr = addr!("x", i);  // More efficient
}

// ✅ Pre-compute addresses when possible
let addresses: Vec<Address> = (0..10000)
    .map(|i| addr!("x", i))
    .collect();
```

### Plate Optimization

```rust
// ❌ Less efficient: Nested prob! blocks
let inefficient = prob! {
    let samples <- plate!(i in 0..1000 => {
        prob! {  // Extra overhead
            let x <- sample(addr!("x", i), Normal::new(0.0, 1.0).unwrap());
            pure(x)
        }
    });
    pure(samples)
};

// ✅ More efficient: Direct expressions in plate
let efficient = plate!(i in 0..1000 => {
    sample(addr!("x", i), Normal::new(0.0, 1.0).unwrap())
});
```

## Debugging Macro Expansions

Sometimes you need to understand what the macros expand to:

```rust
// Use cargo expand to see macro expansions
// $ cargo expand --example your_example

// For debugging, you can also use explicit monadic composition
let debug_model = 
    sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
        .bind(|x| {
            println!("Debug: x = {}", x);  // Add debug prints
            sample(addr!("y"), Normal::new(x, 0.5).unwrap())
        });
```

## Common Patterns and Idioms

### Model Factories with Macros

```rust
fn make_regression_model(data: &[(f64, f64)]) -> Model<(f64, f64)> {
    prob! {
        let slope <- sample(addr!("slope"), Normal::new(0.0, 1.0).unwrap());
        let intercept <- sample(addr!("intercept"), Normal::new(0.0, 1.0).unwrap());
        let noise <- sample(addr!("noise"), LogNormal::new(0.0, 0.5).unwrap());
        
        let _observations <- plate!((i, (x, y)) in data.iter().enumerate() => {
            let predicted = slope * x + intercept;
            observe(addr!("y", i), Normal::new(predicted, noise).unwrap(), *y)
        });
        
        pure((slope, intercept))
    }
}
```

### Conditional Model Selection

```rust
fn adaptive_model(use_complex: bool) -> Model<f64> {
    if use_complex {
        prob! {
            let components <- plate!(i in 0..3 => {
                sample(addr!("component", i), Normal::new(i as f64, 1.0).unwrap())
            });
            let weights <- plate!(i in 0..3 => {
                sample(addr!("weight", i), Gamma::new(1.0, 1.0).unwrap())
            });
            let total_weight: f64 = weights.iter().sum();
            let weighted_sum: f64 = components.iter()
                .zip(weights.iter())
                .map(|(c, w)| c * w / total_weight)
                .sum();
            pure(weighted_sum)
        }
    } else {
        sample(addr!("simple"), Normal::new(0.0, 1.0).unwrap())
    }
}
```

## Complete Working Example

Here's a comprehensive example using all three macros effectively:

```rust
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

// Hierarchical model for A/B testing with multiple metrics
fn ab_test_model(
    conversions_a: &[bool],
    conversions_b: &[bool], 
    revenue_a: &[f64],
    revenue_b: &[f64],
) -> Model<(f64, f64, f64, f64)> {
    prob! {
        // Prior beliefs about conversion rates
        let conversion_rate_a <- sample(addr!("conversion::rate_a"), Beta::new(1.0, 1.0).unwrap());
        let conversion_rate_b <- sample(addr!("conversion::rate_b"), Beta::new(1.0, 1.0).unwrap());
        
        // Prior beliefs about revenue parameters
        let revenue_mu_a <- sample(addr!("revenue::mu_a"), Normal::new(50.0, 20.0).unwrap());
        let revenue_mu_b <- sample(addr!("revenue::mu_b"), Normal::new(50.0, 20.0).unwrap());
        let revenue_sigma <- sample(addr!("revenue::sigma"), LogNormal::new(2.0, 0.5).unwrap());
        
        // Conversion observations for group A
        let _conv_obs_a <- plate!(i in 0..conversions_a.len() => {
            observe(
                addr!("conv_obs_a", i), 
                Bernoulli::new(conversion_rate_a).unwrap(), 
                conversions_a[i]
            )
        });
        
        // Conversion observations for group B
        let _conv_obs_b <- plate!(i in 0..conversions_b.len() => {
            observe(
                addr!("conv_obs_b", i), 
                Bernoulli::new(conversion_rate_b).unwrap(), 
                conversions_b[i]
            )
        });
        
        // Revenue observations for group A (only for converted users)
        let _rev_obs_a <- plate!(i in 0..revenue_a.len() => {
            observe(
                addr!("rev_obs_a", i),
                Normal::new(revenue_mu_a, revenue_sigma).unwrap(),
                revenue_a[i]
            )
        });
        
        // Revenue observations for group B (only for converted users)
        let _rev_obs_b <- plate!(i in 0..revenue_b.len() => {
            observe(
                addr!("rev_obs_b", i),
                Normal::new(revenue_mu_b, revenue_sigma).unwrap(),
                revenue_b[i]
            )
        });
        
        pure((conversion_rate_a, conversion_rate_b, revenue_mu_a, revenue_mu_b))
    }
}

fn main() {
    // Sample data
    let conv_a = vec![true, false, true, true, false];
    let conv_b = vec![true, true, false, true, true];
    let rev_a = vec![45.0, 52.0, 48.0];  // Revenue from converted users
    let rev_b = vec![55.0, 60.0, 58.0, 62.0];  // Revenue from converted users
    
    let model = ab_test_model(&conv_a, &conv_b, &rev_a, &rev_b);
    
    let mut rng = StdRng::seed_from_u64(42);
    let ((conv_a_rate, conv_b_rate, rev_a_mu, rev_b_mu), trace) = runtime::handler::run(
        runtime::interpreters::PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        model,
    );
    
    println!("🧪 A/B Test Analysis:");
    println!("  Conversion Rate A: {:.1}%", conv_a_rate * 100.0);
    println!("  Conversion Rate B: {:.1}%", conv_b_rate * 100.0);
    println!("  Average Revenue A: ${:.2}", rev_a_mu);
    println!("  Average Revenue B: ${:.2}", rev_b_mu);
    println!("  Log probability: {:.4}", trace.total_log_weight());
    
    // Expected value differences
    let conv_diff = conv_b_rate - conv_a_rate;
    let rev_diff = rev_b_mu - rev_a_mu;
    println!("\n📊 Differences (B - A):");
    println!("  Conversion: {:.1}%", conv_diff * 100.0);
    println!("  Revenue: ${:.2}", rev_diff);
}
```

## Key Takeaways

1. **`prob!`** - Use for complex models with multiple steps and control flow
2. **`plate!`** - Use for vectorized operations and independent replications  
3. **`addr!`** - Always provide meaningful, unique addresses for random variables
4. **Combine macros** - They work together naturally for complex hierarchical models
5. **Performance** - Be mindful of address creation and nested structures in hot paths

## Next Steps

- **[Trace Manipulation](trace-manipulation.md)** - Debug and analyze model execution
- **[Custom Handlers](custom-handlers.md)** - Implement your own model interpreters
- **[Linear Regression Tutorial](../tutorials/linear-regression.md)** - See macros in action

---

**Ready to manipulate traces?** → **[Trace Manipulation](trace-manipulation.md)**