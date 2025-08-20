# Working with Distributions

Fugue's type-safe distribution system is one of its key features. This guide covers all available distributions, their type safety benefits, parameter validation, and practical usage patterns.

## The Type Safety Revolution

Traditional probabilistic programming libraries return `f64` for all distributions, leading to error-prone code:

```rust
// ❌ Error-prone in other PPLs
let coin = sample("coin", Bernoulli { p: 0.5 });  // Returns f64
if coin == 1.0 {  // Fragile floating-point comparison!
    // ...
}
```

Fugue distributions return natural, meaningful types:

```rust
// ✅ Type-safe in Fugue
let coin: Model<bool> = sample(addr!("coin"), Bernoulli::new(0.5).unwrap());
coin.map(|is_heads| {
    if is_heads {  // Natural boolean usage!
        "Heads"
    } else {
        "Tails"
    }
})
```

## Distribution Overview

| **Distribution** | **Return Type** | **Parameters** | **Support** | **Use Cases** |
|------------------|-----------------|----------------|-------------|---------------|
| `Normal` | `f64` | `mu`, `sigma` | ℝ | General continuous modeling |
| `LogNormal` | `f64` | `mu`, `sigma` | ℝ⁺ | Positive values, scales |
| `Uniform` | `f64` | `low`, `high` | [low, high] | Uninformative priors |
| `Exponential` | `f64` | `rate` | ℝ⁺ | Waiting times, lifetimes |
| `Beta` | `f64` | `alpha`, `beta` | [0, 1] | Probabilities, proportions |
| `Gamma` | `f64` | `shape`, `rate` | ℝ⁺ | Positive continuous values |
| `Bernoulli` | **`bool`** | `p` | {false, true} | Binary outcomes |
| `Binomial` | **`u64`** | `n`, `p` | {0, 1, ..., n} | Count of successes |
| `Categorical` | **`usize`** | `probs` | {0, 1, ..., k-1} | Discrete choices |
| `Poisson` | **`u64`** | `lambda` | ℕ | Event counts |

## Continuous Distributions

### Normal Distribution

The most commonly used continuous distribution:

```rust
use fugue::*;

// Safe constructor - validates parameters
let normal = Normal::new(0.0, 1.0).unwrap();  // μ=0, σ=1
let model: Model<f64> = sample(addr!("x"), normal);

// Common patterns
let standard_normal = Normal::new(0.0, 1.0).unwrap();
let wide_prior = Normal::new(0.0, 10.0).unwrap();
let narrow_likelihood = Normal::new(observed_mean, 0.1).unwrap();

// Invalid parameters are caught at construction
match Normal::new(0.0, -1.0) {  // Negative sigma
    Ok(_) => unreachable!(),
    Err(e) => println!("Caught invalid parameter: {}", e),
}
```

### LogNormal Distribution

For positive values only:

```rust
// LogNormal parameters are for the underlying normal distribution
let log_normal = LogNormal::new(0.0, 1.0).unwrap();  // Median ≈ 1.0

// Great for scales, variances, positive parameters
let scale_prior: Model<f64> = sample(addr!("scale"), LogNormal::new(0.0, 0.5).unwrap());
let precision: Model<f64> = sample(addr!("precision"), LogNormal::new(1.0, 0.2).unwrap());

// Convert between parameterizations
fn log_normal_from_mean_var(mean: f64, var: f64) -> Result<LogNormal, String> {
    let mu = (mean.powi(2) / (var + mean.powi(2)).sqrt()).ln();
    let sigma = ((var + mean.powi(2)) / mean.powi(2)).ln().sqrt();
    LogNormal::new(mu, sigma)
}
```

### Beta Distribution

For values in [0, 1]:

```rust
// Beta(1, 1) is uniform on [0, 1]
let uniform_prob = Beta::new(1.0, 1.0).unwrap();

// Beta(2, 2) prefers values near 0.5
let centered_prob = Beta::new(2.0, 2.0).unwrap();

// Conjugate prior for Bernoulli/Binomial
let coin_bias_prior = Beta::new(2.0, 2.0).unwrap();
let bias_model = prob! {
    let p <- sample(addr!("bias"), coin_bias_prior);
    observe(addr!("flips"), Binomial::new(10, p).unwrap(), 7u64);
    pure(p)
};
```

### Gamma Distribution

For positive continuous values:

```rust
// Shape and rate parameterization
let gamma = Gamma::new(2.0, 1.0).unwrap();  // shape=2, rate=1

// Common uses
let precision_prior = Gamma::new(1.0, 1.0).unwrap();  // Precision parameter
let arrival_rate = Gamma::new(2.0, 0.5).unwrap();     // Rate parameter

// Convert from shape/scale to shape/rate
fn gamma_from_shape_scale(shape: f64, scale: f64) -> Result<Gamma, String> {
    Gamma::new(shape, 1.0 / scale)  // rate = 1/scale
}
```

### Exponential Distribution

For waiting times and lifetimes:

```rust
let waiting_time = Exponential::new(2.0).unwrap();  // rate=2

// Memoryless property makes it useful for:
let time_to_failure = Exponential::new(0.001).unwrap();
let time_between_events = Exponential::new(lambda).unwrap();

// Related to Poisson: if events follow Poisson(λ), 
// inter-arrival times follow Exponential(λ)
```

### Uniform Distribution

For uninformative priors:

```rust
let uniform_prior = Uniform::new(-5.0, 5.0).unwrap();
let probability_prior = Uniform::new(0.0, 1.0).unwrap();

// Useful for bounded parameters
let angle_prior = Uniform::new(0.0, 2.0 * std::f64::consts::PI).unwrap();
```

## Discrete Distributions

### Bernoulli Distribution (Returns `bool`)

For binary outcomes:

```rust
let fair_coin = Bernoulli::new(0.5).unwrap();
let biased_coin = Bernoulli::new(0.7).unwrap();

let coin_model: Model<bool> = sample(addr!("coin"), fair_coin);

// Type-safe usage - no more == 1.0 comparisons!
let outcome_model = coin_model.map(|is_heads| {
    if is_heads {
        "Success"
    } else {
        "Failure"
    }
});

// Conditional models based on boolean outcomes
let conditional_model = prob! {
    let success <- sample(addr!("success"), Bernoulli::new(0.3).unwrap());
    
    if success {
        let bonus <- sample(addr!("bonus"), Poisson::new(5.0).unwrap());
        pure(bonus)
    } else {
        pure(0u64)
    }
};
```

### Binomial Distribution (Returns `u64`)

For counting successes:

```rust
let coin_flips = Binomial::new(10, 0.5).unwrap();  // 10 flips, 50% success
let trials: Model<u64> = sample(addr!("successes"), coin_flips);

// Natural count usage - no casting needed!
let analysis = trials.map(|count| {
    match count {
        0..=3 => "Few successes",
        4..=6 => "Moderate successes", 
        7..=10 => "Many successes",
    }
});

// Conjugate with Beta prior
let binomial_model = prob! {
    let p <- sample(addr!("success_rate"), Beta::new(2.0, 2.0).unwrap());
    let successes <- sample(addr!("successes"), Binomial::new(20, p).unwrap());
    observe(addr!("observed"), Binomial::new(20, p).unwrap(), 15u64);
    pure((p, successes))
};
```

### Poisson Distribution (Returns `u64`)

For counting events:

```rust
let events = Poisson::new(3.0).unwrap();  // Average 3 events
let count_model: Model<u64> = sample(addr!("events"), events);

// Perfect for count data
let event_analysis = count_model.map(|n_events| {
    format!("Observed {} events", n_events)
});

// Multiple related Poisson processes
let multi_poisson = prob! {
    let rate <- sample(addr!("rate"), Gamma::new(2.0, 1.0).unwrap());
    
    let morning_events <- sample(
        addr!("morning"), 
        Poisson::new(rate * 0.5).unwrap()  // Half rate in morning
    );
    let evening_events <- sample(
        addr!("evening"), 
        Poisson::new(rate * 1.5).unwrap()  // Higher rate in evening
    );
    
    pure((morning_events, evening_events))
};
```

### Categorical Distribution (Returns `usize`)

For discrete choices:

```rust
// Three categories with different probabilities
let categories = Categorical::new(vec![0.3, 0.5, 0.2]).unwrap();
let choice_model: Model<usize> = sample(addr!("category"), categories);

// Safe array indexing with no conversion needed!
let category_names = vec!["Red", "Green", "Blue"];
let named_choice = choice_model.map(|index| {
    category_names[index]  // Direct indexing - no casting!
});

// Dynamic category probabilities
fn categorical_with_softmax(logits: &[f64]) -> Result<Categorical, String> {
    let max_logit = logits.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let exp_logits: Vec<f64> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum: f64 = exp_logits.iter().sum();
    let probs: Vec<f64> = exp_logits.iter().map(|&x| x / sum).collect();
    Categorical::new(probs)
}

// Mixture model using categorical
let mixture_model = prob! {
    let component <- sample(addr!("component"), Categorical::new(vec![0.3, 0.7]).unwrap());
    
    match component {
        0 => sample(addr!("value"), Normal::new(-2.0, 1.0).unwrap()),
        1 => sample(addr!("value"), Normal::new(2.0, 1.0).unwrap()),
        _ => unreachable!(),  // Categorical guarantees valid indices
    }
};
```

## Parameter Validation and Error Handling

All distributions use safe constructors that validate parameters:

```rust
use fugue::*;

// ✅ Valid parameters
let normal = Normal::new(0.0, 1.0).unwrap();
let beta = Beta::new(2.0, 3.0).unwrap();
let poisson = Poisson::new(5.0).unwrap();

// ❌ Invalid parameters are caught
match Normal::new(0.0, -1.0) {  // Negative sigma
    Ok(_) => unreachable!(),
    Err(e) => println!("Invalid sigma: {}", e),
}

match Beta::new(-1.0, 2.0) {  // Negative alpha
    Ok(_) => unreachable!(),
    Err(e) => println!("Invalid alpha: {}", e),
}

match Poisson::new(-2.0) {  // Negative lambda
    Ok(_) => unreachable!(),
    Err(e) => println!("Invalid lambda: {}", e),
}

// Handle validation in model construction
fn safe_model(sigma: f64) -> Result<Model<f64>, String> {
    let normal = Normal::new(0.0, sigma)?;  // Propagate error
    Ok(sample(addr!("x"), normal))
}
```

## Advanced Distribution Patterns

### Custom Distribution Parameters

```rust
// Temperature-dependent distribution
fn temperature_dependent_normal(temp: f64) -> Result<Normal, String> {
    let sigma = 0.1 + 0.01 * temp;  // Variance increases with temperature
    Normal::new(0.0, sigma)
}

// Time-varying Poisson process
fn time_varying_poisson(t: f64, base_rate: f64) -> Result<Poisson, String> {
    let rate = base_rate * (1.0 + 0.5 * (2.0 * std::f64::consts::PI * t).sin());
    Poisson::new(rate)
}
```

### Distribution Factories

```rust
// Factory for creating related distributions
struct DistributionFactory {
    base_mu: f64,
    base_sigma: f64,
}

impl DistributionFactory {
    fn normal(&self, scale: f64) -> Result<Normal, String> {
        Normal::new(self.base_mu, self.base_sigma * scale)
    }
    
    fn log_normal(&self, scale: f64) -> Result<LogNormal, String> {
        LogNormal::new(self.base_mu, self.base_sigma * scale)
    }
}

// Usage
let factory = DistributionFactory { base_mu: 0.0, base_sigma: 1.0 };
let narrow_normal = factory.normal(0.1).unwrap();
let wide_normal = factory.normal(2.0).unwrap();
```

### Mixture Distributions

```rust
// Manual mixture using categorical
fn normal_mixture(weights: Vec<f64>, means: Vec<f64>, stds: Vec<f64>) -> Model<f64> {
    prob! {
        let component <- sample(
            addr!("component"), 
            Categorical::new(weights).unwrap()
        );
        
        let mu = means[component];    // Safe indexing!
        let sigma = stds[component];  // Safe indexing!
        
        sample(addr!("value"), Normal::new(mu, sigma).unwrap())
    }
}

// Three-component mixture
let mixture = normal_mixture(
    vec![0.3, 0.5, 0.2],           // weights
    vec![-2.0, 0.0, 2.0],          // means
    vec![0.5, 1.0, 0.8],           // standard deviations
);
```

## Working with Observations

### Type-Safe Observations

```rust
// Continuous observations
observe(addr!("temp"), Normal::new(20.0, 1.0).unwrap(), 22.5f64);

// Boolean observations  
observe(addr!("success"), Bernoulli::new(0.7).unwrap(), true);

// Count observations
observe(addr!("events"), Poisson::new(5.0).unwrap(), 8u64);

// Categorical observations
observe(addr!("choice"), Categorical::new(vec![0.3, 0.4, 0.3]).unwrap(), 1usize);
```

### Observation Validation

```rust
// Observations are validated against distribution support
fn safe_observation_model() -> Model<()> {
    prob! {
        // ✅ Valid: 0.5 is in [0, 1]
        observe(addr!("prob"), Beta::new(2.0, 2.0).unwrap(), 0.5);
        
        // ❌ Would fail: -0.5 is not in [0, 1]
        // observe(addr!("invalid"), Beta::new(2.0, 2.0).unwrap(), -0.5);
        
        // ✅ Valid: 3 is a valid count
        observe(addr!("count"), Poisson::new(2.0).unwrap(), 3u64);
        
        pure(())
    }
}
```

## Performance Considerations

### Distribution Creation

```rust
// ❌ Inefficient: creating distributions in hot loops
for i in 0..1000 {
    let dist = Normal::new(0.0, 1.0).unwrap();  // Recreated each time
    sample(addr!("x", i), dist);
}

// ✅ Efficient: create once, reuse
let shared_dist = Normal::new(0.0, 1.0).unwrap();
for i in 0..1000 {
    sample(addr!("x", i), shared_dist);
}
```

### Parameter Validation

```rust
// Parameter validation happens at construction, not sampling
let dist = Normal::new(mu, sigma)?;  // Validation here
let model = sample(addr!("x"), dist);  // No validation overhead here

// For dynamic parameters, validate once:
fn validated_normal(mu: f64, sigma: f64) -> Result<Model<f64>, String> {
    let dist = Normal::new(mu, sigma)?;  // Early validation
    Ok(sample(addr!("x"), dist))
}
```

## Complete Example: Multi-Type Model

Here's a complete example using multiple distribution types:

```rust
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn survey_model(responses: &[(bool, usize, u64)]) -> Model<(f64, Vec<f64>)> {
    prob! {
        // Global bias parameter
        let global_bias <- sample(addr!("bias"), Beta::new(2.0, 2.0).unwrap());
        
        // Category preferences
        let prefs <- sample(
            addr!("preferences"), 
            // Dirichlet would be better, but we'll use normalized Gammas
            plate!(i in 0..3 => {
                sample(addr!("pref", i), Gamma::new(1.0, 1.0).unwrap())
            })
        );
        
        // Normalize preferences
        let total: f64 = prefs.iter().sum();
        let normalized_prefs: Vec<f64> = prefs.iter().map(|&p| p / total).collect();
        
        // Event rate
        let event_rate <- sample(addr!("rate"), Gamma::new(2.0, 0.5).unwrap());
        
        // Observations
        for (i, &(success, category, events)) in responses.iter().enumerate() {
            // Boolean observation
            observe(
                addr!("success", i), 
                Bernoulli::new(global_bias).unwrap(), 
                success
            );
            
            // Categorical observation  
            observe(
                addr!("category", i),
                Categorical::new(normalized_prefs.clone()).unwrap(),
                category
            );
            
            // Count observation
            observe(
                addr!("events", i),
                Poisson::new(event_rate).unwrap(),
                events
            );
        }
        
        pure((global_bias, normalized_prefs))
    }
}

fn main() {
    // Survey data: (success, preferred_category, event_count)
    let data = vec![
        (true, 0usize, 3u64),
        (false, 1usize, 2u64), 
        (true, 2usize, 5u64),
        (true, 0usize, 1u64),
    ];
    
    let model = survey_model(&data);
    
    let mut rng = StdRng::seed_from_u64(42);
    let ((bias, prefs), trace) = runtime::handler::run(
        runtime::interpreters::PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        model,
    );
    
    println!("📊 Survey Analysis Results:");
    println!("  Global bias: {:.3}", bias);
    println!("  Category preferences: {:?}", 
             prefs.iter().map(|&p| format!("{:.3}", p)).collect::<Vec<_>>());
    println!("  Log probability: {:.4}", trace.total_log_weight());
}
```

## Key Takeaways

1. **Type Safety**: Distributions return natural types (`bool`, `u64`, `usize`, `f64`)
2. **Validation**: Safe constructors catch parameter errors early
3. **Performance**: Create distributions once, reuse them
4. **Composition**: Mix different distribution types naturally
5. **Observations**: Type-safe conditioning on data

## Next Steps

- **[Using Macros](using-macros.md)** - Master `prob!`, `plate!`, and `addr!` macros
- **[Trace Manipulation](trace-manipulation.md)** - Debug and analyze model execution
- **[Bayesian Coin Flip Tutorial](../tutorials/bayesian-coin-flip.md)** - Complete example with multiple distributions

---

**Ready to master macros?** → **[Using Macros](using-macros.md)**