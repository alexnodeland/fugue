# Type Safety Features Tutorial

**Level: Intermediate** | **Time: 35 minutes**

Fugue provides comprehensive type safety for probabilistic programming. This tutorial demonstrates the advanced type system that prevents common errors and makes your models more reliable.

## Learning Objectives

By the end of this tutorial, you'll understand:

- Type-safe distribution system with natural return types
- Safe trace accessors that eliminate manual pattern matching
- Safe handlers that never panic on type mismatches
- Type-specific diagnostics without lossy conversions
- How type safety improves model reliability

## The Problem with Unsafe Systems

Traditional probabilistic programming systems often:

- Return `f64` for everything (losing semantic meaning)
- Require manual type conversions (error-prone)
- Panic on type mismatches (crashes programs)
- Force lossy conversions in diagnostics

Fugue eliminates these issues with a comprehensive type safety system.

## Part 1: Type-Safe Distributions

### Natural Return Types

**Try it**: Run with `cargo run --example fully_type_safe`

```rust
{{#include ../../../examples/fully_type_safe.rs}}
```

### Key Features

**Before (unsafe):**

```rust
// Everything returns f64 - semantic meaning lost
let coin_result: f64 = bernoulli.sample();  // 0.0 or 1.0?
let count: f64 = poisson.sample();          // Actually an integer
let choice: f64 = categorical.sample();     // Array index as float?

// Requires manual, error-prone conversions
let is_heads = coin_result > 0.5;           // Fragile comparison
let actual_count = count as u64;            // Lossy conversion
let array_index = choice as usize;          // Potential crash
```

**After (type-safe):**

```rust
// Natural types preserve semantic meaning
let coin_result: bool = Bernoulli::new(0.6)?.sample();    // true/false
let count: u64 = Poisson::new(3.0)?.sample();             // Natural integer
let choice: usize = Categorical::new(vec![0.3, 0.7])?.sample(); // Safe array index

// Direct usage - no conversions needed
if coin_result { /* heads */ }
for i in 0..count { /* process events */ }
let selected = options[choice];  // Safe indexing
```

### Benefits

✅ **Compiler enforcement** - Wrong types caught at compile time  
✅ **Zero runtime overhead** - No conversion costs  
✅ **Semantic clarity** - Types match their mathematical meaning  
✅ **Safe operations** - No invalid array accesses or type punning

## Part 2: Safe Trace Operations

### Type-Safe Trace Accessors

**Traditional approach (error-prone):**

```rust
// Manual pattern matching required
match trace.choices.get(&addr!("bias")) {
    Some(choice) => match choice.value {
        ChoiceValue::F64(bias) => bias,
        _ => panic!("Wrong type!"),  // Crashes on mismatch
    },
    None => panic!("Missing choice!"),
}
```

**Type-safe approach:**

```rust
// Clean, safe accessor methods
if let Some(bias) = trace.get_f64(&addr!("bias")) {
    println!("Bias: {:.3}", bias);
}

// Type-specific accessors prevent errors
let coin_result = trace.get_bool(&addr!("coin")).unwrap_or(false);
let event_count = trace.get_u64(&addr!("events")).unwrap_or(0);
let choice_index = trace.get_usize(&addr!("choice")).unwrap_or(0);
```

### Multiple Type Support

```rust
// Extract different types safely
let f64_values = extract_f64_values(&traces, &addr!("continuous_param"));
let bool_values = extract_bool_values(&traces, &addr!("binary_outcome"));
let u64_values = extract_u64_values(&traces, &addr!("count_data"));
let usize_values = extract_usize_values(&traces, &addr!("categories"));

// No lossy conversions - original semantics preserved
assert_eq!(bool_values.len(), traces.len());  // All values preserved
```

## Part 3: Safe Handlers

### Panic-Free Operation

**Try it**: Run with `cargo run --example type_safe_improvements`

```rust
{{#include ../../../examples/type_safe_improvements.rs}}
```

### Safe Replay Handler

```rust
// Traditional handler - panics on type mismatch
let ReplayHandler { base, .. } = /* ... */;
// If base has wrong type for current model - PANIC!

// Safe handler - gracefully handles mismatches
let SafeReplayHandler {
    base,
    warn_on_mismatch: true,
    ..
} = /* ... */;
// Logs warning and falls back to sampling - no crash
```

### Benefits

✅ **No crashes** - Type mismatches handled gracefully  
✅ **Automatic fallback** - Switches to sampling when needed  
✅ **Debugging support** - Optional warnings for development  
✅ **Production ready** - Robust operation in deployment

## Part 4: Advanced Type Safety Examples

### Complex Type-Safe Models

```rust
fn type_safe_mixture_model(data: Vec<f64>) -> Model<(Vec<f64>, Vec<bool>)> {
    prob! {
        let n_components = 3;

        // Sample component means (continuous parameters)
        let mut means = Vec::new();
        for i in 0..n_components {
            let mu <- sample(addr!("mu", i), Normal::new(0.0, 10.0)?);
            means.push(mu);
        }

        // Sample component assignments (discrete choices)
        let mut assignments = Vec::new();
        for (i, &x) in data.iter().enumerate() {
            // Natural boolean for binary choice
            let is_component_1 <- sample(addr!("assignment", i), Bernoulli::new(0.5)?);
            assignments.push(is_component_1);

            // Type-safe conditional observation
            let chosen_mean = if is_component_1 { means[0] } else { means[1] };
            observe(addr!("x", i), Normal::new(chosen_mean, 1.0)?, x);
        }

        pure((means, assignments))
    }
}
```

### Type-Safe Diagnostics

```rust
fn comprehensive_diagnostics(traces: &[Trace]) {
    // Extract continuous parameters without conversion loss
    let continuous_params = extract_f64_values(traces, &addr!("continuous"));
    let r_hat = compute_r_hat(&continuous_params);
    let ess = effective_sample_size(&continuous_params);

    // Extract discrete parameters with correct types
    let binary_outcomes = extract_bool_values(traces, &addr!("binary"));
    let success_rate = binary_outcomes.iter().filter(|&&x| x).count() as f64 / binary_outcomes.len() as f64;

    // Extract counts preserving integer semantics
    let event_counts = extract_u64_values(traces, &addr!("events"));
    let total_events: u64 = event_counts.iter().sum();
    let avg_events = total_events as f64 / event_counts.len() as f64;

    println!("Continuous: R-hat={:.3}, ESS={:.1}", r_hat, ess);
    println!("Binary: Success rate={:.1}%", success_rate * 100.0);
    println!("Counts: Total={}, Average={:.1}", total_events, avg_events);
}
```

## Part 5: Type Safety Best Practices

### 1. Use Natural Types from the Start

```rust
// ✅ Good - types match semantics
fn model_customer_behavior() -> Model<(bool, u64, usize)> {
    prob! {
        let will_purchase <- sample(addr!("purchase"), Bernoulli::new(0.3)?);
        let items_viewed <- sample(addr!("views"), Poisson::new(5.0)?);
        let category <- sample(addr!("category"), Categorical::new(vec![0.4, 0.3, 0.3])?);

        pure((will_purchase, items_viewed, category))
    }
}

// ❌ Bad - everything as f64 loses meaning
fn unsafe_model() -> Model<(f64, f64, f64)> {
    prob! {
        let purchase_float <- sample(addr!("purchase"), /* convert to f64 somehow */);
        let views_float <- sample(addr!("views"), /* awkward conversion */);
        let category_float <- sample(addr!("category"), /* index as float? */);

        pure((purchase_float, views_float, category_float))
    }
}
```

### 2. Leverage Safe Trace Accessors

```rust
// ✅ Good - safe accessors with fallbacks
fn analyze_results(trace: &Trace) -> AnalysisResult {
    let purchase_prob = trace.get_f64(&addr!("purchase_prob")).unwrap_or(0.0);
    let did_purchase = trace.get_bool(&addr!("purchase")).unwrap_or(false);
    let view_count = trace.get_u64(&addr!("views")).unwrap_or(0);
    let category_idx = trace.get_usize(&addr!("category")).unwrap_or(0);

    AnalysisResult {
        purchase_probability: purchase_prob,
        actually_purchased: did_purchase,
        items_viewed: view_count,
        preferred_category: category_idx,
    }
}

// ❌ Bad - manual pattern matching
fn unsafe_analysis(trace: &Trace) -> AnalysisResult {
    let purchase_prob = match trace.choices.get(&addr!("purchase_prob")) {
        Some(choice) => match choice.value {
            ChoiceValue::F64(x) => x,
            _ => panic!("Wrong type!"),
        },
        None => panic!("Missing choice!"),
    };
    // ... repeat for every parameter
}
```

### 3. Handle Type Mismatches Gracefully

```rust
// ✅ Good - safe handlers in production
fn robust_inference() {
    let safe_handler = SafeReplayHandler {
        rng: &mut rng,
        base: potentially_incompatible_trace,
        trace: Trace::default(),
        warn_on_mismatch: false,  // Silent in production
    };

    let result = runtime::handler::run(safe_handler, model);
    // Will never panic, even with type mismatches
}

// ❌ Risky - can crash in production
fn fragile_inference() {
    let handler = ReplayHandler {
        rng: &mut rng,
        base: potentially_incompatible_trace,  // Might crash!
        trace: Trace::default(),
    };

    let result = runtime::handler::run(handler, model);  // May panic
}
```

## Type Safety Checklist

When building models, ensure:

- ✅ **Natural return types**: `bool` for binary, `u64` for counts, `usize` for indices
- ✅ **Safe constructors**: Use `?` for error handling in distribution creation
- ✅ **Type-safe accessors**: Use `trace.get_f64()` etc. instead of manual pattern matching
- ✅ **Safe handlers**: Use `SafeReplayHandler` in production environments
- ✅ **Preserve semantics**: Don't convert discrete types to floating point
- ✅ **Graceful fallbacks**: Handle missing or mistyped trace values

## Performance Benefits

Type safety in Fugue has **zero runtime cost**:

- No boxing/unboxing of values
- No runtime type checking
- No conversion overhead
- Compile-time optimizations preserved

The safety comes from Rust's type system, not runtime checks.

## Migration from Unsafe Systems

If migrating from systems that use `f64` everywhere:

```rust
// Old: Everything as f64
let coin_result: f64 = sample_float();
let is_heads = coin_result > 0.5;

// New: Natural types
let is_heads: bool = sample(addr!("coin"), Bernoulli::new(0.5)?);
```

The type-safe version is both safer and clearer.

## Next Steps

Now that you understand type safety:

1. **[Trace Manipulation](trace-manipulation.md)** - Advanced trace operations with type safety
2. **[Advanced Inference](advanced-inference.md)** - Sophisticated techniques with safe types
3. **[Basic Inference](basic-inference.md)** - Apply type safety to fundamental models

## Key Takeaways

- **Natural types** preserve semantic meaning and prevent errors
- **Safe accessors** eliminate manual pattern matching and potential panics
- **Safe handlers** ensure robust operation even with type mismatches
- **Zero overhead** - safety comes from Rust's compile-time type system
- **Better diagnostics** - no lossy conversions in analysis

Fugue's type safety makes probabilistic programming more reliable, maintainable, and clear.

---

**Ready for advanced trace operations?** → **[Trace Manipulation Tutorial](trace-manipulation.md)**
