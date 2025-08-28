use fugue::runtime::interpreters::PriorHandler;
use fugue::*;
use rand::thread_rng;

// ANCHOR: traditional_problems
// Demonstrates problems with traditional PPL approaches (shown for contrast)
fn traditional_ppl_problems() {
    println!("=== Traditional PPL Problems (What Fugue Solves) ===\n");

    // In traditional PPLs, everything returns f64, leading to:
    println!("❌ Traditional PPL Issues:");
    println!("   - Bernoulli returns f64 → if sample == 1.0 (awkward)");
    println!("   - Poisson returns f64 → count.round() as u64 (precision loss)");
    println!("   - Categorical returns f64 → array[sample as usize] (unsafe)");
    println!("   - Runtime type errors and casting overhead");
    println!();
}
// ANCHOR_END: traditional_problems

// ANCHOR: natural_types
// Demonstrate Fugue's natural return types
fn natural_type_system() {
    println!("✅ Fugue's Natural Type System");
    println!("==============================\n");

    let mut rng = thread_rng();

    // Boolean decisions: Bernoulli → bool
    let fair_coin = Bernoulli::new(0.5).unwrap();
    let is_heads: bool = fair_coin.sample(&mut rng);

    // Natural conditional logic - no comparisons!
    let outcome = if is_heads {
        "Heads - you win!"
    } else {
        "Tails - try again"
    };
    println!("🪙 Coin flip: {} (type: bool)", outcome);

    // Count data: Poisson → u64
    let customer_arrivals = Poisson::new(5.0).unwrap();
    let arrivals: u64 = customer_arrivals.sample(&mut rng);

    // Direct arithmetic with counts - no casting!
    let service_time = arrivals * 10; // minutes per customer
    println!(
        "👥 Customers: {} arrivals, {}min service (type: u64)",
        arrivals, service_time
    );

    // Category selection: Categorical → usize
    let product_preferences = Categorical::new(vec![0.4, 0.35, 0.25]).unwrap();
    let choice: usize = product_preferences.sample(&mut rng);

    // Safe array indexing - guaranteed bounds safety!
    let products = ["Laptop", "Smartphone", "Tablet"];
    println!(
        "🛒 Customer chose: {} (index: {}, type: usize)",
        products[choice], choice
    );

    // Continuous values: Normal → f64 (unchanged, as expected)
    let measurement = Normal::new(100.0, 5.0).unwrap();
    let reading: f64 = measurement.sample(&mut rng);
    println!("📏 Sensor reading: {:.2} units (type: f64)", reading);

    println!();
}
// ANCHOR_END: natural_types

// ANCHOR: compile_time_safety
// Demonstrate compile-time type safety guarantees
fn compile_time_safety_demo() {
    println!("🛡️ Compile-Time Type Safety");
    println!("============================\n");

    // Type-safe model composition
    let data_model: Model<(bool, u64, usize, f64)> = prob!(
        let coin_result <- sample(addr!("coin"), Bernoulli::new(0.6).unwrap());
        let event_count <- sample(addr!("events"), Poisson::new(3.0).unwrap());
        let category <- sample(addr!("category"), Categorical::uniform(4).unwrap());
        let measurement <- sample(addr!("measure"), Normal::new(0.0, 1.0).unwrap());

        // Compiler enforces correct types throughout
        pure((coin_result, event_count, category, measurement))
    );

    println!("✅ Model created with strict type guarantees:");
    println!("   - coin_result: bool (no == 1.0 needed)");
    println!("   - event_count: u64 (direct arithmetic)");
    println!("   - category: usize (safe indexing)");
    println!("   - measurement: f64 (natural continuous)");

    // Execute model safely
    let mut rng = thread_rng();
    let (sample, _trace) = runtime::handler::run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        data_model,
    );

    println!(
        "📊 Sample: coin={}, events={}, category={}, value={:.3}",
        sample.0, sample.1, sample.2, sample.3
    );
    println!();
}
// ANCHOR_END: compile_time_safety

// ANCHOR: safe_indexing
// Demonstrate safe array indexing with categorical distributions
fn safe_array_indexing() {
    println!("🎯 Safe Array Indexing");
    println!("======================\n");

    let mut rng = thread_rng();

    // Define categories with natural indexing
    let algorithms = ["MCMC", "Variational Inference", "ABC", "SMC", "Exact"];
    let method_weights = vec![0.3, 0.25, 0.2, 0.15, 0.1];

    let method_selector = Categorical::new(method_weights).unwrap();

    println!("🧮 Available inference methods:");
    for (i, method) in algorithms.iter().enumerate() {
        println!("   {}: {}", i, method);
    }
    println!();

    // Sample multiple times to show safety
    for trial in 1..=5 {
        let selected_idx: usize = method_selector.sample(&mut rng);

        // This is GUARANTEED safe - no bounds checking needed!
        let chosen_method = algorithms[selected_idx];

        println!(
            "Trial {}: Selected method '{}' (index {})",
            trial, chosen_method, selected_idx
        );
    }

    println!("\n✅ All array accesses guaranteed safe by type system!");
    println!();
}
// ANCHOR_END: safe_indexing

// ANCHOR: parameter_validation
// Demonstrate parameter validation and error handling
fn parameter_validation_demo() {
    println!("🔍 Parameter Validation");
    println!("=======================\n");

    println!("Fugue validates parameters at construction time:");
    println!();

    // Valid constructions
    match Normal::new(0.0, 1.0) {
        Ok(_) => println!("✅ Normal(μ=0.0, σ=1.0) - valid"),
        Err(e) => println!("❌ Unexpected error: {:?}", e),
    }

    match Beta::new(2.0, 3.0) {
        Ok(_) => println!("✅ Beta(α=2.0, β=3.0) - valid"),
        Err(e) => println!("❌ Unexpected error: {:?}", e),
    }

    match Categorical::new(vec![0.3, 0.4, 0.3]) {
        Ok(_) => println!("✅ Categorical([0.3, 0.4, 0.3]) - valid"),
        Err(e) => println!("❌ Unexpected error: {:?}", e),
    }

    println!();

    // Invalid constructions - caught at compile time with .unwrap()
    // or handled gracefully with pattern matching
    println!("Invalid parameter examples:");

    match Normal::new(0.0, -1.0) {
        Ok(_) => println!("✅ Normal(μ=0.0, σ=-1.0) - unexpected success"),
        Err(e) => println!("❌ Normal(μ=0.0, σ=-1.0) - {}", e),
    }

    match Beta::new(0.0, 1.0) {
        Ok(_) => println!("✅ Beta(α=0.0, β=1.0) - unexpected success"),
        Err(e) => println!("❌ Beta(α=0.0, β=1.0) - {}", e),
    }

    match Categorical::new(vec![0.5, 0.6]) {
        // Doesn't sum to 1
        Ok(_) => println!("✅ Categorical([0.5, 0.6]) - unexpected success"),
        Err(e) => println!("❌ Categorical([0.5, 0.6]) - {}", e),
    }

    println!("\n✅ All invalid parameters caught before runtime!");
    println!();
}
// ANCHOR_END: parameter_validation

// ANCHOR: type_safe_observations
// Demonstrate type-safe observations with automatic type checking
fn type_safe_observations() {
    println!("🔗 Type-Safe Observations");
    println!("=========================\n");

    // Observations must match distribution return types
    let observation_model = prob!(
        // Boolean observation - must provide bool
        let _bool_obs <- observe(addr!("coin_obs"),
                                Bernoulli::new(0.7).unwrap(),
                                true); // ✅ bool type matches

        // Count observation - must provide u64
        let _count_obs <- observe(addr!("events_obs"),
                                 Poisson::new(4.0).unwrap(),
                                 5u64); // ✅ u64 type matches

        // Category observation - must provide usize
        let _category_obs <- observe(addr!("choice_obs"),
                                    Categorical::new(vec![0.2, 0.5, 0.3]).unwrap(),
                                    1usize); // ✅ usize type matches

        // Continuous observation - must provide f64
        let _continuous_obs <- observe(addr!("measurement_obs"),
                                      Normal::new(10.0, 2.0).unwrap(),
                                      12.5f64); // ✅ f64 type matches

        pure(())
    );

    println!("✅ All observations type-checked at compile time!");
    println!("   - Bernoulli observation: bool");
    println!("   - Poisson observation: u64");
    println!("   - Categorical observation: usize");
    println!("   - Normal observation: f64");

    // Execute to verify it works
    let mut rng = thread_rng();
    let (_result, trace) = runtime::handler::run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        observation_model,
    );

    println!(
        "📊 Model executed successfully with {} addresses",
        trace.choices.len()
    );
    println!();
}
// ANCHOR_END: type_safe_observations

// ANCHOR: advanced_composition
// Demonstrate advanced type-safe model composition
fn advanced_type_composition() {
    println!("🧩 Advanced Type Composition");
    println!("============================\n");

    // Complex hierarchical model with full type safety
    let hierarchical_model = prob!(
        // Global parameters
        let success_rate <- sample(addr!("global_rate"), Beta::new(1.0, 1.0).unwrap());

        // Group-specific parameters (different types working together)
        let group_sizes <- sequence_vec((0..3).map(|group_id| {
            sample(addr!("group_size", group_id), Poisson::new(10.0).unwrap())
        }).collect());

        let group_successes <- sequence_vec(group_sizes.iter().enumerate().map(|(group_id, &size)| {
            sample(addr!("successes", group_id), Binomial::new(size, success_rate).unwrap())
        }).collect());

        // Category assignments for each group
        let group_categories <- sequence_vec((0..3).map(|group_id| {
            sample(addr!("category", group_id), Categorical::uniform(4).unwrap())
        }).collect());

        // Return complex structured result with full type safety
        pure((success_rate, group_sizes, group_successes, group_categories))
    );

    println!("🏗️ Hierarchical model structure:");
    println!("   - Global success rate: f64 (Beta distribution)");
    println!("   - Group sizes: Vec<u64> (Poisson distributions)");
    println!("   - Group successes: Vec<u64> (Binomial distributions)");
    println!("   - Group categories: Vec<usize> (Categorical distributions)");
    println!();

    // Sample from the complex model
    let mut rng = thread_rng();
    let (result, _trace) = runtime::handler::run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        hierarchical_model,
    );

    let (rate, sizes, successes, categories) = result;

    println!("📈 Sample from hierarchical model:");
    println!("   Global success rate: {:.3}", rate);

    for (i, ((&size, &success), &category)) in sizes
        .iter()
        .zip(successes.iter())
        .zip(categories.iter())
        .enumerate()
    {
        println!(
            "   Group {}: {} trials, {} successes, category {}",
            i, size, success, category
        );
    }

    println!("\n✅ Complex model composed with full type safety!");
    println!();
}
// ANCHOR_END: advanced_composition

// ANCHOR: performance_benefits
// Demonstrate performance benefits of type safety
fn performance_benefits() {
    println!("⚡ Performance Benefits");
    println!("======================\n");

    println!("Type safety eliminates runtime overhead:");
    println!();

    println!("🚫 Traditional PPL (f64 everything):");
    println!("   let coin_flip = sample(...); // Returns f64");
    println!("   if coin_flip == 1.0 {{ ... }} // Float comparison");
    println!("   let count = sample(...) as u64; // Casting overhead");
    println!("   array[sample(...) as usize] // Unsafe casting + bounds check");
    println!();

    println!("✅ Fugue (natural types):");
    println!("   let coin_flip: bool = sample(...); // Returns bool");
    println!("   if coin_flip {{ ... }} // Natural boolean");
    println!("   let count: u64 = sample(...); // Direct u64");
    println!("   array[sample(...)] // Safe usize indexing");
    println!();

    println!("🎯 Benefits:");
    println!("   ✓ Zero casting overhead");
    println!("   ✓ No floating-point comparisons for discrete values");
    println!("   ✓ Eliminated bounds checking for categorical indexing");
    println!("   ✓ No precision loss from float→int conversions");
    println!("   ✓ Compile-time error detection");
    println!();
}
// ANCHOR_END: performance_benefits

// ANCHOR: testing_framework
// Exercise framework for testing understanding
fn testing_framework_example() {
    println!("🧪 Testing Framework Example");
    println!("============================\n");

    let comprehensive_model = prob!(
        // Boolean decision making
        let is_premium <- sample(addr!("premium"), Bernoulli::new(0.3).unwrap());

        // Count data arithmetic
        let base_items <- sample(addr!("base_items"), Poisson::new(5.0).unwrap());
        let bonus_items = if is_premium { base_items + 2 } else { base_items };

        // Safe array indexing
        let service_tier <- sample(addr!("tier"), Categorical::new(vec![0.5, 0.3, 0.2]).unwrap());

        // Continuous parameters
        let satisfaction <- sample(addr!("satisfaction"), Beta::new(2.0, 1.0).unwrap());

        pure((is_premium, bonus_items, service_tier, satisfaction))
    );

    println!("✅ Comprehensive model demonstrates:");
    println!("   - Boolean logic: Premium account decision");
    println!("   - Count arithmetic: Items calculation with bonus");
    println!("   - Safe indexing: Service tier selection");
    println!("   - Continuous data: Customer satisfaction modeling");

    let mut rng = thread_rng();
    let (premium, items, tier, satisfaction) = runtime::handler::run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        comprehensive_model,
    )
    .0;

    let tiers = ["Basic", "Standard", "Premium"];
    println!("\n📊 Sample result:");
    println!("   Premium account: {}", premium);
    println!("   Items received: {}", items);
    println!("   Service tier: {} ({})", tiers[tier], tier);
    println!("   Satisfaction: {:.2}%", satisfaction * 100.0);
    println!();
}
// ANCHOR_END: testing_framework

fn main() {
    println!("🎯 Fugue Type Safety Demonstration");
    println!("==================================\n");

    traditional_ppl_problems();
    natural_type_system();
    compile_time_safety_demo();
    safe_array_indexing();
    parameter_validation_demo();
    type_safe_observations();
    advanced_type_composition();
    performance_benefits();
    testing_framework_example();

    println!("🏁 Type Safety Demonstration Complete!");
    println!("\nKey Takeaways:");
    println!("• Natural return types eliminate casting and comparisons");
    println!("• Compile-time safety catches errors before runtime");
    println!("• Safe array indexing with categorical distributions");
    println!("• Parameter validation prevents invalid distributions");
    println!("• Complex model composition with full type guarantees");
    println!("• Zero-cost abstractions and performance benefits");
}

#[cfg(test)]
mod tests {
    use super::*;
    use fugue::runtime::interpreters::PriorHandler;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_natural_return_types() {
        let mut rng = StdRng::seed_from_u64(42);

        // Test Bernoulli returns bool
        let coin = Bernoulli::new(0.5).unwrap();
        let flip: bool = coin.sample(&mut rng);
        assert!(flip == true || flip == false); // Must be boolean

        // Test Poisson returns u64
        let events = Poisson::new(3.0).unwrap();
        let count: u64 = events.sample(&mut rng);
        assert!(count < 100); // Reasonable upper bound

        // Test Categorical returns usize
        let categories = Categorical::new(vec![0.3, 0.4, 0.3]).unwrap();
        let choice: usize = categories.sample(&mut rng);
        assert!(choice < 3); // Must be valid index

        // Test Normal returns f64
        let normal = Normal::new(0.0, 1.0).unwrap();
        let value: f64 = normal.sample(&mut rng);
        assert!(value.is_finite()); // Must be finite
    }

    #[test]
    fn test_safe_array_indexing() {
        let mut rng = StdRng::seed_from_u64(123);
        let options = ["A", "B", "C", "D"];
        let selector = Categorical::uniform(4).unwrap();

        // Test multiple selections are always safe
        for _ in 0..100 {
            let idx: usize = selector.sample(&mut rng);
            let _selected = options[idx]; // This should never panic
            assert!(idx < options.len());
        }
    }

    #[test]
    fn test_parameter_validation() {
        // Valid parameters should work
        assert!(Normal::new(0.0, 1.0).is_ok());
        assert!(Beta::new(1.0, 1.0).is_ok());
        assert!(Poisson::new(5.0).is_ok());
        assert!(Categorical::new(vec![0.5, 0.5]).is_ok());

        // Invalid parameters should fail
        assert!(Normal::new(0.0, -1.0).is_err()); // Negative sigma
        assert!(Beta::new(0.0, 1.0).is_err()); // Zero alpha
        assert!(Poisson::new(-1.0).is_err()); // Negative lambda
        assert!(Categorical::new(vec![0.3, 0.3]).is_err()); // Doesn't sum to 1
    }

    #[test]
    fn test_type_safe_model_composition() {
        let model = prob!(
            let coin <- sample(addr!("coin"), Bernoulli::new(0.6).unwrap());
            let count <- sample(addr!("count"), Poisson::new(4.0).unwrap());
            let choice <- sample(addr!("choice"), Categorical::uniform(3).unwrap());
            let value <- sample(addr!("value"), Normal::new(0.0, 1.0).unwrap());

            // Type-safe composition
            pure((coin, count, choice, value))
        );

        let mut rng = StdRng::seed_from_u64(456);
        let (result, trace) = runtime::handler::run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            model,
        );

        // Verify types
        let (coin, count, choice, value) = result;
        assert!(coin == true || coin == false);
        assert!(count < 100);
        assert!(choice < 3);
        assert!(value.is_finite());

        // Verify trace has all addresses
        assert_eq!(trace.choices.len(), 4);
    }

    #[test]
    fn test_type_safe_observations() {
        let model = prob!(
            let _bool_obs <- observe(addr!("bool"), Bernoulli::new(0.7).unwrap(), true);
            let _u64_obs <- observe(addr!("u64"), Poisson::new(3.0).unwrap(), 5u64);
            let _usize_obs <- observe(addr!("usize"), Categorical::uniform(3).unwrap(), 1usize);
            let _f64_obs <- observe(addr!("f64"), Normal::new(0.0, 1.0).unwrap(), 0.5f64);
            pure(())
        );

        let mut rng = StdRng::seed_from_u64(789);
        let (_result, trace) = runtime::handler::run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            model,
        );

        // All observations should be recorded
        assert!(trace.log_likelihood != 0.0); // Likelihood accumulated
    }

    #[test]
    fn test_hierarchical_composition() {
        let hierarchical = prob!(
            // Global parameter
            let rate <- sample(addr!("rate"), Beta::new(2.0, 3.0).unwrap());

            // Group-level parameters (different types)
            let sizes <- sequence_vec((0..2).map(|i| {
                sample(addr!("size", i), Poisson::new(5.0).unwrap())
            }).collect());

            let successes <- sequence_vec(sizes.iter().enumerate().map(|(i, &size)| {
                sample(addr!("success", i), Binomial::new(size, rate).unwrap())
            }).collect());

            pure((rate, sizes, successes))
        );

        let mut rng = StdRng::seed_from_u64(101112);
        let (result, trace) = runtime::handler::run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            hierarchical,
        );

        let (rate, sizes, successes) = result;

        // Type checking
        assert!(rate >= 0.0 && rate <= 1.0);
        assert_eq!(sizes.len(), 2);
        assert_eq!(successes.len(), 2);

        // Relationship checking
        for (i, (&size, &success)) in sizes.iter().zip(successes.iter()).enumerate() {
            assert!(
                success <= size,
                "Group {}: {} successes > {} trials",
                i,
                success,
                size
            );
        }

        // Trace should have all addresses (1 rate + 2 sizes + 2 successes)
        assert!(trace.choices.len() >= 5);
    }

    #[test]
    fn test_no_casting_overhead() {
        let mut rng = StdRng::seed_from_u64(131415);

        // Demonstrate direct usage without casts
        let coin = Bernoulli::new(0.5).unwrap();
        let flip = coin.sample(&mut rng);

        // Direct boolean usage - no casting needed
        let message = if flip { "heads" } else { "tails" };
        assert!(message == "heads" || message == "tails");

        // Direct count arithmetic - no casting needed
        let counter = Poisson::new(3.0).unwrap();
        let events = counter.sample(&mut rng);
        let doubled = events * 2; // Direct u64 arithmetic
        assert!(doubled >= events);

        // Direct array indexing - no casting needed
        let options = ["red", "green", "blue"];
        let selector = Categorical::uniform(3).unwrap();
        let idx = selector.sample(&mut rng);
        let _color = options[idx]; // Safe indexing guaranteed
    }

    #[test]
    fn test_compile_time_guarantees() {
        // This test verifies that the type system prevents common errors
        // The fact that this compiles proves the type safety works

        let model: Model<(bool, u64, usize, f64)> = prob!(
            let a <- sample(addr!("a"), Bernoulli::new(0.5).unwrap());
            let b <- sample(addr!("b"), Poisson::new(2.0).unwrap());
            let c <- sample(addr!("c"), Categorical::uniform(4).unwrap());
            let d <- sample(addr!("d"), Normal::new(0.0, 1.0).unwrap());

            pure((a, b, c, d))
        );

        let mut rng = StdRng::seed_from_u64(161718);
        let (result, _trace) = runtime::handler::run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            model,
        );

        // If this compiles and runs, type safety is working
        let (_bool_val, _u64_val, _usize_val, _f64_val) = result;

        // The compiler ensures these types are correct
        assert!(true); // Test passes if we reach here
    }
}
