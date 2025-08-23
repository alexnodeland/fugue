use clap::Parser;
use fugue::*;
use rand::{rngs::StdRng, SeedableRng};

/// Demonstrates the comprehensive type safety improvements in Fugue.
///
/// This example showcases all the new type safety features:
/// - Safe distribution constructors with validation
/// - Type-safe trace accessors
/// - Safe handlers that don't panic
/// - Type-specific diagnostics
/// - Distribution-aware proposals

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value_t = 2.0)]
    obs: f64,

    #[arg(long, default_value_t = 42)]
    seed: u64,

    #[arg(long, default_value_t = 1000)]
    n_samples: usize,
}

/// Type-safe simple model with comprehensive safety features.
fn safe_simple_model(obs: f64) -> FugueResult<Model<f64>> {
    let prior = Normal::new(0.0, 1.0)?;
    let likelihood_dist = Normal::new(obs, 0.5)?;

    let model = sample(addr!("mu"), prior)
        .bind(move |mu| observe(addr!("y"), likelihood_dist, obs).map(move |_| mu));
    Ok(model)
}

/// Type-safe mixture model using safe constructors and natural boolean types.
fn safe_mixture_model(obs: f64) -> FugueResult<Model<(f64, bool)>> {
    let mu1_prior = Normal::new(-2.0, 1.0)?;
    let mu2_prior = Normal::new(2.0, 1.0)?;
    let component_prior = Bernoulli::new(0.5)?;

    let model = sample(addr!("mu1"), mu1_prior).bind(move |mu1| {
        sample(addr!("mu2"), mu2_prior).bind(move |mu2| {
            sample(addr!("component"), component_prior).bind(move |component| {
                let mu = if component { mu2 } else { mu1 }; // Natural boolean usage!
                observe(addr!("y"), Normal::new(mu, 1.0).unwrap(), obs)
                    .map(move |_| (mu1, component))
            })
        })
    });
    Ok(model)
}

/// Demonstrates type-safe count modeling with Poisson.
fn safe_count_model() -> FugueResult<Model<String>> {
    let poisson_dist = Poisson::new(3.0)?;

    let model = sample(addr!("count"), poisson_dist).map(|count| {
        // count is naturally u64, can be used directly with integer operations
        match count {
            0 => "No events occurred".to_string(),
            1 => "One event occurred".to_string(),
            n if n > 10 => "Many events occurred!".to_string(),
            n => format!("{} events occurred", n),
        }
    });
    Ok(model)
}

/// Demonstrates type-safe categorical selection.
fn safe_choice_model() -> FugueResult<Model<String>> {
    let options = vec!["red", "green", "blue"];
    let categorical_dist = Categorical::new(vec![0.5, 0.3, 0.2])?;

    let model = sample(addr!("color_choice"), categorical_dist).map(move |choice_idx| {
        // choice_idx is naturally usize, can be used safely as array index
        let chosen_color = options.get(choice_idx).unwrap_or(&"unknown");
        format!("Chose color: {}", chosen_color)
    });
    Ok(model)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let mut rng = StdRng::seed_from_u64(args.seed);

    println!("Type Safety Improvements in Fugue");
    println!("=================================");

    // 1. Safe Distribution Constructors
    println!("\n1. Safe Distribution Constructors:");

    // Valid construction
    let normal = Normal::new(0.0, 1.0)?;
    println!("   ✅ Valid Normal(0.0, 1.0): {:?}", normal);

    // Invalid construction fails safely
    match Normal::new(0.0, -1.0) {
        Ok(_) => println!("   ❌ Should have failed!"),
        Err(e) => println!("   ✅ Caught invalid Normal(0.0, -1.0): {}", e),
    }

    // 2. Safe Simple Model
    println!("\n2. Type-safe Simple Model:");
    let simple = safe_simple_model(args.obs)?;
    let (mu_value, trace) = runtime::handler::run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        simple,
    );
    println!("   Sampled mu: {:.3}", mu_value);

    // ✅ Type-safe trace access - no more manual pattern matching!
    if let Some(mu_from_trace) = trace.get_f64(&addr!("mu")) {
        println!("   Trace mu value: {:.3}", mu_from_trace);
    }

    // 3. Safe Mixture Model
    println!("\n3. Type-safe Mixture Model:");
    let mixture = safe_mixture_model(args.obs)?;
    let ((mu1, component), mix_trace) = runtime::handler::run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        mixture,
    );
    println!("   Component 1 mean: {:.3}", mu1);
    println!("   Selected component: {}", if component { 2 } else { 1 });

    // ✅ Type-safe trace access for different types
    if let Some(comp_from_trace) = mix_trace.get_bool(&addr!("component")) {
        println!("   Trace component: {}", comp_from_trace);
    }

    // 4. Safe Count Model
    println!("\n4. Type-safe Count Model:");
    let count_model = safe_count_model()?;
    let (message, count_trace) = runtime::handler::run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        count_model,
    );
    println!("   {}", message);

    // ✅ Type-safe u64 access
    if let Some(count) = count_trace.get_u64(&addr!("count")) {
        println!("   Actual count value: {}", count);
    }

    // 5. Safe Choice Model
    println!("\n5. Type-safe Choice Model:");
    let choice_model = safe_choice_model()?;
    let (choice_result, choice_trace) = runtime::handler::run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        choice_model,
    );
    println!("   {}", choice_result);

    // ✅ Type-safe usize access
    if let Some(idx) = choice_trace.get_usize(&addr!("color_choice")) {
        println!("   Chosen index: {}", idx);
    }

    // 6. Safe Handlers - No More Panics!
    println!("\n6. Safe Handlers (no more panics!):");

    // Create a base trace with wrong types intentionally
    let mut base_trace = Trace::default();
    base_trace.insert_choice(addr!("test"), ChoiceValue::Bool(true), -0.5);

    // Safe replay handler won't panic, will warn and fall back to sampling
    let safe_handler = SafeReplayHandler {
        rng: &mut rng,
        base: base_trace,
        trace: Trace::default(),
        warn_on_mismatch: false, // Don't spam output in example
    };

    let test_model = sample(addr!("test"), Normal::new(0.0, 1.0)?);
    let (value, _) = runtime::handler::run(safe_handler, test_model);
    println!(
        "   ✅ Safe handler handled type mismatch gracefully: {:.3}",
        value
    );

    // 7. Type-Safe Diagnostics
    println!("\n7. Type-safe Diagnostics:");

    // Create some sample traces to demonstrate type-safe extraction
    let mut sample_traces = Vec::new();
    for i in 0..5 {
        let mut trace = Trace::default();
        trace.insert_choice(addr!("f64_param"), ChoiceValue::F64(i as f64), -0.5);
        trace.insert_choice(addr!("bool_param"), ChoiceValue::Bool(i % 2 == 0), -0.693);
        trace.insert_choice(addr!("u64_param"), ChoiceValue::U64(i), -2.0);
        sample_traces.push(trace);
    }

    // ✅ Type-safe parameter extraction without lossy conversions
    let f64_values = extract_f64_values(&sample_traces, &addr!("f64_param"));
    let bool_values = extract_bool_values(&sample_traces, &addr!("bool_param"));
    let u64_values = extract_u64_values(&sample_traces, &addr!("u64_param"));

    println!(
        "   ✅ Extracted {} f64 values: {:?}",
        f64_values.len(),
        f64_values
    );
    println!(
        "   ✅ Extracted {} bool values: {:?}",
        bool_values.len(),
        bool_values
    );
    println!(
        "   ✅ Extracted {} u64 values: {:?}",
        u64_values.len(),
        u64_values
    );
    println!("   ✅ No lossy type conversions - preserves original semantics!");

    println!("\n🎉 All Type Safety Features Demonstrated!");
    println!("✅ Safe constructors prevent invalid distributions");
    println!("✅ Type-safe trace accessors eliminate manual matching");
    println!("✅ Safe handlers never panic on type mismatches");
    println!("✅ Type-specific diagnostics avoid lossy conversions");
    println!("✅ Natural types: bool, u64, usize instead of f64 everywhere");

    Ok(())
}
