use clap::Parser;
use fugue::*;
use rand::{rngs::StdRng, SeedableRng};

/// Demonstrates the fully type-safe distribution system.
///
/// This example shows how the new system provides complete type safety:
/// - Bernoulli returns bool
/// - Poisson returns u64  
/// - Categorical returns usize
/// - No type coercion or unsafe conversions

/// Type-safe mixture model using natural types
fn type_safe_mixture_model(obs: f64) -> Model<(f64, f64)> {
    sample(addr!("weight"), Beta::new(1.0, 1.0).unwrap()).bind(move |weight| {
        sample(addr!("mu1"), Normal::new(-2.0, 1.0).unwrap()).bind(move |mu1| {
            sample(addr!("mu2"), Normal::new(2.0, 1.0).unwrap()).bind(move |mu2| {
                // ✅ Type-safe: Bernoulli now returns bool directly!
                sample(addr!("component"), Bernoulli::new(weight).unwrap()).bind(move |comp| {
                    let mu = if comp { mu2 } else { mu1 }; // Natural boolean usage!
                    observe(addr!("y"), Normal::new(mu, 1.0).unwrap(), obs)
                        .bind(move |_| pure((mu1, mu2)))
                })
            })
        })
    })
}

/// Example showing type-safe counting with Poisson
fn count_model_example() -> Model<String> {
    // ✅ Type-safe: Poisson returns u64 directly!
    sample(addr!("count"), Poisson::new(3.0).unwrap()).bind(|count| {
        // count is naturally u64, can be used directly with integer operations
        let message = match count {
            0 => "No events occurred".to_string(),
            1 => "One event occurred".to_string(),
            n if n > 10 => "Many events occurred!".to_string(),
            n => format!("{} events occurred", n),
        };
        pure(message)
    })
}

/// Example showing type-safe categorical selection
fn choice_model_example() -> Model<String> {
    let options = vec!["red", "green", "blue"];

    // ✅ Type-safe: Categorical returns usize directly!
    sample(
        addr!("color_choice"),
        Categorical::new(vec![0.5, 0.3, 0.2]).unwrap(),
    )
    .bind(move |choice_idx| {
        // choice_idx is naturally usize, can be used safely as array index
        // No need for error-prone f64 to usize conversion
        let chosen_color = options.get(choice_idx).unwrap_or(&"unknown").to_string();
        pure(format!("Chose color: {}", chosen_color))
    })
}

/// Example of type-safe observations
fn observation_example() -> Model<()> {
    // Observe boolean outcome directly
    observe(addr!("coin_result"), Bernoulli::new(0.6).unwrap(), true).bind(|_| {
        // Observe integer count directly
        observe(addr!("event_count"), Poisson::new(4.0).unwrap(), 7u64).bind(|_| {
            // Observe categorical choice directly
            observe(
                addr!("user_choice"),
                Categorical::new(vec![0.2, 0.3, 0.5]).unwrap(),
                2usize,
            )
        })
    })
}

/// Example using direct distribution access outside the Model system
fn direct_distributions_example() -> Model<String> {
    // All distributions work both inside and outside the Model system
    let coin = Bernoulli::new(0.7).unwrap();
    let mut rng = StdRng::seed_from_u64(42);

    // Direct sampling - returns bool
    let flip: bool = coin.sample(&mut rng);

    // Direct log_prob - takes &bool
    let log_prob = coin.log_prob(&flip);

    pure(format!(
        "Direct sampling: flip={}, log_prob={:.3}",
        flip, log_prob
    ))
}

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value_t = 0.0)]
    obs: f64,

    #[arg(long, default_value_t = 42)]
    seed: u64,
}

fn main() {
    let args = Args::parse();

    let mut rng = StdRng::seed_from_u64(args.seed);

    println!("Fully Type-Safe Distribution System");
    println!("===================================");

    println!("\n1. Type-safe mixture model:");
    let model = type_safe_mixture_model(args.obs);
    let ((mu1, mu2), _t) = runtime::handler::run(
        runtime::interpreters::PriorHandler {
            rng: &mut rng,
            trace: runtime::trace::Trace::default(),
        },
        model,
    );
    println!("   Component 1 mean: {:.3}", mu1);
    println!("   Component 2 mean: {:.3}", mu2);

    println!("\n2. Type-safe count example:");
    let count_model = count_model_example();
    let (message, _t) = runtime::handler::run(
        runtime::interpreters::PriorHandler {
            rng: &mut rng,
            trace: runtime::trace::Trace::default(),
        },
        count_model,
    );
    println!("   {}", message);

    println!("\n3. Type-safe choice example:");
    let choice_model = choice_model_example();
    let (choice_message, _t) = runtime::handler::run(
        runtime::interpreters::PriorHandler {
            rng: &mut rng,
            trace: runtime::trace::Trace::default(),
        },
        choice_model,
    );
    println!("   {}", choice_message);

    println!("\n4. Type-safe observations:");
    let obs_model = observation_example();
    let ((), trace) = runtime::handler::run(
        runtime::interpreters::PriorHandler {
            rng: &mut rng,
            trace: runtime::trace::Trace::default(),
        },
        obs_model,
    );
    println!("   Observation log-likelihood: {:.3}", trace.log_likelihood);

    println!("\n5. Direct distribution usage:");
    let direct_result = direct_distributions_example();
    let (direct_message, _t) = runtime::handler::run(
        runtime::interpreters::PriorHandler {
            rng: &mut rng,
            trace: runtime::trace::Trace::default(),
        },
        direct_result,
    );
    println!("   {}", direct_message);

    println!("\n🎉 Fully Type-Safe System Features:");
    println!("✅ Bernoulli: sample() -> bool, log_prob(&bool)");
    println!("✅ Poisson:   sample() -> u64,  log_prob(&u64)");
    println!("✅ Categorical: sample() -> usize, log_prob(&usize)");
    println!("✅ Binomial:  sample() -> u64,  log_prob(&u64)");
    println!("✅ Continuous distributions: f64 as appropriate");
    println!("✅ Compiler enforces type safety throughout");
    println!("✅ Zero runtime overhead vs hand-coded type conversions");
}
