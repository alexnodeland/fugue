use clap::Parser;
use fugue::*;
use rand::{rngs::StdRng, SeedableRng};

/// Simple Beta-Binomial conjugate model.
fn beta_binomial_model(n: u64, k: u64) -> Model<f64> {
    sample(addr!("p"), Beta::new(2.0, 2.0).unwrap()).bind(move |p| {
        observe(addr!("obs"), Binomial::new(n, p).unwrap(), k).bind(move |_| pure(p))
    })
}

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value_t = 10)]
    trials: u64,

    #[arg(long, default_value_t = 6)]
    successes: u64,

    #[arg(long, default_value_t = 42)]
    seed: u64,
}

fn main() {
    let args = Args::parse();

    if args.successes > args.trials {
        eprintln!("Error: successes cannot exceed trials");
        std::process::exit(1);
    }

    println!(
        "Data: {}/{} successes ({:.1}%)",
        args.successes,
        args.trials,
        100.0 * args.successes as f64 / args.trials as f64
    );

    let model = beta_binomial_model(args.trials, args.successes);
    let mut rng = StdRng::seed_from_u64(args.seed);
    let (p, t) = runtime::handler::run(
        runtime::interpreters::PriorHandler {
            rng: &mut rng,
            trace: runtime::trace::Trace::default(),
        },
        model,
    );

    println!("\nBeta-Binomial Model Results:");
    println!("  Estimated success probability: {:.3}", p);
    println!("  Total log weight: {:.3}", t.total_log_weight());
    println!("  Posterior sample suggests {:.1}% success rate", p * 100.0);
}
