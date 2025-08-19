use fugue::*;
use rand::{SeedableRng, rngs::StdRng};
use clap::Parser;

/// Simple 2-component mixture model.
fn mixture_model(obs: f64) -> Model<(f64, f64)> {
    sample(addr!("weight"), Beta{alpha: 1.0, beta: 1.0}).bind(move |weight| {
        sample(addr!("mu1"), Normal{mu: -2.0, sigma: 1.0}).bind(move |mu1| {
            sample(addr!("mu2"), Normal{mu: 2.0, sigma: 1.0}).bind(move |mu2| {
                sample(addr!("component"), Bernoulli{p: weight}).bind(move |comp| {
                    let mu = if comp == 1.0 { mu2 } else { mu1 };
                    observe(addr!("y"), Normal{mu, sigma: 1.0}, obs).bind(move |_| {
                        pure((mu1, mu2))
                    })
                })
            })
        })
    })
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
    
    let model = mixture_model(args.obs);
    let mut rng = StdRng::seed_from_u64(args.seed);
    let ((mu1, mu2), t) = runtime::handler::run(
        runtime::interpreters::PriorHandler{rng: &mut rng, trace: runtime::trace::Trace::default()}, 
        model
    );
    
    println!("Simple Mixture Model Results:");
    println!("  Component 1 mean: {:.3}", mu1);
    println!("  Component 2 mean: {:.3}", mu2);
    println!("  Observation: {:.3}", args.obs);
    println!("  Total log weight: {:.3}", t.total_log_weight());
}
