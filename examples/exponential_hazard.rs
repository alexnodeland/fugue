use clap::Parser;
use fugue::*;
use rand::{rngs::StdRng, SeedableRng};

fn hazard_model(obs: f64) -> Model<f64> {
    sample(
        addr!("rate"),
        LogNormal {
            mu: 0.0,
            sigma: 1.0,
        },
    )
    .bind(move |rate| observe(addr!("t"), Exponential { rate }, obs).bind(move |_| pure(rate)))
}

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value_t = 1.0)]
    obs: f64,
    #[arg(long)]
    seed: Option<u64>,
}

fn main() {
    let args = Args::parse();
    let m = hazard_model(args.obs);
    let mut rng = match args.seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };
    let (rate, t) = runtime::handler::run(
        runtime::interpreters::PriorHandler {
            rng: &mut rng,
            trace: runtime::trace::Trace::default(),
        },
        m,
    );
    println!("rate={}, total_logw={}", rate, t.total_log_weight());
}
