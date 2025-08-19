use monadic_ppl::*;
use rand::{SeedableRng, rngs::StdRng};
use rand::thread_rng;
use clap::Parser;
fn gaussian_mean(obs:f64)->Model<f64>{ sample(addr!("mu"), Normal{mu:0.0, sigma:5.0}).bind(move|mu|{ observe(addr!("y"), Normal{mu, sigma:1.0}, obs).bind(move |_| pure(mu)) }) }
#[derive(Parser, Debug)]
#[command(name = "gaussian_mean", about = "Prior sample and log-weight for a Gaussian mean model")]
struct Args {
  /// Observation value for y
  #[arg(long, default_value_t = 2.7)]
  obs: f64,

  /// Optional RNG seed for deterministic runs
  #[arg(long)]
  seed: Option<u64>,
}

fn main(){
  let args = Args::parse();
  let m = gaussian_mean(args.obs);
  let mut rng = match args.seed { Some(s) => StdRng::seed_from_u64(s), None => StdRng::from_rng(thread_rng()).expect("seed from thread_rng") };
  let (mu,t) = runtime::handler::run(PriorHandler{rng:&mut rng, trace:Trace::default()}, m);
  if let Some(s) = args.seed {
    println!("mu={} total_logw={} (obs={}, seed={})", mu, t.total_log_weight(), args.obs, s);
  } else {
    println!("mu={} total_logw={} (obs={})", mu, t.total_log_weight(), args.obs);
  }
}
