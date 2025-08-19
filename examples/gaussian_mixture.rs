use monadic_ppl::*;
use rand::{SeedableRng, rngs::StdRng};
use clap::Parser;

fn mixture_model(obs: f64) -> Model<(f64, f64)> {
  // Simple 2-component mixture with fixed weights
  sample(addr!("z"), Uniform{low:0.0, high:1.0}).bind(move |u|{
    let choose_first = u < 0.5;
    let mu = if choose_first { -2.0 } else { 2.0 };
    let sigma = 1.0;
    sample(addr!("x"), Normal{mu, sigma}).bind(move |x|{
      observe(addr!("y"), Normal{mu:x, sigma:1.0}, obs).bind(move |_| pure((x, obs)))
    })
  })
}

#[derive(Parser, Debug)]
struct Args { #[arg(long, default_value_t=0.0)] obs: f64, #[arg(long)] seed: Option<u64> }

fn main(){
  let args = Args::parse();
  let m = mixture_model(args.obs);
  let mut rng = match args.seed { Some(s)=>StdRng::seed_from_u64(s), None=>StdRng::from_entropy() };
  let ((x,_), t) = runtime::handler::run(runtime::interpreters::PriorHandler{rng:&mut rng, trace: runtime::trace::Trace::default()}, m);
  println!("x={}, total_logw={}", x, t.total_log_weight());
}


