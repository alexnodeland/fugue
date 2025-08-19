use monadic_ppl::*; use rand::thread_rng;
fn gaussian_mean(obs:f64)->Model<f64>{ sample(addr!("mu"), Normal{mu:0.0, sigma:5.0}).bind(move|mu|{ observe(addr!("y"), Normal{mu, sigma:1.0}, obs).bind(move |_| pure(mu)) }) }
fn main(){ let m=gaussian_mean(2.7); let mut rng=thread_rng(); let (mu,t)=runtime::handler::run(PriorHandler{rng:&mut rng, trace:Trace::default()}, m); println!("mu={} total_logw={}", mu, t.total_log_weight()); }
