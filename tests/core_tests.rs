use monadic_ppl::*;
use rand::{SeedableRng, rngs::StdRng};

#[test]
fn address_macro_stability() {
  let a1 = addr!("mu");
  let a2 = addr!("mu");
  let b1 = addr!("mu", 1);
  assert_eq!(a1, a2);
  assert_ne!(a1, b1);
}

#[test]
fn model_pure_and_map() {
  let m = pure(2).map(|x| x + 3);
  let mut rng = StdRng::seed_from_u64(1);
  let (val, _t) = runtime::handler::run(runtime::interpreters::PriorHandler{rng: &mut rng, trace: Trace::default()}, m);
  assert_eq!(val, 5);
}

#[test]
fn model_sampling_and_observe() {
  // mu ~ N(0, 1); observe y ~ N(mu, 1) at 0.0
  let m = sample(addr!("mu"), Normal{mu:0.0, sigma:1.0}).bind(|mu|{
    observe(addr!("y"), Normal{mu, sigma:1.0}, 0.0).bind(move |_| pure(mu))
  });
  let mut rng = StdRng::seed_from_u64(7);
  let (_mu, t) = runtime::handler::run(runtime::interpreters::PriorHandler{rng: &mut rng, trace: Trace::default()}, m);
  assert!(t.choices.contains_key(&addr!("mu")));
  // Likelihood term should have been accumulated
  assert!(t.log_likelihood.is_finite());
}


