use fugue::*;
use rand::{rngs::StdRng, SeedableRng};

fn gm(obs: f64) -> Model<f64> {
    sample(
        addr!("mu"),
        Normal {
            mu: 0.0,
            sigma: 1.0,
        },
    )
    .bind(move |mu| observe(addr!("y"), Normal { mu, sigma: 1.0 }, obs).bind(move |_| pure(mu)))
}

#[test]
fn vi_elbo_produces_finite_estimate() {
    let mut rng = StdRng::seed_from_u64(999);
    let elbo = inference::vi::estimate_elbo(&mut rng, || gm(0.5), 5);
    assert!(elbo.is_finite());
}

#[test]
fn smc_prior_particles_normalizes_weights() {
    let mut rng = StdRng::seed_from_u64(42);
    let parts = inference::smc::smc_prior_particles(&mut rng, 10, || gm(0.0));
    let sum: f64 = parts.iter().map(|p| p.weight).sum();
    assert!((sum - 1.0).abs() < 1e-9);
}

#[test]
fn mh_transition_returns_trace() {
    let mut rng = StdRng::seed_from_u64(5);
    let (_a0, t0) = runtime::handler::run(
        runtime::interpreters::PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        gm(0.2),
    );
    let (_a1, t1) = inference::mh::single_site_random_walk_mh(&mut rng, 0.1, || gm(0.2), &t0);
    // Should at least return a trace with a choice at mu
    assert!(t1.choices.contains_key(&addr!("mu")));
}
