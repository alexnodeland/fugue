use fugue::*;
use rand::thread_rng;
fn gaussian_mean(obs: f64) -> Model<f64> {
    sample(addr!("mu"), Normal::new(0.0, 5.0).unwrap()).bind(move |mu| {
        observe(addr!("y"), Normal::new(mu, 1.0).unwrap(), obs).bind(move |_| pure(mu))
    })
}
#[test]
fn prior_runs() {
    let m = gaussian_mean(0.5);
    let mut rng = thread_rng();
    let (mu, t) = runtime::handler::run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        m,
    );
    assert!(t.choices.contains_key(&addr!("mu")));
    assert!(mu.is_finite());
}
#[test]
fn replay_reuses() {
    let m = gaussian_mean(0.0);
    let mut rng = thread_rng();
    let (_mu, base) = runtime::handler::run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        m,
    );
    let base_mu = &base.choices.get(&addr!("mu")).unwrap().value;
    let m2 = gaussian_mean(3.14);
    let (_mu2, t2) = runtime::handler::run(
        ReplayHandler {
            rng: &mut rng,
            base: base.clone(),
            trace: Trace::default(),
        },
        m2,
    );
    assert_eq!(base_mu, &t2.choices.get(&addr!("mu")).unwrap().value);
}
#[test]
fn factor_adds_weight() {
    let m = factor(-1.23).bind(|_| pure(()));
    let mut rng = thread_rng();
    let (_u, t) = runtime::handler::run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        m,
    );
    assert!((t.total_log_weight() + 1.23).abs() < 1e-9);
}
