use monadic_ppl::*;
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
fn prior_handler_records_choices() {
    let mut rng = StdRng::seed_from_u64(123);
    let (_mu, t) = runtime::handler::run(
        runtime::interpreters::PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        gm(0.0),
    );
    assert!(t.choices.contains_key(&addr!("mu")));
}

#[test]
fn replay_handler_reuses_choice() {
    let mut rng = StdRng::seed_from_u64(1);
    let (_mu, base) = runtime::handler::run(
        runtime::interpreters::PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        gm(0.0),
    );
    let base_mu = base.choices.get(&addr!("mu")).unwrap().value;
    let (_mu2, t2) = runtime::handler::run(
        runtime::interpreters::ReplayHandler {
            rng: &mut rng,
            base: base.clone(),
            trace: Trace::default(),
        },
        gm(1.0),
    );
    assert_eq!(base_mu, t2.choices.get(&addr!("mu")).unwrap().value);
}

#[test]
fn score_given_trace_matches_prior_when_no_observes() {
    // Model with only prior choices
    let m = sample(
        addr!("x"),
        Normal {
            mu: 0.0,
            sigma: 1.0,
        },
    )
    .bind(|_x| pure(()));
    let mut rng = StdRng::seed_from_u64(55);
    let (_a, base) = runtime::handler::run(
        runtime::interpreters::PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        m,
    );
    let (_a2, scored) = runtime::handler::run(
        runtime::interpreters::ScoreGivenTrace {
            base: base.clone(),
            trace: Trace::default(),
        },
        sample(
            addr!("x"),
            Normal {
                mu: 0.0,
                sigma: 1.0,
            },
        )
        .bind(|_x| pure(())),
    );
    assert!((base.total_log_weight() - scored.total_log_weight()).abs() < 1e-9);
}
