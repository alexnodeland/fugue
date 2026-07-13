//! FG-31: `DiscreteUniform` exercises the (now-live) `ChoiceValue::I64` path
//! end-to-end through sample / observe / replay / score, and through full MCMC
//! inference.

use fugue::runtime::handler::run;
use fugue::runtime::interpreters::{PriorHandler, ReplayHandler, ScoreGivenTrace};
use fugue::runtime::trace::{ChoiceValue, Trace};
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

// FG-31: sampling a DiscreteUniform records an I64 choice with the correct
// log-prior and an in-range value.
#[test]
fn fg31_discrete_uniform_prior_sample_records_i64() {
    let mut rng = StdRng::seed_from_u64(1);
    let (k, trace) = run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        sample(addr!("k"), DiscreteUniform::new(1, 6).unwrap()),
    );
    assert!((1..=6).contains(&k));
    // The choice is stored as an I64 variant and is retrievable via get_i64.
    let choice = trace.choices.get(&addr!("k")).unwrap();
    assert!(matches!(choice.value, ChoiceValue::I64(v) if v == k));
    assert_eq!(trace.get_i64(&addr!("k")), Some(k));
    // log-prior = -ln(6).
    assert!((trace.log_prior - -(6.0f64).ln()).abs() < 1e-12);
}

// FG-31: observing an i64 value flows through on_observe_i64 into the
// likelihood; out-of-range observations get -inf.
#[test]
fn fg31_discrete_uniform_observe_i64_likelihood() {
    let mut rng = StdRng::seed_from_u64(2);
    let (_a, trace) = run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        observe(addr!("obs"), DiscreteUniform::new(0, 10).unwrap(), 4i64),
    );
    // In-range: log-likelihood = -ln(11).
    assert!((trace.log_likelihood - -(11.0f64).ln()).abs() < 1e-12);

    let mut rng = StdRng::seed_from_u64(3);
    let (_b, trace_oob) = run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        observe(addr!("obs"), DiscreteUniform::new(0, 10).unwrap(), 42i64),
    );
    assert_eq!(trace_oob.log_likelihood, f64::NEG_INFINITY);
}

// FG-31: ReplayHandler reuses the i64 value from the base trace, and
// ScoreGivenTrace scores that fixed value (re-deriving the same log-prior).
#[test]
fn fg31_discrete_uniform_replay_and_score_i64() {
    // Build a base trace with a known i64 value.
    let mut rng = StdRng::seed_from_u64(4);
    let (k0, base) = run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        sample(addr!("k"), DiscreteUniform::new(-5, 5).unwrap()),
    );

    // Replay reuses the same value.
    let (k_replay, replayed) = run(
        ReplayHandler {
            rng: &mut rng,
            base: base.clone(),
            trace: Trace::default(),
        },
        sample(addr!("k"), DiscreteUniform::new(-5, 5).unwrap()),
    );
    assert_eq!(k_replay, k0);
    assert_eq!(replayed.get_i64(&addr!("k")), Some(k0));

    // Score fixes the value and recomputes the log-prior (-ln(11)).
    let (k_score, scored) = run(
        ScoreGivenTrace {
            base: base.clone(),
            trace: Trace::default(),
        },
        sample(addr!("k"), DiscreteUniform::new(-5, 5).unwrap()),
    );
    assert_eq!(k_score, k0);
    assert!((scored.log_prior - -(11.0f64).ln()).abs() < 1e-12);
    // Re-scoring under a NARROWER support that excludes k0 (if it does) yields
    // -inf, confirming the score path really re-evaluates the i64 log_prob.
    if !(0..=3).contains(&k0) {
        let (_ks, scored_narrow) = run(
            ScoreGivenTrace {
                base: base.clone(),
                trace: Trace::default(),
            },
            sample(addr!("k"), DiscreteUniform::new(0, 3).unwrap()),
        );
        assert_eq!(scored_narrow.log_prior, f64::NEG_INFINITY);
    }
}

// FG-31: full MCMC inference over a DiscreteUniform latent recovers the
// posterior mode — the i64 site is proposed, replayed, and scored across the
// whole adaptive chain without panicking.
#[test]
fn fg31_discrete_uniform_mcmc_recovers_posterior_mode() {
    // Latent k ~ DiscreteUniform(0, 10); observe several y_i ~ N(k, 0.5)
    // concentrated near 7, so the posterior peaks sharply at k = 7.
    let data = [6.9_f64, 7.1, 7.0, 6.8, 7.2, 7.0];
    let model_fn = move || {
        sample(addr!("k"), DiscreteUniform::new(0, 10).unwrap()).bind(move |k| {
            let obs = (0..data.len()).fold(pure(()), move |acc, i| {
                let yi = data[i];
                acc.bind(move |_| observe(addr!("y", i), Normal::new(k as f64, 0.5).unwrap(), yi))
            });
            obs.map(move |_| k)
        })
    };

    let mut rng = StdRng::seed_from_u64(7);
    let samples = adaptive_mcmc_chain(&mut rng, model_fn, 4000, 1000);
    let ks: Vec<i64> = samples.iter().map(|(k, _)| *k).collect();

    // Modal k should be 7.
    let mut counts = [0usize; 11];
    for &k in &ks {
        assert!((0..=10).contains(&k));
        counts[k as usize] += 1;
    }
    let mode = (0..=10).max_by_key(|&k| counts[k as usize]).unwrap();
    assert_eq!(mode, 7, "posterior mode should be 7, counts = {counts:?}");
    // Posterior mean is close to 7 as well.
    let mean = ks.iter().map(|&k| k as f64).sum::<f64>() / ks.len() as f64;
    assert!((mean - 7.0).abs() < 0.3, "posterior mean {mean} off from 7");
}
