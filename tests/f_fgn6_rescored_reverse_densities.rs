//! FG-N6: every re-scoring MH entry point reads its reverse-move (death /
//! block) densities from the freshly re-scored current trace, never from the
//! per-choice `logp` the caller stored.
//!
//! A trace assembled with `insert_choice(.., 0.0)` - which is how every
//! fugue-evo `to_trace` / `trace_of` builds one - stores `logp = 0` at every
//! site. Pre-fix, `block_regeneration_mh` summed those zeros as the reverse-birth
//! densities of the regenerated block, and the single-site kernels summed them
//! for sites a proposal made vanish, inflating `log alpha` on exactly the
//! transition meant to correct for the dimension change.
//!
//! The tests are exact equivalences: from the same seed, a step started on the
//! zero-logp trace must make the same decision and land on the same state as a
//! step started on the properly scored trace with the same values. Because
//! `ScoreGivenTrace` consumes no randomness, any difference is a difference in
//! the acceptance ratio.

use fugue::inference::smc::{rejuvenate_particles, Particle};
use fugue::runtime::handler::run;
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// b ~ Bernoulli(0.9); if b { x ~ N(0,1); y ~ N(x, 1) } else { y ~ N(0, 2) }.
/// Flipping b = true -> false makes `x` vanish (a death), whose reverse-birth
/// density is logp(x) = N(x; 0, 1) - not 0.
///
/// The start state below (b = true, x = 1.5 = y) is chosen so the two ratios
/// differ in *decision*, not just in value: the correct branch-closing
/// log alpha is about -2.5 (accept ~8%), while the inflated one - with the
/// stored logp(x) = 0 in place of N(1.5; 0, 1) = -2.04 - is about -0.4
/// (accept ~65%). With a low switch prior instead, both ratios are positive
/// and the move is accepted either way, which is exactly how a first draft of
/// this test failed to discriminate.
fn switch_model() -> Model<bool> {
    let y = 1.5;
    sample(addr!("b"), Bernoulli::new(0.9).unwrap()).and_then(move |b| {
        if b {
            sample(addr!("x"), Normal::new(0.0, 1.0).unwrap()).and_then(move |x| {
                observe(addr!("y"), Normal::new(x, 1.0).unwrap(), y).map(move |_| b)
            })
        } else {
            observe(addr!("y"), Normal::new(0.0, 2.0).unwrap(), y).map(move |_| b)
        }
    })
}

/// The zero-logp trace for state (b = true, x = 1.5): logp(x) = -2.04, so the
/// stored 0 is badly wrong.
fn zero_logp_start() -> Trace {
    let mut t = Trace::default();
    t.insert_choice(addr!("b"), ChoiceValue::Bool(true), 0.0);
    t.insert_choice(addr!("x"), ChoiceValue::F64(1.5), 0.0);
    t
}

fn scored(t: &Trace) -> Trace {
    run(
        ScoreGivenTrace {
            base: t.clone(),
            trace: Trace::default(),
        },
        switch_model(),
    )
    .1
}

fn same_state(a: &Trace, b: &Trace) -> bool {
    a.choices.len() == b.choices.len()
        && a.choices.iter().all(|(k, c)| {
            b.choices
                .get(k)
                .map(|d| d.value == c.value && (d.logp - c.logp).abs() < 1e-12)
                .unwrap_or(false)
        })
        && (a.total_log_weight() - b.total_log_weight()).abs() < 1e-12
}

#[test]
fn fgn6_single_site_mh_from_zero_logp_trace_matches_scored_start() {
    let bad = zero_logp_start();
    let good = scored(&bad);
    let mut deaths = 0;
    for seed in 0u64..300 {
        let mut ad1 = DiminishingAdaptation::new(0.44, 0.7);
        let mut ad2 = DiminishingAdaptation::new(0.44, 0.7);
        let (_, t_bad) = adaptive_single_site_mh(
            &mut StdRng::seed_from_u64(seed),
            switch_model,
            &bad,
            &mut ad1,
        );
        let (_, t_good) = adaptive_single_site_mh(
            &mut StdRng::seed_from_u64(seed),
            switch_model,
            &good,
            &mut ad2,
        );
        if !t_good.choices.contains_key(&addr!("x")) {
            deaths += 1;
        }
        assert!(
            same_state(&t_bad, &t_good),
            "seed {seed}: step from the zero-logp trace diverged from the scored start"
        );
        // Whatever happened, the returned trace is fully scored.
        assert!(same_state(&t_bad, &scored(&t_bad)));
    }
    assert!(
        deaths > 0,
        "no seed exercised the death move; test is not discriminating"
    );
}

#[test]
fn fgn6_block_regeneration_from_zero_logp_trace_matches_scored_start() {
    let bad = zero_logp_start();
    let good = scored(&bad);
    let block = [addr!("b"), addr!("x")];
    let mut closed = 0;
    for seed in 0u64..300 {
        let (_, t_bad) = block_regeneration_mh(
            &mut StdRng::seed_from_u64(seed),
            switch_model,
            &bad,
            &block,
            1.0,
        );
        let (_, t_good) = block_regeneration_mh(
            &mut StdRng::seed_from_u64(seed),
            switch_model,
            &good,
            &block,
            1.0,
        );
        if !t_good.choices.contains_key(&addr!("x")) {
            closed += 1;
        }
        assert!(
            same_state(&t_bad, &t_good),
            "seed {seed}: block move from the zero-logp trace diverged from the scored start"
        );
    }
    assert!(
        closed > 0,
        "no seed closed the branch; test is not discriminating"
    );
}

#[test]
fn fgn6_smc_rejuvenation_from_zero_logp_particles_matches_scored_start() {
    let bad = zero_logp_start();
    let good = scored(&bad);
    let particle = |t: &Trace| Particle {
        trace: t.clone(),
        weight: 1.0,
        log_weight: 0.0,
    };
    let mut deaths = 0;
    for seed in 0u64..150 {
        let mut p_bad = vec![particle(&bad); 4];
        let mut p_good = vec![particle(&good); 4];
        rejuvenate_particles(
            &mut StdRng::seed_from_u64(seed),
            &mut p_bad,
            switch_model,
            0.7,
            2,
        );
        rejuvenate_particles(
            &mut StdRng::seed_from_u64(seed),
            &mut p_good,
            switch_model,
            0.7,
            2,
        );
        for (i, (a, b)) in p_bad.iter().zip(&p_good).enumerate() {
            if !b.trace.choices.contains_key(&addr!("x")) {
                deaths += 1;
            }
            assert!(
                same_state(&a.trace, &b.trace),
                "seed {seed}, particle {i}: rejuvenation from zero-logp trace diverged"
            );
        }
    }
    assert!(
        deaths > 0,
        "no particle exercised the death move; test is not discriminating"
    );
}
