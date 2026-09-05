//! A strict / safe score of a structurally incompatible trace must be a
//! well-defined `Err` / `-inf`, never a divergence or a panic.
//!
//! `StrictScoreGivenTrace` and `SafeScoreGivenTrace` record the first
//! missing or type-mismatched site and then KEEP EXECUTING the program (a
//! handler cannot abort `run`). They used to hand the program
//! `Default::default()` at that site — `false` for a `Bool`, `0.0` for an
//! `f64` — which is not a value the program's own prior could have produced.
//! fugue-evo's grammar prior reads a missing `#leaf` flag as `false` =
//! "function node, recurse" at every depth and overflowed the stack; a missing
//! `sigma` arrived as `0.0` and `Normal::new(mu, 0.0).unwrap()` panicked
//! inside the likelihood. The scorers now hand the program a deterministic
//! draw from the site's own prior, so execution stays inside the program's
//! support and terminates the way the prior does.

use fugue::inference::smc::{try_decode_particle, Particle};
use fugue::runtime::handler::run;
use fugue::runtime::interpreters::{score_given_trace_strict, PriorHandler, SafeScoreGivenTrace};
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// A grammar-shaped program: at each depth a `Bool` decides leaf vs. recurse.
/// Under its own prior it terminates with probability 1 (P(leaf) = 0.6).
/// Fed `false` at every missing site it never terminates.
fn grow(depth: usize) -> Model<usize> {
    sample(addr!("leaf", depth), Bernoulli::new(0.6).unwrap()).bind(move |leaf| {
        if leaf {
            pure(depth)
        } else {
            grow(depth + 1)
        }
    })
}

/// mu with a scale whose prior is positive; `Normal::new(mu, 0.0)` panics.
fn scale_model() -> Model<f64> {
    sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap()).bind(|mu| {
        sample(addr!("sigma"), LogNormal::new(0.0, 0.5).unwrap()).bind(move |sigma| {
            observe(addr!("y"), Normal::new(mu, sigma).unwrap(), 0.3).map(move |_| mu)
        })
    })
}

#[test]
fn strict_score_of_bool_gated_program_from_empty_trace_terminates_with_err() {
    // Pre-fix: unbounded recursion (stack overflow) — every missing `leaf#d`
    // read as `false`. Post-fix: Err naming the first missing site.
    let err = score_given_trace_strict(Trace::default(), grow(0)).unwrap_err();
    assert_eq!(err.code(), ErrorCode::UnexpectedModelStructure);
    assert!(
        err.to_string().contains("leaf#0"),
        "error should name the first offending site: {err}"
    );
}

#[test]
fn strict_score_of_type_mismatched_gate_terminates_with_err() {
    // `leaf#0` present but as an f64: a type mismatch at the gating site.
    let mut base = Trace::default();
    base.insert_choice(addr!("leaf", 0), ChoiceValue::F64(1.0), 0.0);
    let err = score_given_trace_strict(base, grow(0)).unwrap_err();
    assert_eq!(err.code(), ErrorCode::UnexpectedModelStructure);
}

#[test]
fn strict_score_with_missing_scale_does_not_panic_in_the_likelihood() {
    // Pre-fix: sigma = 0.0 → `Normal::new(mu, 0.0).unwrap()` panics.
    let mut base = Trace::default();
    base.insert_choice(addr!("mu"), ChoiceValue::F64(0.2), 0.0);
    let err = score_given_trace_strict(base, scale_model()).unwrap_err();
    assert_eq!(err.code(), ErrorCode::UnexpectedModelStructure);
    assert!(err.to_string().contains("sigma"));
}

#[test]
fn safe_score_of_incompatible_traces_is_neg_inf_and_terminates() {
    // Bool-gated program from an empty trace.
    let (_, t) = run(
        SafeScoreGivenTrace {
            base: Trace::default(),
            trace: Trace::default(),
            warn_on_error: false,
        },
        grow(0),
    );
    assert_eq!(t.log_prior, f64::NEG_INFINITY);
    assert_eq!(t.total_log_weight(), f64::NEG_INFINITY);
    // The fallback draws are recorded, so the invalid trace is still a complete
    // assignment a caller can inspect.
    assert!(t.get_bool(&addr!("leaf", 0)).is_some());

    // Missing scale.
    let mut base = Trace::default();
    base.insert_choice(addr!("mu"), ChoiceValue::F64(0.2), 0.0);
    let (_, t) = run(
        SafeScoreGivenTrace {
            base,
            trace: Trace::default(),
            warn_on_error: false,
        },
        scale_model(),
    );
    assert_eq!(t.log_prior, f64::NEG_INFINITY);
    assert!(t.get_f64(&addr!("sigma")).unwrap() > 0.0);
}

#[test]
fn incompatible_scores_are_deterministic_functions_of_base_and_model() {
    let score = || {
        run(
            SafeScoreGivenTrace {
                base: Trace::default(),
                trace: Trace::default(),
                warn_on_error: false,
            },
            grow(0),
        )
    };
    let (a1, t1) = score();
    let (a2, t2) = score();
    assert_eq!(a1, a2, "fallback draws must be deterministic");
    assert_eq!(t1.choices.len(), t2.choices.len());
    for (addr, c) in &t1.choices {
        assert_eq!(t2.choices[addr].value, c.value);
    }
    // Different sites get independent draws: a chain of missing `leaf#d`
    // decisions must not repeat one outcome forever. The recorded depth is the
    // number of `false` draws before the first `true`; it is finite and, since
    // the very first draw is not forced, the trace has at least one site.
    assert!(!t1.choices.is_empty());
    let depth = a1;
    assert_eq!(t1.choices.len(), depth + 1);
}

#[test]
fn try_decode_particle_returns_err_for_structurally_incompatible_traces() {
    // A particle carrying an empty trace, decoded under the Bool-gated program:
    // pre-fix this recursed without bound.
    let empty = Particle {
        trace: Trace::default(),
        weight: 1.0,
        log_weight: 0.0,
    };
    assert!(try_decode_particle(&empty, || grow(0)).is_err());

    // Missing scale: pre-fix this panicked in `Normal::new`.
    let mut base = Trace::default();
    base.insert_choice(addr!("mu"), ChoiceValue::F64(0.2), 0.0);
    let p = Particle {
        trace: base,
        weight: 1.0,
        log_weight: 0.0,
    };
    assert!(try_decode_particle(&p, scale_model).is_err());

    // And a genuinely complete trace still decodes.
    let mut rng = StdRng::seed_from_u64(1);
    let (v, t) = run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        grow(0),
    );
    let ok = Particle {
        trace: t,
        weight: 1.0,
        log_weight: 0.0,
    };
    assert_eq!(try_decode_particle(&ok, || grow(0)).unwrap(), v);
}

/// Well-formed traces are untouched: strict and safe scores agree with the fast
/// path to the bit.
#[test]
fn compatible_traces_score_identically_across_the_three_scorers() {
    let mut rng = StdRng::seed_from_u64(9);
    for _ in 0..20 {
        let (_, base) = run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            scale_model(),
        );
        let (_, fast) = run(
            ScoreGivenTrace {
                base: base.clone(),
                trace: Trace::default(),
            },
            scale_model(),
        );
        let (_, strict) = score_given_trace_strict(base.clone(), scale_model()).unwrap();
        let (_, safe) = run(
            SafeScoreGivenTrace {
                base,
                trace: Trace::default(),
                warn_on_error: true,
            },
            scale_model(),
        );
        assert_eq!(fast.total_log_weight(), strict.total_log_weight());
        assert_eq!(fast.total_log_weight(), safe.total_log_weight());
    }
}
