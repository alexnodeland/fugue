//! Regression tests for the July 2026 audit findings owned by the f-runtime
//! work package: FG-19, FG-20, FG-21, FG-26, FG-47, FG-48, FG-52, FG-54, FG-61.
//!
//! Each test names the finding it guards in a comment and is written so it would
//! FAIL on the pre-fix code.

use fugue::core::distribution::Distribution;
use fugue::runtime::handler::run;
use fugue::runtime::interpreters::{
    score_given_trace_reconciled, score_given_trace_strict, PriorHandler, ReplayHandler,
    ScoreGivenTrace,
};
use fugue::runtime::trace::{ChoiceValue, Trace};
use fugue::*;
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};

// ---------------------------------------------------------------------------
// A tiny signed-discrete distribution used only to exercise the i64 sample path
// (FG-54). A real `DiscreteUniform` lands in a later work package; this mock
// stands in so the i64 Model/Handler/Trace plumbing can be tested end to end.
// Inclusive range [lo, hi].
// ---------------------------------------------------------------------------
#[derive(Clone)]
struct DiscreteUniformI64 {
    lo: i64,
    hi: i64,
}
impl Distribution<i64> for DiscreteUniformI64 {
    fn sample(&self, rng: &mut dyn RngCore) -> i64 {
        let n = (self.hi - self.lo + 1) as u64;
        self.lo + (rng.next_u64() % n) as i64
    }
    fn log_prob(&self, x: &i64) -> f64 {
        if *x >= self.lo && *x <= self.hi {
            -((self.hi - self.lo + 1) as f64).ln()
        } else {
            f64::NEG_INFINITY
        }
    }
    fn clone_box(&self) -> Box<dyn Distribution<i64>> {
        Box::new(self.clone())
    }
}

// ===========================================================================
// FG-19: interpretation is stack-safe (trampoline). A 100k-deep sample+bind
// chain must not overflow the stack. Driven on a small (512 KiB) stack so the
// guarantee is explicit; this overflows on the pre-fix recursive interpreter.
// ===========================================================================
#[test]
fn fg19_deep_model_is_stack_safe_via_public_api() {
    fn build(i: usize, n: usize, acc: f64) -> Model<f64> {
        if i >= n {
            pure(acc)
        } else {
            sample(addr!("s", i), Normal::new(0.0, 1.0).unwrap())
                .bind(move |x| build(i + 1, n, acc + x))
        }
    }

    let handle = std::thread::Builder::new()
        .stack_size(512 * 1024)
        .spawn(|| {
            let n = 100_000;
            let mut rng = StdRng::seed_from_u64(19);
            let (sum, trace) = run(
                PriorHandler {
                    rng: &mut rng,
                    trace: Trace::default(),
                },
                build(0, n, 0.0),
            );
            assert!(sum.is_finite());
            assert_eq!(trace.choices.len(), n);
        })
        .expect("spawn");
    handle
        .join()
        .expect("deep interpretation overflowed the stack (FG-19 regression)");
}

// ===========================================================================
// FG-20 / FG-21: structure-varying scoring returns a Result instead of
// panicking.
// ===========================================================================

// FG-21: the STRICT path returns Err (not a panic) when the model reaches an
// address absent from the base trace. Pre-fix, ScoreGivenTrace panicked here.
#[test]
fn fg21_strict_scoring_errors_on_new_address_instead_of_panicking() {
    let mut rng = StdRng::seed_from_u64(20);
    let (_, base) = run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        sample(addr!("x"), Normal::new(0.0, 1.0).unwrap()),
    );

    // Same structure => Ok.
    let ok = score_given_trace_strict(
        base.clone(),
        sample(addr!("x"), Normal::new(0.5, 2.0).unwrap()),
    );
    assert!(ok.is_ok());

    // Model reaches "y", which the base trace never recorded => Err, no panic.
    let err = score_given_trace_strict(base, sample(addr!("y"), Normal::new(0.0, 1.0).unwrap()))
        .unwrap_err();
    assert_eq!(err.code(), ErrorCode::UnexpectedModelStructure);
}

// FG-20: the RECONCILING path samples NEW addresses from the prior (accumulating
// their log_prior) and reports VANISHED addresses so the caller can drop them.
#[test]
fn fg20_reconciling_scoring_samples_fresh_and_reports_vanished() {
    let mut rng = StdRng::seed_from_u64(21);
    // Base trace has two sites: "x" and "z".
    let (_, base) = run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
            .bind(|_| sample(addr!("z"), Normal::new(0.0, 1.0).unwrap())),
    );

    // New model keeps "x" but drops "z" and introduces "y" (a fresh dimension).
    let x_val = base.get_f64(&addr!("x")).unwrap();
    let fresh_dist = Normal::new(3.0, 0.25).unwrap();
    let (_v, trace, report) = score_given_trace_reconciled(
        base.clone(),
        &mut rng,
        sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
            .bind(move |_| sample(addr!("y"), Normal::new(3.0, 0.25).unwrap())),
    )
    .unwrap();

    assert_eq!(report.fresh_addresses, vec![addr!("y")]);
    assert_eq!(report.vanished_addresses, vec![addr!("z")]);

    // "x" replayed from base, "y" proposed fresh; both present, "z" dropped.
    assert_eq!(trace.get_f64(&addr!("x")), Some(x_val));
    assert!(trace.get_f64(&addr!("y")).is_some());
    assert!(!trace.choices.contains_key(&addr!("z")));

    // log_prior == score("x" under model) + log_prob of the fresh "y" draw.
    let y_val = trace.get_f64(&addr!("y")).unwrap();
    let expected = Normal::new(0.0, 1.0).unwrap().log_prob(&x_val) + fresh_dist.log_prob(&y_val);
    assert!(
        (trace.log_prior - expected).abs() < 1e-9,
        "log_prior {} != expected {}",
        trace.log_prior,
        expected
    );
}

// ===========================================================================
// FG-47: duplicate sample addresses are detected instead of silently
// double-counting log_prior and dropping a choice.
// ===========================================================================

// Fast handler (PriorHandler) panics with a precise AddressConflict message.
#[test]
#[should_panic(expected = "AddressConflict")]
fn fg47_prior_handler_panics_on_duplicate_address() {
    let colliding = sample(addr!("dup"), Normal::new(0.0, 1.0).unwrap())
        .bind(|_| sample(addr!("dup"), Normal::new(0.0, 1.0).unwrap()));

    let mut rng = StdRng::seed_from_u64(47);
    run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        colliding,
    );
}

// Fallible strict path surfaces AddressConflict as a real error code.
#[test]
fn fg47_strict_scoring_reports_address_conflict_error_code() {
    let mut rng = StdRng::seed_from_u64(147);
    let (_, base) = run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        sample(addr!("dup"), Normal::new(0.0, 1.0).unwrap()),
    );

    let err = score_given_trace_strict(
        base,
        sample(addr!("dup"), Normal::new(0.0, 1.0).unwrap())
            .bind(|_| sample(addr!("dup"), Normal::new(0.0, 1.0).unwrap())),
    )
    .unwrap_err();
    assert_eq!(err.code(), ErrorCode::AddressConflict);
}

// Safe handler (SafeReplayHandler) invalidates the trace with -inf rather than
// silently double-counting on a duplicate address.
#[test]
fn fg47_safe_replay_invalidates_on_duplicate_address() {
    use fugue::runtime::interpreters::SafeReplayHandler;
    let mut rng = StdRng::seed_from_u64(247);
    let (_v, trace) = run(
        SafeReplayHandler {
            rng: &mut rng,
            base: Trace::default(),
            trace: Trace::default(),
            warn_on_mismatch: false,
        },
        sample(addr!("dup"), Normal::new(0.0, 1.0).unwrap())
            .bind(|_| sample(addr!("dup"), Normal::new(0.0, 1.0).unwrap())),
    );
    assert!(trace.log_prior.is_infinite() && trace.log_prior < 0.0);
}

// ===========================================================================
// FG-48: Score handlers store the FRESHLY computed per-choice logp, so the sum
// of the stored choice logps equals the trace's log_prior. Pre-fix, the score
// handlers cloned the base choice (stale logp under the original distribution).
// ===========================================================================
#[test]
fn fg48_scored_choice_logps_sum_to_log_prior() {
    // Build a base trace under one set of distributions.
    let mut rng = StdRng::seed_from_u64(48);
    let (_, base) = run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        sample(addr!("a"), Normal::new(0.0, 1.0).unwrap())
            .bind(|_| sample(addr!("b"), Normal::new(0.0, 1.0).unwrap()))
            .bind(|_| sample(addr!("c"), Poisson::new(2.0).unwrap())),
    );

    // Re-score under DIFFERENT distributions so stale base logps would not match.
    let (_v, scored) = run(
        ScoreGivenTrace {
            base: base.clone(),
            trace: Trace::default(),
        },
        sample(addr!("a"), Normal::new(1.5, 3.0).unwrap())
            .bind(|_| sample(addr!("b"), Normal::new(-0.5, 0.7).unwrap()))
            .bind(|_| sample(addr!("c"), Poisson::new(5.0).unwrap())),
    );

    let sum_logps: f64 = scored.choices.values().map(|c| c.logp).sum();
    assert!(
        (sum_logps - scored.log_prior).abs() < 1e-9,
        "sum of choice logps {sum_logps} != log_prior {}",
        scored.log_prior
    );

    // And the stored logps really are the RE-SCORED ones, not the base's.
    let base_sum: f64 = base.choices.values().map(|c| c.logp).sum();
    assert!(
        (sum_logps - base_sum).abs() > 1e-6,
        "re-scored logps should differ from the stale base logps"
    );
}

// ===========================================================================
// FG-54: ChoiceValue::I64 is a live value type with a full sample/replay/score
// path, tested via manually built traces. (A DiscreteUniform distribution lands
// in a later work package.)
// ===========================================================================
#[test]
fn fg54_i64_sample_replay_and_score_roundtrip() {
    let dist = DiscreteUniformI64 { lo: 0, hi: 9 };
    let expected_lp = -(10.0_f64).ln();

    // 1. PriorHandler records an I64 choice.
    let mut rng = StdRng::seed_from_u64(54);
    let (drawn, prior_trace) = run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        sample(addr!("k"), dist.clone()),
    );
    assert_eq!(prior_trace.get_i64(&addr!("k")), Some(drawn));
    assert!((prior_trace.log_prior - expected_lp).abs() < 1e-12);
    // The stored choice carries the fresh logp (matches log_prior).
    let stored = &prior_trace.choices[&addr!("k")];
    assert!(matches!(stored.value, ChoiceValue::I64(_)));
    assert!((stored.logp - expected_lp).abs() < 1e-12);

    // 2. Replay from a MANUALLY built base trace uses the stored i64 value.
    let mut base = Trace::default();
    base.insert_choice(addr!("k"), ChoiceValue::I64(3), 0.0);
    let (replayed, replay_trace) = run(
        ReplayHandler {
            rng: &mut rng,
            base: base.clone(),
            trace: Trace::default(),
        },
        sample(addr!("k"), dist.clone()),
    );
    assert_eq!(replayed, 3);
    assert_eq!(replay_trace.get_i64(&addr!("k")), Some(3));
    assert!((replay_trace.log_prior - expected_lp).abs() < 1e-12);

    // 3. Scoring a manually built i64 trace computes the fresh logp.
    let (scored_val, score_trace) = run(
        ScoreGivenTrace {
            base,
            trace: Trace::default(),
        },
        sample(addr!("k"), dist.clone()),
    );
    assert_eq!(scored_val, 3);
    assert!((score_trace.log_prior - expected_lp).abs() < 1e-12);
    assert!((score_trace.choices[&addr!("k")].logp - expected_lp).abs() < 1e-12);

    // 4. An out-of-support i64 observation contributes -inf as expected.
    let (_o, obs_trace) = run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        observe(addr!("obs"), dist, 42_i64),
    );
    assert!(obs_trace.log_likelihood.is_infinite() && obs_trace.log_likelihood < 0.0);
}

// ===========================================================================
// FG-26 / FG-52: `addr!` index encoding is collision-free through the public
// macro. Pre-fix, addr!("a", 1) and addr!("a#1") both produced "a#1".
// ===========================================================================
#[test]
fn fg26_fg52_addr_indexing_is_collision_free() {
    assert_ne!(addr!("a", 1), addr!("a#1"));
    assert_ne!(addr!("a", "b#3"), addr!("a#b", 3));

    // The collision would have corrupted a trace: two "distinct" sites sharing a
    // key double-count log_prior and drop a choice. With the fix, the two sites
    // are separate keys, so both survive.
    let mut rng = StdRng::seed_from_u64(26);
    let (_v, trace) = run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        sample(addr!("a", 1), Normal::new(0.0, 1.0).unwrap())
            .bind(|_| sample(addr!("a#1"), Normal::new(0.0, 1.0).unwrap())),
    );
    assert_eq!(trace.choices.len(), 2);
}

// ===========================================================================
// FG-61: `prob!` do-notation accepts irrefutable patterns on the left of `<-`.
// ===========================================================================
#[test]
fn fg61_prob_accepts_tuple_and_struct_patterns() {
    struct Pair {
        first: f64,
        second: f64,
    }

    let model = prob! {
        let (a, b) <- pure((1.0_f64, 2.0_f64));
        let Pair { first, second } <- pure(Pair { first: a, second: b });
        let s <- sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
        pure(first + second + s - s)
    };

    let mut rng = StdRng::seed_from_u64(61);
    let (val, _t) = run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        model,
    );
    assert!((val - 3.0).abs() < 1e-12);
}
