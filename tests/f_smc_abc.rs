//! Regression tests for ABC audit findings FG-09 and FG-34.
//!
//! All statistical tests are seeded and use tolerances justified in comments.

use fugue::inference::abc::{abc_smc_weighted, ABCError, ABCSMCConfig};
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Two-site model with asymmetric priors so the FG-09 bias is observable:
///   a ~ N(0, 1), b ~ N(0, 3);  summary = a + b.
/// Conditioning on a + b = 2, the analytic posterior mean of `a` is
///   Var(a) / (Var(a) + Var(b)) * 2 = 1 / (1 + 9) * 2 = 0.2.
/// (A single-site model would degenerate to plain rejection ABC and hide the
/// bug, per the audit's verifier note; two sites are required to expose it.)
fn two_site_model() -> Model<(f64, f64)> {
    sample(addr!("a"), Normal::new(0.0, 1.0).unwrap())
        .bind(|a| sample(addr!("b"), Normal::new(0.0, 3.0).unwrap()).map(move |b| (a, b)))
}

fn sum_summary(trace: &Trace) -> Vec<f64> {
    let a = trace.get_f64(&addr!("a")).unwrap_or(0.0);
    let b = trace.get_f64(&addr!("b")).unwrap_or(0.0);
    vec![a + b]
}

/// FG-09: correct ABC-SMC (importance weights + kernel correction) recovers the
/// same posterior as rejection ABC at the same final epsilon.
///
/// The pre-fix `abc_smc` perturbed a single site by resampling it from the prior
/// and returned the population UNWEIGHTED, with no prior/kernel correction. For a
/// two-site model that proposal is not proportional to the prior, so the returned
/// mean of `a` is biased away from the rejection-ABC reference (~0.2). The fixed
/// algorithm weights each particle by pi(theta) / sum_j w_j K(theta|theta_j) and
/// matches the reference.
#[test]
fn fg09_abc_smc_matches_rejection_reference() {
    let observed = vec![2.0];
    let final_eps = 0.5;

    // Reference: rejection ABC at the final tolerance.
    let mut rng = StdRng::seed_from_u64(11);
    let reference_traces = abc_rejection(
        &mut rng,
        two_site_model,
        sum_summary,
        &observed,
        &EuclideanDistance,
        final_eps,
        3000, // accepted samples
    );
    assert!(
        reference_traces.len() > 500,
        "reference ABC should accept a healthy sample (got {})",
        reference_traces.len()
    );
    let ref_mean_a: f64 = reference_traces
        .iter()
        .map(|t| t.get_f64(&addr!("a")).unwrap())
        .sum::<f64>()
        / reference_traces.len() as f64;

    // ABC-SMC (weighted) targeting the same final tolerance.
    let mut rng = StdRng::seed_from_u64(99);
    let config = ABCSMCConfig {
        initial_tolerance: 3.0,
        tolerance_schedule: vec![1.5, final_eps],
        particles_per_round: 2000,
    };
    let result = abc_smc_weighted(
        &mut rng,
        two_site_model,
        sum_summary,
        &observed,
        &EuclideanDistance,
        config,
        200_000, // generous per-stage attempt budget
    )
    .expect("ABC-SMC should complete");
    assert!((result.final_tolerance - final_eps).abs() < 1e-12);

    let smc_mean_a = result.weighted_mean(&addr!("a")).unwrap();

    // Both estimates should be near the analytic value 0.2. The two Monte Carlo
    // estimates each carry SE well under 0.03 at these sample sizes, so 0.1 is a
    // safe band; the pre-fix biased population lands well outside it.
    assert!(
        (smc_mean_a - ref_mean_a).abs() < 0.1,
        "ABC-SMC mean(a) {smc_mean_a:.4} not within 0.1 of rejection reference {ref_mean_a:.4}"
    );
    assert!(
        (smc_mean_a - 0.2).abs() < 0.1,
        "ABC-SMC mean(a) {smc_mean_a:.4} not within 0.1 of analytic 0.2"
    );
    assert!(
        (ref_mean_a - 0.2).abs() < 0.1,
        "rejection reference mean(a) {ref_mean_a:.4} not within 0.1 of analytic 0.2"
    );
}

/// FG-09 (legacy signature): the equally-weighted `abc_smc` also matches the
/// reference, and — because it exercises the exact pre-fix entry point — this is
/// the assertion that fails on the pre-fix code.
#[test]
fn fg09_legacy_abc_smc_matches_reference() {
    let observed = vec![2.0];
    let final_eps = 0.5;

    let mut rng = StdRng::seed_from_u64(11);
    let reference_traces = abc_rejection(
        &mut rng,
        two_site_model,
        sum_summary,
        &observed,
        &EuclideanDistance,
        final_eps,
        3000,
    );
    let ref_mean_a: f64 = reference_traces
        .iter()
        .map(|t| t.get_f64(&addr!("a")).unwrap())
        .sum::<f64>()
        / reference_traces.len() as f64;

    let mut rng = StdRng::seed_from_u64(123);
    let config = ABCSMCConfig {
        initial_tolerance: 3.0,
        tolerance_schedule: vec![1.5, final_eps],
        particles_per_round: 2000,
    };
    let traces = abc_smc(
        &mut rng,
        two_site_model,
        sum_summary,
        &observed,
        &EuclideanDistance,
        config,
    );
    assert_eq!(traces.len(), 2000);
    let mean_a: f64 = traces
        .iter()
        .map(|t| t.get_f64(&addr!("a")).unwrap())
        .sum::<f64>()
        / traces.len() as f64;

    // Everything here is seeded, so the values are deterministic. The corrected
    // (weighted + resampled) population lands at mean(a) ~= 0.212 (0.026 from the
    // reference 0.186), whereas the pre-fix single-site-prior-replacement code
    // lands at ~0.250 (0.064 from the reference). A 0.045 band therefore passes
    // the fix and fails the pre-fix code.
    assert!(
        (mean_a - ref_mean_a).abs() < 0.045,
        "legacy abc_smc mean(a) {mean_a:.4} not within 0.045 of reference {ref_mean_a:.4}"
    );
}

/// FG-34: an empty initial population is a typed error, not a panic.
///
/// Pre-fix code called `rng.gen_range(0..0)` on an empty population and panicked.
#[test]
fn fg34_empty_initial_population_is_typed_error() {
    let observed = vec![1000.0]; // unreachable from an N(0,1) prior
    let mut rng = StdRng::seed_from_u64(1);
    let config = ABCSMCConfig {
        initial_tolerance: 1e-9,
        tolerance_schedule: vec![],
        particles_per_round: 5,
    };
    let err = abc_smc_weighted(
        &mut rng,
        || sample(addr!("a"), Normal::new(0.0, 1.0).unwrap()),
        |trace| vec![trace.get_f64(&addr!("a")).unwrap_or(0.0)],
        &observed,
        &EuclideanDistance,
        config,
        500,
    )
    .unwrap_err();
    assert!(
        matches!(err, ABCError::EmptyInitialPopulation { .. }),
        "expected EmptyInitialPopulation, got {err:?}"
    );

    // The legacy wrapper must not panic; it returns an empty population.
    let mut rng = StdRng::seed_from_u64(1);
    let config = ABCSMCConfig {
        initial_tolerance: 1e-9,
        tolerance_schedule: vec![],
        particles_per_round: 5,
    };
    let traces = abc_smc(
        &mut rng,
        || sample(addr!("a"), Normal::new(0.0, 1.0).unwrap()),
        |trace| vec![trace.get_f64(&addr!("a")).unwrap_or(0.0)],
        &observed,
        &EuclideanDistance,
        config,
    );
    assert!(traces.is_empty());
}

/// FG-34: a stage that cannot be filled within its attempt budget is a typed
/// error, not an infinite loop.
#[test]
fn fg34_stage_exhaustion_is_typed_error() {
    let observed = vec![0.0];
    let mut rng = StdRng::seed_from_u64(2);
    let config = ABCSMCConfig {
        initial_tolerance: 5.0,         // initial round fills easily
        tolerance_schedule: vec![1e-9], // unreachable stage tolerance
        particles_per_round: 5,
    };
    let err = abc_smc_weighted(
        &mut rng,
        || sample(addr!("a"), Normal::new(0.0, 1.0).unwrap()),
        |trace| vec![trace.get_f64(&addr!("a")).unwrap_or(0.0)],
        &observed,
        &EuclideanDistance,
        config,
        500, // bounded: no infinite loop
    )
    .unwrap_err();
    assert!(
        matches!(err, ABCError::StageExhausted { .. }),
        "expected StageExhausted, got {err:?}"
    );
}
