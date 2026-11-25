//! A standard error for SMC log evidence: `log_evidence_estimate` and
//! `adaptive_smc_replicated`.
//!
//! The arithmetic is checked on hand-computed cases, and the error bar is
//! checked for calibration on a conjugate model whose evidence is known in
//! closed form: across many independent estimates, the reported standard
//! error matches the actual spread and the standardized errors look like
//! standard normal draws.

use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

const DATA: [f64; 6] = [0.9, 1.4, 0.2, 1.7, 0.6, 1.1];

fn model() -> Model<f64> {
    sample(addr!("mu"), Normal::new(0.0, 2.0).unwrap()).bind(move |mu| {
        traverse_vec(DATA.iter().copied().enumerate().collect(), move |(i, y)| {
            observe(addr!("y", i), Normal::new(mu, 1.0).unwrap(), y)
        })
        .map(move |_| mu)
    })
}

fn exact() -> f64 {
    normal_normal_log_marginal(0.0, 2.0, 1.0, &DATA).unwrap()
}

#[test]
fn estimate_arithmetic() {
    // Z estimates 1 and 3: mean 2, sd sqrt(2), se = sqrt(2) / (2 sqrt(2)).
    let e = log_evidence_estimate(&[0.0, 3.0f64.ln()]).unwrap();
    assert!((e.log_evidence - 2.0f64.ln()).abs() < 1e-12);
    assert!((e.std_error - 0.5).abs() < 1e-12);
    assert_eq!(e.replicates, vec![0.0, 3.0f64.ln()]);

    // Z estimates 2, 4, 6, 8: mean 5, sample variance 20/3, se = sqrt(20/3/4)/5.
    let logs: Vec<f64> = [2.0f64, 4.0, 6.0, 8.0].iter().map(|z| z.ln()).collect();
    let e = log_evidence_estimate(&logs).unwrap();
    assert!((e.log_evidence - 5.0f64.ln()).abs() < 1e-12);
    assert!((e.std_error - (20.0f64 / 3.0 / 4.0).sqrt() / 5.0).abs() < 1e-12);

    // Identical replicates: no spread.
    let e = log_evidence_estimate(&[-3.5, -3.5, -3.5]).unwrap();
    assert!((e.log_evidence + 3.5).abs() < 1e-12);
    assert!(e.std_error < 1e-12);
}

/// Only ratios of evidences matter: a common factor shifts the estimate and
/// leaves the standard error alone, at any magnitude.
#[test]
fn estimate_is_shift_invariant_and_stable_at_extreme_magnitudes() {
    let base = [-1.2, -0.7, -1.9, -1.0, -1.4];
    let reference = log_evidence_estimate(&base).unwrap();
    for shift in [-1e6, -5000.0, -700.0, 0.0, 700.0, 5000.0, 1e6] {
        let shifted: Vec<f64> = base.iter().map(|l| l + shift).collect();
        let e = log_evidence_estimate(&shifted).unwrap();
        assert!(e.log_evidence.is_finite());
        let tol = 1e-9 * (1.0 + shift.abs() * 1e-6);
        assert!((e.log_evidence - (reference.log_evidence + shift)).abs() < tol);
        assert!((e.std_error - reference.std_error).abs() < 1e-6);
    }
    // The order of the replicates is irrelevant.
    let mut reversed = base;
    reversed.reverse();
    let e = log_evidence_estimate(&reversed).unwrap();
    assert!((e.log_evidence - reference.log_evidence).abs() < 1e-12);
    assert!((e.std_error - reference.std_error).abs() < 1e-12);
}

#[test]
fn estimate_edge_cases() {
    assert_eq!(
        log_evidence_estimate(&[]).unwrap_err().code(),
        ErrorCode::InvalidCount
    );
    assert_eq!(
        log_evidence_estimate(&[-1.0]).unwrap_err().code(),
        ErrorCode::InvalidCount
    );
    // Zero evidence everywhere: the estimate is -inf and has no error bar.
    let none = log_evidence_estimate(&[f64::NEG_INFINITY, f64::NEG_INFINITY]).unwrap();
    assert_eq!(none.log_evidence, f64::NEG_INFINITY);
    assert!(none.std_error.is_nan());
    // Z estimates 1 and 0: mean 1/2, sd sqrt(1/2), se = sqrt(1/2) / (sqrt(2)/2) = 1.
    let half = log_evidence_estimate(&[0.0, f64::NEG_INFINITY]).unwrap();
    assert!((half.log_evidence + 2.0f64.ln()).abs() < 1e-12);
    assert!((half.std_error - 1.0).abs() < 1e-12);
    // NaN counts as zero evidence.
    let nan = log_evidence_estimate(&[0.0, f64::NAN]).unwrap();
    assert_eq!(nan.replicates[1], f64::NEG_INFINITY);
    assert!((nan.log_evidence - half.log_evidence).abs() < 1e-12);
}

#[test]
fn replicated_run_pools_the_populations() {
    let (n, r) = (300, 6);
    let mut rng = StdRng::seed_from_u64(21);
    let (pooled, evidence) =
        adaptive_smc_replicated(&mut rng, n, r, model, &SMCConfig::default()).unwrap();

    assert_eq!(pooled.particles.len(), n * r);
    assert_eq!(evidence.replicates.len(), r);
    assert_eq!(pooled.log_evidence, evidence.log_evidence);
    assert_eq!(
        log_evidence_estimate(&evidence.replicates).unwrap(),
        evidence
    );

    // Weights are normalized over the pool, and each run's share is its
    // share of the summed evidence.
    let total: f64 = pooled.particles.iter().map(|p| p.weight).sum();
    assert!((total - 1.0).abs() < 1e-9);
    let z_total: f64 = evidence.replicates.iter().map(|l| l.exp()).sum();
    for (run, log_z) in pooled.particles.chunks(n).zip(&evidence.replicates) {
        let share: f64 = run.iter().map(|p| p.weight).sum();
        assert!((share - log_z.exp() / z_total).abs() < 1e-9);
        for p in run {
            assert!((p.weight - p.log_weight.exp()).abs() < 1e-15);
        }
    }

    // The pool is a posterior approximation.
    let (mu_n, sigma_n) = normal_normal_posterior(0.0, 2.0, 1.0, &DATA).unwrap();
    let mean: f64 = pooled
        .particles
        .iter()
        .map(|p| p.weight * p.trace.get_f64(&addr!("mu")).unwrap())
        .sum();
    assert!((mean - mu_n).abs() < 4.0 * sigma_n / (effective_sample_size(&pooled)).sqrt() + 0.02);

    // Same seed, same answer; the runs are the plain `adaptive_smc` runs in order.
    let mut rng = StdRng::seed_from_u64(21);
    let (_, again) = adaptive_smc_replicated(&mut rng, n, r, model, &SMCConfig::default()).unwrap();
    assert_eq!(again, evidence);
    let mut rng = StdRng::seed_from_u64(21);
    for log_z in &evidence.replicates {
        let single = adaptive_smc(&mut rng, n, model, SMCConfig::default());
        assert_eq!(single.log_evidence, *log_z);
    }
}

#[test]
fn replicated_run_rejects_degenerate_requests() {
    let mut rng = StdRng::seed_from_u64(1);
    let config = SMCConfig::default();
    for replicates in [0, 1] {
        let err = adaptive_smc_replicated(&mut rng, 10, replicates, model, &config).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidCount);
    }
    let err = adaptive_smc_replicated(&mut rng, 0, 4, model, &config).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidCount);

    // A model with zero evidence: -inf, no error bar, a uniform pool.
    let impossible = || model().bind(|mu| factor(f64::NEG_INFINITY).map(move |_| mu));
    let (pooled, evidence) = adaptive_smc_replicated(&mut rng, 5, 3, impossible, &config).unwrap();
    assert_eq!(evidence.log_evidence, f64::NEG_INFINITY);
    assert!(evidence.std_error.is_nan());
    for p in &pooled.particles {
        assert!((p.weight - 1.0 / 15.0).abs() < 1e-15);
    }
}

/// Calibration against the closed form, for the plain importance-sampling
/// path and for the tempered path with resampling and rejuvenation.
#[test]
fn standard_error_is_calibrated() {
    let truth = exact();
    for (label, config_of, particles) in [
        (
            "importance sampling",
            (|| SMCConfig::default()) as fn() -> SMCConfig,
            150,
        ),
        (
            "tempered with rejuvenation",
            || SMCConfig {
                rejuvenation_steps: 2,
                ..Default::default()
            },
            100,
        ),
    ] {
        let trials = 60;
        let replicates = 10;
        let (mut estimates, mut errors, mut z_scores) = (Vec::new(), Vec::new(), Vec::new());
        for trial in 0..trials {
            let mut rng = StdRng::seed_from_u64(5000 + trial);
            let (_, e) =
                adaptive_smc_replicated(&mut rng, particles, replicates, model, &config_of())
                    .unwrap();
            assert!(
                e.std_error > 0.0 && e.std_error < 0.5,
                "{label}: se {}",
                e.std_error
            );
            z_scores.push((e.log_evidence - truth) / e.std_error);
            estimates.push(e.log_evidence);
            errors.push(e.std_error);
        }
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let spread = {
            let m = mean(&estimates);
            (estimates.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (trials - 1) as f64).sqrt()
        };

        // The reported error is the size of the actual spread of the estimate ...
        let ratio = mean(&errors) / spread;
        assert!(
            (0.6..1.6).contains(&ratio),
            "{label}: mean reported se {} vs empirical sd {spread} (ratio {ratio})",
            mean(&errors)
        );
        // ... the estimate is centred on the closed form ...
        assert!(
            (mean(&estimates) - truth).abs()
                < 4.0 * spread / (trials as f64).sqrt() + 0.5 * spread * spread,
            "{label}: mean estimate {} vs exact {truth}",
            mean(&estimates)
        );
        // ... and the standardized errors behave like unit-variance draws:
        // mean square near 1 (t-like tails with 9 d.o.f. push it above), and
        // the usual coverage.
        let mean_sq = mean(&z_scores.iter().map(|z| z * z).collect::<Vec<_>>());
        assert!(
            (0.5..2.5).contains(&mean_sq),
            "{label}: mean squared z-score {mean_sq}"
        );
        let within_two = z_scores.iter().filter(|z| z.abs() < 2.0).count() as f64 / trials as f64;
        let within_three = z_scores.iter().filter(|z| z.abs() < 3.0).count() as f64 / trials as f64;
        assert!(within_two >= 0.85, "{label}: {within_two} within 2 se");
        assert!(within_three >= 0.95, "{label}: {within_three} within 3 se");
    }
}
