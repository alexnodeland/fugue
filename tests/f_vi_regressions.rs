//! Regression tests for the variational-inference audit findings
//! (FG-04, FG-16, FG-17, FG-18, FG-44, FG-45, FG-46, FG-60).
//!
//! Every statistical assertion is seeded (`StdRng::seed_from_u64`) and its tolerance is
//! justified in a comment. Reference values for the conjugate model are derived from the
//! closed-form Normal-Normal posterior (see individual tests).

use fugue::inference::vi::{
    elbo_gradient_fd, estimate_elbo, optimize_meanfield_vi_with_config, GuideError, MeanFieldGuide,
    ParamCoord, Support, VIConfig, VariationalParam,
};
// `elbo_with_guide`, `optimize_meanfield_vi`, `MeanFieldGuide`, `VariationalParam` are also
// re-exported at the crate root via `fugue::*`.
use fugue::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Conjugate Normal-Normal model used by several tests.
///
/// Prior:  mu ~ Normal(0, 1)          (mu0 = 0, prior sd = 1  => prior precision tau0 = 1)
/// Likelihood: y_i ~ Normal(mu, 1)     (likelihood precision = 1 per observation)
/// Data: y = [2.0, 3.0, 1.5, 2.5]      (n = 4, sum = 9.0)
///
/// Closed-form posterior (Normal-Normal conjugacy):
///   posterior precision  = tau0 + n/sigma_lik^2 = 1 + 4 = 5
///   posterior variance   = 1/5 = 0.2   => posterior sd = 1/sqrt(5) = 0.4472135954999579
///   posterior mean       = (tau0*mu0 + sum(y)/sigma_lik^2) / precision = (0 + 9)/5 = 1.8
fn conjugate_model() -> impl Fn() -> Model<f64> {
    || {
        sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap()).bind(|mu| {
            observe(addr!("y1"), Normal::new(mu, 1.0).unwrap(), 2.0)
                .bind(move |_| observe(addr!("y2"), Normal::new(mu, 1.0).unwrap(), 3.0))
                .bind(move |_| observe(addr!("y3"), Normal::new(mu, 1.0).unwrap(), 1.5))
                .bind(move |_| observe(addr!("y4"), Normal::new(mu, 1.0).unwrap(), 2.5))
                .map(move |_| mu)
        })
    }
}

const POSTERIOR_MEAN: f64 = 1.8;
const POSTERIOR_SD: f64 = 0.447_213_595_499_957_9; // 1/sqrt(5)

/// FG-45 (the proof test) + FG-04: optimizing BOTH location and scale recovers the
/// analytically-known conjugate posterior. This test is impossible to pass without the
/// FG-04 scale fix, because the posterior sd (0.447) differs sharply from any fixed init.
#[test]
fn fg45_vi_recovers_conjugate_posterior_mean_and_scale() {
    let model_fn = conjugate_model();

    // Start deliberately far from the optimum in BOTH coordinates: mu = 0 (truth 1.8) and
    // sigma = 1.0 (truth 0.447). If scale were frozen (the pre-fix bug), sigma could never
    // leave 1.0 and the 25% band below would be unreachable.
    let mut guide = MeanFieldGuide::new();
    guide.params.insert(
        addr!("mu"),
        VariationalParam::Normal {
            mu: 0.0,
            log_sigma: 0.0, // sigma = 1.0
        },
    );

    let config = VIConfig {
        n_iterations: 4000,
        n_samples_per_iter: 16,
        base_learning_rate: 0.05,
        fd_eps: 0.02,
        convergence_tol: 1e-9, // effectively disable early stop; run the full budget
        convergence_window: 50,
        step_decay_exponent: 0.6,
    };

    let mut rng = StdRng::seed_from_u64(2026);
    let result = optimize_meanfield_vi_with_config(&mut rng, &model_fn, guide, &config);

    let (mu, sigma) = match result.guide.params.get(&addr!("mu")).unwrap() {
        VariationalParam::Normal { mu, log_sigma } => (*mu, log_sigma.exp()),
        _ => panic!("expected Normal factor"),
    };

    // Mean within 0.05 (generous, honest: SGD residual noise is well below this).
    assert!(
        (mu - POSTERIOR_MEAN).abs() < 0.05,
        "posterior mean not recovered: got mu={mu}, want {POSTERIOR_MEAN}"
    );
    // Scale within 25% of the analytic posterior sd. Passing this REQUIRES the FG-04 scale
    // optimization: the init sigma of 1.0 is +124% away from 0.447.
    let rel_err = (sigma - POSTERIOR_SD).abs() / POSTERIOR_SD;
    assert!(
        rel_err < 0.25,
        "posterior sd not recovered: got sigma={sigma}, want {POSTERIOR_SD} (rel err {rel_err})"
    );
}

/// FG-04: the scale parameter (log_sigma) is actually moved by the optimizer, not frozen
/// at its initial value. Pre-fix the Normal arm bound `log_sigma: _` and never touched it.
#[test]
fn fg04_scale_parameter_is_optimized_not_frozen() {
    let model_fn = conjugate_model();

    let init_log_sigma = 0.0_f64; // sigma = 1.0
    let mut guide = MeanFieldGuide::new();
    guide.params.insert(
        addr!("mu"),
        VariationalParam::Normal {
            mu: 0.0,
            log_sigma: init_log_sigma,
        },
    );

    let config = VIConfig {
        n_iterations: 1500,
        n_samples_per_iter: 16,
        base_learning_rate: 0.05,
        fd_eps: 0.02,
        convergence_tol: 1e-9,
        convergence_window: 50,
        step_decay_exponent: 0.6,
    };

    let mut rng = StdRng::seed_from_u64(7);
    let result = optimize_meanfield_vi_with_config(&mut rng, &model_fn, guide, &config);

    let final_log_sigma = match result.guide.params.get(&addr!("mu")).unwrap() {
        VariationalParam::Normal { log_sigma, .. } => *log_sigma,
        _ => panic!("expected Normal factor"),
    };

    // The scale must have moved substantially away from its init (toward ln(0.447) = -0.80).
    assert!(
        (final_log_sigma - init_log_sigma).abs() > 0.2,
        "log_sigma was effectively frozen: init={init_log_sigma}, final={final_log_sigma}"
    );
    // ...and moved in the correct (downward) direction, toward the tighter posterior.
    assert!(
        final_log_sigma < init_log_sigma,
        "log_sigma should decrease toward the tighter posterior; got {final_log_sigma}"
    );
}

/// FG-16: the common-random-numbers (CRN) central finite-difference gradient has the
/// correct SIGN. On the 1-D Gaussian target `mu ~ Normal(0,1)`, `y ~ Normal(mu,1)`,
/// `y = 2`, the exact posterior mean is mu* = 1.0 and the ELBO gradient wrt the guide mean
/// m is `-2*(m - 1)`, so its sign is `sign(1 - m)`.
///
/// We assert the CRN estimator matches this sign in >= 95% of seeded trials, while the
/// pre-fix estimator (independent draws for the base/perturbed ELBO, mismatched sample
/// counts, forward difference divided by eps -> ~100x noise amplification) is barely
/// better than a coin flip. This directly exhibits the defect and the fix.
#[test]
fn fg16_crn_gradient_sign_matches_analytic() {
    // 1-D quadratic (Gaussian) target; posterior mean mu* = (0 + 2)/(1 + 1) = 1.0.
    let model_fn = || {
        sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap())
            .bind(|mu| observe(addr!("y"), Normal::new(mu, 1.0).unwrap(), 2.0).map(move |_| mu))
    };
    const MU_STAR: f64 = 1.0;

    let mut seed_rng = StdRng::seed_from_u64(123);
    let n_trials = 300;
    let (mut crn_matches, mut naive_matches, mut counted) = (0usize, 0usize, 0usize);
    for _ in 0..n_trials {
        // Random guide mean in [-3, 5], skipping a band around the optimum where the true
        // gradient is ~0 and its sign is genuinely ambiguous for any finite estimator.
        let m: f64 = -3.0 + 8.0 * seed_rng.gen::<f64>();
        if (m - MU_STAR).abs() < 0.25 {
            continue;
        }
        let mut guide = MeanFieldGuide::new();
        guide.params.insert(
            addr!("mu"),
            VariationalParam::Normal {
                mu: m,
                log_sigma: 0.0,
            },
        );
        let analytic_sign = (MU_STAR - m).signum();

        // Fixed CRN estimator: identical seeded draws for +eps and -eps, matched counts.
        let grad_seed: u64 = seed_rng.gen();
        let crn = elbo_gradient_fd(
            grad_seed,
            model_fn,
            &guide,
            &addr!("mu"),
            ParamCoord::Location,
            0.02,
            16,
        );

        // Pre-fix-style estimator: independent draws off one advancing RNG, base uses 3
        // samples, the +eps evaluation uses 10 (mismatched), forward difference / eps.
        let eps = 0.01;
        let mut nrng = StdRng::seed_from_u64(seed_rng.gen());
        let base = elbo_with_guide(&mut nrng, model_fn, &guide, 3);
        let mut guide_plus = guide.clone();
        guide_plus.params.insert(
            addr!("mu"),
            VariationalParam::Normal {
                mu: m + eps,
                log_sigma: 0.0,
            },
        );
        let plus = elbo_with_guide(&mut nrng, model_fn, &guide_plus, 10);
        let naive = (plus - base) / eps;

        counted += 1;
        if crn.signum() == analytic_sign {
            crn_matches += 1;
        }
        if naive.signum() == analytic_sign {
            naive_matches += 1;
        }
    }

    let crn_frac = crn_matches as f64 / counted as f64;
    let naive_frac = naive_matches as f64 / counted as f64;
    // Fixed estimator: correct sign essentially always (worst observed across seeds ~0.98).
    assert!(
        crn_frac >= 0.95,
        "CRN gradient sign matched only {crn_frac} of trials (want >= 0.95)"
    );
    // Pre-fix estimator: noise-dominated, ~0.54 (near chance). 0.75 cleanly separates them.
    assert!(
        naive_frac < 0.75,
        "pre-fix independent-draw estimator unexpectedly accurate ({naive_frac}); \
         the CRN contrast is the point of this test"
    );
    assert!(
        crn_frac > naive_frac + 0.2,
        "CRN ({crn_frac}) should dominate the pre-fix estimator ({naive_frac})"
    );
}

/// FG-17: discrete latents produce a typed error at guide construction (never a panic,
/// never a silent F64 factor).
#[test]
fn fg17_discrete_latent_is_typed_error() {
    use fugue::runtime::trace::{Choice, ChoiceValue, Trace};

    let mut base = Trace::default();
    base.choices.insert(
        addr!("flag"),
        Choice {
            addr: addr!("flag"),
            value: ChoiceValue::Bool(true),
            logp: -0.7,
        },
    );

    let err = MeanFieldGuide::from_trace(&base).unwrap_err();
    match err {
        GuideError::UnsupportedDiscreteLatent { addr, value_type } => {
            assert_eq!(addr, addr!("flag"));
            assert_eq!(value_type, "bool");
        }
    }

    // Same for a u64 count latent.
    let mut base2 = Trace::default();
    base2.choices.insert(
        addr!("n"),
        Choice {
            addr: addr!("n"),
            value: ChoiceValue::U64(3),
            logp: -0.5,
        },
    );
    assert!(matches!(
        MeanFieldGuide::from_trace(&base2),
        Err(GuideError::UnsupportedDiscreteLatent { .. })
    ));
}

/// FG-17: a positive-support latent gets a LogNormal factor whose samples are always in
/// support, giving a finite ELBO -- whereas a support-mismatched Normal guide proposes
/// negative values and drives the ELBO to -inf.
#[test]
fn fg17_positive_support_guide_matches_and_is_finite() {
    // lambda ~ Gamma(2, 1) on (0, inf). The latent feeds an observation *mean* (any real
    // is a valid mean), so an out-of-support draw is caught purely as a -inf prior
    // log-density at the "lambda" site -- the FG-17 mechanism -- rather than a downstream
    // distribution-constructor panic.
    let model_fn = || {
        sample(addr!("lambda"), Gamma::new(2.0, 1.0).unwrap())
            .bind(|lam| observe(addr!("y"), Normal::new(lam, 1.0).unwrap(), 2.5).map(move |_| lam))
    };

    let mut guide = MeanFieldGuide::new();
    guide.add_latent(addr!("lambda"), Support::Positive, 2.0); // LogNormal factor
    assert!(matches!(
        guide.params.get(&addr!("lambda")),
        Some(VariationalParam::LogNormal { .. })
    ));

    let mut rng = StdRng::seed_from_u64(1);
    let elbo_matched = elbo_with_guide(&mut rng, model_fn, &guide, 32);
    assert!(
        elbo_matched.is_finite(),
        "support-matched LogNormal guide should give a finite ELBO, got {elbo_matched}"
    );

    // A Normal guide on a strictly-positive latent proposes negatives -> Gamma log_prob
    // -inf -> ELBO -inf. This is the pre-fix behavior FG-17 describes.
    let mut bad_guide = MeanFieldGuide::new();
    bad_guide.params.insert(
        addr!("lambda"),
        VariationalParam::Normal {
            mu: 0.0,
            log_sigma: 0.0, // sigma = 1 -> ~50% of draws are negative
        },
    );
    let mut rng2 = StdRng::seed_from_u64(2);
    let elbo_bad = elbo_with_guide(&mut rng2, model_fn, &bad_guide, 32);
    assert!(
        !elbo_bad.is_finite(),
        "support-mismatched Normal guide should collapse the ELBO to -inf, got {elbo_bad}"
    );
}

/// FG-17: a unit-interval latent gets a Beta factor (finite ELBO), while a Normal guide
/// proposes values outside [0,1] and collapses the ELBO.
#[test]
fn fg17_unit_support_guide_matches_and_is_finite() {
    // theta ~ Beta(2, 2) on (0, 1). The latent feeds an observation *mean*, so an
    // out-of-support draw shows up as a -inf prior log-density at the "theta" site (the
    // FG-17 mechanism), not a downstream constructor panic.
    let model_fn = || {
        sample(addr!("theta"), Beta::new(2.0, 2.0).unwrap())
            .bind(|t| observe(addr!("y"), Normal::new(t, 0.5).unwrap(), 0.5).map(move |_| t))
    };

    let mut guide = MeanFieldGuide::new();
    guide.add_latent(addr!("theta"), Support::Unit, 0.5); // Beta factor
    assert!(matches!(
        guide.params.get(&addr!("theta")),
        Some(VariationalParam::Beta { .. })
    ));

    let mut rng = StdRng::seed_from_u64(3);
    let elbo_matched = elbo_with_guide(&mut rng, model_fn, &guide, 32);
    assert!(
        elbo_matched.is_finite(),
        "support-matched Beta guide should give a finite ELBO, got {elbo_matched}"
    );

    let mut bad_guide = MeanFieldGuide::new();
    bad_guide.params.insert(
        addr!("theta"),
        VariationalParam::Normal {
            mu: 0.5,
            log_sigma: 0.0, // sigma = 1 -> many draws fall outside (0,1)
        },
    );
    let mut rng2 = StdRng::seed_from_u64(4);
    let elbo_bad = elbo_with_guide(&mut rng2, model_fn, &bad_guide, 32);
    assert!(
        !elbo_bad.is_finite(),
        "support-mismatched Normal guide should collapse the ELBO to -inf, got {elbo_bad}"
    );
}

/// FG-18: `from_trace` never yields log_sigma = ln(0) = -inf, and is NaN-proof for a
/// value of exactly 0.0. Pre-fix the positive branch used `0.0_f64.ln()` (= -inf), whose
/// sigma = 0 made sampling produce NaN / panic.
#[test]
fn fg18_from_trace_scale_is_finite_and_nan_proof() {
    use fugue::runtime::trace::{Choice, ChoiceValue, Trace};

    let mut base = Trace::default();
    for (name, v) in [("zero", 0.0), ("pos", 4.0), ("neg", -2.5)] {
        base.choices.insert(
            addr!(name),
            Choice {
                addr: addr!(name),
                value: ChoiceValue::F64(v),
                logp: -0.1,
            },
        );
    }

    let guide = MeanFieldGuide::from_trace(&base).expect("all-continuous trace should build");
    for name in ["zero", "pos", "neg"] {
        match guide.params.get(&addr!(name)).unwrap() {
            VariationalParam::Normal { mu, log_sigma } => {
                assert!(mu.is_finite());
                assert!(
                    log_sigma.is_finite(),
                    "log_sigma must be finite for '{name}'"
                );
                let sigma = log_sigma.exp();
                assert!(sigma > 0.0, "sigma must be strictly positive for '{name}'");
            }
            _ => panic!("expected Normal factor for '{name}'"),
        }
    }

    // Sampling must not produce NaN (a degenerate sigma = 0 would).
    let t = guide.sample_trace(&mut StdRng::seed_from_u64(99));
    assert!(t.log_prior.is_finite());
    for choice in t.choices.values() {
        assert!(choice.value.as_f64().unwrap().is_finite());
    }
}

/// FG-44: convergence detection fires (before the full iteration budget) once the ELBO
/// plateaus under the decaying step size, and the ELBO trends upward over optimization.
#[test]
fn fg44_convergence_detection_and_elbo_improves() {
    let model_fn = conjugate_model();

    let mut guide = MeanFieldGuide::new();
    guide.params.insert(
        addr!("mu"),
        VariationalParam::Normal {
            mu: 0.0,
            log_sigma: 0.0,
        },
    );

    let config = VIConfig {
        n_iterations: 6000,
        n_samples_per_iter: 16,
        base_learning_rate: 0.05,
        fd_eps: 0.02,
        convergence_tol: 1e-3, // loose enough that the plateau is detected
        convergence_window: 50,
        step_decay_exponent: 0.6,
    };

    let mut rng = StdRng::seed_from_u64(555);
    let result = optimize_meanfield_vi_with_config(&mut rng, &model_fn, guide, &config);

    assert!(
        result.converged,
        "ELBO-plateau convergence should fire on this well-behaved model"
    );
    assert!(
        result.iterations < config.n_iterations,
        "convergence should stop before the full budget ({} iters)",
        config.n_iterations
    );

    // The ELBO trends upward: mean of the last window exceeds the mean of the first.
    let h = &result.elbo_history;
    let w = 50.min(h.len() / 2).max(1);
    let first: f64 = h[..w].iter().sum::<f64>() / w as f64;
    let last: f64 = h[h.len() - w..].iter().sum::<f64>() / w as f64;
    assert!(
        last > first,
        "ELBO should improve over optimization: first-window mean {first}, last-window mean {last}"
    );
}

/// FG-46: `estimate_elbo` returns the ELBO with q = prior = E_prior[log p(x | z)], not the
/// mislabeled joint E_prior[log p(x, z)].
///
/// Model: mu ~ Normal(0,1), observe y ~ Normal(mu, 1), y = 1.0.
///   E_prior[log p(y|mu)] = -0.5*ln(2pi) - 0.5*E[(1-mu)^2]
///                        = -0.5*ln(2pi) - 0.5*(Var(mu) + (1-E[mu])^2)
///                        = -0.5*ln(2pi) - 0.5*(1 + 1) = -0.5*ln(2pi) - 1.0
/// Reference (scipy): -0.5*np.log(2*np.pi) - 1.0 = -1.9189385332046727
/// The OLD (buggy) joint value would be that minus the prior entropy term, ~ -3.3378771.
#[test]
fn fg46_estimate_elbo_is_prior_elbo_not_joint() {
    let model_fn = || {
        sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap())
            .bind(|mu| observe(addr!("y"), Normal::new(mu, 1.0).unwrap(), 1.0).map(move |_| mu))
    };

    // Reference: E_prior[log p(y|mu)] = -0.5*ln(2pi) - 1.0.
    const REFERENCE: f64 = -1.918_938_533_204_672_7;
    // The old, double-counted joint value (kept only to assert we are NOT returning it).
    const OLD_JOINT: f64 = -3.337_877_066_409_345; // REFERENCE - 1.4189385 (prior cross-entropy)

    let mut rng = StdRng::seed_from_u64(2024);
    // 20000 samples: Var(log p(y|mu)) = 1.5 => SE ~ 1.22/sqrt(20000) ~ 0.0087, so 0.05 is safe.
    let elbo = estimate_elbo(&mut rng, model_fn, 20_000);

    assert!(
        (elbo - REFERENCE).abs() < 0.05,
        "estimate_elbo should equal the prior-ELBO {REFERENCE}, got {elbo}"
    );
    // And it must be clearly distinct from the old joint value (off by the prior entropy).
    assert!(
        (elbo - OLD_JOINT).abs() > 1.0,
        "estimate_elbo must no longer return the double-counted joint {OLD_JOINT}, got {elbo}"
    );
}

/// FG-60: Beta guide sampling is exact (two-Gamma / rand_distr), not a moment-matched
/// Gaussian clamped to [0.001, 0.999]. The clamped-Gaussian pre-fix code piled probability
/// mass exactly on the clamp boundaries and produced a unimodal (bell) shape; an exact
/// Beta(0.5, 0.5) is the bimodal arcsine law with almost no central mass.
#[test]
fn fg60_beta_sampling_is_exact_not_clamped_gaussian() {
    // Beta(0.5, 0.5): mean 0.5, and (arcsine CDF) P(0.4 < X < 0.6) ~ 0.1282.
    let param = VariationalParam::Beta {
        log_alpha: 0.5_f64.ln(),
        log_beta: 0.5_f64.ln(),
    };

    let mut rng = StdRng::seed_from_u64(77);
    let n = 20_000usize;
    let mut sum = 0.0;
    let mut central = 0usize; // in (0.4, 0.6)
    let mut clamped = 0usize; // exactly at old clamp boundaries
    for _ in 0..n {
        let (v, aux) = param.sample_with_aux(&mut rng);
        assert!(
            v > 0.0 && v < 1.0,
            "exact Beta sample must lie in (0,1): {v}"
        );
        // FG-60: Beta has no reparameterization base -> aux is NaN.
        assert!(
            aux.is_nan(),
            "Beta aux must be NaN (no reparameterization base)"
        );
        sum += v;
        if v > 0.4 && v < 0.6 {
            central += 1;
        }
        if v == 0.001 || v == 0.999 {
            clamped += 1;
        }
    }

    // No sample may sit exactly on the pre-fix clamp boundaries.
    assert_eq!(
        clamped, 0,
        "exact Beta sampler must never hit the clamp boundaries"
    );

    // Mean ~ 0.5 (SE ~ sd/sqrt(n); sd of arcsine = sqrt(0.125) ~ 0.354 => SE ~ 0.0025).
    let mean = sum / n as f64;
    assert!(
        (mean - 0.5).abs() < 0.02,
        "Beta(0.5,0.5) mean should be ~0.5, got {mean}"
    );

    // U-shape: central fraction ~ 0.128 for exact arcsine; the clamped Gaussian(0.5,0.354)
    // pre-fix code has ~0.22 central mass. 0.18 cleanly separates the two.
    let central_frac = central as f64 / n as f64;
    assert!(
        central_frac < 0.18,
        "exact Beta(0.5,0.5) should have little central mass (~0.128), got {central_frac}"
    );
}
