//! Regression tests for SMC audit findings FG-03, FG-13, FG-43, FG-58.
//!
//! All statistical tests are seeded (`StdRng::seed_from_u64`) and use tolerances
//! justified in comments. Analytic reference values are derived in-comment.

use fugue::inference::smc::{
    adaptive_smc, adaptive_smc_with_kernel, effective_sample_size, rejuvenate_particles,
    resample_particles, smc_prior_particles, CrossoverKernel, NoKernel, Particle, PopulationKernel,
    ResamplingMethod, SMCConfig,
};
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Beta-Bernoulli data: 12 successes out of 15 trials.
const DATA: [bool; 15] = [
    true, true, true, true, true, true, true, true, true, true, true, true, false, false, false,
];

/// theta ~ Beta(8, 8); x_j ~ Bernoulli(theta) for the 15 observations in DATA.
fn beta_bernoulli_model() -> Model<f64> {
    sample(addr!("theta"), Beta::new(8.0, 8.0).unwrap()).bind(|theta| {
        let mut m: Model<()> = observe(addr!("y", 0usize), Bernoulli::new(theta).unwrap(), DATA[0]);
        for (i, &b) in DATA.iter().enumerate().skip(1) {
            let a = addr!("y", i);
            m = m.bind(move |_| observe(a, Bernoulli::new(theta).unwrap(), b));
        }
        m.map(move |_| theta)
    })
}

/// FG-03: SMC importance weights must NOT double-count the prior.
///
/// Prior-proposed particles carry weight = log_likelihood only (the prior cancels
/// against the prior proposal). For the conjugate Beta(8,8)-Bernoulli model with
/// 12/15 successes:
///
/// - correct posterior: Beta(8+12, 8+3) = Beta(20, 11), mean = 20/31 = 0.645161
/// - the pre-fix (prior-squared) weight targets the effective prior Beta(15,15),
///   giving posterior Beta(27, 18), mean = 27/45 = 0.600000
///
/// The gap between correct and buggy means is 0.0452. We assert the seeded SMC
/// weighted mean is within 0.03 of the correct value; the pre-fix code produces
/// ~0.60 (0.045 away) and therefore fails this test.
#[test]
fn fg03_smc_prior_weights_do_not_square_the_prior() {
    // Analytic reference: mean of Beta(20, 11).
    // python: (8+12)/((8+12)+(8+3)) = 0.6451612903...
    const ANALYTIC_MEAN: f64 = 20.0 / 31.0;
    const BUGGY_MEAN: f64 = 27.0 / 45.0; // 0.6

    let mut rng = StdRng::seed_from_u64(20260710);
    let n = 2000;
    let particles = smc_prior_particles(&mut rng, n, beta_bernoulli_model);

    // Self-normalized importance estimate of the posterior mean of theta.
    let weighted_mean: f64 = particles
        .iter()
        .map(|p| p.weight * p.trace.get_f64(&addr!("theta")).unwrap())
        .sum();

    // With N=2000 the Monte Carlo standard error of this estimate is well under
    // 0.01 (posterior std ~0.085, ESS in the hundreds), so 0.03 comfortably
    // contains the correct value while excluding the pre-fix value 0.60.
    assert!(
        (weighted_mean - ANALYTIC_MEAN).abs() < 0.03,
        "weighted posterior mean {weighted_mean:.4} not within 0.03 of analytic {ANALYTIC_MEAN:.4}"
    );
    // Discrimination guard: the estimate must be clearly closer to the correct
    // posterior mean than to the prior-squared (buggy) mean.
    assert!(
        (weighted_mean - ANALYTIC_MEAN).abs() < (weighted_mean - BUGGY_MEAN).abs(),
        "estimate {weighted_mean:.4} is closer to the prior-squared mean {BUGGY_MEAN} than to {ANALYTIC_MEAN}"
    );
}

/// FG-13: an invariant MH rejuvenation move after resampling must leave the
/// (uniform) particle weights unchanged; post-rejuvenation ESS must equal N.
///
/// The pre-fix code reweighted each particle by the full joint after the move
/// (and renormalized), which skews a just-equalized population and drops ESS
/// below N. The fixed rejuvenation does not touch weights, so ESS stays == N.
#[test]
fn fg13_rejuvenation_preserves_uniform_weights() {
    let model_fn = || {
        sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap())
            .bind(|mu| observe(addr!("y"), Normal::new(mu, 1.0).unwrap(), 1.0).map(move |_| mu))
    };

    let mut rng = StdRng::seed_from_u64(7);
    let n = 50;
    let particles = smc_prior_particles(&mut rng, n, model_fn);

    // Resample: weights become uniform, ESS == N.
    let mut resampled = resample_particles(&mut rng, &particles, ResamplingMethod::Systematic);
    let ess_before = effective_sample_size(&resampled);
    assert!(
        (ess_before - n as f64).abs() < 1e-9,
        "post-resample ESS {ess_before} should equal N={n}"
    );

    // Snapshot mu values to confirm the move actually perturbs particles.
    let mu_before: Vec<f64> = resampled
        .iter()
        .map(|p| p.trace.get_f64(&addr!("mu")).unwrap())
        .collect();

    // Invariant MH rejuvenation at beta = 1 (the posterior). No reweighting.
    rejuvenate_particles(&mut rng, &mut resampled, model_fn, 1.0, 5);

    // Weights must be untouched: still exactly uniform, ESS still == N.
    let ess_after = effective_sample_size(&resampled);
    assert!(
        (ess_after - n as f64).abs() < 1e-9,
        "post-rejuvenation ESS {ess_after} should still equal N={n} (FG-13)"
    );
    for p in &resampled {
        assert!(
            (p.weight - 1.0 / n as f64).abs() < 1e-12,
            "rejuvenation must not change weights"
        );
    }

    // Sanity: the move did change at least one particle (so invariance is
    // non-trivially preserved, not preserved because nothing moved).
    let mu_after: Vec<f64> = resampled
        .iter()
        .map(|p| p.trace.get_f64(&addr!("mu")).unwrap())
        .collect();
    let moved = mu_before
        .iter()
        .zip(&mu_after)
        .any(|(a, b)| (a - b).abs() > 1e-9);
    assert!(moved, "rejuvenation should move at least one particle");
}

/// FG-43 + FG-58: genuine likelihood-tempered SMC recovers both the analytic
/// posterior mean and the analytic log marginal likelihood.
///
/// Model: mu ~ N(0, 1); y_j ~ N(mu, 1) for ys = [1.0, 2.0, 1.5, 0.5, 1.8].
/// Analytic (Normal-Normal conjugate, verified by two independent methods):
///   - posterior mean = 1.133333, var = 0.166667
///   - log marginal likelihood log p(y) = -7.007239
///     (python sequential predictive factorization; see comment below)
#[test]
fn fg43_fg58_tempered_smc_matches_conjugate_evidence_and_mean() {
    // python (pure, no scipy): sequential predictive factorization
    //   m,v = 0,1; logZ=0
    //   for y in [1.0,2.0,1.5,0.5,1.8]:
    //       pv = v + 1.0
    //       logZ += -0.5*log(2*pi*pv) - (y-m)**2/(2*pv)
    //       prec = 1/v + 1/1.0; m = (m/v + y/1.0)/prec; v = 1/prec
    //   -> logZ = -7.007239, posterior mean = 1.133333
    const ANALYTIC_LOG_Z: f64 = -7.007239;
    const ANALYTIC_MEAN: f64 = 1.133333;

    let ys = [1.0_f64, 2.0, 1.5, 0.5, 1.8];
    let model_fn = || {
        sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap()).bind(move |mu| {
            let mut m: Model<()> =
                observe(addr!("y", 0usize), Normal::new(mu, 1.0).unwrap(), ys[0]);
            for (i, &y) in ys.iter().enumerate().skip(1) {
                let a = addr!("y", i);
                m = m.bind(move |_| observe(a, Normal::new(mu, 1.0).unwrap(), y));
            }
            m.map(move |_| mu)
        })
    };

    let mut rng = StdRng::seed_from_u64(2026);
    let config = SMCConfig {
        resampling_method: ResamplingMethod::Systematic,
        ess_threshold: 0.5,
        rejuvenation_steps: 3,
    };
    let n = 2000;
    let result = adaptive_smc(&mut rng, n, model_fn, config);

    // Weighted posterior mean of mu.
    let total_w: f64 = result.iter().map(|p| p.weight).sum();
    let mean: f64 = result
        .iter()
        .map(|p| p.weight * p.trace.get_f64(&addr!("mu")).unwrap())
        .sum::<f64>()
        / total_w;

    // Posterior std ~0.408; with N=2000 and rejuvenation the ESS is in the
    // hundreds+, so SE < 0.02 -> 0.06 is a safe ~3*SE band.
    assert!(
        (mean - ANALYTIC_MEAN).abs() < 0.06,
        "SMC posterior mean {mean:.4} not within 0.06 of analytic {ANALYTIC_MEAN:.4}"
    );

    // Log-evidence: the tempered-SMC estimator of log p(y). Its variance grows
    // with the number of tempering steps; 0.2 in log-space is a conservative
    // band for this 5-observation conjugate model at N=2000 (validated below).
    assert!(
        result.log_evidence.is_finite(),
        "log evidence must be finite"
    );
    assert!(
        (result.log_evidence - ANALYTIC_LOG_Z).abs() < 0.2,
        "SMC log evidence {:.4} not within 0.2 of analytic {ANALYTIC_LOG_Z:.4}",
        result.log_evidence
    );
}

// ---------------------------------------------------------------------------
// FG-N1 (SMC side): the rejuvenation kernel must be π_β-invariant on a bounded
// prior whose support contains negatives but not -1. The pre-fix kernel chose
// its proposal kind per call from the *current value* (log-space iff current > 0
// and the density probe at -1 was -inf): from x < 0 it proposed a Gaussian step
// that could land at x' > 0, from where only the log-space walk (never negative)
// was used - and the code treated both as symmetric. Mass leaked one way until
// the whole population sat in (0, 1/2]. Target rho(x) ∝ exp(2x) on [-1/2, 1/2]:
// E[x] = e^{-1}/(2 sinh 1) = 0.156518, P(x > 0) = ((e-1)/2)/sinh(1) = 0.731059;
// the confined population has E[x] = 0.290988, P(x > 0) = 1.
// ---------------------------------------------------------------------------
#[test]
fn fgn1_smc_rejuvenation_is_invariant_on_bounded_prior_with_negatives() {
    let model_fn = || {
        sample(addr!("x"), Uniform::new(-0.5, 0.5).unwrap())
            .bind(|x| factor(2.0 * x).map(move |_| x))
    };
    let config = SMCConfig {
        resampling_method: ResamplingMethod::Systematic,
        // A low threshold forces several intermediate tempering steps, each
        // followed by resample + rejuvenation, so a non-invariant kernel has
        // many chances to drift the population.
        ess_threshold: 0.9,
        rejuvenation_steps: 10,
    };
    let mut rng = StdRng::seed_from_u64(20260905);
    let result = adaptive_smc(&mut rng, 2_000, model_fn, config);

    let mean: f64 = result
        .iter()
        .map(|p| p.weight * p.trace.get_f64(&addr!("x")).unwrap())
        .sum();
    let p_pos: f64 = result
        .iter()
        .filter(|p| p.trace.get_f64(&addr!("x")).unwrap() > 0.0)
        .map(|p| p.weight)
        .sum();
    assert!(
        (mean - 0.156_518).abs() < 0.03,
        "SMC posterior mean {mean:.4} vs analytic 0.1565 (confined value: 0.291)"
    );
    assert!(
        (p_pos - 0.731_059).abs() < 0.06,
        "SMC P(x > 0) = {p_pos:.3} vs analytic 0.731 (confined value: 1.0)"
    );
    // Log-evidence is untouched by an invariant kernel: log Z = log sinh(1)
    // (the prior density 1 on a unit-width interval times ∫ e^{2x}).
    assert!(
        (result.log_evidence - 1.0_f64.sinh().ln()).abs() < 0.05,
        "log-evidence {:.4} vs analytic {:.4}",
        result.log_evidence,
        1.0_f64.sinh().ln()
    );
}

// ---------------------------------------------------------------------------
// FG-N3: `adaptive_smc_with_kernel` must apply a non-identity kernel even when
// `rejuvenation_steps == 0`. The FG-43 shortcut (single 0 -> 1 reweight, no
// ladder) used to key on `rejuvenation_steps == 0` alone, so the kernel - which
// lives inside the ladder - was never invoked. `PopulationKernel::is_identity`
// now gates the shortcut.
// FG-N7: particles handed to `sweep` carry uniform weights.
// ---------------------------------------------------------------------------

/// Records every sweep and asserts the FG-N7 weight contract on entry.
struct RecordingKernel {
    sweeps: usize,
    betas: Vec<f64>,
}

impl<A> PopulationKernel<A> for RecordingKernel {
    fn sweep(
        &mut self,
        _rng: &mut dyn rand::RngCore,
        particles: &mut [Particle],
        _model_fn: &dyn Fn() -> Model<A>,
        beta: f64,
    ) {
        let n = particles.len() as f64;
        for (i, p) in particles.iter().enumerate() {
            assert!(
                (p.weight - 1.0 / n).abs() < 1e-12,
                "sweep {}: particle {i} entered with weight {} (expected 1/n = {})",
                self.sweeps,
                p.weight,
                1.0 / n
            );
            assert!(
                (p.log_weight + n.ln()).abs() < 1e-12,
                "sweep {}: particle {i} entered with log_weight {} (expected -ln n = {})",
                self.sweeps,
                p.log_weight,
                -n.ln()
            );
        }
        self.sweeps += 1;
        self.betas.push(beta);
    }
}

#[test]
fn fgn3_non_identity_kernel_is_applied_with_zero_rejuvenation_steps() {
    let config = SMCConfig {
        resampling_method: ResamplingMethod::Systematic,
        // A demanding ESS target forces several intermediate tempering steps on
        // this model (12/15 successes against a Beta(8,8) prior).
        ess_threshold: 0.9,
        rejuvenation_steps: 0,
    };
    let mut kernel = RecordingKernel {
        sweeps: 0,
        betas: Vec::new(),
    };
    let mut rng = StdRng::seed_from_u64(20260905);
    let result = adaptive_smc_with_kernel(&mut rng, 500, beta_bernoulli_model, config, &mut kernel);

    assert!(
        kernel.sweeps > 0,
        "kernel was never swept with rejuvenation_steps == 0 (FG-N3 regression)"
    );
    // Never at the terminal beta = 1 step (FG-43).
    assert!(
        kernel.betas.iter().all(|&b| b > 0.0 && b < 1.0),
        "kernel swept outside (0, 1): {:?}",
        kernel.betas
    );
    // The identity kernel with no rejuvenation still takes the shortcut.
    assert!(<NoKernel as PopulationKernel<f64>>::is_identity(&NoKernel));
    assert!(!<RecordingKernel as PopulationKernel<f64>>::is_identity(
        &kernel
    ));
    // And the ladder, being a no-move ladder for this kernel, still returns a
    // sound posterior: Beta(20, 11) mean = 0.6452.
    let mean: f64 = result
        .iter()
        .map(|p| p.weight * p.trace.get_f64(&addr!("theta")).unwrap())
        .sum();
    assert!(
        (mean - 20.0 / 31.0).abs() < 0.03,
        "posterior mean {mean:.4}"
    );
}

/// FG-N3 end to end: a crossover kernel with `rejuvenation_steps == 0` is the
/// configuration fugue-evo's grammar SMC ships by default. It must be applied
/// (population diversity is restored after each resample) and, being
/// pi_beta-invariant on the product, it must leave the posterior mean and the
/// log-evidence correct. Two independent Normal(0,1) sites each observed with
/// sd 1: posterior mean y_i/2, log Z = sum_i log N(y_i; 0, sqrt 2).
#[test]
fn fgn3_crossover_kernel_without_rejuvenation_is_applied_and_invariant() {
    let (y0, y1) = (1.0_f64, -0.5_f64);
    let model_fn = move || {
        sample(addr!("x", 0), Normal::new(0.0, 1.0).unwrap()).bind(move |x0| {
            sample(addr!("x", 1), Normal::new(0.0, 1.0).unwrap()).bind(move |x1| {
                observe(addr!("y", 0), Normal::new(x0, 1.0).unwrap(), y0).bind(move |_| {
                    observe(addr!("y", 1), Normal::new(x1, 1.0).unwrap(), y1).map(move |_| (x0, x1))
                })
            })
        })
    };
    let n = 3_000usize;
    let config = SMCConfig {
        resampling_method: ResamplingMethod::Systematic,
        ess_threshold: 0.8,
        rejuvenation_steps: 0,
    };
    let mut kernel = CrossoverKernel {
        n_pairs: n,
        // Value-independent, pair-symmetric: swap site 0 or site 1 at random.
        mask: Box::new(|_a: &Trace, _b: &Trace, rng: &mut dyn rand::RngCore| {
            let i = rand::Rng::gen_range(rng, 0..2usize);
            vec![addr!("x", i)]
        }),
    };
    let mut rng = StdRng::seed_from_u64(4242);
    let result = adaptive_smc_with_kernel(&mut rng, n, model_fn, config, &mut kernel);

    for (i, y) in [(0usize, y0), (1usize, y1)] {
        let mean: f64 = result
            .iter()
            .map(|p| p.weight * p.trace.get_f64(&addr!("x", i)).unwrap())
            .sum();
        assert!(
            (mean - y / 2.0).abs() < 0.05,
            "site {i}: posterior mean {mean:.4} vs analytic {:.4}",
            y / 2.0
        );
    }
    let log_n = |y: f64| {
        -0.5 * (y * y / 2.0) - (2.0_f64).sqrt().ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
    };
    let log_z = log_n(y0) + log_n(y1);
    assert!(
        (result.log_evidence - log_z).abs() < 0.05,
        "log-evidence {:.4} vs analytic {log_z:.4}",
        result.log_evidence
    );
}
