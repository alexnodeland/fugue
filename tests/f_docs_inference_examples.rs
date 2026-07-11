//! Regression coverage for finding FG-25: SMC, ABC and VI must actually
//! recover a known posterior, not merely "run without panicking".
//!
//! FG-25 found that three of the four headline "Multiple Inference Methods"
//! (SMC, VI, ABC) were re-exported at the crate root and documented at length
//! in rustdoc, but exercised by zero examples and zero mdBook guides -- so
//! nothing in CI checked that a first-time user's copy-paste of these APIs
//! would actually work end-to-end. `examples/smc_inference.rs`,
//! `examples/abc_inference.rs` and `examples/vi_inference.rs` close that gap
//! for humans reading the docs; these tests close it for CI, so a regression
//! in `adaptive_smc`, `abc_smc_weighted` or `optimize_meanfield_vi_with_config`
//! (or their crate-root re-exports going stale/renamed) fails the build
//! instead of silently rotting the example surface again.
//!
//! All tests are seeded (`StdRng::seed_from_u64`) against the same conjugate
//! Normal-Normal target used by the examples, with tolerances justified
//! in-comment from the analytic posterior (see each example's module docs for
//! the precision-weighted-update derivation):
//!
//! ```text
//! prior mu ~ Normal(0, 1); y | mu ~ Normal(mu, 0.5); observed y = 1.5
//! post_prec = 1/1.0**2 + 1/0.5**2 = 5.0
//! post_mean = (0.0*1.0 + 1.5*4.0) / 5.0 = 1.2
//! post_var  = 1 / 5.0 = 0.2   =>  post_sd = 0.4472135954999579
//! ```

use fugue::inference::abc::{abc_smc_weighted, ABCSMCConfig, EuclideanDistance};
use fugue::inference::vi::{optimize_meanfield_vi_with_config, Support, VIConfig};
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

const POSTERIOR_MEAN: f64 = 1.2;
const POSTERIOR_VAR: f64 = 0.2;
const POSTERIOR_SD: f64 = 0.4472135954999579; // sqrt(0.2)

fn observed_model() -> Model<f64> {
    sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap())
        .bind(|mu| observe(addr!("y"), Normal::new(mu, 0.5).unwrap(), 1.5).map(move |_| mu))
}

fn forward_sim_model() -> Model<f64> {
    sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap())
        .bind(|mu| sample(addr!("y_sim"), Normal::new(mu, 0.5).unwrap()).map(move |_| mu))
}

/// FG-25: `adaptive_smc` (crate-root re-export) must recover the analytic
/// posterior on a model with a known closed form, matching
/// `examples/smc_inference.rs`.
#[test]
fn fg25_smc_example_recovers_known_posterior() {
    let mut rng = StdRng::seed_from_u64(42);
    let config = SMCConfig {
        resampling_method: ResamplingMethod::Systematic,
        ess_threshold: 0.5,
        rejuvenation_steps: 3,
    };
    let result = adaptive_smc(&mut rng, 2000, observed_model, config);

    let mean: f64 = result
        .iter()
        .filter_map(|p| p.trace.get_f64(&addr!("mu")).map(|mu| p.weight * mu))
        .sum();
    let var: f64 = result
        .iter()
        .filter_map(|p| {
            p.trace
                .get_f64(&addr!("mu"))
                .map(|mu| p.weight * (mu - mean).powi(2))
        })
        .sum();

    // Same tolerance rationale as the example: 2000 particles + 3
    // rejuvenation moves per step keeps Monte Carlo error well under 0.1 for
    // this 1-D conjugate model.
    assert!(
        (mean - POSTERIOR_MEAN).abs() < 0.15,
        "SMC mean {mean} deviates from exact posterior mean {POSTERIOR_MEAN}"
    );
    assert!(
        (var - POSTERIOR_VAR).abs() < 0.1,
        "SMC var {var} deviates from exact posterior var {POSTERIOR_VAR}"
    );
    assert!(result.log_evidence.is_finite());
}

/// FG-25: `abc_smc_weighted` (crate-root-reachable via `inference::abc`) must
/// approximate the same target in its small-tolerance limit, matching
/// `examples/abc_inference.rs`.
#[test]
fn fg25_abc_example_recovers_known_posterior() {
    let observed: Vec<f64> = vec![1.5];
    let mut rng = StdRng::seed_from_u64(7);
    let config = ABCSMCConfig {
        initial_tolerance: 2.0,
        tolerance_schedule: vec![1.0, 0.5, 0.25, 0.1],
        particles_per_round: 500,
    };
    let result = abc_smc_weighted(
        &mut rng,
        forward_sim_model,
        |trace| vec![trace.get_f64(&addr!("y_sim")).unwrap()],
        &observed,
        &EuclideanDistance,
        config,
        200_000,
    )
    .expect("ABC-SMC should complete with this many particles/attempts");

    let mean = result
        .weighted_mean(&addr!("mu"))
        .expect("mu present in every particle");

    // Wider band than SMC: ABC is only asymptotically exact as tolerance -> 0.
    assert!(
        (mean - POSTERIOR_MEAN).abs() < 0.3,
        "ABC-SMC mean {mean} deviates from target posterior mean {POSTERIOR_MEAN}"
    );
}

/// FG-25: `optimize_meanfield_vi_with_config` (crate-root-reachable via
/// `inference::vi`) must fit the true Gaussian posterior exactly for this
/// conjugate model, matching `examples/vi_inference.rs`.
#[test]
fn fg25_vi_example_recovers_known_posterior() {
    let mut rng = StdRng::seed_from_u64(11);
    let mut guide = MeanFieldGuide::new();
    guide.add_latent(addr!("mu"), Support::Real, 0.0);
    let config = VIConfig {
        n_iterations: 800,
        n_samples_per_iter: 32,
        base_learning_rate: 0.3,
        ..VIConfig::default()
    };
    let result = optimize_meanfield_vi_with_config(&mut rng, observed_model, guide, &config);

    let VariationalParam::Normal { mu, log_sigma } = result
        .guide
        .params
        .get(&addr!("mu"))
        .expect("guide has a factor for mu")
    else {
        panic!("Support::Real latent must produce a Normal factor");
    };
    let sigma = log_sigma.exp();

    // Mean-field VI is exact for this model (the posterior is Gaussian and so
    // is the guide family), so both location and scale should land close to
    // ground truth. This is also a regression guard for FG-04: an
    // un-optimized scale would stay near its ~1.0 init, several sigma from
    // the target 0.4472.
    assert!(
        (*mu - POSTERIOR_MEAN).abs() < 0.15,
        "VI mean {mu} deviates from exact posterior mean {POSTERIOR_MEAN}"
    );
    assert!(
        (sigma - POSTERIOR_SD).abs() < 0.15,
        "VI sd {sigma} deviates from exact posterior sd {POSTERIOR_SD}"
    );
}
