//! Variational Inference (VI) inference (finding FG-25).
//!
//! `optimize_meanfield_vi`/`elbo_with_guide` were re-exported at the crate root
//! and documented at length in `src/inference/vi.rs`'s rustdoc, but -- like SMC
//! and ABC -- were never exercised by any example or mdBook guide. This
//! example fits a mean-field Gaussian guide by stochastic gradient ascent on
//! the ELBO and checks the fitted guide against the same closed-form target
//! used by `smc_inference.rs`.
//!
//! ## Model and why mean-field VI is *exact* here
//!
//! Conjugate Normal-Normal: `mu ~ Normal(0, 1)`, `y | mu ~ Normal(mu, 0.5)`,
//! observed `y = 1.5`. The true posterior `Normal(1.2, sqrt(0.2))` (see
//! `smc_inference.rs` for the derivation) is itself Gaussian, and the
//! mean-field guide family for a [`Support::Real`] latent is exactly a
//! `Normal(mu, sigma)` factor (see [`fugue::inference::vi::VariationalParam`]).
//! So unlike the SMC/ABC examples (which target an approximation that is only
//! asymptotically exact), a *converged* mean-field VI fit on this model has no
//! family-mismatch bias at all: this example is a clean check that the
//! optimizer actually finds the right Gaussian, not just "some" Gaussian.

use fugue::inference::vi::{optimize_meanfield_vi_with_config, Support, VIConfig};
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

// ANCHOR: model
/// `mu ~ Normal(0, 1)`; observe `y ~ Normal(mu, 0.5)` at the fixed value 1.5.
fn model() -> Model<f64> {
    sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap())
        .bind(|mu| observe(addr!("y"), Normal::new(mu, 0.5).unwrap(), 1.5).map(move |_| mu))
}
// ANCHOR_END: model

// Same closed-form target as `smc_inference.rs` (see that file's derivation).
const POSTERIOR_MEAN: f64 = 1.2;
const POSTERIOR_SD: f64 = 0.4472135954999579; // sqrt(0.2)

fn main() {
    println!("=== Variational Inference (VI) ===\n");

    // ANCHOR: run_vi
    let mut rng = StdRng::seed_from_u64(11);

    // A real-supported latent gets a Normal factor -- the family that can
    // represent this model's posterior exactly (finding FG-17: guide families
    // are chosen to match the latent's support instead of a one-size-fits-all
    // Normal that would mismatch bounded latents).
    let mut guide = MeanFieldGuide::new();
    guide.add_latent(addr!("mu"), Support::Real, 0.0);

    let config = VIConfig {
        n_iterations: 800,
        n_samples_per_iter: 32,
        base_learning_rate: 0.3,
        ..VIConfig::default()
    };
    let result = optimize_meanfield_vi_with_config(&mut rng, model, guide, &config);
    // ANCHOR_END: run_vi

    println!("Iterations run: {}", result.iterations);
    println!("Converged (ELBO plateau): {}", result.converged);
    println!(
        "Final ELBO estimate: {:.4}",
        result.elbo_history.last().copied().unwrap_or(f64::NAN)
    );

    // ANCHOR: analyze
    let VariationalParam::Normal { mu, log_sigma } = result
        .guide
        .params
        .get(&addr!("mu"))
        .expect("guide has a factor for mu")
    else {
        panic!("Support::Real latent must produce a Normal factor");
    };
    let fitted_sigma = log_sigma.exp();
    println!("Fitted q(mu) = Normal({mu:.4}, {fitted_sigma:.4})");
    println!("Exact posterior = Normal({POSTERIOR_MEAN}, {POSTERIOR_SD:.4})");
    // ANCHOR_END: analyze

    // ANCHOR: assertions
    // Mean-field VI is exact for this model family (see module docs), so a
    // healthy optimizer run should land close to the true posterior. The
    // tolerances are set from repeated runs at this seed/config: comfortably
    // above run-to-run stochastic-gradient noise, but tight enough that
    // regressing either the location or (log-space) scale update of finding
    // FG-04 -- which previously left the scale un-optimized entirely -- would
    // fail this assertion (an un-optimized scale stays at its ~1.0 init,
    // several sigma away from the target 0.4472).
    assert!(
        (*mu - POSTERIOR_MEAN).abs() < 0.15,
        "VI fitted mean {mu} too far from exact posterior mean {POSTERIOR_MEAN}"
    );
    assert!(
        (fitted_sigma - POSTERIOR_SD).abs() < 0.15,
        "VI fitted sd {fitted_sigma} too far from exact posterior sd {POSTERIOR_SD}"
    );
    // ANCHOR_END: assertions

    println!("\nVI recovered the analytic posterior within tolerance.");
}
