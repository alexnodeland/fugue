//! Approximate Bayesian Computation (ABC) inference (finding FG-25).
//!
//! `abc_rejection`/`abc_smc`/`abc_smc_weighted` were re-exported at the crate
//! root and documented at length in `src/inference/abc.rs`'s rustdoc, but --
//! like SMC and VI -- were never exercised by any example or mdBook guide.
//! This example runs the likelihood-free ABC-SMC algorithm end-to-end and
//! checks its posterior approximation against the same closed-form target used
//! by `smc_inference.rs`.
//!
//! ## Model and why ABC applies here
//!
//! `mu ~ Normal(0, 1)`; the model then *forward-simulates* one synthetic
//! observation `y_sim ~ Normal(mu, 0.5)` (a `sample`, not an `observe` --
//! ABC never scores a likelihood, it only compares simulated data to real
//! data). ABC accepts a draw of `mu` when its simulated `y_sim` lands within
//! `tolerance` of the real observation `y_obs = 1.5`.
//!
//! As `tolerance -> 0`, accepting `|y_sim - y_obs| <= tolerance` converges to
//! conditioning on `y_sim = y_obs` exactly, which is the same event a proper
//! Bayesian update on `y ~ Normal(mu, 0.5) = 1.5` conditions on -- so the ABC
//! posterior converges to the *same* closed-form target as the SMC example:
//! `Normal(1.2, sqrt(0.2))` (see `smc_inference.rs` for the derivation). At
//! finite tolerance ABC is only an approximation of that target, which is why
//! this example's tolerance bands on the *assertions* are wider than the SMC
//! example's -- that gap is the whole point of ABC: exact inference traded for
//! applicability to simulators with no tractable likelihood at all.

use fugue::inference::abc::{abc_smc_weighted, ABCSMCConfig, EuclideanDistance};
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

// ANCHOR: model
/// `mu ~ Normal(0, 1)`; forward-simulate `y_sim ~ Normal(mu, 0.5)` (a `sample`,
/// not an `observe` -- ABC never scores this density, only compares outcomes).
fn model() -> Model<f64> {
    sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap())
        .bind(|mu| sample(addr!("y_sim"), Normal::new(mu, 0.5).unwrap()).map(move |_| mu))
}
// ANCHOR_END: model

// Same closed-form target as `smc_inference.rs` (see that file's derivation):
// the small-tolerance limit of this ABC setup conditions on y_sim = 1.5 under
// likelihood Normal(mu, 0.5) with prior Normal(0, 1).
const POSTERIOR_MEAN: f64 = 1.2;
const POSTERIOR_SD: f64 = 0.4472135954999579; // sqrt(0.2)

fn main() {
    println!("=== Approximate Bayesian Computation (ABC) Inference ===\n");

    // ANCHOR: run_abc
    let observed: Vec<f64> = vec![1.5];
    let mut rng = StdRng::seed_from_u64(7);

    let config = ABCSMCConfig {
        initial_tolerance: 2.0,
        // Geometrically shrinking tolerance schedule (Beaumont/Toni-style):
        // each stage's population is built by perturbing and re-weighting the
        // previous one (see `abc_smc_weighted`'s rustdoc).
        tolerance_schedule: vec![1.0, 0.5, 0.25, 0.1],
        particles_per_round: 500,
    };

    let result = abc_smc_weighted(
        &mut rng,
        model,
        // The simulator just reads back the model's own forward-simulated
        // synthetic observation.
        |trace| vec![trace.get_f64(&addr!("y_sim")).unwrap()],
        &observed,
        &EuclideanDistance,
        config,
        // Attempt budget per stage (finding FG-34: bounded, typed-error
        // instead of an unbounded loop / panic on an empty population).
        200_000,
    )
    .expect("ABC-SMC should complete with this many particles/attempts");
    // ANCHOR_END: run_abc

    println!("Final tolerance: {}", result.final_tolerance);
    println!("Particles: {}", result.particles.len());

    // ANCHOR: analyze
    let posterior_mean = result
        .weighted_mean(&addr!("mu"))
        .expect("mu is present in every particle's trace");
    let posterior_var: f64 = {
        let mut num = 0.0;
        let mut den = 0.0;
        for p in &result.particles {
            let mu = p.trace.get_f64(&addr!("mu")).unwrap();
            num += p.weight * (mu - posterior_mean).powi(2);
            den += p.weight;
        }
        num / den
    };
    println!("Posterior mean(mu) ~= {posterior_mean:.4} (target: {POSTERIOR_MEAN})");
    println!(
        "Posterior sd(mu)   ~= {:.4} (target: {POSTERIOR_SD:.4})",
        posterior_var.sqrt()
    );
    // ANCHOR_END: analyze

    // ANCHOR: assertions
    // ABC is only asymptotically (tolerance -> 0) exact, so these bands are
    // deliberately wider than the SMC example's: at final tolerance 0.1 with
    // 500 particles, the Beaumont-kernel ABC-SMC posterior mean should still
    // land within a few tenths of the exact value, but a broken importance
    // correction (e.g. reverting to the prior-replacement heuristic of
    // finding FG-09) biases it by much more than this band allows.
    assert!(
        (posterior_mean - POSTERIOR_MEAN).abs() < 0.3,
        "ABC posterior mean {posterior_mean} too far from target {POSTERIOR_MEAN}"
    );
    assert!(
        posterior_var.sqrt() < POSTERIOR_SD * 2.0,
        "ABC posterior sd {} implausibly larger than target {POSTERIOR_SD}",
        posterior_var.sqrt()
    );
    // ANCHOR_END: assertions

    println!("\nABC-SMC approximated the target posterior within tolerance.");
}
