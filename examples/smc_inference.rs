//! Sequential Monte Carlo (SMC) inference (finding FG-25).
//!
//! `adaptive_smc` was one of three headline "Multiple Inference Methods" (SMC,
//! Variational Inference, ABC) that were re-exported at the crate root and
//! documented in `src/inference/smc.rs`'s rustdoc but never exercised by any
//! example or mdBook guide — a first-time user following the README's pointer
//! to `examples/` would never see it actually invoked. This example runs
//! likelihood-tempered SMC end-to-end on a model with a known closed-form
//! posterior and checks the particle population recovers it.
//!
//! ## Model
//!
//! Conjugate Normal-Normal: `mu ~ Normal(0, 1)`, `y | mu ~ Normal(mu, 0.5)`,
//! observed `y = 1.5`. The posterior is exactly
//! `Normal(1.2, sqrt(0.2)) = Normal(1.2, 0.4472...)`
//! (precision-weighted combination of prior and likelihood; see the `python3`
//! reference computation below), which lets us check SMC's particle population
//! against ground truth rather than just asserting "it runs".
//!
//! ```text
//! # scipy / by-hand precision-weighted Normal-Normal conjugate update:
//! prior_prec = 1/1.0**2          # = 1.0
//! lik_prec   = 1/0.5**2          # = 4.0
//! post_prec  = prior_prec + lik_prec               # = 5.0
//! post_mean  = (0.0*prior_prec + 1.5*lik_prec) / post_prec  # = 1.2
//! post_var   = 1 / post_prec                       # = 0.2
//! ```

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

// Closed-form posterior mean/variance for the conjugate model above (see the
// module docs for the derivation): precision-weighted combination of the
// N(0, 1) prior and the N(mu, 0.5) likelihood evaluated at y = 1.5.
const POSTERIOR_MEAN: f64 = 1.2;
const POSTERIOR_VAR: f64 = 0.2;

fn main() {
    println!("=== Sequential Monte Carlo (SMC) Inference ===\n");

    // ANCHOR: run_smc
    let mut rng = StdRng::seed_from_u64(42);
    let config = SMCConfig {
        resampling_method: ResamplingMethod::Systematic,
        ess_threshold: 0.5,
        // Rejuvenation moves diversify particles after each resample, which is
        // what makes this genuine (multi-step) tempered SMC rather than a
        // single importance-sampling reweight (see `adaptive_smc`'s rustdoc).
        rejuvenation_steps: 3,
    };
    let num_particles = 2000;
    let result = adaptive_smc(&mut rng, num_particles, model, config);
    // ANCHOR_END: run_smc

    println!("Particles: {}", result.particles.len());
    println!("Log-evidence estimate: {:.4}", result.log_evidence);
    println!(
        "ESS of final population: {:.1}",
        effective_sample_size(&result)
    );

    // ANCHOR: analyze
    // Weighted posterior mean/variance over `mu` from the final particle population.
    let weighted_mean: f64 = result
        .iter()
        .filter_map(|p| p.trace.get_f64(&addr!("mu")).map(|mu| p.weight * mu))
        .sum();
    let weighted_var: f64 = result
        .iter()
        .filter_map(|p| {
            p.trace
                .get_f64(&addr!("mu"))
                .map(|mu| p.weight * (mu - weighted_mean).powi(2))
        })
        .sum();
    println!("Posterior mean(mu) ~= {weighted_mean:.4} (exact: {POSTERIOR_MEAN})");
    println!("Posterior var(mu)  ~= {weighted_var:.4} (exact: {POSTERIOR_VAR})");
    // ANCHOR_END: analyze

    // ANCHOR: assertions
    // With 2000 particles and 3 rejuvenation moves per intermediate temper
    // step, the weighted-mean Monte Carlo error is well under 0.1 for this
    // 1-D conjugate model; 0.15 gives comfortable headroom without being
    // vacuous (a broken importance weight -- e.g. reintroducing the
    // prior-squaring bug of FG-03 -- would shift the mean by several tenths).
    assert!(
        (weighted_mean - POSTERIOR_MEAN).abs() < 0.15,
        "SMC posterior mean {weighted_mean} too far from exact {POSTERIOR_MEAN}"
    );
    assert!(
        (weighted_var - POSTERIOR_VAR).abs() < 0.1,
        "SMC posterior var {weighted_var} too far from exact {POSTERIOR_VAR}"
    );
    assert!(
        result.log_evidence.is_finite(),
        "log-evidence estimate must be finite"
    );
    // ANCHOR_END: assertions

    println!("\nSMC inference recovered the analytic posterior within tolerance.");
}
