//! EA-as-PPL F6 acceptance: every primitive the fugue-evo inference layer
//! consumes is importable from the crate root. This test exists so a future
//! re-export regression fails loudly instead of breaking the downstream crate.

#[allow(unused_imports)]
use fugue::{
    adaptive_smc, adaptive_smc_with_kernel, block_regeneration_mh, decode_particle,
    decode_particles, effective_sample_size, multinomial_resample, normalize_particles,
    rejuvenate_particles, resample_particles, score_given_trace_reconciled, smc_prior_particles,
    stratified_resample, systematic_resample, try_decode_particle, CrossoverKernel, NoKernel,
    Particle, PopulationKernel, ReconcileReport, ResamplingMethod, SMCConfig, SMCResult,
};

use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Exercise the root-exported EA-as-PPL surface end to end: SMC with a
/// population kernel, root-level decode, and a root-level block move.
#[test]
fn root_exports_are_usable() {
    let model_fn = || {
        sample(addr!("x", 0), Normal::new(0.0, 1.0).unwrap())
            .and_then(|x| observe(addr!("y"), Normal::new(x, 1.0).unwrap(), 0.8).map(move |_| x))
    };
    let mut rng = StdRng::seed_from_u64(5);

    let mut kernel = CrossoverKernel {
        n_pairs: 10,
        mask: Box::new(|_: &Trace, _: &Trace, _: &mut dyn rand::RngCore| vec![addr!("x", 0)]),
    };
    let result: SMCResult = adaptive_smc_with_kernel(
        &mut rng,
        50,
        model_fn,
        SMCConfig {
            resampling_method: ResamplingMethod::Systematic,
            ess_threshold: 0.5,
            rejuvenation_steps: 1,
        },
        &mut kernel,
    );
    assert_eq!(result.len(), 50);
    assert!(result.log_evidence.is_finite());

    let decoded = decode_particles(&result, model_fn);
    assert_eq!(decoded.len(), 50);
    assert!(try_decode_particle(&result[0], model_fn).is_ok());

    let (_, moved) =
        block_regeneration_mh(&mut rng, model_fn, &result[0].trace, &[addr!("x", 0)], 1.0);
    assert!(moved.total_log_weight().is_finite());

    // Trace surgery + prefix predicate are part of the same surface.
    let sub = moved.extract_prefix("x");
    assert_eq!(sub.choices.len(), 1);
    assert!(addr!("x", 0).has_prefix("x"));

    let (_, _, report): (_, _, ReconcileReport) =
        score_given_trace_reconciled(moved.clone(), &mut rng, model_fn()).unwrap();
    assert!(report.fresh_addresses.is_empty());
    assert!(report.vanished_addresses.is_empty());
}
