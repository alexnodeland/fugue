//! FG-31: HMC must be more sample-efficient PER MODEL EVALUATION than
//! single-site Metropolis-Hastings on a correlated Gaussian — the exact regime
//! where single-site MH mixes badly and a gradient kernel earns its cost.

use fugue::inference::hmc::{hmc_chain, HMCConfig};
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::cell::Cell;

const RHO: f64 = 0.99;

/// Correlated 2-D Gaussian with unit marginals and correlation `RHO`:
/// `x ~ N(0,1)`, `y|x ~ N(RHO·x, sqrt(1-RHO²))` gives joint covariance
/// `[[1, RHO],[RHO, 1]]`. The long principal axis (`x + y`) is the direction
/// single-site MH struggles to traverse.
fn build_model() -> Model<(f64, f64)> {
    let cond_sd = (1.0 - RHO * RHO).sqrt();
    sample(addr!("x"), Normal::new(0.0, 1.0).unwrap()).bind(move |x| {
        sample(addr!("y"), Normal::new(RHO * x, cond_sd).unwrap()).map(move |y| (x, y))
    })
}

#[test]
fn fg31_hmc_beats_mh_on_ess_per_model_eval() {
    // Count model executions exactly: every `run(...)` calls `model_fn()` once,
    // so incrementing on each call gives a fair, method-agnostic evaluation count.
    // ---- HMC ----
    let hmc_evals = Cell::new(0usize);
    let hmc_model_fn = || {
        hmc_evals.set(hmc_evals.get() + 1);
        build_model()
    };
    let mut rng = StdRng::seed_from_u64(20260711);
    let cfg = HMCConfig {
        n_leapfrog: 12,
        target_accept: 0.8,
        init_step_size: None,
        finite_diff_eps: 1e-5,
        adapt_mass: false,
    };
    let hmc_samples = hmc_chain(&mut rng, hmc_model_fn, 1500, 600, cfg);
    let hmc_eval_count = hmc_evals.get();
    let hmc_s: Vec<f64> = hmc_samples.iter().map(|((x, y), _)| x + y).collect();
    let hmc_ess = effective_sample_size_mcmc(&hmc_s);
    let hmc_ess_per_eval = hmc_ess / hmc_eval_count as f64;

    // ---- adaptive single-site MH ----
    let mh_evals = Cell::new(0usize);
    let mh_model_fn = || {
        mh_evals.set(mh_evals.get() + 1);
        build_model()
    };
    let mut rng2 = StdRng::seed_from_u64(20260711);
    let mh_samples = adaptive_mcmc_chain(&mut rng2, mh_model_fn, 12000, 2000);
    let mh_eval_count = mh_evals.get();
    let mh_s: Vec<f64> = mh_samples.iter().map(|((x, y), _)| x + y).collect();
    let mh_ess = effective_sample_size_mcmc(&mh_s);
    let mh_ess_per_eval = mh_ess / mh_eval_count as f64;

    eprintln!(
        "HMC:  ESS={:.1} evals={} ESS/eval={:.3e}",
        hmc_ess, hmc_eval_count, hmc_ess_per_eval
    );
    eprintln!(
        "MH:   ESS={:.1} evals={} ESS/eval={:.3e}",
        mh_ess, mh_eval_count, mh_ess_per_eval
    );
    eprintln!("ratio HMC/MH = {:.2}", hmc_ess_per_eval / mh_ess_per_eval);

    // Generous margin per the finding: HMC must be at least 2x more efficient
    // per model evaluation.
    assert!(
        hmc_ess_per_eval >= 2.0 * mh_ess_per_eval,
        "HMC ESS/eval {:.3e} should be >= 2x MH ESS/eval {:.3e} (ratio {:.2})",
        hmc_ess_per_eval,
        mh_ess_per_eval,
        hmc_ess_per_eval / mh_ess_per_eval
    );
}
