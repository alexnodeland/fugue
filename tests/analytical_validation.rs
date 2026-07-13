//! Exercises the library's dedicated analytical-posterior validation
//! harness (`src/inference/validation.rs`).
//!
//! Covers finding FG-15: `test_conjugate_normal_model` computes the exact
//! Normal-Normal conjugate posterior and checks MCMC output against it, and
//! is publicly re-exported from the crate root — but before this file, it
//! was never called by any test anywhere in the crate (grep for
//! `test_conjugate_normal_model(` / `ConjugateNormalConfig` outside its own
//! definition returned zero call sites). `tests/inference_integration.rs`
//! even had a comment claiming `ConjugateNormalConfig` "isn't exported",
//! which was false — `inference` and `validation` are both `pub mod`, so it
//! was reachable via the full path the whole time, and is now also
//! re-exported at the crate root (`src/lib.rs`).
//!
//! Also exercises the new `test_conjugate_beta_bernoulli_model` /
//! `ConjugateBetaBernoulliConfig` harness added alongside it, so the
//! reusable validation framework covers both textbook conjugate families
//! (an unbounded symmetric posterior, and a bounded skewed one) rather than
//! just Normal-Normal.

use fugue::*;
use rand::{rngs::StdRng, SeedableRng};

#[test]
fn fg15_conjugate_normal_model_harness_is_exercised() {
    let mut rng = StdRng::seed_from_u64(7);

    // Prior: mu ~ Normal(0, 2). Likelihood: y ~ Normal(mu, 1), observed y = 2.5.
    let model_fn = || {
        sample(addr!("mu"), Normal::new(0.0, 2.0).unwrap()).bind(|mu| {
            observe(addr!("y"), Normal::new(mu, 1.0).unwrap(), 2.5).bind(move |_| pure(mu))
        })
    };

    let config = ConjugateNormalConfig {
        prior_mu: 0.0,
        prior_sigma: 2.0,
        likelihood_sigma: 1.0,
        observation: 2.5,
        n_samples: 2000,
        n_warmup: 300,
    };

    let result = test_conjugate_normal_model(
        &mut rng,
        |r, n_samples, n_warmup| adaptive_mcmc_chain(r, model_fn, n_samples, n_warmup),
        config,
    );

    result.print_summary();
    assert!(
        result.is_valid(),
        "FG-15: test_conjugate_normal_model harness reported an invalid MCMC posterior: {result:?}"
    );

    // The harness's own posterior arithmetic should match the textbook
    // Normal-Normal closed form independently derived here:
    // precision_post = 1/2^2 + 1/1^2 = 1.25 -> var_post = 0.8
    // mu_post = 0.8 * (0/4 + 2.5/1) = 2.0
    if let ValidationResult::Success {
        posterior_mu,
        posterior_sigma,
        ..
    } = result
    {
        assert!((posterior_mu - 2.0).abs() < 1e-9);
        assert!((posterior_sigma - 0.8_f64.sqrt()).abs() < 1e-9);
    } else {
        panic!("expected ValidationResult::Success");
    }
}

#[test]
fn fg15_conjugate_beta_bernoulli_model_harness_is_exercised() {
    let mut rng = StdRng::seed_from_u64(11);

    // Prior: theta ~ Beta(2, 2). Likelihood: 12 iid Bernoulli(theta) draws,
    // 9 successes, 3 failures -> exact posterior Beta(11, 5).
    let observations = vec![
        true, true, true, true, true, true, true, true, true, false, false, false,
    ];

    let model_fn = {
        let observations = observations.clone();
        move || {
            let indexed: Vec<(u64, bool)> = observations
                .iter()
                .enumerate()
                .map(|(i, &o)| (i as u64, o))
                .collect();
            sample(addr!("theta"), Beta::new(2.0, 2.0).unwrap()).bind(move |theta| {
                let valid_theta = theta.clamp(1e-6, 1.0 - 1e-6);
                traverse_vec(indexed.clone(), move |(i, o)| {
                    observe(addr!("obs", i), Bernoulli::new(valid_theta).unwrap(), o)
                })
                .bind(move |_| pure(theta))
            })
        }
    };

    let config = ConjugateBetaBernoulliConfig {
        prior_alpha: 2.0,
        prior_beta: 2.0,
        observations,
        n_samples: 2000,
        n_warmup: 300,
    };

    let result = test_conjugate_beta_bernoulli_model(
        &mut rng,
        |r, n_samples, n_warmup| adaptive_mcmc_chain(r, &model_fn, n_samples, n_warmup),
        config,
    );

    result.print_summary();
    assert!(
        result.is_valid(),
        "FG-15: test_conjugate_beta_bernoulli_model harness reported an invalid MCMC posterior: {result:?}"
    );

    // Posterior Beta(2+9, 2+3) = Beta(11, 5): mean = 11/16 = 0.6875.
    if let ValidationResult::Success { posterior_mu, .. } = result {
        assert!((posterior_mu - 11.0 / 16.0).abs() < 1e-9);
    } else {
        panic!("expected ValidationResult::Success");
    }
}
