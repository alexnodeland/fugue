//! X-5: the single-step MH API honours per-address `SiteProposal` overrides and
//! offers a no-rescore variant that reuses the scored current trace.
//!
//! Downstream chains (fugue-evo's `EvolutionChain::step`) drive fugue one
//! transition at a time. Until now only the batch driver
//! (`adaptive_mcmc_chain_with_overrides`) accepted overrides, so `Reflect`
//! bounds could not be applied incrementally, and every step re-scored the
//! current state (two model executions per transition).

use fugue::inference::mh::{
    adaptive_single_site_mh_cached, adaptive_single_site_mh_with_overrides,
};
use fugue::runtime::handler::run;
use fugue::runtime::interpreters::PriorHandler;
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::cell::Cell;
use std::collections::HashMap;

/// Normal(0.5, 2): ~80% of its mass is outside [0, 1].
fn wide_normal() -> Model<f64> {
    sample(addr!("p"), Normal::new(0.5, 2.0).unwrap())
}

#[test]
fn x5_single_step_honours_reflect_override() {
    let mut overrides: HashMap<Address, SiteProposal> = HashMap::new();
    overrides.insert(
        addr!("p"),
        SiteProposal::Reflect {
            lower: 0.0,
            upper: 1.0,
        },
    );
    let mut rng = StdRng::seed_from_u64(5);
    // Start inside [0, 1]: a reflected walk can then never leave it, whereas the
    // support-derived Gaussian default leaves within a handful of steps.
    let mut current = Trace::default();
    current.insert_choice(addr!("p"), ChoiceValue::F64(0.5), 0.0);
    let mut adaptation = DiminishingAdaptation::new(0.44, 0.7);
    for _ in 0..2_000 {
        let (_p, t) = adaptive_single_site_mh_with_overrides(
            &mut rng,
            wide_normal,
            &current,
            &mut adaptation,
            &overrides,
        );
        current = t;
        let p = current.get_f64(&addr!("p")).unwrap();
        assert!(
            (0.0..=1.0).contains(&p),
            "Reflect override ignored by the single step: p = {p}"
        );
    }

    // Control: without the override the same chain leaves [0, 1].
    let mut rng = StdRng::seed_from_u64(5);
    let mut current = Trace::default();
    current.insert_choice(addr!("p"), ChoiceValue::F64(0.5), 0.0);
    let mut adaptation = DiminishingAdaptation::new(0.44, 0.7);
    let none: HashMap<Address, SiteProposal> = HashMap::new();
    let mut left = false;
    for _ in 0..2_000 {
        let (_p, t) = adaptive_single_site_mh_with_overrides(
            &mut rng,
            wide_normal,
            &current,
            &mut adaptation,
            &none,
        );
        current = t;
        if !(0.0..=1.0).contains(&current.get_f64(&addr!("p")).unwrap()) {
            left = true;
            break;
        }
    }
    assert!(
        left,
        "control chain never left [0, 1]; the test is not discriminating"
    );
}

/// Two-site model so target selection and site ordering are exercised.
fn two_site() -> Model<(f64, f64)> {
    sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap()).and_then(|mu| {
        sample(addr!("sigma"), LogNormal::new(0.0, 1.0).unwrap()).and_then(move |s| {
            observe(addr!("y"), Normal::new(mu, s).unwrap(), 0.7).map(move |_| (mu, s))
        })
    })
}

/// The cached step IS the transition the batch driver runs: driving it by hand
/// from the same prior draw, with adaptation on for the warmup and frozen for
/// the sampling phase (FG-57), reproduces `adaptive_mcmc_chain` bit for bit.
#[test]
fn x5_cached_step_reproduces_the_batch_driver_exactly() {
    let (n_warmup, n_samples) = (150usize, 300usize);
    let seed = 0xC0FFEE;
    let batch = adaptive_mcmc_chain(
        &mut StdRng::seed_from_u64(seed),
        two_site,
        n_samples,
        n_warmup,
    );

    let mut rng = StdRng::seed_from_u64(seed);
    let (mut a, mut current) = run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        two_site(),
    );
    let mut adaptation = DiminishingAdaptation::new(0.44, 0.7);
    let overrides: HashMap<Address, SiteProposal> = HashMap::new();
    let mut manual = Vec::with_capacity(n_samples);
    for i in 0..(n_warmup + n_samples) {
        let adapt = i < n_warmup;
        if let Some((a_new, t, lw)) = adaptive_single_site_mh_cached(
            &mut rng,
            two_site,
            &current,
            &mut adaptation,
            &overrides,
            adapt,
        ) {
            assert_eq!(lw, t.total_log_weight(), "returned log-weight is stale");
            a = a_new;
            current = t;
        }
        if i >= n_warmup {
            manual.push((a, current.clone()));
        }
    }

    assert_eq!(manual.len(), batch.len());
    for (i, ((ma, mt), (ba, bt))) in manual.iter().zip(&batch).enumerate() {
        assert_eq!(ma, ba, "draw {i}: value differs from the batch driver");
        assert_eq!(
            mt.total_log_weight(),
            bt.total_log_weight(),
            "draw {i}: trace weight differs from the batch driver"
        );
    }
}

/// Cost contract: the cached step executes the model exactly once per call
/// (the proposal) and never on rejection; the re-scoring variant exactly twice.
#[test]
#[allow(clippy::needless_borrows_for_generic_args)] // &model_fn is reused across loop iterations
fn x5_cached_step_runs_the_model_once_per_transition() {
    let count = Cell::new(0usize);
    let model_fn = || {
        count.set(count.get() + 1);
        two_site()
    };
    let mut rng = StdRng::seed_from_u64(3);
    let (_, mut current) = run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        model_fn(),
    );
    let mut adaptation = DiminishingAdaptation::new(0.44, 0.7);
    let overrides: HashMap<Address, SiteProposal> = HashMap::new();

    count.set(0);
    let steps = 200;
    for _ in 0..steps {
        if let Some((_, t, _)) = adaptive_single_site_mh_cached(
            &mut rng,
            &model_fn,
            &current,
            &mut adaptation,
            &overrides,
            true,
        ) {
            current = t;
        }
    }
    assert_eq!(
        count.get(),
        steps,
        "cached step: not exactly one model run per step"
    );

    count.set(0);
    for _ in 0..steps {
        let (_, t) = adaptive_single_site_mh_with_overrides(
            &mut rng,
            &model_fn,
            &current,
            &mut adaptation,
            &overrides,
        );
        current = t;
    }
    assert_eq!(
        count.get(),
        2 * steps,
        "re-scoring step: not exactly two model runs per step"
    );
}

/// Every trace the re-scoring step returns — accepted or rejected — is a valid
/// input for the cached step: fresh accumulators and per-choice logp.
#[test]
fn x5_returned_traces_are_fully_scored_in_both_branches() {
    let mut rng = StdRng::seed_from_u64(9);
    // Deliberately hand-built with bogus accumulators and logp.
    let mut current = Trace::default();
    current.insert_choice(addr!("mu"), ChoiceValue::F64(0.3), 0.0);
    current.insert_choice(addr!("sigma"), ChoiceValue::F64(1.2), 0.0);
    let mut adaptation = DiminishingAdaptation::new(0.44, 0.7);
    let overrides: HashMap<Address, SiteProposal> = HashMap::new();
    for _ in 0..30 {
        let (_, t) = adaptive_single_site_mh_with_overrides(
            &mut rng,
            two_site,
            &current,
            &mut adaptation,
            &overrides,
        );
        let (_, fresh) = run(
            ScoreGivenTrace {
                base: t.clone(),
                trace: Trace::default(),
            },
            two_site(),
        );
        assert!((t.total_log_weight() - fresh.total_log_weight()).abs() < 1e-12);
        for (addr, c) in &t.choices {
            assert!(
                (c.logp - fresh.choices[addr].logp).abs() < 1e-12,
                "stale per-choice logp at {addr}"
            );
        }
        current = t;
    }
}
