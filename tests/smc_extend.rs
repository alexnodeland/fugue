//! `smc_extend`: moving a weighted particle population to a new target.
//!
//! The model is conjugate (a Normal mean with known noise), so every evidence
//! here has a closed form from `normal_normal_log_marginal`:
//!
//! - evidence over two batches equals the single-pass evidence, exactly for a
//!   pure reweight from the same particles and within Monte Carlo error with
//!   resampling and rejuvenation, and both equal the closed form;
//! - the increment is the closed-form predictive of the new batch;
//! - a renamed population under a renamed model, and a model with one more
//!   prior-only site, have an increment of exactly zero;
//! - new sites, dropped sites and changed priors are weighted correctly.

use fugue::runtime::handler::run;
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

const MU0: f64 = 0.0;
const SIGMA0: f64 = 2.0;
const SIGMA: f64 = 1.0;

fn batch_one() -> Vec<f64> {
    vec![0.9, 1.4, 0.2, 1.7, 0.6]
}
fn batch_two() -> Vec<f64> {
    vec![1.1, 0.7, 1.9, 0.3, 1.2, 0.8]
}
fn batch_three() -> Vec<f64> {
    vec![1.5, 0.1, 1.0]
}
fn concat(parts: &[Vec<f64>]) -> Vec<f64> {
    parts.iter().flatten().copied().collect()
}

/// `mu ~ Normal(MU0, SIGMA0)`, `y_i ~ Normal(mu, SIGMA)`, sites under `root`.
fn mean_model_at(root: &'static str, data: Vec<f64>) -> Model<f64> {
    sample(
        Address::new(format!("{root}/mu")),
        Normal::new(MU0, SIGMA0).unwrap(),
    )
    .bind(move |mu| {
        traverse_vec(data.into_iter().enumerate().collect(), move |(i, y)| {
            observe(
                Address::new(format!("{root}/y#{i}")),
                Normal::new(mu, SIGMA).unwrap(),
                y,
            )
        })
        .map(move |_| mu)
    })
}

fn mean_model(data: Vec<f64>) -> Model<f64> {
    mean_model_at("m", data)
}

fn exact(data: &[f64]) -> f64 {
    normal_normal_log_marginal(MU0, SIGMA0, SIGMA, data).unwrap()
}

fn never_resample() -> SMCConfig {
    SMCConfig {
        ess_threshold: 0.0,
        ..Default::default()
    }
}

fn moving() -> SMCConfig {
    SMCConfig {
        resampling_method: ResamplingMethod::Systematic,
        ess_threshold: 0.5,
        rejuvenation_steps: 3,
    }
}

fn weighted_mean(particles: &[Particle], addr: &Address) -> f64 {
    particles
        .iter()
        .map(|p| p.weight * p.trace.get_f64(addr).unwrap())
        .sum()
}

fn assert_normalized(particles: &[Particle]) {
    let total: f64 = particles.iter().map(|p| p.weight).sum();
    assert!((total - 1.0).abs() < 1e-9, "weights sum to {total}");
    for p in particles {
        assert!((p.weight - p.log_weight.exp()).abs() < 1e-12);
    }
}

// ---------------------------------------------------------------------------
// Evidence over two batches
// ---------------------------------------------------------------------------

/// Pure reweighting, same particles: the two-batch estimate *is* the
/// single-pass importance-sampling estimate, `mean(L1·L2)`, up to rounding,
/// and so are the final weights.
#[test]
fn two_batch_reweight_equals_single_pass_from_the_same_particles() {
    let (b1, all) = (batch_one(), concat(&[batch_one(), batch_two()]));
    for seed in 0..5 {
        let mut rng = StdRng::seed_from_u64(seed);
        let start = adaptive_smc(
            &mut rng,
            500,
            || mean_model(b1.clone()),
            SMCConfig::default(),
        );
        let step = smc_extend(
            &mut rng,
            start.particles,
            || mean_model(all.clone()),
            &never_resample(),
        )
        .unwrap();
        assert!(!step.resampled);
        assert!(step.fresh_addresses.is_empty() && step.vanished_addresses.is_empty());

        // The same prior draws, weighted in one pass.
        let mut rng = StdRng::seed_from_u64(seed);
        let single = adaptive_smc(
            &mut rng,
            500,
            || mean_model(all.clone()),
            SMCConfig::default(),
        );

        let two_batch = start.log_evidence + step.log_evidence_increment;
        assert!(
            (two_batch - single.log_evidence).abs() < 1e-10,
            "two-batch {two_batch} vs single-pass {}",
            single.log_evidence
        );
        assert_normalized(&step.particles);
        for (a, b) in step.particles.iter().zip(&single.particles) {
            assert_eq!(
                a.trace.get_f64(&addr!("m/mu")),
                b.trace.get_f64(&addr!("m/mu"))
            );
            assert!((a.weight - b.weight).abs() < 1e-12);
            // Traces come back scored under the new model.
            assert_eq!(a.trace.log_likelihood, b.trace.log_likelihood);
        }
    }
}

/// With resampling and rejuvenation the two estimates use different
/// randomness; both must agree with the closed form, and with each other,
/// within Monte Carlo error. Averaged over seeds to make the bound tight.
#[test]
fn two_batch_evidence_matches_single_pass_and_closed_form() {
    let (b1, b2) = (batch_one(), batch_two());
    let all = concat(&[b1.clone(), b2.clone()]);
    let (z1, z_all) = (exact(&b1), exact(&all));
    let n = 3000;
    let seeds = 8;

    let (mut two_batch, mut single_pass, mut increments) = (Vec::new(), Vec::new(), Vec::new());
    let mut resampled_any = false;
    for seed in 0..seeds {
        let mut rng = StdRng::seed_from_u64(1000 + seed);
        let start = adaptive_smc(&mut rng, n, || mean_model(b1.clone()), moving());
        let step = smc_extend(
            &mut rng,
            start.particles,
            || mean_model(all.clone()),
            &moving(),
        )
        .unwrap();
        resampled_any |= step.resampled;
        assert!(step.ess >= 1.0 && step.ess <= n as f64);
        assert_normalized(&step.particles);
        two_batch.push(start.log_evidence + step.log_evidence_increment);
        increments.push(step.log_evidence_increment);

        // The extended population is the posterior given all the data.
        let (mu_n, _) = normal_normal_posterior(MU0, SIGMA0, SIGMA, &all).unwrap();
        assert!((weighted_mean(&step.particles, &addr!("m/mu")) - mu_n).abs() < 0.05);

        let mut rng = StdRng::seed_from_u64(2000 + seed);
        let single = adaptive_smc(&mut rng, n, || mean_model(all.clone()), moving());
        single_pass.push(single.log_evidence);
    }
    assert!(resampled_any, "the test should exercise the resample path");

    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let sd = |v: &[f64]| {
        let m = mean(v);
        (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
    };
    // Each mean is within 4 standard errors of the closed form (with a floor
    // so a lucky tiny spread cannot make the bound vacuous or impossible) ...
    let bound = |v: &[f64]| (4.0 * sd(v) / (v.len() as f64).sqrt()).max(0.01);
    assert!(
        (mean(&two_batch) - z_all).abs() < bound(&two_batch),
        "two-batch {} vs exact {z_all} (bound {})",
        mean(&two_batch),
        bound(&two_batch)
    );
    assert!(
        (mean(&single_pass) - z_all).abs() < bound(&single_pass),
        "single-pass {} vs exact {z_all}",
        mean(&single_pass)
    );
    // ... the two agree with each other within their combined error ...
    assert!(
        (mean(&two_batch) - mean(&single_pass)).abs() < bound(&two_batch) + bound(&single_pass)
    );
    // ... and the increment is the predictive of the second batch. (A fixed
    // tolerance: the increment's error includes that of the population it
    // started from, which its own spread across seeds does not measure.)
    assert!((mean(&increments) - (z_all - z1)).abs() < 0.05);
    // No single run is far off either.
    for z in two_batch.iter().chain(&single_pass) {
        assert!((z - z_all).abs() < 0.15, "run {z} vs exact {z_all}");
    }
}

/// Evidence is additive over any number of epochs, starting from the prior.
#[test]
fn increments_over_three_epochs_sum_to_the_evidence() {
    let batches = [batch_one(), batch_two(), batch_three()];
    let mut rng = StdRng::seed_from_u64(77);
    // Epoch zero: the prior, as a population (no data, evidence 1).
    let mut particles = smc_prior_particles(&mut rng, 4000, || mean_model(Vec::new()));
    let mut seen: Vec<f64> = Vec::new();
    let mut log_evidence = 0.0;
    for batch in &batches {
        let before = exact(&seen);
        seen.extend(batch);
        let data = seen.clone();
        let step = smc_extend(&mut rng, particles, || mean_model(data.clone()), &moving()).unwrap();
        // Each increment is that epoch's closed-form predictive.
        assert!(
            (step.log_evidence_increment - (exact(&seen) - before)).abs() < 0.15,
            "epoch increment {} vs exact {}",
            step.log_evidence_increment,
            exact(&seen) - before
        );
        log_evidence += step.log_evidence_increment;
        particles = step.particles;
    }
    assert!((log_evidence - exact(&seen)).abs() < 0.2);
    let (mu_n, _) = normal_normal_posterior(MU0, SIGMA0, SIGMA, &seen).unwrap();
    assert!((weighted_mean(&particles, &addr!("m/mu")) - mu_n).abs() < 0.06);
}

// ---------------------------------------------------------------------------
// Moves that change the model, not the data
// ---------------------------------------------------------------------------

/// A renamed population under the renamed model is the same target: the
/// increment is exactly zero and no weight changes, per particle.
#[test]
fn renaming_is_an_exact_no_op() {
    let data = batch_one();
    let mut rng = StdRng::seed_from_u64(3);
    let start = adaptive_smc(
        &mut rng,
        300,
        || mean_model(data.clone()),
        SMCConfig::default(),
    );
    let renamed: Vec<Particle> = start
        .particles
        .iter()
        .map(|p| Particle {
            trace: p.trace.rename_prefix("m", "moved").unwrap(),
            weight: p.weight,
            log_weight: p.log_weight,
        })
        .collect();
    let step = smc_extend(
        &mut rng,
        renamed,
        || mean_model_at("moved", data.clone()),
        &never_resample(),
    )
    .unwrap();
    assert_eq!(step.log_evidence_increment, 0.0);
    assert!(step.fresh_addresses.is_empty() && step.vanished_addresses.is_empty());
    for (after, before) in step.particles.iter().zip(&start.particles) {
        assert!((after.log_weight - before.log_weight).abs() < 1e-12);
        assert_eq!(
            after.trace.get_f64(&Address::new("moved/mu")),
            before.trace.get_f64(&Address::new("m/mu"))
        );
        assert_eq!(
            after.trace.total_log_weight(),
            before.trace.total_log_weight()
        );
    }
    assert!((step.ess - effective_sample_size(&start.particles)).abs() < 1e-6);
}

/// Conservative lift: a new site with a prior and nothing observed below it
/// changes no evidence. The increment is exactly zero, per particle, and the
/// new site is reported and filled from its prior.
#[test]
fn a_new_prior_only_site_has_zero_increment() {
    let data = batch_one();
    let mut rng = StdRng::seed_from_u64(4);
    let start = adaptive_smc(
        &mut rng,
        300,
        || mean_model(data.clone()),
        SMCConfig::default(),
    );
    let lifted = || {
        mean_model(data.clone())
            .bind(|mu| sample(addr!("extra"), Gamma::new(2.0, 1.0).unwrap()).map(move |g| (mu, g)))
    };
    let step = smc_extend(&mut rng, start.particles.clone(), lifted, &never_resample()).unwrap();
    assert_eq!(step.log_evidence_increment, 0.0);
    assert_eq!(step.fresh_addresses, vec![addr!("extra")]);
    assert!(step.vanished_addresses.is_empty());
    let mut extras = Vec::new();
    for (after, before) in step.particles.iter().zip(&start.particles) {
        assert!((after.log_weight - before.log_weight).abs() < 1e-12);
        extras.push(after.trace.get_f64(&addr!("extra")).unwrap());
    }
    // Drawn from Gamma(2, 1): mean 2, and not one repeated value.
    let mean = extras.iter().sum::<f64>() / extras.len() as f64;
    assert!((mean - 2.0).abs() < 0.3);
    assert!(extras.iter().any(|g| (g - extras[0]).abs() > 1e-6));

    // Dropping it again is also free: its conditional prior is the backward
    // kernel, so the weight is again exactly one.
    let back = smc_extend(
        &mut rng,
        step.particles,
        || mean_model(data.clone()),
        &never_resample(),
    )
    .unwrap();
    assert_eq!(back.log_evidence_increment, 0.0);
    assert_eq!(back.vanished_addresses, vec![addr!("extra")]);
    assert!(back.fresh_addresses.is_empty());
    for (after, before) in back.particles.iter().zip(&start.particles) {
        assert!((after.log_weight - before.log_weight).abs() < 1e-12);
        assert!(after.trace.get_f64(&addr!("extra")).is_none());
    }
}

/// A new latent with its own observations: the increment is the evidence of
/// the new, independent block, here in closed form.
#[test]
fn a_new_observed_block_adds_its_own_evidence() {
    let (old_data, new_data) = (batch_one(), vec![-0.4, 0.3, -0.9]);
    let (od, nd) = (old_data.clone(), new_data.clone());
    let two_blocks = move || {
        let nd = nd.clone();
        mean_model_at("m", od.clone())
            .bind(move |a| mean_model_at("second", nd.clone()).map(move |b| (a, b)))
    };
    let mut increments = Vec::new();
    for seed in 0..6 {
        let mut rng = StdRng::seed_from_u64(50 + seed);
        let start = adaptive_smc(&mut rng, 4000, || mean_model(old_data.clone()), moving());
        let step = smc_extend(&mut rng, start.particles, &two_blocks, &moving()).unwrap();
        assert_eq!(step.fresh_addresses, vec![Address::new("second/mu")]);
        increments.push(step.log_evidence_increment);
    }
    let mean = increments.iter().sum::<f64>() / increments.len() as f64;
    assert!(
        (mean - exact(&new_data)).abs() < 0.05,
        "increment {mean} vs exact {}",
        exact(&new_data)
    );
}

/// A changed prior on a shared site is part of the weight: the increment is
/// the log ratio of the two evidences.
#[test]
fn a_changed_prior_is_weighted() {
    let data = batch_one();
    let d = data.clone();
    let wide_prior = move || {
        let d = d.clone();
        sample(Address::new("m/mu"), Normal::new(0.5, 4.0).unwrap()).bind(move |mu| {
            traverse_vec(d.into_iter().enumerate().collect(), move |(i, y)| {
                observe(
                    Address::new(format!("m/y#{i}")),
                    Normal::new(mu, SIGMA).unwrap(),
                    y,
                )
            })
            .map(move |_| mu)
        })
    };
    let want = normal_normal_log_marginal(0.5, 4.0, SIGMA, &data).unwrap() - exact(&data);
    let mut increments = Vec::new();
    for seed in 0..6 {
        let mut rng = StdRng::seed_from_u64(90 + seed);
        let start = adaptive_smc(&mut rng, 4000, || mean_model(data.clone()), moving());
        let step = smc_extend(&mut rng, start.particles, &wide_prior, &never_resample()).unwrap();
        increments.push(step.log_evidence_increment);
    }
    let mean = increments.iter().sum::<f64>() / increments.len() as f64;
    assert!(
        (mean - want).abs() < 0.03,
        "increment {mean} vs exact {want}"
    );
}

// ---------------------------------------------------------------------------
// Mechanics and edge cases
// ---------------------------------------------------------------------------

#[test]
fn resampling_follows_the_ess_threshold() {
    let (b1, all) = (batch_one(), concat(&[batch_one(), batch_two()]));
    let start = |rng: &mut StdRng| {
        adaptive_smc(rng, 400, || mean_model(b1.clone()), SMCConfig::default()).particles
    };
    let n = 400.0;

    let mut rng = StdRng::seed_from_u64(6);
    let population = start(&mut rng);
    let kept = smc_extend(
        &mut rng,
        population,
        || mean_model(all.clone()),
        &never_resample(),
    )
    .unwrap();
    assert!(!kept.resampled);
    assert!(kept.ess < n);
    assert!((effective_sample_size(&kept.particles) - kept.ess).abs() < 1e-6);

    let always = SMCConfig {
        ess_threshold: f64::INFINITY,
        rejuvenation_steps: 2,
        resampling_method: ResamplingMethod::Stratified,
    };
    let mut rng = StdRng::seed_from_u64(6);
    let population = start(&mut rng);
    let moved = smc_extend(&mut rng, population, || mean_model(all.clone()), &always).unwrap();
    assert!(moved.resampled);
    // The ESS reported is the one that triggered the resample ...
    assert!((moved.ess - kept.ess).abs() < 1e-9);
    // ... and the weights are now uniform.
    for p in &moved.particles {
        assert!((p.weight - 1.0 / n).abs() < 1e-15);
    }
    // The evidence increment does not depend on what happens after it.
    assert_eq!(moved.log_evidence_increment, kept.log_evidence_increment);
    // Rejuvenated traces are still fully scored under the new model.
    for p in &moved.particles {
        let (_, rescored) = run(
            ScoreGivenTrace {
                base: p.trace.clone(),
                trace: Trace::default(),
            },
            mean_model(all.clone()),
        );
        assert_eq!(rescored.log_likelihood, p.trace.log_likelihood);
        assert_eq!(rescored.log_prior, p.trace.log_prior);
    }
}

#[test]
fn incoming_weights_need_not_be_normalized() {
    let (b1, all) = (batch_one(), concat(&[batch_one(), batch_two()]));
    let mut rng = StdRng::seed_from_u64(8);
    let start = adaptive_smc(
        &mut rng,
        200,
        || mean_model(b1.clone()),
        SMCConfig::default(),
    );
    let shifted: Vec<Particle> = start
        .particles
        .iter()
        .map(|p| Particle {
            trace: p.trace.clone(),
            weight: 123.0, // ignored
            log_weight: p.log_weight + 17.0,
        })
        .collect();
    let a = smc_extend(
        &mut rng,
        start.particles,
        || mean_model(all.clone()),
        &never_resample(),
    )
    .unwrap();
    let b = smc_extend(
        &mut rng,
        shifted,
        || mean_model(all.clone()),
        &never_resample(),
    )
    .unwrap();
    assert!((a.log_evidence_increment - b.log_evidence_increment).abs() < 1e-10);
    assert_normalized(&b.particles);
}

#[test]
fn degenerate_populations() {
    let mut rng = StdRng::seed_from_u64(9);
    // Empty in, empty out.
    let empty = smc_extend(
        &mut rng,
        Vec::new(),
        || mean_model(batch_one()),
        &SMCConfig::default(),
    )
    .unwrap();
    assert!(empty.particles.is_empty());
    assert_eq!(empty.log_evidence_increment, 0.0);
    assert_eq!(empty.ess, 0.0);
    assert!(!empty.resampled);

    // A new target no particle supports: zero evidence, uniform weights, no
    // resample, no NaN.
    let start = smc_prior_particles(&mut rng, 50, || mean_model(batch_one()));
    let impossible =
        || mean_model(batch_one()).bind(|mu| factor(f64::NEG_INFINITY).map(move |_| mu));
    let dead = smc_extend(&mut rng, start.clone(), impossible, &moving()).unwrap();
    assert_eq!(dead.log_evidence_increment, f64::NEG_INFINITY);
    assert!(!dead.resampled);
    assert_eq!(dead.particles.len(), 50);
    for p in &dead.particles {
        assert!((p.weight - 0.02).abs() < 1e-15);
        assert!(p.log_weight.is_finite());
    }

    // Part of the population outside the new support: those particles get
    // zero weight, the rest carry the estimate.
    let half = || mean_model(batch_one()).bind(|mu| guard(mu > 0.9).map(move |_| mu));
    let cut = smc_extend(&mut rng, start, half, &never_resample()).unwrap();
    assert!(cut.log_evidence_increment.is_finite() && cut.log_evidence_increment < 0.0);
    assert_normalized(&cut.particles);
    for p in &cut.particles {
        let mu = p.trace.get_f64(&addr!("m/mu")).unwrap();
        assert_eq!(p.weight > 0.0, mu > 0.9);
    }

    // A model that samples one address twice is an error, not a panic.
    let mut rng = StdRng::seed_from_u64(10);
    let start = smc_prior_particles(&mut rng, 5, || mean_model(batch_one()));
    let twice = || {
        sample(addr!("m/mu"), Normal::new(0.0, 1.0).unwrap())
            .bind(|_| sample(addr!("m/mu"), Normal::new(0.0, 1.0).unwrap()))
    };
    let err = smc_extend(&mut rng, start, twice, &SMCConfig::default()).unwrap_err();
    assert_eq!(err.code(), ErrorCode::AddressConflict);
}
