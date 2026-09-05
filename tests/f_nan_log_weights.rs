//! FG-N2: `NaN` log-weights are sanitised to `-inf` (probability zero) at every
//! accumulation point and in the log-space reductions, and Metropolis-Hastings
//! escapes a state whose density is non-finite instead of freezing on it.
//!
//! Before the fix one `factor(NaN)` made the trace weight `NaN`; in SMC the
//! population normalizer became `NaN`, the ESS non-finite, and the tempering
//! ladder silently jumped to `beta = 1` with uniform (prior) weights and a `NaN`
//! evidence; in MH `NaN >= 0.0` and `u < exp(NaN)` are both false, so a chain
//! started on a `NaN` state rejected every proposal forever.

use fugue::inference::mh::adaptive_single_site_mh;
use fugue::inference::smc::{adaptive_smc, ResamplingMethod, SMCConfig};
use fugue::runtime::handler::run;
use fugue::runtime::interpreters::PriorHandler;
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// x ~ Normal(0, 1), and the negative half-line is scored with a `NaN` factor
/// (an invalid fitness, say). The only well-defined reading is "probability
/// zero", so the target is the half-normal on `x >= 0`:
/// E[x] = sqrt(2/pi) = 0.797885, log Z = ln(1/2).
fn nan_on_negative_half() -> Model<f64> {
    sample(addr!("x"), Normal::new(0.0, 1.0).unwrap()).bind(|x| {
        let w = if x < 0.0 { f64::NAN } else { 0.0 };
        factor(w).map(move |_| x)
    })
}

const HALF_NORMAL_MEAN: f64 = 0.797_884_56;

#[test]
fn fgn2_factor_nan_accumulates_as_neg_inf() {
    let mut rng = StdRng::seed_from_u64(1);
    let (_, t) = run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        factor(f64::NAN),
    );
    assert_eq!(t.log_factors, f64::NEG_INFINITY);
    assert_eq!(t.total_log_weight(), f64::NEG_INFINITY);
    assert!(!t.total_log_weight().is_nan());

    // The observe path: a NaN observed value has NaN log-density under Normal.
    let (_, t) = run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        observe(addr!("y"), Normal::new(0.0, 1.0).unwrap(), f64::NAN),
    );
    assert_eq!(t.log_likelihood, f64::NEG_INFINITY);

    // Finite and -inf weights pass through untouched.
    let (_, t) = run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        factor(-2.5).bind(|_| factor(f64::NEG_INFINITY)),
    );
    assert_eq!(t.log_factors, f64::NEG_INFINITY);
}

#[test]
fn fgn2_mh_chain_escapes_a_nan_start_and_targets_the_half_normal() {
    // Six seeds so that at least some prior initialisations land on x < 0 (the
    // NaN region). Pre-fix such a chain froze there forever with NaN weights.
    let mut started_in_nan_region = 0;
    for seed in 1u64..=6 {
        let mut rng = StdRng::seed_from_u64(seed);
        let (_, init) = run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            nan_on_negative_half(),
        );
        if init.get_f64(&addr!("x")).unwrap() < 0.0 {
            started_in_nan_region += 1;
        }
        // Drive from the exact prior state the chain would have been
        // initialised with, through the public single-step API.
        let mut adaptation = DiminishingAdaptation::new(0.44, 0.7);
        let mut current = init;
        let mut xs = Vec::new();
        for it in 0..6_000 {
            let (_x, t) =
                adaptive_single_site_mh(&mut rng, nan_on_negative_half, &current, &mut adaptation);
            current = t;
            if it >= 1_000 {
                xs.push(current.get_f64(&addr!("x")).unwrap());
            }
        }
        assert!(
            current.total_log_weight().is_finite(),
            "seed {seed}: chain ended on a non-finite state"
        );
        assert!(
            xs.iter().all(|&x| x >= 0.0),
            "seed {seed}: a retained draw sits in the zero-probability region"
        );
        let mean = xs.iter().sum::<f64>() / xs.len() as f64;
        // Half-normal mean 0.798; a frozen chain reports a single value.
        assert!(
            (mean - HALF_NORMAL_MEAN).abs() < 0.12,
            "seed {seed}: mean {mean:.4} vs half-normal {HALF_NORMAL_MEAN}"
        );
    }
    assert!(
        started_in_nan_region > 0,
        "no seed started in the NaN region; the test exercised nothing"
    );
}

#[test]
fn fgn2_mh_full_chain_driver_survives_nan_factor() {
    let mut rng = StdRng::seed_from_u64(20260905);
    let samples = adaptive_mcmc_chain(&mut rng, nan_on_negative_half, 8_000, 2_000);
    assert!(samples
        .iter()
        .all(|(_, t)| t.total_log_weight().is_finite()));
    let xs: Vec<f64> = samples.iter().map(|(x, _)| *x).collect();
    assert!(xs.iter().all(|&x| x >= 0.0));
    let mean = xs.iter().sum::<f64>() / xs.len() as f64;
    assert!(
        (mean - HALF_NORMAL_MEAN).abs() < 0.1,
        "mean {mean:.4} vs half-normal {HALF_NORMAL_MEAN}"
    );
}

#[test]
fn fgn2_smc_does_not_collapse_to_the_prior_on_nan_factor() {
    for rejuvenation_steps in [0usize, 3] {
        let config = SMCConfig {
            resampling_method: ResamplingMethod::Systematic,
            ess_threshold: 0.7,
            rejuvenation_steps,
        };
        let mut rng = StdRng::seed_from_u64(20260905 + rejuvenation_steps as u64);
        let result = adaptive_smc(&mut rng, 4_000, nan_on_negative_half, config);

        assert!(
            result.log_evidence.is_finite(),
            "rejuv={rejuvenation_steps}: log-evidence is {}",
            result.log_evidence
        );
        // log Z = ln P(x >= 0) = ln(1/2). Pre-fix: NaN.
        assert!(
            (result.log_evidence - 0.5_f64.ln()).abs() < 0.1,
            "rejuv={rejuvenation_steps}: log-evidence {:.4} vs ln(1/2) = {:.4}",
            result.log_evidence,
            0.5_f64.ln()
        );
        let total: f64 = result.iter().map(|p| p.weight).sum();
        assert!(
            (total - 1.0).abs() < 1e-9,
            "weights do not sum to 1: {total}"
        );
        assert!(result.iter().all(|p| !p.weight.is_nan()));
        // Particles in the NaN region carry exactly zero weight.
        for p in result.iter() {
            if p.trace.get_f64(&addr!("x")).unwrap() < 0.0 {
                assert_eq!(p.weight, 0.0, "a zero-probability particle kept weight");
            }
        }
        let mean: f64 = result
            .iter()
            .map(|p| p.weight * p.trace.get_f64(&addr!("x")).unwrap())
            .sum();
        // Pre-fix the weights collapsed to uniform over the PRIOR population,
        // whose mean is 0. The half-normal mean is 0.798.
        assert!(
            (mean - HALF_NORMAL_MEAN).abs() < 0.08,
            "rejuv={rejuvenation_steps}: weighted mean {mean:.4} vs {HALF_NORMAL_MEAN}"
        );
    }
}
