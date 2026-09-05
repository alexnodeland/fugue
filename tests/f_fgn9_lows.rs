//! FG-N9 (low): RJMCMC type-change bookkeeping, `ReconcileReport` symmetry,
//! `Address::has_prefix` single-colon edge.

use fugue::runtime::interpreters::score_given_trace_reconciled;
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// The SAME address `v` carries an f64 in one branch and a u64 in the other.
/// Switching `b` is a type change at `v`: the handler samples the new-typed
/// site fresh (a birth) and the old-typed one dies. Pre-fix only the birth was
/// corrected for in the single-site kernel, so `log alpha` for a type-changing
/// move was inflated by `-logp(old v)`. The two priors are chosen with very
/// different entropies (Normal(0, 10): 3.7 nats; Poisson(1): 1.3 nats) so the
/// omission does NOT cancel between the two directions and biases the chain
/// (pre-fix: P(b=1) = 0.147 against the analytic 0.1225, se 0.002); with
/// Normal(0, 1) the two inflations nearly cancel and the pre-fix chain passes
/// this test by accident.
///
/// y = 0.5 observed with sd 1 in both branches:
///   p(y | b=1) = ∫ N(v;0,10) N(0.5;v,1) dv = N(0.5; 0, √101)
///   p(y | b=0) = Σ_k Pois(k;1) N(0.5; k, 1)
fn type_switch_model() -> Model<bool> {
    let y = 0.5;
    sample(addr!("b"), Bernoulli::new(0.5).unwrap()).bind(move |b| {
        if b {
            sample(addr!("v"), Normal::new(0.0, 10.0).unwrap())
                .bind(move |v| observe(addr!("y"), Normal::new(v, 1.0).unwrap(), y).map(move |_| b))
        } else {
            sample(addr!("v"), Poisson::new(1.0).unwrap()).bind(move |k| {
                observe(addr!("y"), Normal::new(k as f64, 1.0).unwrap(), y).map(move |_| b)
            })
        }
    })
}

fn normal_pdf(x: f64, m: f64, sd: f64) -> f64 {
    let z = (x - m) / sd;
    (-0.5 * z * z).exp() / (sd * (2.0 * std::f64::consts::PI).sqrt())
}

fn analytic_p_b1() -> f64 {
    let y = 0.5;
    let l1 = normal_pdf(y, 0.0, 101.0_f64.sqrt());
    let mut l0 = 0.0;
    let mut pois = (-1.0_f64).exp(); // Pois(0; 1)
    for k in 0..40u64 {
        l0 += pois * normal_pdf(y, k as f64, 1.0);
        pois /= (k + 1) as f64; // Pois(k+1; 1) = Pois(k; 1) / (k+1)
    }
    l1 / (l1 + l0)
}

#[test]
fn fgn9_type_change_at_same_address_is_corrected_on_both_sides() {
    let target = analytic_p_b1();
    let mut rng = StdRng::seed_from_u64(20260905);
    let samples = adaptive_mcmc_chain(&mut rng, type_switch_model, 60_000, 6_000);
    let bs: Vec<f64> = samples
        .iter()
        .map(|(b, _)| if *b { 1.0 } else { 0.0 })
        .collect();
    let p_b1 = bs.iter().sum::<f64>() / bs.len() as f64;
    let se = (p_b1 * (1.0 - p_b1) / effective_sample_size_mcmc(&bs)).sqrt();
    // Every retained trace has exactly the two sites, with `v` of the right type.
    for (b, t) in &samples {
        assert_eq!(t.choices.len(), 2);
        if *b {
            assert!(t.get_f64(&addr!("v")).is_some());
        } else {
            assert!(t.get_u64(&addr!("v")).is_some());
        }
    }
    assert!(
        (p_b1 - target).abs() < (4.0 * se).max(0.01) && (p_b1 - target).abs() < 0.04,
        "P(b=1) = {p_b1:.4} vs analytic {target:.4} (se {se:.4}): type-change move is mis-weighted"
    );
}

#[test]
fn fgn9_reconcile_report_lists_a_type_change_as_both_fresh_and_vanished() {
    let mut rng = StdRng::seed_from_u64(1);
    // Base: b = true, v: f64.
    let mut base = Trace::default();
    base.insert_choice(addr!("b"), ChoiceValue::Bool(false), 0.0);
    base.insert_choice(addr!("v"), ChoiceValue::F64(0.3), 0.0);
    // The model with b = false samples v as u64: same address, new type.
    let (_, trace, report) =
        score_given_trace_reconciled(base, &mut rng, type_switch_model()).unwrap();
    assert!(trace.get_u64(&addr!("v")).is_some());
    assert_eq!(report.fresh_addresses, vec![addr!("v")]);
    assert_eq!(
        report.vanished_addresses,
        vec![addr!("v")],
        "the old-typed site must be reported vanished, not silently dropped"
    );
}

#[test]
fn fgn9_has_prefix_single_colon_is_not_a_separator() {
    assert!(Address::new("scope::x").has_prefix("scope"));
    assert!(Address::new("scope::x").has_prefix("scope::"));
    assert!(Address::new("node/0/1").has_prefix("node/0/"));
    assert!(addr!("gene", 3).has_prefix("gene#"));
    assert!(!Address::new("a:b").has_prefix("a:"));
    assert!(!Address::new("scope::x").has_prefix("scope:"));
    assert!(!Address::new("generation").has_prefix("gene"));
}

#[test]
fn fgn9_crossover_mask_is_send() {
    fn assert_send<T: Send>(_: &T) {}
    let kernel = CrossoverKernel {
        n_pairs: 1,
        mask: Box::new(|_a: &Trace, _b: &Trace, _r: &mut dyn rand::RngCore| vec![addr!("x")]),
    };
    assert_send(&kernel);
}
