//! Known-answer regressions for the MCMC diagnostics (audit findings FG-01,
//! FG-35, FG-36, FG-37). Seeded; tolerances justified inline.

use fugue::runtime::trace::{ChoiceValue, Trace};
use fugue::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Build a chain of traces carrying scalar `values` at address `x`.
fn traces_from(values: &[f64]) -> Vec<Trace> {
    values
        .iter()
        .map(|&v| {
            let mut t = Trace::default();
            t.insert_choice(addr!("x"), ChoiceValue::F64(v), 0.0);
            t
        })
        .collect()
}

/// One standard-normal draw (Box-Muller) from a seeded RNG.
fn z(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.gen::<f64>().max(1e-12);
    let u2: f64 = rng.gen();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

// FG-35 / FG-36: two independent chains drawn from the SAME stationary
// distribution give split-R-hat ≈ 1.
#[test]
fn fg36_identical_distribution_chains_split_rhat_near_one() {
    let mut rng = StdRng::seed_from_u64(101);
    let n = 600;
    let c1: Vec<f64> = (0..n).map(|_| z(&mut rng)).collect();
    let c2: Vec<f64> = (0..n).map(|_| z(&mut rng)).collect();
    let chains = vec![traces_from(&c1), traces_from(&c2)];

    let split = r_hat_f64(&chains, &addr!("x"));
    // Both chains are stationary and identically distributed → split-R-hat < 1.01.
    assert!(
        split < 1.01,
        "split-R-hat {split:.4} should be < 1.01 for identical-distribution chains"
    );
}

// FG-35 / FG-36: two chains that both drift the SAME way have nearly-equal chain
// means, so classic (1992) R-hat stays ≈ 1 and misses the non-stationarity — but
// split-R-hat halves each chain and detects the within-chain trend (> 1.1). This
// is the exact failure mode split-R-hat exists to catch.
#[test]
fn fg36_within_chain_drift_only_caught_by_split() {
    let mut rng = StdRng::seed_from_u64(202);
    let n = 600;
    // Linear drift shared by both chains, plus small independent noise so the
    // chains are not bit-identical but their means nearly coincide.
    let drift = |i: usize, rng: &mut StdRng| 0.02 * i as f64 + 0.2 * z(rng);
    let c1: Vec<f64> = (0..n).map(|i| drift(i, &mut rng)).collect();
    let c2: Vec<f64> = (0..n).map(|i| drift(i, &mut rng)).collect();
    let chains = vec![traces_from(&c1), traces_from(&c2)];

    let classic = classic_r_hat_f64(&chains, &addr!("x"));
    let split = r_hat_f64(&chains, &addr!("x"));

    assert!(
        classic < 1.01,
        "classic R-hat {classic:.4} should stay < 1.01 (it cannot see the shared within-chain drift)"
    );
    assert!(
        split > 1.1,
        "split-R-hat {split:.4} should exceed 1.1, flagging the within-chain drift"
    );
}

// FG-37: summarize_f64_parameter must compute ESS across ALL chains, not just the
// first. With M independent iid chains each of ESS ≈ n, the multi-chain ESS is
// ≈ M·n, far larger than a single chain's n. The pre-fix code reported only the
// first chain's ESS.
#[test]
fn fg37_summary_ess_uses_all_chains() {
    let mut rng = StdRng::seed_from_u64(303);
    let n = 500;
    let m = 4;
    let chains: Vec<Vec<Trace>> = (0..m)
        .map(|_| {
            let vals: Vec<f64> = (0..n).map(|_| z(&mut rng)).collect();
            traces_from(&vals)
        })
        .collect();

    let summary = summarize_f64_parameter(&chains, &addr!("x"));
    // ESS of just the first chain, for comparison.
    let first_only =
        effective_sample_size_multichain(&[extract_f64_values(&chains[0], &addr!("x"))]);

    // The pooled ESS must clearly exceed a single chain's (it should be roughly
    // m× larger); require at least 1.8× the single-chain value.
    assert!(
        summary.ess > 1.8 * first_only,
        "summary ESS {:.1} should exceed 1.8× single-chain ESS {:.1}",
        summary.ess,
        first_only
    );
    // And it must not exceed the total number of draws (m·n) by more than noise.
    assert!(
        summary.ess <= (m * n) as f64 * 1.05,
        "summary ESS {:.1} exceeds total draws {}",
        summary.ess,
        m * n
    );
}

// FG-01 end-to-end: ESS reported by summarize is invariant to rescaling the
// parameter. The pre-fix diagnostics estimator (raw autocovariances) scaled ESS
// with the parameter variance; the routed normalized estimator does not.
#[test]
fn fg01_summary_ess_scale_invariant() {
    let mut rng = StdRng::seed_from_u64(404);
    let n = 1500;
    // A correlated (AR(1)) chain so ESS < n and the bug would have bitten.
    let mut x = 0.0;
    let base: Vec<f64> = (0..n)
        .map(|_| {
            x = 0.7 * x + z(&mut rng);
            x
        })
        .collect();
    let scaled: Vec<f64> = base.iter().map(|&v| v * 500.0).collect();

    let s_base = summarize_f64_parameter(&[traces_from(&base)], &addr!("x"));
    let s_scaled = summarize_f64_parameter(&[traces_from(&scaled)], &addr!("x"));

    let rel = (s_base.ess - s_scaled.ess).abs() / s_base.ess;
    assert!(
        rel < 1e-9,
        "summary ESS not scale-invariant: {} vs {}",
        s_base.ess,
        s_scaled.ess
    );
}
