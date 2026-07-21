//! Boundary tests, run in a real wasm runtime via `wasm-pack test --node`.

#![cfg(target_arch = "wasm32")]

use fugue_wasm::*;
use wasm_bindgen_test::*;

const COIN: &str = r#"
    let p <- sample(addr!("p"), Beta(2.0, 2.0));
    for i in 0..data.len() {
        observe(addr!("flip", i), Bernoulli(p), data[i]);
    }
    pure(p)
"#;
const COIN_DATA: &str = "[1,0,1,1,0,1,1,0,1,1]"; // 7 heads, 3 tails

#[wasm_bindgen_test]
fn check_model_reports_errors_and_ok() {
    assert_eq!(check_model(COIN, COIN_DATA), "");
    let err = check_model("pure(nope)", "");
    assert!(err.contains("nope"), "{err}");
}

#[wasm_bindgen_test]
fn mh_recovers_conjugate_posterior_in_wasm() {
    let mut mh = WasmMh::new(COIN, COIN_DATA, 3, 11).unwrap();
    mh.step(3000);
    let sites = mh.site_names();
    assert_eq!(sites, vec!["p".to_string()]);
    let s = mh.summary("p"); // [mean, std, r_hat, ess]
    assert!((s[0] - 9.0 / 14.0).abs() < 0.05, "mean {}", s[0]);
    assert!(s[2] < 1.1, "r_hat {}", s[2]);
    assert!(s[3] > 50.0, "ess {}", s[3]);
    let acc = mh.acceptance_rate();
    assert!(acc > 0.05 && acc < 0.95, "acceptance {acc}");
    // Determinism: same seed, same draws.
    let mut mh2 = WasmMh::new(COIN, COIN_DATA, 3, 11).unwrap();
    mh2.step(100);
    assert_eq!(mh.values_since("p", 0, 0)[..100], mh2.values_since("p", 0, 0)[..]);
}

#[wasm_bindgen_test]
fn hmc_streams_trajectories() {
    let src = r#"
        let mu <- sample(addr!("mu"), Normal(0.0, 2.0));
        for i in 0..data.len() {
            observe(addr!("y", i), Normal(mu, 1.0), data[i]);
        }
        pure(mu)
    "#;
    let mut hmc = WasmHmc::new(src, "[1.3, 0.7, 2.1, 0.4, 1.5]", 7, 30, 12, 0.0).unwrap();
    hmc.step(30); // warmup
    assert!(!hmc.is_warming_up());
    let json = hmc.step_recorded();
    assert!(json.contains("\"trajectory_q\""), "{json}");
    assert!(json.contains("\"accept_prob\""), "{json}");
    hmc.step(200);
    let vals = hmc.values("mu");
    assert!(vals.len() >= 200);
    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
    // Conjugate posterior mean = (sum y / sigma^2) / (1/tau^2 + n/sigma^2) ~ 1.17
    assert!((mean - 1.17).abs() < 0.4, "mean {mean}");
}

#[wasm_bindgen_test]
fn particle_filter_tracks_and_reports() {
    let mut pf = WasmParticleFilter::new(200, 1.6, 0.7, 0.6, 0.5, 11).unwrap();
    let mut resampled_any = false;
    for t in 0..10 {
        let obs = (t as f64) * 0.3;
        let json = pf.step(obs);
        assert!(json.contains("\"posterior\""), "{json}");
        if json.contains("\"resampled\":true") {
            resampled_any = true;
        }
    }
    assert_eq!(pf.t(), 10);
    assert!(pf.log_evidence().is_finite());
    assert!(resampled_any, "expected at least one resampling in 10 steps");
}

#[wasm_bindgen_test]
fn smc_run_returns_evidence() {
    let json = wasm_smc_run(COIN, COIN_DATA, 300, 2, 11).unwrap();
    assert!(json.contains("\"log_evidence\""), "{json}");
    assert!(json.contains("\"p\""), "{json}");
}
