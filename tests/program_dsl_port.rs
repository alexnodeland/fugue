//! The playground interpreter's tests (formerly `crates/fugue-wasm/src/dsl.rs`),
//! ported to `fugue::program` (issue #65). Each test states the same property
//! against the new API; fugue-wasm keeps the originals, run through its shim.

#![cfg(feature = "program")]

use fugue::program::{CompiledProgram, Data, Program, ProgramError, Registry};
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

const COIN: &str = r#"
    let p <- sample(addr!("p"), Beta(2.0, 2.0));
    for i in 0..data.len() {
        observe(addr!("flip", i), Bernoulli(p), data[i]);
    }
    pure(p)
"#;

fn compile(src: &str, data_json: &str) -> Result<CompiledProgram, ProgramError> {
    Program::parse(src)?.compile(&Registry::new(), &Data::from_json(data_json)?)
}

fn prior_run<A>(model: Model<A>, seed: u64) -> (A, Trace) {
    let mut rng = StdRng::seed_from_u64(seed);
    runtime::handler::run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        model,
    )
}

#[test]
fn coin_model_matches_native_addresses_and_posterior() {
    let data = "[1,0,1,1,0,1,1,0,1,1]"; // 7 heads, 3 tails
    let cm = compile(COIN, data).unwrap();

    // Address parity with the addr! macro.
    let (_, t) = prior_run(cm.build(), 11);
    assert!(t.get_f64(&addr!("p")).is_some());
    // Observations are scored, not recorded as choices.
    assert_eq!(t.choices.len(), 1);
    assert!(t.log_likelihood.is_finite() && t.log_likelihood < 0.0);

    // Real inference on the interpreted model recovers the conjugate
    // posterior mean Beta(2+7, 2+3) => 9/14.
    let mut rng = StdRng::seed_from_u64(11);
    let samples = adaptive_mcmc_chain(&mut rng, || cm.build_f64(), 4000, 1000);
    let ps: Vec<f64> = samples
        .iter()
        .filter_map(|(_, t)| t.get_f64(&addr!("p")))
        .collect();
    let mean = ps.iter().sum::<f64>() / ps.len() as f64;
    assert!((mean - 9.0 / 14.0).abs() < 0.03, "mean {mean}");
}

#[test]
fn regression_model_with_named_arrays() {
    let src = r#"
        let a <- sample(addr!("a"), Normal(0.0, 2.5));
        let b <- sample(addr!("b"), Normal(0.0, 2.5));
        for i in 0..x.len() {
            observe(addr!("y", i), Normal(a * x[i] + b, 0.8), y[i]);
        }
        pure(a)
    "#;
    let data = r#"{"x": [-2, -1, 0, 1, 2], "y": [-2.1, -1.3, -0.4, 0.5, 1.2]}"#;
    let cm = compile(src, data).unwrap();
    let mut rng = StdRng::seed_from_u64(3);
    let samples = adaptive_mcmc_chain(&mut rng, || cm.build(), 2000, 500);
    let a_mean: f64 = samples
        .iter()
        .filter_map(|(_, t)| t.get_f64(&addr!("a")))
        .sum::<f64>()
        / samples.len() as f64;
    // True slope ~0.85 for this tiny dataset; just check sanity.
    assert!(a_mean > 0.4 && a_mean < 1.4, "a_mean {a_mean}");
}

#[test]
fn indexed_sample_addresses_match_addr_macro() {
    let src = r#"
        for i in 0..3 {
            let z <- sample(addr!("z", i), Normal(0.0, 1.0));
        }
        pure(0.0)
    "#;
    let cm = compile(src, "").unwrap();
    let (_, t) = prior_run(cm.build(), 1);
    assert!(t.choices.contains_key(&addr!("z", 0)));
    assert!(t.choices.contains_key(&addr!("z", 2)));
    assert_eq!(t.choices.len(), 3);
}

#[test]
fn rust_ish_sugar_parses() {
    let src = r#"
        let mu <- sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap());
        observe(addr!("y"), Normal::new(mu, 1.0).unwrap(), 0.5);
        pure(mu)
    "#;
    assert!(compile(src, "").is_ok());
}

#[test]
fn static_errors_are_caught() {
    assert!(compile("pure(nope)", "").is_err());
    assert!(compile("let x <- sample(addr!(\"x\"), Nope(1.0)); pure(x)", "").is_err());
    assert!(compile("let x <- sample(addr!(\"x\"), Normal(1.0)); pure(x)", "").is_err());
    assert!(compile("let x = 1.0;", "").is_err()); // no pure
    let err = match compile("let x <- sample(addr!(\"x\") Normal(0,1)); pure(x)", "") {
        Err(e) => e.to_string(),
        Ok(_) => panic!("missing comma must not parse"),
    };
    assert!(err.contains("line"), "error should carry a line: {err}");
}

#[test]
fn invalid_params_kill_weight_not_process() {
    // sigma = -1 is impossible: the model must still run (prior draw)
    // with -inf total weight, not crash.
    let src = r#"
        let mu <- sample(addr!("mu"), Normal(0.0, -1.0));
        pure(mu)
    "#;
    let cm = compile(src, "").unwrap();
    let (_, t) = prior_run(cm.build(), 0);
    assert_eq!(t.total_log_weight(), f64::NEG_INFINITY);
    // The site is still recorded, with its declared type.
    assert!(t.get_f64(&addr!("mu")).is_some());
    assert!(!cm.take_warnings().is_empty());
}

#[test]
fn factor_and_math_functions() {
    let src = r#"
        let x <- sample(addr!("x"), Normal(0.0, 1.0));
        factor(-0.5 * pow(x - 1.0, 2.0));
        pure(exp(x) / (1.0 + exp(x)))
    "#;
    let cm = compile(src, "").unwrap();
    let (r, t) = prior_run(cm.build_f64(), 5);
    assert!(r > 0.0 && r < 1.0);
    assert!(t.log_factors < 0.0);
}

#[test]
fn discrete_sites_sample_and_observe() {
    let src = r#"
        let k <- sample(addr!("k"), Poisson(4.0));
        let z <- sample(addr!("z"), Categorical(0.3, 0.7));
        observe(addr!("n"), Binomial(10, 0.5), 7);
        observe(addr!("flag"), Bernoulli(0.5), true);
        pure(k + z)
    "#;
    let cm = compile(src, "").unwrap();
    let (_, t) = prior_run(cm.build(), 2);
    assert!(t.get_u64(&addr!("k")).is_some());
    assert!(t.get_usize(&addr!("z")).is_some());
    assert!(t.total_log_weight().is_finite());
    assert!(cm.take_warnings().is_empty());
}
