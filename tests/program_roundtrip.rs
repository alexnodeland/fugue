//! The playground's programs through both front ends (issue #65).
//!
//! Every model the docs site runs — the playground presets and the
//! explorables' regression model, read straight out of their JavaScript so
//! this test follows the real sources, plus the programs fugue-wasm's own
//! tests use — must:
//!
//! 1. round-trip text -> JSON -> program (and back through the text printer)
//!    to a structurally equal program;
//! 2. build models that produce identical traces (addresses, values, logp,
//!    weights) and identical MH chains before and after the round trip;
//! 3. and, for the presets, build exactly the model hand-written Rust builds:
//!    same addresses as `addr!`, same draws, same weights, same MH chain.

#![cfg(feature = "program")]

use std::fs;
use std::path::Path;
use std::sync::Arc;

use fugue::program::{CompiledProgram, Data, Program, Registry, Value};
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

struct Case {
    name: String,
    source: String,
    data: String,
}

/// The JavaScript string value that follows each occurrence of `key` in
/// `js`: one or more quoted literals joined by `+`.
fn js_values(js: &str, key: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = js;
    while let Some(at) = rest.find(key) {
        let mut s = &rest[at + key.len()..];
        let mut value = String::new();
        loop {
            s = s.trim_start();
            let Some(quote) = s.chars().next().filter(|c| *c == '\'' || *c == '"') else {
                panic!("expected a string literal after `{key}`: {:.40}", s)
            };
            let body = &s[1..];
            let mut chars = body.char_indices();
            let end = loop {
                match chars.next() {
                    Some((i, c)) if c == quote => break i,
                    Some((_, '\\')) => match chars.next() {
                        Some((_, 'n')) => value.push('\n'),
                        Some((_, c)) => value.push(c),
                        None => panic!("unterminated escape"),
                    },
                    Some((_, c)) => value.push(c),
                    None => panic!("unterminated string after `{key}`"),
                }
            };
            s = body[end + 1..].trim_start();
            match s.strip_prefix('+') {
                Some(more) => s = more,
                None => break,
            }
        }
        out.push(value);
        rest = s;
    }
    out
}

fn read(rel: &str) -> String {
    fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join(rel))
        .unwrap_or_else(|e| panic!("{rel}: {e}"))
}

/// The docs tree is part of the repository but not of the published crate
/// (it is its own package); in a packaged copy the docs cases are skipped.
/// In the repository, a missing file under `docs/` still fails loudly.
fn docs_available() -> bool {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("docs").is_dir()
}

/// The docs' programs: playground presets (with their data) and the
/// explorables' model (with data shaped like what the widgets send).
fn docs_cases() -> Vec<Case> {
    if !docs_available() {
        return Vec::new();
    }
    let js = read("docs/viz/playground.js");
    let names = js_values(&js, "name:");
    let sources = js_values(&js, "source:");
    let datas = js_values(&js, "data:");
    assert!(
        sources.len() >= 4,
        "found {} playground presets",
        sources.len()
    );
    assert_eq!(sources.len(), datas.len());
    assert_eq!(sources.len(), names.len());
    let mut cases: Vec<Case> = names
        .into_iter()
        .zip(sources)
        .zip(datas)
        .map(|((name, source), data)| Case { name, source, data })
        .collect();
    for file in ["docs/viz/metropolis.js", "docs/viz/hmc.js"] {
        for source in js_values(&read(file), "var SRC =") {
            cases.push(Case {
                name: file.to_string(),
                source,
                data: r#"{"x": [-1.5, -0.5, 0.25, 1.0, 2.0], "y": [-1.1, -0.2, 0.1, 0.9, 1.4]}"#
                    .to_string(),
            });
        }
    }
    cases
}

/// fugue-wasm's test programs and the old `dsl.rs` module-doc example.
fn fixture_cases() -> Vec<Case> {
    let case = |name: &str, source: &str, data: &str| Case {
        name: name.to_string(),
        source: source.to_string(),
        data: data.to_string(),
    };
    vec![
        case(
            "wasm hmc",
            r#"
            let mu <- sample(addr!("mu"), Normal(0.0, 2.0));
            for i in 0..data.len() {
                observe(addr!("y", i), Normal(mu, 1.0), data[i]);
            }
            pure(mu)"#,
            "[1.3, 0.7, 2.1, 0.4, 1.5]",
        ),
        case(
            "wasm grid",
            r#"
            let a <- sample(addr!("a"), Normal(0.0, 2.0));
            let b <- sample(addr!("b"), Normal(0.0, 2.0));
            observe(addr!("y"), Normal(a + b, 0.5), 2.0);
            pure(a)"#,
            "",
        ),
        case(
            "dsl doc",
            r#"
            let p <- sample(addr!("p"), Beta(2.0, 2.0));
            let mu = 2.0 * p - 1.0;
            for i in 0..y.len() {
                observe(addr!("y", i), Normal(mu, 0.8), y[i]);
            }
            factor(-0.5 * mu * mu);
            pure(p)"#,
            r#"{"y": [0.4, -0.3, 1.1]}"#,
        ),
    ]
}

fn compile(p: &Program, data: &str) -> CompiledProgram {
    p.compile(&Registry::new(), &Data::from_json(data).unwrap())
        .unwrap()
}

/// Everything a trace records, bit for bit.
fn signature(t: &Trace) -> Vec<String> {
    let mut sig: Vec<String> = t
        .choices
        .iter()
        .map(|(a, c)| {
            assert_eq!(&c.addr, a);
            format!("{a} = {:?} @ {:x}", c.value, c.logp.to_bits())
        })
        .collect();
    sig.push(format!(
        "prior {:x} likelihood {:x} factors {:x}",
        t.log_prior.to_bits(),
        t.log_likelihood.to_bits(),
        t.log_factors.to_bits()
    ));
    sig
}

fn prior_run<A>(model: Model<A>, seed: u64) -> (A, Trace) {
    runtime::handler::run(
        PriorHandler {
            rng: &mut StdRng::seed_from_u64(seed),
            trace: Trace::default(),
        },
        model,
    )
}

fn same_value(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::F64(x), Value::F64(y)) => x.to_bits() == y.to_bits(),
        _ => a == b,
    }
}

#[test]
fn docs_programs_are_found() {
    if !docs_available() {
        return;
    }
    let cases = docs_cases();
    // The four presets plus the two explorables' model.
    assert!(cases.len() >= 6);
    assert!(cases.iter().any(|c| c.source.contains("theta")));
    for c in &cases {
        assert!(c.source.trim_end().ends_with(')'), "{}", c.name);
    }
}

#[test]
fn programs_round_trip_through_json_and_text() {
    for c in docs_cases().into_iter().chain(fixture_cases()) {
        let parsed = Program::parse(&c.source).unwrap_or_else(|e| panic!("{}: {e}", c.name));

        let json = parsed.to_json().unwrap();
        let back = Program::from_json(&json).unwrap();
        assert_eq!(back, parsed, "{}: JSON round trip", c.name);
        // Serialization is deterministic, so the JSON is a stable artifact.
        assert_eq!(back.to_json().unwrap(), json, "{}", c.name);
        let pretty = parsed.to_json_pretty().unwrap();
        assert_eq!(Program::from_json(&pretty).unwrap(), parsed, "{}", c.name);

        // The canonical text form reads back to the same program too.
        let text = parsed.to_string();
        assert_eq!(Program::parse(&text).unwrap(), parsed, "{}: {text}", c.name);
    }
}

#[test]
fn round_tripped_programs_build_identical_models() {
    for c in docs_cases().into_iter().chain(fixture_cases()) {
        let parsed = Program::parse(&c.source).unwrap();
        let from_json = Program::from_json(&parsed.to_json().unwrap()).unwrap();
        let (a, b) = (compile(&parsed, &c.data), compile(&from_json, &c.data));
        for seed in 0..25 {
            let (ra, ta) = prior_run(a.build(), seed);
            let (rb, tb) = prior_run(b.build(), seed);
            assert_eq!(signature(&ta), signature(&tb), "{} seed {seed}", c.name);
            assert!(same_value(&ra, &rb), "{} seed {seed}", c.name);
            assert!(!ta.choices.is_empty() && ta.total_log_weight().is_finite());
        }
        let chain_a = adaptive_mcmc_chain(&mut StdRng::seed_from_u64(1), || a.build(), 150, 50);
        let chain_b = adaptive_mcmc_chain(&mut StdRng::seed_from_u64(1), || b.build(), 150, 50);
        for ((ra, ta), (rb, tb)) in chain_a.iter().zip(&chain_b) {
            assert_eq!(
                signature(ta),
                signature(tb),
                "{}: MH chains diverge",
                c.name
            );
            assert!(same_value(ra, rb));
        }
        assert!(a.take_warnings().is_empty() && b.take_warnings().is_empty());
    }
}

// ---------------------------------------------------------------------------
// Interpreted vs hand-written Rust
// ---------------------------------------------------------------------------

fn data_array(data: &Data, name: &str) -> Arc<Vec<f64>> {
    Arc::new(data.get(name).unwrap().as_array().unwrap().to_vec())
}

fn native_coin(flips: Arc<Vec<f64>>) -> Model<f64> {
    sample(addr!("p"), Beta::new(2.0, 2.0).unwrap()).bind(move |p| {
        let obs = (0..flips.len())
            .map(|i| {
                observe(
                    addr!("flip", i),
                    Bernoulli::new(p).unwrap(),
                    flips[i] != 0.0,
                )
            })
            .collect();
        sequence_vec(obs).map(move |_| p)
    })
}

fn native_gauss(y: Arc<Vec<f64>>) -> Model<f64> {
    sample(addr!("mu"), Normal::new(0.0, 2.0).unwrap()).bind(move |mu| {
        let obs = (0..y.len())
            .map(|i| observe(addr!("y", i), Normal::new(mu, 1.0).unwrap(), y[i]))
            .collect();
        sequence_vec(obs).map(move |_| mu)
    })
}

fn native_regression(x: Arc<Vec<f64>>, y: Arc<Vec<f64>>) -> Model<f64> {
    prob! {
        let a <- sample(addr!("a"), Normal::new(0.0, 2.5).unwrap());
        let b <- sample(addr!("b"), Normal::new(0.0, 2.5).unwrap());
        let obs = (0..x.len())
            .map(|i| observe(addr!("y", i), Normal::new(a * x[i] + b, 0.8).unwrap(), y[i]))
            .collect();
        sequence_vec(obs).map(move |_| a)
    }
}

fn native_schools(y: Arc<Vec<f64>>, se: Arc<Vec<f64>>) -> Model<f64> {
    fn school(j: usize, mu: f64, tau: f64, y: Arc<Vec<f64>>, se: Arc<Vec<f64>>) -> Model<()> {
        if j == y.len() {
            return pure(());
        }
        sample(addr!("theta", j), Normal::new(mu, tau).unwrap()).bind(move |theta| {
            observe(addr!("obs", j), Normal::new(theta, se[j]).unwrap(), y[j])
                .bind(move |_| school(j + 1, mu, tau, y, se))
        })
    }
    prob! {
        let mu <- sample(addr!("mu"), Normal::new(0.0, 5.0).unwrap());
        let tau <- sample(addr!("tau"), LogNormal::new(1.0, 0.5).unwrap());
        school(0, mu, tau, y, se).map(move |_| mu)
    }
}

#[test]
fn presets_build_exactly_the_hand_written_models() {
    let presets = docs_cases();
    let mut checked = 0;
    for c in presets.iter().filter(|c| !c.name.starts_with("docs/")) {
        let data = Data::from_json(&c.data).unwrap();
        let native: Box<dyn Fn() -> Model<f64>> = if c.source.contains("theta") {
            let (y, se) = (data_array(&data, "y"), data_array(&data, "se"));
            Box::new(move || native_schools(y.clone(), se.clone()))
        } else if c.source.contains("addr!(\"b\")") {
            let (x, y) = (data_array(&data, "x"), data_array(&data, "y"));
            Box::new(move || native_regression(x.clone(), y.clone()))
        } else if c.source.contains("flip") {
            let flips = data_array(&data, "data");
            Box::new(move || native_coin(flips.clone()))
        } else {
            let y = data_array(&data, "data");
            Box::new(move || native_gauss(y.clone()))
        };
        let compiled = compile(&Program::parse(&c.source).unwrap(), &c.data);
        for seed in 0..25 {
            let (rn, tn) = prior_run(native(), seed);
            let (ri, ti) = prior_run(compiled.build_f64(), seed);
            assert_eq!(signature(&tn), signature(&ti), "{} seed {seed}", c.name);
            assert_eq!(rn.to_bits(), ri.to_bits());
        }
        let chain_n = adaptive_mcmc_chain(&mut StdRng::seed_from_u64(4), &native, 200, 100);
        let chain_i = adaptive_mcmc_chain(
            &mut StdRng::seed_from_u64(4),
            || compiled.build_f64(),
            200,
            100,
        );
        for ((rn, tn), (ri, ti)) in chain_n.iter().zip(&chain_i) {
            assert_eq!(
                signature(tn),
                signature(ti),
                "{}: MH chains diverge",
                c.name
            );
            assert_eq!(rn.to_bits(), ri.to_bits());
        }
        checked += 1;
    }
    if docs_available() {
        assert_eq!(checked, 4, "one native twin per playground preset");
    }
}
