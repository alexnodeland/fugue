//! Extension points (issue #65): a host registers named distributions and
//! pure functions; programs call them with their own variables; unknown
//! names and wrong arities are compile-time errors; a constructor's or
//! function's `Err` at runtime is a soft error, never a panic.

#![cfg(feature = "program")]

use std::sync::{Arc, Mutex};

use fugue::program::{
    Arity, CompiledProgram, Data, HostDist, Program, ProgramError, Registry, SiteType, Value,
};
use fugue::*;
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};

/// Metadata a host attaches to each `Next` site.
#[derive(Clone, Debug, PartialEq)]
struct SiteMeta {
    prev: usize,
    failed: bool,
}

/// A host wrapper: the distribution to sample plus the site's metadata. It
/// delegates everything to the inner distribution and logs the metadata of
/// every site it is asked to sample, the way a harness handler would read it.
#[derive(Clone)]
struct Tagged {
    inner: Categorical,
    meta: SiteMeta,
    log: Arc<Mutex<Vec<SiteMeta>>>,
}

impl Distribution<usize> for Tagged {
    fn sample(&self, rng: &mut dyn RngCore) -> usize {
        self.log.lock().unwrap().push(self.meta.clone());
        self.inner.sample(rng)
    }
    fn log_prob(&self, x: &usize) -> f64 {
        self.inner.log_prob(x)
    }
    fn clone_box(&self) -> Box<dyn Distribution<usize>> {
        Box::new(self.clone())
    }
}

/// The habit's next-tool probabilities given the last tool and whether it
/// failed (three tools; tool 2 finishes).
fn next_probs(prev: usize, failed: bool) -> Vec<f64> {
    match (prev, failed) {
        (_, true) => vec![0.6, 0.3, 0.1],
        (0, false) => vec![0.1, 0.6, 0.3],
        _ => vec![0.2, 0.2, 0.6],
    }
}

fn host_registry(log: Arc<Mutex<Vec<SiteMeta>>>) -> Registry {
    let mut registry = Registry::new();
    registry
        .register_distribution("Next", SiteType::Usize, Arity::Exact(2), move |args| {
            let prev = args[0]
                .as_usize()
                .filter(|&p| p < 3)
                .ok_or_else(|| format!("Next: unknown previous tool {}", args[0]))?;
            let failed = args[1]
                .as_bool()
                .ok_or_else(|| format!("Next: `failed` must be a bool, found {}", args[1]))?;
            let inner = Categorical::new(next_probs(prev, failed)).map_err(|e| e.to_string())?;
            Ok(HostDist::Usize(Box::new(Tagged {
                inner,
                meta: SiteMeta { prev, failed },
                log: log.clone(),
            })))
        })
        .register_function("penalty", Arity::Exact(2), |args| {
            let tool = args[0].as_usize().ok_or("penalty: tool must be an index")?;
            let failed = args[1].as_bool().ok_or("penalty: failed must be a bool")?;
            Ok(Value::F64(if failed && tool != 0 { -1.0 } else { 0.0 }))
        })
        .register_function("bump", Arity::Between(1, 2), |args| {
            let n = args[0].as_i64().ok_or("bump: n must be an integer")?;
            let on = args.get(1).map_or(Some(true), Value::as_bool);
            Ok(Value::Int(if on == Some(true) { n + 1 } else { n }))
        });
    registry
}

const FLOW: &str = r#"
    let prev = 0;
    let failed = false;
    for i in 0..6 {
        let d <- sample(addr!("decide", i), Next(prev, failed));
        factor(penalty(d, failed));
        if d == 2 { break; }
        let ok <- sample(addr!("outcome", i), Bernoulli(0.8));
        prev = d;
        failed = !ok;
    }
    pure(prev)
"#;

fn run_seed<A>(model: Model<A>, seed: u64) -> (A, Trace) {
    runtime::handler::run(
        PriorHandler {
            rng: &mut StdRng::seed_from_u64(seed),
            trace: Trace::default(),
        },
        model,
    )
}

fn compile(src: &str, registry: &Registry) -> Result<CompiledProgram, ProgramError> {
    Program::parse(src)?.compile(registry, &Data::new())
}

#[test]
fn host_distribution_takes_program_variables_and_carries_site_metadata() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let registry = host_registry(log.clone());
    let compiled = compile(FLOW, &registry).unwrap();
    for seed in 0..100 {
        log.lock().unwrap().clear();
        let (value, t) = run_seed(compiled.build(), seed);
        // The metadata each visited `decide` site saw is the state the
        // program carried into it.
        let metas = log.lock().unwrap().clone();
        let (mut prev, mut failed, mut factors) = (0usize, false, 0.0);
        for (i, meta) in metas.iter().enumerate() {
            assert_eq!(meta, &SiteMeta { prev, failed }, "seed {seed} site {i}");
            let d = t.get_usize(&addr!("decide", i)).unwrap();
            // Scored under the host's distribution.
            let logp = t.choices[&addr!("decide", i)].logp;
            assert_eq!(logp, next_probs(prev, failed)[d].ln());
            if failed && d != 0 {
                factors -= 1.0;
            }
            if d == 2 {
                assert_eq!(i + 1, metas.len());
                break;
            }
            prev = d;
            failed = !t.get_bool(&addr!("outcome", i)).unwrap();
        }
        assert_eq!(value.as_usize(), Some(prev), "seed {seed}");
        assert_eq!(t.log_factors, factors);
        assert!(t.total_log_weight().is_finite());
    }
    // Real inference runs on it.
    let draws = adaptive_mcmc_chain(&mut StdRng::seed_from_u64(2), || compiled.build(), 300, 100);
    assert!(draws.iter().all(|(_, t)| t.total_log_weight().is_finite()));
    assert!(compiled.take_warnings().is_empty());

    // Observing under a host distribution scores with it too.
    let compiled = compile(
        r#"observe(addr!("seen"), Next(1, true), 2); observe(addr!("u"), Next(0, false), 1); pure(0)"#,
        &registry,
    )
    .unwrap();
    let (_, t) = run_seed(compiled.build(), 0);
    assert_eq!(t.log_likelihood, 0.1f64.ln() + 0.6f64.ln());
    assert!(t.choices.is_empty());
}

#[test]
fn host_functions_take_ints_and_bools() {
    let registry = host_registry(Arc::default());
    for (expr, expected) in [
        ("bump(3)", Value::Int(4)),
        ("bump(3, true)", Value::Int(4)),
        ("bump(3, false)", Value::Int(3)),
        ("bump(3, 1 > 2)", Value::Int(3)),
        ("bump(2.0, 1)", Value::Int(3)),
        ("penalty(1, true)", Value::F64(-1.0)),
    ] {
        let compiled = compile(&format!("pure({expr})"), &registry).unwrap();
        assert_eq!(run_seed(compiled.build(), 0).0, expected, "{expr}");
    }
}

#[test]
fn unknown_names_and_wrong_arities_are_compile_time_errors() {
    let registry = host_registry(Arc::default());
    for (src, needle) in [
        (
            r#"let d <- sample(addr!("d"), Prev(0, true)); pure(d)"#,
            "unknown distribution `Prev`",
        ),
        (
            r#"observe(addr!("d"), Nope(1.0), 0); pure(0)"#,
            "unknown distribution `Nope`",
        ),
        ("pure(bumpp(1))", "unknown function `bumpp`"),
        (
            r#"let d <- sample(addr!("d"), Next(0)); pure(d)"#,
            "`Next` takes 2 argument(s), got 1",
        ),
        ("pure(penalty(1))", "`penalty` takes 2 argument(s), got 1"),
        (
            "pure(bump(1, true, 3))",
            "`bump` takes 1..=2 argument(s), got 3",
        ),
        (
            r#"let z <- sample(addr!("z"), Categorical()); pure(z)"#,
            "at least 1",
        ),
        ("pure(exp(1.0, 2.0))", "`exp` takes 1 argument(s), got 2"),
    ] {
        match compile(src, &registry) {
            Err(ProgramError::Check(m)) => assert!(m.contains(needle), "{src}: {m}"),
            Err(other) => panic!("{src}: {other:?}"),
            Ok(_) => panic!("{src}: compiled"),
        }
    }
    // The host's names are not in the default registry.
    assert!(compile(FLOW, &Registry::new()).is_err());
    // An empty registry knows nothing, not even the built-ins.
    assert!(compile(
        r#"let x <- sample(addr!("x"), Normal(0.0, 1.0)); pure(x)"#,
        &Registry::empty()
    )
    .is_err());
}

#[test]
fn constructor_and_function_errors_are_soft_errors() {
    let registry = host_registry(Arc::default());
    let mut registry = registry;
    registry
        .register_distribution("Wrong", SiteType::Usize, Arity::Exact(0), |_| {
            Ok(HostDist::F64(Box::new(Normal::standard())))
        })
        .register_function("fails", Arity::Exact(0), |_| {
            Err("fails: always".to_string())
        });

    for (src, needle, site) in [
        // The constructor's own `Err`, surfaced as a warning.
        (
            r#"let d <- sample(addr!("d"), Next(7, false)); pure(d)"#,
            "unknown previous tool 7",
            Some(addr!("d")),
        ),
        (
            r#"let d <- sample(addr!("d"), Next(0, [1.0])); pure(d)"#,
            "`failed` must be a bool",
            Some(addr!("d")),
        ),
        // A distribution of another type than the one registered.
        (
            r#"let d <- sample(addr!("d"), Wrong()); pure(d)"#,
            "registered with usize sites but built a f64 distribution",
            Some(addr!("d")),
        ),
        (
            r#"observe(addr!("o"), Next(9, true), 1); pure(0)"#,
            "unknown previous tool 9",
            None,
        ),
        ("pure(fails())", "fails: always", None),
        (
            "let x = bump(1.5); pure(x)",
            "bump: n must be an integer",
            None,
        ),
    ] {
        let compiled = compile(src, &registry).unwrap();
        let (_, t) = run_seed(compiled.build(), 0);
        assert_eq!(t.total_log_weight(), f64::NEG_INFINITY, "{src}");
        let warnings = compiled.take_warnings();
        assert_eq!(warnings.len(), 1, "{src}: {warnings:?}");
        assert!(warnings[0].contains(needle), "{src}: {warnings:?}");
        if let Some(site) = site {
            // The site keeps its address and its declared type.
            assert_eq!(t.get_usize(&site), Some(0), "{src}");
        }
    }
}

#[test]
fn registry_entries_can_be_replaced_and_queried() {
    let mut registry = Registry::new();
    assert!(registry.has_distribution("Normal") && registry.has_function("exp"));
    assert_eq!(
        registry.distribution_site_type("Categorical"),
        Some(SiteType::Usize)
    );
    assert_eq!(registry.distribution_site_type("Next"), None);
    // A host can replace a built-in; programs then call the host's version.
    registry.register_function("exp", Arity::Exact(1), |_| Ok(Value::Int(42)));
    let compiled = compile("pure(exp(1.0))", &registry).unwrap();
    assert_eq!(run_seed(compiled.build(), 0).0, Value::Int(42));

    fn shareable<T: Send + Sync + Clone>() {}
    shareable::<Registry>();
    shareable::<CompiledProgram>();
    shareable::<Program>();

    // A compiled program is shared across threads, each building its own
    // models.
    let compiled = Arc::new(compile(FLOW, &host_registry(Arc::default())).unwrap());
    let handles: Vec<_> = (0..4)
        .map(|seed| {
            let compiled = compiled.clone();
            std::thread::spawn(move || run_seed(compiled.build(), seed).1.choices.len())
        })
        .collect();
    for h in handles {
        assert!(h.join().unwrap() >= 1);
    }
}
