//! The playground's model language: a `prob!` subset interpreted into real
//! fugue [`Model`]s.
//!
//! The parser and interpreter that used to live here moved into `fugue-ppl`
//! as its `program` feature ([`fugue::program`], issue #65). There they gained
//! a serializable AST with text and JSON front ends, typed values, `if`/`else`,
//! comparison and boolean operators, `break`, reassignment, array literals,
//! and a registry a host can extend. This module keeps fugue-wasm's own API
//! (compile a source string plus a JSON data payload, build `Model<f64>`s,
//! drain warnings) as a thin adapter over it.
//!
//! What the adapter preserves, all documented on `fugue::program`: model
//! *construction* is interpreted but everything downstream is the real crate;
//! addresses are built with fugue's `make_name`/`make_indexed`, so they are
//! byte-identical to `addr!(..)` in compiled Rust; and runtime soft errors
//! (index out of bounds, invalid distribution parameters, ...) degrade to
//! `factor(-inf)` plus a warning the host can surface, while static errors
//! (syntax, unknown variables/distributions, wrong arities) are rejected at
//! compile time.

use std::fmt::Write as _;
use std::sync::OnceLock;

use fugue::program::{CompiledProgram, Data, Program, Registry, Value};
use fugue::*;

/// The built-in registry, built once and shared by every compile.
fn registry() -> &'static Registry {
    static REGISTRY: OnceLock<Registry> = OnceLock::new();
    REGISTRY.get_or_init(Registry::new)
}

/// A parsed, validated model plus its data environment. `build()` constructs
/// a fresh single-use [`Model`], so `|| compiled.build()` is the `model_fn`
/// every fugue inference driver wants.
pub struct CompiledModel {
    program: CompiledProgram,
    data: Data,
}

impl CompiledModel {
    /// Parse `source` and bind data from `data_json` — a JSON object of
    /// number arrays (`{"y": [...], ...}`) or scalars, a bare array (bound to
    /// `data`), or empty/`null` for no data. Booleans in arrays are coerced
    /// to 0/1.
    pub fn compile(source: &str, data_json: &str) -> Result<CompiledModel, String> {
        let program = Program::parse(source).map_err(|e| e.to_string())?;
        let data = Data::from_json(data_json).map_err(|e| e.to_string())?;
        let program = program
            .compile(registry(), &data)
            .map_err(|e| e.to_string())?;
        Ok(CompiledModel { program, data })
    }

    /// Build a fresh single-use model. The program's value is returned as an
    /// `f64` (bools as 0/1, arrays as NaN).
    pub fn build(&self) -> Model<f64> {
        self.program.build_f64()
    }

    /// Drain accumulated runtime warnings (soft errors mapped to `-inf`).
    pub fn take_warnings(&self) -> Vec<String> {
        self.program.take_warnings()
    }

    /// Human-readable summary of the data environment (for error messages).
    pub fn data_summary(&self) -> String {
        let mut s = String::new();
        for (k, v) in self.data.iter() {
            if let Value::Arr(a) = v {
                let _ = write!(s, "{k}[{}] ", a.len());
            }
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fugue::inference::mh::adaptive_mcmc_chain;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    const COIN: &str = r#"
        let p <- sample(addr!("p"), Beta(2.0, 2.0));
        for i in 0..data.len() {
            observe(addr!("flip", i), Bernoulli(p), data[i]);
        }
        pure(p)
    "#;

    #[test]
    fn coin_model_matches_native_addresses_and_posterior() {
        let data = "[1,0,1,1,0,1,1,0,1,1]"; // 7 heads, 3 tails
        let cm = CompiledModel::compile(COIN, data).unwrap();

        // Address parity with the addr! macro.
        let mut rng = StdRng::seed_from_u64(11);
        let (_, t) = fugue::runtime::handler::run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            cm.build(),
        );
        assert!(t.get_f64(&addr!("p")).is_some());
        // Observations are scored, not recorded as choices.
        assert_eq!(t.choices.len(), 1);
        assert!(t.log_likelihood.is_finite() && t.log_likelihood < 0.0);

        // Real inference on the interpreted model recovers the conjugate
        // posterior mean Beta(2+7, 2+3) => 9/14.
        let mut rng = StdRng::seed_from_u64(11);
        let samples = adaptive_mcmc_chain(&mut rng, || cm.build(), 4000, 1000);
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
        let cm = CompiledModel::compile(src, data).unwrap();
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
        let cm = CompiledModel::compile(src, "").unwrap();
        let mut rng = StdRng::seed_from_u64(1);
        let (_, t) = fugue::runtime::handler::run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            cm.build(),
        );
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
        assert!(CompiledModel::compile(src, "").is_ok());
    }

    #[test]
    fn static_errors_are_caught() {
        assert!(CompiledModel::compile("pure(nope)", "").is_err());
        assert!(
            CompiledModel::compile("let x <- sample(addr!(\"x\"), Nope(1.0)); pure(x)", "")
                .is_err()
        );
        assert!(
            CompiledModel::compile("let x <- sample(addr!(\"x\"), Normal(1.0)); pure(x)", "")
                .is_err()
        );
        assert!(CompiledModel::compile("let x = 1.0;", "").is_err()); // no pure
        let err = match CompiledModel::compile(
            "let x <- sample(addr!(\"x\") Normal(0,1)); pure(x)",
            "",
        ) {
            Err(e) => e,
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
        let cm = CompiledModel::compile(src, "").unwrap();
        let mut rng = StdRng::seed_from_u64(0);
        let (_, t) = fugue::runtime::handler::run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            cm.build(),
        );
        assert_eq!(t.total_log_weight(), f64::NEG_INFINITY);
        assert!(!cm.take_warnings().is_empty());
    }

    #[test]
    fn factor_and_math_functions() {
        let src = r#"
            let x <- sample(addr!("x"), Normal(0.0, 1.0));
            factor(-0.5 * pow(x - 1.0, 2.0));
            pure(exp(x) / (1.0 + exp(x)))
        "#;
        let cm = CompiledModel::compile(src, "").unwrap();
        let mut rng = StdRng::seed_from_u64(5);
        let (r, t) = fugue::runtime::handler::run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            cm.build(),
        );
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
        let cm = CompiledModel::compile(src, "").unwrap();
        let mut rng = StdRng::seed_from_u64(2);
        let (_, t) = fugue::runtime::handler::run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            cm.build(),
        );
        assert!(t.get_u64(&addr!("k")).is_some());
        assert!(t.get_usize(&addr!("z")).is_some());
        assert!(t.total_log_weight().is_finite());
        assert!(cm.take_warnings().is_empty());
    }

    #[test]
    fn a_site_built_from_the_language_downcasts_like_a_compiled_one() {
        // #63: a handler at a site the language built sees the distribution
        // itself, as it would in compiled Rust.
        struct Probs<'a>(&'a mut Vec<f64>);
        impl<H: Handler> Overrides<H> for Probs<'_> {
            fn on_sample_usize(
                &mut self,
                inner: &mut H,
                addr: &Address,
                dist: &dyn Distribution<usize>,
            ) -> usize {
                let cat = dist.downcast_ref::<Categorical>().expect("a Categorical");
                self.0.extend_from_slice(cat.probs());
                inner.on_sample_usize(addr, dist)
            }
        }
        let cm = CompiledModel::compile(
            "let z <- sample(addr!(\"z\"), Categorical([0.3, 0.7]));\npure(z)",
            "",
        )
        .unwrap();
        let mut probs = Vec::new();
        let mut rng = StdRng::seed_from_u64(1);
        fugue::runtime::handler::run(
            Delegate::with(
                PriorHandler {
                    rng: &mut rng,
                    trace: Trace::default(),
                },
                Probs(&mut probs),
            ),
            cm.build(),
        );
        assert_eq!(probs, [0.3, 0.7]);
    }

    #[test]
    fn control_flow_reaches_the_playground() {
        // The language extensions of fugue::program (issue #65) are
        // available to the playground unchanged: comparisons, if/else,
        // break, reassignment, array literals, typed values.
        let src = r#"
            let tries = 0;
            for i in 0..20 {
                let hit <- sample(addr!("hit", i), Bernoulli(0.3));
                tries = tries + 1;
                if hit && tries >= 1 { break; }
            }
            let z <- sample(addr!("z"), Categorical([0.2, 0.8]));
            if z == 1 { factor(0.0); } else { factor(-1.0); }
            pure(tries)
        "#;
        let cm = CompiledModel::compile(src, "").unwrap();
        let mut rng = StdRng::seed_from_u64(9);
        let (tries, t) = fugue::runtime::handler::run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            cm.build(),
        );
        // One `hit` site per iteration actually run, then `z`.
        assert_eq!(t.choices.len() as f64, tries + 1.0);
        assert_eq!(t.get_bool(&addr!("hit", tries as usize - 1)), Some(true));
        assert!(t.total_log_weight().is_finite());
        assert!(cm.take_warnings().is_empty());
        assert!(CompiledModel::compile("break; pure(0)", "").is_err());
    }
}
