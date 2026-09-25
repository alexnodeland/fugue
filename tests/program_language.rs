//! The program language's semantics (issue #65): reassignment and lexical
//! scoping, `if`/`else`, comparison and boolean operators, `break`, typed
//! values, array literals, data scalars, addresses, and the total-evaluation
//! policy for runtime soft errors.

#![cfg(feature = "program")]

use fugue::program::{CompiledProgram, Data, Program, ProgramError, Registry, Value};
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn compile_with(src: &str, data: Data) -> CompiledProgram {
    Program::parse(src)
        .unwrap_or_else(|e| panic!("{e}\n{src}"))
        .compile(&Registry::new(), &data)
        .unwrap_or_else(|e| panic!("{e}\n{src}"))
}

fn compile(src: &str) -> CompiledProgram {
    compile_with(src, Data::new())
}

fn compile_err(src: &str) -> ProgramError {
    match Program::parse(src).and_then(|p| p.compile(&Registry::new(), &Data::new())) {
        Err(e) => e,
        Ok(_) => panic!("expected a compile error:\n{src}"),
    }
}

fn run_seed<A>(model: Model<A>, seed: u64) -> (A, Trace) {
    runtime::handler::run(
        PriorHandler {
            rng: &mut StdRng::seed_from_u64(seed),
            trace: Trace::default(),
        },
        model,
    )
}

/// The program's value, run once under the prior; it must not warn.
fn value_of(src: &str) -> Value {
    let compiled = compile(src);
    let (v, _) = run_seed(compiled.build(), 0);
    assert_eq!(compiled.take_warnings(), Vec::<String>::new(), "{src}");
    v
}

fn eval(expr: &str) -> Value {
    value_of(&format!("pure({expr})"))
}

// ---------------------------------------------------------------------------
// Reassignment and scoping
// ---------------------------------------------------------------------------

const FLOW: &str = r#"
    let prev = 3;
    let failed = false;
    for i in 0..8 {
        let d <- sample(addr!("decide", i), Categorical([0.5, 0.25, 0.25]));
        if d == 0 { break; }
        let ok <- sample(addr!("outcome", i), Bernoulli(0.9));
        prev = d;
        failed = !ok;
    }
    pure(prev)
"#;

/// The same flow as hand-written Rust.
fn native_flow(i: usize, prev: usize) -> Model<usize> {
    if i == 8 {
        return pure(prev);
    }
    let decide = Categorical::new(vec![0.5, 0.25, 0.25]).unwrap();
    sample(addr!("decide", i), decide).bind(move |d| {
        if d == 0 {
            return pure(prev);
        }
        sample(addr!("outcome", i), Bernoulli::new(0.9).unwrap())
            .bind(move |_ok| native_flow(i + 1, d))
    })
}

#[test]
fn assignment_carries_state_across_iterations_and_break_stops_the_loop() {
    let compiled = compile(FLOW);
    let mut saw_break = false;
    let mut saw_full_run = false;
    for seed in 0..300 {
        let (value, trace) = run_seed(compiled.build(), seed);
        // The sites visited: decide#0..=k (k = the first zero, or 7) and
        // outcome#0..k, and nothing else.
        let k = (0..8)
            .find(|&i| trace.get_usize(&addr!("decide", i)) == Some(0))
            .unwrap_or(8);
        let decides = if k < 8 { k + 1 } else { 8 };
        assert_eq!(trace.choices.len(), decides + k, "seed {seed}");
        for i in 0..decides {
            assert!(
                trace.get_usize(&addr!("decide", i)).is_some(),
                "seed {seed}"
            );
        }
        for i in 0..k {
            assert!(
                trace.get_bool(&addr!("outcome", i)).is_some(),
                "seed {seed}"
            );
        }
        assert!(!trace.choices.contains_key(&addr!("outcome", k)));
        // The value is the last tool chosen, or the initial 3.
        let expected = if k == 0 {
            3
        } else {
            trace.get_usize(&addr!("decide", k - 1)).unwrap()
        };
        assert_eq!(value.as_usize(), Some(expected), "seed {seed}");
        saw_break |= k < 8;
        saw_full_run |= k == 8;

        // Bit-identical to the hand-written flow.
        let (native_value, native_trace) = run_seed(native_flow(0, 3), seed);
        assert_eq!(native_value, expected);
        assert_eq!(native_trace.choices.len(), trace.choices.len());
        for (a, c) in &trace.choices {
            let n = &native_trace.choices[a];
            assert_eq!((&n.value, n.logp.to_bits()), (&c.value, c.logp.to_bits()));
        }
        assert_eq!(
            native_trace.total_log_weight().to_bits(),
            trace.total_log_weight().to_bits()
        );
    }
    assert!(saw_break && saw_full_run);
    assert!(compiled.take_warnings().is_empty());
}

#[test]
fn let_in_a_block_shadows_only_until_the_block_ends() {
    // Neither the loop body's nor the branch's `let x` leaks out.
    let src = r#"
        let x = 1;
        for i in 0..3 {
            let x = 10 + i;
            factor(0.0 * x);
        }
        if true { let x = 99; }
        pure(x)
    "#;
    assert_eq!(value_of(src), Value::Int(1));
    // Each iteration's shadowing `let` starts again from the outer value...
    let src = "let total = 0; for i in 0..3 { let total = total + 100; } pure(total)";
    assert_eq!(value_of(src), Value::Int(0));
    // ...while an assignment persists across iterations and after the loop.
    let src = "let total = 0; for i in 0..4 { total = total + i; } pure(total)";
    assert_eq!(value_of(src), Value::Int(6));
    // Assigning a shadowed name assigns the innermost binding only.
    let src = "let x = 1; if true { let x = 2; x = 5; } pure(x)";
    assert_eq!(value_of(src), Value::Int(1));
    // A sample reads the outer binding of the name it binds.
    let src = r#"let p = 0.25; let p <- sample(addr!("p"), Bernoulli(p)); pure(p)"#;
    assert!(matches!(value_of(src), Value::Bool(_)));
    // Data can be shadowed, and assigned, like any outer variable.
    let data = Data::new().with("n", 2);
    let compiled = compile_with("n = n + 1; let n = n * 10; pure(n)", data);
    assert_eq!(run_seed(compiled.build(), 0).0, Value::Int(30));
}

#[test]
fn unbound_names_are_static_errors() {
    for src in [
        "x = 1; pure(0)",
        "for i in 0..2 { let y = 1; } y = 2; pure(0)",
        "for i in 0..2 { } pure(i)",
        "if true { let t = 1; } pure(t)",
        r#"for i in 0..2 { let s <- sample(addr!("s", i), Normal(0.0, 1.0)); } pure(s)"#,
        "let x = x + 1; pure(x)",
    ] {
        match compile_err(src) {
            ProgramError::Check(m) => assert!(m.contains("variable"), "{src}: {m}"),
            other => panic!("{src}: {other:?}"),
        }
    }
    match compile_err("x = 1; pure(0)") {
        ProgramError::Check(m) => {
            assert!(m.contains("`x`") && m.contains("assign"), "{m}")
        }
        other => panic!("{other:?}"),
    }
    assert!(matches!(
        compile_err("break; pure(0)"),
        ProgramError::Check(m) if m.contains("break")
    ));
    assert!(matches!(
        compile_err("if true { break; } pure(0)"),
        ProgramError::Check(_)
    ));
}

// ---------------------------------------------------------------------------
// Control flow and operators
// ---------------------------------------------------------------------------

#[test]
fn if_else_chains() {
    let src = r#"
        let s = 0;
        for i in 0..6 {
            if i < 2 {
                s = s + 1;
            } else if i == 2 || i == 5 {
                s = s + 10;
            } else {
                s = s + 100;
            }
        }
        pure(s)
    "#;
    assert_eq!(value_of(src), Value::Int(222));
    assert_eq!(
        value_of("let v = 1.0; if v > 2.0 { v = 0.0; } pure(v)"),
        Value::F64(1.0)
    );
    // Effects in branches: only the taken branch's sites exist.
    let compiled = compile(
        r#"
        let coin <- sample(addr!("coin"), Bernoulli(0.5));
        if coin {
            let h <- sample(addr!("heads"), Normal(1.0, 1.0));
        } else {
            let t <- sample(addr!("tails"), Normal(-1.0, 1.0));
        }
        pure(coin)
    "#,
    );
    for seed in 0..40 {
        let (coin, t) = run_seed(compiled.build(), seed);
        let heads = coin == Value::Bool(true);
        assert_eq!(t.choices.contains_key(&addr!("heads")), heads);
        assert_eq!(t.choices.contains_key(&addr!("tails")), !heads);
        assert_eq!(t.choices.len(), 2);
    }
}

#[test]
fn comparison_and_boolean_operators() {
    for (expr, expected) in [
        ("1 < 2", true),
        ("2 <= 2", true),
        ("3 > 4", false),
        ("2.5 >= 2.5", true),
        ("1 == 1.0", true),
        ("1 != 2", true),
        ("true != false", true),
        ("false < true", true),
        ("!(1 > 2)", true),
        ("true && false || true", true),
        ("true && (false || false)", false),
        ("!true || !false", true),
        ("0.0 / 0.0 == 0.0 / 0.0", false),
        ("0.0 / 0.0 != 0.0 / 0.0", true),
        ("0.0 / 0.0 < 1.0", false),
        ("true == 1", true),
        ("1 + 2 * 3 == 7", true),
        ("-2 < -1", true),
        ("9223372036854775807 > 9223372036854775806", true),
    ] {
        assert_eq!(eval(expr), Value::Bool(expected), "{expr}");
    }
    // `&&`/`||` short-circuit: the right side is not evaluated (no
    // out-of-bounds soft error, no warning).
    let data = Data::new().with("y", vec![1.0, 2.0, 3.0]);
    let compiled = compile_with(
        "let i = 5; pure(i < y.len() && y[i] > 0.0 || i >= y.len() || y[100] > 0.0)",
        data,
    );
    let (v, t) = run_seed(compiled.build(), 0);
    assert_eq!(v, Value::Bool(true));
    assert!(t.total_log_weight().is_finite());
    assert!(compiled.take_warnings().is_empty());
}

#[test]
fn break_samples_until_the_first_success() {
    let compiled = compile(
        r#"
        let n = 0;
        for i in 0..10 {
            let hit <- sample(addr!("hit", i), Bernoulli(0.3));
            n = n + 1;
            if hit { break; }
        }
        pure(n)
    "#,
    );
    let mut lengths = std::collections::BTreeSet::new();
    for seed in 0..200 {
        let (n, t) = run_seed(compiled.build(), seed);
        let n = n.as_usize().unwrap();
        lengths.insert(n);
        // Addresses only for the iterations that ran.
        assert_eq!(t.choices.len(), n, "seed {seed}");
        for i in 0..n {
            let hit = t.get_bool(&addr!("hit", i)).unwrap();
            // Misses until the last visited site, which is the first hit
            // (unless all ten missed).
            if i + 1 < n {
                assert!(!hit, "seed {seed}");
            } else if n < 10 {
                assert!(hit, "seed {seed}");
            }
        }
        assert!(!t.choices.contains_key(&addr!("hit", n)));
    }
    assert!(lengths.len() > 4, "{lengths:?}");

    // `break` leaves the innermost loop only.
    let src = r#"
        let count = 0;
        for i in 0..3 {
            for j in 0..10 {
                if j == 2 { break; }
                count = count + 1;
            }
            count = count + 100;
        }
        pure(count)
    "#;
    assert_eq!(value_of(src), Value::Int(306));
}

// ---------------------------------------------------------------------------
// Values
// ---------------------------------------------------------------------------

#[test]
fn sites_bind_and_programs_return_natural_types() {
    let src = r#"
        let b <- sample(addr!("b"), Bernoulli(0.5));
        let k <- sample(addr!("k"), Poisson(3.0));
        let n <- sample(addr!("n"), Binomial(10, 0.5));
        let z <- sample(addr!("z"), Categorical(0.2, 0.3, 0.5));
        let d <- sample(addr!("d"), DiscreteUniform(-3, 3));
        let x <- sample(addr!("x"), Normal(0.0, 1.0));
        pure(b)
    "#;
    let compiled = compile(src);
    let (v, t) = run_seed(compiled.build(), 3);
    assert_eq!(v.as_bool(), t.get_bool(&addr!("b")));
    assert!(matches!(v, Value::Bool(_)));
    assert!(t.get_u64(&addr!("k")).is_some() && t.get_u64(&addr!("n")).is_some());
    assert!(t.get_usize(&addr!("z")).is_some());
    assert!(t.get_i64(&addr!("d")).is_some());
    assert!(t.get_f64(&addr!("x")).is_some());

    for (ret, type_name) in [
        ("k", "u64"),
        ("n", "u64"),
        ("z", "usize"),
        ("d", "int"),
        ("x", "f64"),
        ("k + z", "int"),
        ("x * 2", "f64"),
        ("z == 2", "bool"),
    ] {
        let program = src.replace("pure(b)", &format!("pure({ret})"));
        let (v, _) = run_seed(compile(&program).build(), 3);
        assert_eq!(v.type_name(), type_name, "pure({ret}) gave {v:?}");
    }

    // Typed conveniences for hosts.
    let counts = compile(r#"let k <- sample(addr!("k"), Poisson(3.0)); pure(k)"#);
    let (k, t) = run_seed(counts.build().map(|v| v.as_u64().unwrap()), 8);
    assert_eq!(Some(k), t.get_u64(&addr!("k")));
    let (kf, _) = run_seed(counts.build_f64(), 8);
    assert_eq!(kf, k as f64);
    let (b, _) = run_seed(compile("pure(true)").build_f64(), 0);
    assert_eq!(b, 1.0);
}

#[test]
fn arithmetic_types_follow_the_documented_rules() {
    assert_eq!(eval("1 + 2"), Value::Int(3));
    assert_eq!(eval("1 + 2.0"), Value::F64(3.0));
    assert_eq!(eval("7 / 2"), Value::F64(3.5)); // `/` is float division
    assert_eq!(eval("-(2)"), Value::Int(-2));
    assert_eq!(eval("2 * -3"), Value::Int(-6));
    assert_eq!(eval("true + true"), Value::Int(2)); // bools read as 0/1
    assert_eq!(eval("floor(2.7)"), Value::F64(2.0));
    assert_eq!(eval("max(1, 2)"), Value::F64(2.0));
    assert_eq!(eval("1e3"), Value::F64(1000.0));
}

#[test]
fn array_literals_and_categoricals_of_any_length() {
    assert_eq!(eval("[1, 2.5, true]"), Value::from(vec![1.0, 2.5, 1.0]));
    assert_eq!(eval("[1.0, 2.0, 3.0].len()"), Value::Int(3));
    assert_eq!(eval("[1.0, 2.0][1]"), Value::F64(2.0));
    assert_eq!(eval("[]"), Value::from(Vec::<f64>::new()));

    // 100 categories, from data: the old 64-probability limit is gone.
    let probs = vec![0.01; 100];
    let compiled = compile_with(
        r#"let z <- sample(addr!("z"), Categorical(probs)); pure(z)"#,
        Data::new().with("probs", probs),
    );
    let mut seen = std::collections::BTreeSet::new();
    for seed in 0..300 {
        let (z, t) = run_seed(compiled.build(), seed);
        let z = z.as_usize().unwrap();
        assert!(z < 100);
        assert_eq!(t.get_usize(&addr!("z")), Some(z));
        assert!((t.log_prior - 0.01f64.ln()).abs() < 1e-12);
        seen.insert(z);
    }
    assert!(seen.iter().any(|&z| z >= 64), "{seen:?}");

    // Computed probabilities in a literal.
    let compiled = compile(
        r#"
        let p <- sample(addr!("p"), Beta(2.0, 2.0));
        let z <- sample(addr!("z"), Categorical([p, 1.0 - p]));
        pure(z)
    "#,
    );
    for seed in 0..20 {
        let (_, t) = run_seed(compiled.build(), seed);
        assert!(t.total_log_weight().is_finite());
    }
    assert!(compiled.take_warnings().is_empty());
}

#[test]
fn data_binds_arrays_and_scalars() {
    let data = Data::from_json(r#"{"n": 3, "sigma": 0.5, "strict": false, "ys": [0.1, 0.2, 0.3]}"#)
        .unwrap();
    let compiled = compile_with(
        r#"
        let mu <- sample(addr!("mu"), Normal(0.0, sigma));
        for i in 0..n {
            if strict || ys[i] > 0.15 {
                observe(addr!("y", i), Normal(mu, sigma), ys[i]);
            }
        }
        pure(mu)
    "#,
        data,
    );
    let (mu, t) = run_seed(compiled.build(), 2);
    let mu = mu.as_f64().unwrap();
    let normal = Normal::new(mu, 0.5).unwrap();
    let expected = normal.log_prob(&0.2) + normal.log_prob(&0.3);
    assert_eq!(t.log_likelihood.to_bits(), expected.to_bits());
}

// ---------------------------------------------------------------------------
// Addresses
// ---------------------------------------------------------------------------

#[test]
fn addresses_match_addr_macro_for_every_index_type() {
    let src = r#"
        let z <- sample(addr!("z"), Categorical(0.0, 1.0));
        let k <- sample(addr!("k"), Poisson(2.0));
        let b <- sample(addr!("b"), Bernoulli(1.0));
        observe(addr!("int", 3), Normal(0.0, 1.0), 0.0);
        let a <- sample(addr!("float", 2.0), Normal(0.0, 1.0));
        let c <- sample(addr!("frac", 0.5), Normal(0.0, 1.0));
        let d <- sample(addr!("neg", -4), Normal(0.0, 1.0));
        let e <- sample(addr!("bool", b), Normal(0.0, 1.0));
        let f <- sample(addr!("usize", z), Normal(0.0, 1.0));
        let g <- sample(addr!("u64", k), Normal(0.0, 1.0));
        let h <- sample(addr!("a#1"), Normal(0.0, 1.0));
        let m <- sample(addr!("x", "b#3"), Normal(0.0, 1.0));
        let q <- sample(addr!("scope::name", 3), Normal(0.0, 1.0));
        let r <- sample(addr!("back\\slash", 1), Normal(0.0, 1.0));
        pure(0)
    "#;
    // (`addr!("x", "b#3")` has a string index, which the language does not
    // have: the text parser must reject it.)
    assert!(Program::parse(src).is_err());
    let src = src.replace(
        "let m <- sample(addr!(\"x\", \"b#3\"), Normal(0.0, 1.0));\n",
        "",
    );
    let compiled = compile(&src);
    let (_, t) = run_seed(compiled.build(), 1);
    let k = t.get_u64(&addr!("k")).unwrap();
    for expected in [
        addr!("float", 2.0),
        addr!("float", 2),
        addr!("frac", 0.5),
        addr!("neg", -4),
        addr!("bool", true),
        addr!("usize", 1usize),
        addr!("u64", k),
        addr!("a#1"),
        scoped_addr!("scope", "name", "{}", 3),
        addr!("back\\slash", 1),
    ] {
        assert!(t.choices.contains_key(&expected), "missing {expected}");
    }
    assert_eq!(t.choices.len(), 12);
    assert!(compiled.take_warnings().is_empty());
}

// ---------------------------------------------------------------------------
// Total evaluation: soft errors reject the execution and warn
// ---------------------------------------------------------------------------

#[test]
fn runtime_soft_errors_reject_and_warn_without_panicking() {
    let data = || Data::new().with("y", vec![1.0, 2.0, 3.0]);
    for (src, needle) in [
        ("let v = y[10]; pure(v)", "out of bounds"),
        ("let v = y[-1]; pure(v)", "out of bounds"),
        ("let v = y[0.5]; pure(v)", "integer"),
        ("let v = y + 1.0; pure(v)", "numbers"),
        ("let v = 9223372036854775807 + 1; pure(v)", "overflow"),
        ("let v = -(-9223372036854775807 - 1); pure(v)", "overflow"),
        ("let v = y < 1.0; pure(v)", "compare"),
        ("let v = !y; pure(v)", "bool"),
        ("let v = (0.0 / 0.0) && true; pure(v)", "NaN"),
        ("let v = [1.0, y]; pure(v)", "array elements"),
        ("let v = y[1].len(); pure(v)", "len"),
        ("let v = exp(y); pure(v)", "number"),
        ("for i in 0..2.5 { } pure(0)", "range bound"),
        ("for i in 0..y { } pure(0)", "range bound"),
        ("if y { } pure(0)", "array"),
        ("if 0.0 / 0.0 { } pure(0)", "NaN"),
        ("factor(y); pure(0)", "factor"),
        ("pure(y[99])", "pure"),
        (r#"observe(addr!("k"), Poisson(3.0), -1); pure(0)"#, "u64"),
        (
            r#"observe(addr!("k"), Binomial(5, 0.5), 2.5); pure(0)"#,
            "u64",
        ),
        (
            r#"observe(addr!("z"), Categorical(0.5, 0.5), -2); pure(0)"#,
            "usize",
        ),
        (
            r#"observe(addr!("d"), DiscreteUniform(0, 3), 1.5); pure(0)"#,
            "i64",
        ),
        (
            r#"observe(addr!("x"), Normal(0.0, 1.0), y); pure(0)"#,
            "f64",
        ),
        (
            r#"observe(addr!("b"), Bernoulli(0.5), 0.0 / 0.0); pure(0)"#,
            "bool",
        ),
        (
            r#"observe(addr!("o", y), Normal(0.0, 1.0), 0.0); pure(0)"#,
            "scalar",
        ),
        (
            r#"observe(addr!("o"), Normal(0.0, -1.0), 0.0); pure(0)"#,
            "Normal",
        ),
        (
            r#"observe(addr!("o"), Normal(y[7], 1.0), 0.0); pure(0)"#,
            "out of bounds",
        ),
        (
            r#"let s <- sample(addr!("s", y), Normal(0.0, 1.0)); pure(s)"#,
            "scalar",
        ),
        (
            r#"let s <- sample(addr!("s"), Normal(y, 1.0)); pure(s)"#,
            "number",
        ),
        (
            r#"let s <- sample(addr!("s"), Binomial(-1, 0.5)); pure(s)"#,
            "Binomial",
        ),
        (
            r#"let s <- sample(addr!("s"), DiscreteUniform(0.5, 2)); pure(s)"#,
            "integers",
        ),
        (
            r#"let s <- sample(addr!("s"), Categorical(0.5, 0.6)); pure(s)"#,
            "Categorical",
        ),
    ] {
        let compiled = compile_with(src, data());
        let (_, t) = run_seed(compiled.build(), 0);
        assert_eq!(t.total_log_weight(), f64::NEG_INFINITY, "{src}");
        let warnings = compiled.take_warnings();
        assert_eq!(warnings.len(), 1, "{src}: {warnings:?}");
        assert!(warnings[0].contains(needle), "{src}: {warnings:?}");
    }
}

#[test]
fn failed_sites_keep_their_address_and_type() {
    // A site whose parameters are invalid is still recorded, at its address,
    // with a value of its declared type, so replay and scoring see a stable
    // structure; the execution continues and the weight is -inf.
    let compiled = compile(
        r#"
        let p = 1.5;
        let b <- sample(addr!("b"), Bernoulli(p));
        let k <- sample(addr!("k"), Poisson(-1.0));
        let z <- sample(addr!("z"), Categorical(2.0));
        let d <- sample(addr!("d"), DiscreteUniform(3, 1));
        let x <- sample(addr!("x"), Gamma(-1.0, 1.0));
        let after <- sample(addr!("after"), Normal(0.0, 1.0));
        pure(b)
    "#,
    );
    let (v, t) = run_seed(compiled.build(), 0);
    assert_eq!(v, Value::Bool(false));
    assert_eq!(t.get_bool(&addr!("b")), Some(false));
    assert_eq!(t.get_u64(&addr!("k")), Some(0));
    assert_eq!(t.get_usize(&addr!("z")), Some(0));
    assert_eq!(t.get_i64(&addr!("d")), Some(0));
    assert!(t.get_f64(&addr!("x")).is_some());
    assert!(t.get_f64(&addr!("after")).is_some());
    assert_eq!(t.log_factors, f64::NEG_INFINITY);
    assert_eq!(compiled.take_warnings().len(), 5);
}

#[test]
fn warnings_are_capped_and_drained() {
    let compiled = compile("for i in 0..500 { let v = [1.0][i + 1]; } pure(0)");
    let (_, t) = run_seed(compiled.build(), 0);
    assert_eq!(t.total_log_weight(), f64::NEG_INFINITY);
    assert_eq!(compiled.take_warnings().len(), 64);
    assert!(compiled.take_warnings().is_empty());
    // Clones share the buffer.
    let clone = compiled.clone();
    let _ = run_seed(clone.build(), 0);
    assert_eq!(compiled.take_warnings().len(), 64);
}

#[test]
fn long_programs_run_in_constant_stack() {
    // A long effect-free loop runs in the interpreter's own loop, and a long
    // run of sites through `run`'s trampoline: neither grows the stack.
    let handle = std::thread::Builder::new()
        .stack_size(256 * 1024)
        .spawn(|| {
            let compiled = compile("let s = 0; for i in 0..200000 { s = s + i; } pure(s)");
            assert_eq!(run_seed(compiled.build(), 0).0, Value::Int(19_999_900_000));
            let compiled = compile(
                r#"
                let s = 0.0;
                for i in 0..20000 {
                    let x <- sample(addr!("x", i), Normal(0.0, 1.0));
                    if x > 10.0 { break; }
                    s = s + x;
                }
                pure(s)
            "#,
            );
            let (s, t) = run_seed(compiled.build_f64(), 0);
            assert!(s.is_finite());
            assert_eq!(t.choices.len(), 20000);
        })
        .unwrap();
    handle
        .join()
        .expect("interpretation overflowed a small stack");
}
