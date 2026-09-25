//! The serialized program format (issue #65): its exact JSON shape, the
//! version check, strict decoding, structural validation, the nesting cap,
//! and the canonical text printer.

#![cfg(feature = "program")]

use fugue::program::{
    Addr, BinOp, Data, DistCall, Expr, Program, ProgramError, Registry, Stmt, Value,
    FORMAT_VERSION, MAX_NESTING,
};
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

/// The wire format, pinned: a change here is a format change, which needs a
/// `FORMAT_VERSION` bump (or a reason it is backward compatible).
const COIN_JSON: &str = concat!(
    r#"{"fugue_program":1,"body":["#,
    r#"{"stmt":"sample","var":"p","addr":{"name":"p"},"dist":{"name":"Beta","args":[{"op":"f64","value":2.0},{"op":"f64","value":2.0}]}},"#,
    r#"{"stmt":"for","var":"i","start":{"op":"int","value":0},"end":{"op":"len","array":{"op":"var","name":"data"}},"body":["#,
    r#"{"stmt":"observe","addr":{"name":"flip","index":{"op":"var","name":"i"}},"dist":{"name":"Bernoulli","args":[{"op":"var","name":"p"}]},"value":{"op":"index","array":{"op":"var","name":"data"},"index":{"op":"var","name":"i"}}}"#,
    r#"]}],"ret":{"op":"var","name":"p"}}"#
);

#[test]
fn json_form_is_pinned() {
    assert_eq!(FORMAT_VERSION, 1);
    let program = Program::parse(COIN).unwrap();
    assert_eq!(program.to_json().unwrap(), COIN_JSON);
    assert_eq!(Program::from_json(COIN_JSON).unwrap(), program);
    // Every statement and expression kind, and its tag.
    let all = Program::parse(
        r#"
        let x <- sample(addr!("x"), Normal(0.0, 1.0));
        let y = [1, -2.5, true][0];
        y = -x + y - x * y / 2.0;
        observe(addr!("o"), Normal(x, 1.0), y);
        factor(0.0);
        for i in 0..3 { if i == 1 { break; } }
        if !(x < y) && x <= y || x > y && x >= y && x != y { } else { y = abs(y); }
        pure(y)
    "#,
    )
    .unwrap();
    let json = all.to_json().unwrap();
    for tag in [
        r#""stmt":"sample""#,
        r#""stmt":"let""#,
        r#""stmt":"assign""#,
        r#""stmt":"observe""#,
        r#""stmt":"factor""#,
        r#""stmt":"for""#,
        r#""stmt":"if""#,
        r#""stmt":"break""#,
        r#""then":["#,
        r#""else":["#,
        r#""op":"f64""#,
        r#""op":"int""#,
        r#""op":"bool""#,
        r#""op":"var""#,
        r#""op":"array","items":["#,
        r#""op":"index""#,
        r#""op":"neg""#,
        r#""op":"not""#,
        r#""op":"add""#,
        r#""op":"sub""#,
        r#""op":"mul""#,
        r#""op":"div""#,
        r#""op":"eq""#,
        r#""op":"ne""#,
        r#""op":"lt""#,
        r#""op":"le""#,
        r#""op":"gt""#,
        r#""op":"ge""#,
        r#""op":"and""#,
        r#""op":"or""#,
        r#""op":"call","func":"abs""#,
    ] {
        assert!(json.contains(tag), "{tag} missing from {json}");
    }
    assert_eq!(Program::from_json(&json).unwrap(), all);
}

#[test]
fn a_program_written_as_json_runs() {
    // The module docs' example: a loop with state, written directly as JSON.
    let json = r#"{
      "fugue_program": 1,
      "body": [
        {"stmt": "let", "var": "tries", "value": {"op": "int", "value": 0}},
        {"stmt": "for", "var": "i",
         "start": {"op": "int", "value": 0}, "end": {"op": "int", "value": 10},
         "body": [
           {"stmt": "sample", "var": "hit", "addr": {"name": "hit", "index": {"op": "var", "name": "i"}},
            "dist": {"name": "Bernoulli", "args": [{"op": "var", "name": "p"}]}},
           {"stmt": "assign", "var": "tries",
            "value": {"op": "add", "lhs": {"op": "var", "name": "tries"}, "rhs": {"op": "int", "value": 1}}},
           {"stmt": "if", "cond": {"op": "var", "name": "hit"}, "then": [{"stmt": "break"}]}
         ]}
      ],
      "ret": {"op": "var", "name": "tries"}
    }"#;
    let from_json = Program::from_json(json).unwrap();
    let from_text = Program::parse(
        r#"
        let tries = 0;
        for i in 0..10 {
            let hit <- sample(addr!("hit", i), Bernoulli(p));
            tries = tries + 1;
            if hit { break; }
        }
        pure(tries)
    "#,
    )
    .unwrap();
    assert_eq!(from_json, from_text);
    let compiled = from_json
        .compile(&Registry::new(), &Data::new().with("p", 0.4))
        .unwrap();
    let (tries, trace) = runtime::handler::run(
        PriorHandler {
            rng: &mut StdRng::seed_from_u64(3),
            trace: Trace::default(),
        },
        compiled.build(),
    );
    assert_eq!(Some(trace.choices.len()), tries.as_usize());
}

#[test]
fn unknown_versions_are_rejected_with_a_clear_error() {
    let body = r#""body": [], "ret": {"op": "int", "value": 0}"#;
    for (version, found) in [("2", 2u64), ("0", 0), ("4294967296", 4_294_967_296)] {
        let doc = format!(r#"{{"fugue_program": {version}, {body}}}"#);
        let err = Program::from_json(&doc).unwrap_err();
        assert_eq!(
            err,
            ProgramError::UnsupportedVersion {
                found,
                supported: FORMAT_VERSION
            }
        );
        let msg = err.to_string();
        assert!(
            msg.contains(&format!("version {found}")) && msg.contains("reads version 1"),
            "{msg}"
        );
    }
    // The version is checked before anything else, so a newer program's new
    // syntax is reported as a version problem, not as a strange decode error.
    let newer = r#"{"fugue_program": 2, "body": [{"stmt": "while", "cond": {"op": "bool", "value": true}}],
                   "ret": {"op": "tuple", "items": []}, "meta": {}}"#;
    assert!(matches!(
        Program::from_json(newer),
        Err(ProgramError::UnsupportedVersion { found: 2, .. })
    ));
    // A missing or malformed version is a JSON error that names the field.
    for doc in [
        format!("{{{body}}}"),
        format!(r#"{{"fugue_program": "1", {body}}}"#),
        format!(r#"{{"fugue_program": -1, {body}}}"#),
        format!(r#"{{"fugue_program": 1.5, {body}}}"#),
    ] {
        match Program::from_json(&doc) {
            Err(ProgramError::Json(m)) => assert!(m.contains("fugue_program"), "{doc}: {m}"),
            other => panic!("{doc}: {other:?}"),
        }
    }
    // A program built in Rust with another version neither serializes nor
    // compiles.
    let mut p = Program::new(Vec::new(), Expr::Int { value: 0 });
    p.fugue_program = 7;
    assert!(matches!(
        p.to_json(),
        Err(ProgramError::UnsupportedVersion { found: 7, .. })
    ));
    assert!(p.compile(&Registry::new(), &Data::new()).is_err());
}

#[test]
fn json_decoding_is_strict() {
    let ok = Program::parse(COIN).unwrap().to_json().unwrap();
    for (from, to) in [
        // Unknown fields anywhere are errors, not silently different programs.
        (r#""var":"p","addr""#, r#""var":"p","extra":1,"addr""#),
        (r#"{"name":"flip","index""#, r#"{"name":"flip","indx""#),
        (
            r#""op":"var","name":"i"}"#,
            r#""op":"var","name":"i","x":0}"#,
        ),
        (r#""name":"Beta","args""#, r#""name":"Beta","argz""#),
        (r#""ret":"#, r#""meta":{},"ret":"#),
        // Unknown tags, missing fields, wrong types.
        (r#""op":"len""#, r#""op":"length""#),
        (r#""stmt":"for""#, r#""stmt":"loop""#),
        (r#""var":"i","#, ""),
        (r#""value":2.0"#, r#""value":"2.0""#),
        (r#""op":"int","value":0"#, r#""op":"int","value":0.5"#),
    ] {
        assert!(ok.contains(from), "{from}");
        let bad = ok.replacen(from, to, 1);
        match Program::from_json(&bad) {
            Err(ProgramError::Json(_)) => {}
            other => panic!("{bad}: {other:?}"),
        }
    }
    assert!(Program::from_json(&format!("{ok} x")).is_err());
    assert!(Program::from_json("[]").is_err());
    let brk = r#"{"fugue_program":1,"body":[{"stmt":"for","var":"i","start":{"op":"int","value":0},
        "end":{"op":"int","value":1},"body":[{"stmt":"break","label":"outer"}]}],"ret":{"op":"int","value":0}}"#;
    assert!(Program::from_json(brk).is_err());
}

#[test]
fn structural_validation_applies_to_every_entry_point() {
    let nan = Program::new(Vec::new(), Expr::F64 { value: f64::NAN });
    let inf = Program::new(
        vec![Stmt::Factor {
            logw: Expr::F64 {
                value: f64::NEG_INFINITY,
            },
        }],
        Expr::Int { value: 0 },
    );
    for p in [&nan, &inf] {
        assert!(matches!(p.to_json(), Err(ProgramError::Check(m)) if m.contains("finite")));
        assert!(p.compile(&Registry::new(), &Data::new()).is_err());
    }
    // Names must be identifiers, so text and JSON are equally expressive.
    for (field, name) in [
        ("var", "1x"),
        ("var", "if"),
        ("var", "a b"),
        ("func", "a-b"),
    ] {
        let doc = format!(
            r#"{{"fugue_program": 1, "body": [{{"stmt": "let", "{field}": "{name}", "value": {{"op": "int", "value": 0}}}}], "ret": {{"op": "int", "value": 0}}}}"#
        );
        let doc = if field == "func" {
            format!(
                r#"{{"fugue_program": 1, "body": [], "ret": {{"op": "call", "func": "{name}", "args": []}}}}"#
            )
        } else {
            doc
        };
        match Program::from_json(&doc) {
            Err(ProgramError::Check(m)) => assert!(m.contains("identifier"), "{m}"),
            other => panic!("{doc}: {other:?}"),
        }
    }
    // Address names are data: any string round-trips, escaped as `addr!` does.
    let odd = Program::new(
        vec![Stmt::Observe {
            addr: Addr {
                name: "a#b\\c \"q\"\n\u{e9}".to_string(),
                index: Some(Expr::Int { value: 1 }),
            },
            dist: DistCall {
                name: "Normal".to_string(),
                args: vec![Expr::F64 { value: 0.0 }, Expr::F64 { value: 1.0 }],
            },
            value: Expr::F64 { value: 0.5 },
        }],
        Expr::Int { value: 0 },
    );
    assert_eq!(Program::from_json(&odd.to_json().unwrap()).unwrap(), odd);
    assert_eq!(Program::parse(&odd.to_string()).unwrap(), odd);
}

fn nots(k: usize) -> String {
    format!("pure({}true)", "!".repeat(k))
}

#[test]
fn nesting_is_capped_the_same_way_by_every_entry_point() {
    // `pure(!..!true)`: the program object, then one level per node.
    let at_limit = Program::parse(&nots(MAX_NESTING - 2)).unwrap();
    let json = at_limit.to_json().unwrap();
    assert_eq!(Program::from_json(&json).unwrap(), at_limit);
    let compiled = at_limit.compile(&Registry::new(), &Data::new()).unwrap();
    let (v, _) = runtime::handler::run(
        PriorHandler {
            rng: &mut StdRng::seed_from_u64(0),
            trace: Trace::default(),
        },
        compiled.build(),
    );
    assert_eq!(v, Value::Bool(true)); // an even number of `!`

    // The text parser refuses as it reads, so the error has a line.
    let too_deep = nots(MAX_NESTING - 1);
    assert!(matches!(
        Program::parse(&too_deep),
        Err(ProgramError::Syntax { line: 1, message })
            if message.contains(&format!("at most {MAX_NESTING}"))
    ));
    // The same program built in Rust, or written as JSON, is refused too.
    let mut e = Expr::Bool { value: true };
    for _ in 0..MAX_NESTING - 1 {
        e = Expr::Not { arg: Box::new(e) };
    }
    let built = Program::new(Vec::new(), e);
    assert!(matches!(built.to_json(), Err(ProgramError::Check(_))));
    assert!(built.compile(&Registry::new(), &Data::new()).is_err());
    let deeper_json = json
        .replacen(
            r#"{"op":"not","arg":"#,
            r#"{"op":"not","arg":{"op":"not","arg":"#,
            1,
        )
        .replacen("}}", "}}}", 1);
    assert!(matches!(
        Program::from_json(&deeper_json),
        Err(ProgramError::Check(_))
    ));
    // Far deeper than serde_json will read: still an error, not a crash.
    let way_deeper = format!(
        r#"{{"fugue_program":1,"body":[],"ret":{}{}{}}}"#,
        r#"{"op":"neg","arg":"#.repeat(5000),
        r#"{"op":"int","value":1}"#,
        "}".repeat(5000)
    );
    assert!(matches!(
        Program::from_json(&way_deeper),
        Err(ProgramError::Json(_))
    ));
}

#[test]
fn the_text_printer_reads_back() {
    for src in [
        COIN,
        "pure(-(2.0))",
        "pure(-2.0)",
        "pure(-(-2))",
        "pure(--x)",
        "pure(-x.len())",
        "pure((-2.0).len())",
        "pure(-9223372036854775808)",
        "pure(1 - (2 - 3))",
        "pure((1 - 2) - 3)",
        "pure(1 / (2 * 3))",
        "pure((a < b) == (c < d))",
        "pure((a || b) && !(c && d))",
        "pure(!(a == b))",
        "pure([1, [2.5][0], x[1 + i]].len())",
        "pure(1e-7 + 1.5e300 + 0.1 + 5e-324)",
        r#"observe(addr!("q\"uo\\te#", 1), Normal(0.0, 1.0), 0.0); pure(0)"#,
        "for i in 0..3 { if i == 0 { } else if i == 1 { break; } else { let y = i; y = y + 1; } } pure(0)",
        "if a { if b { } } else { if c { } else { } } pure(0)",
        "let mut x = 1; x = x * -1; pure(x)",
    ] {
        let p = Program::parse(src).unwrap_or_else(|e| panic!("{src}: {e}"));
        let text = p.to_string();
        assert_eq!(Program::parse(&text).unwrap(), p, "{src}\n=>\n{text}");
        assert_eq!(Program::from_json(&p.to_json().unwrap()).unwrap(), p, "{src}");
    }
    // The printer is canonical: printing a reparsed program is a fixed point.
    let p = Program::parse(COIN).unwrap();
    assert_eq!(
        p.to_string(),
        "let p <- sample(addr!(\"p\"), Beta(2.0, 2.0));\n\
         for i in 0..data.len() {\n    observe(addr!(\"flip\", i), Bernoulli(p), data[i]);\n}\n\
         pure(p)"
    );
    // `FromStr` is `Program::parse`.
    assert_eq!(
        "pure(1)".parse::<Program>().unwrap(),
        Program::parse("pure(1)").unwrap()
    );
    // Building expressions generically.
    let e = Expr::binary(BinOp::Le, Expr::var("a"), Expr::Int { value: 3 });
    assert_eq!(e.to_string(), "a <= 3");
    assert_eq!(e.as_binary().map(|(op, _, _)| op.symbol()), Some("<="));
}
