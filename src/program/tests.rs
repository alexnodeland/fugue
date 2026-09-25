//! Property tests over random ASTs: the text printer and the parser are
//! inverse, the JSON form round-trips, and `nesting_depth` is exactly the
//! nesting of the serialized JSON (the quantity serde_json's recursion limit
//! bounds, which `MAX_NESTING` keeps every accepted program below).

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::*;

struct Gen {
    rng: StdRng,
}

impl Gen {
    fn pick<T: Clone>(&mut self, items: &[T]) -> T {
        items[self.rng.gen_range(0..items.len())].clone()
    }

    fn name(&mut self) -> String {
        self.pick(&[
            "a", "b", "x_1", "_t", "data", "sample", "addr", "len", "Normal",
        ])
        .to_string()
    }

    fn addr_name(&mut self) -> String {
        self.pick(&[
            "p",
            "flip",
            "a#1",
            "b\\c",
            "q\"uote",
            "new\nline",
            "t\tab",
            "\u{e9}t\u{e9}",
            "",
            "scope::x",
            "nul\0",
            "\u{1}",
        ])
        .to_string()
    }

    fn f64_literal(&mut self) -> f64 {
        if self.rng.gen_bool(0.5) {
            self.pick(&[
                0.0,
                -0.0,
                1.0,
                -2.5,
                0.1,
                1e-7,
                1e16,
                1e300,
                5e-324,
                123_456_789.125,
                f64::MAX,
                -f64::MAX,
                f64::MIN_POSITIVE,
            ])
        } else {
            loop {
                let x = f64::from_bits(self.rng.gen());
                if x.is_finite() {
                    return x;
                }
            }
        }
    }

    fn expr(&mut self, depth: usize) -> Expr {
        let leaf = depth == 0 || self.rng.gen_bool(0.3);
        let choice = if leaf {
            self.rng.gen_range(0..4)
        } else {
            self.rng.gen_range(4..11)
        };
        let sub = |g: &mut Gen| Box::new(g.expr(depth.saturating_sub(1)));
        match choice {
            0 => Expr::F64 {
                value: self.f64_literal(),
            },
            1 => {
                let random: i64 = self.rng.gen();
                Expr::Int {
                    value: self.pick(&[0, 7, -3, i64::MIN, i64::MAX, random]),
                }
            }
            2 => Expr::Bool {
                value: self.rng.gen(),
            },
            3 => Expr::Var { name: self.name() },
            4 => Expr::Array {
                items: (0..self.rng.gen_range(0..3))
                    .map(|_| self.expr(depth - 1))
                    .collect(),
            },
            5 => Expr::Index {
                array: sub(self),
                index: sub(self),
            },
            6 => Expr::Len { array: sub(self) },
            7 => Expr::Neg { arg: sub(self) },
            8 => Expr::Not { arg: sub(self) },
            9 => Expr::Call {
                func: self.name(),
                args: (0..self.rng.gen_range(0..3))
                    .map(|_| self.expr(depth - 1))
                    .collect(),
            },
            _ => {
                let op = self.pick(&[
                    BinOp::Add,
                    BinOp::Sub,
                    BinOp::Mul,
                    BinOp::Div,
                    BinOp::Eq,
                    BinOp::Ne,
                    BinOp::Lt,
                    BinOp::Le,
                    BinOp::Gt,
                    BinOp::Ge,
                    BinOp::And,
                    BinOp::Or,
                ]);
                let (l, r) = (self.expr(depth - 1), self.expr(depth - 1));
                Expr::binary(op, l, r)
            }
        }
    }

    fn addr(&mut self) -> Addr {
        Addr {
            name: self.addr_name(),
            index: self.rng.gen_bool(0.5).then(|| self.expr(2)),
        }
    }

    fn dist(&mut self) -> DistCall {
        DistCall {
            name: self.name(),
            args: (0..self.rng.gen_range(0..3))
                .map(|_| self.expr(2))
                .collect(),
        }
    }

    fn block(&mut self, depth: usize) -> Vec<Stmt> {
        (0..self.rng.gen_range(0..3))
            .map(|_| self.stmt(depth))
            .collect()
    }

    fn stmt(&mut self, depth: usize) -> Stmt {
        let top = if depth == 0 { 6 } else { 8 };
        match self.rng.gen_range(0..top) {
            0 => Stmt::Sample {
                var: self.name(),
                addr: self.addr(),
                dist: self.dist(),
            },
            1 => Stmt::Let {
                var: self.name(),
                value: self.expr(3),
            },
            2 => Stmt::Assign {
                var: self.name(),
                value: self.expr(3),
            },
            3 => Stmt::Observe {
                addr: self.addr(),
                dist: self.dist(),
                value: self.expr(2),
            },
            4 => Stmt::Factor { logw: self.expr(3) },
            5 => Stmt::Break {},
            6 => Stmt::For {
                var: self.name(),
                start: self.expr(2),
                end: self.expr(2),
                body: self.block(depth - 1),
            },
            _ => Stmt::If {
                cond: self.expr(3),
                then_branch: self.block(depth - 1),
                else_branch: match self.rng.gen_range(0..3) {
                    0 => Vec::new(),
                    1 => vec![self.stmt_if(depth - 1)],
                    _ => self.block(depth - 1),
                },
            },
        }
    }

    fn stmt_if(&mut self, depth: usize) -> Stmt {
        Stmt::If {
            cond: self.expr(2),
            then_branch: self.block(depth),
            else_branch: if depth > 0 && self.rng.gen_bool(0.5) {
                vec![self.stmt_if(depth - 1)]
            } else {
                Vec::new()
            },
        }
    }

    fn program(&mut self) -> Program {
        let body = (0..self.rng.gen_range(0..5))
            .map(|_| self.stmt(3))
            .collect();
        Program::new(body, self.expr(4))
    }
}

fn json_depth(v: &serde_json::Value) -> usize {
    match v {
        serde_json::Value::Array(items) => 1 + items.iter().map(json_depth).max().unwrap_or(0),
        serde_json::Value::Object(map) => 1 + map.values().map(json_depth).max().unwrap_or(0),
        _ => 0,
    }
}

#[test]
fn random_programs_round_trip_through_text_and_json() {
    let mut g = Gen {
        rng: StdRng::seed_from_u64(0x65),
    };
    for case in 0..2000 {
        let p = g.program();
        let text = p.to_string();
        let reparsed =
            Program::parse(&text).unwrap_or_else(|e| panic!("case {case}: {e}\n{text}\n{p:?}"));
        assert_eq!(reparsed, p, "case {case}: text round trip\n{text}");
        // Literal bits survive too (`PartialEq` on f64 cannot see -0.0).
        assert_eq!(reparsed.to_string(), text, "case {case}");

        let json = p.to_json().unwrap();
        let back = Program::from_json(&json).unwrap();
        assert_eq!(back, p, "case {case}: JSON round trip\n{json}");
        assert_eq!(back.to_json().unwrap(), json, "case {case}");

        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(p.nesting_depth(), json_depth(&value), "case {case}\n{json}");
    }
}

// ---------------------------------------------------------------------------
// The nesting cap, shape by shape
// ---------------------------------------------------------------------------

fn var(name: &str) -> Expr {
    Expr::var(name)
}

fn int(value: i64) -> Expr {
    Expr::Int { value }
}

fn nest(k: usize, leaf: Expr, wrap: impl Fn(Expr) -> Expr) -> Expr {
    (0..k).fold(leaf, |e, _| wrap(e))
}

fn ret(e: Expr) -> Program {
    Program::new(Vec::new(), e)
}

fn in_block(k: usize, inner: Vec<Stmt>, wrap: impl Fn(Vec<Stmt>) -> Stmt) -> Vec<Stmt> {
    (0..k).fold(inner, |body, _| vec![wrap(body)])
}

fn normal(mu: Expr) -> DistCall {
    DistCall {
        name: "Normal".to_string(),
        args: vec![mu, Expr::F64 { value: 1.0 }],
    }
}

/// Every entry point agrees on the cap for one shape of nesting: the
/// deepest program within `MAX_NESTING` is accepted by the text parser (from
/// its printed form: no false rejections), by both JSON directions, and by
/// `compile`, and runs; one step deeper is refused by all of them.
fn check_cap(shape: &str, build: impl Fn(usize) -> Program) {
    let mut k = 0;
    while build(k + 1).nesting_depth() <= MAX_NESTING {
        k += 1;
    }
    let (ok, over) = (build(k), build(k + 1));
    assert!(
        ok.nesting_depth() > MAX_NESTING - 4,
        "{shape}: {}",
        ok.nesting_depth()
    );

    assert_eq!(Program::parse(&ok.to_string()).as_ref(), Ok(&ok), "{shape}");
    let json = ok.to_json().unwrap_or_else(|e| panic!("{shape}: {e}"));
    assert_eq!(Program::from_json(&json).as_ref(), Ok(&ok), "{shape}");
    let data = Data::new().with("x", vec![0.0]);
    let compiled = ok
        .compile(&Registry::new(), &data)
        .unwrap_or_else(|e| panic!("{shape}: {e}"));
    let mut rng = StdRng::seed_from_u64(0);
    let _ = crate::runtime::handler::run(
        crate::runtime::interpreters::PriorHandler {
            rng: &mut rng,
            trace: crate::runtime::trace::Trace::default(),
        },
        compiled.build(),
    );

    assert!(Program::parse(&over.to_string()).is_err(), "{shape}");
    assert!(over.to_json().is_err(), "{shape}");
    let unchecked = serde_json::to_string(&over).unwrap();
    assert!(Program::from_json(&unchecked).is_err(), "{shape}");
    assert!(over.compile(&Registry::new(), &data).is_err(), "{shape}");
}

fn factor_zero() -> Vec<Stmt> {
    vec![Stmt::Factor {
        logw: Expr::F64 { value: 0.0 },
    }]
}

fn cap_shapes() {
    check_cap("prefix run", |k| {
        ret(nest(k, Expr::Bool { value: true }, |e| Expr::Not {
            arg: Box::new(e),
        }))
    });
    check_cap("left chain", |k| {
        ret(nest(k, int(1), |e| Expr::binary(BinOp::Add, e, int(1))))
    });
    check_cap("right chain (parenthesized)", |k| {
        ret(nest(k, int(1), |e| Expr::binary(BinOp::Sub, int(1), e)))
    });
    check_cap("chain with a deep last operand", |k| {
        let deep = nest(k, var("x"), |e| Expr::Len { array: Box::new(e) });
        let chain = nest(k / 2, int(1), |e| Expr::binary(BinOp::Mul, e, int(2)));
        ret(Expr::binary(BinOp::Add, chain, deep))
    });
    check_cap("mixed precedence", |k| {
        ret(nest(k, var("x"), |e| {
            let sum = Expr::binary(BinOp::Add, e, int(1));
            Expr::binary(BinOp::Lt, int(0), sum)
        }))
    });
    check_cap("calls", |k| {
        ret(nest(k, Expr::F64 { value: 0.5 }, |e| Expr::Call {
            func: "exp".to_string(),
            args: vec![e],
        }))
    });
    check_cap("arrays", |k| {
        ret(nest(k, int(1), |e| Expr::Array { items: vec![e] }))
    });
    check_cap("index chain", |k| {
        ret(nest(k, var("x"), |e| Expr::Index {
            array: Box::new(e),
            index: Box::new(int(0)),
        }))
    });
    check_cap("nested indices", |k| {
        ret(nest(k, int(0), |e| Expr::Index {
            array: Box::new(var("x")),
            index: Box::new(e),
        }))
    });
    check_cap("negation of negative literals", |k| {
        ret(nest(k, Expr::F64 { value: -2.0 }, |e| Expr::Neg {
            arg: Box::new(e),
        }))
    });
    check_cap("nested ifs", |k| {
        let body = in_block(k, factor_zero(), |then_branch| Stmt::If {
            cond: Expr::Bool { value: true },
            then_branch,
            else_branch: Vec::new(),
        });
        Program::new(body, int(0))
    });
    check_cap("else-if ladder", |k| {
        let body = in_block(k, factor_zero(), |else_branch| Stmt::If {
            cond: Expr::Bool { value: false },
            then_branch: Vec::new(),
            else_branch,
        });
        Program::new(body, int(0))
    });
    check_cap("nested loops with a deep value", |k| {
        let value = nest(k, int(0), |e| Expr::binary(BinOp::Sub, int(1), e));
        let leaf = vec![
            Stmt::Let {
                var: "v".to_string(),
                value,
            },
            Stmt::Break {},
        ];
        let body = in_block(k / 4, leaf, |body| Stmt::For {
            var: "i".to_string(),
            start: int(0),
            end: int(1),
            body,
        });
        Program::new(body, int(0))
    });
    check_cap("deep dist argument and address index", |k| {
        let arg = nest(k, Expr::F64 { value: 0.0 }, |e| Expr::Neg {
            arg: Box::new(e),
        });
        let index = nest(k, int(0), |e| Expr::Neg { arg: Box::new(e) });
        let sample = Stmt::Sample {
            var: "s".to_string(),
            addr: Addr {
                name: "s".to_string(),
                index: Some(index.clone()),
            },
            dist: normal(arg.clone()),
        };
        let observe = Stmt::Observe {
            addr: Addr {
                name: "o".to_string(),
                index: Some(index),
            },
            dist: normal(arg.clone()),
            value: arg,
        };
        let body = in_block(k / 8, vec![sample, observe], |then_branch| Stmt::If {
            cond: Expr::Bool { value: true },
            then_branch,
            else_branch: Vec::new(),
        });
        Program::new(body, int(0))
    });
}

#[test]
fn every_entry_point_enforces_the_same_nesting_cap() {
    // On a 1 MiB stack, wasm's default, in whatever profile the tests use.
    std::thread::Builder::new()
        .stack_size(1024 * 1024)
        .spawn(cap_shapes)
        .unwrap()
        .join()
        .expect("programs at the nesting cap must fit a 1 MiB stack");
}
