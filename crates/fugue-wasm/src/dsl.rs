//! A `prob!`-subset model language, interpreted into real fugue [`Model`]s.
//!
//! The playground (and the WASM-backed doc widgets) need to build models at
//! runtime from user-editable source. This module parses a Rust-ish subset of
//! the `prob!` macro language and folds it into fugue's actual combinators —
//! `sample`, `observe`, `factor`, `pure`, `bind` — so everything downstream
//! (handlers, traces, MH/HMC/SMC, diagnostics) is the real crate, not a
//! mirror. Only model *construction* is interpreted.
//!
//! Supported grammar (statements end with `;`, the program ends with `pure`):
//!
//! ```text
//! let p <- sample(addr!("p"), Beta(2.0, 2.0));
//! let mu = 2.0 * p - 1.0;
//! for i in 0..y.len() {
//!     observe(addr!("y", i), Normal(mu, 0.8), y[i]);
//! }
//! factor(-0.5 * mu * mu);
//! pure(p)
//! ```
//!
//! Rust-style `Dist::new(...)` and a trailing `.unwrap()` are accepted and
//! ignored sugar, so examples copied from the docs parse unchanged. Data
//! arrays come from a JSON object (`{"y": [...], "x": [...]}`) or a bare
//! JSON array (bound to `data`). Addresses are built with fugue's own
//! `make_name`/`make_indexed`, so they are byte-identical to `addr!(..)` in
//! compiled Rust.
//!
//! Total-evaluation policy: model-build code runs inside `bind` continuations
//! where errors cannot propagate as `Result` (and wasm has no unwinding), so
//! runtime soft errors (index out of bounds, invalid distribution parameters)
//! degrade to `factor(-inf)` — "this parameter region is impossible" — and
//! push a warning the host can surface. Static errors (syntax, unknown
//! variables/distributions, wrong arities) are rejected at compile time.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::{Arc, Mutex};

use fugue::core::address::{make_indexed, make_name};
use fugue::*;

// ---------------------------------------------------------------------------
// Values
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub enum Value {
    F64(f64),
    Int(i64),
    Bool(bool),
    Arr(Arc<Vec<f64>>),
}

impl Value {
    pub fn as_f64(&self) -> f64 {
        match self {
            Value::F64(x) => *x,
            Value::Int(i) => *i as f64,
            Value::Bool(b) => {
                if *b {
                    1.0
                } else {
                    0.0
                }
            }
            Value::Arr(_) => f64::NAN,
        }
    }

    fn as_index(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            Value::F64(x) if x.fract() == 0.0 && x.is_finite() => Some(*x as i64),
            _ => None,
        }
    }

    fn as_bool(&self) -> bool {
        match self {
            Value::Bool(b) => *b,
            other => other.as_f64() != 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// AST
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub enum Expr {
    Num(f64),
    Int(i64),
    Bool(bool),
    Var(String),
    Index(Box<Expr>, Box<Expr>),
    Len(Box<Expr>),
    Neg(Box<Expr>),
    Bin(char, Box<Expr>, Box<Expr>),
    Call(String, Vec<Expr>),
}

#[derive(Clone, Debug)]
pub struct AddrSpec {
    name: String,
    index: Option<Expr>,
}

#[derive(Clone, Debug)]
pub struct DistSpec {
    name: String,
    args: Vec<Expr>,
}

#[derive(Clone, Debug)]
pub enum Stmt {
    SampleLet {
        name: String,
        addr: AddrSpec,
        dist: DistSpec,
    },
    LetExpr {
        name: String,
        expr: Expr,
    },
    Observe {
        addr: AddrSpec,
        dist: DistSpec,
        value: Expr,
    },
    Factor(Expr),
    For {
        var: String,
        lo: Expr,
        hi: Expr,
        body: Arc<Vec<Stmt>>,
    },
}

#[derive(Debug)]
pub struct Program {
    stmts: Arc<Vec<Stmt>>,
    ret: Arc<Expr>,
}

// ---------------------------------------------------------------------------
// Lexer
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Ident(String),
    Num(f64, bool), // value, had_dot_or_exp
    Str(String),
    Sym(&'static str),
    Eof,
}

struct Lexer {
    toks: Vec<(Tok, usize)>, // (token, byte offset for error reporting)
    pos: usize,
    src_line_starts: Vec<usize>,
}

const SYMS: &[&str] = &[
    "<-", "::", "..", "(", ")", "{", "}", "[", "]", ",", ";", "+", "-", "*", "/", "=", ".", "!",
];

impl Lexer {
    fn new(src: &str) -> Result<Lexer, String> {
        let bytes = src.as_bytes();
        let mut toks = Vec::new();
        let mut i = 0;
        let line_starts = std::iter::once(0)
            .chain(src.match_indices('\n').map(|(p, _)| p + 1))
            .collect();
        'outer: while i < bytes.len() {
            let c = bytes[i] as char;
            if c.is_whitespace() {
                i += 1;
                continue;
            }
            if c == '/' && bytes.get(i + 1) == Some(&b'/') {
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
                continue;
            }
            if c == '"' {
                let start = i + 1;
                let mut j = start;
                while j < bytes.len() && bytes[j] != b'"' {
                    j += 1;
                }
                if j >= bytes.len() {
                    return Err(format!("unterminated string at byte {i}"));
                }
                toks.push((Tok::Str(src[start..j].to_string()), i));
                i = j + 1;
                continue;
            }
            if c.is_ascii_digit() {
                let start = i;
                let mut seen_dot = false;
                while i < bytes.len() {
                    let d = bytes[i] as char;
                    if d.is_ascii_digit() {
                        i += 1;
                    } else if d == '.'
                        && !seen_dot
                        && bytes.get(i + 1).is_some_and(|b| b.is_ascii_digit())
                    {
                        // Only consume '.' as a decimal point when a digit
                        // follows: `0..n` must lex as Num(0) '..' ident.
                        seen_dot = true;
                        i += 1;
                    } else if (d == 'e' || d == 'E')
                        && bytes
                            .get(i + 1)
                            .is_some_and(|&b| b.is_ascii_digit() || b == b'-' || b == b'+')
                    {
                        seen_dot = true;
                        i += 2;
                        while i < bytes.len() && bytes[i].is_ascii_digit() {
                            i += 1;
                        }
                        break;
                    } else {
                        break;
                    }
                }
                let text = &src[start..i];
                let v: f64 = text
                    .parse()
                    .map_err(|_| format!("bad number `{text}` at byte {start}"))?;
                toks.push((Tok::Num(v, seen_dot), start));
                continue;
            }
            if c.is_ascii_alphabetic() || c == '_' {
                let start = i;
                while i < bytes.len()
                    && ((bytes[i] as char).is_ascii_alphanumeric() || bytes[i] == b'_')
                {
                    i += 1;
                }
                toks.push((Tok::Ident(src[start..i].to_string()), start));
                continue;
            }
            for s in SYMS {
                if src[i..].starts_with(s) {
                    toks.push((Tok::Sym(s), i));
                    i += s.len();
                    continue 'outer;
                }
            }
            return Err(format!("unexpected character `{c}` at byte {i}"));
        }
        toks.push((Tok::Eof, src.len()));
        Ok(Lexer {
            toks,
            pos: 0,
            src_line_starts: line_starts,
        })
    }

    fn line_of(&self, byte: usize) -> usize {
        match self.src_line_starts.binary_search(&byte) {
            Ok(l) => l + 1,
            Err(l) => l,
        }
    }

    fn peek(&self) -> &Tok {
        &self.toks[self.pos].0
    }

    fn peek2(&self) -> &Tok {
        &self.toks[(self.pos + 1).min(self.toks.len() - 1)].0
    }

    fn next(&mut self) -> Tok {
        let t = self.toks[self.pos].0.clone();
        if self.pos < self.toks.len() - 1 {
            self.pos += 1;
        }
        t
    }

    fn err(&self, what: &str) -> String {
        let (tok, byte) = &self.toks[self.pos];
        format!(
            "line {}: expected {what}, found {}",
            self.line_of(*byte),
            match tok {
                Tok::Ident(s) => format!("`{s}`"),
                Tok::Num(v, _) => format!("`{v}`"),
                Tok::Str(s) => format!("\"{s}\""),
                Tok::Sym(s) => format!("`{s}`"),
                Tok::Eof => "end of input".to_string(),
            }
        )
    }

    fn expect_sym(&mut self, s: &'static str) -> Result<(), String> {
        if self.peek() == &Tok::Sym(s) {
            self.next();
            Ok(())
        } else {
            Err(self.err(&format!("`{s}`")))
        }
    }

    fn expect_ident(&mut self) -> Result<String, String> {
        match self.peek().clone() {
            Tok::Ident(s) => {
                self.next();
                Ok(s)
            }
            _ => Err(self.err("an identifier")),
        }
    }

    fn eat_kw(&mut self, kw: &str) -> bool {
        if matches!(self.peek(), Tok::Ident(s) if s == kw) {
            self.next();
            true
        } else {
            false
        }
    }
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

fn parse_program(src: &str) -> Result<Program, String> {
    let mut lx = Lexer::new(src)?;
    let mut stmts = Vec::new();
    loop {
        if lx.eat_kw("pure") {
            lx.expect_sym("(")?;
            let ret = parse_expr(&mut lx)?;
            lx.expect_sym(")")?;
            let _ = lx.expect_sym(";"); // optional trailing semicolon
            if lx.peek() != &Tok::Eof {
                return Err(lx.err("end of input after `pure(..)`"));
            }
            return Ok(Program {
                stmts: Arc::new(stmts),
                ret: Arc::new(ret),
            });
        }
        if lx.peek() == &Tok::Eof {
            return Err("model must end with `pure(<expr>)`".to_string());
        }
        stmts.push(parse_stmt(&mut lx)?);
    }
}

fn parse_stmt(lx: &mut Lexer) -> Result<Stmt, String> {
    if lx.eat_kw("let") {
        let name = lx.expect_ident()?;
        if lx.peek() == &Tok::Sym("<-") {
            lx.next();
            if !lx.eat_kw("sample") {
                return Err(lx.err("`sample`"));
            }
            lx.expect_sym("(")?;
            let addr = parse_addr(lx)?;
            lx.expect_sym(",")?;
            let dist = parse_dist(lx)?;
            lx.expect_sym(")")?;
            lx.expect_sym(";")?;
            return Ok(Stmt::SampleLet { name, addr, dist });
        }
        lx.expect_sym("=")?;
        let expr = parse_expr(lx)?;
        lx.expect_sym(";")?;
        return Ok(Stmt::LetExpr { name, expr });
    }
    if lx.eat_kw("observe") {
        lx.expect_sym("(")?;
        let addr = parse_addr(lx)?;
        lx.expect_sym(",")?;
        let dist = parse_dist(lx)?;
        lx.expect_sym(",")?;
        let value = parse_expr(lx)?;
        lx.expect_sym(")")?;
        lx.expect_sym(";")?;
        return Ok(Stmt::Observe { addr, dist, value });
    }
    if lx.eat_kw("factor") {
        lx.expect_sym("(")?;
        let e = parse_expr(lx)?;
        lx.expect_sym(")")?;
        lx.expect_sym(";")?;
        return Ok(Stmt::Factor(e));
    }
    if lx.eat_kw("for") {
        let var = lx.expect_ident()?;
        if !lx.eat_kw("in") {
            return Err(lx.err("`in`"));
        }
        let lo = parse_expr(lx)?;
        lx.expect_sym("..")?;
        let hi = parse_expr(lx)?;
        lx.expect_sym("{")?;
        let mut body = Vec::new();
        while lx.peek() != &Tok::Sym("}") {
            if lx.peek() == &Tok::Eof {
                return Err(lx.err("`}`"));
            }
            body.push(parse_stmt(lx)?);
        }
        lx.next(); // }
        return Ok(Stmt::For {
            var,
            lo,
            hi,
            body: Arc::new(body),
        });
    }
    Err(lx.err("a statement (`let`, `observe`, `factor`, `for`, or `pure`)"))
}

fn parse_addr(lx: &mut Lexer) -> Result<AddrSpec, String> {
    if !lx.eat_kw("addr") {
        return Err(lx.err("`addr!(..)`"));
    }
    lx.expect_sym("!")?;
    lx.expect_sym("(")?;
    let name = match lx.peek().clone() {
        Tok::Str(s) => {
            lx.next();
            s
        }
        _ => return Err(lx.err("a string literal address name")),
    };
    let index = if lx.peek() == &Tok::Sym(",") {
        lx.next();
        Some(parse_expr(lx)?)
    } else {
        None
    };
    lx.expect_sym(")")?;
    Ok(AddrSpec { name, index })
}

fn parse_dist(lx: &mut Lexer) -> Result<DistSpec, String> {
    let name = lx.expect_ident()?;
    if lx.peek() == &Tok::Sym("::") {
        lx.next();
        let m = lx.expect_ident()?;
        if m != "new" {
            return Err(format!("unknown distribution constructor `{name}::{m}`"));
        }
    }
    lx.expect_sym("(")?;
    let mut args = Vec::new();
    if lx.peek() != &Tok::Sym(")") {
        loop {
            args.push(parse_expr(lx)?);
            if lx.peek() == &Tok::Sym(",") {
                lx.next();
            } else {
                break;
            }
        }
    }
    lx.expect_sym(")")?;
    // Optional Rust-ism: `.unwrap()`
    if lx.peek() == &Tok::Sym(".") && matches!(lx.peek2(), Tok::Ident(s) if s == "unwrap") {
        lx.next();
        lx.next();
        lx.expect_sym("(")?;
        lx.expect_sym(")")?;
    }
    dist_arity(&name)
        .ok_or_else(|| format!("unknown distribution `{name}`"))
        .and_then(|(lo, hi)| {
            if args.len() < lo || args.len() > hi {
                Err(format!(
                    "`{name}` takes {} argument(s), got {}",
                    if lo == hi {
                        lo.to_string()
                    } else {
                        format!("{lo}..={hi}")
                    },
                    args.len()
                ))
            } else {
                Ok(DistSpec { name, args })
            }
        })
}

fn parse_expr(lx: &mut Lexer) -> Result<Expr, String> {
    parse_additive(lx)
}

fn parse_additive(lx: &mut Lexer) -> Result<Expr, String> {
    let mut lhs = parse_multiplicative(lx)?;
    loop {
        let op = match lx.peek() {
            Tok::Sym("+") => '+',
            Tok::Sym("-") => '-',
            _ => return Ok(lhs),
        };
        lx.next();
        let rhs = parse_multiplicative(lx)?;
        lhs = Expr::Bin(op, Box::new(lhs), Box::new(rhs));
    }
}

fn parse_multiplicative(lx: &mut Lexer) -> Result<Expr, String> {
    let mut lhs = parse_unary(lx)?;
    loop {
        let op = match lx.peek() {
            Tok::Sym("*") => '*',
            Tok::Sym("/") => '/',
            _ => return Ok(lhs),
        };
        lx.next();
        let rhs = parse_unary(lx)?;
        lhs = Expr::Bin(op, Box::new(lhs), Box::new(rhs));
    }
}

fn parse_unary(lx: &mut Lexer) -> Result<Expr, String> {
    if lx.peek() == &Tok::Sym("-") {
        lx.next();
        return Ok(Expr::Neg(Box::new(parse_unary(lx)?)));
    }
    parse_postfix(lx)
}

fn parse_postfix(lx: &mut Lexer) -> Result<Expr, String> {
    let mut e = parse_primary(lx)?;
    loop {
        match lx.peek() {
            Tok::Sym("[") => {
                lx.next();
                let idx = parse_expr(lx)?;
                lx.expect_sym("]")?;
                e = Expr::Index(Box::new(e), Box::new(idx));
            }
            Tok::Sym(".") => {
                // Only `.len()` — and NOT when the next token would start
                // `.unwrap()` (handled by parse_dist) or a number.
                if let Tok::Ident(m) = lx.peek2().clone() {
                    if m == "len" {
                        lx.next();
                        lx.next();
                        lx.expect_sym("(")?;
                        lx.expect_sym(")")?;
                        e = Expr::Len(Box::new(e));
                        continue;
                    }
                }
                return Ok(e);
            }
            _ => return Ok(e),
        }
    }
}

const MATH_FNS: &[(&str, usize)] = &[
    ("exp", 1),
    ("ln", 1),
    ("log", 1),
    ("sqrt", 1),
    ("abs", 1),
    ("floor", 1),
    ("sin", 1),
    ("cos", 1),
    ("tanh", 1),
    ("pow", 2),
    ("min", 2),
    ("max", 2),
];

fn parse_primary(lx: &mut Lexer) -> Result<Expr, String> {
    match lx.peek().clone() {
        Tok::Num(v, dotted) => {
            lx.next();
            if dotted {
                Ok(Expr::Num(v))
            } else {
                Ok(Expr::Int(v as i64))
            }
        }
        Tok::Sym("(") => {
            lx.next();
            let e = parse_expr(lx)?;
            lx.expect_sym(")")?;
            Ok(e)
        }
        Tok::Ident(name) => {
            lx.next();
            match name.as_str() {
                "true" => return Ok(Expr::Bool(true)),
                "false" => return Ok(Expr::Bool(false)),
                _ => {}
            }
            if lx.peek() == &Tok::Sym("(") {
                let arity = MATH_FNS
                    .iter()
                    .find(|(f, _)| *f == name)
                    .map(|(_, a)| *a)
                    .ok_or_else(|| format!("unknown function `{name}`"))?;
                lx.next();
                let mut args = Vec::new();
                if lx.peek() != &Tok::Sym(")") {
                    loop {
                        args.push(parse_expr(lx)?);
                        if lx.peek() == &Tok::Sym(",") {
                            lx.next();
                        } else {
                            break;
                        }
                    }
                }
                lx.expect_sym(")")?;
                if args.len() != arity {
                    return Err(format!("`{name}` takes {arity} argument(s)"));
                }
                Ok(Expr::Call(name, args))
            } else {
                Ok(Expr::Var(name))
            }
        }
        _ => Err(lx.err("an expression")),
    }
}

// ---------------------------------------------------------------------------
// Static validation (names, so runtime never sees an unknown identifier)
// ---------------------------------------------------------------------------

fn check_expr(e: &Expr, scope: &mut Vec<String>) -> Result<(), String> {
    match e {
        Expr::Var(n) => {
            if scope.iter().any(|s| s == n) {
                Ok(())
            } else {
                Err(format!("unknown variable `{n}`"))
            }
        }
        Expr::Num(_) | Expr::Int(_) | Expr::Bool(_) => Ok(()),
        Expr::Index(a, i) => check_expr(a, scope).and_then(|_| check_expr(i, scope)),
        Expr::Len(a) | Expr::Neg(a) => check_expr(a, scope),
        Expr::Bin(_, a, b) => check_expr(a, scope).and_then(|_| check_expr(b, scope)),
        Expr::Call(_, args) => args.iter().try_for_each(|a| check_expr(a, scope)),
    }
}

fn check_stmts(stmts: &[Stmt], scope: &mut Vec<String>) -> Result<(), String> {
    for s in stmts {
        match s {
            Stmt::SampleLet { name, addr, dist } => {
                if let Some(ix) = &addr.index {
                    check_expr(ix, scope)?;
                }
                dist.args.iter().try_for_each(|a| check_expr(a, scope))?;
                scope.push(name.clone());
            }
            Stmt::LetExpr { name, expr } => {
                check_expr(expr, scope)?;
                scope.push(name.clone());
            }
            Stmt::Observe { addr, dist, value } => {
                if let Some(ix) = &addr.index {
                    check_expr(ix, scope)?;
                }
                dist.args.iter().try_for_each(|a| check_expr(a, scope))?;
                check_expr(value, scope)?;
            }
            Stmt::Factor(e) => check_expr(e, scope)?,
            Stmt::For { var, lo, hi, body } => {
                check_expr(lo, scope)?;
                check_expr(hi, scope)?;
                let depth = scope.len();
                scope.push(var.clone());
                check_stmts(body, scope)?;
                scope.truncate(depth);
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct Env {
    vars: HashMap<String, Value>,
    warnings: Arc<Mutex<Vec<String>>>,
}

impl Env {
    fn warn(&self, msg: String) {
        if let Ok(mut w) = self.warnings.lock() {
            if w.len() < 64 {
                w.push(msg);
            }
        }
    }
}

fn eval(e: &Expr, env: &Env) -> Value {
    match e {
        Expr::Num(v) => Value::F64(*v),
        Expr::Int(i) => Value::Int(*i),
        Expr::Bool(b) => Value::Bool(*b),
        Expr::Var(n) => env.vars.get(n).cloned().unwrap_or_else(|| {
            env.warn(format!("unknown variable `{n}` at runtime"));
            Value::F64(f64::NAN)
        }),
        Expr::Index(a, ie) => {
            let arr = eval(a, env);
            let idx = eval(ie, env);
            match (&arr, idx.as_index()) {
                (Value::Arr(v), Some(i)) if i >= 0 && (i as usize) < v.len() => {
                    Value::F64(v[i as usize])
                }
                (Value::Arr(v), Some(i)) => {
                    env.warn(format!("index {i} out of bounds (len {})", v.len()));
                    Value::F64(f64::NAN)
                }
                _ => {
                    env.warn("indexing a non-array value".to_string());
                    Value::F64(f64::NAN)
                }
            }
        }
        Expr::Len(a) => match eval(a, env) {
            Value::Arr(v) => Value::Int(v.len() as i64),
            _ => {
                env.warn("`.len()` on a non-array value".to_string());
                Value::Int(0)
            }
        },
        Expr::Neg(a) => match eval(a, env) {
            Value::Int(i) => Value::Int(-i),
            other => Value::F64(-other.as_f64()),
        },
        Expr::Bin(op, a, b) => {
            let va = eval(a, env);
            let vb = eval(b, env);
            if let (Value::Int(x), Value::Int(y), '+' | '-' | '*') = (&va, &vb, *op) {
                return Value::Int(match op {
                    '+' => x + y,
                    '-' => x - y,
                    _ => x * y,
                });
            }
            let (x, y) = (va.as_f64(), vb.as_f64());
            Value::F64(match op {
                '+' => x + y,
                '-' => x - y,
                '*' => x * y,
                _ => x / y,
            })
        }
        Expr::Call(name, args) => {
            let a: Vec<f64> = args.iter().map(|x| eval(x, env).as_f64()).collect();
            Value::F64(match name.as_str() {
                "exp" => a[0].exp(),
                "ln" | "log" => a[0].ln(),
                "sqrt" => a[0].sqrt(),
                "abs" => a[0].abs(),
                "floor" => a[0].floor(),
                "sin" => a[0].sin(),
                "cos" => a[0].cos(),
                "tanh" => a[0].tanh(),
                "pow" => a[0].powf(a[1]),
                "min" => a[0].min(a[1]),
                "max" => a[0].max(a[1]),
                _ => f64::NAN,
            })
        }
    }
}

fn eval_addr(a: &AddrSpec, env: &Env) -> Address {
    match &a.index {
        None => Address::new(make_name(&a.name)),
        Some(ix) => {
            let v = eval(ix, env);
            match v.as_index() {
                Some(i) => Address::new(make_indexed(&a.name, i)),
                None => Address::new(make_indexed(&a.name, v.as_f64())),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Distributions
// ---------------------------------------------------------------------------

/// (min_args, max_args) per distribution; `Categorical` is variadic.
fn dist_arity(name: &str) -> Option<(usize, usize)> {
    Some(match name {
        "Normal" | "Uniform" | "LogNormal" | "Beta" | "Gamma" | "InverseGamma" | "Cauchy"
        | "Laplace" | "Weibull" | "Binomial" | "DiscreteUniform" => (2, 2),
        "Exponential" | "Poisson" | "Bernoulli" | "ChiSquared" => (1, 1),
        "StudentT" => (3, 3),
        "Categorical" => (1, 64),
        _ => return None,
    })
}

enum BuiltDist {
    F64(Box<dyn Distribution<f64>>),
    Bool(Box<dyn Distribution<bool>>),
    U64(Box<dyn Distribution<u64>>),
    Usize(Box<dyn Distribution<usize>>),
    I64(Box<dyn Distribution<i64>>),
}

fn build_dist(spec: &DistSpec, env: &Env) -> Result<BuiltDist, String> {
    let a: Vec<f64> = spec.args.iter().map(|x| eval(x, env).as_f64()).collect();
    let d = |r: Result<BuiltDist, FugueError>| r.map_err(|e| e.to_string());
    match spec.name.as_str() {
        "Normal" => d(Normal::new(a[0], a[1]).map(|x| BuiltDist::F64(Box::new(x)))),
        "Uniform" => d(Uniform::new(a[0], a[1]).map(|x| BuiltDist::F64(Box::new(x)))),
        "LogNormal" => d(LogNormal::new(a[0], a[1]).map(|x| BuiltDist::F64(Box::new(x)))),
        "Beta" => d(Beta::new(a[0], a[1]).map(|x| BuiltDist::F64(Box::new(x)))),
        "Gamma" => d(Gamma::new(a[0], a[1]).map(|x| BuiltDist::F64(Box::new(x)))),
        "InverseGamma" => d(InverseGamma::new(a[0], a[1]).map(|x| BuiltDist::F64(Box::new(x)))),
        "Cauchy" => d(Cauchy::new(a[0], a[1]).map(|x| BuiltDist::F64(Box::new(x)))),
        "Laplace" => d(Laplace::new(a[0], a[1]).map(|x| BuiltDist::F64(Box::new(x)))),
        "Weibull" => d(Weibull::new(a[0], a[1]).map(|x| BuiltDist::F64(Box::new(x)))),
        "Exponential" => d(Exponential::new(a[0]).map(|x| BuiltDist::F64(Box::new(x)))),
        "ChiSquared" => d(ChiSquared::new(a[0]).map(|x| BuiltDist::F64(Box::new(x)))),
        "StudentT" => d(StudentT::new(a[0], a[1], a[2]).map(|x| BuiltDist::F64(Box::new(x)))),
        "Bernoulli" => d(Bernoulli::new(a[0]).map(|x| BuiltDist::Bool(Box::new(x)))),
        "Binomial" => {
            if a[0] < 0.0 || a[0].fract() != 0.0 {
                return Err(format!("Binomial n must be a non-negative integer, got {}", a[0]));
            }
            d(Binomial::new(a[0] as u64, a[1]).map(|x| BuiltDist::U64(Box::new(x))))
        }
        "Poisson" => d(Poisson::new(a[0]).map(|x| BuiltDist::U64(Box::new(x)))),
        "Categorical" => d(Categorical::new(a.clone()).map(|x| BuiltDist::Usize(Box::new(x)))),
        "DiscreteUniform" => d(DiscreteUniform::new(a[0] as i64, a[1] as i64)
            .map(|x| BuiltDist::I64(Box::new(x)))),
        other => Err(format!("unknown distribution `{other}`")),
    }
}

// ---------------------------------------------------------------------------
// Interpretation: AST -> Model
// ---------------------------------------------------------------------------

/// Explicit continuation so `for` bodies and trailing statements compose
/// without recursion at build time (the `run` trampoline drives everything).
enum Cont {
    Stmts(Arc<Vec<Stmt>>, usize, Box<Cont>),
    Loop {
        var: String,
        i: i64,
        hi: i64,
        body: Arc<Vec<Stmt>>,
        after: Box<Cont>,
    },
    Ret(Arc<Expr>),
}

fn interp(env: Env, cont: Cont) -> Model<f64> {
    match cont {
        Cont::Ret(ret) => pure(eval(&ret, &env).as_f64()),
        Cont::Loop {
            var,
            i,
            hi,
            body,
            after,
        } => {
            if i >= hi {
                interp(env, *after)
            } else {
                let mut env2 = env;
                env2.vars.insert(var.clone(), Value::Int(i));
                interp(
                    env2,
                    Cont::Stmts(
                        body.clone(),
                        0,
                        Box::new(Cont::Loop {
                            var,
                            i: i + 1,
                            hi,
                            body,
                            after,
                        }),
                    ),
                )
            }
        }
        Cont::Stmts(stmts, idx, after) => {
            if idx >= stmts.len() {
                return interp(env, *after);
            }
            let next = |after: Box<Cont>| Cont::Stmts(stmts.clone(), idx + 1, after);
            match stmts[idx].clone() {
                Stmt::LetExpr { name, expr } => {
                    let mut env2 = env;
                    let v = eval(&expr, &env2);
                    env2.vars.insert(name, v);
                    interp(env2, next(after))
                }
                Stmt::Factor(e) => {
                    let lw = eval(&e, &env).as_f64();
                    let k = next(after);
                    factor(if lw.is_nan() { f64::NEG_INFINITY } else { lw })
                        .bind(move |_| interp(env, k))
                }
                Stmt::SampleLet { name, addr, dist } => {
                    let a = eval_addr(&addr, &env);
                    match build_dist(&dist, &env) {
                        Ok(built) => {
                            let k = next(after);
                            match built {
                                BuiltDist::F64(d) => sample(a, DynDist(d)).bind(move |v| {
                                    let mut env2 = env;
                                    env2.vars.insert(name, Value::F64(v));
                                    interp(env2, k)
                                }),
                                BuiltDist::Bool(d) => sample(a, DynDist(d)).bind(move |v| {
                                    let mut env2 = env;
                                    env2.vars.insert(name, Value::Bool(v));
                                    interp(env2, k)
                                }),
                                BuiltDist::U64(d) => sample(a, DynDist(d)).bind(move |v| {
                                    let mut env2 = env;
                                    env2.vars.insert(name, Value::Int(v as i64));
                                    interp(env2, k)
                                }),
                                BuiltDist::Usize(d) => sample(a, DynDist(d)).bind(move |v| {
                                    let mut env2 = env;
                                    env2.vars.insert(name, Value::Int(v as i64));
                                    interp(env2, k)
                                }),
                                BuiltDist::I64(d) => sample(a, DynDist(d)).bind(move |v| {
                                    let mut env2 = env;
                                    env2.vars.insert(name, Value::Int(v));
                                    interp(env2, k)
                                }),
                            }
                        }
                        Err(msg) => {
                            // Invalid parameters at this point of parameter
                            // space: impossible region. Sample a placeholder
                            // so downstream code still has a value, and kill
                            // the weight.
                            env.warn(format!("sample `{name}`: {msg}"));
                            let k = next(after);
                            let a2 = a.clone();
                            sample(a2, Normal::new(0.0, 1.0).unwrap())
                                .bind(move |v| {
                                    factor(f64::NEG_INFINITY).bind(move |_| {
                                        let mut env2 = env;
                                        env2.vars.insert(name, Value::F64(v));
                                        interp(env2, k)
                                    })
                                })
                        }
                    }
                }
                Stmt::Observe { addr, dist, value } => {
                    let a = eval_addr(&addr, &env);
                    let v = eval(&value, &env);
                    match build_dist(&dist, &env) {
                        Ok(built) => {
                            let k = next(after);
                            let m: Model<()> = match built {
                                BuiltDist::F64(d) => observe(a, DynDist(d), v.as_f64()),
                                BuiltDist::Bool(d) => observe(a, DynDist(d), v.as_bool()),
                                BuiltDist::U64(d) => {
                                    observe(a, DynDist(d), v.as_index().unwrap_or(0).max(0) as u64)
                                }
                                BuiltDist::Usize(d) => observe(
                                    a,
                                    DynDist(d),
                                    v.as_index().unwrap_or(0).max(0) as usize,
                                ),
                                BuiltDist::I64(d) => observe(a, DynDist(d), v.as_index().unwrap_or(0)),
                            };
                            m.bind(move |_| interp(env, k))
                        }
                        Err(msg) => {
                            env.warn(format!("observe at `{}`: {msg}", a));
                            let k = next(after);
                            factor(f64::NEG_INFINITY).bind(move |_| interp(env, k))
                        }
                    }
                }
                Stmt::For { var, lo, hi, body } => {
                    let lo_v = eval(&lo, &env).as_index().unwrap_or_else(|| {
                        env.warn("`for` lower bound is not an integer".to_string());
                        0
                    });
                    let hi_v = eval(&hi, &env).as_index().unwrap_or_else(|| {
                        env.warn("`for` upper bound is not an integer".to_string());
                        0
                    });
                    interp(
                        env,
                        Cont::Loop {
                            var,
                            i: lo_v,
                            hi: hi_v,
                            body,
                            after: Box::new(next(after)),
                        },
                    )
                }
            }
        }
    }
}

/// Adapter: fugue's `sample`/`observe` take `impl Distribution<T> + 'static`
/// by value, but the DSL builds `Box<dyn Distribution<T>>` at runtime.
struct DynDist<T>(Box<dyn Distribution<T>>);

impl<T: Clone + 'static> Distribution<T> for DynDist<T> {
    fn sample(&self, rng: &mut dyn rand::RngCore) -> T {
        self.0.sample(rng)
    }
    fn log_prob(&self, x: &T) -> f64 {
        self.0.log_prob(x)
    }
    fn clone_box(&self) -> Box<dyn Distribution<T>> {
        self.0.clone_box()
    }
}

// ---------------------------------------------------------------------------
// Public compiled-model handle
// ---------------------------------------------------------------------------

/// A parsed, validated model plus its data environment. `build()` constructs
/// a fresh single-use [`Model`], so `|| compiled.build()` is the `model_fn`
/// every fugue inference driver wants.
pub struct CompiledModel {
    prog: Arc<Program>,
    base_env: Env,
}

impl CompiledModel {
    /// Parse `source` and bind data arrays from `data_json` — a JSON object
    /// of number arrays (`{"y": [...], ...}`), a bare array (bound to
    /// `data`), or empty/`null` for no data. Booleans are coerced to 0/1.
    pub fn compile(source: &str, data_json: &str) -> Result<CompiledModel, String> {
        let prog = parse_program(source)?;
        let mut vars = HashMap::new();
        let trimmed = data_json.trim();
        if !trimmed.is_empty() && trimmed != "null" {
            let parsed: serde_json::Value =
                serde_json::from_str(trimmed).map_err(|e| format!("data is not valid JSON: {e}"))?;
            let to_arr = |v: &serde_json::Value| -> Result<Vec<f64>, String> {
                v.as_array()
                    .ok_or_else(|| "data arrays must be JSON arrays".to_string())?
                    .iter()
                    .map(|x| {
                        x.as_f64()
                            .or(x.as_bool().map(|b| if b { 1.0 } else { 0.0 }))
                            .ok_or_else(|| "data arrays must hold numbers or booleans".to_string())
                    })
                    .collect()
            };
            match &parsed {
                serde_json::Value::Array(_) => {
                    vars.insert("data".to_string(), Value::Arr(Arc::new(to_arr(&parsed)?)));
                }
                serde_json::Value::Object(map) => {
                    for (k, v) in map {
                        vars.insert(k.clone(), Value::Arr(Arc::new(to_arr(v)?)));
                    }
                }
                _ => return Err("data must be a JSON array or object of arrays".to_string()),
            }
        }
        let mut scope: Vec<String> = vars.keys().cloned().collect();
        check_stmts(&prog.stmts, &mut scope)?;
        check_expr(&prog.ret, &mut scope)?;
        Ok(CompiledModel {
            prog: Arc::new(prog),
            base_env: Env {
                vars,
                warnings: Arc::new(Mutex::new(Vec::new())),
            },
        })
    }

    /// Build a fresh single-use model.
    pub fn build(&self) -> Model<f64> {
        let env = self.base_env.clone();
        interp(
            env,
            Cont::Stmts(
                self.prog.stmts.clone(),
                0,
                Box::new(Cont::Ret(self.prog.ret.clone())),
            ),
        )
    }

    /// Drain accumulated runtime warnings (soft errors mapped to `-inf`).
    pub fn take_warnings(&self) -> Vec<String> {
        self.base_env
            .warnings
            .lock()
            .map(|mut w| std::mem::take(&mut *w))
            .unwrap_or_default()
    }

    /// Human-readable summary of the data environment (for error messages).
    pub fn data_summary(&self) -> String {
        let mut s = String::new();
        for (k, v) in &self.base_env.vars {
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
        assert!(CompiledModel::compile("let x <- sample(addr!(\"x\"), Nope(1.0)); pure(x)", "")
            .is_err());
        assert!(CompiledModel::compile(
            "let x <- sample(addr!(\"x\"), Normal(1.0)); pure(x)",
            ""
        )
        .is_err());
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
}
