//! Interpretation of a resolved program into fugue's own combinators.
//!
//! The executor walks the statements with an explicit frame stack (blocks and
//! loops) and a slot environment. Everything without an effect — `let`,
//! assignment, `if`, loop bookkeeping, `break` — runs in a plain loop, so a
//! long effect-free stretch costs no stack; each effect (`sample`, `observe`,
//! `factor`) returns a real `Model` node whose continuation resumes the loop,
//! so `run`'s trampoline drives the rest in constant stack as well.
//!
//! Total-evaluation policy: model-building code runs inside `bind`
//! continuations, where an error cannot propagate as a `Result` (and wasm has
//! no unwinding), so a runtime soft error — an index out of bounds, invalid
//! distribution parameters, a type error, a host constructor's `Err` — adds
//! `factor(-inf)` ("this region of parameter space is impossible"), records a
//! warning, and lets the execution continue with a placeholder value.

use std::cmp::Ordering;
use std::sync::{Arc, Mutex};

use rand::RngCore;

use crate::core::address::{make_indexed, make_name, Address};
use crate::core::distribution::{Distribution, Normal};
use crate::core::model::{factor, pure, Model, ModelExt, SampleType};

use super::ast::BinOp;
use super::compile::{RAddr, RDist, RExpr, RStmt};
use super::registry::{HostDist, SiteType};
use super::value::Num;
use super::Value;

/// Warnings kept per compiled program (the oldest are kept; later ones are
/// dropped until the buffer is drained).
const MAX_WARNINGS: usize = 64;

/// The immutable, shared part of a compiled program.
pub(crate) struct Shared {
    pub(crate) body: Arc<[RStmt]>,
    pub(crate) ret: RExpr,
    pub(crate) warnings: Mutex<Vec<String>>,
}

impl Shared {
    fn warn(&self, msg: String) {
        if let Ok(mut w) = self.warnings.lock() {
            if w.len() < MAX_WARNINGS {
                w.push(msg);
            }
        }
    }
}

/// One execution's mutable state, moved through the continuation chain.
struct Exec {
    slots: Vec<Value>,
    frames: Vec<Frame>,
}

enum Frame {
    Block {
        stmts: Arc<[RStmt]>,
        next: usize,
    },
    Loop {
        slot: usize,
        next: i64,
        end: i64,
        body: Arc<[RStmt]>,
    },
}

/// Build a fresh single-use model of the program.
pub(crate) fn start(shared: Arc<Shared>, base: &[Value]) -> Model<Value> {
    let ex = Exec {
        slots: base.to_vec(),
        frames: vec![Frame::Block {
            stmts: shared.body.clone(),
            next: 0,
        }],
    };
    drive(shared, ex)
}

/// Kill the execution's weight, then carry on.
fn reject(sh: Arc<Shared>, ex: Exec) -> Model<Value> {
    factor(f64::NEG_INFINITY).bind(move |()| drive(sh, ex))
}

fn drive(sh: Arc<Shared>, mut ex: Exec) -> Model<Value> {
    loop {
        let (stmts, idx) = match ex.frames.last_mut() {
            None => return finish(sh, ex),
            Some(Frame::Loop {
                slot,
                next,
                end,
                body,
            }) => {
                if *next >= *end {
                    ex.frames.pop();
                    continue;
                }
                let i = *next;
                *next += 1; // cannot overflow: next < end <= i64::MAX
                let (slot, body) = (*slot, body.clone());
                ex.slots[slot] = Value::Int(i);
                ex.frames.push(Frame::Block {
                    stmts: body,
                    next: 0,
                });
                continue;
            }
            Some(Frame::Block { stmts, next }) => {
                if *next >= stmts.len() {
                    ex.frames.pop();
                    continue;
                }
                *next += 1;
                (stmts.clone(), *next - 1)
            }
        };
        match &stmts[idx] {
            RStmt::Set { slot, var, value } => match eval(value, &ex.slots) {
                Ok(v) => ex.slots[*slot] = v,
                Err(e) => {
                    sh.warn(format!("`{var}`: {e}"));
                    ex.slots[*slot] = Value::F64(f64::NAN);
                    return reject(sh, ex);
                }
            },
            RStmt::Break => {
                while let Some(frame) = ex.frames.pop() {
                    if matches!(frame, Frame::Loop { .. }) {
                        break;
                    }
                }
            }
            RStmt::If {
                cond,
                then_branch,
                else_branch,
            } => match eval(cond, &ex.slots).and_then(|c| truth(&c)) {
                Ok(c) => {
                    let branch = if c { then_branch } else { else_branch };
                    if !branch.is_empty() {
                        ex.frames.push(Frame::Block {
                            stmts: branch.clone(),
                            next: 0,
                        });
                    }
                }
                Err(e) => {
                    sh.warn(format!("`if` condition: {e}"));
                    return reject(sh, ex);
                }
            },
            RStmt::For {
                slot,
                var,
                start,
                end,
                body,
            } => match range_bound(start, &ex.slots)
                .and_then(|s| range_bound(end, &ex.slots).map(|e| (s, e)))
            {
                Ok((next, end)) => ex.frames.push(Frame::Loop {
                    slot: *slot,
                    next,
                    end,
                    body: body.clone(),
                }),
                Err(e) => {
                    sh.warn(format!("`for {var}`: {e}"));
                    return reject(sh, ex);
                }
            },
            RStmt::Factor(logw) => {
                return match eval(logw, &ex.slots).and_then(|v| {
                    v.as_f64()
                        .ok_or_else(|| format!("expected a number, found {}", v.type_name()))
                }) {
                    // `factor` itself maps a NaN weight to -inf (FG-N2).
                    Ok(lw) => factor(lw).bind(move |()| drive(sh, ex)),
                    Err(e) => {
                        sh.warn(format!("`factor`: {e}"));
                        reject(sh, ex)
                    }
                };
            }
            RStmt::Sample {
                slot,
                var,
                addr,
                dist,
            } => return sample_stmt(sh, ex, *slot, var, addr, dist),
            RStmt::Observe { addr, dist, value } => return observe_stmt(sh, ex, addr, dist, value),
        }
    }
}

fn finish(sh: Arc<Shared>, ex: Exec) -> Model<Value> {
    match eval(&sh.ret, &ex.slots) {
        Ok(v) => pure(v),
        Err(e) => {
            sh.warn(format!("`pure`: {e}"));
            factor(f64::NEG_INFINITY).bind(|()| pure(Value::F64(f64::NAN)))
        }
    }
}

fn sample_stmt(
    sh: Arc<Shared>,
    mut ex: Exec,
    slot: usize,
    var: &str,
    addr: &RAddr,
    dist: &RDist,
) -> Model<Value> {
    let site = dist.entry.site;
    let address = match eval_addr(addr, &ex.slots) {
        Ok(a) => a,
        Err(e) => {
            // No address, so no site to record: bind a placeholder and reject.
            sh.warn(format!("sample `{var}`: {e}"));
            ex.slots[slot] = placeholder_value(site);
            return reject(sh, ex);
        }
    };
    match build_dist(dist, &ex.slots) {
        Ok(d) => sample_site(sh, ex, slot, address, d, false),
        Err(e) => {
            // Invalid parameters here: an impossible region. Sample a
            // placeholder of the site's declared type — so the site keeps its
            // address and type, and downstream code still has a value — and
            // kill the weight.
            sh.warn(format!("sample `{var}`: {e}"));
            sample_site(sh, ex, slot, address, placeholder_dist(site), true)
        }
    }
}

fn sample_site(
    sh: Arc<Shared>,
    ex: Exec,
    slot: usize,
    address: Address,
    d: HostDist,
    rejected: bool,
) -> Model<Value> {
    fn resume(
        sh: Arc<Shared>,
        mut ex: Exec,
        slot: usize,
        v: Value,
        rejected: bool,
    ) -> Model<Value> {
        ex.slots[slot] = v;
        if rejected {
            reject(sh, ex)
        } else {
            drive(sh, ex)
        }
    }
    match d {
        HostDist::F64(d) => f64::make_sample_model(address, d)
            .bind(move |x| resume(sh, ex, slot, Value::F64(x), rejected)),
        HostDist::Bool(d) => bool::make_sample_model(address, d)
            .bind(move |x| resume(sh, ex, slot, Value::Bool(x), rejected)),
        HostDist::U64(d) => u64::make_sample_model(address, d)
            .bind(move |x| resume(sh, ex, slot, Value::U64(x), rejected)),
        HostDist::Usize(d) => usize::make_sample_model(address, d)
            .bind(move |x| resume(sh, ex, slot, Value::Usize(x), rejected)),
        HostDist::I64(d) => i64::make_sample_model(address, d)
            .bind(move |x| resume(sh, ex, slot, Value::Int(x), rejected)),
    }
}

fn observe_stmt(
    sh: Arc<Shared>,
    ex: Exec,
    addr: &RAddr,
    dist: &RDist,
    value: &RExpr,
) -> Model<Value> {
    let address = match eval_addr(addr, &ex.slots) {
        Ok(a) => a,
        Err(e) => {
            sh.warn(format!("observe `{}`: {e}", addr.name));
            return reject(sh, ex);
        }
    };
    let built = eval(value, &ex.slots).and_then(|v| build_dist(dist, &ex.slots).map(|d| (v, d)));
    let (v, d) = match built {
        Ok(x) => x,
        Err(e) => {
            sh.warn(format!("observe at `{address}`: {e}"));
            return reject(sh, ex);
        }
    };
    let shown = address.clone();
    let site = d.site_type();
    // Convert the observed value to the site's type with the language's
    // coercions; a value outside the type (a negative count, 2.5 as an
    // index) has probability zero, and is also almost surely a data bug.
    let m: Option<Model<()>> = match d {
        HostDist::F64(d) => v.as_f64().map(|x| f64::make_observe_model(address, d, x)),
        HostDist::Bool(d) => v.as_bool().map(|x| bool::make_observe_model(address, d, x)),
        HostDist::U64(d) => v.as_u64().map(|x| u64::make_observe_model(address, d, x)),
        HostDist::Usize(d) => v
            .as_usize()
            .map(|x| usize::make_observe_model(address, d, x)),
        HostDist::I64(d) => v.as_i64().map(|x| i64::make_observe_model(address, d, x)),
    };
    match m {
        Some(m) => m.bind(move |()| drive(sh, ex)),
        None => {
            sh.warn(format!(
                "observe at `{shown}`: observed value {v} is not a valid {site}"
            ));
            reject(sh, ex)
        }
    }
}

fn build_dist(d: &RDist, slots: &[Value]) -> Result<HostDist, String> {
    let args = d
        .args
        .iter()
        .map(|a| eval(a, slots))
        .collect::<Result<Vec<_>, _>>()?;
    let built = (d.entry.ctor)(&args)?;
    if built.site_type() != d.entry.site {
        return Err(format!(
            "`{}` is registered with {} sites but built a {} distribution",
            d.name,
            d.entry.site,
            built.site_type()
        ));
    }
    Ok(built)
}

/// The address `addr!(name)` / `addr!(name, index)` would build in Rust: the
/// same `make_name`/`make_indexed` encoding, with the index formatted by the
/// `Display` of its natural type.
fn eval_addr(a: &RAddr, slots: &[Value]) -> Result<Address, String> {
    Ok(match &a.index {
        None => Address::new(make_name(&a.name)),
        Some(ix) => match eval(ix, slots)? {
            Value::Arr(_) => {
                return Err(format!(
                    "the index of `addr!(\"{}\", ..)` must be a scalar, found an array",
                    a.name
                ))
            }
            v => Address::new(make_indexed(&a.name, &v)),
        },
    })
}

fn range_bound(e: &RExpr, slots: &[Value]) -> Result<i64, String> {
    let v = eval(e, slots)?;
    v.as_i64()
        .ok_or_else(|| format!("range bound must be an integer, found {v}"))
}

fn truth(v: &Value) -> Result<bool, String> {
    v.as_bool().ok_or_else(|| match v {
        Value::Arr(_) => "expected a bool, found an array".to_string(),
        _ => "NaN is neither true nor false".to_string(),
    })
}

fn operand(op: BinOp, v: &Value) -> Result<Num, String> {
    v.num()
        .map_err(|_| format!("`{}` needs numbers, found {}", op.symbol(), v.type_name()))
}

/// A variable's value by reference: `data[i]` and `data.len()` then read the
/// array in place instead of cloning its `Arc` on every access.
fn borrow_var<'a>(e: &RExpr, slots: &'a [Value]) -> Option<&'a Value> {
    match e {
        RExpr::Var(slot) => Some(&slots[*slot]),
        _ => None,
    }
}

pub(crate) fn eval(e: &RExpr, slots: &[Value]) -> Result<Value, String> {
    Ok(match e {
        RExpr::Lit(v) => v.clone(),
        RExpr::Var(slot) => slots[*slot].clone(),
        RExpr::Array(items) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                let v = eval(item, slots)?;
                out.push(v.as_f64().ok_or_else(|| {
                    format!("array elements must be numbers, found {}", v.type_name())
                })?);
            }
            Value::Arr(Arc::new(out))
        }
        RExpr::Index(array, index) => {
            let owned;
            let arr = match borrow_var(array, slots) {
                Some(v) => v,
                None => {
                    owned = eval(array, slots)?;
                    &owned
                }
            };
            let idx = eval(index, slots)?;
            let Value::Arr(items) = arr else {
                return Err(format!("cannot index a {}", arr.type_name()));
            };
            let i = idx
                .as_i64()
                .ok_or_else(|| format!("array index must be an integer, found {idx}"))?;
            match usize::try_from(i).ok().and_then(|k| items.get(k)) {
                Some(x) => Value::F64(*x),
                None => return Err(format!("index {i} out of bounds (len {})", items.len())),
            }
        }
        RExpr::Len(array) => {
            let owned;
            let arr = match borrow_var(array, slots) {
                Some(v) => v,
                None => {
                    owned = eval(array, slots)?;
                    &owned
                }
            };
            match arr {
                Value::Arr(items) => Value::Int(items.len() as i64),
                other => return Err(format!("`.len()` on a {}", other.type_name())),
            }
        }
        RExpr::Neg(arg) => match operand(BinOp::Sub, &eval(arg, slots)?)? {
            Num::I(i) => Value::Int(
                i.checked_neg()
                    .ok_or_else(|| "integer overflow in `-`".to_string())?,
            ),
            Num::F(x) => Value::F64(-x),
        },
        RExpr::Not(arg) => Value::Bool(!truth(&eval(arg, slots)?)?),
        RExpr::Bin(op, lhs, rhs) => match op {
            // Short-circuit, as in Rust: `i < n && y[i] > 0.0` never
            // evaluates `y[i]` when `i >= n`.
            BinOp::And => Value::Bool(truth(&eval(lhs, slots)?)? && truth(&eval(rhs, slots)?)?),
            BinOp::Or => Value::Bool(truth(&eval(lhs, slots)?)? || truth(&eval(rhs, slots)?)?),
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div => {
                let a = operand(*op, &eval(lhs, slots)?)?;
                let b = operand(*op, &eval(rhs, slots)?)?;
                arith(*op, a, b)?
            }
            _ => Value::Bool(compare(*op, &eval(lhs, slots)?, &eval(rhs, slots)?)?),
        },
        RExpr::Call(entry, args) => {
            let vals = args
                .iter()
                .map(|a| eval(a, slots))
                .collect::<Result<Vec<_>, _>>()?;
            (entry.f)(&vals)?
        }
    })
}

/// `+ - *` stay exact on integers (overflow is an error, not a wrap) and are
/// float otherwise; `/` is always float division.
fn arith(op: BinOp, a: Num, b: Num) -> Result<Value, String> {
    if let (BinOp::Add | BinOp::Sub | BinOp::Mul, Num::I(x), Num::I(y)) = (op, a, b) {
        let r = match op {
            BinOp::Add => x.checked_add(y),
            BinOp::Sub => x.checked_sub(y),
            _ => x.checked_mul(y),
        };
        return r
            .map(Value::Int)
            .ok_or_else(|| format!("integer overflow in `{}`", op.symbol()));
    }
    let (x, y) = (a.to_f64(), b.to_f64());
    Ok(Value::F64(match op {
        BinOp::Add => x + y,
        BinOp::Sub => x - y,
        BinOp::Mul => x * y,
        _ => x / y,
    }))
}

/// Bools compare with bools (`false < true`); numbers compare exactly when
/// both are integers and as floats otherwise (a bool against a number reads
/// as `0`/`1`); `NaN` is unordered, so only `!=` holds for it.
fn compare(op: BinOp, a: &Value, b: &Value) -> Result<bool, String> {
    let ord = match (a, b) {
        (Value::Bool(x), Value::Bool(y)) => Some(x.cmp(y)),
        _ => match (a.num(), b.num()) {
            (Ok(Num::I(x)), Ok(Num::I(y))) => Some(x.cmp(&y)),
            (Ok(x), Ok(y)) => x.to_f64().partial_cmp(&y.to_f64()),
            _ => {
                return Err(format!(
                    "cannot compare {} with {} using `{}`",
                    a.type_name(),
                    b.type_name(),
                    op.symbol()
                ))
            }
        },
    };
    Ok(match op {
        BinOp::Eq => ord == Some(Ordering::Equal),
        BinOp::Ne => ord != Some(Ordering::Equal),
        BinOp::Lt => ord == Some(Ordering::Less),
        BinOp::Le => matches!(ord, Some(Ordering::Less | Ordering::Equal)),
        BinOp::Gt => ord == Some(Ordering::Greater),
        _ => matches!(ord, Some(Ordering::Greater | Ordering::Equal)),
    })
}

/// The value a sample binds when its address cannot even be built.
fn placeholder_value(site: SiteType) -> Value {
    match site {
        SiteType::F64 => Value::F64(f64::NAN),
        SiteType::Bool => Value::Bool(false),
        SiteType::U64 => Value::U64(0),
        SiteType::Usize => Value::Usize(0),
        SiteType::I64 => Value::Int(0),
    }
}

/// The distribution a sample site draws from when its own could not be
/// built: a standard normal for `f64` sites (what the playground interpreter
/// always drew), a point mass at `false`/`0` for the discrete types (`0` is
/// in the support of `Poisson` and `Binomial`, so a replay that repairs the
/// parameters can usually score the placeholder instead of rejecting it).
fn placeholder_dist(site: SiteType) -> HostDist {
    match site {
        SiteType::F64 => HostDist::F64(Box::new(Normal::standard())),
        SiteType::Bool => HostDist::Bool(Box::new(PointMass(false))),
        SiteType::U64 => HostDist::U64(Box::new(PointMass(0u64))),
        SiteType::Usize => HostDist::Usize(Box::new(PointMass(0usize))),
        SiteType::I64 => HostDist::I64(Box::new(PointMass(0i64))),
    }
}

/// A distribution with all its mass on one value.
#[derive(Clone)]
struct PointMass<T>(T);

impl<T: Clone + PartialEq + Send + Sync + 'static> Distribution<T> for PointMass<T> {
    fn sample(&self, _rng: &mut dyn RngCore) -> T {
        self.0.clone()
    }

    fn log_prob(&self, x: &T) -> f64 {
        if *x == self.0 {
            0.0
        } else {
            f64::NEG_INFINITY
        }
    }

    fn clone_box(&self) -> Box<dyn Distribution<T>> {
        Box::new(self.clone())
    }
}
