//! Static checking: resolve every name — variables to environment slots,
//! distributions and functions to registry entries — so the interpreter
//! never looks a name up, and every unknown name is a compile-time error.
//!
//! Variables are lexically scoped as in Rust. Each binding site (`let`,
//! `let .. <- sample`, a loop variable, a data entry) gets its own slot, so a
//! `let` inside a block shadows an outer variable only until the block ends,
//! while an assignment resolves to the slot of the binding it names and so
//! persists across loop iterations and past the loop.

use std::sync::Arc;

use super::ast::{Addr, BinOp, DistCall, Expr, Stmt};
use super::registry::{DistEntry, FuncEntry, Registry};
use super::{ProgramError, Value};

/// A resolved expression.
pub(crate) enum RExpr {
    Lit(Value),
    Var(usize),
    Array(Vec<RExpr>),
    Index(Box<RExpr>, Box<RExpr>),
    Len(Box<RExpr>),
    Neg(Box<RExpr>),
    Not(Box<RExpr>),
    Bin(BinOp, Box<RExpr>, Box<RExpr>),
    Call(FuncEntry, Vec<RExpr>),
}

/// A resolved address: the name and the optional index.
pub(crate) struct RAddr {
    pub(crate) name: String,
    pub(crate) index: Option<RExpr>,
}

/// A resolved distribution call.
pub(crate) struct RDist {
    pub(crate) name: String,
    pub(crate) entry: DistEntry,
    pub(crate) args: Vec<RExpr>,
}

/// A resolved statement. `Let` and `Assign` both become `Set`: they differ
/// only in which slot the resolver picked.
pub(crate) enum RStmt {
    Sample {
        slot: usize,
        var: String,
        addr: RAddr,
        dist: RDist,
    },
    Set {
        slot: usize,
        var: String,
        value: RExpr,
    },
    Observe {
        addr: RAddr,
        dist: RDist,
        value: RExpr,
    },
    Factor(RExpr),
    For {
        slot: usize,
        var: String,
        start: RExpr,
        end: RExpr,
        body: Arc<[RStmt]>,
    },
    If {
        cond: RExpr,
        then_branch: Arc<[RStmt]>,
        else_branch: Arc<[RStmt]>,
    },
    Break,
}

pub(crate) struct Resolver<'r> {
    registry: &'r Registry,
    /// Innermost scope last; within a scope, later bindings shadow earlier.
    scopes: Vec<Vec<(String, usize)>>,
    n_slots: usize,
    loops: usize,
}

fn check(msg: String) -> ProgramError {
    ProgramError::Check(msg)
}

impl<'r> Resolver<'r> {
    pub(crate) fn new(registry: &'r Registry) -> Self {
        Resolver {
            registry,
            scopes: vec![Vec::new()],
            n_slots: 0,
            loops: 0,
        }
    }

    pub(crate) fn slot_count(&self) -> usize {
        self.n_slots
    }

    pub(crate) fn push_scope(&mut self) {
        self.scopes.push(Vec::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    /// Bind `name` in the innermost scope to a fresh slot.
    pub(crate) fn bind(&mut self, name: &str) -> usize {
        let slot = self.n_slots;
        self.n_slots += 1;
        self.scopes
            .last_mut()
            .expect("there is always a scope")
            .push((name.to_string(), slot));
        slot
    }

    fn lookup(&self, name: &str) -> Option<usize> {
        self.scopes
            .iter()
            .rev()
            .flat_map(|scope| scope.iter().rev())
            .find(|(n, _)| n == name)
            .map(|&(_, slot)| slot)
    }

    /// Resolve a block in a fresh scope; `loop_var` is bound first (a `for`
    /// body's variable lives in the body's scope).
    fn block(
        &mut self,
        stmts: &[Stmt],
        loop_var: Option<&str>,
    ) -> Result<(Arc<[RStmt]>, Option<usize>), ProgramError> {
        self.push_scope();
        let slot = loop_var.map(|v| self.bind(v));
        let out = self.stmts(stmts);
        self.pop_scope();
        Ok((out?.into(), slot))
    }

    pub(crate) fn stmts(&mut self, stmts: &[Stmt]) -> Result<Vec<RStmt>, ProgramError> {
        stmts.iter().map(|s| self.stmt(s)).collect()
    }

    fn stmt(&mut self, s: &Stmt) -> Result<RStmt, ProgramError> {
        Ok(match s {
            Stmt::Sample { var, addr, dist } => {
                // Arguments first: `let p <- sample(.., Normal(p, 1.0))`
                // reads the outer `p`, as in Rust.
                let addr = self.addr(addr)?;
                let dist = self.dist(dist)?;
                RStmt::Sample {
                    slot: self.bind(var),
                    var: var.clone(),
                    addr,
                    dist,
                }
            }
            Stmt::Let { var, value } => {
                let value = self.expr(value)?;
                RStmt::Set {
                    slot: self.bind(var),
                    var: var.clone(),
                    value,
                }
            }
            Stmt::Assign { var, value } => {
                let value = self.expr(value)?;
                let slot = self.lookup(var).ok_or_else(|| {
                    check(format!(
                        "cannot assign to unknown variable `{var}` (bind it with `let` first)"
                    ))
                })?;
                RStmt::Set {
                    slot,
                    var: var.clone(),
                    value,
                }
            }
            Stmt::Observe { addr, dist, value } => RStmt::Observe {
                addr: self.addr(addr)?,
                dist: self.dist(dist)?,
                value: self.expr(value)?,
            },
            Stmt::Factor { logw } => RStmt::Factor(self.expr(logw)?),
            Stmt::For {
                var,
                start,
                end,
                body,
            } => {
                let start = self.expr(start)?;
                let end = self.expr(end)?;
                self.loops += 1;
                let resolved = self.block(body, Some(var));
                self.loops -= 1;
                let (body, slot) = resolved?;
                RStmt::For {
                    slot: slot.expect("a loop variable was bound"),
                    var: var.clone(),
                    start,
                    end,
                    body,
                }
            }
            Stmt::If {
                cond,
                then_branch,
                else_branch,
            } => RStmt::If {
                cond: self.expr(cond)?,
                then_branch: self.block(then_branch, None)?.0,
                else_branch: self.block(else_branch, None)?.0,
            },
            Stmt::Break {} => {
                if self.loops == 0 {
                    return Err(check("`break` outside of a `for` loop".to_string()));
                }
                RStmt::Break
            }
        })
    }

    fn addr(&mut self, a: &Addr) -> Result<RAddr, ProgramError> {
        Ok(RAddr {
            name: a.name.clone(),
            index: a.index.as_ref().map(|ix| self.expr(ix)).transpose()?,
        })
    }

    fn dist(&mut self, d: &DistCall) -> Result<RDist, ProgramError> {
        let entry = self
            .registry
            .dist(&d.name)
            .ok_or_else(|| check(format!("unknown distribution `{}`", d.name)))?
            .clone();
        if !entry.arity.accepts(d.args.len()) {
            return Err(check(format!(
                "`{}` takes {} argument(s), got {}",
                d.name,
                entry.arity,
                d.args.len()
            )));
        }
        Ok(RDist {
            name: d.name.clone(),
            entry,
            args: d
                .args
                .iter()
                .map(|a| self.expr(a))
                .collect::<Result<_, _>>()?,
        })
    }

    pub(crate) fn expr(&mut self, e: &Expr) -> Result<RExpr, ProgramError> {
        let boxed = |r: &mut Self, e: &Expr| r.expr(e).map(Box::new);
        Ok(match e {
            Expr::F64 { value } => RExpr::Lit(Value::F64(*value)),
            Expr::Int { value } => RExpr::Lit(Value::Int(*value)),
            Expr::Bool { value } => RExpr::Lit(Value::Bool(*value)),
            Expr::Var { name } => RExpr::Var(
                self.lookup(name)
                    .ok_or_else(|| check(format!("unknown variable `{name}`")))?,
            ),
            Expr::Array { items } => RExpr::Array(
                items
                    .iter()
                    .map(|i| self.expr(i))
                    .collect::<Result<_, _>>()?,
            ),
            Expr::Index { array, index } => RExpr::Index(boxed(self, array)?, boxed(self, index)?),
            Expr::Len { array } => RExpr::Len(boxed(self, array)?),
            Expr::Neg { arg } => RExpr::Neg(boxed(self, arg)?),
            Expr::Not { arg } => RExpr::Not(boxed(self, arg)?),
            Expr::Call { func, args } => {
                let entry = self
                    .registry
                    .func(func)
                    .ok_or_else(|| check(format!("unknown function `{func}`")))?
                    .clone();
                if !entry.arity.accepts(args.len()) {
                    return Err(check(format!(
                        "`{func}` takes {} argument(s), got {}",
                        entry.arity,
                        args.len()
                    )));
                }
                RExpr::Call(
                    entry,
                    args.iter()
                        .map(|a| self.expr(a))
                        .collect::<Result<_, _>>()?,
                )
            }
            other => {
                let (op, lhs, rhs) = other.as_binary().expect("remaining variants are binary");
                RExpr::Bin(op, boxed(self, lhs)?, boxed(self, rhs)?)
            }
        })
    }
}
