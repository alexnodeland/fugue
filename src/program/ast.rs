//! The serializable AST, its JSON front end, structural validation, and the
//! canonical text printer.

use std::fmt;

use serde::{Deserialize, Serialize};

use super::ProgramError;

/// The program format version this build reads and writes (the
/// `fugue_program` field of every serialized [`Program`]).
///
/// It changes only when a program that one version accepts would mean
/// something else, or fail, under another; new node kinds that an old reader
/// would reject outright also bump it, so an old reader fails with
/// [`ProgramError::UnsupportedVersion`] instead of an obscure decode error.
pub const FORMAT_VERSION: u32 = 1;

/// How deeply a program may nest, counted as the nesting depth of its JSON
/// form (every expression node is one level, a block is two).
///
/// serde_json refuses documents nested deeper than 127 levels; capping
/// programs below that, identically in *both* front ends (and in
/// [`Program::compile`](Program::compile)), guarantees that every program one
/// front end accepts the other accepts too, and bounds the parser's and the
/// interpreter's recursion on hostile input.
pub const MAX_NESTING: usize = 100;

/// A program: statements and a return expression, plus the format version.
///
/// A `Program` is plain data. Build one with [`Program::parse`] (the text
/// syntax), [`Program::from_json`], or directly from the AST types; then
/// [`compile`](Program::compile) it against a [`Registry`](super::Registry)
/// and [`Data`](super::Data) to get a [`CompiledProgram`](super::CompiledProgram)
/// that builds real `Model`s.
///
/// `Display` prints the canonical text form, which [`Program::parse`] reads
/// back to an equal program.
///
/// ```rust
/// use fugue::program::Program;
///
/// let text = Program::parse(
///     r#"let p <- sample(addr!("p"), Beta(2.0, 2.0));
///        observe(addr!("y"), Bernoulli(p), true);
///        pure(p)"#,
/// )
/// .unwrap();
/// let json = Program::from_json(&text.to_json().unwrap()).unwrap();
/// assert_eq!(text, json);
/// assert_eq!(Program::parse(&text.to_string()).unwrap(), text);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Program {
    /// Format version; [`FORMAT_VERSION`] for programs this build writes.
    pub fugue_program: u32,
    /// The statements, in order.
    pub body: Vec<Stmt>,
    /// The expression `pure(..)` returns.
    pub ret: Expr,
}

/// A statement. The JSON form is an object tagged by `"stmt"`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "stmt", rename_all = "snake_case", deny_unknown_fields)]
pub enum Stmt {
    /// `let var <- sample(addr, dist);` — draw from `dist` at `addr` and bind
    /// the draw (with the site's natural type) to `var`.
    Sample {
        /// The variable the draw is bound to.
        var: String,
        /// The site's address.
        addr: Addr,
        /// The distribution to draw from.
        dist: DistCall,
    },
    /// `let var = value;` — bind a new variable (shadowing any outer one
    /// until the end of the enclosing block).
    Let {
        /// The new variable.
        var: String,
        /// Its value.
        value: Expr,
    },
    /// `var = value;` — reassign a variable bound by an enclosing `let`
    /// (or by the data). The new value is visible for the rest of that
    /// variable's scope, including later iterations of an enclosing loop.
    Assign {
        /// The variable to reassign.
        var: String,
        /// Its new value.
        value: Expr,
    },
    /// `observe(addr, dist, value);` — condition on `value` under `dist`.
    Observe {
        /// The observation's address.
        addr: Addr,
        /// The distribution the value is scored under.
        dist: DistCall,
        /// The observed value.
        value: Expr,
    },
    /// `factor(logw);` — add `logw` to the execution's log-weight.
    Factor {
        /// The log-weight.
        logw: Expr,
    },
    /// `for var in start..end { body }` — run `body` with `var` bound to each
    /// integer of the half-open range, evaluated once on entry.
    For {
        /// The loop variable (scoped to the body).
        var: String,
        /// Inclusive start.
        start: Expr,
        /// Exclusive end.
        end: Expr,
        /// The loop body.
        body: Vec<Stmt>,
    },
    /// `if cond { then } else { else }` (`else` optional; `else if` chains
    /// nest an `If` as the only statement of the else branch).
    If {
        /// The condition.
        cond: Expr,
        /// Statements run when the condition is true.
        #[serde(rename = "then")]
        then_branch: Vec<Stmt>,
        /// Statements run when it is false (omitted from JSON when empty).
        #[serde(rename = "else", default, skip_serializing_if = "Vec::is_empty")]
        else_branch: Vec<Stmt>,
    },
    /// `break;` — leave the innermost `for` loop.
    Break {},
}

/// A site address: `addr!("name")` or `addr!("name", index)`, built with the
/// same encoding as fugue's `addr!` macro.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Addr {
    /// The address name (any string; escaped exactly as `addr!` escapes it).
    pub name: String,
    /// The optional index expression (omitted from JSON when absent).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index: Option<Expr>,
}

/// A distribution constructor call: a registered distribution's name and its
/// argument expressions (`Normal(mu, 1.0)`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DistCall {
    /// The registered distribution name.
    pub name: String,
    /// The constructor's arguments.
    #[serde(default)]
    pub args: Vec<Expr>,
}

/// An expression. The JSON form is an object tagged by `"op"`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum Expr {
    /// A float literal (`2.0`, `1e-3`); must be finite.
    F64 {
        /// The literal's value.
        value: f64,
    },
    /// An integer literal (`2`).
    Int {
        /// The literal's value.
        value: i64,
    },
    /// `true` or `false`.
    Bool {
        /// The literal's value.
        value: bool,
    },
    /// A variable reference.
    Var {
        /// The variable's name.
        name: String,
    },
    /// An array literal `[a, b, c]` (its elements must evaluate to numbers).
    Array {
        /// The element expressions.
        items: Vec<Expr>,
    },
    /// `array[index]`.
    Index {
        /// The indexed array.
        array: Box<Expr>,
        /// The index.
        index: Box<Expr>,
    },
    /// `array.len()`.
    Len {
        /// The array.
        array: Box<Expr>,
    },
    /// `-arg`.
    Neg {
        /// The operand.
        arg: Box<Expr>,
    },
    /// `!arg`.
    Not {
        /// The operand.
        arg: Box<Expr>,
    },
    /// `lhs + rhs`.
    Add {
        /// Left operand.
        lhs: Box<Expr>,
        /// Right operand.
        rhs: Box<Expr>,
    },
    /// `lhs - rhs`.
    Sub {
        /// Left operand.
        lhs: Box<Expr>,
        /// Right operand.
        rhs: Box<Expr>,
    },
    /// `lhs * rhs`.
    Mul {
        /// Left operand.
        lhs: Box<Expr>,
        /// Right operand.
        rhs: Box<Expr>,
    },
    /// `lhs / rhs` (always float division).
    Div {
        /// Left operand.
        lhs: Box<Expr>,
        /// Right operand.
        rhs: Box<Expr>,
    },
    /// `lhs == rhs`.
    Eq {
        /// Left operand.
        lhs: Box<Expr>,
        /// Right operand.
        rhs: Box<Expr>,
    },
    /// `lhs != rhs`.
    Ne {
        /// Left operand.
        lhs: Box<Expr>,
        /// Right operand.
        rhs: Box<Expr>,
    },
    /// `lhs < rhs`.
    Lt {
        /// Left operand.
        lhs: Box<Expr>,
        /// Right operand.
        rhs: Box<Expr>,
    },
    /// `lhs <= rhs`.
    Le {
        /// Left operand.
        lhs: Box<Expr>,
        /// Right operand.
        rhs: Box<Expr>,
    },
    /// `lhs > rhs`.
    Gt {
        /// Left operand.
        lhs: Box<Expr>,
        /// Right operand.
        rhs: Box<Expr>,
    },
    /// `lhs >= rhs`.
    Ge {
        /// Left operand.
        lhs: Box<Expr>,
        /// Right operand.
        rhs: Box<Expr>,
    },
    /// `lhs && rhs` (short-circuit).
    And {
        /// Left operand.
        lhs: Box<Expr>,
        /// Right operand.
        rhs: Box<Expr>,
    },
    /// `lhs || rhs` (short-circuit).
    Or {
        /// Left operand.
        lhs: Box<Expr>,
        /// Right operand.
        rhs: Box<Expr>,
    },
    /// A call of a registered pure function: `exp(x)`, `max(a, b)`.
    Call {
        /// The registered function name.
        func: String,
        /// The arguments.
        #[serde(default)]
        args: Vec<Expr>,
    },
}

/// The binary operators, for building and inspecting [`Expr`] generically.
///
/// ```rust
/// use fugue::program::{BinOp, Expr};
///
/// let e = Expr::binary(BinOp::Mul, Expr::F64 { value: 2.0 }, Expr::var("p"));
/// assert_eq!(e.to_string(), "2.0 * p");
/// assert!(matches!(e.as_binary(), Some((BinOp::Mul, _, _))));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinOp {
    /// `+`
    Add,
    /// `-`
    Sub,
    /// `*`
    Mul,
    /// `/`
    Div,
    /// `==`
    Eq,
    /// `!=`
    Ne,
    /// `<`
    Lt,
    /// `<=`
    Le,
    /// `>`
    Gt,
    /// `>=`
    Ge,
    /// `&&`
    And,
    /// `||`
    Or,
}

impl BinOp {
    /// The operator's text spelling (`"+"`, `"<="`, `"&&"`, ...).
    pub fn symbol(self) -> &'static str {
        match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Eq => "==",
            BinOp::Ne => "!=",
            BinOp::Lt => "<",
            BinOp::Le => "<=",
            BinOp::Gt => ">",
            BinOp::Ge => ">=",
            BinOp::And => "&&",
            BinOp::Or => "||",
        }
    }

    /// Binding strength, Rust's: `||` < `&&` < comparisons < `+ -` < `* /`.
    pub(crate) fn precedence(self) -> u8 {
        match self {
            BinOp::Or => 1,
            BinOp::And => 2,
            BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => 3,
            BinOp::Add | BinOp::Sub => 4,
            BinOp::Mul | BinOp::Div => 5,
        }
    }
}

impl Expr {
    /// A variable reference.
    pub fn var(name: impl Into<String>) -> Expr {
        Expr::Var { name: name.into() }
    }

    /// The binary expression `lhs op rhs`.
    pub fn binary(op: BinOp, lhs: Expr, rhs: Expr) -> Expr {
        let (lhs, rhs) = (Box::new(lhs), Box::new(rhs));
        match op {
            BinOp::Add => Expr::Add { lhs, rhs },
            BinOp::Sub => Expr::Sub { lhs, rhs },
            BinOp::Mul => Expr::Mul { lhs, rhs },
            BinOp::Div => Expr::Div { lhs, rhs },
            BinOp::Eq => Expr::Eq { lhs, rhs },
            BinOp::Ne => Expr::Ne { lhs, rhs },
            BinOp::Lt => Expr::Lt { lhs, rhs },
            BinOp::Le => Expr::Le { lhs, rhs },
            BinOp::Gt => Expr::Gt { lhs, rhs },
            BinOp::Ge => Expr::Ge { lhs, rhs },
            BinOp::And => Expr::And { lhs, rhs },
            BinOp::Or => Expr::Or { lhs, rhs },
        }
    }

    /// `Some((op, lhs, rhs))` for a binary expression, `None` otherwise.
    pub fn as_binary(&self) -> Option<(BinOp, &Expr, &Expr)> {
        let (op, lhs, rhs) = match self {
            Expr::Add { lhs, rhs } => (BinOp::Add, lhs, rhs),
            Expr::Sub { lhs, rhs } => (BinOp::Sub, lhs, rhs),
            Expr::Mul { lhs, rhs } => (BinOp::Mul, lhs, rhs),
            Expr::Div { lhs, rhs } => (BinOp::Div, lhs, rhs),
            Expr::Eq { lhs, rhs } => (BinOp::Eq, lhs, rhs),
            Expr::Ne { lhs, rhs } => (BinOp::Ne, lhs, rhs),
            Expr::Lt { lhs, rhs } => (BinOp::Lt, lhs, rhs),
            Expr::Le { lhs, rhs } => (BinOp::Le, lhs, rhs),
            Expr::Gt { lhs, rhs } => (BinOp::Gt, lhs, rhs),
            Expr::Ge { lhs, rhs } => (BinOp::Ge, lhs, rhs),
            Expr::And { lhs, rhs } => (BinOp::And, lhs, rhs),
            Expr::Or { lhs, rhs } => (BinOp::Or, lhs, rhs),
            _ => return None,
        };
        Some((op, lhs, rhs))
    }
}

// ---------------------------------------------------------------------------
// Construction, JSON front end
// ---------------------------------------------------------------------------

impl Program {
    /// A program at the current [`FORMAT_VERSION`].
    pub fn new(body: Vec<Stmt>, ret: Expr) -> Program {
        Program {
            fugue_program: FORMAT_VERSION,
            body,
            ret,
        }
    }

    /// Read a program from its JSON form.
    ///
    /// The `fugue_program` version is checked **before** the rest of the
    /// document is decoded, so a program written by a newer format fails with
    /// [`ProgramError::UnsupportedVersion`] rather than with whatever node its
    /// newer syntax trips over first. Unknown fields are rejected, so a
    /// misspelt key is an error, not a silently different program.
    ///
    /// ```rust
    /// use fugue::program::{Program, ProgramError};
    ///
    /// let p = Program::from_json(
    ///     r#"{"fugue_program": 1, "body": [], "ret": {"op": "f64", "value": 1.5}}"#,
    /// )
    /// .unwrap();
    /// assert_eq!(p, Program::parse("pure(1.5)").unwrap());
    ///
    /// let newer = r#"{"fugue_program": 2, "body": [], "ret": {"op": "f64", "value": 1.5}}"#;
    /// assert!(matches!(
    ///     Program::from_json(newer),
    ///     Err(ProgramError::UnsupportedVersion { found: 2, supported: 1 })
    /// ));
    /// ```
    pub fn from_json(json: &str) -> Result<Program, ProgramError> {
        #[derive(Deserialize)]
        struct VersionProbe {
            fugue_program: Option<serde_json::Value>,
        }
        let probe: VersionProbe =
            serde_json::from_str(json).map_err(|e| ProgramError::Json(e.to_string()))?;
        let version = match probe.fugue_program {
            None => {
                return Err(ProgramError::Json(
                    "missing field `fugue_program` (the program format version)".to_string(),
                ))
            }
            Some(v) => v.as_u64().ok_or_else(|| {
                ProgramError::Json(format!(
                    "`fugue_program` must be a non-negative integer format version, found {v}"
                ))
            })?,
        };
        if version != u64::from(FORMAT_VERSION) {
            return Err(ProgramError::UnsupportedVersion {
                found: version,
                supported: FORMAT_VERSION,
            });
        }
        let program: Program =
            serde_json::from_str(json).map_err(|e| ProgramError::Json(e.to_string()))?;
        program.validate()?;
        Ok(program)
    }

    /// The program's JSON form (compact). Fails for a program that would not
    /// read back: a non-current version, a non-finite literal, a name that is
    /// not an identifier, or nesting beyond [`MAX_NESTING`].
    pub fn to_json(&self) -> Result<String, ProgramError> {
        self.validate()?;
        serde_json::to_string(self).map_err(|e| ProgramError::Json(e.to_string()))
    }

    /// The program's JSON form, pretty-printed. Same checks as
    /// [`to_json`](Program::to_json).
    pub fn to_json_pretty(&self) -> Result<String, ProgramError> {
        self.validate()?;
        serde_json::to_string_pretty(self).map_err(|e| ProgramError::Json(e.to_string()))
    }

    /// Structural checks shared by both front ends, `to_json` and `compile`:
    /// the version, finite literals, identifier names, and the nesting cap.
    /// Name *resolution* (variables, registry entries, `break` placement) is
    /// [`compile`](Program::compile)'s job.
    pub(crate) fn validate(&self) -> Result<(), ProgramError> {
        if self.fugue_program != FORMAT_VERSION {
            return Err(ProgramError::UnsupportedVersion {
                found: u64::from(self.fugue_program),
                supported: FORMAT_VERSION,
            });
        }
        let depth = self.nesting_depth();
        if depth > MAX_NESTING {
            return Err(ProgramError::Check(format!(
                "program nests {depth} levels deep; the format allows at most {MAX_NESTING}"
            )));
        }
        // Iterative walk (the depth is bounded now, but a walk that cannot
        // overflow is no dearer than one that can).
        let mut stmts: Vec<&Stmt> = self.body.iter().collect();
        let mut exprs: Vec<&Expr> = vec![&self.ret];
        let ident = |kind: &str, name: &str| -> Result<(), ProgramError> {
            if is_ident(name) {
                Ok(())
            } else {
                Err(ProgramError::Check(format!(
                    "{kind} name `{name}` is not an identifier (or is a reserved word)"
                )))
            }
        };
        while let Some(s) = stmts.pop() {
            match s {
                Stmt::Sample { var, addr, dist } => {
                    ident("variable", var)?;
                    ident("distribution", &dist.name)?;
                    exprs.extend(addr.index.iter());
                    exprs.extend(dist.args.iter());
                }
                Stmt::Let { var, value } | Stmt::Assign { var, value } => {
                    ident("variable", var)?;
                    exprs.push(value);
                }
                Stmt::Observe { addr, dist, value } => {
                    ident("distribution", &dist.name)?;
                    exprs.extend(addr.index.iter());
                    exprs.extend(dist.args.iter());
                    exprs.push(value);
                }
                Stmt::Factor { logw } => exprs.push(logw),
                Stmt::For {
                    var,
                    start,
                    end,
                    body,
                } => {
                    ident("variable", var)?;
                    exprs.push(start);
                    exprs.push(end);
                    stmts.extend(body.iter());
                }
                Stmt::If {
                    cond,
                    then_branch,
                    else_branch,
                } => {
                    exprs.push(cond);
                    stmts.extend(then_branch.iter());
                    stmts.extend(else_branch.iter());
                }
                Stmt::Break {} => {}
            }
        }
        while let Some(e) = exprs.pop() {
            match e {
                Expr::F64 { value } if !value.is_finite() => {
                    return Err(ProgramError::Check(format!(
                        "number literal {value} is not finite (JSON cannot represent it)"
                    )))
                }
                Expr::F64 { .. } | Expr::Int { .. } | Expr::Bool { .. } => {}
                Expr::Var { name } => ident("variable", name)?,
                Expr::Array { items } => exprs.extend(items.iter()),
                Expr::Index { array, index } => {
                    exprs.push(array);
                    exprs.push(index);
                }
                Expr::Len { array: arg } | Expr::Neg { arg } | Expr::Not { arg } => exprs.push(arg),
                Expr::Call { func, args } => {
                    ident("function", func)?;
                    exprs.extend(args.iter());
                }
                other => {
                    let (_, lhs, rhs) = other.as_binary().expect("remaining variants are binary");
                    exprs.push(lhs);
                    exprs.push(rhs);
                }
            }
        }
        Ok(())
    }

    /// The nesting depth of the program's JSON form, computed without
    /// recursion (so it is safe on any AST, however deep).
    pub(crate) fn nesting_depth(&self) -> usize {
        // Depths follow the serialized shape: the program object is level 1,
        // its `body` array level 2, each statement object level 3; a
        // statement's direct expressions sit one level below it, a block two
        // (array + statement object), a distribution's arguments three
        // (dist object + args array + expression).
        let mut max = 2; // program object + body array
        let mut stmts: Vec<(&Stmt, usize)> = self.body.iter().map(|s| (s, 3)).collect();
        let mut exprs: Vec<(&Expr, usize)> = vec![(&self.ret, 2)];
        while let Some((s, d)) = stmts.pop() {
            max = max.max(d);
            match s {
                Stmt::Sample { addr, dist, .. } | Stmt::Observe { addr, dist, .. } => {
                    // addr/dist objects at d + 1, the (always written) args
                    // array at d + 2, its expressions at d + 3.
                    max = max.max(d + 2);
                    if let Some(ix) = &addr.index {
                        exprs.push((ix, d + 2));
                    }
                    exprs.extend(dist.args.iter().map(|a| (a, d + 3)));
                    if let Stmt::Observe { value, .. } = s {
                        exprs.push((value, d + 1));
                    }
                }
                Stmt::Let { value, .. } | Stmt::Assign { value, .. } => exprs.push((value, d + 1)),
                Stmt::Factor { logw } => exprs.push((logw, d + 1)),
                Stmt::For {
                    start, end, body, ..
                } => {
                    exprs.push((start, d + 1));
                    exprs.push((end, d + 1));
                    max = max.max(d + 1); // body array (even when empty)
                    stmts.extend(body.iter().map(|s| (s, d + 2)));
                }
                Stmt::If {
                    cond,
                    then_branch,
                    else_branch,
                } => {
                    exprs.push((cond, d + 1));
                    max = max.max(d + 1); // then array (always written)
                    stmts.extend(then_branch.iter().map(|s| (s, d + 2)));
                    stmts.extend(else_branch.iter().map(|s| (s, d + 2)));
                }
                Stmt::Break {} => {}
            }
        }
        while let Some((e, d)) = exprs.pop() {
            max = max.max(d);
            match e {
                Expr::F64 { .. } | Expr::Int { .. } | Expr::Bool { .. } | Expr::Var { .. } => {}
                Expr::Array { items: args } | Expr::Call { args, .. } => {
                    max = max.max(d + 1); // the array itself
                    exprs.extend(args.iter().map(|a| (a, d + 2)));
                }
                Expr::Index { array, index } => {
                    exprs.push((array, d + 1));
                    exprs.push((index, d + 1));
                }
                Expr::Len { array: arg } | Expr::Neg { arg } | Expr::Not { arg } => {
                    exprs.push((arg, d + 1))
                }
                other => {
                    let (_, lhs, rhs) = other.as_binary().expect("remaining variants are binary");
                    exprs.push((lhs, d + 1));
                    exprs.push((rhs, d + 1));
                }
            }
        }
        max
    }
}

/// Words the text syntax reserves; they cannot name variables, distributions
/// or functions.
pub(crate) const RESERVED: &[&str] = &[
    "let", "mut", "observe", "factor", "for", "in", "if", "else", "break", "pure", "true", "false",
];

/// An ASCII identifier (`[A-Za-z_][A-Za-z0-9_]*`) that is not reserved.
pub(crate) fn is_ident(s: &str) -> bool {
    let mut chars = s.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !RESERVED.contains(&s)
}

// ---------------------------------------------------------------------------
// Canonical text printer
// ---------------------------------------------------------------------------

/// The canonical text form; [`Program::parse`] reads it back to an equal
/// program (for any program [`to_json`](Program::to_json) accepts).
impl fmt::Display for Program {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for s in &self.body {
            write_stmt(f, s, 0)?;
        }
        write!(f, "pure({})", self.ret)
    }
}

/// Prints the expression in text syntax with the minimal parentheses that
/// read back to the same tree.
impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_expr(f, self)
    }
}

fn indent(f: &mut fmt::Formatter<'_>, depth: usize) -> fmt::Result {
    for _ in 0..depth {
        f.write_str("    ")?;
    }
    Ok(())
}

fn write_block(f: &mut fmt::Formatter<'_>, stmts: &[Stmt], depth: usize) -> fmt::Result {
    f.write_str("{\n")?;
    for s in stmts {
        write_stmt(f, s, depth + 1)?;
    }
    indent(f, depth)?;
    f.write_str("}")
}

fn write_stmt(f: &mut fmt::Formatter<'_>, s: &Stmt, depth: usize) -> fmt::Result {
    indent(f, depth)?;
    match s {
        Stmt::Sample { var, addr, dist } => {
            writeln!(f, "let {var} <- sample({addr}, {dist});")
        }
        Stmt::Let { var, value } => writeln!(f, "let {var} = {value};"),
        Stmt::Assign { var, value } => writeln!(f, "{var} = {value};"),
        Stmt::Observe { addr, dist, value } => {
            writeln!(f, "observe({addr}, {dist}, {value});")
        }
        Stmt::Factor { logw } => writeln!(f, "factor({logw});"),
        Stmt::For {
            var,
            start,
            end,
            body,
        } => {
            write!(f, "for {var} in {start}..{end} ")?;
            write_block(f, body, depth)?;
            f.write_str("\n")
        }
        Stmt::If { .. } => {
            write_if(f, s, depth)?;
            f.write_str("\n")
        }
        Stmt::Break {} => f.write_str("break;\n"),
    }
}

fn write_if(f: &mut fmt::Formatter<'_>, s: &Stmt, depth: usize) -> fmt::Result {
    let Stmt::If {
        cond,
        then_branch,
        else_branch,
    } = s
    else {
        unreachable!("write_if is only called on `If`")
    };
    write!(f, "if {cond} ")?;
    write_block(f, then_branch, depth)?;
    match else_branch.as_slice() {
        [] => Ok(()),
        [nested @ Stmt::If { .. }] => {
            f.write_str(" else ")?;
            write_if(f, nested, depth)
        }
        stmts => {
            f.write_str(" else ")?;
            write_block(f, stmts, depth)
        }
    }
}

impl fmt::Display for Addr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("addr!(")?;
        write_str_literal(f, &self.name)?;
        if let Some(ix) = &self.index {
            write!(f, ", {ix}")?;
        }
        f.write_str(")")
    }
}

impl fmt::Display for DistCall {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}(", self.name)?;
        write_list(f, &self.args)?;
        f.write_str(")")
    }
}

fn write_list(f: &mut fmt::Formatter<'_>, items: &[Expr]) -> fmt::Result {
    for (i, a) in items.iter().enumerate() {
        if i > 0 {
            f.write_str(", ")?;
        }
        write_expr(f, a)?;
    }
    Ok(())
}

/// A Rust string literal the lexer reads back to exactly `s`.
fn write_str_literal(f: &mut fmt::Formatter<'_>, s: &str) -> fmt::Result {
    f.write_str("\"")?;
    for c in s.chars() {
        match c {
            '\\' => f.write_str("\\\\")?,
            '"' => f.write_str("\\\"")?,
            '\n' => f.write_str("\\n")?,
            '\r' => f.write_str("\\r")?,
            '\t' => f.write_str("\\t")?,
            '\0' => f.write_str("\\0")?,
            c => write!(f, "{c}")?,
        }
    }
    f.write_str("\"")
}

/// Precedence of an expression as printed: binary operators their own,
/// prefix operators and negative literals 6, postfix/primary forms 7.
fn expr_prec(e: &Expr) -> u8 {
    match e {
        Expr::F64 { value } if value.is_sign_negative() => 6,
        Expr::Int { value } if *value < 0 => 6,
        Expr::Neg { .. } | Expr::Not { .. } => 6,
        other => other.as_binary().map_or(7, |(op, _, _)| op.precedence()),
    }
}

fn write_expr(f: &mut fmt::Formatter<'_>, e: &Expr) -> fmt::Result {
    match e {
        // `{:?}` keeps a `.0` or an exponent, so the literal reads back as a
        // float; it prints the shortest digits that round-trip.
        Expr::F64 { value } => write!(f, "{value:?}"),
        Expr::Int { value } => write!(f, "{value}"),
        Expr::Bool { value } => write!(f, "{value}"),
        Expr::Var { name } => f.write_str(name),
        Expr::Array { items } => {
            f.write_str("[")?;
            write_list(f, items)?;
            f.write_str("]")
        }
        Expr::Index { array, index } => {
            write_operand(f, array, 7)?;
            write!(f, "[{index}]")
        }
        Expr::Len { array } => {
            write_operand(f, array, 7)?;
            f.write_str(".len()")
        }
        Expr::Neg { arg } => {
            // The parser folds `-2.0` into one negative literal, so a `Neg`
            // of a literal prints as `-(2.0)` to read back as a `Neg`.
            if matches!(**arg, Expr::F64 { .. } | Expr::Int { .. }) {
                write!(f, "-({arg})")
            } else {
                f.write_str("-")?;
                write_operand(f, arg, 6)
            }
        }
        Expr::Not { arg } => {
            f.write_str("!")?;
            write_operand(f, arg, 6)
        }
        Expr::Call { func, args } => {
            write!(f, "{func}(")?;
            write_list(f, args)?;
            f.write_str(")")
        }
        other => {
            let (op, lhs, rhs) = other.as_binary().expect("remaining variants are binary");
            let p = op.precedence();
            // Left-associative, and comparisons do not chain: the left operand
            // needs parentheses below this precedence (at it, for a
            // comparison), the right operand at or below it.
            let lhs_min = if p == 3 { p + 1 } else { p };
            write_operand(f, lhs, lhs_min)?;
            write!(f, " {} ", op.symbol())?;
            write_operand(f, rhs, p + 1)
        }
    }
}

/// Print `e`, parenthesized unless it binds at least as tightly as `min`.
fn write_operand(f: &mut fmt::Formatter<'_>, e: &Expr, min: u8) -> fmt::Result {
    if expr_prec(e) >= min {
        write_expr(f, e)
    } else {
        write!(f, "({e})")
    }
}
