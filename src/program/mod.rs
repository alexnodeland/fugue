#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/docs/program/README.md"))]

mod ast;
mod compile;
mod data;
mod interp;
mod parse;
mod registry;
#[cfg(test)]
mod tests;
mod value;

use std::fmt;
use std::str::FromStr;
use std::sync::{Arc, Mutex};

use crate::core::model::{Model, ModelExt};

pub use ast::{Addr, BinOp, DistCall, Expr, Program, Stmt, FORMAT_VERSION, MAX_NESTING};
pub use data::Data;
pub use registry::{Arity, HostDist, Registry, SiteType};
pub use value::Value;

/// Why a program could not be read, written or compiled.
///
/// Runtime problems are not errors: once a program compiles, building and
/// running its model never fails — see the module docs' error policy.
#[derive(Clone, Debug, PartialEq)]
pub enum ProgramError {
    /// The text front end rejected the source.
    Syntax {
        /// 1-based line of the offending token.
        line: usize,
        /// What was expected and what was found.
        message: String,
    },
    /// The JSON front end could not decode the document.
    Json(String),
    /// The document's `fugue_program` version is not one this build reads.
    UnsupportedVersion {
        /// The version the document declares.
        found: u64,
        /// The version this build reads ([`FORMAT_VERSION`]).
        supported: u32,
    },
    /// The program is well-formed but does not check: an unknown variable,
    /// distribution or function, a wrong argument count, a `break` outside a
    /// loop, a non-finite literal, a non-identifier name, or nesting beyond
    /// [`MAX_NESTING`] (which the text front end, refusing as it reads,
    /// reports as a [`Syntax`](ProgramError::Syntax) error with its line).
    Check(String),
    /// The data payload is malformed.
    Data(String),
}

impl fmt::Display for ProgramError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProgramError::Syntax { line, message } => write!(f, "line {line}: {message}"),
            ProgramError::Json(m) => write!(f, "invalid program JSON: {m}"),
            ProgramError::UnsupportedVersion { found, supported } => write!(
                f,
                "unsupported program format version {found}: this build of fugue reads \
                 version {supported}"
            ),
            ProgramError::Check(m) | ProgramError::Data(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for ProgramError {}

impl Program {
    /// Parse the text syntax (the grammar is in the [module docs](self)).
    ///
    /// Parsing needs no registry: names are resolved by
    /// [`compile`](Program::compile), so an unknown distribution is a compile
    /// error, not a syntax error.
    ///
    /// ```rust
    /// use fugue::program::{Program, ProgramError};
    ///
    /// let p = Program::parse(
    ///     r#"let mu <- sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap());
    ///        observe(addr!("y"), Normal(mu, 1.0), 0.5);
    ///        pure(mu)"#,
    /// )
    /// .unwrap();
    /// assert_eq!(p.body.len(), 2);
    ///
    /// match Program::parse("let x = 1.0;\nlet y = ;\npure(x)") {
    ///     Err(ProgramError::Syntax { line, .. }) => assert_eq!(line, 2),
    ///     other => panic!("{other:?}"),
    /// }
    /// ```
    pub fn parse(src: &str) -> Result<Program, ProgramError> {
        parse::parse_program(src)
    }

    /// Check the program against a registry and data, resolving every name,
    /// and prepare it for building models.
    ///
    /// Everything that can be known before running is checked here: unknown
    /// variables (including assignment to one), unknown distributions and
    /// functions, argument counts, `break` outside a loop, and the structural
    /// checks of [`to_json`](Program::to_json). The data's names are bound as
    /// variables in an outer scope (a program's own `let` may shadow them).
    ///
    /// ```rust
    /// use fugue::program::{Data, Program, Registry};
    ///
    /// let program = Program::parse("pure(y.len())").unwrap();
    /// let registry = Registry::new();
    /// assert!(program.compile(&registry, &Data::new()).is_err()); // unknown `y`
    /// let compiled = program
    ///     .compile(&registry, &Data::new().with("y", vec![1.0, 2.0]))
    ///     .unwrap();
    /// # let _ = compiled;
    /// ```
    pub fn compile(
        &self,
        registry: &Registry,
        data: &Data,
    ) -> Result<CompiledProgram, ProgramError> {
        self.validate()?;
        let mut resolver = compile::Resolver::new(registry);
        let mut base: Vec<Value> = Vec::with_capacity(data.len());
        for (name, value) in data.iter() {
            let slot = resolver.bind(name);
            debug_assert_eq!(slot, base.len());
            base.push(value.clone());
        }
        resolver.push_scope();
        let body = resolver.stmts(&self.body)?;
        let ret = resolver.expr(&self.ret)?;
        base.resize(resolver.slot_count(), Value::F64(f64::NAN));
        Ok(CompiledProgram {
            shared: Arc::new(interp::Shared {
                body: body.into(),
                ret,
                warnings: Mutex::new(Vec::new()),
            }),
            base: base.into(),
        })
    }
}

impl FromStr for Program {
    type Err = ProgramError;

    /// Same as [`Program::parse`].
    fn from_str(s: &str) -> Result<Program, ProgramError> {
        Program::parse(s)
    }
}

/// A checked program bound to its data, ready to build models.
///
/// [`build`](CompiledProgram::build) constructs a fresh single-use `Model`,
/// so `|| compiled.build()` is the `model_fn` every fugue inference driver
/// takes. A `CompiledProgram` is `Send + Sync`; clones share their warning
/// buffer.
///
/// ```rust
/// use fugue::program::{Data, Program, Registry, Value};
/// use fugue::*;
/// use rand::{rngs::StdRng, SeedableRng};
///
/// let program = Program::parse(
///     r#"let p <- sample(addr!("p"), Beta(2.0, 2.0));
///        for i in 0..flips.len() {
///            observe(addr!("flip", i), Bernoulli(p), flips[i]);
///        }
///        pure(p > 0.5)"#,
/// )
/// .unwrap();
/// let data = Data::from_json(r#"{"flips": [true, false, true, true]}"#).unwrap();
/// let compiled = program.compile(&Registry::new(), &data).unwrap();
///
/// let mut rng = StdRng::seed_from_u64(7);
/// let (favours_heads, trace) = runtime::handler::run(
///     PriorHandler { rng: &mut rng, trace: Trace::default() },
///     compiled.build(),
/// );
/// assert!(matches!(favours_heads, Value::Bool(_)));
/// // Addresses are exactly what `addr!` builds in Rust.
/// assert!(trace.get_f64(&addr!("p")).is_some());
/// assert!(trace.log_likelihood.is_finite());
///
/// // Any inference driver takes it.
/// let draws = adaptive_mcmc_chain(&mut rng, || compiled.build(), 200, 100);
/// assert_eq!(draws.len(), 200);
/// ```
#[derive(Clone)]
pub struct CompiledProgram {
    shared: Arc<interp::Shared>,
    base: Arc<[Value]>,
}

impl fmt::Debug for CompiledProgram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompiledProgram")
            .field("statements", &self.shared.body.len())
            .field("slots", &self.base.len())
            .finish_non_exhaustive()
    }
}

impl CompiledProgram {
    /// Build a fresh single-use model returning the program's value.
    pub fn build(&self) -> Model<Value> {
        interp::start(self.shared.clone(), &self.base)
    }

    /// Build a model returning the program's value as an `f64`: a number
    /// converts, a bool reads as `1.0`/`0.0`, an array as `NaN` — the
    /// playground interpreter's `Model<f64>` contract. For other result
    /// types, map [`build`](CompiledProgram::build) with [`Value`]'s
    /// accessors:
    ///
    /// ```rust
    /// use fugue::program::{Data, Program, Registry};
    /// use fugue::*;
    ///
    /// let compiled = Program::parse(r#"let k <- sample(addr!("k"), Poisson(3.0)); pure(k)"#)
    ///     .unwrap()
    ///     .compile(&Registry::new(), &Data::new())
    ///     .unwrap();
    /// let as_f64: Model<f64> = compiled.build_f64();
    /// let as_u64: Model<u64> = compiled.build().map(|v| v.as_u64().unwrap_or(0));
    /// # let _ = (as_f64, as_u64);
    /// ```
    pub fn build_f64(&self) -> Model<f64> {
        self.build().map(|v| v.to_f64_lossy())
    }

    /// Drain the runtime warnings accumulated by this program's models so
    /// far (each a soft error that was mapped to weight `-inf`). At most 64
    /// are kept between drains.
    ///
    /// ```rust
    /// use fugue::program::{Data, Program, Registry};
    /// use fugue::*;
    /// use rand::{rngs::StdRng, SeedableRng};
    ///
    /// let compiled = Program::parse(r#"let s <- sample(addr!("s"), Normal(0.0, -1.0)); pure(s)"#)
    ///     .unwrap()
    ///     .compile(&Registry::new(), &Data::new())
    ///     .unwrap();
    /// let (_, trace) = runtime::handler::run(
    ///     PriorHandler { rng: &mut StdRng::seed_from_u64(1), trace: Trace::default() },
    ///     compiled.build(),
    /// );
    /// assert_eq!(trace.total_log_weight(), f64::NEG_INFINITY);
    /// let warnings = compiled.take_warnings();
    /// assert!(warnings[0].starts_with("sample `s`"));
    /// assert!(compiled.take_warnings().is_empty());
    /// ```
    pub fn take_warnings(&self) -> Vec<String> {
        self.shared
            .warnings
            .lock()
            .map(|mut w| std::mem::take(&mut *w))
            .unwrap_or_default()
    }
}
