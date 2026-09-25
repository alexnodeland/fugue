//! Named distributions and pure functions a program can call: the built-ins,
//! and whatever a host registers.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use crate::core::distribution::*;
use crate::error::FugueResult;

use super::Value;

/// The value type of a sample site: which `Model` sample variant (and which
/// `ChoiceValue`) the site uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SiteType {
    /// A continuous site (`Distribution<f64>`); binds [`Value::F64`].
    F64,
    /// A boolean site (`Distribution<bool>`); binds [`Value::Bool`].
    Bool,
    /// A count site (`Distribution<u64>`); binds [`Value::U64`].
    U64,
    /// An index site (`Distribution<usize>`); binds [`Value::Usize`].
    Usize,
    /// A signed-integer site (`Distribution<i64>`); binds [`Value::Int`].
    I64,
}

impl SiteType {
    /// The Rust type's name: `"f64"`, `"bool"`, `"u64"`, `"usize"` or `"i64"`.
    pub fn name(self) -> &'static str {
        match self {
            SiteType::F64 => "f64",
            SiteType::Bool => "bool",
            SiteType::U64 => "u64",
            SiteType::Usize => "usize",
            SiteType::I64 => "i64",
        }
    }
}

impl fmt::Display for SiteType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A distribution built by a registered constructor, boxed at its site type.
///
/// Any `Distribution<T>` implementation fits, including a host's wrapper
/// type that carries site metadata alongside the distribution it delegates
/// to.
pub enum HostDist {
    /// A continuous distribution.
    F64(Box<dyn Distribution<f64>>),
    /// A distribution over `bool`.
    Bool(Box<dyn Distribution<bool>>),
    /// A distribution over counts.
    U64(Box<dyn Distribution<u64>>),
    /// A distribution over indices.
    Usize(Box<dyn Distribution<usize>>),
    /// A distribution over signed integers.
    I64(Box<dyn Distribution<i64>>),
}

impl HostDist {
    /// The site type this distribution samples.
    pub fn site_type(&self) -> SiteType {
        match self {
            HostDist::F64(_) => SiteType::F64,
            HostDist::Bool(_) => SiteType::Bool,
            HostDist::U64(_) => SiteType::U64,
            HostDist::Usize(_) => SiteType::Usize,
            HostDist::I64(_) => SiteType::I64,
        }
    }
}

impl fmt::Debug for HostDist {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HostDist::{:?}(..)", self.site_type())
    }
}

/// How many arguments a registered distribution or function takes. Checked
/// when a program is compiled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arity {
    /// Exactly this many.
    Exact(usize),
    /// Between the two bounds, inclusive.
    Between(usize, usize),
    /// This many or more.
    AtLeast(usize),
}

impl Arity {
    /// Whether a call with `n` arguments fits.
    pub fn accepts(self, n: usize) -> bool {
        match self {
            Arity::Exact(k) => n == k,
            Arity::Between(lo, hi) => (lo..=hi).contains(&n),
            Arity::AtLeast(k) => n >= k,
        }
    }
}

impl fmt::Display for Arity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Arity::Exact(k) => write!(f, "{k}"),
            Arity::Between(lo, hi) => write!(f, "{lo}..={hi}"),
            Arity::AtLeast(k) => write!(f, "at least {k}"),
        }
    }
}

type DistCtor = dyn Fn(&[Value]) -> Result<HostDist, String> + Send + Sync;
type FuncImpl = dyn Fn(&[Value]) -> Result<Value, String> + Send + Sync;

/// A registered distribution: its declared site type, arity and constructor.
#[derive(Clone)]
pub(crate) struct DistEntry {
    pub(crate) site: SiteType,
    pub(crate) arity: Arity,
    pub(crate) ctor: Arc<DistCtor>,
}

/// A registered pure function.
#[derive(Clone)]
pub(crate) struct FuncEntry {
    pub(crate) arity: Arity,
    pub(crate) f: Arc<FuncImpl>,
}

/// The distributions and pure functions programs can call, by name.
///
/// [`Registry::new`] (also `Default`) holds the built-ins: the distributions
/// `Normal`, `Uniform`, `LogNormal`, `Exponential`, `Beta`, `Gamma`,
/// `InverseGamma`, `StudentT`, `Cauchy`, `Laplace`, `Weibull`, `ChiSquared`
/// (`f64` sites), `Bernoulli` (`bool`), `Binomial`, `Poisson` (`u64`),
/// `Categorical` (`usize`) and `DiscreteUniform` (`i64`), with fugue's
/// constructors and parameterizations; and the functions `exp`, `ln` (alias
/// `log`), `sqrt`, `abs`, `floor`, `sin`, `cos`, `tanh` (one argument) and
/// `pow`, `min`, `max` (two), all on `f64`. `Categorical` takes either its
/// probabilities as arguments, `Categorical(0.2, 0.8)`, or one array,
/// `Categorical(probs)`, of any length.
///
/// A host adds its own with [`register_distribution`](Registry::register_distribution)
/// and [`register_function`](Registry::register_function); registering an
/// existing name replaces it. Names a program can call are identifiers (the
/// text syntax's); an entry under any other name is unreachable.
///
/// ```rust
/// use fugue::program::{Arity, Data, HostDist, Program, Registry, SiteType, Value};
/// use fugue::{Categorical, Distribution};
///
/// let mut registry = Registry::new();
/// // A usize site whose probabilities come from the program's arguments.
/// registry.register_distribution("Pick", SiteType::Usize, Arity::Exact(1), |args| {
///     let p = args[0].as_f64().ok_or("Pick: p must be a number")?;
///     let d = Categorical::new(vec![1.0 - p, p]).map_err(|e| e.to_string())?;
///     Ok(HostDist::Usize(d.clone_box()))
/// });
/// // A pure function over ints and bools.
/// registry.register_function("bump", Arity::Exact(2), |args| {
///     let n = args[0].as_i64().ok_or("bump: n must be an integer")?;
///     let on = args[1].as_bool().ok_or("bump: flag must be a bool")?;
///     Ok(Value::Int(if on { n + 1 } else { n }))
/// });
///
/// let program = Program::parse(
///     r#"let k <- sample(addr!("k"), Pick(0.25));
///        pure(bump(3, k == 1))"#,
/// )
/// .unwrap();
/// assert!(program.compile(&registry, &Data::new()).is_ok());
/// // Unknown names are compile-time errors.
/// assert!(program.compile(&Registry::new(), &Data::new()).is_err());
/// ```
#[derive(Clone)]
pub struct Registry {
    dists: HashMap<String, DistEntry>,
    funcs: HashMap<String, FuncEntry>,
}

impl Default for Registry {
    fn default() -> Self {
        Registry::new()
    }
}

impl fmt::Debug for Registry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut dists: Vec<_> = self.dists.keys().collect();
        let mut funcs: Vec<_> = self.funcs.keys().collect();
        dists.sort();
        funcs.sort();
        f.debug_struct("Registry")
            .field("distributions", &dists)
            .field("functions", &funcs)
            .finish()
    }
}

impl Registry {
    /// A registry holding the built-in distributions and functions.
    pub fn new() -> Registry {
        let mut r = Registry::empty();
        register_builtins(&mut r);
        r
    }

    /// A registry with nothing in it, for hosts that want to expose only
    /// their own vocabulary.
    pub fn empty() -> Registry {
        Registry {
            dists: HashMap::new(),
            funcs: HashMap::new(),
        }
    }

    /// Register a distribution constructor under `name`.
    ///
    /// `site` declares the type of the sites it builds; the constructor must
    /// return a [`HostDist`] of that type. The declaration is what lets a site
    /// keep a stable type when construction fails: an `Err` from the
    /// constructor (or a `HostDist` of another type) is a runtime soft error —
    /// the site is filled with a placeholder of the declared type, the
    /// execution's weight becomes `-inf`, and a warning is recorded (see the
    /// module docs' error policy). `arity` is checked at compile time.
    ///
    /// The constructor sees the argument values exactly as the program
    /// computed them; [`Value`]'s accessors apply the language's coercions.
    /// It must be deterministic in its arguments (replay and scoring rebuild
    /// the same site many times) and must not panic.
    pub fn register_distribution<F>(
        &mut self,
        name: impl Into<String>,
        site: SiteType,
        arity: Arity,
        ctor: F,
    ) -> &mut Registry
    where
        F: Fn(&[Value]) -> Result<HostDist, String> + Send + Sync + 'static,
    {
        self.dists.insert(
            name.into(),
            DistEntry {
                site,
                arity,
                ctor: Arc::new(ctor),
            },
        );
        self
    }

    /// Register a pure function under `name`. `arity` is checked at compile
    /// time; an `Err` at runtime is a soft error (weight `-inf` plus a
    /// warning). The function must be deterministic and must not panic.
    pub fn register_function<F>(
        &mut self,
        name: impl Into<String>,
        arity: Arity,
        f: F,
    ) -> &mut Registry
    where
        F: Fn(&[Value]) -> Result<Value, String> + Send + Sync + 'static,
    {
        self.funcs.insert(
            name.into(),
            FuncEntry {
                arity,
                f: Arc::new(f),
            },
        );
        self
    }

    /// Whether a distribution is registered under `name`.
    pub fn has_distribution(&self, name: &str) -> bool {
        self.dists.contains_key(name)
    }

    /// Whether a function is registered under `name`.
    pub fn has_function(&self, name: &str) -> bool {
        self.funcs.contains_key(name)
    }

    /// The declared site type of the distribution registered under `name`.
    pub fn distribution_site_type(&self, name: &str) -> Option<SiteType> {
        self.dists.get(name).map(|d| d.site)
    }

    pub(crate) fn dist(&self, name: &str) -> Option<&DistEntry> {
        self.dists.get(name)
    }

    pub(crate) fn func(&self, name: &str) -> Option<&FuncEntry> {
        self.funcs.get(name)
    }
}

// ---------------------------------------------------------------------------
// Built-ins
// ---------------------------------------------------------------------------

/// Argument `i` of `name` as an `f64` (numbers, and bools as 0/1).
fn num(args: &[Value], i: usize, name: &str) -> Result<f64, String> {
    args[i].as_f64().ok_or_else(|| {
        format!(
            "`{name}` argument {} must be a number, found {}",
            i + 1,
            args[i].type_name()
        )
    })
}

fn boxed_f64<D: Distribution<f64> + 'static>(r: FugueResult<D>) -> Result<HostDist, String> {
    r.map(|d| HostDist::F64(Box::new(d)))
        .map_err(|e| e.to_string())
}

fn register_builtins(r: &mut Registry) {
    macro_rules! f64_dist {
        ($name:literal, $ty:ident, 1) => {
            r.register_distribution($name, SiteType::F64, Arity::Exact(1), |a| {
                boxed_f64($ty::new(num(a, 0, $name)?))
            });
        };
        ($name:literal, $ty:ident, 2) => {
            r.register_distribution($name, SiteType::F64, Arity::Exact(2), |a| {
                boxed_f64($ty::new(num(a, 0, $name)?, num(a, 1, $name)?))
            });
        };
        ($name:literal, $ty:ident, 3) => {
            r.register_distribution($name, SiteType::F64, Arity::Exact(3), |a| {
                boxed_f64($ty::new(
                    num(a, 0, $name)?,
                    num(a, 1, $name)?,
                    num(a, 2, $name)?,
                ))
            });
        };
    }
    f64_dist!("Normal", Normal, 2);
    f64_dist!("Uniform", Uniform, 2);
    f64_dist!("LogNormal", LogNormal, 2);
    f64_dist!("Beta", Beta, 2);
    f64_dist!("Gamma", Gamma, 2);
    f64_dist!("InverseGamma", InverseGamma, 2);
    f64_dist!("Cauchy", Cauchy, 2);
    f64_dist!("Laplace", Laplace, 2);
    f64_dist!("Weibull", Weibull, 2);
    f64_dist!("Exponential", Exponential, 1);
    f64_dist!("ChiSquared", ChiSquared, 1);
    f64_dist!("StudentT", StudentT, 3);

    r.register_distribution("Bernoulli", SiteType::Bool, Arity::Exact(1), |a| {
        Bernoulli::new(num(a, 0, "Bernoulli")?)
            .map(|d| HostDist::Bool(Box::new(d)))
            .map_err(|e| e.to_string())
    });
    r.register_distribution("Poisson", SiteType::U64, Arity::Exact(1), |a| {
        Poisson::new(num(a, 0, "Poisson")?)
            .map(|d| HostDist::U64(Box::new(d)))
            .map_err(|e| e.to_string())
    });
    r.register_distribution("Binomial", SiteType::U64, Arity::Exact(2), |a| {
        let n = a[0]
            .as_u64()
            .ok_or_else(|| format!("Binomial n must be a non-negative integer, got {}", a[0]))?;
        Binomial::new(n, num(a, 1, "Binomial")?)
            .map(|d| HostDist::U64(Box::new(d)))
            .map_err(|e| e.to_string())
    });
    r.register_distribution("Categorical", SiteType::Usize, Arity::AtLeast(1), |a| {
        // One array argument, or the probabilities themselves.
        let probs = match a {
            [Value::Arr(p)] => p.as_ref().clone(),
            _ => (0..a.len())
                .map(|i| num(a, i, "Categorical"))
                .collect::<Result<Vec<f64>, String>>()?,
        };
        Categorical::new(probs)
            .map(|d| HostDist::Usize(Box::new(d)))
            .map_err(|e| e.to_string())
    });
    r.register_distribution("DiscreteUniform", SiteType::I64, Arity::Exact(2), |a| {
        let bound = |i: usize| {
            a[i].as_i64()
                .ok_or_else(|| format!("DiscreteUniform bounds must be integers, got {}", a[i]))
        };
        DiscreteUniform::new(bound(0)?, bound(1)?)
            .map(|d| HostDist::I64(Box::new(d)))
            .map_err(|e| e.to_string())
    });

    macro_rules! math1 {
        ($($name:literal => $f:expr),* $(,)?) => {$(
            r.register_function($name, Arity::Exact(1), |a| {
                let f: fn(f64) -> f64 = $f;
                Ok(Value::F64(f(num(a, 0, $name)?)))
            });
        )*};
    }
    math1! {
        "exp" => f64::exp,
        "ln" => f64::ln,
        "log" => f64::ln,
        "sqrt" => f64::sqrt,
        "abs" => f64::abs,
        "floor" => f64::floor,
        "sin" => f64::sin,
        "cos" => f64::cos,
        "tanh" => f64::tanh,
    }
    macro_rules! math2 {
        ($($name:literal => $f:expr),* $(,)?) => {$(
            r.register_function($name, Arity::Exact(2), |a| {
                let f: fn(f64, f64) -> f64 = $f;
                Ok(Value::F64(f(num(a, 0, $name)?, num(a, 1, $name)?)))
            });
        )*};
    }
    math2! {
        "pow" => f64::powf,
        "min" => f64::min,
        "max" => f64::max,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtins_are_registered_with_their_site_types() {
        let r = Registry::new();
        for (name, site) in [
            ("Normal", SiteType::F64),
            ("StudentT", SiteType::F64),
            ("Bernoulli", SiteType::Bool),
            ("Poisson", SiteType::U64),
            ("Binomial", SiteType::U64),
            ("Categorical", SiteType::Usize),
            ("DiscreteUniform", SiteType::I64),
        ] {
            assert_eq!(r.distribution_site_type(name), Some(site), "{name}");
        }
        assert_eq!(r.dists.len(), 17);
        for f in [
            "exp", "ln", "log", "sqrt", "abs", "floor", "sin", "cos", "tanh", "pow", "min", "max",
        ] {
            assert!(r.has_function(f), "{f}");
        }
        assert!(!Registry::empty().has_distribution("Normal"));
        assert!(format!("{r:?}").contains("Normal"));
    }

    #[test]
    fn builtin_constructors_validate_arguments() {
        let r = Registry::new();
        let build = |name: &str, args: &[Value]| (r.dist(name).unwrap().ctor)(args);
        assert!(build("Normal", &[Value::F64(0.0), Value::F64(-1.0)]).is_err());
        assert!(build("Normal", &[Value::from(vec![1.0]), Value::F64(1.0)]).is_err());
        assert!(build("Binomial", &[Value::F64(2.5), Value::F64(0.5)]).is_err());
        assert!(build("Binomial", &[Value::Int(-1), Value::F64(0.5)]).is_err());
        assert!(build("DiscreteUniform", &[Value::F64(0.5), Value::Int(3)]).is_err());
        let many = Value::from(vec![1.0 / 100.0; 100]);
        match build("Categorical", &[many]) {
            Ok(HostDist::Usize(_)) => {}
            other => panic!("{other:?}"),
        }
        assert!(build("Categorical", &[Value::F64(0.5), Value::from(vec![0.5])]).is_err());
        let exp = r.func("exp").unwrap();
        assert_eq!((exp.f)(&[Value::Int(0)]), Ok(Value::F64(1.0)));
        assert!((exp.f)(&[Value::from(vec![0.0])]).is_err());
        assert!(Arity::Between(1, 2).accepts(2) && !Arity::Between(1, 2).accepts(3));
        assert_eq!(Arity::AtLeast(1).to_string(), "at least 1");
    }
}
