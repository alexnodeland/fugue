//! Runtime values of the program language, and the coercion rules the
//! interpreter applies to them.

use std::fmt;
use std::sync::Arc;

/// A value computed by a [`Program`](super::Program).
///
/// Sample sites bind their **natural** type — `Bernoulli` binds [`Value::Bool`],
/// `Poisson`/`Binomial` bind [`Value::U64`], `Categorical` binds
/// [`Value::Usize`], `DiscreteUniform` binds [`Value::Int`], continuous
/// distributions bind [`Value::F64`] — and a compiled program builds a
/// `Model<Value>`, so a program returns whichever of these it computes.
///
/// The language is dynamically typed with a few numeric coercions, and the
/// accessors below expose exactly those coercions, so a host function reading
/// its arguments sees the same conversions the built-ins do:
///
/// | accessor | accepts |
/// |---|---|
/// | [`as_f64`](Value::as_f64) | any number; `true`/`false` as `1.0`/`0.0` |
/// | [`as_bool`](Value::as_bool) | a bool; a number as "is nonzero" (`NaN` has no truth value) |
/// | [`as_i64`](Value::as_i64) | an integer in range, an integral finite `F64`, a bool as `0`/`1` |
/// | [`as_u64`](Value::as_u64) / [`as_usize`](Value::as_usize) | as `as_i64`, and non-negative |
/// | [`as_array`](Value::as_array) | an array |
///
/// ```rust
/// use fugue::program::Value;
///
/// assert_eq!(Value::Usize(2).as_f64(), Some(2.0));
/// assert_eq!(Value::F64(3.0).as_usize(), Some(3));
/// assert_eq!(Value::F64(2.5).as_i64(), None);
/// assert_eq!(Value::Int(-1).as_u64(), None);
/// assert_eq!(Value::Bool(true).as_f64(), Some(1.0));
/// assert_eq!(Value::F64(0.0).as_bool(), Some(false));
/// assert_eq!(Value::from(vec![0.5, 0.5]).as_array(), Some(&[0.5, 0.5][..]));
/// ```
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// A float: `2.0`-style literals, continuous sites, float arithmetic.
    F64(f64),
    /// A signed integer: `2`-style literals, loop variables, `.len()`,
    /// integer arithmetic, and `i64` sites (`DiscreteUniform`).
    Int(i64),
    /// An unsigned count: the value of a `u64` site (`Poisson`, `Binomial`).
    U64(u64),
    /// An index: the value of a `usize` site (`Categorical`).
    Usize(usize),
    /// A boolean: `true`/`false`, comparisons, `&&`/`||`/`!`, and `bool`
    /// sites (`Bernoulli`).
    Bool(bool),
    /// An array of numbers: a data array or an array literal `[a, b, c]`.
    Arr(Arc<Vec<f64>>),
}

impl Value {
    /// The value's type as the language names it in messages: `"f64"`,
    /// `"int"`, `"u64"`, `"usize"`, `"bool"` or `"array"`.
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::F64(_) => "f64",
            Value::Int(_) => "int",
            Value::U64(_) => "u64",
            Value::Usize(_) => "usize",
            Value::Bool(_) => "bool",
            Value::Arr(_) => "array",
        }
    }

    /// The value as an `f64`: any number converts, and a bool is `1.0`/`0.0`.
    /// `None` for an array.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::F64(x) => Some(*x),
            Value::Int(i) => Some(*i as f64),
            Value::U64(u) => Some(*u as f64),
            Value::Usize(u) => Some(*u as f64),
            Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
            Value::Arr(_) => None,
        }
    }

    /// The value's truth: a bool is itself, a number is "nonzero". `None` for
    /// `NaN` (which is neither) and for an array.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            Value::F64(x) if x.is_nan() => None,
            Value::F64(x) => Some(*x != 0.0),
            Value::Int(i) => Some(*i != 0),
            Value::U64(u) => Some(*u != 0),
            Value::Usize(u) => Some(*u != 0),
            Value::Arr(_) => None,
        }
    }

    /// The value as an `i64`: an integer in range, a finite integral `F64` in
    /// range, or a bool as `0`/`1`. `None` otherwise.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            Value::U64(u) => i64::try_from(*u).ok(),
            Value::Usize(u) => i64::try_from(*u).ok(),
            Value::Bool(b) => Some(i64::from(*b)),
            // [-2^63, 2^63): both bounds are exactly representable as f64.
            Value::F64(x) if x.fract() == 0.0 && *x >= -9.223_372_036_854_776e18 => {
                if *x < 9.223_372_036_854_776e18 {
                    Some(*x as i64)
                } else {
                    None
                }
            }
            Value::F64(_) | Value::Arr(_) => None,
        }
    }

    /// The value as a `u64`: as [`as_i64`](Value::as_i64), but non-negative
    /// (and a `U64` above `i64::MAX` converts too).
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::U64(u) => Some(*u),
            Value::Usize(u) => u64::try_from(*u).ok(),
            // [0, 2^64): both bounds are exactly representable as f64.
            Value::F64(x) if x.fract() == 0.0 && *x >= 0.0 => {
                if *x < 1.844_674_407_370_955_2e19 {
                    Some(*x as u64)
                } else {
                    None
                }
            }
            other => other.as_i64().and_then(|i| u64::try_from(i).ok()),
        }
    }

    /// The value as a `usize`: as [`as_u64`](Value::as_u64), and in range for
    /// the platform's `usize`.
    pub fn as_usize(&self) -> Option<usize> {
        match self {
            Value::Usize(u) => Some(*u),
            other => other.as_u64().and_then(|u| usize::try_from(u).ok()),
        }
    }

    /// The array's elements, or `None` for a scalar.
    pub fn as_array(&self) -> Option<&[f64]> {
        match self {
            Value::Arr(a) => Some(a.as_slice()),
            _ => None,
        }
    }

    /// The value as an `f64` for [`CompiledProgram::build_f64`]: as
    /// [`as_f64`](Value::as_f64), with an array reading as `NaN` (the playground
    /// interpreter's historical conversion).
    ///
    /// [`CompiledProgram::build_f64`]: super::CompiledProgram::build_f64
    pub(crate) fn to_f64_lossy(&self) -> f64 {
        self.as_f64().unwrap_or(f64::NAN)
    }

    /// The value as an arithmetic operand: integers (and bools, as `0`/`1`)
    /// stay exact, floats stay floats, an array is an error.
    pub(crate) fn num(&self) -> Result<Num, String> {
        match self {
            Value::F64(x) => Ok(Num::F(*x)),
            Value::Int(i) => Ok(Num::I(*i)),
            Value::Bool(b) => Ok(Num::I(i64::from(*b))),
            Value::U64(u) => Ok(i64::try_from(*u).map_or(Num::F(*u as f64), Num::I)),
            Value::Usize(u) => Ok(i64::try_from(*u).map_or(Num::F(*u as f64), Num::I)),
            Value::Arr(_) => Err("expected a number, found an array".to_string()),
        }
    }
}

/// An arithmetic operand: exact integer or float.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Num {
    I(i64),
    F(f64),
}

impl Num {
    pub(crate) fn to_f64(self) -> f64 {
        match self {
            Num::I(i) => i as f64,
            Num::F(x) => x,
        }
    }
}

/// Formats the value the way Rust's `Display` formats the corresponding Rust
/// value, which is what makes `addr!("x", i)` in a program byte-identical to
/// `addr!("x", i)` in compiled Rust: `3`, `0.5`, `true`. An array prints as
/// `[1, 2.5]` (arrays are not valid address indices).
impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::F64(x) => write!(f, "{x}"),
            Value::Int(i) => write!(f, "{i}"),
            Value::U64(u) => write!(f, "{u}"),
            Value::Usize(u) => write!(f, "{u}"),
            Value::Bool(b) => write!(f, "{b}"),
            Value::Arr(a) => {
                f.write_str("[")?;
                for (i, x) in a.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{x}")?;
                }
                f.write_str("]")
            }
        }
    }
}

impl From<f64> for Value {
    fn from(x: f64) -> Self {
        Value::F64(x)
    }
}

impl From<i64> for Value {
    fn from(i: i64) -> Self {
        Value::Int(i)
    }
}

impl From<i32> for Value {
    fn from(i: i32) -> Self {
        Value::Int(i64::from(i))
    }
}

impl From<u64> for Value {
    fn from(u: u64) -> Self {
        Value::U64(u)
    }
}

impl From<usize> for Value {
    fn from(u: usize) -> Self {
        Value::Usize(u)
    }
}

impl From<bool> for Value {
    fn from(b: bool) -> Self {
        Value::Bool(b)
    }
}

impl From<Vec<f64>> for Value {
    fn from(v: Vec<f64>) -> Self {
        Value::Arr(Arc::new(v))
    }
}

impl From<&[f64]> for Value {
    fn from(v: &[f64]) -> Self {
        Value::Arr(Arc::new(v.to_vec()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integral_conversions_respect_ranges() {
        assert_eq!(Value::F64(-0.0).as_i64(), Some(0));
        assert_eq!(Value::F64(9.223_372_036_854_776e18).as_i64(), None);
        assert_eq!(
            Value::F64(-9.223_372_036_854_776e18).as_i64(),
            Some(i64::MIN)
        );
        assert_eq!(Value::F64(f64::NAN).as_i64(), None);
        assert_eq!(Value::F64(f64::INFINITY).as_u64(), None);
        assert_eq!(Value::F64(1.844_674_407_370_955_2e19).as_u64(), None);
        assert_eq!(Value::U64(u64::MAX).as_u64(), Some(u64::MAX));
        assert_eq!(Value::U64(u64::MAX).as_i64(), None);
        assert_eq!(Value::Bool(true).as_u64(), Some(1));
        assert_eq!(Value::Int(-3).as_usize(), None);
        assert_eq!(Value::from(vec![1.0]).as_i64(), None);
    }

    #[test]
    fn truthiness_and_display() {
        assert_eq!(Value::F64(f64::NAN).as_bool(), None);
        assert_eq!(Value::Int(2).as_bool(), Some(true));
        assert_eq!(Value::from(vec![1.0]).as_bool(), None);
        assert_eq!(Value::F64(3.0).to_string(), "3");
        assert_eq!(Value::F64(0.5).to_string(), "0.5");
        assert_eq!(Value::Bool(true).to_string(), "true");
        assert_eq!(Value::from(vec![1.0, 2.5]).to_string(), "[1, 2.5]");
        assert_eq!(
            Value::from(vec![1.0]).to_f64_lossy().to_bits(),
            f64::NAN.to_bits()
        );
    }
}
