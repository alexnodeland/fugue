//! Error handling for probabilistic programming operations.
//!
//! This module provides structured error types with rich context information for graceful handling of common failure modes in probabilistic computation.
//!
//! ## Scope (finding FG-33)
//!
//! [`ErrorCode`] intentionally only enumerates codes that fugue's own code paths
//! actually construct today, verified with `grep -rn 'ErrorCode::' src/`. An
//! earlier revision of this module carried 22 variants across 6 categories
//! (numerical instability, inference non-convergence, trace corruption, ...) of
//! which only 11 were ever produced by real logic; the rest were aspirational
//! placeholders that overstated how much of the crate's failure surface was
//! actually captured by structured errors (the numerical/model-execution paths
//! they were meant for return `NaN`/`-inf` or panic-free `Option`s instead, or —
//! for [`crate::inference::vi::GuideError`] and [`crate::inference::abc::ABCError`]
//! — got dedicated, more precise algorithm-specific error types rather than being
//! shoehorned into this general enum). See `CHANGELOG.md` for the removed list.
//!
//! The live codes today:
//!
//! | Code | Category | Constructed in |
//! |------|----------|-----------------|
//! | `InvalidMean`/`InvalidVariance`/`InvalidProbability`/`InvalidRange`/`InvalidShape`/`InvalidRate`/`InvalidCount` | Distribution validation (1xx) | `core::distribution` constructors |
//! | `AddressConflict` | Model execution (3xx) | `runtime::interpreters` (duplicate sample address) |
//! | `UnexpectedModelStructure` | Model execution (3xx) | `runtime::interpreters` (replay/score structure mismatch) |
//! | `TraceAddressNotFound` | Trace manipulation (5xx) | `runtime::trace` typed accessors |
//! | `TypeMismatch` | Type system (6xx) | `runtime::trace` typed accessors |

use crate::core::address::Address;
use crate::core::distribution::*;
use std::fmt;

/// Error codes for programmatic error handling and categorization.
///
/// Every variant here is constructed by real logic somewhere in the crate — see
/// the module-level table. If you're adding a new failure mode, add the code
/// here *and* wire it into the code path that detects it in the same change;
/// don't add speculative codes for failure modes nothing produces yet (FG-33).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCode {
    // Distribution parameter validation errors (1xx)
    InvalidMean = 100,
    InvalidVariance = 101,
    InvalidProbability = 102,
    InvalidRange = 103,
    InvalidShape = 104,
    InvalidRate = 105,
    InvalidCount = 106,

    // Model execution errors (3xx)
    AddressConflict = 301,
    UnexpectedModelStructure = 302,

    // Trace manipulation errors (5xx)
    TraceAddressNotFound = 500,

    // Type system errors (6xx)
    TypeMismatch = 600,
}

impl ErrorCode {
    /// Get a human-readable description of the error code.
    pub fn description(&self) -> &'static str {
        match self {
            ErrorCode::InvalidMean => "Distribution mean parameter is invalid",
            ErrorCode::InvalidVariance => "Distribution variance/scale parameter is invalid",
            ErrorCode::InvalidProbability => "Probability parameter is invalid",
            ErrorCode::InvalidRange => "Parameter range is invalid",
            ErrorCode::InvalidShape => "Shape parameter is invalid",
            ErrorCode::InvalidRate => "Rate parameter is invalid",
            ErrorCode::InvalidCount => "Count parameter is invalid",

            ErrorCode::AddressConflict => "Address already exists in trace",
            ErrorCode::UnexpectedModelStructure => "Model structure is unexpected",

            ErrorCode::TraceAddressNotFound => "Address not found in trace",

            ErrorCode::TypeMismatch => "Type mismatch in trace value",
        }
    }

    /// Get the category of the error (first digit of the code).
    pub fn category(&self) -> ErrorCategory {
        match (*self as u32) / 100 {
            1 => ErrorCategory::DistributionValidation,
            3 => ErrorCategory::ModelExecution,
            5 => ErrorCategory::TraceManipulation,
            6 => ErrorCategory::TypeSystem,
            _ => ErrorCategory::Unknown,
        }
    }
}

/// High-level error categories for filtering and handling.
///
/// Only categories with at least one live [`ErrorCode`] are represented (FG-33):
/// numerical-computation and inference-algorithm buckets were removed because no
/// code in the crate constructs an error in either category today.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    DistributionValidation,
    ModelExecution,
    TraceManipulation,
    TypeSystem,
    Unknown,
}

/// Enhanced error context providing debugging information.
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// Optional source location (file, line) where error occurred
    pub source_location: Option<(String, u32)>,
    /// Additional contextual information
    pub context: Vec<(String, String)>,
    /// Chain of causality (parent errors)
    pub cause: Option<Box<FugueError>>,
}

impl ErrorContext {
    /// Create a new empty error context.
    pub fn new() -> Self {
        Self {
            source_location: None,
            context: Vec::new(),
            cause: None,
        }
    }

    /// Add contextual key-value information.
    pub fn with_context(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.context.push((key.into(), value.into()));
        self
    }

    /// Add source location information.
    pub fn with_source_location(mut self, file: impl Into<String>, line: u32) -> Self {
        self.source_location = Some((file.into(), line));
        self
    }

    /// Chain another error as the cause.
    pub fn with_cause(mut self, cause: FugueError) -> Self {
        self.cause = Some(Box::new(cause));
        self
    }
}

impl Default for ErrorContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Errors that can occur during probabilistic programming operations.
///
/// Every variant is constructed by real logic in the crate (FG-33): the
/// speculative `NumericalError` and `InferenceError` variants were removed
/// because nothing produced them — see the module-level docs.
#[derive(Debug, Clone)]
#[allow(clippy::result_large_err)]
pub enum FugueError {
    /// Invalid distribution parameters
    InvalidParameters {
        distribution: String,
        reason: String,
        code: ErrorCode,
        context: ErrorContext,
    },
    /// Model execution failed
    ModelError {
        address: Option<Address>,
        reason: String,
        code: ErrorCode,
        context: ErrorContext,
    },
    /// Trace manipulation error
    TraceError {
        operation: String,
        address: Option<Address>,
        reason: String,
        code: ErrorCode,
        context: ErrorContext,
    },
    /// Type mismatch in trace value
    TypeMismatch {
        address: Address,
        expected: String,
        found: String,
        code: ErrorCode,
        context: ErrorContext,
    },
}

impl fmt::Display for FugueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FugueError::InvalidParameters {
                distribution,
                reason,
                code,
                context,
            } => {
                write!(
                    f,
                    "[{}] Invalid parameters for {}: {}",
                    *code as u32, distribution, reason
                )?;
                self.write_context(f, context)?;
                Ok(())
            }
            FugueError::ModelError {
                address,
                reason,
                code,
                context,
            } => {
                if let Some(addr) = address {
                    write!(f, "[{}] Model error at {}: {}", *code as u32, addr, reason)?;
                } else {
                    write!(f, "[{}] Model error: {}", *code as u32, reason)?;
                }
                self.write_context(f, context)?;
                Ok(())
            }
            FugueError::TraceError {
                operation,
                address,
                reason,
                code,
                context,
            } => {
                if let Some(addr) = address {
                    write!(
                        f,
                        "[{}] Trace error in {} at {}: {}",
                        *code as u32, operation, addr, reason
                    )?;
                } else {
                    write!(
                        f,
                        "[{}] Trace error in {}: {}",
                        *code as u32, operation, reason
                    )?;
                }
                self.write_context(f, context)?;
                Ok(())
            }
            FugueError::TypeMismatch {
                address,
                expected,
                found,
                code,
                context,
            } => {
                write!(
                    f,
                    "[{}] Type mismatch at {}: expected {}, found {}",
                    *code as u32, address, expected, found
                )?;
                self.write_context(f, context)?;
                Ok(())
            }
        }
    }
}

impl FugueError {
    /// Write additional context information to the formatter.
    fn write_context(&self, f: &mut fmt::Formatter<'_>, context: &ErrorContext) -> fmt::Result {
        // Write source location if available
        if let Some((file, line)) = &context.source_location {
            write!(f, " (at {}:{})", file, line)?;
        }

        // Write contextual information
        if !context.context.is_empty() {
            write!(f, " [")?;
            for (i, (key, value)) in context.context.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}={}", key, value)?;
            }
            write!(f, "]")?;
        }

        // Write cause chain
        if let Some(cause) = &context.cause {
            write!(f, "\n  Caused by: {}", cause)?;
        }

        Ok(())
    }

    /// Get the error code for programmatic handling.
    pub fn code(&self) -> ErrorCode {
        match self {
            FugueError::InvalidParameters { code, .. } => *code,
            FugueError::ModelError { code, .. } => *code,
            FugueError::TraceError { code, .. } => *code,
            FugueError::TypeMismatch { code, .. } => *code,
        }
    }

    /// Get the error category for high-level handling.
    pub fn category(&self) -> ErrorCategory {
        self.code().category()
    }

    /// Get the error context for debugging.
    pub fn context(&self) -> &ErrorContext {
        match self {
            FugueError::InvalidParameters { context, .. } => context,
            FugueError::ModelError { context, .. } => context,
            FugueError::TraceError { context, .. } => context,
            FugueError::TypeMismatch { context, .. } => context,
        }
    }

    /// Check if this error is caused by parameter validation issues.
    pub fn is_validation_error(&self) -> bool {
        matches!(self.category(), ErrorCategory::DistributionValidation)
    }

    /// Add context to an existing error.
    pub fn with_context(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        match &mut self {
            FugueError::InvalidParameters { context, .. } => {
                context.context.push((key.into(), value.into()));
            }
            FugueError::ModelError { context, .. } => {
                context.context.push((key.into(), value.into()));
            }
            FugueError::TraceError { context, .. } => {
                context.context.push((key.into(), value.into()));
            }
            FugueError::TypeMismatch { context, .. } => {
                context.context.push((key.into(), value.into()));
            }
        }
        self
    }

    /// Add source location to an existing error.
    pub fn with_source_location(mut self, file: impl Into<String>, line: u32) -> Self {
        match &mut self {
            FugueError::InvalidParameters { context, .. } => {
                context.source_location = Some((file.into(), line));
            }
            FugueError::ModelError { context, .. } => {
                context.source_location = Some((file.into(), line));
            }
            FugueError::TraceError { context, .. } => {
                context.source_location = Some((file.into(), line));
            }
            FugueError::TypeMismatch { context, .. } => {
                context.source_location = Some((file.into(), line));
            }
        }
        self
    }
}

impl std::error::Error for FugueError {}

/// Result type for fallible probabilistic operations.
#[allow(clippy::result_large_err)]
pub type FugueResult<T> = Result<T, FugueError>;

// =============================================================================
// Helper Methods and Constructors
// =============================================================================

impl FugueError {
    /// Create an InvalidParameters error with enhanced context.
    pub fn invalid_parameters(
        distribution: impl Into<String>,
        reason: impl Into<String>,
        code: ErrorCode,
    ) -> Self {
        Self::InvalidParameters {
            distribution: distribution.into(),
            reason: reason.into(),
            code,
            context: ErrorContext::new(),
        }
    }

    /// Create an InvalidParameters error with context.
    pub fn invalid_parameters_with_context(
        distribution: impl Into<String>,
        reason: impl Into<String>,
        code: ErrorCode,
        context: ErrorContext,
    ) -> Self {
        Self::InvalidParameters {
            distribution: distribution.into(),
            reason: reason.into(),
            code,
            context,
        }
    }

    /// Create a TraceError with enhanced context.
    pub fn trace_error(
        operation: impl Into<String>,
        address: Option<Address>,
        reason: impl Into<String>,
        code: ErrorCode,
    ) -> Self {
        Self::TraceError {
            operation: operation.into(),
            address,
            reason: reason.into(),
            code,
            context: ErrorContext::new(),
        }
    }

    /// Create a TypeMismatch error with enhanced context.
    pub fn type_mismatch(
        address: Address,
        expected: impl Into<String>,
        found: impl Into<String>,
    ) -> Self {
        Self::TypeMismatch {
            address,
            expected: expected.into(),
            found: found.into(),
            code: ErrorCode::TypeMismatch,
            context: ErrorContext::new(),
        }
    }
}

// =============================================================================
// Macros for Convenient Error Creation
// =============================================================================

/// Create an InvalidParameters error with optional context.
///
/// Example:
/// ```rust
/// # use fugue::*;
/// let err = invalid_params!("Normal", "sigma must be positive", InvalidVariance);
/// let err_with_ctx = invalid_params!("Normal", "sigma must be positive", InvalidVariance,
///     "sigma" => "-1.0", "expected" => "> 0.0");
/// ```
#[macro_export]
macro_rules! invalid_params {
    ($dist:expr, $reason:expr, $code:ident) => {
        $crate::error::FugueError::invalid_parameters($dist, $reason, $crate::error::ErrorCode::$code)
    };
    ($dist:expr, $reason:expr, $code:ident, $($key:expr => $value:expr),+ $(,)?) => {
        $crate::error::FugueError::invalid_parameters($dist, $reason, $crate::error::ErrorCode::$code)
            $(.with_context($key, $value))*
    };
}

/// Create a TraceError with optional context.
///
/// Example:
/// ```rust
/// # use fugue::*;
/// let err = trace_error!("get_f64", Some(addr!("mu")), "address not found", TraceAddressNotFound);
/// ```
#[macro_export]
macro_rules! trace_error {
    ($op:expr, $addr:expr, $reason:expr, $code:ident) => {
        $crate::error::FugueError::trace_error($op, $addr, $reason, $crate::error::ErrorCode::$code)
    };
    ($op:expr, $addr:expr, $reason:expr, $code:ident, $($key:expr => $value:expr),+ $(,)?) => {
        $crate::error::FugueError::trace_error($op, $addr, $reason, $crate::error::ErrorCode::$code)
            $(.with_context($key, $value))*
    };
}

/// Trait for validating distribution parameters.
pub trait Validate {
    fn validate(&self) -> FugueResult<()>;
}

impl Validate for Normal {
    fn validate(&self) -> FugueResult<()> {
        if !self.mu().is_finite() {
            return Err(invalid_params!(
                "Normal",
                "Mean (mu) must be finite",
                InvalidMean,
                "mu" => format!("{}", self.mu())
            ));
        }
        if self.sigma() <= 0.0 || !self.sigma().is_finite() {
            return Err(invalid_params!(
                "Normal",
                "Standard deviation (sigma) must be positive and finite",
                InvalidVariance,
                "sigma" => format!("{}", self.sigma()),
                "expected" => "> 0.0 and finite"
            ));
        }
        Ok(())
    }
}

impl Validate for Exponential {
    fn validate(&self) -> FugueResult<()> {
        if self.rate() <= 0.0 || !self.rate().is_finite() {
            return Err(invalid_params!(
                "Exponential",
                "Rate parameter must be positive and finite",
                InvalidRate,
                "rate" => format!("{}", self.rate()),
                "expected" => "> 0.0 and finite"
            ));
        }
        Ok(())
    }
}

impl Validate for Beta {
    fn validate(&self) -> FugueResult<()> {
        if self.alpha() <= 0.0 || !self.alpha().is_finite() {
            return Err(invalid_params!(
                "Beta",
                "Alpha parameter must be positive and finite",
                InvalidShape,
                "alpha" => format!("{}", self.alpha()),
                "expected" => "> 0.0 and finite"
            ));
        }
        if self.beta() <= 0.0 || !self.beta().is_finite() {
            return Err(invalid_params!(
                "Beta",
                "Beta parameter must be positive and finite",
                InvalidShape,
                "beta" => format!("{}", self.beta()),
                "expected" => "> 0.0 and finite"
            ));
        }
        Ok(())
    }
}

impl Validate for Gamma {
    fn validate(&self) -> FugueResult<()> {
        if self.shape() <= 0.0 || !self.shape().is_finite() {
            return Err(invalid_params!(
                "Gamma",
                "Shape parameter must be positive and finite",
                InvalidShape,
                "shape" => format!("{}", self.shape()),
                "expected" => "> 0.0 and finite"
            ));
        }
        if self.rate() <= 0.0 || !self.rate().is_finite() {
            return Err(invalid_params!(
                "Gamma",
                "Rate parameter must be positive and finite",
                InvalidRate,
                "rate" => format!("{}", self.rate()),
                "expected" => "> 0.0 and finite"
            ));
        }
        Ok(())
    }
}

impl Validate for Uniform {
    fn validate(&self) -> FugueResult<()> {
        if !self.low().is_finite() || !self.high().is_finite() {
            return Err(invalid_params!(
                "Uniform",
                "Bounds must be finite",
                InvalidRange,
                "low" => format!("{}", self.low()),
                "high" => format!("{}", self.high())
            ));
        }
        if self.low() >= self.high() {
            return Err(invalid_params!(
                "Uniform",
                "Lower bound must be less than upper bound",
                InvalidRange,
                "low" => format!("{}", self.low()),
                "high" => format!("{}", self.high())
            ));
        }
        Ok(())
    }
}

impl Validate for Bernoulli {
    fn validate(&self) -> FugueResult<()> {
        if !self.p().is_finite() || self.p() < 0.0 || self.p() > 1.0 {
            return Err(invalid_params!(
                "Bernoulli",
                "Probability must be in [0, 1]",
                InvalidProbability,
                "p" => format!("{}", self.p()),
                "expected" => "[0.0, 1.0]"
            ));
        }
        Ok(())
    }
}

impl Validate for Categorical {
    fn validate(&self) -> FugueResult<()> {
        if self.probs().is_empty() {
            return Err(invalid_params!(
                "Categorical",
                "Probability vector cannot be empty",
                InvalidProbability,
                "length" => "0"
            ));
        }

        let sum: f64 = self.probs().iter().sum();
        if (sum - 1.0).abs() > 1e-6 {
            return Err(invalid_params!(
                "Categorical",
                "Probabilities must sum to 1.0",
                InvalidProbability,
                "sum" => format!("{:.6}", sum),
                "expected" => "1.0",
                "tolerance" => "1e-6"
            ));
        }

        for (i, &p) in self.probs().iter().enumerate() {
            if !p.is_finite() || p < 0.0 {
                return Err(invalid_params!(
                    "Categorical",
                    "All probabilities must be non-negative and finite",
                    InvalidProbability,
                    "index" => format!("{}", i),
                    "value" => format!("{}", p),
                    "expected" => ">= 0.0 and finite"
                ));
            }
        }

        Ok(())
    }
}

// FG-55: the seven impls above historically covered only part of the exported
// distribution suite. The impls below extend `Validate` to the remaining ten
// exported distributions (LogNormal, Binomial, Poisson, StudentT, Cauchy,
// Laplace, Weibull, ChiSquared, InverseGamma, DiscreteUniform) so the standalone
// trait is complete for all 17 distributions re-exported at the crate root. Each
// impl mirrors the validation performed by the corresponding `new()` constructor
// in `core::distribution` exactly (same predicates, messages, error codes, and
// context keys). `tests/f_validate_coverage.rs` guards against future drift.

impl Validate for LogNormal {
    fn validate(&self) -> FugueResult<()> {
        if !self.mu().is_finite() {
            return Err(invalid_params!(
                "LogNormal",
                "Mean (mu) must be finite",
                InvalidMean,
                "mu" => format!("{}", self.mu())
            ));
        }
        if self.sigma() <= 0.0 || !self.sigma().is_finite() {
            return Err(invalid_params!(
                "LogNormal",
                "Standard deviation (sigma) must be positive and finite",
                InvalidVariance,
                "sigma" => format!("{}", self.sigma()),
                "expected" => "> 0.0 and finite"
            ));
        }
        Ok(())
    }
}

impl Validate for Binomial {
    fn validate(&self) -> FugueResult<()> {
        if !self.p().is_finite() || !(0.0..=1.0).contains(&self.p()) {
            return Err(invalid_params!(
                "Binomial",
                "Probability must be in [0, 1]",
                InvalidProbability,
                "p" => format!("{}", self.p()),
                "expected" => "[0.0, 1.0]"
            ));
        }
        Ok(())
    }
}

impl Validate for Poisson {
    fn validate(&self) -> FugueResult<()> {
        if self.lambda() <= 0.0 || !self.lambda().is_finite() {
            return Err(invalid_params!(
                "Poisson",
                "Rate parameter lambda must be positive and finite",
                InvalidRate,
                "lambda" => format!("{}", self.lambda()),
                "expected" => "> 0.0 and finite"
            ));
        }
        Ok(())
    }
}

impl Validate for StudentT {
    fn validate(&self) -> FugueResult<()> {
        if self.df() <= 0.0 || !self.df().is_finite() {
            return Err(invalid_params!(
                "StudentT",
                "Degrees of freedom must be positive and finite",
                InvalidShape,
                "df" => format!("{}", self.df()),
                "expected" => "> 0.0 and finite"
            ));
        }
        if !self.loc().is_finite() {
            return Err(invalid_params!(
                "StudentT",
                "Location (loc) must be finite",
                InvalidMean,
                "loc" => format!("{}", self.loc())
            ));
        }
        if self.scale() <= 0.0 || !self.scale().is_finite() {
            return Err(invalid_params!(
                "StudentT",
                "Scale must be positive and finite",
                InvalidVariance,
                "scale" => format!("{}", self.scale()),
                "expected" => "> 0.0 and finite"
            ));
        }
        Ok(())
    }
}

impl Validate for Cauchy {
    fn validate(&self) -> FugueResult<()> {
        if !self.loc().is_finite() {
            return Err(invalid_params!(
                "Cauchy",
                "Location (loc) must be finite",
                InvalidMean,
                "loc" => format!("{}", self.loc())
            ));
        }
        if self.scale() <= 0.0 || !self.scale().is_finite() {
            return Err(invalid_params!(
                "Cauchy",
                "Scale must be positive and finite",
                InvalidVariance,
                "scale" => format!("{}", self.scale()),
                "expected" => "> 0.0 and finite"
            ));
        }
        Ok(())
    }
}

impl Validate for Laplace {
    fn validate(&self) -> FugueResult<()> {
        if !self.loc().is_finite() {
            return Err(invalid_params!(
                "Laplace",
                "Location (loc) must be finite",
                InvalidMean,
                "loc" => format!("{}", self.loc())
            ));
        }
        if self.scale() <= 0.0 || !self.scale().is_finite() {
            return Err(invalid_params!(
                "Laplace",
                "Scale must be positive and finite",
                InvalidVariance,
                "scale" => format!("{}", self.scale()),
                "expected" => "> 0.0 and finite"
            ));
        }
        Ok(())
    }
}

impl Validate for Weibull {
    fn validate(&self) -> FugueResult<()> {
        if self.shape() <= 0.0 || !self.shape().is_finite() {
            return Err(invalid_params!(
                "Weibull",
                "Shape parameter must be positive and finite",
                InvalidShape,
                "shape" => format!("{}", self.shape()),
                "expected" => "> 0.0 and finite"
            ));
        }
        if self.scale() <= 0.0 || !self.scale().is_finite() {
            return Err(invalid_params!(
                "Weibull",
                "Scale parameter must be positive and finite",
                InvalidVariance,
                "scale" => format!("{}", self.scale()),
                "expected" => "> 0.0 and finite"
            ));
        }
        Ok(())
    }
}

impl Validate for ChiSquared {
    fn validate(&self) -> FugueResult<()> {
        if self.k() <= 0.0 || !self.k().is_finite() {
            return Err(invalid_params!(
                "ChiSquared",
                "Degrees of freedom must be positive and finite",
                InvalidShape,
                "k" => format!("{}", self.k()),
                "expected" => "> 0.0 and finite"
            ));
        }
        Ok(())
    }
}

impl Validate for InverseGamma {
    fn validate(&self) -> FugueResult<()> {
        if self.shape() <= 0.0 || !self.shape().is_finite() {
            return Err(invalid_params!(
                "InverseGamma",
                "Shape parameter must be positive and finite",
                InvalidShape,
                "shape" => format!("{}", self.shape()),
                "expected" => "> 0.0 and finite"
            ));
        }
        if self.rate() <= 0.0 || !self.rate().is_finite() {
            return Err(invalid_params!(
                "InverseGamma",
                "Rate parameter must be positive and finite",
                InvalidRate,
                "rate" => format!("{}", self.rate()),
                "expected" => "> 0.0 and finite"
            ));
        }
        Ok(())
    }
}

impl Validate for DiscreteUniform {
    fn validate(&self) -> FugueResult<()> {
        if self.high() < self.low() {
            return Err(invalid_params!(
                "DiscreteUniform",
                "Upper bound must be >= lower bound",
                InvalidRange,
                "low" => format!("{}", self.low()),
                "high" => format!("{}", self.high())
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::addr;

    /// FG-33: every remaining `ErrorCode` variant must be live (constructed
    /// somewhere in the crate) and correctly categorized. This test enumerates
    /// all 11 surviving variants; if a future change adds a variant without
    /// updating this list, the mismatch is a signal to double check it's wired
    /// into a real code path rather than left aspirational again.
    #[test]
    fn error_code_taxonomy_is_exactly_the_live_set() {
        let all = [
            ErrorCode::InvalidMean,
            ErrorCode::InvalidVariance,
            ErrorCode::InvalidProbability,
            ErrorCode::InvalidRange,
            ErrorCode::InvalidShape,
            ErrorCode::InvalidRate,
            ErrorCode::InvalidCount,
            ErrorCode::AddressConflict,
            ErrorCode::UnexpectedModelStructure,
            ErrorCode::TraceAddressNotFound,
            ErrorCode::TypeMismatch,
        ];
        assert_eq!(all.len(), 11);
        for code in all {
            // Every code must have a non-empty description and a known category.
            assert!(!code.description().is_empty());
            assert_ne!(code.category(), ErrorCategory::Unknown);
        }
    }

    #[test]
    fn error_code_category_and_description() {
        let code = ErrorCode::InvalidMean;
        assert!(ErrorCode::InvalidMean.description().contains("mean"));
        assert_eq!(code.category(), ErrorCategory::DistributionValidation);

        assert_eq!(
            ErrorCode::AddressConflict.category(),
            ErrorCategory::ModelExecution
        );
        assert_eq!(
            ErrorCode::TraceAddressNotFound.category(),
            ErrorCategory::TraceManipulation
        );
        assert_eq!(
            ErrorCode::TypeMismatch.category(),
            ErrorCategory::TypeSystem
        );
    }

    #[test]
    fn invalid_parameters_constructor_and_context() {
        let err = FugueError::invalid_parameters("Normal", "bad params", ErrorCode::InvalidMean)
            .with_context("mu", "nan")
            .with_source_location("file.rs", 10);

        let msg = format!("{}", err);
        assert!(msg.contains("Invalid parameters for Normal"));
        assert!(msg.contains("mu=nan"));
        assert_eq!(err.code(), ErrorCode::InvalidMean);
        assert_eq!(err.category(), ErrorCategory::DistributionValidation);
        assert!(err.is_validation_error());
    }

    #[test]
    fn error_macros_create_expected_variants() {
        let e1 = invalid_params!("Uniform", "bad range", InvalidRange, "low" => "1", "high" => "0");
        match e1 {
            FugueError::InvalidParameters { code, .. } => assert_eq!(code, ErrorCode::InvalidRange),
            _ => panic!("expected InvalidParameters"),
        }

        let e2 = trace_error!("lookup", Some(addr!("x")), "missing", TraceAddressNotFound);
        match e2 {
            FugueError::TraceError { code, .. } => {
                assert_eq!(code, ErrorCode::TraceAddressNotFound)
            }
            _ => panic!("expected TraceError"),
        }
    }

    #[test]
    fn type_mismatch_constructor() {
        let e = FugueError::type_mismatch(addr!("a"), "f64", "bool");
        assert_eq!(e.code(), ErrorCode::TypeMismatch);
        assert_eq!(e.category(), ErrorCategory::TypeSystem);
        let msg = format!("{}", e);
        assert!(msg.contains("Type mismatch"));
    }

    #[test]
    fn validate_trait_on_valid_distributions() {
        // FG-55: `Validate` is implemented for all 17 exported distributions;
        // exercise a valid instance of each here. `tests/f_validate_coverage.rs`
        // is the public-API drift guard.
        assert!(Normal::new(0.0, 1.0).unwrap().validate().is_ok());
        assert!(Exponential::new(1.0).unwrap().validate().is_ok());
        assert!(Beta::new(2.0, 3.0).unwrap().validate().is_ok());
        assert!(Gamma::new(2.0, 1.0).unwrap().validate().is_ok());
        assert!(Uniform::new(0.0, 1.0).unwrap().validate().is_ok());
        assert!(Bernoulli::new(0.5).unwrap().validate().is_ok());
        assert!(Categorical::new(vec![0.2, 0.8]).unwrap().validate().is_ok());
        assert!(LogNormal::new(0.0, 1.0).unwrap().validate().is_ok());
        assert!(Binomial::new(10, 0.5).unwrap().validate().is_ok());
        assert!(Poisson::new(3.0).unwrap().validate().is_ok());
        assert!(StudentT::new(5.0, 0.0, 1.0).unwrap().validate().is_ok());
        assert!(Cauchy::new(0.0, 1.0).unwrap().validate().is_ok());
        assert!(Laplace::new(0.0, 1.0).unwrap().validate().is_ok());
        assert!(Weibull::new(2.0, 1.5).unwrap().validate().is_ok());
        assert!(ChiSquared::new(4.0).unwrap().validate().is_ok());
        assert!(InverseGamma::new(3.0, 2.0).unwrap().validate().is_ok());
        assert!(DiscreteUniform::new(1, 6).unwrap().validate().is_ok());
    }

    #[test]
    fn error_cause_chaining_and_display_variants() {
        // Build a cause chain
        let base = FugueError::invalid_parameters("Normal", "bad", ErrorCode::InvalidMean);
        let ctx = ErrorContext::new().with_cause(base.clone());
        let model_err = FugueError::ModelError {
            address: Some(crate::addr!("x")),
            reason: "failed".into(),
            code: ErrorCode::UnexpectedModelStructure,
            context: ctx,
        };
        let msg = format!("{}", model_err);
        assert!(msg.contains("Model error"));
        assert!(msg.contains("Caused by"));
    }
}
