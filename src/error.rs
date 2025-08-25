#![doc = include_str!("../docs/api/error/README.md")]

use crate::core::address::Address;
use crate::core::distribution::*;
use std::fmt;

/// Error codes for programmatic error handling and categorization.
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
    
    // Numerical computation errors (2xx)
    NumericalOverflow = 200,
    NumericalUnderflow = 201,
    NumericalInstability = 202,
    InvalidLogDensity = 203,
    
    // Model execution errors (3xx)
    ModelExecutionFailed = 300,
    AddressConflict = 301,
    UnexpectedModelStructure = 302,
    
    // Inference algorithm errors (4xx)
    InferenceConvergenceFailed = 400,
    InsufficientSamples = 401,
    InvalidInferenceConfig = 402,
    
    // Trace manipulation errors (5xx)
    TraceAddressNotFound = 500,
    TraceCorrupted = 501,
    TraceReplayFailed = 502,
    
    // Type system errors (6xx)
    TypeMismatch = 600,
    UnsupportedType = 601,
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
            
            ErrorCode::NumericalOverflow => "Numerical computation resulted in overflow",
            ErrorCode::NumericalUnderflow => "Numerical computation resulted in underflow", 
            ErrorCode::NumericalInstability => "Numerical computation is unstable",
            ErrorCode::InvalidLogDensity => "Log density computation is invalid",
            
            ErrorCode::ModelExecutionFailed => "Model execution failed",
            ErrorCode::AddressConflict => "Address already exists in trace",
            ErrorCode::UnexpectedModelStructure => "Model structure is unexpected",
            
            ErrorCode::InferenceConvergenceFailed => "Inference algorithm failed to converge",
            ErrorCode::InsufficientSamples => "Insufficient samples for reliable inference",
            ErrorCode::InvalidInferenceConfig => "Inference configuration is invalid",
            
            ErrorCode::TraceAddressNotFound => "Address not found in trace",
            ErrorCode::TraceCorrupted => "Trace data is corrupted",
            ErrorCode::TraceReplayFailed => "Trace replay failed",
            
            ErrorCode::TypeMismatch => "Type mismatch in trace value",
            ErrorCode::UnsupportedType => "Unsupported type for operation",
        }
    }

    /// Get the category of the error (first digit of the code).
    pub fn category(&self) -> ErrorCategory {
        match (*self as u32) / 100 {
            1 => ErrorCategory::DistributionValidation,
            2 => ErrorCategory::NumericalComputation,
            3 => ErrorCategory::ModelExecution,
            4 => ErrorCategory::InferenceAlgorithm,
            5 => ErrorCategory::TraceManipulation,
            6 => ErrorCategory::TypeSystem,
            _ => ErrorCategory::Unknown,
        }
    }
}

/// High-level error categories for filtering and handling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    DistributionValidation,
    NumericalComputation,
    ModelExecution,
    InferenceAlgorithm,
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
#[derive(Debug, Clone)]
pub enum FugueError {
    /// Invalid distribution parameters
    InvalidParameters {
        distribution: String,
        reason: String,
        code: ErrorCode,
        context: ErrorContext,
    },
    /// Numerical computation failed
    NumericalError { 
        operation: String, 
        details: String,
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
    /// Inference algorithm failed
    InferenceError { 
        algorithm: String, 
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
                write!(f, "[{}] Invalid parameters for {}: {}", *code as u32, distribution, reason)?;
                self.write_context(f, context)?;
                Ok(())
            }
            FugueError::NumericalError { 
                operation, 
                details,
                code,
                context,
            } => {
                write!(f, "[{}] Numerical error in {}: {}", *code as u32, operation, details)?;
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
            FugueError::InferenceError { 
                algorithm, 
                reason,
                code,
                context,
            } => {
                write!(f, "[{}] Inference error in {}: {}", *code as u32, algorithm, reason)?;
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
                    write!(f, "[{}] Trace error in {} at {}: {}", *code as u32, operation, addr, reason)?;
                } else {
                    write!(f, "[{}] Trace error in {}: {}", *code as u32, operation, reason)?;
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
            FugueError::NumericalError { code, .. } => *code,
            FugueError::ModelError { code, .. } => *code,
            FugueError::InferenceError { code, .. } => *code,
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
            FugueError::NumericalError { context, .. } => context,
            FugueError::ModelError { context, .. } => context,
            FugueError::InferenceError { context, .. } => context,
            FugueError::TraceError { context, .. } => context,
            FugueError::TypeMismatch { context, .. } => context,
        }
    }

    /// Check if this error is caused by parameter validation issues.
    pub fn is_validation_error(&self) -> bool {
        matches!(self.category(), ErrorCategory::DistributionValidation)
    }

    /// Check if this error is caused by numerical computation issues.
    pub fn is_numerical_error(&self) -> bool {
        matches!(self.category(), ErrorCategory::NumericalComputation)
    }

    /// Check if this error is recoverable (can be handled and retried).
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self.code(),
            ErrorCode::InsufficientSamples
                | ErrorCode::NumericalInstability
                | ErrorCode::InferenceConvergenceFailed
        )
    }
}

impl std::error::Error for FugueError {}

/// Result type for fallible probabilistic operations.
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

    /// Create a NumericalError with enhanced context.
    pub fn numerical_error(
        operation: impl Into<String>,
        details: impl Into<String>,
        code: ErrorCode,
    ) -> Self {
        Self::NumericalError {
            operation: operation.into(),
            details: details.into(),
            code,
            context: ErrorContext::new(),
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

    /// Add context to an existing error.
    pub fn with_context(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        match &mut self {
            FugueError::InvalidParameters { context, .. } => {
                context.context.push((key.into(), value.into()));
            }
            FugueError::NumericalError { context, .. } => {
                context.context.push((key.into(), value.into()));
            }
            FugueError::ModelError { context, .. } => {
                context.context.push((key.into(), value.into()));
            }
            FugueError::InferenceError { context, .. } => {
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
            FugueError::NumericalError { context, .. } => {
                context.source_location = Some((file.into(), line));
            }
            FugueError::ModelError { context, .. } => {
                context.source_location = Some((file.into(), line));
            }
            FugueError::InferenceError { context, .. } => {
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

// =============================================================================
// From Trait Implementations for Common Conversions
// =============================================================================

/// Convert from standard library errors to FugueError.
impl From<std::num::ParseFloatError> for FugueError {
    fn from(err: std::num::ParseFloatError) -> Self {
        FugueError::numerical_error(
            "parse_float",
            format!("Failed to parse float: {}", err),
            ErrorCode::NumericalInstability,
        )
    }
}

impl From<std::num::ParseIntError> for FugueError {
    fn from(err: std::num::ParseIntError) -> Self {
        FugueError::numerical_error(
            "parse_int",
            format!("Failed to parse integer: {}", err),
            ErrorCode::NumericalInstability,
        )
    }
}

/// Helper for converting string errors (common in examples).
impl From<&str> for FugueError {
    fn from(msg: &str) -> Self {
        FugueError::ModelError {
            address: None,
            reason: msg.to_string(),
            code: ErrorCode::ModelExecutionFailed,
            context: ErrorContext::new(),
        }
    }
}

impl From<String> for FugueError {
    fn from(msg: String) -> Self {
        FugueError::ModelError {
            address: None,
            reason: msg,
            code: ErrorCode::ModelExecutionFailed,
            context: ErrorContext::new(),
        }
    }
}

// =============================================================================
// Macros for Convenient Error Creation
// =============================================================================

/// Create an InvalidParameters error with optional context.
/// 
/// # Examples
/// ```
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

/// Create a NumericalError with optional context.
/// 
/// # Examples
/// ```
/// # use fugue::*;
/// let err = numerical_error!("log", "input was negative", NumericalInstability);
/// let err_with_ctx = numerical_error!("log", "input was negative", NumericalInstability,
///     "input" => "-1.5");
/// ```
#[macro_export]
macro_rules! numerical_error {
    ($op:expr, $details:expr, $code:ident) => {
        $crate::error::FugueError::numerical_error($op, $details, $crate::error::ErrorCode::$code)
    };
    ($op:expr, $details:expr, $code:ident, $($key:expr => $value:expr),+ $(,)?) => {
        $crate::error::FugueError::numerical_error($op, $details, $crate::error::ErrorCode::$code)
            $(.with_context($key, $value))*
    };
}

/// Create a TraceError with optional context.
/// 
/// # Examples 
/// ```
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
