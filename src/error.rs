//! Error handling for probabilistic programming operations.
//!
//! This module provides structured error types for graceful handling
//! of common failure modes in probabilistic computation.

use crate::core::address::Address;
use crate::core::distribution::*;
use std::fmt;

/// Errors that can occur during probabilistic programming operations.
#[derive(Debug, Clone)]
pub enum FugueError {
    /// Invalid distribution parameters
    InvalidParameters {
        distribution: String,
        reason: String,
    },
    /// Numerical computation failed
    NumericalError { operation: String, details: String },
    /// Model execution failed
    ModelError {
        address: Option<Address>,
        reason: String,
    },
    /// Inference algorithm failed
    InferenceError { algorithm: String, reason: String },
    /// Trace manipulation error
    TraceError {
        operation: String,
        address: Option<Address>,
        reason: String,
    },
    /// Type mismatch in trace value
    TypeMismatch {
        address: Address,
        expected: String,
        found: String,
    },
}

impl fmt::Display for FugueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FugueError::InvalidParameters {
                distribution,
                reason,
            } => {
                write!(f, "Invalid parameters for {}: {}", distribution, reason)
            }
            FugueError::NumericalError { operation, details } => {
                write!(f, "Numerical error in {}: {}", operation, details)
            }
            FugueError::ModelError { address, reason } => {
                if let Some(addr) = address {
                    write!(f, "Model error at {}: {}", addr, reason)
                } else {
                    write!(f, "Model error: {}", reason)
                }
            }
            FugueError::InferenceError { algorithm, reason } => {
                write!(f, "Inference error in {}: {}", algorithm, reason)
            }
            FugueError::TraceError {
                operation,
                address,
                reason,
            } => {
                if let Some(addr) = address {
                    write!(f, "Trace error in {} at {}: {}", operation, addr, reason)
                } else {
                    write!(f, "Trace error in {}: {}", operation, reason)
                }
            }
            FugueError::TypeMismatch {
                address,
                expected,
                found,
            } => {
                write!(
                    f,
                    "Type mismatch at {}: expected {}, found {}",
                    address, expected, found
                )
            }
        }
    }
}

impl std::error::Error for FugueError {}

/// Result type for fallible probabilistic operations.
pub type FugueResult<T> = Result<T, FugueError>;

/// Trait for validating distribution parameters.
pub trait Validate {
    fn validate(&self) -> FugueResult<()>;
}

impl Validate for Normal {
    fn validate(&self) -> FugueResult<()> {
        if !self.mu().is_finite() {
            return Err(FugueError::InvalidParameters {
                distribution: "Normal".to_string(),
                reason: "Mean (mu) must be finite".to_string(),
            });
        }
        if self.sigma() <= 0.0 || !self.sigma().is_finite() {
            return Err(FugueError::InvalidParameters {
                distribution: "Normal".to_string(),
                reason: "Standard deviation (sigma) must be positive and finite".to_string(),
            });
        }
        Ok(())
    }
}

impl Validate for Exponential {
    fn validate(&self) -> FugueResult<()> {
        if self.rate() <= 0.0 || !self.rate().is_finite() {
            return Err(FugueError::InvalidParameters {
                distribution: "Exponential".to_string(),
                reason: "Rate parameter must be positive and finite".to_string(),
            });
        }
        Ok(())
    }
}

impl Validate for Beta {
    fn validate(&self) -> FugueResult<()> {
        if self.alpha() <= 0.0 || !self.alpha().is_finite() {
            return Err(FugueError::InvalidParameters {
                distribution: "Beta".to_string(),
                reason: "Alpha parameter must be positive and finite".to_string(),
            });
        }
        if self.beta() <= 0.0 || !self.beta().is_finite() {
            return Err(FugueError::InvalidParameters {
                distribution: "Beta".to_string(),
                reason: "Beta parameter must be positive and finite".to_string(),
            });
        }
        Ok(())
    }
}

impl Validate for Gamma {
    fn validate(&self) -> FugueResult<()> {
        if self.shape() <= 0.0 || !self.shape().is_finite() {
            return Err(FugueError::InvalidParameters {
                distribution: "Gamma".to_string(),
                reason: "Shape parameter must be positive and finite".to_string(),
            });
        }
        if self.rate() <= 0.0 || !self.rate().is_finite() {
            return Err(FugueError::InvalidParameters {
                distribution: "Gamma".to_string(),
                reason: "Rate parameter must be positive and finite".to_string(),
            });
        }
        Ok(())
    }
}

impl Validate for Uniform {
    fn validate(&self) -> FugueResult<()> {
        if !self.low().is_finite() || !self.high().is_finite() {
            return Err(FugueError::InvalidParameters {
                distribution: "Uniform".to_string(),
                reason: "Bounds must be finite".to_string(),
            });
        }
        if self.low() >= self.high() {
            return Err(FugueError::InvalidParameters {
                distribution: "Uniform".to_string(),
                reason: "Lower bound must be less than upper bound".to_string(),
            });
        }
        Ok(())
    }
}

impl Validate for Bernoulli {
    fn validate(&self) -> FugueResult<()> {
        if !self.p().is_finite() || self.p() < 0.0 || self.p() > 1.0 {
            return Err(FugueError::InvalidParameters {
                distribution: "Bernoulli".to_string(),
                reason: "Probability must be in [0, 1]".to_string(),
            });
        }
        Ok(())
    }
}

impl Validate for Categorical {
    fn validate(&self) -> FugueResult<()> {
        if self.probs().is_empty() {
            return Err(FugueError::InvalidParameters {
                distribution: "Categorical".to_string(),
                reason: "Probability vector cannot be empty".to_string(),
            });
        }

        let sum: f64 = self.probs().iter().sum();
        if (sum - 1.0).abs() > 1e-6 {
            return Err(FugueError::InvalidParameters {
                distribution: "Categorical".to_string(),
                reason: format!("Probabilities must sum to 1.0, got {:.6}", sum),
            });
        }

        for (i, &p) in self.probs().iter().enumerate() {
            if !p.is_finite() || p < 0.0 {
                return Err(FugueError::InvalidParameters {
                    distribution: "Categorical".to_string(),
                    reason: format!(
                        "Probability at index {} must be non-negative and finite, got {}",
                        i, p
                    ),
                });
            }
        }

        Ok(())
    }
}
