#![doc = include_str!("../../docs/api/runtime/trace/README.md")]

use crate::core::address::Address;
use crate::error::{FugueError, FugueResult};
use std::collections::BTreeMap;

#[doc = include_str!("../../docs/api/runtime/trace/choice_value.md")]
#[derive(Clone, Debug, PartialEq)]
pub enum ChoiceValue {
    /// Floating-point value (continuous distributions).
    F64(f64),
    /// Signed integer value.
    I64(i64),
    /// Unsigned integer value (Poisson, Binomial counts).
    U64(u64),
    /// Array index value (Categorical choices).
    Usize(usize),
    /// Boolean value (Bernoulli outcomes).
    Bool(bool),
}
impl ChoiceValue {
    /// Try to extract an f64 value, returning None if the type doesn't match.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            ChoiceValue::F64(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to extract a bool value, returning None if the type doesn't match.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            ChoiceValue::Bool(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to extract a u64 value, returning None if the type doesn't match.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            ChoiceValue::U64(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to extract a usize value, returning None if the type doesn't match.
    pub fn as_usize(&self) -> Option<usize> {
        match self {
            ChoiceValue::Usize(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to extract an i64 value, returning None if the type doesn't match.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            ChoiceValue::I64(v) => Some(*v),
            _ => None,
        }
    }

    /// Get the type name as a string for error messages.
    pub fn type_name(&self) -> &'static str {
        match self {
            ChoiceValue::F64(_) => "f64",
            ChoiceValue::Bool(_) => "bool",
            ChoiceValue::U64(_) => "u64",
            ChoiceValue::Usize(_) => "usize",
            ChoiceValue::I64(_) => "i64",
        }
    }
}

#[doc = include_str!("../../docs/api/runtime/trace/choice.md")]
#[derive(Clone, Debug)]
pub struct Choice {
    /// Address where this choice was made.
    pub addr: Address,
    /// Value that was chosen.
    pub value: ChoiceValue,
    /// Log-probability of this value under the generating distribution.
    pub logp: f64,
}

#[doc = include_str!("../../docs/api/runtime/trace/trace.md")]
#[derive(Clone, Debug, Default)]
pub struct Trace {
    /// Map from addresses to the choices made at those sites.
    pub choices: BTreeMap<Address, Choice>,
    /// Accumulated log-prior probability from all sampling sites.
    pub log_prior: f64,
    /// Accumulated log-likelihood from all observation sites.
    pub log_likelihood: f64,
    /// Accumulated log-weight from all factor statements.
    pub log_factors: f64,
}

impl Trace {
    #[doc = include_str!("../../docs/api/runtime/trace/total_log_weight.md")]
    pub fn total_log_weight(&self) -> f64 {
        self.log_prior + self.log_likelihood + self.log_factors
    }

    /// Type-safe accessor for f64 values in the trace.
    pub fn get_f64(&self, addr: &Address) -> Option<f64> {
        self.choices.get(addr)?.value.as_f64()
    }

    /// Type-safe accessor for bool values in the trace.
    pub fn get_bool(&self, addr: &Address) -> Option<bool> {
        self.choices.get(addr)?.value.as_bool()
    }

    /// Type-safe accessor for u64 values in the trace.
    pub fn get_u64(&self, addr: &Address) -> Option<u64> {
        self.choices.get(addr)?.value.as_u64()
    }

    /// Type-safe accessor for usize values in the trace.
    pub fn get_usize(&self, addr: &Address) -> Option<usize> {
        self.choices.get(addr)?.value.as_usize()
    }

    /// Type-safe accessor for i64 values in the trace.
    pub fn get_i64(&self, addr: &Address) -> Option<i64> {
        self.choices.get(addr)?.value.as_i64()
    }

    /// Type-safe accessor that returns a Result for better error handling.
    pub fn get_f64_result(&self, addr: &Address) -> FugueResult<f64> {
        let choice = self
            .choices
            .get(addr)
            .ok_or_else(|| FugueError::trace_error(
                "get_f64",
                Some(addr.clone()),
                "Address not found in trace",
                crate::error::ErrorCode::TraceAddressNotFound,
            ))?;

        choice
            .value
            .as_f64()
            .ok_or_else(|| FugueError::type_mismatch(
                addr.clone(),
                "f64",
                choice.value.type_name(),
            ))
    }

    /// Type-safe accessor that returns a Result for better error handling.
    pub fn get_bool_result(&self, addr: &Address) -> FugueResult<bool> {
        let choice = self
            .choices
            .get(addr)
            .ok_or_else(|| FugueError::trace_error(
                "get_bool",
                Some(addr.clone()),
                "Address not found in trace",
                crate::error::ErrorCode::TraceAddressNotFound,
            ))?;

        choice
            .value
            .as_bool()
            .ok_or_else(|| FugueError::type_mismatch(
                addr.clone(),
                "bool",
                choice.value.type_name(),
            ))
    }

    /// Type-safe accessor that returns a Result for better error handling.
    pub fn get_u64_result(&self, addr: &Address) -> FugueResult<u64> {
        let choice = self
            .choices
            .get(addr)
            .ok_or_else(|| FugueError::trace_error(
                "get_u64",
                Some(addr.clone()),
                "Address not found in trace",
                crate::error::ErrorCode::TraceAddressNotFound,
            ))?;

        choice
            .value
            .as_u64()
            .ok_or_else(|| FugueError::type_mismatch(
                addr.clone(),
                "u64",
                choice.value.type_name(),
            ))
    }

    /// Type-safe accessor that returns a Result for better error handling.
    pub fn get_usize_result(&self, addr: &Address) -> FugueResult<usize> {
        let choice = self
            .choices
            .get(addr)
            .ok_or_else(|| FugueError::trace_error(
                "get_usize",
                Some(addr.clone()),
                "Address not found in trace",
                crate::error::ErrorCode::TraceAddressNotFound,
            ))?;

        choice
            .value
            .as_usize()
            .ok_or_else(|| FugueError::type_mismatch(
                addr.clone(),
                "usize",
                choice.value.type_name(),
            ))
    }

    /// Type-safe accessor that returns a Result for better error handling.
    pub fn get_i64_result(&self, addr: &Address) -> FugueResult<i64> {
        let choice = self
            .choices
            .get(addr)
            .ok_or_else(|| FugueError::trace_error(
                "get_i64",
                Some(addr.clone()),
                "Address not found in trace",
                crate::error::ErrorCode::TraceAddressNotFound,
            ))?;

        choice
            .value
            .as_i64()
            .ok_or_else(|| FugueError::type_mismatch(
                addr.clone(),
                "i64",
                choice.value.type_name(),
            ))
    }

    #[doc = include_str!("../../docs/api/runtime/trace/insert_choice.md")]
    pub fn insert_choice(&mut self, addr: Address, value: ChoiceValue, logp: f64) {
        let choice = Choice {
            addr: addr.clone(),
            value,
            logp,
        };
        self.choices.insert(addr, choice);
    }
}
