//! Execution traces capturing choices and accumulated log-weights.
//!
//! A `Trace` records named random choices and additive log terms for prior,
//! likelihood, and arbitrary factors. It is used for replay and scoring.
use crate::core::address::Address;
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq)]
pub enum ChoiceValue {
    F64(f64),
    I64(i64),
    Bool(bool),
}

#[derive(Clone, Debug)]
pub struct Choice {
    pub addr: Address,
    pub value: ChoiceValue,
    pub logp: f64,
}

#[derive(Clone, Debug, Default)]
pub struct Trace {
    pub choices: BTreeMap<Address, Choice>,
    pub log_prior: f64,
    pub log_likelihood: f64,
    pub log_factors: f64,
}

impl Trace {
    pub fn total_log_weight(&self) -> f64 {
        self.log_prior + self.log_likelihood + self.log_factors
    }
}
