//! Inference methods for fitting models to data.
//!
//! This module provides various inference methods for fitting models to data.
//! The `inference` module provides a unified interface for running inference
//! methods on models.

pub mod abc;
pub mod diagnostics;
pub mod mcmc_utils;
pub mod mh;
pub mod smc;
pub mod validation;
pub mod vi;
