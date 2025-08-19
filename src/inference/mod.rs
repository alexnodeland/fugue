//! Inference methods for fitting models to data.
//!
//! This module provides various inference methods for fitting models to data.
//! The `inference` module provides a unified interface for running inference
//! methods on models.

pub mod mh;
pub mod smc;
pub mod vi;
