#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/docs/inference/README.md"))]
pub mod abc;
pub mod diagnostics;
pub mod mcmc_utils;
pub mod mh;
pub mod smc;
pub mod validation;
pub mod vi;
