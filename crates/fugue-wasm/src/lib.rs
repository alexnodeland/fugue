//! WebAssembly bindings for fugue: real models, real handlers, real inference
//! in the browser.
//!
//! The docs' interactive widgets and playground drive these types one
//! animation frame at a time: construct a sampler with a model source (the
//! `prob!`-subset DSL in [`dsl`]), a JSON data payload, and a seed, then call
//! `step(n)` per frame and read back draws/diagnostics as typed arrays.
//! Everything statistical — distributions, trace scoring, MH/HMC transitions,
//! SMC resampling, R̂/ESS — executes the actual `fugue` crate compiled to
//! wasm, so the docs cannot drift from the library.
//!
//! All samplers are explicitly seeded (`StdRng::seed_from_u64`); no browser
//! entropy is consumed, and a seed is a replayable recording — the same
//! promise a fugue `Trace` makes.

mod dsl;
mod hmc;
mod mh;
mod pf;
mod smc;

pub use dsl::CompiledModel;
pub use hmc::WasmHmc;
pub use mh::WasmMh;
pub use pf::WasmParticleFilter;
pub use smc::wasm_smc_run;

use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

/// The version of the fugue-wasm bindings package.
#[wasm_bindgen]
pub fn fugue_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Validate a model source + data payload without running anything. Returns
/// the empty string when the model compiles, otherwise the compile error
/// (with a line number where applicable). Used for live editor feedback.
#[wasm_bindgen]
pub fn check_model(source: &str, data_json: &str) -> String {
    match dsl::CompiledModel::compile(source, data_json) {
        Ok(_) => String::new(),
        Err(e) => e,
    }
}
