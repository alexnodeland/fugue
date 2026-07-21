//! One-shot adaptive tempered SMC (fugue's `adaptive_smc`) for the
//! playground: run the full tempering ladder on a compiled model and return
//! particle values + the unbiased log-evidence estimate.

use fugue::inference::smc::{adaptive_smc, ResamplingMethod, SMCConfig};
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::Serialize;
use wasm_bindgen::prelude::*;

use crate::dsl::CompiledModel;

#[derive(Serialize)]
struct SmcOut {
    /// Per-site posterior particle values: site name -> weighted values.
    sites: Vec<String>,
    /// Row per site, matching `sites`: particle values.
    values: Vec<Vec<f64>>,
    /// Normalized particle weights (shared across sites).
    weights: Vec<f64>,
    log_evidence: f64,
    ess: f64,
    warnings: Vec<String>,
}

/// Run fugue's `adaptive_smc` on a DSL model. Returns a JSON string of
/// `{sites, values, weights, log_evidence, ess, warnings}`.
#[wasm_bindgen]
pub fn wasm_smc_run(
    source: &str,
    data_json: &str,
    n_particles: usize,
    rejuvenation_steps: usize,
    seed: u64,
) -> Result<String, JsValue> {
    let model = CompiledModel::compile(source, data_json).map_err(|e| JsValue::from_str(&e))?;
    let mut rng = StdRng::seed_from_u64(seed);
    let config = SMCConfig {
        resampling_method: ResamplingMethod::Systematic,
        ess_threshold: 0.5,
        rejuvenation_steps: rejuvenation_steps.min(20),
    };
    let result = adaptive_smc(
        &mut rng,
        n_particles.clamp(10, 5000),
        || model.build(),
        config,
    );

    // Collect the f64 sites present in the first live particle.
    let sites: Vec<Address> = result
        .particles
        .iter()
        .find(|p| !p.trace.choices.is_empty())
        .map(|p| {
            p.trace
                .choices
                .iter()
                .filter(|(_, c)| c.value.as_f64().is_some())
                .map(|(a, _)| a.clone())
                .collect()
        })
        .unwrap_or_default();

    let values: Vec<Vec<f64>> = sites
        .iter()
        .map(|a| {
            result
                .particles
                .iter()
                .map(|p| p.trace.get_f64(a).unwrap_or(f64::NAN))
                .collect()
        })
        .collect();
    let weights: Vec<f64> = result.particles.iter().map(|p| p.weight).collect();
    let ess = fugue::inference::smc::effective_sample_size(&result.particles);

    let out = SmcOut {
        sites: sites.iter().map(|a| a.to_string()).collect(),
        values,
        weights,
        log_evidence: result.log_evidence,
        ess,
        warnings: model.take_warnings(),
    };
    serde_json::to_string(&out).map_err(|e| JsValue::from_str(&e.to_string()))
}
