//! Log-joint evaluation over a 2-D parameter grid, for posterior heatmaps.
//!
//! The widgets draw their posterior-density backgrounds from this: every
//! grid point is scored by running the real model under `ScoreGivenTrace`
//! with the two chosen sites pinned — the same evaluation MH/HMC use for
//! their accept ratios, so the heat is exactly the surface the samplers walk.

use fugue::runtime::interpreters::PriorHandler;
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use wasm_bindgen::prelude::*;

use crate::dsl::CompiledModel;

/// Evaluate the unnormalized log-joint on the grid `a_values x b_values`
/// for the two `f64` sites `site_a`/`site_b`. Any other latent sites are
/// held at a fixed prior draw (seeded, so the surface is reproducible).
/// Returns row-major values: `out[j * a_values.len() + i]` scores
/// `(a_values[i], b_values[j])`.
#[wasm_bindgen]
pub fn log_joint_grid(
    source: &str,
    data_json: &str,
    site_a: &str,
    site_b: &str,
    a_values: &[f64],
    b_values: &[f64],
    seed: u64,
) -> Result<Vec<f64>, JsValue> {
    let model = CompiledModel::compile(source, data_json).map_err(|e| JsValue::from_str(&e))?;
    let mut rng = StdRng::seed_from_u64(seed);
    let (_, base) = fugue::runtime::handler::run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        model.build(),
    );
    let addr_a = Address::new(site_a.to_string());
    let addr_b = Address::new(site_b.to_string());
    if base.get_f64(&addr_a).is_none() || base.get_f64(&addr_b).is_none() {
        return Err(JsValue::from_str(&format!(
            "sites `{site_a}`/`{site_b}` must be f64 sample sites of the model"
        )));
    }

    let mut out = Vec::with_capacity(a_values.len() * b_values.len());
    for &b in b_values {
        for &a in a_values {
            let mut t = base.clone();
            if let Some(c) = t.choices.get_mut(&addr_a) {
                c.value = ChoiceValue::F64(a);
            }
            if let Some(c) = t.choices.get_mut(&addr_b) {
                c.value = ChoiceValue::F64(b);
            }
            let (_, scored) = fugue::runtime::handler::run(
                ScoreGivenTrace {
                    base: t,
                    trace: Trace::default(),
                },
                model.build(),
            );
            out.push(scored.total_log_weight());
        }
    }
    Ok(out)
}
