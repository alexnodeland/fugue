//! Incremental HMC for the browser, over fugue's `HmcSession`.
//!
//! One `step_recorded()` per widget transition returns the full leapfrog
//! trajectory (positions + Hamiltonians) so the renderer can animate the
//! "rolling" proposal exactly as fugue computed it.

use fugue::inference::hmc::{HMCConfig, HmcSession};
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::Serialize;
use wasm_bindgen::prelude::*;

use crate::dsl::CompiledModel;

#[derive(Serialize)]
struct StepOut {
    accepted: bool,
    divergent: bool,
    accept_prob: f64,
    step_size: f64,
    /// Row-major [n_points][dim] leapfrog positions (start included).
    trajectory_q: Vec<Vec<f64>>,
    /// Hamiltonian at each trajectory point.
    trajectory_h: Vec<f64>,
    /// Position after the transition.
    position: Vec<f64>,
    log_weight: f64,
}

/// A single incremental HMC chain over one compiled model.
#[wasm_bindgen]
pub struct WasmHmc {
    model: CompiledModel,
    session: HmcSession<f64>,
    rng: StdRng,
    history: Vec<Trace>,
}

#[wasm_bindgen]
impl WasmHmc {
    /// Compile `source` against `data_json` and initialize a chain.
    /// `step_size <= 0` lets fugue's reasonable-epsilon heuristic pick one;
    /// `n_warmup > 0` enables dual-averaging warmup for that many steps.
    #[wasm_bindgen(constructor)]
    pub fn new(
        source: &str,
        data_json: &str,
        seed: u64,
        n_warmup: usize,
        n_leapfrog: usize,
        step_size: f64,
    ) -> Result<WasmHmc, JsValue> {
        let model = CompiledModel::compile(source, data_json).map_err(|e| JsValue::from_str(&e))?;
        let config = HMCConfig {
            n_leapfrog: n_leapfrog.clamp(1, 200),
            init_step_size: if step_size > 0.0 {
                Some(step_size)
            } else {
                None
            },
            ..HMCConfig::default()
        };
        let mut rng = StdRng::seed_from_u64(seed);
        let session = HmcSession::new(&mut rng, &(|| model.build()), n_warmup, config);
        Ok(WasmHmc {
            model,
            session,
            rng,
            history: Vec::new(),
        })
    }

    /// The continuous site addresses the chain moves (coordinate order of
    /// every position/trajectory array).
    pub fn site_names(&self) -> Vec<String> {
        self.session
            .sites()
            .iter()
            .map(|a| a.to_string())
            .collect()
    }

    /// One transition WITH trajectory recording; returns a JSON string of
    /// `{accepted, divergent, accept_prob, step_size, trajectory_q,
    /// trajectory_h, position, log_weight}`.
    pub fn step_recorded(&mut self) -> String {
        let model = &self.model;
        let info = self
            .session
            .step_recorded(&mut self.rng, &(|| model.build()));
        self.push_history();
        let out = StepOut {
            accepted: info.accepted,
            divergent: info.divergent,
            accept_prob: info.accept_prob,
            step_size: info.step_size,
            trajectory_q: info.trajectory.iter().map(|p| p.q.clone()).collect(),
            trajectory_h: info.trajectory.iter().map(|p| p.h).collect(),
            position: self.session.position().to_vec(),
            log_weight: self.session.trace().total_log_weight(),
        };
        serde_json::to_string(&out).unwrap_or_else(|_| "{}".to_string())
    }

    /// Advance `n` transitions without recording (warmup, fast-forward).
    /// Returns how many were accepted.
    pub fn step(&mut self, n: usize) -> usize {
        let mut accepted = 0;
        for _ in 0..n {
            let model = &self.model;
            let info = self.session.step(&mut self.rng, &(|| model.build()));
            if info.accepted {
                accepted += 1;
            }
            self.push_history();
        }
        accepted
    }

    /// Whether the next step still adapts the step size.
    pub fn is_warming_up(&self) -> bool {
        self.session.is_warming_up()
    }

    /// The step size the next transition will use.
    pub fn step_size(&self) -> f64 {
        self.session.step_size()
    }

    /// History of one coordinate (by site name) across recorded transitions.
    pub fn values(&self, site: &str) -> Vec<f64> {
        let addr = Address::new(site.to_string());
        self.history
            .iter()
            .filter_map(|t| t.get_f64(&addr))
            .collect()
    }

    /// Single-chain ESS of a coordinate over the recorded history, via
    /// fugue's `effective_sample_size_mcmc`.
    pub fn ess(&self, site: &str) -> f64 {
        effective_sample_size_mcmc(&self.values(site))
    }

    /// Drain runtime model warnings.
    pub fn warnings(&self) -> Vec<String> {
        self.model.take_warnings()
    }
}

impl WasmHmc {
    fn push_history(&mut self) {
        if self.history.len() >= 20_000 {
            self.history.drain(0..10_000);
        }
        self.history.push(self.session.trace().clone());
    }
}
