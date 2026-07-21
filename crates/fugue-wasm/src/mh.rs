//! Incremental multi-chain adaptive Metropolis-Hastings for the browser.
//!
//! Each chain is fugue's real `adaptive_single_site_mh` transition driven one
//! call at a time, with its own seeded `StdRng` and `DiminishingAdaptation`
//! state. Histories are kept as full `Trace`s so R̂/ESS come from fugue's own
//! `diagnostics` module, not a re-implementation.

use fugue::inference::mh::adaptive_single_site_mh;
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use wasm_bindgen::prelude::*;

use crate::dsl::CompiledModel;

struct Chain {
    rng: StdRng,
    trace: Trace,
    adapt: DiminishingAdaptation,
    history: Vec<Trace>,
    steps: usize,
}

/// A set of independent adaptive-MH chains over one compiled model.
#[wasm_bindgen]
pub struct WasmMh {
    model: CompiledModel,
    chains: Vec<Chain>,
    // Cap on stored history per chain, so an always-running widget cannot
    // grow memory without bound. Oldest halves are dropped in blocks.
    max_history: usize,
    dropped: usize,
}

#[wasm_bindgen]
impl WasmMh {
    /// Compile `source` (the `prob!`-subset DSL) against `data_json` and
    /// initialize `n_chains` chains from prior draws, seeded from `seed`
    /// (chain c uses `seed + c`, so chains are independent but replayable).
    #[wasm_bindgen(constructor)]
    pub fn new(
        source: &str,
        data_json: &str,
        n_chains: usize,
        seed: u64,
    ) -> Result<WasmMh, JsValue> {
        let model = CompiledModel::compile(source, data_json).map_err(|e| JsValue::from_str(&e))?;
        let n = n_chains.clamp(1, 16);
        let chains = (0..n)
            .map(|c| {
                let mut rng = StdRng::seed_from_u64(seed.wrapping_add(c as u64));
                let (_, trace) = fugue::runtime::handler::run(
                    PriorHandler {
                        rng: &mut rng,
                        trace: Trace::default(),
                    },
                    model.build(),
                );
                Chain {
                    rng,
                    trace,
                    adapt: DiminishingAdaptation::new(0.44, 0.7),
                    history: Vec::new(),
                    steps: 0,
                }
            })
            .collect();
        Ok(WasmMh {
            model,
            chains,
            max_history: 20_000,
            dropped: 0,
        })
    }

    /// Advance every chain by `n` transitions. Returns the total number of
    /// transitions each chain has taken so far.
    pub fn step(&mut self, n: usize) -> usize {
        for chain in &mut self.chains {
            for _ in 0..n {
                let (_, next) = adaptive_single_site_mh(
                    &mut chain.rng,
                    || self.model.build(),
                    &chain.trace,
                    &mut chain.adapt,
                );
                chain.trace = next;
                chain.steps += 1;
                chain.history.push(chain.trace.clone());
            }
        }
        if self.chains[0].history.len() > self.max_history {
            let drop = self.max_history / 2;
            for chain in &mut self.chains {
                chain.history.drain(0..drop);
            }
            self.dropped += drop;
        }
        self.chains[0].steps
    }

    /// Number of chains.
    pub fn n_chains(&self) -> usize {
        self.chains.len()
    }

    /// Total transitions taken per chain.
    pub fn total_steps(&self) -> usize {
        self.chains[0].steps
    }

    /// The `f64` site addresses of the model (from chain 0's current trace),
    /// in trace order.
    pub fn site_names(&self) -> Vec<String> {
        self.chains[0]
            .trace
            .choices
            .iter()
            .filter(|(_, c)| c.value.as_f64().is_some())
            .map(|(a, _)| a.to_string())
            .collect()
    }

    /// History of a site's values for one chain, from `start` (a step index
    /// as previously returned by `step`/`total_steps`) onward. Lets the
    /// renderer pull only the new draws each frame.
    pub fn values_since(&self, site: &str, chain: usize, start: usize) -> Vec<f64> {
        let addr = Address::new(site.to_string());
        let Some(chain) = self.chains.get(chain) else {
            return Vec::new();
        };
        let from = start.saturating_sub(self.dropped);
        chain
            .history
            .iter()
            .skip(from)
            .filter_map(|t| t.get_f64(&addr))
            .collect()
    }

    /// Current value of a site in each chain (one entry per chain; NaN when
    /// the site is missing or not `f64`).
    pub fn current_values(&self, site: &str) -> Vec<f64> {
        let addr = Address::new(site.to_string());
        self.chains
            .iter()
            .map(|c| c.trace.get_f64(&addr).unwrap_or(f64::NAN))
            .collect()
    }

    /// Current total log-weight of each chain.
    pub fn log_weights(&self) -> Vec<f64> {
        self.chains
            .iter()
            .map(|c| c.trace.total_log_weight())
            .collect()
    }

    /// Split-R̂ (Vehtari et al. 2021) across chains for a site, over the last
    /// `window` retained draws (0 = all retained history). Uses fugue's
    /// `r_hat_f64`.
    pub fn r_hat(&self, site: &str, window: usize) -> f64 {
        let addr = Address::new(site.to_string());
        let chains: Vec<Vec<Trace>> = self
            .chains
            .iter()
            .map(|c| {
                let h = &c.history;
                let from = if window > 0 && h.len() > window {
                    h.len() - window
                } else {
                    0
                };
                h[from..].to_vec()
            })
            .collect();
        r_hat_f64(&chains, &addr)
    }

    /// Multi-chain effective sample size for a site over all retained draws,
    /// via fugue's `effective_sample_size_multichain`.
    pub fn ess(&self, site: &str) -> f64 {
        let addr = Address::new(site.to_string());
        let values: Vec<Vec<f64>> = self
            .chains
            .iter()
            .map(|c| c.history.iter().filter_map(|t| t.get_f64(&addr)).collect())
            .collect();
        effective_sample_size_multichain(&values)
    }

    /// Overall acceptance rate across chains, from the adaptation's exact
    /// per-address accept/total counts.
    pub fn acceptance_rate(&self) -> f64 {
        let (acc, tot) = self.chains.iter().fold((0usize, 0usize), |(a, t), c| {
            (
                a + c.adapt.accept_counts.values().sum::<usize>(),
                t + c.adapt.total_counts.values().sum::<usize>(),
            )
        });
        if tot == 0 {
            f64::NAN
        } else {
            acc as f64 / tot as f64
        }
    }

    /// Posterior summary of a site over retained draws:
    /// `[mean, std, r_hat, ess]` from fugue's `summarize_f64_parameter`.
    pub fn summary(&self, site: &str) -> Vec<f64> {
        let addr = Address::new(site.to_string());
        let chains: Vec<Vec<Trace>> = self.chains.iter().map(|c| c.history.clone()).collect();
        let s = summarize_f64_parameter(&chains, &addr);
        vec![s.mean, s.std, s.r_hat, s.ess]
    }

    /// Drain runtime model warnings (soft errors mapped to `-inf` regions).
    pub fn warnings(&self) -> Vec<String> {
        self.model.take_warnings()
    }
}
