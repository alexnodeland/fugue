//! A 1-D bootstrap particle filter driven by fugue's real SMC primitives.
//!
//! The SMC explorable animates particle filtering through time on a
//! random-walk state-space model. This type keeps the exact semantics of
//! that widget but computes every statistical step with the fugue crate:
//! particles are `fugue::Particle`s, transition draws and weight updates use
//! `Normal::sample`/`log_prob`, normalization is `normalize_particles`, ESS
//! is `smc::effective_sample_size`, and resampling is fugue's
//! `systematic_resample` — the same routine `adaptive_smc` uses.

use fugue::inference::smc::{
    effective_sample_size, normalize_particles, systematic_resample, Particle,
};
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::Serialize;
use wasm_bindgen::prelude::*;

fn make_particle(x: f64, log_weight: f64) -> Particle {
    let mut trace = Trace::default();
    trace.insert_choice(addr!("x"), ChoiceValue::F64(x), 0.0);
    Particle {
        trace,
        weight: 0.0,
        log_weight,
    }
}

fn positions(particles: &[Particle]) -> Vec<f64> {
    particles
        .iter()
        .map(|p| p.trace.get_f64(&addr!("x")).unwrap_or(f64::NAN))
        .collect()
}

#[derive(Serialize)]
struct StepOut {
    /// Particle positions before propagation.
    prev: Vec<f64>,
    /// Positions after the transition draw.
    propagated: Vec<f64>,
    /// Normalized linear weights after the observation update.
    weights: Vec<f64>,
    /// ESS / N after weighting (before any resampling).
    ess_frac: f64,
    /// Log-evidence increment from this observation.
    log_z_inc: f64,
    /// Whether the adaptive threshold triggered resampling.
    resampled: bool,
    /// Ancestor index per particle (identity when not resampled).
    parents: Vec<usize>,
    /// Final positions after (possible) resampling.
    posterior: Vec<f64>,
}

/// Bootstrap particle filter: latent random walk `x_t ~ N(x_{t-1}, sig_step)`
/// observed through `y_t ~ N(x_t, sig_obs)`.
#[wasm_bindgen]
pub struct WasmParticleFilter {
    particles: Vec<Particle>,
    rng: StdRng,
    sig_step: f64,
    sig_obs: f64,
    ess_threshold: f64,
    log_z: f64,
    t: usize,
}

#[wasm_bindgen]
impl WasmParticleFilter {
    /// `n` particles from the prior `x_0 ~ N(0, prior_sig)`, seeded.
    #[wasm_bindgen(constructor)]
    pub fn new(
        n: usize,
        prior_sig: f64,
        sig_step: f64,
        sig_obs: f64,
        ess_threshold: f64,
        seed: u64,
    ) -> Result<WasmParticleFilter, JsValue> {
        let n = n.clamp(2, 5000);
        let mut rng = StdRng::seed_from_u64(seed);
        let prior = Normal::new(0.0, prior_sig.max(1e-6))
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let particles = (0..n)
            .map(|_| make_particle(prior.sample(&mut rng), 0.0))
            .collect();
        Ok(WasmParticleFilter {
            particles,
            rng,
            sig_step: sig_step.max(1e-6),
            sig_obs: sig_obs.max(1e-6),
            ess_threshold: ess_threshold.clamp(0.0, 1.0),
            log_z: 0.0,
            t: 0,
        })
    }

    /// The filter's assumed observation noise (widget slider).
    pub fn set_sig_obs(&mut self, sig_obs: f64) {
        self.sig_obs = sig_obs.max(1e-6);
    }

    /// Advance one time step with observation `obs`. Returns a JSON string
    /// with every intermediate array the animation renders (see `StepOut`).
    pub fn step(&mut self, obs: f64) -> String {
        let n = self.particles.len();
        let prev = positions(&self.particles);

        // Propagate through the transition kernel (bootstrap proposal).
        let trans_ok = Normal::new(0.0, self.sig_step).unwrap();
        for p in &mut self.particles {
            let x = p.trace.get_f64(&addr!("x")).unwrap_or(0.0);
            let x_new = x + trans_ok.sample(&mut self.rng);
            p.trace = make_particle(x_new, p.log_weight).trace;
        }
        let propagated = positions(&self.particles);

        // Weight by the observation likelihood — fugue's Normal::log_prob.
        let lik = Normal::new(obs, self.sig_obs).unwrap();
        for p in &mut self.particles {
            let x = p.trace.get_f64(&addr!("x")).unwrap_or(0.0);
            p.log_weight += lik.log_prob(&x);
        }

        // Log-evidence increment: log mean of the incremental weights. With
        // equal weights entering the step this is log_sum_exp(lw) - log N.
        let lws: Vec<f64> = self.particles.iter().map(|p| p.log_weight).collect();
        let log_z_inc = log_sum_exp(&lws) - (n as f64).ln();
        self.log_z += log_z_inc;

        normalize_particles(&mut self.particles);
        let weights: Vec<f64> = self.particles.iter().map(|p| p.weight).collect();
        let ess = effective_sample_size(&self.particles);
        let ess_frac = ess / n as f64;

        let resampled = ess_frac < self.ess_threshold;
        let parents: Vec<usize> = if resampled {
            let idx = systematic_resample(&mut self.rng, &self.particles);
            self.particles = idx
                .iter()
                .map(|&i| {
                    let x = self.particles[i].trace.get_f64(&addr!("x")).unwrap_or(0.0);
                    make_particle(x, 0.0)
                })
                .collect();
            normalize_particles(&mut self.particles);
            idx
        } else {
            // Carry normalized log-weights forward (identity ancestry). The
            // next step's evidence increment then needs weights relative to
            // equal, so store log(N·w) — standard weighted-filter bookkeeping.
            for p in &mut self.particles {
                p.log_weight = (p.weight * n as f64).max(1e-300).ln();
            }
            (0..n).collect()
        };

        self.t += 1;
        let out = StepOut {
            prev,
            propagated,
            weights,
            ess_frac,
            log_z_inc,
            resampled,
            parents,
            posterior: positions(&self.particles),
        };
        serde_json::to_string(&out).unwrap_or_else(|_| "{}".to_string())
    }

    /// Accumulated log-evidence estimate.
    pub fn log_evidence(&self) -> f64 {
        self.log_z
    }

    /// Time steps taken.
    pub fn t(&self) -> usize {
        self.t
    }
}
