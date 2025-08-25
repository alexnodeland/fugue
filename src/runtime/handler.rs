#![doc = include_str!("../../docs/api/runtime/handler/README.md")]

use crate::core::address::Address;
use crate::core::distribution::Distribution;
use crate::core::model::Model;
use crate::runtime::trace::Trace;

#[doc = include_str!("../../docs/api/runtime/handler/handler.md")]
pub trait Handler {
    /// Handle an f64 sampling operation (continuous distributions).
    fn on_sample_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>) -> f64;

    /// Handle a bool sampling operation (Bernoulli).
    fn on_sample_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>) -> bool;

    /// Handle a u64 sampling operation (Poisson, Binomial).
    fn on_sample_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>) -> u64;

    /// Handle a usize sampling operation (Categorical).
    fn on_sample_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>) -> usize;

    /// Handle an f64 observation operation.
    fn on_observe_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>, value: f64);

    /// Handle a bool observation operation.
    fn on_observe_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>, value: bool);

    /// Handle a u64 observation operation.
    fn on_observe_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>, value: u64);

    /// Handle a usize observation operation.
    fn on_observe_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>, value: usize);

    /// Handle a factor operation.
    ///
    /// This method is called when the model encounters a `factor` operation.
    /// The handler typically adds the log-weight to the trace.
    ///
    /// # Arguments
    ///
    /// * `logw` - Log-weight to add to the model's total weight
    fn on_factor(&mut self, logw: f64);

    /// Finalize the handler and return the accumulated trace.
    ///
    /// This method is called after model execution completes to retrieve
    /// the final trace containing all choices and log-weights.
    fn finish(self) -> Trace
    where
        Self: Sized;
}

#[doc = include_str!("../../docs/api/runtime/handler/run.md")]
pub fn run<A>(mut h: impl Handler, m: Model<A>) -> (A, Trace) {
    fn go<A>(h: &mut impl Handler, m: Model<A>) -> A {
        match m {
            Model::Pure(a) => a,
            Model::SampleF64 { addr, dist, k } => {
                let x = h.on_sample_f64(&addr, &*dist);
                go(h, k(x))
            }
            Model::SampleBool { addr, dist, k } => {
                let x = h.on_sample_bool(&addr, &*dist);
                go(h, k(x))
            }
            Model::SampleU64 { addr, dist, k } => {
                let x = h.on_sample_u64(&addr, &*dist);
                go(h, k(x))
            }
            Model::SampleUsize { addr, dist, k } => {
                let x = h.on_sample_usize(&addr, &*dist);
                go(h, k(x))
            }
            Model::ObserveF64 {
                addr,
                dist,
                value,
                k,
            } => {
                h.on_observe_f64(&addr, &*dist, value);
                go(h, k(()))
            }
            Model::ObserveBool {
                addr,
                dist,
                value,
                k,
            } => {
                h.on_observe_bool(&addr, &*dist, value);
                go(h, k(()))
            }
            Model::ObserveU64 {
                addr,
                dist,
                value,
                k,
            } => {
                h.on_observe_u64(&addr, &*dist, value);
                go(h, k(()))
            }
            Model::ObserveUsize {
                addr,
                dist,
                value,
                k,
            } => {
                h.on_observe_usize(&addr, &*dist, value);
                go(h, k(()))
            }
            Model::Factor { logw, k } => {
                h.on_factor(logw);
                go(h, k(()))
            }
        }
    }
    let a = go(&mut h, m);
    let t = h.finish();
    (a, t)
}
