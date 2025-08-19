use rand::Rng;
use rand_distr::{Distribution as RandDistr, Normal as RDNormal};
pub type LogF64 = f64;
#[derive(Clone, Copy, Debug)]
pub struct Normal { pub mu: f64, pub sigma: f64 }
impl Normal {
  pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 { RDNormal::new(self.mu, self.sigma).unwrap().sample(rng) }
  pub fn log_prob(&self, x: f64) -> LogF64 {
    let z = (x - self.mu) / self.sigma;
    -0.5*z*z - self.sigma.ln() - 0.5*(2.0*std::f64::consts::PI).ln()
  }
}
