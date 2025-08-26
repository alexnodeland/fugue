#![doc = include_str!("../../docs/api/core/distribution/README.md")]
use rand::{Rng, RngCore};
use rand_distr::{
    Beta as RDBeta, Binomial as RDBinomial, Distribution as RandDistr, Exp as RDExp,
    Gamma as RDGamma, LogNormal as RDLogNormal, Normal as RDNormal, Poisson as RDPoisson,
};
/// Type alias for log-probabilities.
///
/// Log-probabilities are represented as `f64` values. Negative infinity represents
/// zero probability, while finite values represent the natural logarithm of probabilities.
pub type LogF64 = f64;

#[doc = include_str!("../../docs/api/core/distribution/distribution.md")]
pub trait Distribution<T>: Send + Sync {
    #[doc = include_str!("../../docs/api/core/distribution/sample.md")]
    fn sample(&self, rng: &mut dyn RngCore) -> T;

    #[doc = include_str!("../../docs/api/core/distribution/log_prob.md")]
    fn log_prob(&self, x: &T) -> LogF64;

    #[doc = include_str!("../../docs/api/core/distribution/clone_box.md")]
    fn clone_box(&self) -> Box<dyn Distribution<T>>;
}

#[doc = include_str!("../../docs/api/core/distribution/distributions/normal.md")]
#[derive(Clone, Copy, Debug)]
pub struct Normal {
    /// Mean of the normal distribution.
    mu: f64,
    /// Standard deviation of the normal distribution (must be positive).
    sigma: f64,
}
impl Normal {
    /// Create a new Normal distribution with validated parameters.
    pub fn new(mu: f64, sigma: f64) -> crate::error::FugueResult<Self> {
        if !mu.is_finite() {
            return Err(crate::error::FugueError::invalid_parameters(
                "Normal", 
                "Mean (mu) must be finite", 
                crate::error::ErrorCode::InvalidMean
            ).with_context("mu", format!("{}", mu)));
        }
        if sigma <= 0.0 || !sigma.is_finite() {
            return Err(crate::error::FugueError::invalid_parameters(
                "Normal", 
                "Standard deviation (sigma) must be positive and finite", 
                crate::error::ErrorCode::InvalidVariance
            ).with_context("sigma", format!("{}", sigma))
             .with_context("expected", "> 0.0 and finite"));
        }
        Ok(Normal { mu, sigma })
    }

    /// Get the mean of the distribution.
    pub fn mu(&self) -> f64 {
        self.mu
    }

    /// Get the standard deviation of the distribution.
    pub fn sigma(&self) -> f64 {
        self.sigma
    }
}
impl Distribution<f64> for Normal {
    fn sample(&self, rng: &mut dyn RngCore) -> f64 {
        if self.sigma <= 0.0 {
            return f64::NAN;
        }
        RDNormal::new(self.mu, self.sigma).unwrap().sample(rng)
    }
    fn log_prob(&self, x: &f64) -> LogF64 {
        // Parameter validation
        if self.sigma <= 0.0 || !self.sigma.is_finite() || !self.mu.is_finite() || !x.is_finite() {
            return f64::NEG_INFINITY;
        }

        // Numerically stable computation
        let z = (x - self.mu) / self.sigma;

        // Prevent overflow for extreme values (|z| > 37 gives exp(-z²/2) < machine epsilon)
        if z.abs() > 37.0 {
            return f64::NEG_INFINITY;
        }

        // Use precomputed constant for better precision
        const LN_2PI: f64 = 1.837_877_066_409_345_6; // ln(2π)
        -0.5 * z * z - self.sigma.ln() - 0.5 * LN_2PI
    }
    fn clone_box(&self) -> Box<dyn Distribution<f64>> {
        Box::new(*self)
    }
}

#[doc = include_str!("../../docs/api/core/distribution/distributions/uniform.md")]
#[derive(Clone, Copy, Debug)]
pub struct Uniform {
    /// Lower bound of the uniform distribution (inclusive).
    low: f64,
    /// Upper bound of the uniform distribution (exclusive).
    high: f64,
}
impl Uniform {
    /// Create a new Uniform distribution with validated parameters.
    pub fn new(low: f64, high: f64) -> crate::error::FugueResult<Self> {
        if !low.is_finite() || !high.is_finite() {
            return Err(crate::error::FugueError::invalid_parameters(
                "Uniform", 
                "Bounds must be finite", 
                crate::error::ErrorCode::InvalidRange
            ).with_context("low", format!("{}", low))
             .with_context("high", format!("{}", high)));
        }
        if low >= high {
            return Err(crate::error::FugueError::invalid_parameters(
                "Uniform", 
                "Lower bound must be less than upper bound", 
                crate::error::ErrorCode::InvalidRange
            ).with_context("low", format!("{}", low))
             .with_context("high", format!("{}", high)));
        }
        Ok(Uniform { low, high })
    }

    /// Get the lower bound.
    pub fn low(&self) -> f64 {
        self.low
    }

    /// Get the upper bound.
    pub fn high(&self) -> f64 {
        self.high
    }
}
impl Distribution<f64> for Uniform {
    fn sample(&self, rng: &mut dyn RngCore) -> f64 {
        // Parameter validation
        if self.low >= self.high || !self.low.is_finite() || !self.high.is_finite() {
            return f64::NAN;
        }
        Rng::gen_range(rng, self.low..self.high)
    }
    fn log_prob(&self, x: &f64) -> LogF64 {
        // Parameter validation
        if self.low >= self.high
            || !self.low.is_finite()
            || !self.high.is_finite()
            || !x.is_finite()
        {
            return f64::NEG_INFINITY;
        }

        // Check support with proper boundary handling
        if *x < self.low || *x >= self.high {
            f64::NEG_INFINITY
        } else {
            let width = self.high - self.low;
            if width <= 0.0 {
                f64::NEG_INFINITY
            } else {
                -width.ln()
            }
        }
    }
    fn clone_box(&self) -> Box<dyn Distribution<f64>> {
        Box::new(*self)
    }
}

#[doc = include_str!("../../docs/api/core/distribution/distributions/lognormal.md")]
#[derive(Clone, Copy, Debug)]
pub struct LogNormal {
    /// Mean of the underlying normal distribution.
    mu: f64,
    /// Standard deviation of the underlying normal distribution (must be positive).
    sigma: f64,
}
impl LogNormal {
    /// Create a new LogNormal distribution with validated parameters.
    pub fn new(mu: f64, sigma: f64) -> crate::error::FugueResult<Self> {
        if !mu.is_finite() {
            return Err(crate::error::FugueError::invalid_parameters(
                "LogNormal", 
                "Mean (mu) must be finite", 
                crate::error::ErrorCode::InvalidMean
            ).with_context("mu", format!("{}", mu)));
        }
        if sigma <= 0.0 || !sigma.is_finite() {
            return Err(crate::error::FugueError::invalid_parameters(
                "LogNormal", 
                "Standard deviation (sigma) must be positive and finite", 
                crate::error::ErrorCode::InvalidVariance
            ).with_context("sigma", format!("{}", sigma))
             .with_context("expected", "> 0.0 and finite"));
        }
        Ok(LogNormal { mu, sigma })
    }

    /// Get the mean of the underlying normal distribution.
    pub fn mu(&self) -> f64 {
        self.mu
    }

    /// Get the standard deviation of the underlying normal distribution.
    pub fn sigma(&self) -> f64 {
        self.sigma
    }
}
impl Distribution<f64> for LogNormal {
    fn sample(&self, rng: &mut dyn RngCore) -> f64 {
        if self.sigma <= 0.0 {
            return f64::NAN;
        }
        RDLogNormal::new(self.mu, self.sigma).unwrap().sample(rng)
    }
    fn log_prob(&self, x: &f64) -> LogF64 {
        // Parameter and input validation
        if self.sigma <= 0.0 || !self.sigma.is_finite() || !self.mu.is_finite() {
            return f64::NEG_INFINITY;
        }
        if *x <= 0.0 || !x.is_finite() {
            return f64::NEG_INFINITY;
        }

        // Numerically stable computation
        let lx = x.ln();
        let z = (lx - self.mu) / self.sigma;

        // Prevent overflow
        if z.abs() > 37.0 {
            return f64::NEG_INFINITY;
        }

        // Stable computation: log_prob = -0.5*z² - ln(x) - ln(σ) - 0.5*ln(2π)
        const LN_2PI: f64 = 1.837_877_066_409_345_6; // ln(2π)
        -0.5 * z * z - lx - self.sigma.ln() - 0.5 * LN_2PI
    }
    fn clone_box(&self) -> Box<dyn Distribution<f64>> {
        Box::new(*self)
    }
}

#[doc = include_str!("../../docs/api/core/distribution/distributions/exponential.md")]
#[derive(Clone, Copy, Debug)]
pub struct Exponential {
    /// Rate parameter λ of the exponential distribution (must be positive).
    rate: f64,
}
impl Exponential {
    /// Create a new Exponential distribution with validated parameters.
    pub fn new(rate: f64) -> crate::error::FugueResult<Self> {
        if rate <= 0.0 || !rate.is_finite() {
            return Err(crate::error::FugueError::invalid_parameters(
                "Exponential", 
                "Rate parameter must be positive and finite", 
                crate::error::ErrorCode::InvalidRate
            ).with_context("rate", format!("{}", rate))
             .with_context("expected", "> 0.0 and finite"));
        }
        Ok(Exponential { rate })
    }

    /// Get the rate parameter.
    pub fn rate(&self) -> f64 {
        self.rate
    }
}
impl Distribution<f64> for Exponential {
    fn sample(&self, rng: &mut dyn RngCore) -> f64 {
        if self.rate <= 0.0 {
            return f64::NAN;
        }
        RDExp::new(self.rate).unwrap().sample(rng)
    }
    fn log_prob(&self, x: &f64) -> LogF64 {
        // Parameter validation
        if self.rate <= 0.0 || !self.rate.is_finite() || !x.is_finite() {
            return f64::NEG_INFINITY;
        }

        if *x < 0.0 {
            f64::NEG_INFINITY
        } else {
            // Check for overflow: if rate * x > 700, exp(-rate*x) underflows
            if self.rate * x > 700.0 {
                return f64::NEG_INFINITY;
            }
            self.rate.ln() - self.rate * x
        }
    }
    fn clone_box(&self) -> Box<dyn Distribution<f64>> {
        Box::new(*self)
    }
}

#[doc = include_str!("../../docs/api/core/distribution/distributions/bernoulli.md")]
#[derive(Clone, Copy, Debug)]
pub struct Bernoulli {
    /// Probability of success (must be in [0, 1]).
    p: f64,
}
impl Bernoulli {
    /// Create a new Bernoulli distribution with validated parameters.
    pub fn new(p: f64) -> crate::error::FugueResult<Self> {
        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return Err(crate::error::FugueError::invalid_parameters(
                "Bernoulli", 
                "Probability must be in [0, 1]", 
                crate::error::ErrorCode::InvalidProbability
            ).with_context("p", format!("{}", p))
             .with_context("expected", "[0.0, 1.0]"));
        }
        Ok(Bernoulli { p })
    }

    /// Get the success probability.
    pub fn p(&self) -> f64 {
        self.p
    }
}
impl Distribution<bool> for Bernoulli {
    fn sample(&self, rng: &mut dyn RngCore) -> bool {
        if self.p < 0.0 || self.p > 1.0 || !self.p.is_finite() {
            return false; // Default to false for invalid parameters
        }
        use rand::Rng;
        rng.gen::<f64>() < self.p
    }
    fn log_prob(&self, x: &bool) -> LogF64 {
        // Parameter validation
        if self.p < 0.0 || self.p > 1.0 || !self.p.is_finite() {
            return f64::NEG_INFINITY;
        }

        if *x {
            // P(X = true) = p
            if self.p <= 0.0 {
                f64::NEG_INFINITY
            } else {
                self.p.ln()
            }
        } else {
            // P(X = false) = 1 - p
            if self.p >= 1.0 {
                f64::NEG_INFINITY
            } else {
                (1.0 - self.p).ln()
            }
        }
    }
    fn clone_box(&self) -> Box<dyn Distribution<bool>> {
        Box::new(*self)
    }
}

#[doc = include_str!("../../docs/api/core/distribution/distributions/categorical.md")]
#[derive(Clone, Debug)]
pub struct Categorical {
    /// Probabilities for each category (should sum to 1.0).
    probs: Vec<f64>,
}
impl Categorical {
    /// Create a new Categorical distribution with validated parameters.
    pub fn new(probs: Vec<f64>) -> crate::error::FugueResult<Self> {
        if probs.is_empty() {
            return Err(crate::error::FugueError::invalid_parameters(
                "Categorical", 
                "Probability vector cannot be empty", 
                crate::error::ErrorCode::InvalidProbability
            ).with_context("length", "0"));
        }

        let sum: f64 = probs.iter().sum();
        if (sum - 1.0).abs() > 1e-6 {
            return Err(crate::error::FugueError::invalid_parameters(
                "Categorical", 
                "Probabilities must sum to 1.0", 
                crate::error::ErrorCode::InvalidProbability
            ).with_context("sum", format!("{:.6}", sum))
             .with_context("expected", "1.0")
             .with_context("tolerance", "1e-6"));
        }

        for (i, &p) in probs.iter().enumerate() {
            if !p.is_finite() || p < 0.0 {
                return Err(crate::error::FugueError::invalid_parameters(
                    "Categorical", 
                    "All probabilities must be non-negative and finite", 
                    crate::error::ErrorCode::InvalidProbability
                ).with_context("index", format!("{}", i))
                 .with_context("value", format!("{}", p))
                 .with_context("expected", ">= 0.0 and finite"));
            }
        }

        Ok(Categorical { probs })
    }

    /// Create a uniform categorical distribution over k categories.
    pub fn uniform(k: usize) -> crate::error::FugueResult<Self> {
        if k == 0 {
            return Err(crate::error::FugueError::invalid_parameters(
                "Categorical", 
                "Number of categories must be positive", 
                crate::error::ErrorCode::InvalidCount
            ).with_context("k", "0"));
        }

        let prob = 1.0 / k as f64;
        let probs = vec![prob; k];
        Ok(Categorical { probs })
    }

    /// Get the probability vector.
    pub fn probs(&self) -> &[f64] {
        &self.probs
    }

    /// Get the number of categories.
    pub fn len(&self) -> usize {
        self.probs.len()
    }

    /// Check if the distribution has no categories.
    pub fn is_empty(&self) -> bool {
        self.probs.is_empty()
    }
}
impl Distribution<usize> for Categorical {
    fn sample(&self, rng: &mut dyn RngCore) -> usize {
        // Parameter validation
        if self.probs.is_empty() {
            return 0;
        }

        let prob_sum: f64 = self.probs.iter().sum();
        if (prob_sum - 1.0).abs() > 1e-6 || self.probs.iter().any(|&p| p < 0.0 || !p.is_finite()) {
            return 0;
        }

        use rand::Rng;
        let u: f64 = rng.gen();
        let mut cum = 0.0;
        for (i, &p) in self.probs.iter().enumerate() {
            cum += p;
            if u <= cum {
                return i;
            }
        }
        self.probs.len() - 1
    }
    fn log_prob(&self, x: &usize) -> LogF64 {
        // Parameter validation
        if self.probs.is_empty() || *x >= self.probs.len() {
            return f64::NEG_INFINITY;
        }

        let prob_sum: f64 = self.probs.iter().sum();
        if (prob_sum - 1.0).abs() > 1e-6 || self.probs.iter().any(|&p| p < 0.0 || !p.is_finite()) {
            return f64::NEG_INFINITY;
        }

        if self.probs[*x] <= 0.0 {
            f64::NEG_INFINITY
        } else {
            self.probs[*x].ln()
        }
    }
    fn clone_box(&self) -> Box<dyn Distribution<usize>> {
        Box::new(self.clone())
    }
}

#[doc = include_str!("../../docs/api/core/distribution/distributions/beta.md")]
#[derive(Clone, Copy, Debug)]
pub struct Beta {
    /// First shape parameter α (must be positive).
    alpha: f64,
    /// Second shape parameter β (must be positive).
    beta: f64,
}
impl Beta {
    /// Create a new Beta distribution with validated parameters.
    pub fn new(alpha: f64, beta: f64) -> crate::error::FugueResult<Self> {
        if alpha <= 0.0 || !alpha.is_finite() {
            return Err(crate::error::FugueError::invalid_parameters(
                "Beta", 
                "Alpha parameter must be positive and finite", 
                crate::error::ErrorCode::InvalidShape
            ).with_context("alpha", format!("{}", alpha))
             .with_context("expected", "> 0.0 and finite"));
        }
        if beta <= 0.0 || !beta.is_finite() {
            return Err(crate::error::FugueError::invalid_parameters(
                "Beta", 
                "Beta parameter must be positive and finite", 
                crate::error::ErrorCode::InvalidShape
            ).with_context("beta", format!("{}", beta))
             .with_context("expected", "> 0.0 and finite"));
        }
        Ok(Beta { alpha, beta })
    }

    /// Get the alpha parameter.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Get the beta parameter.
    pub fn beta(&self) -> f64 {
        self.beta
    }
}
impl Distribution<f64> for Beta {
    fn sample(&self, rng: &mut dyn RngCore) -> f64 {
        if self.alpha <= 0.0 || self.beta <= 0.0 {
            return f64::NAN;
        }
        RDBeta::new(self.alpha, self.beta).unwrap().sample(rng)
    }
    fn log_prob(&self, x: &f64) -> LogF64 {
        // Parameter validation
        if self.alpha <= 0.0
            || self.beta <= 0.0
            || !self.alpha.is_finite()
            || !self.beta.is_finite()
            || !x.is_finite()
        {
            return f64::NEG_INFINITY;
        }

        // Support validation
        if *x <= 0.0 || *x >= 1.0 {
            return f64::NEG_INFINITY;
        }

        // Handle edge cases near boundaries
        if *x < 1e-100 || *x > 1.0 - 1e-100 {
            return f64::NEG_INFINITY;
        }

        // Numerically stable computation using log-gamma
        // log Beta(x; α, β) = (α-1)ln(x) + (β-1)ln(1-x) - log B(α,β)
        let log_beta_fn = libm::lgamma(self.alpha) + libm::lgamma(self.beta)
            - libm::lgamma(self.alpha + self.beta);

        let ln_x = x.ln();
        let ln_1_minus_x = (1.0 - x).ln();

        // Check for extreme log values
        if ln_x < -700.0 || ln_1_minus_x < -700.0 {
            return f64::NEG_INFINITY;
        }

        (self.alpha - 1.0) * ln_x + (self.beta - 1.0) * ln_1_minus_x - log_beta_fn
    }
    fn clone_box(&self) -> Box<dyn Distribution<f64>> {
        Box::new(*self)
    }
}

#[doc = include_str!("../../docs/api/core/distribution/distributions/gamma.md")]
#[derive(Clone, Copy, Debug)]
pub struct Gamma {
    /// Shape parameter k (must be positive).
    shape: f64,
    /// Rate parameter λ (must be positive).
    rate: f64,
}
impl Gamma {
    /// Create a new Gamma distribution with validated parameters.
    pub fn new(shape: f64, rate: f64) -> crate::error::FugueResult<Self> {
        if shape <= 0.0 || !shape.is_finite() {
            return Err(crate::error::FugueError::invalid_parameters(
                "Gamma", 
                "Shape parameter must be positive and finite", 
                crate::error::ErrorCode::InvalidShape
            ).with_context("shape", format!("{}", shape))
             .with_context("expected", "> 0.0 and finite"));
        }
        if rate <= 0.0 || !rate.is_finite() {
            return Err(crate::error::FugueError::invalid_parameters(
                "Gamma", 
                "Rate parameter must be positive and finite", 
                crate::error::ErrorCode::InvalidRate
            ).with_context("rate", format!("{}", rate))
             .with_context("expected", "> 0.0 and finite"));
        }
        Ok(Gamma { shape, rate })
    }

    /// Get the shape parameter.
    pub fn shape(&self) -> f64 {
        self.shape
    }

    /// Get the rate parameter.
    pub fn rate(&self) -> f64 {
        self.rate
    }
}
impl Distribution<f64> for Gamma {
    fn sample(&self, rng: &mut dyn RngCore) -> f64 {
        if self.shape <= 0.0 || self.rate <= 0.0 {
            return f64::NAN;
        }
        RDGamma::new(self.shape, 1.0 / self.rate)
            .unwrap()
            .sample(rng)
    }
    fn log_prob(&self, x: &f64) -> LogF64 {
        // Parameter validation
        if self.shape <= 0.0
            || self.rate <= 0.0
            || !self.shape.is_finite()
            || !self.rate.is_finite()
            || !x.is_finite()
        {
            return f64::NEG_INFINITY;
        }

        if *x <= 0.0 {
            return f64::NEG_INFINITY;
        }

        // Check for overflow conditions
        if self.rate * x > 700.0 || x.ln() * (self.shape - 1.0) < -700.0 {
            return f64::NEG_INFINITY;
        }

        // Numerically stable computation
        // log Gamma(x; k, λ) = k*ln(λ) + (k-1)*ln(x) - λ*x - ln Γ(k)
        let log_rate = self.rate.ln();
        let log_x = x.ln();
        let log_gamma_shape = libm::lgamma(self.shape);

        self.shape * log_rate + (self.shape - 1.0) * log_x - self.rate * x - log_gamma_shape
    }
    fn clone_box(&self) -> Box<dyn Distribution<f64>> {
        Box::new(*self)
    }
}

#[doc = include_str!("../../docs/api/core/distribution/distributions/binomial.md")]
#[derive(Clone, Copy, Debug)]
pub struct Binomial {
    /// Number of trials.
    n: u64,
    /// Probability of success on each trial (must be in [0, 1]).
    p: f64,
}
impl Binomial {
    /// Create a new Binomial distribution with validated parameters.
    pub fn new(n: u64, p: f64) -> crate::error::FugueResult<Self> {
        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return Err(crate::error::FugueError::invalid_parameters(
                "Binomial", 
                "Probability must be in [0, 1]", 
                crate::error::ErrorCode::InvalidProbability
            ).with_context("p", format!("{}", p))
             .with_context("expected", "[0.0, 1.0]"));
        }
        Ok(Binomial { n, p })
    }

    /// Get the number of trials.
    pub fn n(&self) -> u64 {
        self.n
    }

    /// Get the success probability.
    pub fn p(&self) -> f64 {
        self.p
    }
}
impl Distribution<u64> for Binomial {
    fn sample(&self, rng: &mut dyn RngCore) -> u64 {
        RDBinomial::new(self.n, self.p).unwrap().sample(rng)
    }
    fn log_prob(&self, x: &u64) -> LogF64 {
        let k = *x;
        if k > self.n {
            return f64::NEG_INFINITY;
        }
        // log Binomial(k; n, p) = log C(n,k) + k*ln(p) + (n-k)*ln(1-p)
        let log_binom_coeff = libm::lgamma(self.n as f64 + 1.0)
            - libm::lgamma(k as f64 + 1.0)
            - libm::lgamma((self.n - k) as f64 + 1.0);
        log_binom_coeff + (k as f64) * self.p.ln() + ((self.n - k) as f64) * (1.0 - self.p).ln()
    }
    fn clone_box(&self) -> Box<dyn Distribution<u64>> {
        Box::new(*self)
    }
}

#[doc = include_str!("../../docs/api/core/distribution/distributions/poisson.md")]
#[derive(Clone, Copy, Debug)]
pub struct Poisson {
    /// Rate parameter λ (must be positive). Mean and variance of the distribution.
    lambda: f64,
}
impl Poisson {
    /// Create a new Poisson distribution with validated parameters.
    pub fn new(lambda: f64) -> crate::error::FugueResult<Self> {
        if lambda <= 0.0 || !lambda.is_finite() {
            return Err(crate::error::FugueError::invalid_parameters(
                "Poisson", 
                "Rate parameter lambda must be positive and finite", 
                crate::error::ErrorCode::InvalidRate
            ).with_context("lambda", format!("{}", lambda))
             .with_context("expected", "> 0.0 and finite"));
        }
        Ok(Poisson { lambda })
    }

    /// Get the rate parameter.
    pub fn lambda(&self) -> f64 {
        self.lambda
    }
}
impl Distribution<u64> for Poisson {
    fn sample(&self, rng: &mut dyn RngCore) -> u64 {
        if self.lambda <= 0.0 || !self.lambda.is_finite() {
            return 0;
        }
        RDPoisson::new(self.lambda).unwrap().sample(rng) as u64
    }
    fn log_prob(&self, x: &u64) -> LogF64 {
        // Parameter validation
        if self.lambda <= 0.0 || !self.lambda.is_finite() {
            return f64::NEG_INFINITY;
        }

        let k = *x;

        // Handle extreme cases
        if self.lambda > 700.0 && k == 0 {
            return -self.lambda; // Direct computation to avoid lgamma issues
        }

        // Numerically stable computation
        // log Poisson(k; λ) = k*ln(λ) - λ - ln(k!)
        let k_f64 = k as f64;
        let log_lambda = self.lambda.ln();
        let log_factorial = libm::lgamma(k_f64 + 1.0);

        k_f64 * log_lambda - self.lambda - log_factorial
    }
    fn clone_box(&self) -> Box<dyn Distribution<u64>> {
        Box::new(*self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn normal_constructor_and_log_prob() {
        assert!(Normal::new(0.0, 1.0).is_ok());
        assert!(Normal::new(f64::NAN, 1.0).is_err());
        assert!(Normal::new(0.0, 0.0).is_err());

        let n = Normal::new(0.0, 1.0).unwrap();
        assert!(n.log_prob(&0.0).is_finite());
        assert_eq!(n.log_prob(&f64::INFINITY), f64::NEG_INFINITY);
    }

    #[test]
    fn uniform_support_and_log_prob() {
        assert!(Uniform::new(0.0, 1.0).is_ok());
        assert!(Uniform::new(1.0, 0.0).is_err());
        let u = Uniform::new(-2.0, 2.0).unwrap();
        // Inside support
        let lp0 = u.log_prob(&0.0);
        assert!(lp0.is_finite());
        // Outside support
        assert_eq!(u.log_prob(&2.0), f64::NEG_INFINITY);
        assert_eq!(u.log_prob(&-2.1), f64::NEG_INFINITY);
    }

    #[test]
    fn lognormal_validation() {
        assert!(LogNormal::new(0.0, 1.0).is_ok());
        assert!(LogNormal::new(0.0, 0.0).is_err());
        let ln = LogNormal::new(0.0, 1.0).unwrap();
        assert_eq!(ln.log_prob(&0.0), f64::NEG_INFINITY);
        assert!(ln.log_prob(&1.0).is_finite());
    }

    #[test]
    fn exponential_validation() {
        assert!(Exponential::new(1.0).is_ok());
        assert!(Exponential::new(0.0).is_err());
        let e = Exponential::new(2.0).unwrap();
        assert_eq!(e.log_prob(&-1.0), f64::NEG_INFINITY);
        assert!((e.log_prob(&0.0) - (2.0f64).ln()).abs() < 1e-12);
    }

    #[test]
    fn bernoulli_validation() {
        assert!(Bernoulli::new(0.5).is_ok());
        assert!(Bernoulli::new(-0.1).is_err());
        let b = Bernoulli::new(0.25).unwrap();
        assert!((b.log_prob(&true) - (0.25f64).ln()).abs() < 1e-12);
        assert!((b.log_prob(&false) - (0.75f64).ln()).abs() < 1e-12);
    }

    #[test]
    fn categorical_validation_and_log_prob() {
        assert!(Categorical::new(vec![0.5, 0.5]).is_ok());
        assert!(Categorical::new(vec![]).is_err());
        assert!(Categorical::new(vec![0.6, 0.5]).is_err());

        let c = Categorical::new(vec![0.2, 0.8]).unwrap();
        assert!((c.log_prob(&1) - (0.8f64).ln()).abs() < 1e-12);
        assert_eq!(c.log_prob(&2), f64::NEG_INFINITY);
    }

    #[test]
    fn beta_validation_and_support() {
        assert!(Beta::new(2.0, 3.0).is_ok());
        assert!(Beta::new(0.0, 1.0).is_err());
        let b = Beta::new(2.0, 5.0).unwrap();
        assert_eq!(b.log_prob(&0.0), f64::NEG_INFINITY);
        assert_eq!(b.log_prob(&1.0), f64::NEG_INFINITY);
        assert!(b.log_prob(&0.5).is_finite());
    }

    #[test]
    fn gamma_validation_and_support() {
        assert!(Gamma::new(1.5, 2.0).is_ok());
        assert!(Gamma::new(0.0, 2.0).is_err());
        assert!(Gamma::new(1.0, 0.0).is_err());
        let g = Gamma::new(2.0, 1.0).unwrap();
        assert_eq!(g.log_prob(&-1.0), f64::NEG_INFINITY);
        assert!(g.log_prob(&1.0).is_finite());
    }

    #[test]
    fn binomial_validation_and_log_prob() {
        assert!(Binomial::new(10, 0.5).is_ok());
        assert!(Binomial::new(10, 1.5).is_err());
        let bi = Binomial::new(5, 0.3).unwrap();
        assert_eq!(bi.log_prob(&6), f64::NEG_INFINITY); // k > n
        assert!(bi.log_prob(&3).is_finite());
    }

    #[test]
    fn poisson_validation_and_log_prob() {
        assert!(Poisson::new(1.0).is_ok());
        assert!(Poisson::new(0.0).is_err());
        let p = Poisson::new(3.0).unwrap();
        assert!(p.log_prob(&0).is_finite());
        assert!(p.log_prob(&5).is_finite());
    }

    #[test]
    fn sampling_basic_sanity() {
        let mut rng = StdRng::seed_from_u64(42);
        let n = Normal::new(0.0, 1.0).unwrap();
        let x = n.sample(&mut rng);
        assert!(x.is_finite());

        let u = Uniform::new(-1.0, 2.0).unwrap();
        let y = u.sample(&mut rng);
        assert!(y >= -1.0 && y < 2.0);

        let b = Bernoulli::new(0.7).unwrap();
        let _z = b.sample(&mut rng);
    }
}

