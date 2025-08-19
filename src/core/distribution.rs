//! Probability distributions over `f64` with sampling and log-density.
//!
//! This module provides a unified interface for probability distributions used in Fugue models.
//! All distributions implement the `DistributionF64` trait, which provides sampling and
//! log-probability density computation. The trait is designed to be dyn-object safe,
//! allowing distributions to be stored as trait objects within `Model` computations.
//!
//! ## Available Distributions
//!
//! ### Continuous Distributions
//! - [`Normal`]: Normal/Gaussian distribution
//! - [`LogNormal`]: Log-normal distribution  
//! - [`Uniform`]: Uniform distribution over an interval
//! - [`Exponential`]: Exponential distribution
//! - [`Beta`]: Beta distribution on \[0,1\]
//! - [`Gamma`]: Gamma distribution
//!
//! ### Discrete Distributions
//! - [`Bernoulli`]: Bernoulli distribution (0 or 1)
//! - [`Binomial`]: Binomial distribution
//! - [`Categorical`]: Categorical distribution over discrete choices
//! - [`Poisson`]: Poisson distribution
//!
//! ## Usage
//!
//! ```rust
//! use fugue::*;
//!
//! // Create and use distributions in models
//! let model = sample(addr!("x"), Normal { mu: 0.0, sigma: 1.0 })
//!     .bind(|x| {
//!         let transformed = if x > 0.0 { 
//!             Exponential { rate: x } 
//!         } else { 
//!             Exponential { rate: 0.1 } 
//!         };
//!         sample(addr!("y"), transformed)
//!     });
//! ```
use rand::{Rng, RngCore};
use rand_distr::{
    Bernoulli as RDBernoulli, Beta as RDBeta, Binomial as RDBinomial, Distribution as RandDistr,
    Exp as RDExp, Gamma as RDGamma, LogNormal as RDLogNormal, Normal as RDNormal,
    Poisson as RDPoisson,
};
/// Type alias for log-probabilities.
///
/// Log-probabilities are represented as `f64` values. Negative infinity represents
/// zero probability, while finite values represent the natural logarithm of probabilities.
pub type LogF64 = f64;

/// Common interface for probability distributions over `f64` values.
///
/// This trait provides the essential operations needed for probabilistic programming:
/// sampling from the distribution and computing log-probability densities. The trait
/// is object-safe, allowing distributions to be stored as trait objects.
///
/// All distributions in Fugue implement this trait, enabling generic probabilistic
/// programming where the specific distribution can be chosen at runtime.
///
/// # Required Methods
///
/// - [`sample`](Self::sample): Generate a random sample from the distribution
/// - [`log_prob`](Self::log_prob): Compute the log-probability density at a point
/// - [`clone_box`](Self::clone_box): Clone the distribution into a boxed trait object
///
/// # Examples
///
/// ```rust
/// use fugue::*;
/// use rand::rngs::StdRng;
/// use rand::SeedableRng;
///
/// // Use trait methods directly
/// let normal = Normal { mu: 0.0, sigma: 1.0 };
/// let mut rng = StdRng::seed_from_u64(42);
/// 
/// let sample = normal.sample(&mut rng);
/// let log_prob = normal.log_prob(0.0); // Should be near -0.92 (for standard normal)
/// ```
pub trait DistributionF64: Send + Sync {
    /// Generate a random sample from this distribution.
    ///
    /// # Arguments
    ///
    /// * `rng` - Random number generator to use for sampling
    ///
    /// # Returns
    ///
    /// A sample from the distribution as an `f64`.
    fn sample(&self, rng: &mut dyn RngCore) -> f64;
    
    /// Compute the log-probability density of a value under this distribution.
    ///
    /// # Arguments
    ///
    /// * `x` - Value to compute log-probability for
    ///
    /// # Returns
    ///
    /// The natural logarithm of the probability density at `x`.
    /// Returns negative infinity for values outside the distribution's support.
    fn log_prob(&self, x: f64) -> LogF64;
    
    /// Clone this distribution into a boxed trait object.
    ///
    /// This method is required for the trait to be object-safe, allowing
    /// distributions to be stored as `Box<dyn DistributionF64>`.
    fn clone_box(&self) -> Box<dyn DistributionF64>;
}

/// Normal (Gaussian) distribution.
///
/// The normal distribution is a continuous probability distribution characterized by
/// its mean (μ) and standard deviation (σ). It's one of the most important distributions
/// in statistics and is commonly used as a prior or likelihood in Bayesian models.
///
/// **Probability density function:**
/// ```text
/// f(x) = (1 / (σ√(2π))) * exp(-0.5 * ((x - μ) / σ)²)
/// ```
///
/// **Support:** All real numbers (-∞, +∞)
///
/// # Fields
///
/// * `mu` - Mean of the distribution
/// * `sigma` - Standard deviation (must be positive)
///
/// # Examples
///
/// ```rust
/// use fugue::*;
///
/// // Standard normal distribution
/// let std_normal = Normal { mu: 0.0, sigma: 1.0 };
///
/// // Normal prior for a parameter
/// let model = sample(addr!("theta"), Normal { mu: 0.0, sigma: 2.0 });
///
/// // Normal likelihood for observations
/// let model = observe(addr!("y"), Normal { mu: 1.5, sigma: 0.5 }, 2.0);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Normal {
    /// Mean of the normal distribution.
    pub mu: f64,
    /// Standard deviation of the normal distribution (must be positive).
    pub sigma: f64,
}
impl DistributionF64 for Normal {
    fn sample(&self, rng: &mut dyn RngCore) -> f64 {
        RDNormal::new(self.mu, self.sigma).unwrap().sample(rng)
    }
    fn log_prob(&self, x: f64) -> LogF64 {
        let z = (x - self.mu) / self.sigma;
        -0.5 * z * z - self.sigma.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
    }
    fn clone_box(&self) -> Box<dyn DistributionF64> {
        Box::new(*self)
    }
}

/// Uniform distribution over a continuous interval.
///
/// The uniform distribution assigns equal probability density to all values
/// within a specified interval [low, high) and zero probability outside.
///
/// **Probability density function:**
/// ```text
/// f(x) = 1 / (high - low)  for low ≤ x < high
/// f(x) = 0                 otherwise
/// ```
///
/// **Support:** [low, high)
///
/// # Fields
///
/// * `low` - Lower bound of the distribution (inclusive)
/// * `high` - Upper bound of the distribution (exclusive)
///
/// # Examples
///
/// ```rust
/// use fugue::*;
///
/// // Unit interval
/// let unit_uniform = Uniform { low: 0.0, high: 1.0 };
///
/// // Symmetric interval around zero
/// let symmetric = Uniform { low: -5.0, high: 5.0 };
///
/// // Use as uninformative prior
/// let model = sample(addr!("weight"), Uniform { low: 0.0, high: 100.0 });
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Uniform {
    /// Lower bound of the uniform distribution (inclusive).
    pub low: f64,
    /// Upper bound of the uniform distribution (exclusive).
    pub high: f64,
}
impl DistributionF64 for Uniform {
    fn sample(&self, rng: &mut dyn RngCore) -> f64 {
        Rng::gen_range(rng, self.low..self.high)
    }
    fn log_prob(&self, x: f64) -> LogF64 {
        if x < self.low || x > self.high {
            f64::NEG_INFINITY
        } else {
            -(self.high - self.low).ln()
        }
    }
    fn clone_box(&self) -> Box<dyn DistributionF64> {
        Box::new(*self)
    }
}

/// Log-normal distribution.
///
/// A continuous distribution where the logarithm of the random variable follows
/// a normal distribution. This distribution is useful for modeling positive-valued
/// quantities that are naturally multiplicative or skewed.
///
/// **Relationship to Normal:** If X ~ LogNormal(μ, σ), then ln(X) ~ Normal(μ, σ)
///
/// **Probability density function:**
/// ```text
/// f(x) = (1 / (x * σ√(2π))) * exp(-0.5 * ((ln(x) - μ) / σ)²)  for x > 0
/// f(x) = 0                                                      for x ≤ 0
/// ```
///
/// **Support:** (0, +∞)
///
/// # Fields
///
/// * `mu` - Mean of the underlying normal distribution
/// * `sigma` - Standard deviation of the underlying normal distribution (must be positive)
///
/// # Examples
///
/// ```rust
/// use fugue::*;
///
/// // Standard log-normal
/// let log_normal = LogNormal { mu: 0.0, sigma: 1.0 };
///
/// // Model for positive scale parameters
/// let model = sample(addr!("scale"), LogNormal { mu: 0.0, sigma: 0.5 });
///
/// // Income distribution (often log-normal)
/// let income_model = sample(addr!("income"), LogNormal { mu: 10.0, sigma: 0.8 });
/// ```
#[derive(Clone, Copy, Debug)]
pub struct LogNormal {
    /// Mean of the underlying normal distribution.
    pub mu: f64,
    /// Standard deviation of the underlying normal distribution (must be positive).
    pub sigma: f64,
}
impl DistributionF64 for LogNormal {
    fn sample(&self, rng: &mut dyn RngCore) -> f64 {
        RDLogNormal::new(self.mu, self.sigma).unwrap().sample(rng)
    }
    fn log_prob(&self, x: f64) -> LogF64 {
        if x <= 0.0 {
            return f64::NEG_INFINITY;
        }
        let lx = x.ln();
        let z = (lx - self.mu) / self.sigma;
        -0.5 * z * z - (self.sigma * x).ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
    }
    fn clone_box(&self) -> Box<dyn DistributionF64> {
        Box::new(*self)
    }
}

/// Exponential distribution.
///
/// A continuous probability distribution often used to model waiting times
/// between events in a Poisson process. It has a single parameter (rate) and
/// is characterized by the memoryless property.
///
/// **Probability density function:**
/// ```text
/// f(x) = λ * exp(-λx)  for x ≥ 0
/// f(x) = 0             for x < 0
/// ```
///
/// **Support:** [0, +∞)
///
/// # Fields
///
/// * `rate` - Rate parameter λ (must be positive). Higher values = shorter waiting times.
///
/// # Examples
///
/// ```rust
/// use fugue::*;
///
/// // Model time between events (rate = 2 events per unit time)
/// let waiting_time = Exponential { rate: 2.0 };
///
/// // Survival analysis / hazard modeling
/// let model = sample(addr!("survival_time"), Exponential { rate: 0.1 });
///
/// // Prior for precision parameters (inverse of variance)
/// let precision_prior = sample(addr!("precision"), Exponential { rate: 1.0 });
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Exponential {
    /// Rate parameter λ of the exponential distribution (must be positive).
    pub rate: f64,
}
impl DistributionF64 for Exponential {
    fn sample(&self, rng: &mut dyn RngCore) -> f64 {
        RDExp::new(self.rate).unwrap().sample(rng)
    }
    fn log_prob(&self, x: f64) -> LogF64 {
        if x < 0.0 {
            f64::NEG_INFINITY
        } else {
            self.rate.ln() - self.rate * x
        }
    }
    fn clone_box(&self) -> Box<dyn DistributionF64> {
        Box::new(*self)
    }
}

/// Bernoulli distribution.
///
/// A discrete distribution representing a single trial with two possible outcomes:
/// success (1.0) with probability p, or failure (0.0) with probability 1-p.
/// This is the building block for binomial distributions and binary classification.
///
/// **Probability mass function:**
/// ```text
/// P(X = 1) = p
/// P(X = 0) = 1 - p
/// ```
///
/// **Support:** {0.0, 1.0} (represented as f64 for trait compatibility)
///
/// # Fields
///
/// * `p` - Probability of success (must be in [0, 1])
///
/// # Examples
///
/// ```rust
/// use fugue::*;
///
/// // Fair coin flip
/// let coin = Bernoulli { p: 0.5 };
///
/// // Biased coin
/// let biased_coin = Bernoulli { p: 0.7 };
///
/// // Binary classification model
/// let model = sample(addr!("class"), Bernoulli { p: 0.8 });
///
/// // Mixture component indicator
/// let component = sample(addr!("component"), Bernoulli { p: 0.3 });
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Bernoulli {
    /// Probability of success (must be in [0, 1]).
    pub p: f64,
}
impl DistributionF64 for Bernoulli {
    fn sample(&self, rng: &mut dyn RngCore) -> f64 {
        if RDBernoulli::new(self.p).unwrap().sample(rng) {
            1.0
        } else {
            0.0
        }
    }
    fn log_prob(&self, x: f64) -> LogF64 {
        if x == 1.0 {
            self.p.ln()
        } else if x == 0.0 {
            (1.0 - self.p).ln()
        } else {
            f64::NEG_INFINITY
        }
    }
    fn clone_box(&self) -> Box<dyn DistributionF64> {
        Box::new(*self)
    }
}

/// Categorical distribution over discrete choices.
///
/// A discrete distribution that represents choosing among k different categories
/// with specified probabilities. The outcome is the index of the chosen category
/// (as an f64 for trait compatibility).
///
/// **Probability mass function:**
/// ```text
/// P(X = i) = probs[i]  for i ∈ {0, 1, ..., k-1}
/// ```
///
/// **Support:** {0.0, 1.0, ..., k-1.0} where k = probs.len()
///
/// # Fields
///
/// * `probs` - Vector of probabilities for each category (should sum to 1.0)
///
/// # Examples
///
/// ```rust
/// use fugue::*;
///
/// // Three-way choice
/// let choice = Categorical { 
///     probs: vec![0.5, 0.3, 0.2] 
/// };
///
/// // Mixture component selection
/// let component = sample(addr!("component"), Categorical {
///     probs: vec![0.4, 0.6]
/// });
///
/// // Discrete outcome modeling
/// let outcome = sample(addr!("outcome"), Categorical {
///     probs: vec![0.1, 0.2, 0.3, 0.4]
/// });
/// ```
#[derive(Clone, Debug)]
pub struct Categorical {
    /// Probabilities for each category (should sum to 1.0).
    pub probs: Vec<f64>,
}
impl DistributionF64 for Categorical {
    fn sample(&self, rng: &mut dyn RngCore) -> f64 {
        let u: f64 = rng.gen();
        let mut cum = 0.0;
        for (i, &p) in self.probs.iter().enumerate() {
            cum += p;
            if u <= cum {
                return i as f64;
            }
        }
        (self.probs.len() - 1) as f64
    }
    fn log_prob(&self, x: f64) -> LogF64 {
        let idx = x as usize;
        if idx < self.probs.len() && (x - idx as f64).abs() < 1e-12 {
            self.probs[idx].ln()
        } else {
            f64::NEG_INFINITY
        }
    }
    fn clone_box(&self) -> Box<dyn DistributionF64> {
        Box::new(self.clone())
    }
}

/// Beta distribution on the interval [0, 1].
///
/// A continuous distribution over the unit interval, commonly used for modeling
/// probabilities, proportions, and as a conjugate prior for Bernoulli/Binomial
/// distributions. The shape is controlled by two positive parameters α and β.
///
/// **Probability density function:**
/// ```text
/// f(x) = (x^(α-1) * (1-x)^(β-1)) / B(α,β)  for 0 < x < 1
/// f(x) = 0                                  otherwise
/// ```
/// where B(α,β) is the beta function.
///
/// **Support:** (0, 1)
///
/// # Fields
///
/// * `alpha` - First shape parameter α (must be positive)
/// * `beta` - Second shape parameter β (must be positive)
///
/// # Examples
///
/// ```rust
/// use fugue::*;
///
/// // Uniform on [0,1] (alpha=1, beta=1)
/// let uniform_beta = Beta { alpha: 1.0, beta: 1.0 };
///
/// // Prior for a probability parameter
/// let prob_prior = sample(addr!("p"), Beta { alpha: 2.0, beta: 5.0 });
///
/// // Conjugate prior for Bernoulli likelihood
/// let model = sample(addr!("success_rate"), Beta { alpha: 3.0, beta: 7.0 })
///     .bind(|p| observe(addr!("trial"), Bernoulli { p }, 1.0));
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Beta {
    /// First shape parameter α (must be positive).
    pub alpha: f64,
    /// Second shape parameter β (must be positive).
    pub beta: f64,
}
impl DistributionF64 for Beta {
    fn sample(&self, rng: &mut dyn RngCore) -> f64 {
        RDBeta::new(self.alpha, self.beta).unwrap().sample(rng)
    }
    fn log_prob(&self, x: f64) -> LogF64 {
        if x <= 0.0 || x >= 1.0 {
            return f64::NEG_INFINITY;
        }
        // log Beta(x; α, β) = (α-1)ln(x) + (β-1)ln(1-x) - log B(α,β)
        let log_beta_fn = libm::lgamma(self.alpha) + libm::lgamma(self.beta)
            - libm::lgamma(self.alpha + self.beta);
        (self.alpha - 1.0) * x.ln() + (self.beta - 1.0) * (1.0 - x).ln() - log_beta_fn
    }
    fn clone_box(&self) -> Box<dyn DistributionF64> {
        Box::new(*self)
    }
}

/// Gamma distribution.
///
/// A continuous probability distribution over positive real numbers, parameterized
/// by shape (k) and rate (λ). The Gamma distribution is commonly used for modeling
/// waiting times, scale parameters, and as a conjugate prior for Poisson distributions.
///
/// **Probability density function:**
/// ```text
/// f(x) = (λ^k / Γ(k)) * x^(k-1) * exp(-λx)  for x > 0
/// f(x) = 0                                   for x ≤ 0
/// ```
/// where Γ(k) is the gamma function.
///
/// **Support:** (0, +∞)
///
/// # Fields
///
/// * `shape` - Shape parameter k (must be positive)
/// * `rate` - Rate parameter λ (must be positive). Note: rate = 1/scale
///
/// # Examples
///
/// ```rust
/// use fugue::*;
///
/// // Exponential is Gamma(1, rate)
/// let exponential_like = Gamma { shape: 1.0, rate: 2.0 };
///
/// // Prior for precision (inverse variance)
/// let precision = sample(addr!("precision"), Gamma { shape: 2.0, rate: 1.0 });
///
/// // Conjugate prior for Poisson rate
/// let model = sample(addr!("rate"), Gamma { shape: 3.0, rate: 2.0 })
///     .bind(|lambda| observe(addr!("count"), Poisson { lambda }, 5.0));
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Gamma {
    /// Shape parameter k (must be positive).
    pub shape: f64,
    /// Rate parameter λ (must be positive).
    pub rate: f64,
}
impl DistributionF64 for Gamma {
    fn sample(&self, rng: &mut dyn RngCore) -> f64 {
        RDGamma::new(self.shape, 1.0 / self.rate)
            .unwrap()
            .sample(rng)
    }
    fn log_prob(&self, x: f64) -> LogF64 {
        if x <= 0.0 {
            return f64::NEG_INFINITY;
        }
        // log Gamma(x; k, λ) = k*ln(λ) + (k-1)*ln(x) - λ*x - ln Γ(k)
        self.shape * self.rate.ln() + (self.shape - 1.0) * x.ln()
            - self.rate * x
            - libm::lgamma(self.shape)
    }
    fn clone_box(&self) -> Box<dyn DistributionF64> {
        Box::new(*self)
    }
}

/// Binomial distribution.
///
/// A discrete distribution representing the number of successes in n independent
/// Bernoulli trials, each with success probability p. This distribution models
/// counting processes and is widely used in statistics.
///
/// **Probability mass function:**
/// ```text
/// P(X = k) = C(n,k) * p^k * (1-p)^(n-k)  for k ∈ {0, 1, ..., n}
/// ```
/// where C(n,k) is the binomial coefficient "n choose k".
///
/// **Support:** {0.0, 1.0, ..., n.0} (represented as f64 for trait compatibility)
///
/// # Fields
///
/// * `n` - Number of trials (must be non-negative)
/// * `p` - Probability of success on each trial (must be in [0, 1])
///
/// # Examples
///
/// ```rust
/// use fugue::*;
///
/// // Count successes in 10 coin flips
/// let coin_flips = Binomial { n: 10, p: 0.5 };
///
/// // Clinical trial success count
/// let trial_model = sample(addr!("success_rate"), Beta { alpha: 1.0, beta: 1.0 })
///     .bind(|p| sample(addr!("successes"), Binomial { n: 100, p }));
///
/// // Observe data from a binomial process
/// let model = observe(addr!("observed_successes"), Binomial { n: 20, p: 0.3 }, 7.0);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Binomial {
    /// Number of trials.
    pub n: u64,
    /// Probability of success on each trial (must be in [0, 1]).
    pub p: f64,
}
impl DistributionF64 for Binomial {
    fn sample(&self, rng: &mut dyn RngCore) -> f64 {
        RDBinomial::new(self.n, self.p).unwrap().sample(rng) as f64
    }
    fn log_prob(&self, x: f64) -> LogF64 {
        let k = x as u64;
        if k > self.n || (x - k as f64).abs() > 1e-12 {
            return f64::NEG_INFINITY;
        }
        // log Binomial(k; n, p) = log C(n,k) + k*ln(p) + (n-k)*ln(1-p)
        let log_binom_coeff = libm::lgamma(self.n as f64 + 1.0)
            - libm::lgamma(k as f64 + 1.0)
            - libm::lgamma((self.n - k) as f64 + 1.0);
        log_binom_coeff + (k as f64) * self.p.ln() + ((self.n - k) as f64) * (1.0 - self.p).ln()
    }
    fn clone_box(&self) -> Box<dyn DistributionF64> {
        Box::new(*self)
    }
}

/// Poisson distribution.
///
/// A discrete probability distribution expressing the probability of a given number
/// of events occurring in a fixed interval of time or space, given that these events
/// occur with a known constant mean rate and independently of each other.
///
/// **Probability mass function:**
/// ```text
/// P(X = k) = (λ^k * exp(-λ)) / k!  for k ∈ {0, 1, 2, ...}
/// ```
///
/// **Support:** {0.0, 1.0, 2.0, ...} (represented as f64 for trait compatibility)
///
/// # Fields
///
/// * `lambda` - Rate parameter λ (must be positive). This is both the mean and variance.
///
/// # Examples
///
/// ```rust
/// use fugue::*;
///
/// // Model number of events per time unit
/// let events = Poisson { lambda: 3.5 };
///
/// // Count data modeling
/// let model = sample(addr!("rate"), Gamma { shape: 2.0, rate: 1.0 })
///     .bind(|lambda| observe(addr!("count"), Poisson { lambda }, 4.0));
///
/// // Rare events modeling
/// let rare_events = sample(addr!("occurrences"), Poisson { lambda: 0.1 });
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Poisson {
    /// Rate parameter λ (must be positive). Mean and variance of the distribution.
    pub lambda: f64,
}
impl DistributionF64 for Poisson {
    fn sample(&self, rng: &mut dyn RngCore) -> f64 {
        RDPoisson::new(self.lambda).unwrap().sample(rng) as f64
    }
    fn log_prob(&self, x: f64) -> LogF64 {
        let k = x as u64;
        if (x - k as f64).abs() > 1e-12 || x < 0.0 {
            return f64::NEG_INFINITY;
        }
        // log Poisson(k; λ) = k*ln(λ) - λ - ln(k!)
        (k as f64) * self.lambda.ln() - self.lambda - libm::lgamma(k as f64 + 1.0)
    }
    fn clone_box(&self) -> Box<dyn DistributionF64> {
        Box::new(*self)
    }
}
