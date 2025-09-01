#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/docs/api/core/distribution.md"))]
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

/// Generic interface for type-safe probability distributions.
/// All distributions implement `Distribution<T>` where `T` is the natural return type.
/// Example:
///
/// ```rust
/// # use fugue::*;
/// # use rand::thread_rng;
///
/// let mut rng = thread_rng();
///
/// // Type-safe sampling
/// let coin = Bernoulli::new(0.5).unwrap();
/// let flip: bool = coin.sample(&mut rng);  // Natural boolean
/// let prob = coin.log_prob(&flip);
///
/// // Safe indexing
/// let choice = Categorical::uniform(3).unwrap();
/// let idx: usize = choice.sample(&mut rng);  // Safe for arrays
/// let choice_prob = choice.log_prob(&idx);
///
/// // Natural counting
/// let events = Poisson::new(3.0).unwrap();
/// let count: u64 = events.sample(&mut rng);  // Natural count type
/// let count_prob = events.log_prob(&count);
/// ```
pub trait Distribution<T>: Send + Sync {
    /// Generate a random sample (with its natural type), `T`, from the distribution, using the provided random number generator, `rng`.
    ///
    /// Example:
    /// ```rust
    /// # use fugue::*;
    /// # use rand::thread_rng;
    ///
    /// let mut rng = thread_rng();
    ///
    /// // Sample different distribution types
    /// let normal_sample: f64 = Normal::new(0.0, 1.0).unwrap().sample(&mut rng);
    /// let coin_flip: bool = Bernoulli::new(0.5).unwrap().sample(&mut rng);
    /// let event_count: u64 = Poisson::new(3.0).unwrap().sample(&mut rng);
    /// let category_idx: usize = Categorical::uniform(5).unwrap().sample(&mut rng);
    /// ```
    fn sample(&self, rng: &mut dyn RngCore) -> T;

    /// Compute the log-probability density (continuous) or mass (discrete) of a value, `x`, from the distribution.
    ///
    /// Example:
    /// ```rust
    /// # use fugue::*;
    ///
    /// // Continuous distribution (probability density)
    /// let normal = Normal::new(0.0, 1.0).unwrap();
    /// let density = normal.log_prob(&0.0);  // Peak of standard normal
    ///
    /// // Discrete distribution (probability mass)
    /// let coin = Bernoulli::new(0.7).unwrap();
    /// let prob_true = coin.log_prob(&true);   // ln(0.7)
    /// let prob_false = coin.log_prob(&false); // ln(0.3)
    ///
    /// // Outside support returns -∞
    /// let poisson = Poisson::new(3.0).unwrap();
    /// let invalid = poisson.log_prob(&u64::MAX); // Very unlikely, returns -∞
    /// ```
    fn log_prob(&self, x: &T) -> LogF64;

    /// Clone the distribution into a boxed trait object, `Box<dyn Distribution<T>>`.
    ///
    /// Example:
    /// ```rust
    /// # use fugue::*;
    ///
    /// // Clone a distribution into a box
    /// let original = Normal::new(0.0, 1.0).unwrap();
    /// let boxed: Box<dyn Distribution<f64>> = original.clone_box();
    ///
    /// // Useful for storing different distribution types
    /// let mut distributions: Vec<Box<dyn Distribution<f64>>> = vec![];
    /// distributions.push(Normal::new(0.0, 1.0).unwrap().clone_box());
    /// distributions.push(Uniform::new(-1.0, 1.0).unwrap().clone_box());
    /// ```
    fn clone_box(&self) -> Box<dyn Distribution<T>>;
}

/// A continuous distribution characterized by its mean, `mu`, and standard deviation, `sigma`.
///
/// Mathematical Properties:
/// - **Support**: (-∞, +∞)
/// - **PDF**: f(x) = (1/(σ√(2π))) × exp(-0.5 × ((x-μ)/σ)²)
/// - **Mean**: μ
/// - **Variance**: σ²
/// - **68-95-99.7 rule**: ~68% within 1σ, ~95% within 2σ, ~99.7% within 3σ
///
/// Example:
/// ```rust
/// # use fugue::*;
///
/// // Standard normal (mean=0, std=1)
/// let standard = sample(addr!("z"), Normal::new(0.0, 1.0).unwrap());
///
/// // Parameter with prior
/// let theta = sample(addr!("theta"), Normal::new(0.0, 2.0).unwrap());
///
/// // Likelihood with observation
/// let likelihood = observe(addr!("y"), Normal::new(1.5, 0.5).unwrap(), 2.0);
///
/// // Measurement error model
/// let true_value = sample(addr!("true_val"), Normal::new(100.0, 10.0).unwrap());
/// let measurement = true_value.bind(|val| {
///     observe(addr!("measured"), Normal::new(val, 2.0).unwrap(), 98.5)
/// });
/// ```
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
                crate::error::ErrorCode::InvalidMean,
            )
            .with_context("mu", format!("{}", mu)));
        }
        if sigma <= 0.0 || !sigma.is_finite() {
            return Err(crate::error::FugueError::invalid_parameters(
                "Normal",
                "Standard deviation (sigma) must be positive and finite",
                crate::error::ErrorCode::InvalidVariance,
            )
            .with_context("sigma", format!("{}", sigma))
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

/// A continuous distribution that assigns equal probability density to all values within a specified interval, from `low` to `high`.
///
/// Commonly used as an uninformative prior when you want to express complete uncertainty over a bounded range.
///
/// Mathematical Properties:
/// - **Support**: [low, high)
/// - **PDF**: f(x) = 1/(high-low) for low ≤ x < high, 0 otherwise
/// - **Mean**: (low + high) / 2
/// - **Variance**: (high - low)² / 12
///
/// Example:
///
/// ```rust
/// # use fugue::*;
///
/// // Unit interval [0, 1)
/// let unit = sample(addr!("p"), Uniform::new(0.0, 1.0).unwrap());
///
/// // Symmetric around zero
/// let symmetric = sample(addr!("x"), Uniform::new(-5.0, 5.0).unwrap());
///
/// // Uninformative prior for weight
/// let weight = sample(addr!("weight"), Uniform::new(0.0, 100.0).unwrap());
///
/// // Random angle in radians
/// let angle = sample(addr!("angle"), Uniform::new(0.0, 2.0 * std::f64::consts::PI).unwrap());
/// ```
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
                crate::error::ErrorCode::InvalidRange,
            )
            .with_context("low", format!("{}", low))
            .with_context("high", format!("{}", high)));
        }
        if low >= high {
            return Err(crate::error::FugueError::invalid_parameters(
                "Uniform",
                "Lower bound must be less than upper bound",
                crate::error::ErrorCode::InvalidRange,
            )
            .with_context("low", format!("{}", low))
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

/// A continuous distribution where the logarithm follows a normal distribution.
///
/// Useful for modeling positive-valued quantities that are naturally multiplicative or skewed.
///
/// Mathematical Properties:
/// - **Support**: (0, +∞)
/// - **PDF**: f(x) = (1/(xσ√(2π))) × exp(-0.5 × ((ln(x)-μ)/σ)²)
/// - **Mean**: exp(μ + σ²/2)
/// - **Variance**: (exp(σ²) - 1) × exp(2μ + σ²)
/// - **Relationship**: If X ~ LogNormal(μ, σ), then ln(X) ~ Normal(μ, σ)
///
/// Example:
/// ```rust
/// # use fugue::*;
///
/// // Standard log-normal (median = 1)
/// let standard = sample(addr!("x"), LogNormal::new(0.0, 1.0).unwrap());
///
/// // Positive scale parameter
/// let scale = sample(addr!("scale"), LogNormal::new(0.0, 0.5).unwrap());
///
/// // Income distribution
/// let income = sample(addr!("income"), LogNormal::new(10.0, 0.8).unwrap())
///     .map(|x| x.round() as u64); // Convert to dollars
///
/// // Multiplicative error model
/// let true_value = 100.0;
/// let measured = sample(addr!("error"), LogNormal::new(0.0, 0.1).unwrap())
///     .map(move |error| true_value * error);
/// ```
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
                crate::error::ErrorCode::InvalidMean,
            )
            .with_context("mu", format!("{}", mu)));
        }
        if sigma <= 0.0 || !sigma.is_finite() {
            return Err(crate::error::FugueError::invalid_parameters(
                "LogNormal",
                "Standard deviation (sigma) must be positive and finite",
                crate::error::ErrorCode::InvalidVariance,
            )
            .with_context("sigma", format!("{}", sigma))
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

/// A continuous distribution often used to model waiting times between events.
///
/// Characterized by the memoryless property.
///
/// Mathematical Properties:
/// - **Support**: [0, +∞)
/// - **PDF**: f(x) = λ × exp(-λx) for x ≥ 0
/// - **Mean**: 1 / λ
/// - **Variance**: 1 / λ²
/// - **Memoryless**: P(X > s + t | X > s) = P(X > t)
///
/// Example:
/// ```rust
/// # use fugue::*;
///
/// // Average wait time of 2 minutes (rate = 0.5 per minute)
/// let wait_time = sample(addr!("wait"), Exponential::new(0.5).unwrap());
///
/// // Service time model
/// let service = sample(addr!("service_time"), Exponential::new(1.5).unwrap())
///     .bind(|time| {
///         if time > 5.0 {
///             pure("slow")
///         } else {
///             pure("fast")
///         }
///     });
///
/// // Observe actual waiting time
/// let observed = observe(addr!("actual_wait"), Exponential::new(0.3).unwrap(), 4.2);
/// ```
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
                crate::error::ErrorCode::InvalidRate,
            )
            .with_context("rate", format!("{}", rate))
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

/// A discrete distribution for binary outcomes (true/false, success/failure).
///
/// Returns `bool` directly for type-safe boolean logic.
///
/// Mathematical Properties:
/// - **Support**: {false, true}
/// - **PMF**: P(X = true) = p, P(X = false) = 1 - p
/// - **Mean**: p
/// - **Variance**: p(1 - p)
///
/// Example:
/// ```rust
/// # use fugue::*;
///
/// // Fair coin flip
/// let coin = sample(addr!("coin"), Bernoulli::new(0.5).unwrap());
/// let result = coin.bind(|heads| {
///     if heads {
///         pure("Heads!")
///     } else {
///         pure("Tails!")
///     }
/// });
///
/// // Biased coin with observation
/// let biased = observe(addr!("biased_coin"), Bernoulli::new(0.7).unwrap(), true);
/// ```
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
                crate::error::ErrorCode::InvalidProbability,
            )
            .with_context("p", format!("{}", p))
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

/// A discrete distribution for choosing among multiple categories with specified probabilities.
///
/// Returns `usize` for safe array indexing.
///
/// Mathematical Properties:
/// - **Support**: {0, 1, ..., k-1} where k = number of categories
/// - **PMF**: P(X = i) = probs[i]
/// - **Mean**: Σ(i × probs[i])
/// - **Variance**: Σ(i² × probs[i]) - mean²
///
/// Example:
/// ```rust
/// # use fugue::*;
///
/// // Custom probabilities
/// let weighted = Categorical::new(vec![0.1, 0.2, 0.3, 0.4]).unwrap();
///
/// // Uniform distribution over k categories
/// let uniform = Categorical::uniform(4).unwrap();
///
/// // Choose from three options
/// let options = vec!["red", "green", "blue"];
/// let choice = sample(addr!("color"), Categorical::new(vec![0.5, 0.3, 0.2]).unwrap())
///     .map(move |idx| options[idx].to_string());
///
/// // Observe a specific choice
/// let observed = observe(addr!("user_choice"),
///     Categorical::uniform(3).unwrap(), 1usize);
/// ```
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
                crate::error::ErrorCode::InvalidProbability,
            )
            .with_context("length", "0"));
        }

        let sum: f64 = probs.iter().sum();
        if (sum - 1.0).abs() > 1e-6 {
            return Err(crate::error::FugueError::invalid_parameters(
                "Categorical",
                "Probabilities must sum to 1.0",
                crate::error::ErrorCode::InvalidProbability,
            )
            .with_context("sum", format!("{:.6}", sum))
            .with_context("expected", "1.0")
            .with_context("tolerance", "1e-6"));
        }

        for (i, &p) in probs.iter().enumerate() {
            if !p.is_finite() || p < 0.0 {
                return Err(crate::error::FugueError::invalid_parameters(
                    "Categorical",
                    "All probabilities must be non-negative and finite",
                    crate::error::ErrorCode::InvalidProbability,
                )
                .with_context("index", format!("{}", i))
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
                crate::error::ErrorCode::InvalidCount,
            )
            .with_context("k", "0"));
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

/// A continuous distribution on the interval (0, 1), commonly used for modeling probabilities and proportions.
///
/// Conjugate prior for Bernoulli/Binomial distributions.
///
/// Mathematical Properties:
/// - **Support**: (0, 1)
/// - **PDF**: f(x) = (x^(α-1) × (1-x)^(β-1)) / B(α,β)
/// - **Mean**: α / (α + β)
/// - **Variance**: (αβ) / ((α+β)²(α+β+1))
///
/// Example:
/// ```rust
/// # use fugue::*;
///
/// // Uniform on [0,1]
/// let uniform = sample(addr!("p"), Beta::new(1.0, 1.0).unwrap());
///
/// // Prior for success probability
/// let prob_prior = sample(addr!("success_rate"), Beta::new(2.0, 5.0).unwrap());
///
/// // Conjugate prior-likelihood pair
/// let model = sample(addr!("p"), Beta::new(3.0, 7.0).unwrap())
///     .bind(|p| observe(addr!("trial"), Bernoulli::new(p).unwrap(), true));
///
/// // Skewed towards 0 (beta > alpha)
/// let skewed = sample(addr!("proportion"), Beta::new(2.0, 8.0).unwrap());
/// ```
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
                crate::error::ErrorCode::InvalidShape,
            )
            .with_context("alpha", format!("{}", alpha))
            .with_context("expected", "> 0.0 and finite"));
        }
        if beta <= 0.0 || !beta.is_finite() {
            return Err(crate::error::FugueError::invalid_parameters(
                "Beta",
                "Beta parameter must be positive and finite",
                crate::error::ErrorCode::InvalidShape,
            )
            .with_context("beta", format!("{}", beta))
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

/// A continuous distribution over positive real numbers, parameterized by shape and rate.
///
/// Commonly used for modeling waiting times and as a conjugate prior for Poisson distributions.
///
/// Mathematical Properties:
/// - **Support**: (0, +∞)
/// - **PDF**: f(x) = (λ^k / Γ(k)) × x^(k-1) × exp(-λx)
/// - **Mean**: k / λ
/// - **Variance**: k / λ²
///
/// Example:
/// ```rust
/// # use fugue::*;
///
/// // Shape=1 gives Exponential distribution
/// let exponential_like = sample(addr!("wait_time"), Gamma::new(1.0, 2.0).unwrap());
///
/// // Prior for precision parameter
/// let precision = sample(addr!("precision"), Gamma::new(2.0, 1.0).unwrap());
///
/// // Conjugate prior for Poisson rate
/// let model = sample(addr!("rate"), Gamma::new(3.0, 2.0).unwrap())
///     .bind(|lambda| observe(addr!("count"), Poisson::new(lambda).unwrap(), 5u64));
///
/// // Scale parameter (rate = 1/scale)
/// let scale_param = sample(addr!("scale"), Gamma::new(2.0, 0.5).unwrap()); // mean = 4
/// ```
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
                crate::error::ErrorCode::InvalidShape,
            )
            .with_context("shape", format!("{}", shape))
            .with_context("expected", "> 0.0 and finite"));
        }
        if rate <= 0.0 || !rate.is_finite() {
            return Err(crate::error::FugueError::invalid_parameters(
                "Gamma",
                "Rate parameter must be positive and finite",
                crate::error::ErrorCode::InvalidRate,
            )
            .with_context("rate", format!("{}", rate))
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

/// A discrete distribution representing the number of successes in n independent trials, with probability of success p.
///
/// Returns `u64` for natural success counting.
///
/// Mathematical Properties:
/// - **Support**: {0, 1, ..., n}
/// - **PMF**: P(X = k) = C(n,k) × p^k × (1-p)^(n-k)
/// - **Mean**: n × p
/// - **Variance**: n × p × (1-p)
///
/// Example:
/// ```rust
/// # use fugue::*;
///
/// // 10 coin flips
/// let successes = sample(addr!("heads"), Binomial::new(10, 0.5).unwrap())
///     .bind(|count| {
///         let rate = count as f64 / 10.0;
///         pure(format!("Success rate: {:.1}%", rate * 100.0))
///     });
///
/// // Clinical trial
/// let trial = sample(addr!("success_rate"), Beta::new(1.0, 1.0).unwrap())
///     .bind(|p| sample(addr!("successes"), Binomial::new(100, p).unwrap()));
///
/// // Observe trial results
/// let observed = observe(addr!("trial_successes"), Binomial::new(20, 0.3).unwrap(), 7u64);
/// ```
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
                crate::error::ErrorCode::InvalidProbability,
            )
            .with_context("p", format!("{}", p))
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

/// A discrete distribution for modeling the number of events occurring in a fixed interval.
///
/// Returns `u64` for natural counting arithmetic.
///
/// Mathematical Properties:
/// - **Support**: {0, 1, 2, 3, ...}
/// - **PMF**: P(X = k) = (λ^k × e^(-λ)) / k!
/// - **Mean**: λ
/// - **Variance**: λ
/// - **Memoryless**: Past events don't affect future rates
///
/// Example:
/// ```rust
/// # use fugue::*;
///
/// // Model event counts
/// let events = sample(addr!("events"), Poisson::new(3.0).unwrap())
///     .bind(|count| {
///         let status = match count {
///             0 => "No events",
///             1 => "Single event",
///             n if n > 10 => "High activity",
///             _ => "Normal activity"
///         };
///         pure(status.to_string())
///     });
///
/// // Hierarchical model with Gamma prior
/// let hierarchical = sample(addr!("rate"), Gamma::new(2.0, 1.0).unwrap())
///     .bind(|lambda| sample(addr!("count"), Poisson::new(lambda).unwrap()));
///
/// // Observe count data
/// let observed = observe(addr!("observed_count"), Poisson::new(4.0).unwrap(), 7u64);
/// ```
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
                crate::error::ErrorCode::InvalidRate,
            )
            .with_context("lambda", format!("{}", lambda))
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
        assert!((-1.0..2.0).contains(&y));

        let b = Bernoulli::new(0.7).unwrap();
        let _z = b.sample(&mut rng);
    }

    #[test]
    fn categorical_uniform_constructor() {
        let cu = Categorical::uniform(4).unwrap();
        assert_eq!(cu.len(), 4);
        for &p in cu.probs() {
            assert!((p - 0.25).abs() < 1e-12);
        }
    }
}
