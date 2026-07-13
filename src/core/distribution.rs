#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/docs/core/distribution.md"))]
use rand::{Rng, RngCore};
use rand_distr::{
    Beta as RDBeta, Binomial as RDBinomial, Cauchy as RDCauchy, ChiSquared as RDChiSquared,
    Distribution as RandDistr, Exp as RDExp, Gamma as RDGamma, LogNormal as RDLogNormal,
    Normal as RDNormal, Poisson as RDPoisson, StudentT as RDStudentT, Weibull as RDWeibull,
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

    /// Create the standard normal distribution `N(0, 1)`.
    ///
    /// FG-29: infallible constructor for the statically-valid `mu = 0`,
    /// `sigma = 1` case, so common code does not need `new(...).unwrap()`.
    ///
    /// ```rust
    /// # use fugue::*;
    /// let z = Normal::standard();
    /// assert_eq!(z.mu(), 0.0);
    /// assert_eq!(z.sigma(), 1.0);
    /// ```
    pub fn standard() -> Self {
        Normal {
            mu: 0.0,
            sigma: 1.0,
        }
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

        // Numerically stable computation.
        //
        // FG-08: the log-density is computed entirely in log-space
        // (`-0.5·z² - ln(σ) - 0.5·ln(2π)`) and never evaluates `exp`, so it is
        // finite for every finite `z`. The previous `|z| > 37` short-circuit
        // returned `-inf` for perfectly finite densities (e.g. a tight-sigma
        // likelihood with a moderate residual), silently collapsing whole
        // models; it has been removed.
        let z = (x - self.mu) / self.sigma;

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

    /// Create the unit uniform distribution on `[0, 1)`.
    ///
    /// FG-29: infallible constructor for the statically-valid `low = 0`,
    /// `high = 1` case (the canonical uninformative prior over a probability),
    /// avoiding `new(0.0, 1.0).unwrap()`.
    ///
    /// ```rust
    /// # use fugue::*;
    /// let u = Uniform::unit();
    /// assert_eq!(u.low(), 0.0);
    /// assert_eq!(u.high(), 1.0);
    /// ```
    pub fn unit() -> Self {
        Uniform {
            low: 0.0,
            high: 1.0,
        }
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

        // Numerically stable computation.
        //
        // FG-08: like Normal, this is pure log-space and finite for any finite
        // standardized residual `z`; the old `|z| > 37` guard wrongly returned
        // `-inf` for finite densities (e.g. tight-sigma multiplicative error
        // models) and has been removed.
        let lx = x.ln();
        let z = (lx - self.mu) / self.sigma;

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
            // FG-30: `ln(λ) - λx` is computed entirely in log-space and is
            // finite for every finite `x` (`-λx` is just a subtraction, no
            // `exp`). The previous `rate*x > 700` short-circuit returned `-inf`
            // for finite tail log-densities and has been removed.
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

    /// Create a fair Bernoulli distribution (`p = 0.5`).
    ///
    /// FG-29: infallible constructor for the statically-valid fair-coin case,
    /// avoiding `new(0.5).unwrap()`.
    ///
    /// ```rust
    /// # use fugue::*;
    /// let coin = Bernoulli::fair();
    /// assert_eq!(coin.p(), 0.5);
    /// ```
    pub fn fair() -> Self {
        Bernoulli { p: 0.5 }
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
    /// Probabilities for each category (validated to sum to 1.0 in the constructor).
    probs: Vec<f64>,
    /// Cached inclusive cumulative distribution: `cumulative[i] = Σ probs[0..=i]`.
    ///
    /// FG-53: computed once at construction so `sample` can binary-search the CDF
    /// (O(log k)) and neither `sample` nor `log_prob` re-sums/re-validates the
    /// full probability vector on the hot inference path.
    cumulative: Vec<f64>,
}
impl Categorical {
    /// Build the inclusive cumulative distribution from a validated probability slice.
    fn compute_cumulative(probs: &[f64]) -> Vec<f64> {
        let mut cumulative = Vec::with_capacity(probs.len());
        let mut acc = 0.0;
        for &p in probs {
            acc += p;
            cumulative.push(acc);
        }
        cumulative
    }

    /// Validate a probability vector against the Categorical invariants
    /// (non-empty, every entry non-negative and finite, sum ≈ 1.0).
    fn validate_probs(probs: &[f64]) -> crate::error::FugueResult<()> {
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

        Ok(())
    }

    /// Create a new Categorical distribution with validated parameters.
    ///
    /// FG-53: the probability vector is validated exactly once here and the
    /// cumulative distribution is cached; `sample`/`log_prob` then rely on the
    /// established invariant instead of re-validating on every call.
    pub fn new(probs: Vec<f64>) -> crate::error::FugueResult<Self> {
        Self::validate_probs(&probs)?;
        let cumulative = Self::compute_cumulative(&probs);
        Ok(Categorical { probs, cumulative })
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
        let cumulative = Self::compute_cumulative(&probs);
        Ok(Categorical { probs, cumulative })
    }

    /// Re-check the constructor invariants on the cached probability vector.
    ///
    /// The public constructors ([`Categorical::new`]/[`Categorical::uniform`])
    /// already guarantee these invariants, so this is only needed if a
    /// `Categorical` is obtained through some future unchecked path (e.g.
    /// deserialization) and the caller wants to reassert validity.
    pub fn revalidate(&self) -> crate::error::FugueResult<()> {
        Self::validate_probs(&self.probs)
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
        // FG-53: the probability vector was validated once at construction, so
        // no per-call re-sum/re-scan is needed. Draw u ~ Uniform[0,1) and binary
        // search the cached CDF for the first index i with cumulative[i] >= u —
        // the exact same mapping the previous linear scan produced, in O(log k).
        if self.cumulative.is_empty() {
            return 0;
        }

        use rand::Rng;
        let u: f64 = rng.gen();
        let idx = self.cumulative.partition_point(|&c| c < u);
        idx.min(self.probs.len() - 1)
    }
    fn log_prob(&self, x: &usize) -> LogF64 {
        // FG-53: bounds-checked index into the validated probability vector.
        match self.probs.get(*x) {
            Some(&p) if p > 0.0 => p.ln(),
            _ => f64::NEG_INFINITY,
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
/// - **Support**: (0, 1); the closed endpoints 0 and 1 are handled as limits
/// - **PDF**: f(x) = (x^(α-1) × (1-x)^(β-1)) / B(α,β)
/// - **Mean**: α / (α + β)
/// - **Variance**: (αβ) / ((α+β)²(α+β+1))
///
/// Boundary semantics (matching `scipy.stats.beta.logpdf`): at `x = 0`,
/// `log_prob` is `-∞` when `α > 1` (density → 0), `ln(β)` when `α == 1`, and
/// `+∞` when `α < 1` (density diverges, e.g. the Jeffreys prior Beta(0.5, 0.5)).
/// The endpoint `x = 1` is symmetric in `β`.
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

    /// Create the uniform-prior Beta distribution `Beta(1, 1)`.
    ///
    /// FG-29: infallible constructor for the statically-valid `α = β = 1` case,
    /// which is exactly the uniform distribution on `(0, 1)` and the standard
    /// uninformative conjugate prior for a Bernoulli/Binomial probability;
    /// avoids `new(1.0, 1.0).unwrap()`.
    ///
    /// ```rust
    /// # use fugue::*;
    /// let prior = Beta::uniform_prior();
    /// assert_eq!(prior.alpha(), 1.0);
    /// assert_eq!(prior.beta(), 1.0);
    /// ```
    pub fn uniform_prior() -> Self {
        Beta {
            alpha: 1.0,
            beta: 1.0,
        }
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

        let x = *x;

        // Outside the closed support [0, 1] the density is 0.
        if !(0.0..=1.0).contains(&x) {
            return f64::NEG_INFINITY;
        }

        // log B(α, β), the (log) normalizing constant.
        let log_beta_fn = libm::lgamma(self.alpha) + libm::lgamma(self.beta)
            - libm::lgamma(self.alpha + self.beta);

        // FG-27: boundary limits matching `scipy.stats.beta.logpdf`. The density
        // behaves like x^(α-1) at 0 and (1-x)^(β-1) at 1, so at each endpoint:
        //   - shape param > 1  ⇒ density → 0   ⇒ -inf
        //   - shape param == 1 ⇒ density finite ⇒ the finite limit (-log B)
        //   - shape param < 1  ⇒ density → ∞   ⇒ +inf
        // The previous `1e-100`/`ln < -700` cutoffs returned -inf here even where
        // the true log-density is a large *positive* number (e.g. Jeffreys prior
        // Beta(0.5,0.5)), which is wrong in sign, not merely over-conservative.
        if x == 0.0 {
            return if self.alpha > 1.0 {
                f64::NEG_INFINITY
            } else if self.alpha < 1.0 {
                f64::INFINITY
            } else {
                // α == 1: (α-1)·ln(x) = 0 and (β-1)·ln(1) = 0, so log_prob = -log B(1,β) = ln(β).
                -log_beta_fn
            };
        }
        if x == 1.0 {
            return if self.beta > 1.0 {
                f64::NEG_INFINITY
            } else if self.beta < 1.0 {
                f64::INFINITY
            } else {
                // β == 1: log_prob = -log B(α,1) = ln(α).
                -log_beta_fn
            };
        }

        // Interior x ∈ (0, 1): computed exactly with no ln guards. f64::ln
        // handles subnormals fine, and for α<1 (or β<1) near a boundary the
        // (α-1)·ln(x) term correctly diverges to +∞ rather than being clipped.
        // log Beta(x; α, β) = (α-1)·ln(x) + (β-1)·ln(1-x) - log B(α, β)
        let ln_x = x.ln();
        let ln_1_minus_x = (1.0 - x).ln();

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

        // FG-07: the formula below is pure log-space and never evaluates
        // `exp`, so `-λx` and `(k-1)·ln(x)` are finite for every `x > 0`. The
        // previous `rate*x > 700` / `ln(x)·(k-1) < -700` guards returned `-inf`
        // across the entire high-density region (including the mode) of any
        // Gamma with mean ≳ 700, silently zeroing large-shape posteriors. They
        // have been removed; only the genuine `x <= 0` support check remains.
        //
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
        // Parameter validation (defensive; `new` already enforces p ∈ [0, 1]).
        if !self.p.is_finite() || !(0.0..=1.0).contains(&self.p) {
            return f64::NEG_INFINITY;
        }
        let k = *x;
        if k > self.n {
            return f64::NEG_INFINITY;
        }

        // FG-28: `new` accepts the degenerate boundaries p = 0 and p = 1, which
        // are valid parameters. Evaluating the general formula there produces
        // `0 * ln(0) = 0 * -inf = NaN`, which is materially worse than -inf
        // because it poisons every downstream comparison. Handle them exactly:
        // p = 0 puts all mass on k = 0, p = 1 puts all mass on k = n.
        if self.p == 0.0 {
            return if k == 0 { 0.0 } else { f64::NEG_INFINITY };
        }
        if self.p == 1.0 {
            return if k == self.n { 0.0 } else { f64::NEG_INFINITY };
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

// =============================================================================
// FG-31: seven additional univariate distributions.
//
// Each follows the established style exactly: a validating `new` constructor
// returning `FugueResult`, natural `f64`/`i64` return types, and a `log_prob`
// that carries the FULL normalizing constant (no dropped `lgamma`/`ln` terms).
// Samplers use `rand_distr` where a matching generator exists and an exact
// inverse-CDF / reciprocal-Gamma construction otherwise. The closed-form
// `log_prob` expressions match `scipy.stats.<dist>.logpdf` (constants in the
// tests were derived from those closed forms).
// =============================================================================

/// Student's t-distribution with a location and scale, `StudentT(ν, μ, σ)`.
///
/// Heavy-tailed generalization of the Normal; as `ν → ∞` it converges to
/// `Normal(μ, σ)`. Widely used as a robust likelihood/prior because its tails
/// tolerate outliers. `ν` need not be an integer.
///
/// Mathematical Properties:
/// - **Support**: (-∞, +∞)
/// - **PDF**: f(x) = Γ((ν+1)/2) / (Γ(ν/2)·√(νπ)·σ) · (1 + z²/ν)^(-(ν+1)/2),
///   where z = (x−μ)/σ
/// - **Mean**: μ for ν > 1 (undefined otherwise)
/// - **Variance**: σ²·ν/(ν−2) for ν > 2 (infinite for 1 < ν ≤ 2)
///
/// Example:
/// ```rust
/// # use fugue::*;
/// // Robust prior with 3 degrees of freedom.
/// let robust = sample(addr!("theta"), StudentT::new(3.0, 0.0, 1.0).unwrap());
/// // Robust likelihood tolerant of outliers.
/// let obs = observe(addr!("y"), StudentT::new(4.0, 1.0, 0.5).unwrap(), 2.0);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct StudentT {
    /// Degrees of freedom ν (must be positive).
    df: f64,
    /// Location parameter μ.
    loc: f64,
    /// Scale parameter σ (must be positive).
    scale: f64,
}
impl StudentT {
    /// Create a new Student's t-distribution with validated parameters.
    pub fn new(df: f64, loc: f64, scale: f64) -> crate::error::FugueResult<Self> {
        if df <= 0.0 || !df.is_finite() {
            return Err(crate::error::FugueError::invalid_parameters(
                "StudentT",
                "Degrees of freedom must be positive and finite",
                crate::error::ErrorCode::InvalidShape,
            )
            .with_context("df", format!("{}", df))
            .with_context("expected", "> 0.0 and finite"));
        }
        if !loc.is_finite() {
            return Err(crate::error::FugueError::invalid_parameters(
                "StudentT",
                "Location (loc) must be finite",
                crate::error::ErrorCode::InvalidMean,
            )
            .with_context("loc", format!("{}", loc)));
        }
        if scale <= 0.0 || !scale.is_finite() {
            return Err(crate::error::FugueError::invalid_parameters(
                "StudentT",
                "Scale must be positive and finite",
                crate::error::ErrorCode::InvalidVariance,
            )
            .with_context("scale", format!("{}", scale))
            .with_context("expected", "> 0.0 and finite"));
        }
        Ok(StudentT { df, loc, scale })
    }

    /// Get the degrees of freedom ν.
    pub fn df(&self) -> f64 {
        self.df
    }

    /// Get the location parameter μ.
    pub fn loc(&self) -> f64 {
        self.loc
    }

    /// Get the scale parameter σ.
    pub fn scale(&self) -> f64 {
        self.scale
    }
}
impl Distribution<f64> for StudentT {
    fn sample(&self, rng: &mut dyn RngCore) -> f64 {
        if self.df <= 0.0 || self.scale <= 0.0 {
            return f64::NAN;
        }
        // rand_distr's StudentT is standardized (location 0, scale 1); apply the
        // affine location-scale transform.
        let t = RDStudentT::new(self.df).unwrap().sample(rng);
        self.loc + self.scale * t
    }
    fn log_prob(&self, x: &f64) -> LogF64 {
        if self.df <= 0.0
            || self.scale <= 0.0
            || !self.df.is_finite()
            || !self.scale.is_finite()
            || !self.loc.is_finite()
            || !x.is_finite()
        {
            return f64::NEG_INFINITY;
        }
        const LN_PI: f64 = 1.144_729_885_849_400_2; // ln(π)
        let z = (x - self.loc) / self.scale;
        // log f = lnΓ((ν+1)/2) − lnΓ(ν/2) − 0.5·ln(νπ) − ln(σ)
        //         − ((ν+1)/2)·ln(1 + z²/ν)
        libm::lgamma((self.df + 1.0) / 2.0)
            - libm::lgamma(self.df / 2.0)
            - 0.5 * (self.df.ln() + LN_PI)
            - self.scale.ln()
            - 0.5 * (self.df + 1.0) * (z * z / self.df).ln_1p()
    }
    fn clone_box(&self) -> Box<dyn Distribution<f64>> {
        Box::new(*self)
    }
}

/// The Cauchy (Lorentz) distribution `Cauchy(x₀, γ)`.
///
/// The heavy-tailed limit `StudentT(1, x₀, γ)`. It has **no** finite mean or
/// variance; `x₀` is the median/mode and `γ` the half-width at half-maximum.
///
/// Mathematical Properties:
/// - **Support**: (-∞, +∞)
/// - **PDF**: f(x) = 1 / (πγ·(1 + ((x−x₀)/γ)²))
/// - **Mean/Variance**: undefined (heavy tails)
/// - **Median/Mode**: x₀
///
/// Example:
/// ```rust
/// # use fugue::*;
/// // Weakly-informative heavy-tailed prior.
/// let prior = sample(addr!("beta"), Cauchy::new(0.0, 2.5).unwrap());
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Cauchy {
    /// Location (median) parameter x₀.
    loc: f64,
    /// Scale parameter γ (must be positive).
    scale: f64,
}
impl Cauchy {
    /// Create a new Cauchy distribution with validated parameters.
    pub fn new(loc: f64, scale: f64) -> crate::error::FugueResult<Self> {
        if !loc.is_finite() {
            return Err(crate::error::FugueError::invalid_parameters(
                "Cauchy",
                "Location (loc) must be finite",
                crate::error::ErrorCode::InvalidMean,
            )
            .with_context("loc", format!("{}", loc)));
        }
        if scale <= 0.0 || !scale.is_finite() {
            return Err(crate::error::FugueError::invalid_parameters(
                "Cauchy",
                "Scale must be positive and finite",
                crate::error::ErrorCode::InvalidVariance,
            )
            .with_context("scale", format!("{}", scale))
            .with_context("expected", "> 0.0 and finite"));
        }
        Ok(Cauchy { loc, scale })
    }

    /// Get the location (median) parameter x₀.
    pub fn loc(&self) -> f64 {
        self.loc
    }

    /// Get the scale parameter γ.
    pub fn scale(&self) -> f64 {
        self.scale
    }
}
impl Distribution<f64> for Cauchy {
    fn sample(&self, rng: &mut dyn RngCore) -> f64 {
        if self.scale <= 0.0 {
            return f64::NAN;
        }
        RDCauchy::new(self.loc, self.scale).unwrap().sample(rng)
    }
    fn log_prob(&self, x: &f64) -> LogF64 {
        if self.scale <= 0.0 || !self.scale.is_finite() || !self.loc.is_finite() || !x.is_finite() {
            return f64::NEG_INFINITY;
        }
        const LN_PI: f64 = 1.144_729_885_849_400_2; // ln(π)
        let z = (x - self.loc) / self.scale;
        // log f = −ln(π) − ln(γ) − ln(1 + z²)
        -LN_PI - self.scale.ln() - (z * z).ln_1p()
    }
    fn clone_box(&self) -> Box<dyn Distribution<f64>> {
        Box::new(*self)
    }
}

/// The Laplace (double-exponential) distribution `Laplace(μ, b)`.
///
/// A symmetric distribution with a sharp peak at `μ` and exponential tails;
/// its log-density is `−|x−μ|/b` up to a constant, which is why it underlies
/// L1/LASSO-style priors.
///
/// Mathematical Properties:
/// - **Support**: (-∞, +∞)
/// - **PDF**: f(x) = (1/(2b))·exp(−|x−μ|/b)
/// - **Mean**: μ
/// - **Variance**: 2b²
///
/// Example:
/// ```rust
/// # use fugue::*;
/// // Sparsity-inducing prior on a coefficient.
/// let coef = sample(addr!("w"), Laplace::new(0.0, 1.0).unwrap());
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Laplace {
    /// Location (mean) parameter μ.
    loc: f64,
    /// Scale parameter b (must be positive).
    scale: f64,
}
impl Laplace {
    /// Create a new Laplace distribution with validated parameters.
    pub fn new(loc: f64, scale: f64) -> crate::error::FugueResult<Self> {
        if !loc.is_finite() {
            return Err(crate::error::FugueError::invalid_parameters(
                "Laplace",
                "Location (loc) must be finite",
                crate::error::ErrorCode::InvalidMean,
            )
            .with_context("loc", format!("{}", loc)));
        }
        if scale <= 0.0 || !scale.is_finite() {
            return Err(crate::error::FugueError::invalid_parameters(
                "Laplace",
                "Scale must be positive and finite",
                crate::error::ErrorCode::InvalidVariance,
            )
            .with_context("scale", format!("{}", scale))
            .with_context("expected", "> 0.0 and finite"));
        }
        Ok(Laplace { loc, scale })
    }

    /// Get the location (mean) parameter μ.
    pub fn loc(&self) -> f64 {
        self.loc
    }

    /// Get the scale parameter b.
    pub fn scale(&self) -> f64 {
        self.scale
    }
}
impl Distribution<f64> for Laplace {
    fn sample(&self, rng: &mut dyn RngCore) -> f64 {
        if self.scale <= 0.0 {
            return f64::NAN;
        }
        // Exact inverse-CDF sampling (rand_distr has no Laplace generator):
        // draw u ∈ (−½, ½) and map through the quantile function. The sign of u
        // picks the tail and −b·sign(u)·ln(1 − 2|u|) is the corresponding
        // exponential deviate.
        let u: f64 = rng.gen::<f64>() - 0.5;
        self.loc - self.scale * u.signum() * (1.0 - 2.0 * u.abs()).ln()
    }
    fn log_prob(&self, x: &f64) -> LogF64 {
        if self.scale <= 0.0 || !self.scale.is_finite() || !self.loc.is_finite() || !x.is_finite() {
            return f64::NEG_INFINITY;
        }
        // log f = −ln(2b) − |x−μ|/b
        -(2.0 * self.scale).ln() - (x - self.loc).abs() / self.scale
    }
    fn clone_box(&self) -> Box<dyn Distribution<f64>> {
        Box::new(*self)
    }
}

/// The Weibull distribution `Weibull(k, λ)` with shape `k` and scale `λ`.
///
/// A flexible positive distribution used for reliability/survival modeling;
/// `k < 1` is a decreasing hazard, `k = 1` is the Exponential, and `k > 1` is
/// an increasing hazard.
///
/// Mathematical Properties:
/// - **Support**: [0, +∞)
/// - **PDF**: f(x) = (k/λ)·(x/λ)^(k−1)·exp(−(x/λ)^k) for x ≥ 0
/// - **Mean**: λ·Γ(1 + 1/k)
/// - **Variance**: λ²·[Γ(1 + 2/k) − Γ(1 + 1/k)²]
///
/// Boundary semantics (matching `scipy.stats.weibull_min.logpdf`): at `x = 0`,
/// `log_prob` is `−∞` when `k > 1`, `−ln(λ)` when `k == 1`, and `+∞` when
/// `k < 1`.
///
/// Example:
/// ```rust
/// # use fugue::*;
/// // Time-to-failure prior with increasing hazard.
/// let ttf = sample(addr!("t"), Weibull::new(1.5, 2.0).unwrap());
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Weibull {
    /// Shape parameter k (must be positive).
    shape: f64,
    /// Scale parameter λ (must be positive).
    scale: f64,
}
impl Weibull {
    /// Create a new Weibull distribution with validated parameters.
    pub fn new(shape: f64, scale: f64) -> crate::error::FugueResult<Self> {
        if shape <= 0.0 || !shape.is_finite() {
            return Err(crate::error::FugueError::invalid_parameters(
                "Weibull",
                "Shape parameter must be positive and finite",
                crate::error::ErrorCode::InvalidShape,
            )
            .with_context("shape", format!("{}", shape))
            .with_context("expected", "> 0.0 and finite"));
        }
        if scale <= 0.0 || !scale.is_finite() {
            return Err(crate::error::FugueError::invalid_parameters(
                "Weibull",
                "Scale parameter must be positive and finite",
                crate::error::ErrorCode::InvalidVariance,
            )
            .with_context("scale", format!("{}", scale))
            .with_context("expected", "> 0.0 and finite"));
        }
        Ok(Weibull { shape, scale })
    }

    /// Get the shape parameter k.
    pub fn shape(&self) -> f64 {
        self.shape
    }

    /// Get the scale parameter λ.
    pub fn scale(&self) -> f64 {
        self.scale
    }
}
impl Distribution<f64> for Weibull {
    fn sample(&self, rng: &mut dyn RngCore) -> f64 {
        if self.shape <= 0.0 || self.scale <= 0.0 {
            return f64::NAN;
        }
        // rand_distr::Weibull::new takes (scale, shape) in that order.
        RDWeibull::new(self.scale, self.shape).unwrap().sample(rng)
    }
    fn log_prob(&self, x: &f64) -> LogF64 {
        if self.shape <= 0.0
            || self.scale <= 0.0
            || !self.shape.is_finite()
            || !self.scale.is_finite()
            || !x.is_finite()
        {
            return f64::NEG_INFINITY;
        }
        let x = *x;
        if x < 0.0 {
            return f64::NEG_INFINITY;
        }
        if x == 0.0 {
            // Endpoint limit of (x/λ)^(k−1): k>1 ⇒ 0, k==1 ⇒ 1/λ, k<1 ⇒ ∞.
            return if self.shape > 1.0 {
                f64::NEG_INFINITY
            } else if self.shape < 1.0 {
                f64::INFINITY
            } else {
                -self.scale.ln()
            };
        }
        // log f = ln(k) − k·ln(λ) + (k−1)·ln(x) − (x/λ)^k
        self.shape.ln() - self.shape * self.scale.ln() + (self.shape - 1.0) * x.ln()
            - (x / self.scale).powf(self.shape)
    }
    fn clone_box(&self) -> Box<dyn Distribution<f64>> {
        Box::new(*self)
    }
}

/// The chi-squared distribution `ChiSquared(k)` with `k` degrees of freedom.
///
/// The distribution of a sum of `k` squared standard normals; the special case
/// `Gamma(k/2, 1/2)`. `k` need not be an integer.
///
/// Mathematical Properties:
/// - **Support**: (0, +∞)
/// - **PDF**: f(x) = 1/(2^(k/2)·Γ(k/2))·x^(k/2−1)·exp(−x/2)
/// - **Mean**: k
/// - **Variance**: 2k
///
/// Example:
/// ```rust
/// # use fugue::*;
/// // Sampling distribution of a scaled variance statistic.
/// let s = sample(addr!("s"), ChiSquared::new(4.0).unwrap());
/// ```
#[derive(Clone, Copy, Debug)]
pub struct ChiSquared {
    /// Degrees of freedom k (must be positive).
    k: f64,
}
impl ChiSquared {
    /// Create a new chi-squared distribution with validated parameters.
    pub fn new(k: f64) -> crate::error::FugueResult<Self> {
        if k <= 0.0 || !k.is_finite() {
            return Err(crate::error::FugueError::invalid_parameters(
                "ChiSquared",
                "Degrees of freedom must be positive and finite",
                crate::error::ErrorCode::InvalidShape,
            )
            .with_context("k", format!("{}", k))
            .with_context("expected", "> 0.0 and finite"));
        }
        Ok(ChiSquared { k })
    }

    /// Get the degrees of freedom k.
    pub fn k(&self) -> f64 {
        self.k
    }
}
impl Distribution<f64> for ChiSquared {
    fn sample(&self, rng: &mut dyn RngCore) -> f64 {
        if self.k <= 0.0 {
            return f64::NAN;
        }
        RDChiSquared::new(self.k).unwrap().sample(rng)
    }
    fn log_prob(&self, x: &f64) -> LogF64 {
        if self.k <= 0.0 || !self.k.is_finite() || !x.is_finite() {
            return f64::NEG_INFINITY;
        }
        if *x <= 0.0 {
            return f64::NEG_INFINITY;
        }
        // log f = −(k/2)·ln(2) − lnΓ(k/2) + (k/2 − 1)·ln(x) − x/2
        let half_k = self.k / 2.0;
        -half_k * std::f64::consts::LN_2 - libm::lgamma(half_k) + (half_k - 1.0) * x.ln() - x / 2.0
    }
    fn clone_box(&self) -> Box<dyn Distribution<f64>> {
        Box::new(*self)
    }
}

/// The inverse-gamma distribution `InverseGamma(α, β)` with shape `α` and rate
/// `β`.
///
/// If `X ~ InverseGamma(α, β)` then `1/X ~ Gamma(α, rate = β)` — hence the
/// second parameter is named `rate` to parallel [`Gamma`]. It is the standard
/// conjugate prior for the variance of a Normal.
///
/// Mathematical Properties:
/// - **Support**: (0, +∞)
/// - **PDF**: f(x) = β^α/Γ(α)·x^(−α−1)·exp(−β/x)
/// - **Mean**: β/(α−1) for α > 1
/// - **Variance**: β²/((α−1)²(α−2)) for α > 2
///
/// This matches `scipy.stats.invgamma.logpdf(x, a = α, scale = β)`.
///
/// Example:
/// ```rust
/// # use fugue::*;
/// // Conjugate prior for an unknown variance.
/// let var = sample(addr!("sigma2"), InverseGamma::new(3.0, 2.0).unwrap());
/// ```
#[derive(Clone, Copy, Debug)]
pub struct InverseGamma {
    /// Shape parameter α (must be positive).
    shape: f64,
    /// Rate parameter β (must be positive).
    rate: f64,
}
impl InverseGamma {
    /// Create a new inverse-gamma distribution with validated parameters.
    pub fn new(shape: f64, rate: f64) -> crate::error::FugueResult<Self> {
        if shape <= 0.0 || !shape.is_finite() {
            return Err(crate::error::FugueError::invalid_parameters(
                "InverseGamma",
                "Shape parameter must be positive and finite",
                crate::error::ErrorCode::InvalidShape,
            )
            .with_context("shape", format!("{}", shape))
            .with_context("expected", "> 0.0 and finite"));
        }
        if rate <= 0.0 || !rate.is_finite() {
            return Err(crate::error::FugueError::invalid_parameters(
                "InverseGamma",
                "Rate parameter must be positive and finite",
                crate::error::ErrorCode::InvalidRate,
            )
            .with_context("rate", format!("{}", rate))
            .with_context("expected", "> 0.0 and finite"));
        }
        Ok(InverseGamma { shape, rate })
    }

    /// Get the shape parameter α.
    pub fn shape(&self) -> f64 {
        self.shape
    }

    /// Get the rate parameter β.
    pub fn rate(&self) -> f64 {
        self.rate
    }
}
impl Distribution<f64> for InverseGamma {
    fn sample(&self, rng: &mut dyn RngCore) -> f64 {
        if self.shape <= 0.0 || self.rate <= 0.0 {
            return f64::NAN;
        }
        // X = 1/Y with Y ~ Gamma(shape = α, rate = β). rand_distr::Gamma takes a
        // scale, so pass scale = 1/β.
        let y = RDGamma::new(self.shape, 1.0 / self.rate)
            .unwrap()
            .sample(rng);
        1.0 / y
    }
    fn log_prob(&self, x: &f64) -> LogF64 {
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
        // log f = α·ln(β) − lnΓ(α) − (α+1)·ln(x) − β/x
        self.shape * self.rate.ln()
            - libm::lgamma(self.shape)
            - (self.shape + 1.0) * x.ln()
            - self.rate / x
    }
    fn clone_box(&self) -> Box<dyn Distribution<f64>> {
        Box::new(*self)
    }
}

/// A discrete distribution assigning equal probability to every integer in an
/// inclusive range `[low, high]`, returning `i64`.
///
/// This is the first-class consumer of the `i64` sample path (`ChoiceValue::I64`
/// end-to-end through sample/observe/replay/score).
///
/// Mathematical Properties:
/// - **Support**: {low, low+1, ..., high}
/// - **PMF**: P(X = k) = 1/(high − low + 1) for low ≤ k ≤ high, 0 otherwise
/// - **Mean**: (low + high) / 2
/// - **Variance**: ((high − low + 1)² − 1) / 12
///
/// Example:
/// ```rust
/// # use fugue::*;
/// // A fair six-sided die labelled 1..=6.
/// let die = sample(addr!("die"), DiscreteUniform::new(1, 6).unwrap());
/// // Condition on an observed roll.
/// let obs = observe(addr!("roll"), DiscreteUniform::new(1, 6).unwrap(), 4i64);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct DiscreteUniform {
    /// Inclusive lower bound.
    low: i64,
    /// Inclusive upper bound (must satisfy `high >= low`).
    high: i64,
}
impl DiscreteUniform {
    /// Create a new discrete-uniform distribution over the inclusive range
    /// `[low, high]`.
    pub fn new(low: i64, high: i64) -> crate::error::FugueResult<Self> {
        if high < low {
            return Err(crate::error::FugueError::invalid_parameters(
                "DiscreteUniform",
                "Upper bound must be >= lower bound",
                crate::error::ErrorCode::InvalidRange,
            )
            .with_context("low", format!("{}", low))
            .with_context("high", format!("{}", high)));
        }
        Ok(DiscreteUniform { low, high })
    }

    /// Get the inclusive lower bound.
    pub fn low(&self) -> i64 {
        self.low
    }

    /// Get the inclusive upper bound.
    pub fn high(&self) -> i64 {
        self.high
    }

    /// Number of points in the support (`high − low + 1`).
    ///
    /// Exact for every range except the full `i64` domain, whose support has
    /// `2^64` points — one more than fits in `u64` — so `len()` **saturates to
    /// `u64::MAX`** for `DiscreteUniform::new(i64::MIN, i64::MAX)`. Sampling and
    /// scoring never round-trip through `len()`; they use the exact `u128`
    /// [`Self::count`], so the full-range case is handled correctly regardless.
    pub fn len(&self) -> u64 {
        u64::try_from(self.count()).unwrap_or(u64::MAX)
    }

    /// Exact number of support points as a `u128`.
    ///
    /// `high >= low` is a constructor invariant, so `high − low` ranges over
    /// `[0, 2^64 − 1]` and `+ 1` over `[1, 2^64]` — always representable in
    /// `u128`. Only the full `i64` domain reaches `2^64`.
    fn count(&self) -> u128 {
        (self.high as i128 - self.low as i128 + 1) as u128
    }

    /// Whether `[low, high]` spans the entire `i64` domain. This is the one range
    /// whose `2^64`-point support overflows a `u64` offset, so `sample`/`log_prob`
    /// special-case it.
    fn is_full_i64_range(&self) -> bool {
        self.low == i64::MIN && self.high == i64::MAX
    }

    /// Whether the support is empty. Always `false` for a validly-constructed
    /// distribution (kept for clippy's `len`/`is_empty` pairing).
    pub fn is_empty(&self) -> bool {
        false
    }
}
impl Distribution<i64> for DiscreteUniform {
    fn sample(&self, rng: &mut dyn RngCore) -> i64 {
        if self.high < self.low {
            return self.low;
        }
        if self.is_full_i64_range() {
            // The support IS the whole i64 domain, so a raw uniform i64 draw is
            // already a uniform sample over [low, high]. The offset arithmetic
            // below is unusable here: the count is 2^64, which does not fit in the
            // u64 that `gen_range` needs.
            return Rng::gen::<i64>(rng);
        }
        // The range is not full, so the count fits in u64. Draw an offset in
        // [0, n) and shift; the shift is done in i128 to avoid any overflow at the
        // extremes of the i64 range.
        let n = self.count() as u64;
        let offset = Rng::gen_range(rng, 0..n) as i128;
        (self.low as i128 + offset) as i64
    }
    fn log_prob(&self, x: &i64) -> LogF64 {
        if self.high < self.low {
            return f64::NEG_INFINITY;
        }
        if *x < self.low || *x > self.high {
            return f64::NEG_INFINITY;
        }
        // log P = −ln(n). For the full i64 domain n = 2^64, whose logarithm is
        // exactly 64·ln 2; computing it directly is both exact and avoids the
        // `2^64 as f64` round-trip.
        if self.is_full_i64_range() {
            -(64.0 * std::f64::consts::LN_2)
        } else {
            -(self.count() as f64).ln()
        }
    }
    fn clone_box(&self) -> Box<dyn Distribution<i64>> {
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

    // Helper: assert closeness with 1e-9 tolerance.
    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-9, "expected {b}, got {a}");
    }

    #[test]
    fn fg06_interior_point_known_answers() {
        // Interior-point closed-form checks (scipy-equivalent constants).
        close(
            Normal::new(0.0, 1.0).unwrap().log_prob(&0.0),
            -0.9189385332046727,
        );
        close(
            Normal::new(1.0, 2.0).unwrap().log_prob(&2.5),
            -1.893335713764618,
        );
        close(
            Uniform::new(-2.0, 2.0).unwrap().log_prob(&1.5),
            -1.3862943611198906,
        );
        close(
            LogNormal::new(0.0, 1.0).unwrap().log_prob(&2.0),
            -1.8523122207237186,
        );
        close(
            Exponential::new(2.0).unwrap().log_prob(&1.0),
            -1.3068528194400546,
        );
        close(
            Beta::new(2.0, 3.0).unwrap().log_prob(&0.5),
            0.4054651081081637,
        );
        close(
            Gamma::new(3.0, 2.0).unwrap().log_prob(&1.5),
            -0.8027754226637804,
        );
        close(
            Binomial::new(20, 0.3).unwrap().log_prob(&7),
            -1.8062926549204255,
        );
        close(Poisson::new(3.0).unwrap().log_prob(&2), -1.4959226032237254);
        close(
            Categorical::new(vec![0.2, 0.3, 0.5]).unwrap().log_prob(&2),
            -std::f64::consts::LN_2, // ln(0.5) = -ln(2)
        );
    }

    #[test]
    fn fg07_fg08_fg30_removed_overflow_guards_return_finite() {
        // Each point was previously forced to -inf by a bogus overflow guard.
        close(
            Gamma::new(2.0, 1.0).unwrap().log_prob(&800.0),
            -793.315388272332,
        ); // FG-07
        close(
            Normal::new(0.0, 0.001).unwrap().log_prob(&0.05),
            -1244.0111832542225,
        ); // FG-08
        close(
            LogNormal::new(0.0, 0.001).unwrap().log_prob(&1.05),
            -1184.3000332584572,
        ); // FG-08
        close(
            Exponential::new(2.0).unwrap().log_prob(&400.0),
            -799.3068528194401,
        ); // FG-30
    }

    #[test]
    fn fg27_beta_boundaries() {
        // Subnormal interior no longer clipped to -inf.
        close(
            Beta::new(0.5, 0.5).unwrap().log_prob(&1e-100),
            113.98452476385289,
        );
        // Endpoint limits.
        close(
            Beta::new(1.0, 5.0).unwrap().log_prob(&0.0),
            1.6094379124341003,
        ); // ln(5)
        close(
            Beta::new(3.0, 1.0).unwrap().log_prob(&1.0),
            1.0986122886681098,
        ); // ln(3)
        assert_eq!(
            Beta::new(2.0, 5.0).unwrap().log_prob(&0.0),
            f64::NEG_INFINITY
        );
        assert_eq!(Beta::new(0.5, 3.0).unwrap().log_prob(&0.0), f64::INFINITY);
    }

    #[test]
    fn fg28_binomial_degenerate_p_not_nan() {
        let b0 = Binomial::new(5, 0.0).unwrap();
        assert!(!b0.log_prob(&0).is_nan());
        close(b0.log_prob(&0), 0.0);
        assert_eq!(b0.log_prob(&1), f64::NEG_INFINITY);
        let b1 = Binomial::new(5, 1.0).unwrap();
        assert!(!b1.log_prob(&5).is_nan());
        close(b1.log_prob(&5), 0.0);
        assert_eq!(b1.log_prob(&3), f64::NEG_INFINITY);
    }

    #[test]
    fn fg29_infallible_constructors() {
        assert_eq!(
            (Normal::standard().mu(), Normal::standard().sigma()),
            (0.0, 1.0)
        );
        assert_eq!((Uniform::unit().low(), Uniform::unit().high()), (0.0, 1.0));
        assert_eq!(
            (Beta::uniform_prior().alpha(), Beta::uniform_prior().beta()),
            (1.0, 1.0)
        );
        assert_eq!(Bernoulli::fair().p(), 0.5);
    }

    #[test]
    fn fg53_categorical_cached_cdf_and_revalidate() {
        let c = Categorical::new(vec![0.1, 0.2, 0.3, 0.4]).unwrap();
        close(c.log_prob(&3), (0.4f64).ln());
        assert_eq!(c.log_prob(&4), f64::NEG_INFINITY);
        assert!(c.revalidate().is_ok());

        // Seeded binary-search sampling stays in-range and roughly matches probs.
        let mut rng = StdRng::seed_from_u64(7);
        let mut counts = [0usize; 4];
        let n = 40_000usize;
        for _ in 0..n {
            counts[c.sample(&mut rng)] += 1;
        }
        for (k, &p) in [0.1, 0.2, 0.3, 0.4].iter().enumerate() {
            // ~1.1e-2 is > 7 std for the tightest bin at N = 40_000.
            assert!((counts[k] as f64 / n as f64 - p).abs() < 1.1e-2);
        }
    }

    // -------------------------------------------------------------------------
    // FG-31: the seven new distributions.
    // -------------------------------------------------------------------------

    // FG-31: interior-point log_prob against the scipy-equivalent closed forms.
    // Constants were derived with the standard log-pdf expressions in python3
    // (math.lgamma), identical to `scipy.stats.<dist>.logpdf`.
    #[test]
    fn fg31_new_distributions_interior_point_log_prob() {
        // scipy: stats.t.logpdf(2.5, 3, 1, 2)
        close(
            StudentT::new(3.0, 1.0, 2.0).unwrap().log_prob(&2.5),
            -2.0377365440367736,
        );
        // scipy: stats.t.logpdf(0.0, 5, 0, 1)
        close(
            StudentT::new(5.0, 0.0, 1.0).unwrap().log_prob(&0.0),
            -0.9686195890547249,
        );
        // scipy: stats.t.logpdf(1.0, 10, 2, 0.5)
        close(
            StudentT::new(10.0, 2.0, 0.5).unwrap().log_prob(&1.0),
            -2.1013474730076767,
        );
        // scipy: stats.cauchy.logpdf(1.5, 0, 1)
        close(
            Cauchy::new(0.0, 1.0).unwrap().log_prob(&1.5),
            -2.3233848821910463,
        );
        // scipy: stats.cauchy.logpdf(5.0, 2, 3)
        close(
            Cauchy::new(2.0, 3.0).unwrap().log_prob(&5.0),
            -2.9364893550774553,
        );
        // scipy: stats.laplace.logpdf(1.5, 0, 1)
        close(
            Laplace::new(0.0, 1.0).unwrap().log_prob(&1.5),
            -2.1931471805599454,
        );
        // scipy: stats.laplace.logpdf(-0.5, 1, 2)
        close(
            Laplace::new(1.0, 2.0).unwrap().log_prob(&-0.5),
            -2.136294361119891,
        );
        // scipy: stats.weibull_min.logpdf(1.0, 1.5, scale=2)
        close(
            Weibull::new(1.5, 2.0).unwrap().log_prob(&1.0),
            -0.9878090533250272,
        );
        // scipy: stats.weibull_min.logpdf(2.0, 2.0, scale=1.5)
        close(
            Weibull::new(2.0, 1.5).unwrap().log_prob(&2.0),
            -1.2024136328742159,
        );
        // scipy: stats.chi2.logpdf(3.0, 4)
        close(
            ChiSquared::new(4.0).unwrap().log_prob(&3.0),
            -1.7876820724517808,
        );
        // scipy: stats.chi2.logpdf(0.5, 1)
        close(
            ChiSquared::new(1.0).unwrap().log_prob(&0.5),
            -0.8223649429247004,
        );
        // scipy: stats.chi2.logpdf(2.0, 2.5)
        close(
            ChiSquared::new(2.5).unwrap().log_prob(&2.0),
            -1.5948753441381327,
        );
        // scipy: stats.invgamma.logpdf(1.5, 3, scale=2)
        close(
            InverseGamma::new(3.0, 2.0).unwrap().log_prob(&1.5),
            -1.5688994046461,
        );
        // scipy: stats.invgamma.logpdf(0.5, 2, scale=1)
        close(
            InverseGamma::new(2.0, 1.0).unwrap().log_prob(&0.5),
            0.07944154167983575,
        );
        // DiscreteUniform over {-2,...,5}: 8 points, log P = -ln(8).
        close(
            DiscreteUniform::new(-2, 5).unwrap().log_prob(&0),
            -2.0794415416798357,
        );
    }

    // FG-31: constructor validation and support/boundary behavior.
    #[test]
    fn fg31_new_distributions_validation_and_support() {
        // Constructor validation.
        assert!(StudentT::new(0.0, 0.0, 1.0).is_err()); // df must be > 0
        assert!(StudentT::new(3.0, f64::NAN, 1.0).is_err());
        assert!(StudentT::new(3.0, 0.0, 0.0).is_err()); // scale must be > 0
        assert!(Cauchy::new(0.0, -1.0).is_err());
        assert!(Cauchy::new(f64::INFINITY, 1.0).is_err());
        assert!(Laplace::new(0.0, 0.0).is_err());
        assert!(Weibull::new(0.0, 1.0).is_err());
        assert!(Weibull::new(1.0, 0.0).is_err());
        assert!(ChiSquared::new(0.0).is_err());
        assert!(ChiSquared::new(-1.0).is_err());
        assert!(InverseGamma::new(0.0, 1.0).is_err());
        assert!(InverseGamma::new(1.0, 0.0).is_err());
        assert!(DiscreteUniform::new(5, 4).is_err()); // high < low

        // Support boundaries.
        assert_eq!(
            Weibull::new(2.0, 1.0).unwrap().log_prob(&-0.5),
            f64::NEG_INFINITY
        );
        // Weibull endpoint limits at x = 0.
        assert_eq!(
            Weibull::new(2.0, 1.0).unwrap().log_prob(&0.0),
            f64::NEG_INFINITY
        ); // k > 1
        close(
            Weibull::new(1.0, 2.0).unwrap().log_prob(&0.0),
            -(2.0f64).ln(),
        ); // k == 1
        assert_eq!(
            Weibull::new(0.5, 1.0).unwrap().log_prob(&0.0),
            f64::INFINITY
        ); // k < 1
        assert_eq!(
            ChiSquared::new(3.0).unwrap().log_prob(&0.0),
            f64::NEG_INFINITY
        );
        assert_eq!(
            ChiSquared::new(3.0).unwrap().log_prob(&-1.0),
            f64::NEG_INFINITY
        );
        assert_eq!(
            InverseGamma::new(2.0, 1.0).unwrap().log_prob(&0.0),
            f64::NEG_INFINITY
        );
        // StudentT/Cauchy/Laplace are full-support: finite everywhere finite.
        assert!(StudentT::new(2.0, 0.0, 1.0)
            .unwrap()
            .log_prob(&-100.0)
            .is_finite());
        assert!(Cauchy::new(0.0, 1.0).unwrap().log_prob(&1e6).is_finite());
        assert!(Laplace::new(0.0, 1.0).unwrap().log_prob(&-42.0).is_finite());
        // DiscreteUniform: outside the inclusive range -> -inf.
        let du = DiscreteUniform::new(1, 6).unwrap();
        assert_eq!(du.log_prob(&0), f64::NEG_INFINITY);
        assert_eq!(du.log_prob(&7), f64::NEG_INFINITY);
        assert!(du.log_prob(&1).is_finite());
        assert!(du.log_prob(&6).is_finite());
        assert_eq!(du.len(), 6);
    }

    // FG-31: seeded moment sanity — sample means/variances match analytic
    // values within Monte-Carlo tolerance. Tolerances are set well above the
    // standard error at N = 60_000 so the seeded assertions are stable.
    #[test]
    fn fg31_new_distributions_moment_sanity() {
        let mut rng = StdRng::seed_from_u64(31);
        let n = 60_000usize;

        // Helper: sample mean of a distribution.
        fn mean_of(d: &impl Distribution<f64>, rng: &mut StdRng, n: usize) -> f64 {
            (0..n).map(|_| d.sample(rng)).sum::<f64>() / n as f64
        }

        // StudentT(df=6, loc=1, scale=2): mean = loc = 1 (df > 1).
        let t = StudentT::new(6.0, 1.0, 2.0).unwrap();
        assert!((mean_of(&t, &mut rng, n) - 1.0).abs() < 0.1);

        // Laplace(0, 2): mean 0, variance 2b^2 = 8.
        let lap = Laplace::new(0.0, 2.0).unwrap();
        let lap_samples: Vec<f64> = (0..n).map(|_| lap.sample(&mut rng)).collect();
        let lap_mean = lap_samples.iter().sum::<f64>() / n as f64;
        let lap_var = lap_samples
            .iter()
            .map(|x| (x - lap_mean).powi(2))
            .sum::<f64>()
            / n as f64;
        assert!(lap_mean.abs() < 0.1);
        assert!((lap_var - 8.0).abs() < 0.6);

        // Weibull(shape=2, scale=1.5): mean = scale*Γ(1+1/2) = 1.3293403881791368.
        let w = Weibull::new(2.0, 1.5).unwrap();
        assert!((mean_of(&w, &mut rng, n) - 1.3293403881791368).abs() < 0.05);

        // ChiSquared(4): mean 4, variance 2k = 8.
        let c = ChiSquared::new(4.0).unwrap();
        let c_samples: Vec<f64> = (0..n).map(|_| c.sample(&mut rng)).collect();
        let c_mean = c_samples.iter().sum::<f64>() / n as f64;
        let c_var = c_samples.iter().map(|x| (x - c_mean).powi(2)).sum::<f64>() / n as f64;
        assert!((c_mean - 4.0).abs() < 0.1);
        assert!((c_var - 8.0).abs() < 0.6);

        // InverseGamma(shape=4, rate=3): mean = β/(α-1) = 1.
        let ig = InverseGamma::new(4.0, 3.0).unwrap();
        assert!((mean_of(&ig, &mut rng, n) - 1.0).abs() < 0.05);

        // Cauchy: no mean; check the empirical MEDIAN converges to loc instead.
        let cau = Cauchy::new(2.0, 1.0).unwrap();
        let mut cau_samples: Vec<f64> = (0..n).map(|_| cau.sample(&mut rng)).collect();
        cau_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = cau_samples[n / 2];
        assert!((median - 2.0).abs() < 0.1);

        // DiscreteUniform(1, 6): mean 3.5, in-range always.
        let du = DiscreteUniform::new(1, 6).unwrap();
        let du_samples: Vec<i64> = (0..n).map(|_| du.sample(&mut rng)).collect();
        assert!(du_samples.iter().all(|&k| (1..=6).contains(&k)));
        let du_mean = du_samples.iter().map(|&k| k as f64).sum::<f64>() / n as f64;
        assert!((du_mean - 3.5).abs() < 0.05);
    }

    // FG-55: `Validate` is now implemented for every exported distribution. The
    // trait mirrors each `new()` constructor, so an *invalid* instance can only
    // be built here — via a struct literal with private fields, which is only
    // possible inside this module. One case per newly implemented distribution
    // (LogNormal, Binomial, Poisson, StudentT, Cauchy, Laplace, Weibull,
    // ChiSquared, InverseGamma, DiscreteUniform), asserting the same error code
    // the corresponding constructor emits. (The public, valid-instance
    // exhaustiveness guard lives in `tests/f_validate_coverage.rs`.)
    #[test]
    fn fg55_validate_rejects_invalid_parameters() {
        use crate::error::{ErrorCode, Validate};

        // LogNormal: non-positive sigma -> InvalidVariance.
        assert_eq!(
            LogNormal {
                mu: 0.0,
                sigma: 0.0
            }
            .validate()
            .unwrap_err()
            .code(),
            ErrorCode::InvalidVariance
        );
        // Binomial: probability outside [0, 1] -> InvalidProbability.
        assert_eq!(
            Binomial { n: 10, p: 1.5 }.validate().unwrap_err().code(),
            ErrorCode::InvalidProbability
        );
        // Poisson: non-positive rate -> InvalidRate.
        assert_eq!(
            Poisson { lambda: -1.0 }.validate().unwrap_err().code(),
            ErrorCode::InvalidRate
        );
        // StudentT: non-positive degrees of freedom -> InvalidShape.
        assert_eq!(
            StudentT {
                df: 0.0,
                loc: 0.0,
                scale: 1.0
            }
            .validate()
            .unwrap_err()
            .code(),
            ErrorCode::InvalidShape
        );
        // Cauchy: non-positive scale -> InvalidVariance.
        assert_eq!(
            Cauchy {
                loc: 0.0,
                scale: -1.0
            }
            .validate()
            .unwrap_err()
            .code(),
            ErrorCode::InvalidVariance
        );
        // Laplace: non-finite location -> InvalidMean.
        assert_eq!(
            Laplace {
                loc: f64::NAN,
                scale: 1.0
            }
            .validate()
            .unwrap_err()
            .code(),
            ErrorCode::InvalidMean
        );
        // Weibull: non-positive shape -> InvalidShape.
        assert_eq!(
            Weibull {
                shape: -2.0,
                scale: 1.0
            }
            .validate()
            .unwrap_err()
            .code(),
            ErrorCode::InvalidShape
        );
        // ChiSquared: non-positive degrees of freedom -> InvalidShape.
        assert_eq!(
            ChiSquared { k: 0.0 }.validate().unwrap_err().code(),
            ErrorCode::InvalidShape
        );
        // InverseGamma: non-positive rate -> InvalidRate.
        assert_eq!(
            InverseGamma {
                shape: 2.0,
                rate: -1.0
            }
            .validate()
            .unwrap_err()
            .code(),
            ErrorCode::InvalidRate
        );
        // DiscreteUniform: high < low -> InvalidRange.
        assert_eq!(
            DiscreteUniform { low: 5, high: 1 }
                .validate()
                .unwrap_err()
                .code(),
            ErrorCode::InvalidRange
        );
    }

    // Re-verification (low): `DiscreteUniform` over the full i64 domain has a
    // support of 2^64 points. The pre-fix `len()` computed the count as
    // `(high - low + 1) as u64`, which truncates 2^64 to 0 — so `sample()`
    // panicked on `gen_range(0..0)` and `log_prob()` for an in-range `x` returned
    // `-(0.0).ln() = +INF`. The fix keeps the count in `u128`, samples the full
    // domain with a raw uniform `i64`, and scores it as `-64·ln 2`.
    #[test]
    fn discrete_uniform_full_i64_range_samples_and_scores() {
        let du = DiscreteUniform::new(i64::MIN, i64::MAX).unwrap();

        // `len()` saturates (2^64 doesn't fit in u64) but the distribution stays
        // usable.
        assert_eq!(du.len(), u64::MAX);
        assert!(!du.is_empty());

        // sample() must not panic and must return real i64 values across the whole
        // domain (seeded for determinism). A truncated count would panic here.
        let mut rng = StdRng::seed_from_u64(0xF017_2026);
        let mut saw_negative = false;
        let mut saw_positive = false;
        for _ in 0..10_000 {
            let x = du.sample(&mut rng);
            // Every i64 is in support, so log_prob is finite for every draw.
            assert!(du.log_prob(&x).is_finite());
            saw_negative |= x < 0;
            saw_positive |= x > 0;
        }
        // A raw uniform i64 spans both signs; a broken offset path (or a fixed
        // low) would not.
        assert!(
            saw_negative && saw_positive,
            "full-range sampler is not uniform"
        );

        // log_prob for any in-range x is exactly -ln(2^64) = -64·ln 2. The pre-fix
        // code returned +INF here.
        let expected = -(64.0 * std::f64::consts::LN_2);
        for &x in &[i64::MIN, -1_000_000_i64, -1, 0, 1, 1_000_000_i64, i64::MAX] {
            let lp = du.log_prob(&x);
            assert!(
                (lp - expected).abs() < 1e-12,
                "full-range log_prob({x}) = {lp}, expected {expected}"
            );
        }
    }

    // Re-verification (low): ranges one short of the full domain (span 2^64 − 1,
    // the largest that fits in a u64 count) must still sample without overflow and
    // score as -ln(2^64 − 1).
    #[test]
    fn discrete_uniform_near_full_ranges_are_exact() {
        let mut rng = StdRng::seed_from_u64(0xBEEF_2026);

        for du in [
            DiscreteUniform::new(i64::MIN, i64::MAX - 1).unwrap(),
            DiscreteUniform::new(i64::MIN + 1, i64::MAX).unwrap(),
        ] {
            // Count = 2^64 − 1 fits exactly in u64.
            assert_eq!(du.len(), u64::MAX);
            let expected = -((u64::MAX as f64).ln());
            for _ in 0..2_000 {
                let x = du.sample(&mut rng);
                assert!(du.log_prob(&x).is_finite());
            }
            // In-range score is -ln(2^64 − 1); the excluded endpoint is -inf.
            close(du.log_prob(&0), expected);
            let excluded = if du.high() == i64::MAX - 1 {
                i64::MAX
            } else {
                i64::MIN
            };
            assert_eq!(du.log_prob(&excluded), f64::NEG_INFINITY);
        }
    }
}
