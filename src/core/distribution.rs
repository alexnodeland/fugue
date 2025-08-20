//! Type-safe probability distributions with natural return types.
//!
//! This module provides a unified, type-safe interface for probability distributions used in Fugue models.
//! All distributions implement the `Distribution<T>` trait, which provides sampling and
//! log-probability density computation. The trait is generic over the sample type `T`,
//! enabling natural return types for each distribution.
//!
//! ## Key Innovation: Type Safety
//!
//! Unlike traditional probabilistic programming libraries that force all distributions
//! to return `f64`, Fugue's distributions return their natural types:
//!
//! - **Continuous distributions** → `f64` (as expected)
//! - **Bernoulli** → `bool` (not 0.0/1.0!)
//! - **Poisson/Binomial** → `u64` (natural counting)
//! - **Categorical** → `usize` (safe array indexing)
//!
//! ## Available Distributions
//!
//! ### Continuous Distributions (return `f64`)
//! - [`Normal`]: Normal/Gaussian distribution
//! - [`LogNormal`]: Log-normal distribution  
//! - [`Uniform`]: Uniform distribution over an interval
//! - [`Exponential`]: Exponential distribution
//! - [`Beta`]: Beta distribution on \[0,1\]
//! - [`Gamma`]: Gamma distribution
//!
//! ### Discrete Distributions (return natural types!)
//! - [`Bernoulli`]: Bernoulli distribution → **`bool`**
//! - [`Binomial`]: Binomial distribution → **`u64`**
//! - [`Categorical`]: Categorical distribution → **`usize`**
//! - [`Poisson`]: Poisson distribution → **`u64`**
//!
//! All distributions can be used both within the Model system (with `sample()` and `observe()`)
//! and for direct statistical computation outside the Model system by calling `sample()` and
//! `log_prob()` directly on distribution instances.
//!
//! ## Usage Examples
//!
//! ### Type-Safe Sampling
//! ```rust
//! use fugue::*;
//!
//! // Continuous distribution returns f64
//! let normal_model: Model<f64> = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
//!
//! // Bernoulli returns bool - no more awkward comparisons!
//! let coin_model: Model<bool> = sample(addr!("coin"), Bernoulli::new(0.5).unwrap());
//! let decision = coin_model.bind(|heads| {
//!     if heads {
//!         pure("Take action!".to_string())
//!     } else {
//!         pure("Wait...".to_string())
//!     }
//! });
//!
//! // Poisson returns u64 - perfect for counting!
//! let count_model: Model<u64> = sample(addr!("events"), Poisson::new(3.0).unwrap());
//! let analysis = count_model.bind(|count| {
//!     let status = match count {
//!         0 => "No events",
//!         1 => "Single event",
//!         n if n > 10 => "Many events!",
//!         n => &format!("{} events", n),
//!     };
//!     pure(status.to_string())
//! });
//!
//! // Categorical returns usize - safe array indexing!
//! let choice_model: Model<usize> = sample(addr!("color"),
//!     Categorical::new(vec![0.5, 0.3, 0.2]).unwrap());
//! let colors = vec!["red", "green", "blue"];
//! let result = choice_model.bind(move |color_idx| {
//!     let chosen_color = colors[color_idx]; // Direct indexing - no casting!
//!     pure(chosen_color.to_string())
//! });
//! ```
//!
//! ### Type-Safe Observations
//! ```rust
//! use fugue::*;
//!
//! // Observe with natural types
//! let model = observe(addr!("coin_result"), Bernoulli::new(0.6).unwrap(), true)      // bool
//!     .bind(|_| observe(addr!("count"), Poisson::new(4.0).unwrap(), 7u64))      // u64
//!     .bind(|_| observe(addr!("choice"), Categorical::new(
//!         vec![0.3, 0.5, 0.2]
//!     ).unwrap(), 1usize))  // usize
//!     .bind(|_| observe(addr!("temp"), Normal::new(20.0, 2.0).unwrap(), 18.5)); // f64
//! ```
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
///
/// This trait provides the essential operations needed for probabilistic programming
/// with **full type safety**. Unlike traditional PPLs that force all distributions to return `f64`,
/// Fugue's `Distribution<T>` trait is generic over the sample type `T`, enabling:
///
/// - **Natural return types**: Each distribution returns its mathematically appropriate type
/// - **Compile-time safety**: Type errors are caught by the compiler, not at runtime
/// - **Zero overhead**: No unnecessary type conversions or boxing
/// - **Intuitive code**: Write code that matches statistical intuition
///
/// ## Type Safety Benefits
///
/// | Distribution | Traditional PPL | Fugue Type-Safe |
/// |--------------|-----------------|-----------------|
/// | Bernoulli | `f64` (0.0/1.0) | **`bool`** (true/false) |
/// | Poisson | `f64` (needs casting) | **`u64`** (natural counts) |
/// | Categorical | `f64` (risky indexing) | **`usize`** (safe indexing) |
/// | Binomial | `f64` (needs casting) | **`u64`** (natural counts) |
/// | Normal | `f64` ✓ | **`f64`** ✓ |
///
/// # Required Methods
///
/// - [`sample`](Self::sample): Generate a random sample of type `T`
/// - [`log_prob`](Self::log_prob): Compute log-probability of a value of type `T`
/// - [`clone_box`](Self::clone_box): Clone into a boxed trait object
///
/// # Examples
///
/// ```rust
/// use fugue::*;
/// use rand::rngs::StdRng;
/// use rand::SeedableRng;
///
/// let mut rng = StdRng::seed_from_u64(42);
///
/// // Continuous distribution returns f64
/// let normal = Normal::new(0.0, 1.0).unwrap();
/// let value: f64 = normal.sample(&mut rng);
/// let log_prob = normal.log_prob(&value);
///
/// // Discrete distributions return natural types!
/// let coin = Bernoulli::new(0.5).unwrap();
/// let flip: bool = coin.sample(&mut rng);  // bool, not f64!
/// let coin_prob = coin.log_prob(&flip);
///
/// let counter = Poisson::new(3.0).unwrap();
/// let count: u64 = counter.sample(&mut rng);  // u64, not f64!
/// let count_prob = counter.log_prob(&count);
///
/// let choice = Categorical::new(vec![0.3, 0.5, 0.2]).unwrap();
/// let idx: usize = choice.sample(&mut rng);  // usize for safe indexing!
/// let choice_prob = choice.log_prob(&idx);
/// ```
pub trait Distribution<T>: Send + Sync {
    /// Generate a random sample from this distribution.
    ///
    /// Returns a value of type `T` drawn from this distribution. The return type
    /// is naturally suited to the distribution:
    /// - `f64` for continuous distributions (Normal, Beta, etc.)
    /// - `bool` for Bernoulli (true/false outcomes)
    /// - `u64` for count distributions (Poisson, Binomial)
    /// - `usize` for categorical indices (safe array indexing)
    ///
    /// # Arguments
    ///
    /// * `rng` - Random number generator to use for sampling
    ///
    /// # Returns
    ///
    /// A sample from the distribution of type `T`.
    fn sample(&self, rng: &mut dyn RngCore) -> T;

    /// Compute the log-probability density/mass of a value under this distribution.
    ///
    /// Accepts a reference to a value of type `T` to avoid unnecessary copying
    /// and to maintain consistency across all distribution types.
    ///
    /// # Arguments
    ///
    /// * `x` - Reference to the value to compute log-probability for
    ///
    /// # Returns
    ///
    /// The natural logarithm of the probability density/mass at `x`.
    /// Returns negative infinity for values outside the distribution's support.
    fn log_prob(&self, x: &T) -> LogF64;

    /// Clone this distribution into a boxed trait object.
    ///
    /// This method is required for the trait to be object-safe, allowing
    /// distributions to be stored as `Box<dyn Distribution<T>>`.
    fn clone_box(&self) -> Box<dyn Distribution<T>>;
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
/// let std_normal = Normal::new(0.0, 1.0).unwrap();
///
/// // Normal prior for a parameter
/// let model = sample(addr!("theta"), Normal::new(0.0, 2.0).unwrap());
///
/// // Normal likelihood for observations
/// let model = observe(addr!("y"), Normal::new(1.5, 0.5).unwrap(), 2.0);
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
    ///
    /// This is the only way to construct a Normal distribution, ensuring
    /// parameter validity at construction time.
    ///
    /// # Arguments
    ///
    /// * `mu` - Mean of the distribution (must be finite)
    /// * `sigma` - Standard deviation (must be positive and finite)
    ///
    /// # Errors
    ///
    /// Returns `FugueError::InvalidParameters` if parameters are invalid.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fugue::*;
    ///
    /// // Valid construction
    /// let normal = Normal::new(0.0, 1.0)?;
    ///
    /// // Invalid parameters will fail
    /// let bad_normal = Normal::new(0.0, -1.0); // Error: negative sigma
    /// assert!(bad_normal.is_err());
    /// # Ok::<(), FugueError>(())
    /// ```
    pub fn new(mu: f64, sigma: f64) -> crate::error::FugueResult<Self> {
        if !mu.is_finite() {
            return Err(crate::error::FugueError::InvalidParameters {
                distribution: "Normal".to_string(),
                reason: "Mean (mu) must be finite".to_string(),
            });
        }
        if sigma <= 0.0 || !sigma.is_finite() {
            return Err(crate::error::FugueError::InvalidParameters {
                distribution: "Normal".to_string(),
                reason: "Standard deviation (sigma) must be positive and finite".to_string(),
            });
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
        const LN_2PI: f64 = 1.8378770664093454835606594728112; // ln(2π)
        -0.5 * z * z - self.sigma.ln() - 0.5 * LN_2PI
    }
    fn clone_box(&self) -> Box<dyn Distribution<f64>> {
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
/// let unit_uniform = Uniform::new(0.0, 1.0).unwrap();
///
/// // Symmetric interval around zero
/// let symmetric = Uniform::new(-5.0, 5.0).unwrap();
///
/// // Use as uninformative prior
/// let model = sample(addr!("weight"), Uniform::new(0.0, 100.0).unwrap());
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
            return Err(crate::error::FugueError::InvalidParameters {
                distribution: "Uniform".to_string(),
                reason: "Bounds must be finite".to_string(),
            });
        }
        if low >= high {
            return Err(crate::error::FugueError::InvalidParameters {
                distribution: "Uniform".to_string(),
                reason: "Lower bound must be less than upper bound".to_string(),
            });
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
/// let log_normal = LogNormal::new(0.0, 1.0).unwrap();
///
/// // Model for positive scale parameters
/// let model = sample(addr!("scale"), LogNormal::new(0.0, 0.5).unwrap());
///
/// // Income distribution (often log-normal)
/// let income_model = sample(addr!("income"), LogNormal::new(10.0, 0.8).unwrap());
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
            return Err(crate::error::FugueError::InvalidParameters {
                distribution: "LogNormal".to_string(),
                reason: "Mean (mu) must be finite".to_string(),
            });
        }
        if sigma <= 0.0 || !sigma.is_finite() {
            return Err(crate::error::FugueError::InvalidParameters {
                distribution: "LogNormal".to_string(),
                reason: "Standard deviation (sigma) must be positive and finite".to_string(),
            });
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
        const LN_2PI: f64 = 1.8378770664093454835606594728112; // ln(2π)
        -0.5 * z * z - lx - self.sigma.ln() - 0.5 * LN_2PI
    }
    fn clone_box(&self) -> Box<dyn Distribution<f64>> {
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
/// let waiting_time = Exponential::new(2.0).unwrap();
///
/// // Survival analysis / hazard modeling
/// let model = sample(addr!("survival_time"), Exponential::new(0.1).unwrap());
///
/// // Prior for precision parameters (inverse of variance)
/// let precision_prior = sample(addr!("precision"), Exponential::new(1.0).unwrap());
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
            return Err(crate::error::FugueError::InvalidParameters {
                distribution: "Exponential".to_string(),
                reason: "Rate parameter must be positive and finite".to_string(),
            });
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

/// **Type-safe Bernoulli distribution** → returns `bool`
///
/// A discrete distribution representing a single trial with two possible outcomes:
/// success (true) with probability p, or failure (false) with probability 1-p.
/// This is the building block for binomial distributions and binary classification.
///
/// ## 🎯 Type Safety Innovation
///
/// **Unlike traditional PPLs**, Fugue's Bernoulli distribution returns **`bool` directly**,
/// eliminating error-prone floating-point comparisons like `if sample == 1.0`.
///
/// **Probability mass function:**
/// ```text
/// P(X = true) = p
/// P(X = false) = 1 - p
/// ```
///
/// **Support:** {false, true} (natural boolean values!)
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
/// // Type-safe boolean sampling - no more f64 comparisons!
/// let coin_model: Model<bool> = sample(addr!("coin"), Bernoulli::new(0.5).unwrap());
/// let decision = coin_model.bind(|heads| {
///     if heads {  // ✅ Natural boolean usage!
///         pure("Heads - take action!".to_string())
///     } else {
///         pure("Tails - wait...".to_string())
///     }
/// });
///
/// // Mixture component selection with natural boolean logic
/// let component_model = sample(addr!("component"), Bernoulli::new(0.3).unwrap())
///     .bind(|is_component_2| {
///         let component_name = if is_component_2 {
///             "Component 2"
///         } else {
///             "Component 1"  
///         };
///         pure(component_name.to_string())
///     });
///
/// // Type-safe observation of boolean outcomes
/// let obs_model = observe(addr!("success"), Bernoulli::new(0.8).unwrap(), true);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Bernoulli {
    /// Probability of success (must be in [0, 1]).
    p: f64,
}

impl Bernoulli {
    /// Create a new Bernoulli distribution with validated parameters.
    pub fn new(p: f64) -> crate::error::FugueResult<Self> {
        if !p.is_finite() || p < 0.0 || p > 1.0 {
            return Err(crate::error::FugueError::InvalidParameters {
                distribution: "Bernoulli".to_string(),
                reason: "Probability must be in [0, 1]".to_string(),
            });
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

/// **Type-safe Categorical distribution** → returns `usize`
///
/// A discrete distribution that represents choosing among k different categories
/// with specified probabilities. The outcome is the index of the chosen category
/// as a `usize`, making it **naturally suitable for safe array indexing**.
///
/// ## 🎯 Type Safety Innovation
///
/// **Unlike traditional PPLs**, Fugue's Categorical distribution returns **`usize` directly**,
/// enabling safe array indexing without error-prone casting from `f64`.
///
/// **Probability mass function:**
/// ```text
/// P(X = i) = probs[i]  for i ∈ {0, 1, ..., k-1}
/// ```
///
/// **Support:** {0, 1, ..., k-1} where k = probs.len() (natural array indices!)
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
/// // Type-safe categorical choice - returns usize directly!
/// let options = vec!["red", "green", "blue"];
/// let color_model: Model<usize> = sample(addr!("color"),
///     Categorical::new(vec![0.5, 0.3, 0.2]).unwrap());
/// let result = color_model.bind(move |color_idx| {
///     // color_idx is naturally usize - safe for direct array indexing!
///     let chosen_color = options[color_idx]; // No casting, no bounds checking needed!
///     pure(chosen_color.to_string())
/// });
///
/// // Multi-armed bandit with type-safe action selection
/// let action_model = sample(addr!("action"),
///     Categorical::new(vec![0.4, 0.3, 0.2, 0.1]).unwrap()  // 4 possible actions
/// ).bind(|action_idx| {
///     let action_rewards = vec![10.0, 15.0, 5.0, 20.0];
///     let reward = action_rewards[action_idx]; // Direct, safe indexing!
///     pure(reward)
/// });
///
/// // Type-safe observation of categorical outcomes
/// let obs_model = observe(addr!("user_choice"),
///     Categorical::new(vec![0.2, 0.3, 0.3, 0.2]).unwrap(), 2usize);  // Observed choice was index 2
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
            return Err(crate::error::FugueError::InvalidParameters {
                distribution: "Categorical".to_string(),
                reason: "Probability vector cannot be empty".to_string(),
            });
        }

        let sum: f64 = probs.iter().sum();
        if (sum - 1.0).abs() > 1e-6 {
            return Err(crate::error::FugueError::InvalidParameters {
                distribution: "Categorical".to_string(),
                reason: format!("Probabilities must sum to 1.0, got {:.6}", sum),
            });
        }

        for (i, &p) in probs.iter().enumerate() {
            if !p.is_finite() || p < 0.0 {
                return Err(crate::error::FugueError::InvalidParameters {
                    distribution: "Categorical".to_string(),
                    reason: format!(
                        "Probability at index {} must be non-negative and finite, got {}",
                        i, p
                    ),
                });
            }
        }

        Ok(Categorical { probs })
    }

    /// Create a uniform categorical distribution over k categories.
    pub fn uniform(k: usize) -> crate::error::FugueResult<Self> {
        if k == 0 {
            return Err(crate::error::FugueError::InvalidParameters {
                distribution: "Categorical".to_string(),
                reason: "Number of categories must be positive".to_string(),
            });
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
/// let uniform_beta = Beta::new(1.0, 1.0).unwrap();
///
/// // Prior for a probability parameter
/// let prob_prior = sample(addr!("p"), Beta::new(2.0, 5.0).unwrap());
///
/// // Conjugate prior for Bernoulli likelihood
/// let model = sample(addr!("success_rate"), Beta::new(3.0, 7.0).unwrap())
///     .bind(|p| observe(addr!("trial"), Bernoulli::new(p).unwrap(), true));
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
            return Err(crate::error::FugueError::InvalidParameters {
                distribution: "Beta".to_string(),
                reason: "Alpha parameter must be positive and finite".to_string(),
            });
        }
        if beta <= 0.0 || !beta.is_finite() {
            return Err(crate::error::FugueError::InvalidParameters {
                distribution: "Beta".to_string(),
                reason: "Beta parameter must be positive and finite".to_string(),
            });
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
/// let exponential_like = Gamma::new(1.0, 2.0).unwrap();
///
/// // Prior for precision (inverse variance)
/// let precision = sample(addr!("precision"), Gamma::new(2.0, 1.0).unwrap());
///
/// // Conjugate prior for Poisson rate
/// let model = sample(addr!("rate"), Gamma::new(3.0, 2.0).unwrap())
///     .bind(|lambda| observe(addr!("count"), Poisson::new(lambda).unwrap(), 5u64));
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
            return Err(crate::error::FugueError::InvalidParameters {
                distribution: "Gamma".to_string(),
                reason: "Shape parameter must be positive and finite".to_string(),
            });
        }
        if rate <= 0.0 || !rate.is_finite() {
            return Err(crate::error::FugueError::InvalidParameters {
                distribution: "Gamma".to_string(),
                reason: "Rate parameter must be positive and finite".to_string(),
            });
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

/// **Type-safe Binomial distribution** → returns `u64`
///
/// A discrete distribution representing the number of successes in n independent
/// Bernoulli trials, each with success probability p. This distribution models
/// counting processes and is widely used in statistics.
///
/// ## 🎯 Type Safety Innovation
///
/// **Unlike traditional PPLs**, Fugue's Binomial distribution returns **`u64` directly**,
/// providing natural counting semantics for the number of successes without casting.
///
/// **Probability mass function:**
/// ```text
/// P(X = k) = C(n,k) * p^k * (1-p)^(n-k)  for k ∈ {0, 1, ..., n}
/// ```
/// where C(n,k) is the binomial coefficient "n choose k".
///
/// **Support:** {0, 1, ..., n} (natural success counts!)
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
/// // Type-safe success counting - returns u64 directly!
/// let trial_model: Model<u64> = sample(addr!("successes"), Binomial::new(10, 0.5).unwrap());
/// let analysis = trial_model.bind(|success_count| {
///     // success_count is naturally u64 - can be used in arithmetic directly
///     let success_rate = success_count as f64 / 10.0;
///     let verdict = if success_rate > 0.7 {
///         "High success rate!"
///     } else if success_rate < 0.3 {
///         "Low success rate"
///     } else {
///         "Moderate success rate"
///     };
///     pure(verdict.to_string())
/// });
///
/// // Clinical trial with type-safe counting
/// let clinical_trial = sample(addr!("success_rate"), Beta::new(1.0, 1.0).unwrap())
///     .bind(|p| {
///         sample(addr!("successes"), Binomial::new(100, p).unwrap())
///             .bind(|successes| {
///                 // successes is naturally u64 - no casting needed!
///                 let efficacy = successes as f64 / 100.0;
///                 pure(efficacy)
///             })
///     });
///
/// // Type-safe observation of trial results
/// let obs_model = observe(addr!("trial_successes"), Binomial::new(20, 0.3).unwrap(), 7u64);
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
        if !p.is_finite() || p < 0.0 || p > 1.0 {
            return Err(crate::error::FugueError::InvalidParameters {
                distribution: "Binomial".to_string(),
                reason: "Probability must be in [0, 1]".to_string(),
            });
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
        RDBinomial::new(self.n, self.p).unwrap().sample(rng) as u64
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

/// **Type-safe Poisson distribution** → returns `u64`
///
/// A discrete probability distribution expressing the probability of a given number
/// of events occurring in a fixed interval of time or space, given that these events
/// occur with a known constant mean rate and independently of each other.
///
/// ## 🎯 Type Safety Innovation
///
/// **Unlike traditional PPLs**, Fugue's Poisson distribution returns **`u64` directly**,
/// providing natural counting semantics without error-prone casting from `f64`.
///
/// **Probability mass function:**
/// ```text
/// P(X = k) = (λ^k * exp(-λ)) / k!  for k ∈ {0, 1, 2, ...}
/// ```
///
/// **Support:** {0, 1, 2, ...} (natural non-negative integers!)
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
/// // Type-safe count modeling - returns u64 directly!
/// let count_model: Model<u64> = sample(addr!("events"), Poisson::new(3.5).unwrap());
/// let analysis = count_model.bind(|count| {
///     // count is naturally u64 - can be used directly in match patterns
///     let status = match count {
///         0 => "No events occurred",
///         1 => "Single event occurred",
///         n if n > 10 => "High activity period!",
///         n => &format!("{} events occurred", n),
///     };
///     pure(status.to_string())
/// });
///
/// // Hierarchical count modeling with type safety
/// let hierarchical = sample(addr!("rate"), Gamma::new(2.0, 1.0).unwrap())
///     .bind(|lambda| {
///         sample(addr!("count"), Poisson::new(lambda).unwrap())
///             .bind(|count| {
///                 // count is naturally u64 - no casting needed!
///                 let bonus = if count > 5 { count * 2 } else { count };
///                 pure(bonus)
///             })
///     });
///
/// // Type-safe observation of count data
/// let obs_model = observe(addr!("observed_count"), Poisson::new(4.0).unwrap(), 7u64);
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
            return Err(crate::error::FugueError::InvalidParameters {
                distribution: "Poisson".to_string(),
                reason: "Rate parameter lambda must be positive and finite".to_string(),
            });
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
