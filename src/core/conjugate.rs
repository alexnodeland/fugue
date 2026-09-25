#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/docs/core/conjugate.md"))]
use crate::addr;
use crate::core::distribution::{Distribution, Gamma, LogF64};
use crate::core::model::{factor, pure, sample, sequence_vec, Model, ModelExt};
use crate::core::numerical::log_sum_exp;
use crate::error::{ErrorCode, FugueError, FugueResult};
use rand::distributions::Open01;
use rand::{Rng, RngCore};
use rand_distr::{Binomial as RDBinomial, Distribution as RandDistr, Gamma as RDGamma};

/// Absolute tolerance on `|Σᵢ xᵢ − 1|` for a vector to count as a point of the
/// probability simplex.
///
/// It is the same value [`Categorical`](crate::Categorical) uses for its sum
/// check, and it is used in two places here: [`dirichlet_log_pdf`] (and
/// `Dirichlet::log_prob`) return `-∞` for an `x` further than this from the
/// simplex, and [`multinomial_log_pmf`] / [`Multinomial::new`] reject a
/// probability vector further than this from it. It guards against gross
/// errors such as an unnormalized vector; it is not a precision claim. Vectors
/// produced by this module sum to 1 within a few ulps.
///
/// ```rust
/// # use fugue::*;
/// use fugue::core::conjugate::SIMPLEX_TOLERANCE;
/// let off = 1.0 + 10.0 * SIMPLEX_TOLERANCE;
/// assert_eq!(
///     dirichlet_log_pdf(&[1.0, 1.0], &[0.5, off - 0.5]).unwrap(),
///     f64::NEG_INFINITY
/// );
/// ```
pub const SIMPLEX_TOLERANCE: f64 = 1e-6;

// =============================================================================
// Validation
//
// Every public function and both distribution types validate through these
// helpers, and so do the `Validate` impls in `crate::error`, so the rules
// cannot drift apart. Invalid *parameters* are errors; a *value* outside the
// support (an `x` off the simplex, `k > n`, a count vector with the wrong
// total) is a `-∞` log-density, never an error.
// =============================================================================

/// Validate a concentration vector and return its total `A = Σᵢ αᵢ`.
///
/// Rules: at least one entry (`InvalidCount`); every entry positive and finite
/// (`InvalidShape`); and a total whose `lnΓ` is finite, i.e. `A ≲ 2.5·10³⁰⁵`
/// (`InvalidShape`). Past that bound every normalizing constant overflows, and
/// accepting it would turn the log-densities into `NaN`.
pub(crate) fn validate_concentrations(who: &str, alpha: &[f64]) -> FugueResult<f64> {
    if alpha.is_empty() {
        return Err(FugueError::invalid_parameters(
            who,
            "Concentration vector cannot be empty",
            ErrorCode::InvalidCount,
        )
        .with_context("length", "0"));
    }
    for (i, &a) in alpha.iter().enumerate() {
        check_positive_finite(who, "concentration", a)
            .map_err(|e| e.with_context("index", i.to_string()))?;
    }
    let total: f64 = alpha.iter().sum();
    check_total_concentration(who, total)?;
    Ok(total)
}

/// Validate a Beta pair `(a, b)` with the same rules as a two-entry
/// concentration vector, naming the parameters in the error context.
fn validate_beta_pair(who: &str, a: f64, b: f64) -> FugueResult<()> {
    check_positive_finite(who, "a", a)?;
    check_positive_finite(who, "b", b)?;
    check_total_concentration(who, a + b)
}

fn check_positive_finite(who: &str, name: &str, value: f64) -> FugueResult<()> {
    if value > 0.0 && value.is_finite() {
        return Ok(());
    }
    Err(FugueError::invalid_parameters(
        who,
        format!("Parameter {} must be positive and finite", name),
        ErrorCode::InvalidShape,
    )
    .with_context(name, value.to_string())
    .with_context("expected", "> 0.0 and finite"))
}

fn check_total_concentration(who: &str, total: f64) -> FugueResult<()> {
    // lnΓ(A) is finite for every finite A up to about 2.5e305 and infinite
    // beyond (and for A = +inf, which a sum of large finite values can reach).
    if libm::lgamma(total).is_finite() {
        return Ok(());
    }
    Err(FugueError::invalid_parameters(
        who,
        "Sum of concentrations is too large: its lnΓ overflows f64",
        ErrorCode::InvalidShape,
    )
    .with_context("sum", total.to_string())
    .with_context("expected", "<= 2.5e305"))
}

/// Validate a probability vector with [`Categorical`](crate::Categorical)'s
/// rules: non-empty, every entry finite and non-negative, and a sum within
/// [`SIMPLEX_TOLERANCE`] of 1 (all `InvalidProbability`).
pub(crate) fn validate_probabilities(who: &str, p: &[f64]) -> FugueResult<()> {
    if p.is_empty() {
        return Err(FugueError::invalid_parameters(
            who,
            "Probability vector cannot be empty",
            ErrorCode::InvalidProbability,
        )
        .with_context("length", "0"));
    }
    for (i, &pi) in p.iter().enumerate() {
        if !pi.is_finite() || pi < 0.0 {
            return Err(FugueError::invalid_parameters(
                who,
                "All probabilities must be non-negative and finite",
                ErrorCode::InvalidProbability,
            )
            .with_context("index", i.to_string())
            .with_context("value", pi.to_string())
            .with_context("expected", ">= 0.0 and finite"));
        }
    }
    let sum: f64 = p.iter().sum();
    if (sum - 1.0).abs() > SIMPLEX_TOLERANCE {
        return Err(FugueError::invalid_parameters(
            who,
            "Probabilities must sum to 1.0",
            ErrorCode::InvalidProbability,
        )
        .with_context("sum", sum.to_string())
        .with_context("expected", "1.0")
        .with_context("tolerance", SIMPLEX_TOLERANCE.to_string()));
    }
    Ok(())
}

/// Require one entry per category (`InvalidCount`).
fn check_same_len(who: &str, what: &str, categories: usize, len: usize) -> FugueResult<()> {
    if len == categories {
        return Ok(());
    }
    Err(FugueError::invalid_parameters(
        who,
        format!("{} must have one entry per category", what),
        ErrorCode::InvalidCount,
    )
    .with_context("categories", categories.to_string())
    .with_context(what, len.to_string()))
}

/// The total `N = Σᵢ nᵢ`, or `InvalidCount` if it overflows `u64`.
fn total_count(who: &str, counts: &[u64]) -> FugueResult<u64> {
    counts
        .iter()
        .try_fold(0u64, |acc, &n| acc.checked_add(n))
        .ok_or_else(|| {
            FugueError::invalid_parameters(
                who,
                "Total count overflows u64",
                ErrorCode::InvalidCount,
            )
            .with_context("max", u64::MAX.to_string())
        })
}

/// Validate `(α, counts)` together and return `(A, N)`.
fn validate_alpha_counts(who: &str, alpha: &[f64], counts: &[u64]) -> FugueResult<(f64, u64)> {
    let a_total = validate_concentrations(who, alpha)?;
    check_same_len(who, "counts", alpha.len(), counts.len())?;
    let n_total = total_count(who, counts)?;
    Ok((a_total, n_total))
}

// =============================================================================
// Numerical kernels
//
// Every marginal below is a sum of `lnΓ` differences, and at large counts the
// two `lnΓ` values in a difference are huge and nearly equal: `lnΓ(10⁹)` is
// about 2·10¹⁰, so subtracting two of them leaves an absolute error near 10⁻⁶
// (10⁻³ at 10¹²) in a result that may be of order 1. The kernels never form
// such a difference directly: `ln_gamma_ratio` evaluates `lnΓ(z + a) − lnΓ(z + b)`
// from a Stirling-series *difference*, and the marginals are assembled from
// log generalized binomial coefficients `ln[(x)ₘ / m!]`, which are small
// unless both `x` and `m` are large, rather than from the raw `lnΓ` terms.
// =============================================================================

/// Below this argument `lnΓ` is taken from `libm::lgamma` directly; at and
/// above it the Stirling series with seven correction terms is accurate to
/// better than 1e-16 (the first omitted term is below 3·10⁻¹⁷ at z = 10).
const STIRLING_MIN: f64 = 10.0;

/// The Stirling correction `lnΓ(z) − [(z − ½)·ln z − z + ½·ln 2π]`, as the
/// asymptotic series `Σₖ B₂ₖ / (2k(2k − 1)·z^(2k−1))` for `k = 1..=7`.
/// Only called with `z ≥ STIRLING_MIN`.
fn stirling_tail(z: f64) -> f64 {
    let y = 1.0 / (z * z);
    (1.0 / 12.0
        - y * (1.0 / 360.0
            - y * (1.0 / 1260.0
                - y * (1.0 / 1680.0 - y * (1.0 / 1188.0 - y * (691.0 / 360_360.0 - y / 156.0))))))
        / z
}

/// `lnΓ(z + a) − lnΓ(z + b)`, accurate even when both arguments are huge and
/// close together.
///
/// When both arguments are at least [`STIRLING_MIN`], with `x = z + a`,
/// `y = z + b` and `d = a − b` (taken from the small parts, so no information
/// about `a` and `b` is lost to rounding `z + a`):
///
/// ```text
/// lnΓ(x) − lnΓ(y) = (x − ½)·ln(x / y) + d·(ln y − 1) + S(x) − S(y)
/// ```
///
/// with `ln(x / y) = ln1p(d / y)` and `S` the Stirling tail. Every term has the
/// size of the result, so nothing cancels. Below the threshold both `lnΓ`
/// values are small and are subtracted directly.
fn ln_gamma_ratio(z: f64, a: f64, b: f64) -> f64 {
    let x = z + a;
    let y = z + b;
    if x < STIRLING_MIN || y < STIRLING_MIN {
        return libm::lgamma(x) - libm::lgamma(y);
    }
    let d = a - b;
    let t = d / y;
    // ln1p is exact for small |t|; near t = −1 the argument 1 + t itself
    // cancels, and ln x − ln y is the accurate form there.
    let ln_x_over_y = if t > -0.5 { t.ln_1p() } else { x.ln() - y.ln() };
    (x - 0.5) * ln_x_over_y + d * (y.ln() - 1.0) + (stirling_tail(x) - stirling_tail(y))
}

/// The log generalized binomial coefficient
/// `ln[Γ(x + m) / (Γ(x)·Γ(m + 1))] = ln C(x + m − 1, m)` for real `x > 0` and
/// integer `m ≥ 0` (the Pólya / negative-binomial coefficient `(x)ₘ / m!`).
///
/// The larger of `x` and `m + 1` is paired with `x + m` through
/// [`ln_gamma_ratio`]; the smaller one's `lnΓ` is subtracted directly, so
/// neither piece is a difference of two large `lnΓ` values. The two pieces
/// can still partly cancel, by at most a factor of about `ln(x + m)` and only
/// when `x` and `m` are of the same size, in which case the result is itself of
/// order `m` and keeps its relative accuracy.
fn ln_polya_coefficient(x: f64, m: f64) -> f64 {
    if m == 0.0 {
        return 0.0;
    }
    if x >= m + 1.0 {
        ln_gamma_ratio(x, m, 0.0) - libm::lgamma(m + 1.0)
    } else {
        ln_gamma_ratio(m, x, 1.0) - libm::lgamma(x)
    }
}

/// `ln B(α)` for a validated `α`: the largest `lnΓ(αⱼ)` is paired with
/// `lnΓ(Σα)` through [`ln_gamma_ratio`], the others are added directly.
fn ln_mbeta(alpha: &[f64]) -> f64 {
    let j = (0..alpha.len()).fold(0, |best, i| if alpha[i] > alpha[best] { i } else { best });
    let rest: f64 = alpha
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != j)
        .map(|(_, &a)| a)
        .sum();
    let others: f64 = alpha
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != j)
        .map(|(_, &a)| libm::lgamma(a))
        .sum();
    others - ln_gamma_ratio(alpha[j], rest, 0.0)
}

/// `ln(N! / Πᵢ nᵢ!)` as the sum of the non-negative terms
/// `ln C(n₁ + … + nᵢ, nᵢ)`: no cancellation. The total must fit in `u64`.
fn ln_mcoef(counts: &[u64]) -> f64 {
    let mut before = 0u64;
    let mut out = 0.0;
    for &n in counts {
        out += ln_polya_coefficient(before as f64 + 1.0, n as f64);
        before += n;
    }
    out
}

/// `ln P(n | α)` for the Dirichlet–multinomial, from validated inputs:
/// `Σᵢ ln[(αᵢ)ₙᵢ / nᵢ!] − ln[(A)_N / N!]`.
fn dm_log_marginal(alpha: &[f64], a_total: f64, counts: &[u64], n_total: u64) -> f64 {
    let per_category: f64 = alpha
        .iter()
        .zip(counts)
        .map(|(&a, &n)| ln_polya_coefficient(a, n as f64))
        .sum();
    per_category - ln_polya_coefficient(a_total, n_total as f64)
}

/// The Dirichlet log-density at `x` for a validated `α` with cached `ln B(α)`.
/// Any `x` that is not a point of the simplex (wrong length, a negative or
/// non-finite coordinate, a sum off by more than [`SIMPLEX_TOLERANCE`]) is `-∞`.
fn dirichlet_log_density(alpha: &[f64], ln_beta: f64, x: &[f64]) -> LogF64 {
    if x.len() != alpha.len() {
        return f64::NEG_INFINITY;
    }
    let mut sum = 0.0;
    for &xi in x {
        if !xi.is_finite() || xi < 0.0 {
            return f64::NEG_INFINITY;
        }
        sum += xi;
    }
    if (sum - 1.0).abs() > SIMPLEX_TOLERANCE {
        return f64::NEG_INFINITY;
    }
    let mut log_density = -ln_beta;
    let mut diverges = false;
    for (&a, &xi) in alpha.iter().zip(x) {
        if xi == 0.0 {
            // Boundary limits of xᵢ^(αᵢ−1), as for `Beta::log_prob`: a vanishing
            // factor makes the density 0 whatever the other coordinates do, a
            // diverging one makes it +∞ unless some factor vanishes, and αᵢ = 1
            // contributes nothing.
            if a > 1.0 {
                return f64::NEG_INFINITY;
            }
            diverges |= a < 1.0;
        } else {
            log_density += (a - 1.0) * xi.ln();
        }
    }
    if diverges {
        f64::INFINITY
    } else {
        log_density
    }
}

/// The multinomial log-mass for a validated `p` and counts whose total fits
/// in `u64`. An observed category with `pᵢ = 0` makes it `-∞`; an unobserved
/// one contributes nothing.
fn multinomial_log_mass(p: &[f64], counts: &[u64]) -> LogF64 {
    let mut log_mass = ln_mcoef(counts);
    for (&pi, &n) in p.iter().zip(counts) {
        if n == 0 {
            continue;
        }
        if pi <= 0.0 {
            return f64::NEG_INFINITY;
        }
        log_mass += n as f64 * pi.ln();
    }
    log_mass
}

// =============================================================================
// Special functions
// =============================================================================

/// The log multivariate Beta function, `ln B(α) = Σᵢ lnΓ(αᵢ) − lnΓ(Σᵢ αᵢ)`.
///
/// `B(α)` is the normalizing constant of the Dirichlet density; for `K = 2` it
/// is the ordinary Beta function `B(a, b)`. The largest `lnΓ(αⱼ)` is paired
/// with `lnΓ(Σα)` through a Stirling-series difference rather than
/// subtracting two large `lnΓ` values, so the result stays accurate when one
/// concentration dominates: `ln B(10⁹, 1) = −ln 10⁹` to full precision, where
/// the plain difference is off by about 10⁻⁶.
///
/// # Errors
///
/// `FugueError::InvalidParameters` if `alpha` is empty (`InvalidCount`), has a
/// non-positive or non-finite entry, or sums past about `2.5·10³⁰⁵`, where
/// `lnΓ` overflows (`InvalidShape`).
///
/// # Example
///
/// ```rust
/// # use fugue::*;
/// // B(1, 1, 1) = Γ(1)³ / Γ(3) = 1/2.
/// let ln_b = ln_multivariate_beta(&[1.0, 1.0, 1.0]).unwrap();
/// assert!((ln_b + std::f64::consts::LN_2).abs() < 1e-12);
///
/// // K = 2 is the Beta function: B(2, 3) = Γ(2)Γ(3) / Γ(5) = 1/12.
/// let ln_b2 = ln_multivariate_beta(&[2.0, 3.0]).unwrap();
/// assert!((ln_b2 + 12f64.ln()).abs() < 1e-12);
///
/// // An invalid concentration is an error, not a NaN.
/// assert!(ln_multivariate_beta(&[1.0, 0.0]).is_err());
/// ```
pub fn ln_multivariate_beta(alpha: &[f64]) -> FugueResult<f64> {
    validate_concentrations("Dirichlet", alpha)?;
    Ok(ln_mbeta(alpha))
}

/// The log multinomial coefficient `ln(N! / Πᵢ nᵢ!)`, with `N = Σᵢ nᵢ`: the
/// log number of distinct sequences whose category counts are `counts`.
///
/// It is exactly the difference between [`dirichlet_multinomial_log_marginal`]
/// (a count vector) and [`dirichlet_categorical_log_marginal`] (one sequence),
/// and between [`beta_binomial_log_marginal`] and [`beta_bernoulli_log_marginal`];
/// it does not depend on the prior. It is computed as the sum of the
/// non-negative terms `ln C(n₁ + … + nᵢ, nᵢ)`, so nothing cancels.
///
/// # Errors
///
/// `FugueError::InvalidParameters` (`InvalidCount`) if `counts` is empty or its
/// total overflows `u64`.
///
/// # Example
///
/// ```rust
/// # use fugue::*;
/// // 4! / (2!·1!·1!) = 12 sequences have the counts (2, 1, 1).
/// let c = ln_multinomial_coefficient(&[2, 1, 1]).unwrap();
/// assert!((c - 12f64.ln()).abs() < 1e-12);
/// ```
pub fn ln_multinomial_coefficient(counts: &[u64]) -> FugueResult<f64> {
    if counts.is_empty() {
        return Err(FugueError::invalid_parameters(
            "Multinomial",
            "Count vector cannot be empty",
            ErrorCode::InvalidCount,
        )
        .with_context("length", "0"));
    }
    total_count("Multinomial", counts)?;
    Ok(ln_mcoef(counts))
}

// =============================================================================
// Log-densities
// =============================================================================

/// Log-density of the Dirichlet distribution at a point `x` of the simplex:
///
/// ```text
/// ln Dir(x | α) = Σᵢ (αᵢ − 1)·ln xᵢ − ln B(α)
/// ```
///
/// The density is with respect to Lebesgue measure on the first `K − 1`
/// coordinates (the usual convention; it is what makes `Dirichlet(1, 1, 1)`
/// the constant 2 on the triangle). For `K = 2` it equals
/// `Beta::new(α₁, α₂)?.log_prob(&x₁)`.
///
/// **Support.** `x` must have every coordinate finite and `≥ 0`, and
/// `|Σᵢ xᵢ − 1| ≤` [`SIMPLEX_TOLERANCE`]; any other `x` returns `-∞`
/// (probability zero), not an error. At the boundary the semantics match
/// [`Beta`](crate::Beta)'s: a coordinate `xᵢ = 0` makes the log-density `-∞` if
/// `αᵢ > 1`, contributes nothing if `αᵢ = 1`, and makes it `+∞` if `αᵢ < 1`.
/// When one zero coordinate vanishes (`αᵢ > 1`) and another diverges
/// (`αⱼ < 1`), the limit depends on the path and the result is `-∞`.
///
/// # Errors
///
/// `FugueError::InvalidParameters` for an invalid `alpha` (see
/// [`ln_multivariate_beta`]) or if `x.len() != alpha.len()` (`InvalidCount`).
///
/// # Example
///
/// ```rust
/// # use fugue::*;
/// // Dirichlet(1, 1, 1) is uniform on the triangle: density 2 everywhere inside.
/// let lp = dirichlet_log_pdf(&[1.0, 1.0, 1.0], &[0.2, 0.3, 0.5]).unwrap();
/// assert!((lp - std::f64::consts::LN_2).abs() < 1e-12);
///
/// // K = 2 is the Beta distribution of the first coordinate.
/// let beta = Beta::new(2.0, 5.0).unwrap();
/// let lp2 = dirichlet_log_pdf(&[2.0, 5.0], &[0.3, 0.7]).unwrap();
/// assert!((lp2 - beta.log_prob(&0.3)).abs() < 1e-12);
///
/// // Off the simplex: probability zero, not an error.
/// let off = dirichlet_log_pdf(&[1.0, 1.0, 1.0], &[0.2, 0.3, 0.6]).unwrap();
/// assert_eq!(off, f64::NEG_INFINITY);
///
/// // A length mismatch is an error.
/// assert!(dirichlet_log_pdf(&[1.0, 1.0], &[0.5, 0.25, 0.25]).is_err());
/// ```
pub fn dirichlet_log_pdf(alpha: &[f64], x: &[f64]) -> FugueResult<f64> {
    validate_concentrations("Dirichlet", alpha)?;
    check_same_len("Dirichlet", "x", alpha.len(), x.len())?;
    Ok(dirichlet_log_density(alpha, ln_mbeta(alpha), x))
}

/// Log-mass of the multinomial distribution: the probability of the count
/// vector `counts` in `N = Σᵢ nᵢ` independent draws with category
/// probabilities `p`,
///
/// ```text
/// ln Mult(n | N, p) = ln(N! / Πᵢ nᵢ!) + Σᵢ nᵢ·ln pᵢ
/// ```
///
/// A category with `pᵢ = 0` contributes nothing when `nᵢ = 0` (`0·ln 0 = 0`)
/// and makes the result `-∞` when `nᵢ > 0`.
///
/// # Errors
///
/// `FugueError::InvalidParameters` if `p` is not a probability vector (empty, an
/// entry negative or non-finite, or a sum further than [`SIMPLEX_TOLERANCE`]
/// from 1: `InvalidProbability`), or if `counts` has the wrong length or a total
/// that overflows `u64` (`InvalidCount`).
///
/// # Example
///
/// ```rust
/// # use fugue::*;
/// // 4!/(2!·1!·1!) · 0.5² · 0.25 · 0.25 = 12 · 0.015625 = 0.1875.
/// let lp = multinomial_log_pmf(&[0.5, 0.25, 0.25], &[2, 1, 1]).unwrap();
/// assert!((lp - 0.1875f64.ln()).abs() < 1e-12);
///
/// // A zero-probability category is fine as long as it was never observed.
/// let lp0 = multinomial_log_pmf(&[0.5, 0.5, 0.0], &[1, 1, 0]).unwrap();
/// assert!((lp0 - 0.5f64.ln()).abs() < 1e-12);
/// let impossible = multinomial_log_pmf(&[0.5, 0.5, 0.0], &[1, 0, 1]).unwrap();
/// assert_eq!(impossible, f64::NEG_INFINITY);
/// ```
pub fn multinomial_log_pmf(p: &[f64], counts: &[u64]) -> FugueResult<f64> {
    validate_probabilities("Multinomial", p)?;
    check_same_len("Multinomial", "counts", p.len(), counts.len())?;
    total_count("Multinomial", counts)?;
    Ok(multinomial_log_mass(p, counts))
}

// =============================================================================
// Marginal likelihoods
// =============================================================================

/// Log marginal likelihood of a **count vector** under a Dirichlet prior
/// (the Dirichlet–multinomial, or Pólya, distribution):
///
/// ```text
/// ln P(n | α) = ln ∫ Mult(n | N, θ)·Dir(θ | α) dθ
///             = ln(N! / Πᵢ nᵢ!) + ln B(α + n) − ln B(α)
/// ```
///
/// with `N = Σᵢ nᵢ`. It is a distribution over count vectors: it sums to 1 over
/// the `C(N + K − 1, K − 1)` vectors with total `N`. Use it when only the
/// counts were observed. For the probability of **one particular sequence**
/// with these counts, use [`dirichlet_categorical_log_marginal`]; the two
/// differ by [`ln_multinomial_coefficient`], which does not depend on `α`, so
/// they give the same posterior over `α` and the same Bayes factors between
/// concentrations.
///
/// **Numerics.** It is evaluated as `Σᵢ ln[(αᵢ)ₙᵢ / nᵢ!] − ln[(A)_N / N!]`
/// (`(x)ₘ` the rising factorial), each term through a Stirling-series
/// difference, never as a difference of two large `lnΓ` values. The relative
/// error stays near 10⁻¹⁵ for tiny concentrations (down to the smallest
/// positive `f64`) and counts far beyond 10⁹; with `α = (1, 1)` and `N = 10¹²`
/// it returns `−ln(N + 1)` to full precision, where the plain `lnΓ` formula is
/// off by about 2·10⁻³. The one regime with a residual error is
/// concentrations *and* counts both large and comparable (≳ 10⁷), where terms
/// of order `N` cancel down to a small result: the absolute error there is
/// about `10⁻¹⁶·N·ln N` (around 10⁻⁵ at `α ≈ N ≈ 10⁹`).
///
/// # Errors
///
/// `FugueError::InvalidParameters` for an invalid `alpha` (see
/// [`ln_multivariate_beta`]), a `counts` of the wrong length, or a total count
/// that overflows `u64` (`InvalidCount`).
///
/// # Example
///
/// ```rust
/// # use fugue::*;
/// // α = (1, 1): all N + 1 count vectors with total N are equally likely.
/// for k in 0..=10u64 {
///     let lp = dirichlet_multinomial_log_marginal(&[1.0, 1.0], &[k, 10 - k]).unwrap();
///     assert!((lp + 11f64.ln()).abs() < 1e-12);
/// }
///
/// // It is the sequence probability plus the multinomial coefficient.
/// let (alpha, counts) = ([0.5, 2.0, 1.0], [3u64, 1, 4]);
/// let dm = dirichlet_multinomial_log_marginal(&alpha, &counts).unwrap();
/// let dc = dirichlet_categorical_log_marginal(&alpha, &counts).unwrap();
/// let coef = ln_multinomial_coefficient(&counts).unwrap();
/// assert!((dm - (dc + coef)).abs() < 1e-12);
/// ```
pub fn dirichlet_multinomial_log_marginal(alpha: &[f64], counts: &[u64]) -> FugueResult<f64> {
    let (a_total, n_total) = validate_alpha_counts("DirichletMultinomial", alpha, counts)?;
    Ok(dm_log_marginal(alpha, a_total, counts, n_total))
}

/// Log probability of **one particular sequence** of `N` categorical draws
/// whose category counts are `counts`, under a Dirichlet prior on the category
/// probabilities:
///
/// ```text
/// ln P(x₁, …, x_N | α) = ln B(α + n) − ln B(α)
///                      = Σᵢ ln[Γ(αᵢ + nᵢ) / Γ(αᵢ)] − ln[Γ(A + N) / Γ(A)]
/// ```
///
/// with `A = Σᵢ αᵢ` and `N = Σᵢ nᵢ`. There is no multinomial coefficient: this
/// is a probability over sequences, not over count vectors. It depends on the
/// sequence only through its counts (exchangeability) and equals the product
/// of the sequential predictive probabilities, the Pólya urn:
///
/// ```text
/// P(x₁, …, x_N | α) = Πₜ (α_{xₜ} + n_{xₜ}⁽ᵗ⁾) / (A + t − 1)
/// ```
///
/// where `n⁽ᵗ⁾` counts the draws before `t`. That product is the prequential
/// likelihood a sequential predictor computes step by step, so this is its
/// closed form for one context with a fixed base measure. Adding
/// [`ln_multinomial_coefficient`] gives [`dirichlet_multinomial_log_marginal`].
/// Numerics are as for that function.
///
/// # Errors
///
/// As for [`dirichlet_multinomial_log_marginal`].
///
/// # Example
///
/// ```rust
/// # use fugue::*;
/// // The sequence a, a, b under α = (1, 1), draw by draw:
/// // P(a) · P(a | a) · P(b | a, a) = 1/2 · 2/3 · 1/4 = 1/12.
/// let lp = dirichlet_categorical_log_marginal(&[1.0, 1.0], &[2, 1]).unwrap();
/// assert!((lp - (1.0f64 / 12.0).ln()).abs() < 1e-12);
/// ```
pub fn dirichlet_categorical_log_marginal(alpha: &[f64], counts: &[u64]) -> FugueResult<f64> {
    let (a_total, n_total) = validate_alpha_counts("DirichletCategorical", alpha, counts)?;
    Ok(dm_log_marginal(alpha, a_total, counts, n_total) - ln_mcoef(counts))
}

/// Log marginal likelihood of `k` successes in `n` Bernoulli trials whose
/// success probability has a `Beta(a, b)` prior (the Beta–binomial):
///
/// ```text
/// ln P(k | n, a, b) = ln C(n, k) + ln B(a + k, b + n − k) − ln B(a, b)
/// ```
///
/// This is [`dirichlet_multinomial_log_marginal`] with `K = 2`, with the same
/// numerics; `k > n` is outside the support and returns `-∞`.
///
/// # Errors
///
/// `FugueError::InvalidParameters` (`InvalidShape`) if `a` or `b` is not
/// positive and finite, or `a + b` is past about `2.5·10³⁰⁵`.
///
/// # Example
///
/// ```rust
/// # use fugue::*;
/// // Under a Beta(1, 1) prior the number of successes is uniform on {0, …, n}.
/// let lp = beta_binomial_log_marginal(1.0, 1.0, 3, 9).unwrap();
/// assert!((lp + 10f64.ln()).abs() < 1e-12);
///
/// // k > n is outside the support.
/// let none = beta_binomial_log_marginal(1.0, 1.0, 10, 9).unwrap();
/// assert_eq!(none, f64::NEG_INFINITY);
/// ```
pub fn beta_binomial_log_marginal(a: f64, b: f64, k: u64, n: u64) -> FugueResult<f64> {
    validate_beta_pair("BetaBinomial", a, b)?;
    if k > n {
        return Ok(f64::NEG_INFINITY);
    }
    Ok(dm_log_marginal(&[a, b], a + b, &[k, n - k], n))
}

/// Log probability of **one particular sequence** of Bernoulli trials with
/// `successes` successes and `failures` failures, when the success probability
/// has a `Beta(a, b)` prior:
///
/// ```text
/// ln P(sequence | a, b) = ln B(a + s, b + f) − ln B(a, b)
/// ```
///
/// This is [`dirichlet_categorical_log_marginal`] with `K = 2`, and
/// [`beta_binomial_log_marginal`] minus `ln C(s + f, s)`.
///
/// # Errors
///
/// `FugueError::InvalidParameters` for an invalid `(a, b)` (see
/// [`beta_binomial_log_marginal`]), or `InvalidCount` if
/// `successes + failures` overflows `u64`.
///
/// # Example
///
/// ```rust
/// # use fugue::*;
/// // The sequence 1, 1, 0 under Beta(1, 1): 1/2 · 2/3 · 1/4 = 1/12.
/// let lp = beta_bernoulli_log_marginal(1.0, 1.0, 2, 1).unwrap();
/// assert!((lp - (1.0f64 / 12.0).ln()).abs() < 1e-12);
/// ```
pub fn beta_bernoulli_log_marginal(
    a: f64,
    b: f64,
    successes: u64,
    failures: u64,
) -> FugueResult<f64> {
    validate_beta_pair("BetaBernoulli", a, b)?;
    let counts = [successes, failures];
    let n = total_count("BetaBernoulli", &counts)?;
    Ok(dm_log_marginal(&[a, b], a + b, &counts, n) - ln_mcoef(&counts))
}

// =============================================================================
// Posterior updates and predictives
// =============================================================================

/// Posterior concentrations of a Dirichlet prior after observing category
/// counts: `Dir(α)` and counts `n` give `Dir(α + n)`.
///
/// # Errors
///
/// As for [`dirichlet_multinomial_log_marginal`].
///
/// # Example
///
/// ```rust
/// # use fugue::*;
/// let post = dirichlet_posterior(&[0.5, 0.5, 0.5], &[12, 3, 0]).unwrap();
/// assert_eq!(post, vec![12.5, 3.5, 0.5]);
/// // The posterior is a Dirichlet like any other.
/// let posterior = Dirichlet::new(post).unwrap();
/// assert_eq!(posterior.len(), 3);
/// ```
pub fn dirichlet_posterior(alpha: &[f64], counts: &[u64]) -> FugueResult<Vec<f64>> {
    validate_alpha_counts("Dirichlet", alpha, counts)?;
    Ok(alpha
        .iter()
        .zip(counts)
        .map(|(&a, &n)| a + n as f64)
        .collect())
}

/// Posterior of a `Beta(a, b)` prior after `successes` successes and `failures`
/// failures: `Beta(a + successes, b + failures)`, as a pair.
///
/// # Errors
///
/// As for [`beta_binomial_log_marginal`].
///
/// # Example
///
/// ```rust
/// # use fugue::*;
/// let (a, b) = beta_posterior(2.0, 2.0, 7, 3).unwrap();
/// assert_eq!((a, b), (9.0, 5.0));
/// let posterior = Beta::new(a, b).unwrap();
/// assert!((posterior.alpha() / (posterior.alpha() + posterior.beta()) - 9.0 / 14.0).abs() < 1e-12);
/// ```
pub fn beta_posterior(a: f64, b: f64, successes: u64, failures: u64) -> FugueResult<(f64, f64)> {
    validate_beta_pair("Beta", a, b)?;
    Ok((a + successes as f64, b + failures as f64))
}

/// Posterior predictive probabilities of the next draw after observing
/// `counts` under a `Dir(α)` prior (the posterior mean of the category
/// probabilities):
///
/// ```text
/// P(x_{N+1} = i | n, α) = (αᵢ + nᵢ) / (A + N)
/// ```
///
/// One level of a hierarchical back-off model is this function with a
/// concentration spread over the level below:
/// `pᵢ = (nᵢ + α·baseᵢ) / (N + α)` is `dirichlet_predictive(α·base, n)`.
/// The probabilities are normalized by their own sum, so they sum to 1 within
/// a few ulps. For probabilities that may underflow (tiny `αᵢ`, huge `N`), use
/// [`dirichlet_log_predictive`].
///
/// # Errors
///
/// As for [`dirichlet_multinomial_log_marginal`].
///
/// # Example
///
/// ```rust
/// # use fugue::*;
/// let p = dirichlet_predictive(&[0.5, 0.5, 0.5], &[12, 3, 0]).unwrap();
/// assert!((p[0] - 12.5 / 16.5).abs() < 1e-12);
/// assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
///
/// // One back-off level: pᵢ = (nᵢ + α·baseᵢ) / (N + α).
/// let (alpha, base) = (2.0, [0.7, 0.2, 0.1]);
/// let conc: Vec<f64> = base.iter().map(|b| alpha * b).collect();
/// let p = dirichlet_predictive(&conc, &[5, 0, 1]).unwrap();
/// assert!((p[1] - (0.0 + 2.0 * 0.2) / (6.0 + 2.0)).abs() < 1e-12);
/// ```
pub fn dirichlet_predictive(alpha: &[f64], counts: &[u64]) -> FugueResult<Vec<f64>> {
    let post = dirichlet_posterior(alpha, counts)?;
    let total: f64 = post.iter().sum();
    Ok(post.into_iter().map(|w| w / total).collect())
}

/// Log posterior predictive probabilities,
/// `ln P(x_{N+1} = i | n, α) = ln(αᵢ + nᵢ) − ln(A + N)`.
///
/// Every entry is finite, including those whose probability underflows
/// `f64` in [`dirichlet_predictive`].
///
/// # Errors
///
/// As for [`dirichlet_multinomial_log_marginal`].
///
/// # Example
///
/// ```rust
/// # use fugue::*;
/// let alpha = [1e-320, 1.0];
/// let counts = [0, 1_000_000_000];
/// // The first predictive probability is about 1e-329: 0 in linear space...
/// assert_eq!(dirichlet_predictive(&alpha, &counts).unwrap()[0], 0.0);
/// // ...but its logarithm is finite and accurate.
/// let lp = dirichlet_log_predictive(&alpha, &counts).unwrap();
/// assert!((lp[0] - (1e-320f64.ln() - 1_000_000_001f64.ln())).abs() < 1e-9);
/// ```
pub fn dirichlet_log_predictive(alpha: &[f64], counts: &[u64]) -> FugueResult<Vec<f64>> {
    let post = dirichlet_posterior(alpha, counts)?;
    let ln_total = post.iter().sum::<f64>().ln();
    Ok(post.into_iter().map(|w| w.ln() - ln_total).collect())
}

// =============================================================================
// Distribution types
// =============================================================================

/// One draw of `ln G` for `G ~ Gamma(shape, 1)`, computed in log space.
///
/// For `shape ≥ 1` this is the logarithm of a `rand_distr` Gamma draw, which is
/// never 0 there. For `shape < 1` it uses the boost `G = G′·U^(1/shape)` with
/// `G′ ~ Gamma(shape + 1, 1)` and `U ~ Uniform(0, 1)`, taken in log space:
/// `ln G = ln G′ + ln U / shape`. `rand_distr` applies the same identity in
/// linear space, where `U^(1/shape)` underflows to 0 for small shapes (about
/// half the time at `shape = 10⁻³`). Here the result is finite for every
/// `shape ≳ 2·10⁻³⁰⁷`, and can be `-∞` only below that.
fn ln_gamma_variate(shape: f64, rng: &mut dyn RngCore) -> f64 {
    if shape >= 1.0 {
        RDGamma::new(shape, 1.0)
            .expect("shape is positive and finite")
            .sample(rng)
            .ln()
    } else {
        let boosted = RDGamma::new(shape + 1.0, 1.0)
            .expect("shape + 1 is positive and finite")
            .sample(rng);
        let u: f64 = rng.sample(Open01);
        boosted.ln() + u.ln() / shape
    }
}

/// A vertex index drawn with probabilities `αᵢ / A`: the exact limit of
/// `Dir(α)` as every concentration goes to 0 with fixed proportions, used when
/// every log-Gamma draw is `-∞`.
fn vertex_index(alpha: &[f64], rng: &mut dyn RngCore) -> usize {
    let total: f64 = alpha.iter().sum();
    let target = rng.gen::<f64>() * total;
    let mut acc = 0.0;
    for (i, &a) in alpha.iter().enumerate() {
        acc += a;
        if target < acc {
            return i;
        }
    }
    alpha.len() - 1
}

/// The Dirichlet distribution `Dir(α)` over the probability simplex, as a
/// `Distribution<Vec<f64>>`.
///
/// Mathematical Properties:
/// - **Support**: `{x ∈ ℝᴷ : xᵢ ≥ 0, Σᵢ xᵢ = 1}`
/// - **PDF**: `f(x) = Πᵢ xᵢ^(αᵢ−1) / B(α)`, with `B(α) = Πᵢ Γ(αᵢ) / Γ(Σᵢ αᵢ)`
/// - **Mean**: `αᵢ / A`, with `A = Σᵢ αᵢ`
/// - **Variance**: `αᵢ(A − αᵢ) / (A²(A + 1))`
/// - **Conjugacy**: the conjugate prior for categorical and multinomial
///   probabilities ([`dirichlet_posterior`])
///
/// **Not a site distribution.** `Vec<f64>` is not a
/// [`SampleType`](crate::SampleType), so a `Dirichlet` cannot be passed to
/// [`sample`] or [`observe`](crate::observe). Use it standalone,
/// to sample with an RNG and to score with `log_prob`. Inside a model, draw a
/// Dirichlet vector with [`sample_dirichlet`], which builds it from `K` scalar
/// Gamma sites; to condition on observed counts, use the conjugate helpers
/// ([`dirichlet_categorical_log_marginal`] with [`factor`], or
/// [`dirichlet_posterior`]).
///
/// **Sampling** normalizes `K` independent `Gamma(αᵢ, 1)` draws, generated as
/// log-Gammas and normalized with a log-sum-exp, so tiny concentrations are
/// stable: a draw is never `NaN` and never all-zero. With tiny `α` a draw sits
/// near a vertex, and coordinates below the smallest positive `f64` (relative
/// to the largest) come out as exactly `0.0`; [`Dirichlet::sample_log`] returns
/// their logarithms instead. If every log-Gamma draw overflows to `-∞` (only
/// possible when every `αᵢ` is below about `2·10⁻³⁰⁷`), the draw is the vertex
/// `eᵢ` with probability `αᵢ / A`, the exact small-`α` limit.
///
/// **Scoring**: `log_prob(x)` is [`dirichlet_log_pdf`] with the same support
/// and boundary rules; an `x` of the wrong length is outside the support
/// (`-∞`). `ln B(α)` is computed once, at construction.
///
/// # Example
///
/// ```rust
/// # use fugue::*;
/// # use rand::rngs::StdRng;
/// # use rand::SeedableRng;
/// let d = Dirichlet::new(vec![2.0, 3.0, 5.0]).unwrap();
/// let mut rng = StdRng::seed_from_u64(42);
/// let x = d.sample(&mut rng);
/// assert_eq!(x.len(), 3);
/// assert!((x.iter().sum::<f64>() - 1.0).abs() < 1e-12);
/// assert!(d.log_prob(&x).is_finite());
///
/// // Tiny concentrations: draws sit near a vertex but are never NaN or all-zero.
/// let sparse = Dirichlet::new(vec![1e-3; 4]).unwrap();
/// let y = sparse.sample(&mut rng);
/// assert!(y.iter().all(|v| v.is_finite() && *v >= 0.0));
/// assert!((y.iter().sum::<f64>() - 1.0).abs() < 1e-12);
/// ```
///
/// It cannot be sampled at an address:
///
/// ```rust,compile_fail
/// # use fugue::*;
/// // `Vec<f64>` is not a site type: this does not compile.
/// let w = sample(addr!("w"), Dirichlet::new(vec![1.0, 1.0]).unwrap());
/// ```
#[derive(Clone, Debug)]
pub struct Dirichlet {
    /// Concentration parameters (each positive and finite, validated in `new`).
    alpha: Vec<f64>,
    /// Cached `ln B(α)`, so `log_prob` makes no `lnΓ` calls.
    ln_beta: f64,
}
impl Dirichlet {
    /// Create a Dirichlet distribution with validated concentrations.
    ///
    /// # Errors
    ///
    /// `FugueError::InvalidParameters` if `alpha` is empty (`InvalidCount`), has a
    /// non-positive or non-finite entry, or sums past about `2.5·10³⁰⁵`
    /// (`InvalidShape`).
    ///
    /// ```rust
    /// # use fugue::*;
    /// assert!(Dirichlet::new(vec![0.5, 0.5]).is_ok());
    /// assert!(Dirichlet::new(vec![]).is_err());
    /// assert!(Dirichlet::new(vec![1.0, -1.0]).is_err());
    /// ```
    pub fn new(alpha: Vec<f64>) -> FugueResult<Self> {
        validate_concentrations("Dirichlet", &alpha)?;
        let ln_beta = ln_mbeta(&alpha);
        Ok(Dirichlet { alpha, ln_beta })
    }

    /// The concentration parameters `α`.
    ///
    /// ```rust
    /// # use fugue::*;
    /// let d = Dirichlet::new(vec![1.0, 2.0]).unwrap();
    /// assert_eq!(d.alpha(), &[1.0, 2.0]);
    /// ```
    pub fn alpha(&self) -> &[f64] {
        &self.alpha
    }

    /// The number of categories `K`.
    ///
    /// ```rust
    /// # use fugue::*;
    /// assert_eq!(Dirichlet::new(vec![1.0; 4]).unwrap().len(), 4);
    /// ```
    pub fn len(&self) -> usize {
        self.alpha.len()
    }

    /// Whether there are no categories. Always `false` for a validly
    /// constructed distribution (kept for clippy's `len`/`is_empty` pairing).
    ///
    /// ```rust
    /// # use fugue::*;
    /// assert!(!Dirichlet::new(vec![1.0]).unwrap().is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.alpha.is_empty()
    }

    /// Draw a point of the simplex and return the **logarithms** of its
    /// coordinates, `ln xᵢ`.
    ///
    /// It uses the same log-Gamma draws as `sample`, so from the same RNG state
    /// `exp` of this is `sample`'s draw up to rounding. Coordinates that
    /// `sample` returns as exactly `0.0` (with `α = 10⁻⁸` they are around
    /// `e^(−10⁸)`) are finite here. An entry is `-∞` only when its own
    /// log-Gamma draw overflowed, which needs `αᵢ ≲ 2·10⁻³⁰⁷` (see
    /// [`Dirichlet`]).
    ///
    /// ```rust
    /// # use fugue::*;
    /// # use rand::rngs::StdRng;
    /// # use rand::SeedableRng;
    /// let d = Dirichlet::new(vec![1e-8; 3]).unwrap();
    /// let mut rng = StdRng::seed_from_u64(3);
    /// let ln_x = d.sample_log(&mut rng);
    /// assert!(ln_x.iter().all(|l| l.is_finite() && *l <= 0.0));
    /// assert!((ln_x.iter().map(|l| l.exp()).sum::<f64>() - 1.0).abs() < 1e-12);
    /// ```
    pub fn sample_log(&self, rng: &mut dyn RngCore) -> Vec<f64> {
        match self.shifted_log_gammas(rng) {
            Ok(shifted) => {
                let ln_total = log_sum_exp(&shifted);
                shifted.into_iter().map(|l| l - ln_total).collect()
            }
            Err(vertex) => (0..self.alpha.len())
                .map(|i| if i == vertex { 0.0 } else { f64::NEG_INFINITY })
                .collect(),
        }
    }

    /// `ln Gᵢ − maxⱼ ln Gⱼ` for independent `Gᵢ ~ Gamma(αᵢ, 1)`, or `Err(i)`
    /// with a vertex index when every `ln Gᵢ` is `-∞`.
    fn shifted_log_gammas(&self, rng: &mut dyn RngCore) -> Result<Vec<f64>, usize> {
        let ln_g: Vec<f64> = self
            .alpha
            .iter()
            .map(|&a| ln_gamma_variate(a, rng))
            .collect();
        let max = ln_g.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if max == f64::NEG_INFINITY {
            return Err(vertex_index(&self.alpha, rng));
        }
        Ok(ln_g.into_iter().map(|l| l - max).collect())
    }
}
impl Distribution<Vec<f64>> for Dirichlet {
    fn sample(&self, rng: &mut dyn RngCore) -> Vec<f64> {
        match self.shifted_log_gammas(rng) {
            Ok(shifted) => {
                // The largest weight is exactly 1, so the total is in [1, K]:
                // no overflow, and the division cannot produce NaN.
                let weights: Vec<f64> = shifted.into_iter().map(f64::exp).collect();
                let total: f64 = weights.iter().sum();
                weights.into_iter().map(|w| w / total).collect()
            }
            Err(vertex) => (0..self.alpha.len())
                .map(|i| if i == vertex { 1.0 } else { 0.0 })
                .collect(),
        }
    }
    fn log_prob(&self, x: &Vec<f64>) -> LogF64 {
        dirichlet_log_density(&self.alpha, self.ln_beta, x)
    }
    fn clone_box(&self) -> Box<dyn Distribution<Vec<f64>>> {
        Box::new(self.clone())
    }
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
}

/// The multinomial distribution `Mult(n, p)`: the category counts of `n`
/// independent draws with probabilities `p`, as a `Distribution<Vec<u64>>`.
///
/// Mathematical Properties:
/// - **Support**: count vectors `c ∈ ℕᴷ` with `Σᵢ cᵢ = n`
/// - **PMF**: `P(c) = n! / Πᵢ cᵢ! · Πᵢ pᵢ^cᵢ` ([`multinomial_log_pmf`])
/// - **Mean**: `n·pᵢ`
/// - **Variance**: `n·pᵢ(1 − pᵢ)`; covariance `−n·pᵢpⱼ`
///
/// **Not a site distribution**: `Vec<u64>` is not a
/// [`SampleType`](crate::SampleType), so it cannot be passed to
/// [`sample`] or [`observe`](crate::observe). Use it standalone,
/// or enter a count likelihood into a model with [`factor`]:
/// `factor(multinomial_log_pmf(&p, &counts)?)`, or, with the probabilities
/// integrated out under a Dirichlet prior,
/// `factor(dirichlet_multinomial_log_marginal(&alpha, &counts)?)`.
///
/// **Sampling** draws the counts one category at a time from conditional
/// binomials, `cᵢ ~ Binomial(n − c₁ − … − cᵢ₋₁, pᵢ / (pᵢ + … + p_K))`, so it
/// costs `O(K)` binomial draws whatever `n` is. **Scoring**: `log_prob(c)` is
/// [`multinomial_log_pmf`], and a vector of the wrong length or with a total
/// other than `n` is outside the support (`-∞`).
///
/// # Example
///
/// ```rust
/// # use fugue::*;
/// # use rand::rngs::StdRng;
/// # use rand::SeedableRng;
/// let m = Multinomial::new(20, vec![0.2, 0.3, 0.5]).unwrap();
/// let mut rng = StdRng::seed_from_u64(7);
/// let counts = m.sample(&mut rng);
/// assert_eq!(counts.iter().sum::<u64>(), 20);
/// assert!(m.log_prob(&counts).is_finite());
///
/// // A count vector with the wrong total is outside the support.
/// assert_eq!(m.log_prob(&vec![1, 1, 1]), f64::NEG_INFINITY);
/// ```
#[derive(Clone, Debug)]
pub struct Multinomial {
    /// Number of draws.
    n: u64,
    /// Category probabilities (validated as a probability vector in `new`).
    probs: Vec<f64>,
}
impl Multinomial {
    /// Create a multinomial distribution over `n` draws with probabilities
    /// `probs`.
    ///
    /// # Errors
    ///
    /// `FugueError::InvalidParameters` (`InvalidProbability`) if `probs` is empty,
    /// has a negative or non-finite entry, or sums further than
    /// [`SIMPLEX_TOLERANCE`] from 1 (the same rules as
    /// [`Categorical::new`](crate::Categorical::new)).
    ///
    /// ```rust
    /// # use fugue::*;
    /// assert!(Multinomial::new(10, vec![0.5, 0.5]).is_ok());
    /// assert!(Multinomial::new(10, vec![0.5, 0.6]).is_err());
    /// ```
    pub fn new(n: u64, probs: Vec<f64>) -> FugueResult<Self> {
        validate_probabilities("Multinomial", &probs)?;
        Ok(Multinomial { n, probs })
    }

    /// The number of draws `n`.
    ///
    /// ```rust
    /// # use fugue::*;
    /// assert_eq!(Multinomial::new(12, vec![1.0]).unwrap().n(), 12);
    /// ```
    pub fn n(&self) -> u64 {
        self.n
    }

    /// The category probabilities `p`.
    ///
    /// ```rust
    /// # use fugue::*;
    /// let m = Multinomial::new(3, vec![0.25, 0.75]).unwrap();
    /// assert_eq!(m.probs(), &[0.25, 0.75]);
    /// ```
    pub fn probs(&self) -> &[f64] {
        &self.probs
    }

    /// The number of categories `K`.
    ///
    /// ```rust
    /// # use fugue::*;
    /// assert_eq!(Multinomial::new(3, vec![0.25, 0.75]).unwrap().len(), 2);
    /// ```
    pub fn len(&self) -> usize {
        self.probs.len()
    }

    /// Whether there are no categories. Always `false` for a validly
    /// constructed distribution (kept for clippy's `len`/`is_empty` pairing).
    ///
    /// ```rust
    /// # use fugue::*;
    /// assert!(!Multinomial::new(3, vec![1.0]).unwrap().is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.probs.is_empty()
    }
}
impl Distribution<Vec<u64>> for Multinomial {
    fn sample(&self, rng: &mut dyn RngCore) -> Vec<u64> {
        let k = self.probs.len();
        // Suffix sums `tail[i] = pᵢ + … + p_K`, so every conditional
        // probability `pᵢ / tail[i]` lies in [0, 1] in floating point, and the
        // last category with positive probability gets exactly 1.
        let mut tail = vec![0.0; k + 1];
        for i in (0..k).rev() {
            tail[i] = tail[i + 1] + self.probs[i];
        }
        let mut counts = vec![0u64; k];
        let mut remaining = self.n;
        for i in 0..k {
            if remaining == 0 || tail[i] <= 0.0 {
                break;
            }
            let q = (self.probs[i] / tail[i]).min(1.0);
            let c = if q >= 1.0 {
                remaining
            } else {
                RDBinomial::new(remaining, q)
                    .expect("conditional probability is in [0, 1]")
                    .sample(rng)
            };
            counts[i] = c;
            remaining -= c;
        }
        // Unreachable for a validated `p` (see `tail`), kept so the total is
        // `n` by construction.
        if remaining > 0 {
            if let Some(j) = self.probs.iter().rposition(|&p| p > 0.0) {
                counts[j] += remaining;
            }
        }
        counts
    }
    fn log_prob(&self, x: &Vec<u64>) -> LogF64 {
        if x.len() != self.probs.len() {
            return f64::NEG_INFINITY;
        }
        match x.iter().try_fold(0u64, |acc, &c| acc.checked_add(c)) {
            Some(total) if total == self.n => multinomial_log_mass(&self.probs, x),
            _ => f64::NEG_INFINITY,
        }
    }
    fn clone_box(&self) -> Box<dyn Distribution<Vec<u64>>> {
        Box::new(self.clone())
    }
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
}

// =============================================================================
// Gamma normalization: a Dirichlet draw from K scalar sites
// =============================================================================

/// Normalize non-negative weights onto the simplex by max-scaling (the largest
/// becomes exactly 1, so the total is in `[1, K]`: no overflow, no `NaN`), or
/// `None` if they do not define a point of the simplex: a negative or
/// non-finite weight, or all weights zero.
fn normalize_weights(w: &[f64]) -> Option<Vec<f64>> {
    let mut max = 0.0_f64;
    for &v in w {
        if !(v >= 0.0 && v.is_finite()) {
            return None;
        }
        max = max.max(v);
    }
    if max <= 0.0 {
        return None;
    }
    let scaled: Vec<f64> = w.iter().map(|&v| v / max).collect();
    let total: f64 = scaled.iter().sum();
    Some(scaled.into_iter().map(|v| v / total).collect())
}

/// The continuation of [`sample_dirichlet`]: normalize the Gamma draws, or
/// zero the trace's weight and return the uniform vector if they do not
/// define a point of the simplex.
fn normalize_gamma_sites(g: Vec<f64>) -> Model<Vec<f64>> {
    match normalize_weights(&g) {
        Some(x) => pure(x),
        None => {
            let k = g.len();
            factor(f64::NEG_INFINITY).map(move |()| vec![1.0 / k as f64; k])
        }
    }
}

/// Draw a `Dirichlet(α)` vector inside a model from `K` scalar Gamma sites.
///
/// The model samples `gᵢ ~ Gamma(αᵢ, 1)` at the address `addr!(name, i)` for
/// `i = 0..K` (built with the `addr!` macro itself, so the addresses are
/// byte-identical to yours), sequenced with
/// [`sequence_vec`], and returns `xᵢ = gᵢ / Σⱼ gⱼ`, which is
/// exactly `Dirichlet(α)`-distributed. No new site type is involved: every
/// handler and inference algorithm sees `K` ordinary `f64` sites.
///
/// **What the trace holds.** The `K` choices are the Gamma draws `gᵢ`, not
/// the simplex coordinates, each with `logp = Gamma(αᵢ, 1).log_prob(gᵢ)`, and
/// `log_prior` gains their joint density
///
/// ```text
/// Σᵢ [(αᵢ − 1)·ln gᵢ − gᵢ − lnΓ(αᵢ)]
///   = ln Dir(x | α) + ln Gamma(S | A, 1) − (K − 1)·ln S,   S = Σᵢ gᵢ
/// ```
///
/// That is **not** the Dirichlet density of the returned vector: it also
/// carries the density of the total `S` and the Jacobian `S^(K−1)` of
/// `g ↦ (x, S)`. Because the sites are scored with their own density,
/// importance weights, MH acceptance ratios and SMC weights computed from the
/// trace are correct over `(g₁, …, g_K)`, and `S`, a nuisance dimension with
/// a proper `Gamma(A, 1)` prior, integrates out. When you need `ln Dir(x | α)`
/// itself, call [`dirichlet_log_pdf`] on the returned vector.
///
/// **Underflow.** A `Gamma(αᵢ, 1)` draw falls below the smallest positive
/// `f64` (about `4.9·10⁻³²⁴`) with probability about `e^(−744·αᵢ)`:
/// negligible (below 10⁻¹⁶) for `αᵢ ≥ 0.05`, about `6·10⁻⁴` at `αᵢ = 0.01`,
/// and about one half at `αᵢ = 10⁻³`. Such a draw is recorded as `0.0`, where
/// the Gamma density is zero, so that trace has `log_prior = -∞` and
/// inference discards it. If every draw underflows (probability about
/// `e^(−744·Σα)`, so only when the total concentration is below about 0.05),
/// the proportions are undefined: the model then adds `factor(-∞)` (the
/// crate's rule that an undefined weight is probability zero, FG-N2) and
/// returns the uniform vector `[1/K; K]`, so the program keeps running on a
/// valid simplex point. The same fallback applies when a replayed trace holds
/// values that do not define a simplex point (negative or non-finite). For
/// concentrations below about 0.05, integrate the simplex out with the
/// conjugate helpers instead, or sample it outside the model with
/// [`Dirichlet`], which samples in log space.
///
/// # Errors
///
/// `FugueError::InvalidParameters` for an invalid `alpha`, as for
/// [`Dirichlet::new`]. Like a distribution constructor, the check happens when
/// the model is built, so a bad `α` surfaces as a structured error instead of
/// a zero-weight trace; for an `α` computed from other sites, map the error to
/// `factor(f64::NEG_INFINITY)` yourself if that is the semantics you want.
///
/// # Example
///
/// ```rust
/// # use fugue::*;
/// # use rand::rngs::StdRng;
/// # use rand::SeedableRng;
/// let alpha = [2.0, 3.0, 5.0];
/// let model = sample_dirichlet("w", &alpha).unwrap();
/// let mut rng = StdRng::seed_from_u64(1);
/// let (w, trace) = runtime::handler::run(
///     PriorHandler { rng: &mut rng, trace: Trace::default() },
///     model,
/// );
/// assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-12);
///
/// // The trace holds the Gamma draws, not the simplex coordinates...
/// let g: Vec<f64> = (0..3).map(|i| trace.get_f64(&addr!("w", i)).unwrap()).collect();
/// let s: f64 = g.iter().sum();
/// assert!((w[0] - g[0] / s).abs() < 1e-12);
///
/// // ...and their joint Gamma density as log_prior.
/// let lp: f64 = alpha
///     .iter()
///     .zip(&g)
///     .map(|(&a, gi)| Gamma::new(a, 1.0).unwrap().log_prob(gi))
///     .sum();
/// assert!((trace.log_prior - lp).abs() < 1e-9);
/// ```
pub fn sample_dirichlet(name: &str, alpha: &[f64]) -> FugueResult<Model<Vec<f64>>> {
    validate_concentrations("Dirichlet", alpha)?;
    let sites = alpha
        .iter()
        .enumerate()
        .map(|(i, &a)| Ok(sample(addr!(name, i), Gamma::new(a, 1.0)?)))
        .collect::<FugueResult<Vec<Model<f64>>>>()?;
    Ok(sequence_vec(sites).bind(normalize_gamma_sites))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::address::Address;
    use crate::core::distribution::{Beta, Binomial, Categorical, Normal};
    use crate::core::model::{observe, traverse_vec};
    use crate::error::Validate;
    use crate::inference::mh::adaptive_mcmc_chain;
    use crate::runtime::handler::run;
    use crate::runtime::interpreters::{PriorHandler, ScoreGivenTrace};
    use crate::runtime::trace::{ChoiceValue, Trace};
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use std::f64::consts::{FRAC_PI_2, LN_2};

    #[test]
    fn the_vector_distributions_downcast_to_themselves() {
        let d: Box<dyn Distribution<Vec<f64>>> = Box::new(Dirichlet::new(vec![1.0, 2.0]).unwrap());
        assert_eq!(
            d.downcast_ref::<Dirichlet>().map(|d| d.alpha().to_vec()),
            Some(vec![1.0, 2.0])
        );
        let m: Box<dyn Distribution<Vec<u64>>> =
            Box::new(Multinomial::new(3, vec![0.5, 0.5]).unwrap());
        assert_eq!(m.downcast_ref::<Multinomial>().map(|m| m.n()), Some(3));
        assert!(m.clone_box().downcast_ref::<Multinomial>().is_some());
    }

    const NEG_INF: f64 = f64::NEG_INFINITY;

    /// `|got − want| ≤ tol`.
    fn close(got: f64, want: f64, tol: f64) {
        assert!(
            (got - want).abs() <= tol,
            "expected {want}, got {got} (|diff| = {:e}, tol {tol:e})",
            (got - want).abs()
        );
    }

    /// `|got − want| ≤ tol·max(1, |want|)`: relative, with an absolute floor.
    fn close_rel(got: f64, want: f64, tol: f64) {
        close(got, want, tol * want.abs().max(1.0));
    }

    fn lgamma(x: f64) -> f64 {
        libm::lgamma(x)
    }

    /// The error code of a result expected to be an error (works for any `T`,
    /// including `Model`, which is not `Debug`).
    fn err_code<T>(result: FugueResult<T>) -> ErrorCode {
        match result {
            Ok(_) => panic!("expected an error"),
            Err(e) => e.code(),
        }
    }

    fn prior_run<A>(seed: u64, model: Model<A>) -> (A, Trace) {
        let mut rng = StdRng::seed_from_u64(seed);
        run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            model,
        )
    }

    // -------------------------------------------------------------------------
    // Numerical kernels
    // -------------------------------------------------------------------------

    // References: mpmath at 80 digits, `loggamma(z + a) - loggamma(z + b)`.
    #[test]
    fn ln_gamma_ratio_matches_high_precision_references() {
        close_rel(ln_gamma_ratio(1e9, 1e-8, 1.0), -20.723265629713755, 4e-15);
        close_rel(ln_gamma_ratio(12.5, 0.5, 1.0), -1.272861660582815, 4e-15);
        close_rel(ln_gamma_ratio(1e15, 2.5, 0.0), 86.34694098727671, 4e-15);
        close_rel(ln_gamma_ratio(1e6, 1e3, 1.0), 13802.194381072677, 4e-15);
        close_rel(ln_gamma_ratio(10.0, 1e12, 0.0), 26631021116179.16, 4e-15);
        // Below the Stirling threshold: plain lnΓ values.
        close_rel(ln_gamma_ratio(3.0, 1e-3, 2.5), -3.2637438052823198, 4e-15);
        // Exact identities: Γ(z + 1) = z·Γ(z), and a zero shift.
        for &z in &[10.5, 1e3, 1e9, 1e15] {
            close_rel(ln_gamma_ratio(z, 1.0, 0.0), z.ln(), 4e-15);
            assert_eq!(ln_gamma_ratio(z, 0.25, 0.25), 0.0);
        }
        // Far apart, on the ln(x) − ln(y) branch (d / y close to −1):
        // lnΓ(10) − lnΓ(1e12) and lnΓ(20) − lnΓ(1e6).
        close_rel(ln_gamma_ratio(0.0, 10.0, 1e12), -26631021115902.85, 4e-15);
        close_rel(
            ln_gamma_ratio(10.0, 10.0, 1e6 - 10.0),
            -12815465.229263425,
            4e-15,
        );
    }

    #[test]
    fn ln_polya_coefficient_matches_exact_binomial_coefficients() {
        // ln C(n, k) = ln[Γ(n + 1) / (Γ(k + 1)·Γ(n − k + 1))] = lnC(k + 1, n − k).
        for n in 0u64..=60 {
            let mut c: u128 = 1;
            for k in 0..=n {
                if k > 0 {
                    c = c * u128::from(n - k + 1) / u128::from(k);
                }
                close_rel(
                    ln_polya_coefficient(k as f64 + 1.0, (n - k) as f64),
                    (c as f64).ln(),
                    1e-14,
                );
            }
        }
    }

    // -------------------------------------------------------------------------
    // Special functions and known log-densities
    // -------------------------------------------------------------------------

    #[test]
    fn ln_multivariate_beta_known_values() {
        // B(1, 1, 1) = 1/2, B(2, 3) = 1/12, and B(a) = 1 for K = 1.
        close(
            ln_multivariate_beta(&[1.0, 1.0, 1.0]).unwrap(),
            -LN_2,
            1e-15,
        );
        close(
            ln_multivariate_beta(&[2.0, 3.0]).unwrap(),
            -(12f64.ln()),
            1e-14,
        );
        close(ln_multivariate_beta(&[3.7]).unwrap(), 0.0, 1e-15);
        // K = 2 agrees with the lnΓ formula `Beta::log_prob` normalizes with.
        for &(a, b) in &[(0.5, 0.5), (2.0, 5.0), (0.3, 7.5), (12.0, 40.0)] {
            close(
                ln_multivariate_beta(&[a, b]).unwrap(),
                lgamma(a) + lgamma(b) - lgamma(a + b),
                1e-12,
            );
        }
        // One dominant concentration: B(1e9, 1) = 1/1e9 exactly.
        close_rel(
            ln_multivariate_beta(&[1e9, 1.0]).unwrap(),
            -(1e9f64.ln()),
            4e-15,
        );
        // mpmath references.
        close_rel(
            ln_multivariate_beta(&[0.3, 7.0, 1e5]).unwrap(),
            -76.36953663224247,
            1e-14,
        );
        close_rel(
            ln_multivariate_beta(&[1e-8, 1e-8]).unwrap(),
            19.11382792451231,
            1e-14,
        );
        close_rel(
            ln_multivariate_beta(&[1e9, 1e9]).unwrap(),
            -1386294370.2160115,
            1e-14,
        );
        close_rel(
            ln_multivariate_beta(&[2.5, 0.7, 11.0, 1e-4]).unwrap(),
            1.7857625624285316,
            1e-13,
        );
    }

    #[test]
    fn ln_multinomial_coefficient_known_values() {
        close(
            ln_multinomial_coefficient(&[2, 1, 1]).unwrap(),
            12f64.ln(),
            1e-14,
        );
        close(ln_multinomial_coefficient(&[0, 0, 0]).unwrap(), 0.0, 0.0);
        close(ln_multinomial_coefficient(&[7]).unwrap(), 0.0, 0.0);
        // mpmath: loggamma(3e12 + 1) − 3·loggamma(1e12 + 1).
        close_rel(
            ln_multinomial_coefficient(&[1_000_000_000_000; 3]).unwrap(),
            3295836865975.4097,
            1e-14,
        );
        // One dominant count: C(1e9, 1) = 1e9.
        close_rel(
            ln_multinomial_coefficient(&[999_999_999, 1]).unwrap(),
            1e9f64.ln(),
            4e-15,
        );
    }

    #[test]
    fn uniform_dirichlet_density_is_ln_2_on_the_triangle() {
        let alpha = [1.0, 1.0, 1.0];
        let third = 1.0 / 3.0;
        for x in [
            [0.2, 0.3, 0.5],
            [third, third, third],
            [0.98, 0.01, 0.01],
            [1e-12, 0.5, 0.5 - 1e-12],
        ] {
            close(dirichlet_log_pdf(&alpha, &x).unwrap(), LN_2, 1e-14);
        }
        // With α = 1 a zero coordinate is harmless: still ln 2 on the edges.
        close(
            dirichlet_log_pdf(&alpha, &[0.0, 0.4, 0.6]).unwrap(),
            LN_2,
            1e-14,
        );
        close(
            dirichlet_log_pdf(&alpha, &[1.0, 0.0, 0.0]).unwrap(),
            LN_2,
            1e-14,
        );
    }

    #[test]
    fn two_category_dirichlet_is_the_beta_distribution() {
        for &(a, b) in &[
            (1.0, 1.0),
            (2.0, 5.0),
            (0.5, 0.5),
            (0.3, 2.7),
            (40.0, 3.5),
            (1e-3, 1e-3),
        ] {
            let beta = Beta::new(a, b).unwrap();
            for &x in &[1e-9, 0.01, 0.25, 0.5, 0.8, 0.999] {
                let want = beta.log_prob(&x);
                close_rel(
                    dirichlet_log_pdf(&[a, b], &[x, 1.0 - x]).unwrap(),
                    want,
                    1e-12,
                );
            }
            // The boundary follows Beta's semantics exactly.
            assert_eq!(
                dirichlet_log_pdf(&[a, b], &[0.0, 1.0]).unwrap(),
                beta.log_prob(&0.0)
            );
            assert_eq!(
                dirichlet_log_pdf(&[a, b], &[1.0, 0.0]).unwrap(),
                beta.log_prob(&1.0)
            );
        }
    }

    // References: mpmath, Σ (αᵢ − 1)·ln xᵢ − ln B(α).
    #[test]
    fn dirichlet_log_pdf_known_values() {
        close_rel(
            dirichlet_log_pdf(&[2.0, 3.0, 4.0], &[0.2, 0.3, 0.5]).unwrap(),
            2.0228711901914416,
            1e-14,
        );
        close_rel(
            dirichlet_log_pdf(&[0.5; 4], &[0.1, 0.2, 0.3, 0.4]).unwrap(),
            0.7266834991153182,
            1e-14,
        );
        // A coordinate at 1e-300 with α = 1e-3: large, finite, positive.
        close_rel(
            dirichlet_log_pdf(&[1e-3, 1e-3], &[1e-300, 1.0]).unwrap(),
            682.4838515533071,
            1e-14,
        );
        // K = 1 is the point mass at (1).
        close(dirichlet_log_pdf(&[4.2], &[1.0]).unwrap(), 0.0, 1e-15);
    }

    #[test]
    fn dirichlet_log_pdf_support_and_boundaries() {
        let alpha = [2.0, 0.5, 1.0];
        // Off the simplex, negative, or non-finite: -inf, not an error.
        assert_eq!(
            dirichlet_log_pdf(&alpha, &[0.5, 0.5, 0.5]).unwrap(),
            NEG_INF
        );
        assert_eq!(
            dirichlet_log_pdf(&alpha, &[1.2, -0.1, -0.1]).unwrap(),
            NEG_INF
        );
        assert_eq!(
            dirichlet_log_pdf(&alpha, &[f64::NAN, 0.5, 0.5]).unwrap(),
            NEG_INF
        );
        assert_eq!(
            dirichlet_log_pdf(&alpha, &[f64::INFINITY, 0.0, 0.0]).unwrap(),
            NEG_INF
        );
        // Within SIMPLEX_TOLERANCE of the simplex is on it; beyond is not.
        let inside = [0.5 + 0.5 * SIMPLEX_TOLERANCE, 0.25, 0.25];
        assert!(dirichlet_log_pdf(&alpha, &inside).unwrap().is_finite());
        let outside = [0.5 + 2.0 * SIMPLEX_TOLERANCE, 0.25, 0.25];
        assert_eq!(dirichlet_log_pdf(&alpha, &outside).unwrap(), NEG_INF);
        // Zero coordinates: αᵢ > 1 vanishes, αᵢ < 1 diverges, αᵢ = 1 is neutral,
        // and a vanishing factor beats a diverging one.
        assert_eq!(
            dirichlet_log_pdf(&alpha, &[0.0, 0.5, 0.5]).unwrap(),
            NEG_INF
        );
        assert_eq!(
            dirichlet_log_pdf(&alpha, &[0.5, 0.0, 0.5]).unwrap(),
            f64::INFINITY
        );
        assert!(dirichlet_log_pdf(&alpha, &[0.5, 0.5, 0.0])
            .unwrap()
            .is_finite());
        assert_eq!(
            dirichlet_log_pdf(&alpha, &[0.0, 0.0, 1.0]).unwrap(),
            NEG_INF
        );
        // A wrong length is an error for the function and -inf for the type.
        assert_eq!(
            err_code(dirichlet_log_pdf(&alpha, &[0.5, 0.5])),
            ErrorCode::InvalidCount
        );
        let d = Dirichlet::new(alpha.to_vec()).unwrap();
        assert_eq!(d.log_prob(&vec![0.5, 0.5]), NEG_INF);
    }

    #[test]
    fn multinomial_matches_the_factorial_formula() {
        fn factorial(n: u64) -> f64 {
            (1..=n).map(|k| k as f64).product()
        }
        let p: [f64; 4] = [0.1, 0.25, 0.4, 0.25];
        for counts in [
            [0u64, 0, 0, 0],
            [1, 0, 0, 0],
            [2, 1, 3, 0],
            [5, 5, 5, 5],
            [0, 7, 1, 2],
            [3, 0, 0, 9],
        ] {
            let n: u64 = counts.iter().sum();
            let coef = factorial(n) / counts.iter().map(|&c| factorial(c)).product::<f64>();
            let prob = coef
                * p.iter()
                    .zip(&counts)
                    .map(|(&pi, &c)| pi.powi(c as i32))
                    .product::<f64>();
            close_rel(multinomial_log_pmf(&p, &counts).unwrap(), prob.ln(), 1e-13);
        }
        // One draw is a categorical draw.
        let cat = Categorical::new(p.to_vec()).unwrap();
        for i in 0..4 {
            let mut one_hot = [0u64; 4];
            one_hot[i] = 1;
            close(
                multinomial_log_pmf(&p, &one_hot).unwrap(),
                cat.log_prob(&i),
                1e-15,
            );
        }
        // Two categories are the binomial.
        let binomial = Binomial::new(17, 0.3).unwrap();
        for k in 0..=17u64 {
            close_rel(
                multinomial_log_pmf(&[0.3, 0.7], &[k, 17 - k]).unwrap(),
                binomial.log_prob(&k),
                1e-12,
            );
        }
    }

    #[test]
    fn multinomial_zero_probability_categories() {
        // An unobserved zero-probability category contributes nothing.
        close(
            multinomial_log_pmf(&[0.5, 0.0, 0.5], &[2, 0, 1]).unwrap(),
            multinomial_log_pmf(&[0.5, 0.5], &[2, 1]).unwrap(),
            1e-15,
        );
        // An observed one makes the count vector impossible.
        assert_eq!(
            multinomial_log_pmf(&[0.5, 0.0, 0.5], &[2, 1, 1]).unwrap(),
            NEG_INF
        );
        // A degenerate p puts all mass on one count vector.
        close(multinomial_log_pmf(&[1.0, 0.0], &[9, 0]).unwrap(), 0.0, 0.0);
        assert_eq!(multinomial_log_pmf(&[1.0, 0.0], &[8, 1]).unwrap(), NEG_INF);
    }

    // -------------------------------------------------------------------------
    // Marginal likelihoods against brute force
    // -------------------------------------------------------------------------

    #[test]
    fn beta_binomial_matches_the_lgamma_formula_and_sums_to_one() {
        let n = 20u64;
        for &(a, b) in &[(1.0, 1.0), (0.5, 2.0), (2.5, 0.7), (0.5, 0.5), (7.0, 3.0)] {
            let mut terms = Vec::new();
            for k in 0..=n {
                let (kf, nf) = (k as f64, n as f64);
                // Plain lnΓ formula: accurate at this size.
                let formula = lgamma(nf + 1.0) - lgamma(kf + 1.0) - lgamma(nf - kf + 1.0)
                    + lgamma(a + kf)
                    + lgamma(b + nf - kf)
                    - lgamma(a + b + nf)
                    - (lgamma(a) + lgamma(b) - lgamma(a + b));
                let got = beta_binomial_log_marginal(a, b, k, n).unwrap();
                close(got, formula, 1e-12);
                terms.push(got);
            }
            close(log_sum_exp(&terms), 0.0, 1e-13);
        }
    }

    // P(k | n, a, b) = ∫₀¹ Binomial(k | n, p)·Beta(p | a, b) dp. With p = sin²θ
    // the integrand is 2·C(n, k)/B(a, b)·sin^(2(k+a)−1)θ·cos^(2(n−k+b)−1)θ on
    // [0, π/2], which is smooth for these cases (both exponents ≥ 0), so
    // composite Simpson with 20 000 panels is accurate far below the 1e-9
    // relative tolerance.
    #[test]
    fn beta_binomial_matches_numerical_integration() {
        fn simpson(f: impl Fn(f64) -> f64, lo: f64, hi: f64, panels: usize) -> f64 {
            let h = (hi - lo) / panels as f64;
            let inner: f64 = (1..panels)
                .map(|i| {
                    let w = if i % 2 == 1 { 4.0 } else { 2.0 };
                    w * f(lo + i as f64 * h)
                })
                .sum();
            h / 3.0 * (f(lo) + inner + f(hi))
        }
        for &(a, b, k, n) in &[
            (0.5, 2.0, 7u64, 20u64),
            (2.5, 0.7, 13, 20),
            (1.0, 1.0, 5, 10),
            (0.5, 0.5, 0, 10),
            (3.0, 4.5, 20, 20),
        ] {
            let e1 = 2.0 * (k as f64 + a) - 1.0;
            let e2 = 2.0 * ((n - k) as f64 + b) - 1.0;
            let integral = simpson(
                |t: f64| 2.0 * t.sin().powf(e1) * t.cos().powf(e2),
                0.0,
                FRAC_PI_2,
                20_000,
            );
            let want = ln_multinomial_coefficient(&[k, n - k]).unwrap()
                - ln_multivariate_beta(&[a, b]).unwrap()
                + integral.ln();
            close_rel(beta_binomial_log_marginal(a, b, k, n).unwrap(), want, 1e-9);
        }
    }

    #[test]
    fn beta_binomial_uniform_prior_is_uniform_at_any_scale() {
        // Beta(1, 1): P(k | n) = 1/(n + 1) for every k, exactly, even at 10¹²,
        // where the plain lnΓ formula is off by about 2e-3.
        for &n in &[0u64, 1, 10, 1_000_000, 1_000_000_000, 1_000_000_000_000] {
            for &k in &[0, n / 3, n / 2, n] {
                close_rel(
                    beta_binomial_log_marginal(1.0, 1.0, k, n).unwrap(),
                    -(n as f64 + 1.0).ln(),
                    4e-15,
                );
            }
        }
    }

    #[test]
    fn categorical_marginal_is_the_polya_urn_product() {
        let sequence = [0usize, 2, 2, 1, 0, 2, 2, 2, 0, 1, 3, 2, 2, 0];
        for alpha in [
            vec![1.0, 1.0, 1.0, 1.0],
            vec![0.5, 1.5, 0.25, 3.0],
            vec![1e-3, 1e-3, 1e-3, 1e-3],
            vec![50.0, 20.0, 5.0, 0.1],
        ] {
            let a_total: f64 = alpha.iter().sum();
            let mut counts = [0u64; 4];
            let mut urn = 0.0;
            for (t, &c) in sequence.iter().enumerate() {
                let by_hand = (alpha[c] + counts[c] as f64) / (a_total + t as f64);
                // The helper's predictive is the urn's next-draw probability.
                let p = dirichlet_predictive(&alpha, &counts).unwrap();
                close_rel(p[c], by_hand, 1e-14);
                urn += by_hand.ln();
                counts[c] += 1;
            }
            close_rel(
                dirichlet_categorical_log_marginal(&alpha, &counts).unwrap(),
                urn,
                1e-13,
            );
        }
    }

    #[test]
    fn marginals_sum_to_one_by_enumeration() {
        // Over count vectors for the Dirichlet–multinomial...
        for alpha in [
            vec![1.0, 1.0, 1.0],
            vec![0.5, 2.0, 1e-3],
            vec![30.0, 0.2, 4.0],
        ] {
            for n in [0u64, 1, 6, 11] {
                let mut terms = Vec::new();
                for n0 in 0..=n {
                    for n1 in 0..=(n - n0) {
                        let counts = [n0, n1, n - n0 - n1];
                        terms.push(dirichlet_multinomial_log_marginal(&alpha, &counts).unwrap());
                    }
                }
                close(log_sum_exp(&terms), 0.0, 1e-13);
            }
        }
        let alpha = [1e-3, 2.0, 0.5, 30.0];
        let n = 5u64;
        let mut terms = Vec::new();
        for a in 0..=n {
            for b in 0..=(n - a) {
                for c in 0..=(n - a - b) {
                    let counts = [a, b, c, n - a - b - c];
                    terms.push(dirichlet_multinomial_log_marginal(&alpha, &counts).unwrap());
                }
            }
        }
        close(log_sum_exp(&terms), 0.0, 1e-13);

        // ...and over all K^N sequences for the Dirichlet–categorical.
        let alpha = [0.5, 1.5, 0.25];
        let len = 6u32;
        let mut terms = Vec::new();
        for code in 0..3usize.pow(len) {
            let mut counts = [0u64; 3];
            let mut rest = code;
            for _ in 0..len {
                counts[rest % 3] += 1;
                rest /= 3;
            }
            terms.push(dirichlet_categorical_log_marginal(&alpha, &counts).unwrap());
        }
        close(log_sum_exp(&terms), 0.0, 1e-13);
    }

    #[test]
    fn two_category_marginals_are_the_beta_ones() {
        for &(a, b) in &[(0.5, 2.0), (1e-3, 7.0), (30.0, 30.0)] {
            for &(k, n) in &[(0u64, 0u64), (3, 10), (10, 10), (400, 1000)] {
                let bb = beta_binomial_log_marginal(a, b, k, n).unwrap();
                let bbern = beta_bernoulli_log_marginal(a, b, k, n - k).unwrap();
                close(
                    dirichlet_multinomial_log_marginal(&[a, b], &[k, n - k]).unwrap(),
                    bb,
                    1e-12,
                );
                close(
                    dirichlet_categorical_log_marginal(&[a, b], &[k, n - k]).unwrap(),
                    bbern,
                    1e-12,
                );
                close_rel(
                    bb - bbern,
                    ln_multinomial_coefficient(&[k, n - k]).unwrap(),
                    1e-12,
                );
            }
        }
    }

    // -------------------------------------------------------------------------
    // Stability: tiny concentrations and huge counts
    // -------------------------------------------------------------------------

    // References: mpmath at 80 digits, from the lnΓ definitions. The plain
    // lnΓ formula misses the 10⁹ and 10¹² cases by 1e-6 to 1e-2.
    #[test]
    fn marginals_match_references_for_tiny_alpha_and_huge_counts() {
        type Case = (&'static [f64], &'static [u64], f64, f64);
        let cases: [Case; 10] = [
            (
                &[1e-8, 1e-8],
                &[1_000_000_000, 0],
                -0.6931473935647601,
                -0.6931473935647601,
            ),
            (
                &[1e-8, 1e-8, 1e-8],
                &[1_000_000_000, 3, 0],
                -20.617905729298215,
                -80.9959437769094,
            ),
            (
                &[1e-3, 1e-3, 1e-3],
                &[1_000_000, 1_000_000, 1],
                -28.052952146279214,
                -1386329.4426099285,
            ),
            (
                &[0.5, 0.5, 0.5],
                &[300_000, 300_000, 400_000],
                -27.80678153290809,
                -1088913.790856622,
            ),
            (
                &[30.0, 30.0, 30.0],
                &[1_000_000_000_000; 3],
                -53.15682287497383,
                -3295836866028.5664,
            ),
            (
                &[1e9, 1e9],
                &[1, 2],
                -0.9808292535117262,
                -2.079441542179836,
            ),
            (
                &[2.5, 0.7, 11.0, 1e-4],
                &[3, 0, 17, 1],
                -12.21625409320739,
                -22.299560072319355,
            ),
            (
                &[1e-8; 4],
                &[1_000_000, 0, 0, 0],
                -1.3862947929016611,
                -1.3862947929016611,
            ),
            (
                &[1e-8; 4],
                &[1_000_000, 1_000_000, 0, 7],
                -53.295926166570936,
                -1386433.212382519,
            ),
            (
                &[0.5, 2.0],
                &[1_000_000_000, 2_000_000_000],
                -21.965719162090412,
                -1909542515.7723186,
            ),
        ];
        for (alpha, counts, dm_ref, dc_ref) in cases {
            close_rel(
                dirichlet_multinomial_log_marginal(alpha, counts).unwrap(),
                dm_ref,
                1e-13,
            );
            close_rel(
                dirichlet_categorical_log_marginal(alpha, counts).unwrap(),
                dc_ref,
                1e-13,
            );
        }
        close_rel(
            beta_binomial_log_marginal(1.0, 1.0, 5, 1_000_000_000_000).unwrap(),
            -27.63102111592955,
            1e-14,
        );
        close_rel(
            beta_binomial_log_marginal(0.5, 0.5, 1_000_000_000, 2_000_000_000).unwrap(),
            -21.867995723045812,
            1e-13,
        );
        close_rel(
            beta_binomial_log_marginal(1e-8, 1e-8, 0, 1_000_000_000).unwrap(),
            -0.6931473935647601,
            1e-13,
        );
    }

    #[test]
    fn marginals_match_the_lgamma_formula_where_it_is_accurate() {
        // Up to 10⁶ the plain lnΓ difference is good to about 1e-8, which makes
        // it a usable reference across a grid that includes tiny α.
        let naive = |alpha: &[f64], counts: &[u64]| -> f64 {
            let a_total: f64 = alpha.iter().sum();
            let n_total = counts.iter().sum::<u64>() as f64;
            let mut out = lgamma(n_total + 1.0) + lgamma(a_total) - lgamma(a_total + n_total);
            for (&a, &n) in alpha.iter().zip(counts) {
                let n = n as f64;
                out += lgamma(a + n) - lgamma(a) - lgamma(n + 1.0);
            }
            out
        };
        for &a in &[1e-8, 1e-3, 0.5, 3.0] {
            for counts in [
                [0u64, 0, 0],
                [1, 0, 0],
                [5, 2, 9],
                [1_000, 0, 7],
                [1_000_000, 3, 0],
                [400_000, 350_000, 250_000],
            ] {
                let alpha = [a, 2.0 * a, 0.5 * a];
                close(
                    dirichlet_multinomial_log_marginal(&alpha, &counts).unwrap(),
                    naive(&alpha, &counts),
                    1e-6,
                );
            }
        }
    }

    #[test]
    fn nothing_is_nan_across_an_extreme_parameter_grid() {
        let alphas = [5e-324, 1e-300, 1e-8, 1e-3, 0.5, 1.0, 7.5, 1e3, 1e8, 1e15];
        let counts = [
            0u64,
            1,
            17,
            1_000,
            1_000_000,
            1_000_000_000,
            1_000_000_000_000,
            u64::MAX / 4,
        ];
        for &a in &alphas {
            for &b in &alphas {
                for &n0 in &counts {
                    for &n1 in &counts {
                        let (alpha, c) = ([a, b], [n0, n1]);
                        let dm = dirichlet_multinomial_log_marginal(&alpha, &c).unwrap();
                        let dc = dirichlet_categorical_log_marginal(&alpha, &c).unwrap();
                        let bb = beta_binomial_log_marginal(a, b, n0, n0 + n1).unwrap();
                        assert!(
                            dm.is_finite() && dc.is_finite() && bb.is_finite(),
                            "alpha {alpha:?} counts {c:?}: {dm} {dc} {bb}"
                        );
                        let lp = dirichlet_log_predictive(&alpha, &c).unwrap();
                        assert!(lp.iter().all(|l| l.is_finite() && *l <= 1e-12));
                    }
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Posterior updates and predictives
    // -------------------------------------------------------------------------

    #[test]
    fn posterior_updates_and_predictives() {
        let (alpha, counts) = ([0.5, 1.0, 2.0], [3u64, 0, 7]);
        assert_eq!(
            dirichlet_posterior(&alpha, &counts).unwrap(),
            vec![3.5, 1.0, 9.0]
        );
        assert_eq!(beta_posterior(2.0, 3.0, 5, 1).unwrap(), (7.0, 4.0));

        let want = [3.5 / 13.5, 1.0 / 13.5, 9.0 / 13.5];
        let p = dirichlet_predictive(&alpha, &counts).unwrap();
        let lp = dirichlet_log_predictive(&alpha, &counts).unwrap();
        for i in 0..3 {
            close(p[i], want[i], 1e-15);
            close(lp[i], want[i].ln(), 1e-14);
        }
        close(p.iter().sum::<f64>(), 1.0, 1e-15);

        // The Beta predictive is the K = 2 Dirichlet one.
        let (a_post, b_post) = beta_posterior(2.0, 3.0, 5, 1).unwrap();
        let p2 = dirichlet_predictive(&[2.0, 3.0], &[5, 1]).unwrap();
        close(p2[0], a_post / (a_post + b_post), 1e-15);

        // Tiny α and huge counts: the log predictive is finite and normalized.
        let lp = dirichlet_log_predictive(&[1e-300; 3], &[0, 1_000_000_000_000, 5]).unwrap();
        assert!(lp.iter().all(|l| l.is_finite()));
        close(log_sum_exp(&lp), 0.0, 1e-14);
    }

    // -------------------------------------------------------------------------
    // Validation
    // -------------------------------------------------------------------------

    #[test]
    fn invalid_parameters_are_errors_with_codes() {
        use ErrorCode::{InvalidCount, InvalidProbability, InvalidShape};

        // Concentrations.
        assert_eq!(err_code(ln_multivariate_beta(&[])), InvalidCount);
        for bad in [
            vec![1.0, 0.0],
            vec![1.0, -2.0],
            vec![f64::NAN, 1.0],
            vec![f64::INFINITY],
            vec![1e306],
            vec![1e308, 1e308],
        ] {
            assert_eq!(
                err_code(ln_multivariate_beta(&bad)),
                InvalidShape,
                "{bad:?}"
            );
            assert_eq!(err_code(Dirichlet::new(bad.clone())), InvalidShape);
            assert_eq!(err_code(sample_dirichlet("w", &bad)), InvalidShape);
            let x = vec![1.0 / bad.len() as f64; bad.len()];
            assert_eq!(err_code(dirichlet_log_pdf(&bad, &x)), InvalidShape);
            let zeros = vec![0u64; bad.len()];
            assert_eq!(
                err_code(dirichlet_multinomial_log_marginal(&bad, &zeros)),
                InvalidShape
            );
            assert_eq!(
                err_code(dirichlet_categorical_log_marginal(&bad, &zeros)),
                InvalidShape
            );
            assert_eq!(err_code(dirichlet_posterior(&bad, &zeros)), InvalidShape);
            assert_eq!(err_code(dirichlet_predictive(&bad, &zeros)), InvalidShape);
            assert_eq!(
                err_code(dirichlet_log_predictive(&bad, &zeros)),
                InvalidShape
            );
        }
        assert_eq!(err_code(Dirichlet::new(vec![])), InvalidCount);
        assert_eq!(err_code(sample_dirichlet("w", &[])), InvalidCount);

        // Lengths and counts.
        assert_eq!(
            err_code(dirichlet_log_pdf(&[1.0, 1.0], &[1.0])),
            InvalidCount
        );
        assert_eq!(
            err_code(dirichlet_multinomial_log_marginal(&[1.0, 1.0], &[1])),
            InvalidCount
        );
        assert_eq!(
            err_code(dirichlet_categorical_log_marginal(&[1.0], &[1, 2])),
            InvalidCount
        );
        assert_eq!(
            err_code(dirichlet_multinomial_log_marginal(
                &[1.0, 1.0],
                &[u64::MAX, 1]
            )),
            InvalidCount
        );
        assert_eq!(
            err_code(beta_bernoulli_log_marginal(1.0, 1.0, u64::MAX, 1)),
            InvalidCount
        );
        assert_eq!(err_code(ln_multinomial_coefficient(&[])), InvalidCount);
        assert_eq!(
            err_code(ln_multinomial_coefficient(&[u64::MAX, 1])),
            InvalidCount
        );

        // Probability vectors (Categorical's rules).
        assert_eq!(err_code(multinomial_log_pmf(&[], &[])), InvalidProbability);
        assert_eq!(
            err_code(multinomial_log_pmf(&[0.5, 0.6], &[1, 1])),
            InvalidProbability
        );
        assert_eq!(
            err_code(multinomial_log_pmf(&[1.5, -0.5], &[1, 1])),
            InvalidProbability
        );
        assert_eq!(
            err_code(multinomial_log_pmf(&[f64::NAN, 1.0], &[1, 1])),
            InvalidProbability
        );
        assert_eq!(
            err_code(multinomial_log_pmf(&[0.5, 0.5], &[1])),
            InvalidCount
        );
        assert_eq!(
            err_code(Multinomial::new(3, vec![0.5, 0.6])),
            InvalidProbability
        );
        assert_eq!(err_code(Multinomial::new(3, vec![])), InvalidProbability);

        // Beta pairs.
        assert_eq!(
            err_code(beta_binomial_log_marginal(0.0, 1.0, 1, 2)),
            InvalidShape
        );
        assert_eq!(
            err_code(beta_binomial_log_marginal(1.0, f64::NAN, 1, 2)),
            InvalidShape
        );
        assert_eq!(
            err_code(beta_bernoulli_log_marginal(-1.0, 1.0, 1, 2)),
            InvalidShape
        );
        assert_eq!(
            err_code(beta_posterior(1.0, f64::INFINITY, 0, 0)),
            InvalidShape
        );
        assert_eq!(err_code(beta_posterior(1e308, 1e308, 0, 0)), InvalidShape);

        // The errors are structured and say what was wrong where.
        let err = dirichlet_log_pdf(&[1.0, -1.0], &[0.5, 0.5]).unwrap_err();
        assert!(err.is_validation_error());
        let msg = err.to_string();
        assert!(
            msg.contains("Dirichlet") && msg.contains("index=1"),
            "{msg}"
        );

        // A value outside the support is not an error.
        assert_eq!(beta_binomial_log_marginal(1.0, 1.0, 3, 2).unwrap(), NEG_INF);
    }

    #[test]
    fn validate_impls_mirror_the_constructors() {
        // Invalid instances can only be built here, through the private fields.
        let bad = Dirichlet {
            alpha: vec![1.0, 0.0],
            ln_beta: 0.0,
        };
        assert_eq!(
            bad.validate().unwrap_err().code(),
            Dirichlet::new(vec![1.0, 0.0]).unwrap_err().code()
        );
        let empty = Dirichlet {
            alpha: vec![],
            ln_beta: 0.0,
        };
        assert_eq!(
            empty.validate().unwrap_err().code(),
            ErrorCode::InvalidCount
        );
        let bad = Multinomial {
            n: 3,
            probs: vec![0.5, 0.6],
        };
        assert_eq!(
            bad.validate().unwrap_err().code(),
            Multinomial::new(3, vec![0.5, 0.6]).unwrap_err().code()
        );
        assert!(Dirichlet::new(vec![0.1]).unwrap().validate().is_ok());
        assert!(Multinomial::new(0, vec![1.0]).unwrap().validate().is_ok());
    }

    // -------------------------------------------------------------------------
    // The distribution types
    // -------------------------------------------------------------------------

    #[test]
    fn dirichlet_sampling_is_stable_for_tiny_concentrations() {
        let mut rng = StdRng::seed_from_u64(66);
        for alpha in [vec![1e-3; 5], vec![1e-8; 3], vec![1e-3, 2.0, 1e-5]] {
            let d = Dirichlet::new(alpha).unwrap();
            for _ in 0..4_000 {
                let x = d.sample(&mut rng);
                assert!(x.iter().all(|v| v.is_finite() && *v >= 0.0), "{x:?}");
                assert!(x.iter().any(|&v| v > 0.0));
                close(x.iter().sum::<f64>(), 1.0, 1e-12);
                let ln_x = d.sample_log(&mut rng);
                assert!(ln_x.iter().all(|l| l.is_finite() && *l <= 0.0), "{ln_x:?}");
                close(log_sum_exp(&ln_x), 0.0, 1e-12);
            }
        }
        // Subnormal concentrations: every log-Gamma draw overflows to -inf and
        // the draw is the vertex eᵢ with probability αᵢ / A (here 1/4 and 3/4).
        let d = Dirichlet::new(vec![1e-320, 3e-320]).unwrap();
        let mut first = 0usize;
        for _ in 0..4_000 {
            let x = d.sample(&mut rng);
            assert!(x == vec![1.0, 0.0] || x == vec![0.0, 1.0], "{x:?}");
            first += usize::from(x[0] == 1.0);
            let ln_x = d.sample_log(&mut rng);
            assert!(ln_x == vec![0.0, NEG_INF] || ln_x == vec![NEG_INF, 0.0]);
        }
        close(first as f64 / 4_000.0, 0.25, 0.03);
    }

    #[test]
    fn dirichlet_sample_moments_match() {
        let alpha = [0.5, 1.0, 2.5];
        let a_total = 4.0;
        let d = Dirichlet::new(alpha.to_vec()).unwrap();
        let mut rng = StdRng::seed_from_u64(2026);
        let n = 40_000;
        let mut sum = [0.0; 3];
        let mut sum_sq = [0.0; 3];
        for _ in 0..n {
            let x = d.sample(&mut rng);
            for i in 0..3 {
                sum[i] += x[i];
                sum_sq[i] += x[i] * x[i];
            }
        }
        for i in 0..3 {
            let mean = sum[i] / n as f64;
            let var = sum_sq[i] / n as f64 - mean * mean;
            let m = alpha[i] / a_total;
            let v = m * (1.0 - m) / (a_total + 1.0);
            // Five standard errors on the mean, 5% on the variance.
            close(mean, m, 5.0 * (v / n as f64).sqrt());
            close(var, v, 0.05 * v);
        }
    }

    #[test]
    fn distribution_types_agree_with_the_free_functions() {
        let alpha = vec![0.7, 2.0, 5.0];
        let d = Dirichlet::new(alpha.clone()).unwrap();
        let x = vec![0.1, 0.3, 0.6];
        assert_eq!(d.log_prob(&x), dirichlet_log_pdf(&alpha, &x).unwrap());
        let boxed: Box<dyn Distribution<Vec<f64>>> = d.clone_box();
        assert_eq!(boxed.log_prob(&x), d.log_prob(&x));
        assert_eq!((d.alpha(), d.len(), d.is_empty()), (&alpha[..], 3, false));

        // `sample` and `sample_log` make the same draw from the same state.
        let (mut r1, mut r2) = (StdRng::seed_from_u64(4), StdRng::seed_from_u64(4));
        let (x1, l2) = (d.sample(&mut r1), d.sample_log(&mut r2));
        for i in 0..3 {
            close(x1[i], l2[i].exp(), 1e-14);
        }

        let p = vec![0.2, 0.3, 0.5];
        let m = Multinomial::new(6, p.clone()).unwrap();
        let counts = vec![1, 2, 3];
        assert_eq!(
            m.log_prob(&counts),
            multinomial_log_pmf(&p, &counts).unwrap()
        );
        let boxed: Box<dyn Distribution<Vec<u64>>> = m.clone_box();
        assert_eq!(boxed.log_prob(&counts), m.log_prob(&counts));
        assert_eq!(
            (m.n(), m.probs(), m.len(), m.is_empty()),
            (6, &p[..], 3, false)
        );
    }

    #[test]
    fn multinomial_sampling_and_support() {
        let p = vec![0.1, 0.0, 0.6, 0.3];
        let m = Multinomial::new(50, p.clone()).unwrap();
        let mut rng = StdRng::seed_from_u64(9);
        let draws = 20_000;
        let mut mean = [0.0; 4];
        for _ in 0..draws {
            let c = m.sample(&mut rng);
            assert_eq!(c.iter().sum::<u64>(), 50);
            assert_eq!(c[1], 0);
            assert!(m.log_prob(&c).is_finite());
            for i in 0..4 {
                mean[i] += c[i] as f64 / draws as f64;
            }
        }
        for i in 0..4 {
            let se = (50.0 * p[i] * (1.0 - p[i]) / draws as f64).sqrt();
            close(mean[i], 50.0 * p[i], 5.0 * se + 1e-12);
        }
        // Huge n costs K binomial draws.
        let big = Multinomial::new(1_000_000_000_000, vec![0.25, 0.25, 0.5]).unwrap();
        let c = big.sample(&mut rng);
        assert_eq!(c.iter().sum::<u64>(), 1_000_000_000_000);
        assert!(big.log_prob(&c).is_finite());
        // Outside the support: wrong length, wrong total, overflowing total, an
        // observed zero-probability category.
        assert_eq!(m.log_prob(&vec![10, 0, 40]), NEG_INF);
        assert_eq!(m.log_prob(&vec![10, 0, 30, 9]), NEG_INF);
        assert_eq!(m.log_prob(&vec![u64::MAX, 0, 1, 0]), NEG_INF);
        assert_eq!(m.log_prob(&vec![10, 1, 30, 9]), NEG_INF);
        // Zero draws.
        let zero = Multinomial::new(0, vec![0.5, 0.5]).unwrap();
        assert_eq!(zero.sample(&mut rng), vec![0, 0]);
        assert_eq!(zero.log_prob(&vec![0, 0]), 0.0);
    }

    // -------------------------------------------------------------------------
    // sample_dirichlet: a Dirichlet draw from K Gamma sites
    // -------------------------------------------------------------------------

    #[test]
    fn sample_dirichlet_records_gamma_sites_at_indexed_addresses() {
        let alpha = [0.7, 2.0, 5.0, 1.3];
        for name in ["mix", "a#b\\c"] {
            let (x, trace) = prior_run(11, sample_dirichlet(name, &alpha).unwrap());
            // Exactly the addresses `addr!(name, i)` builds, escaping included.
            let mut expected: Vec<Address> = (0..4).map(|i| addr!(name, i)).collect();
            expected.sort();
            let recorded: Vec<Address> = trace.choices.keys().cloned().collect();
            assert_eq!(recorded, expected);
            // The returned vector is on the simplex and is g / Σg.
            let g: Vec<f64> = (0..4)
                .map(|i| trace.get_f64(&addr!(name, i)).unwrap())
                .collect();
            let total: f64 = g.iter().sum();
            for i in 0..4 {
                assert!(x[i] >= 0.0);
                close(x[i], g[i] / total, 1e-15);
            }
            close(x.iter().sum::<f64>(), 1.0, 1e-12);
            // Only the Gamma sites contribute to the weight.
            assert_eq!((trace.log_likelihood, trace.log_factors), (0.0, 0.0));
        }
        assert_eq!(addr!("mix", 3).as_str(), "mix#3");
    }

    #[test]
    fn sample_dirichlet_log_prior_is_the_joint_gamma_density() {
        let alpha = [0.7, 2.0, 5.0, 1.3];
        let a_total: f64 = alpha.iter().sum();
        for seed in 0..20 {
            let (x, trace) = prior_run(seed, sample_dirichlet("w", &alpha).unwrap());
            let g: Vec<f64> = (0..4)
                .map(|i| trace.get_f64(&addr!("w", i)).unwrap())
                .collect();
            let per_site: Vec<f64> = alpha
                .iter()
                .zip(&g)
                .map(|(&a, gi)| Gamma::new(a, 1.0).unwrap().log_prob(gi))
                .collect();
            for (i, &lp) in per_site.iter().enumerate() {
                assert_eq!(trace.choices[&addr!("w", i)].logp, lp);
            }
            close_rel(trace.log_prior, per_site.iter().sum(), 1e-14);
            // The change of variables g ↦ (x, S):
            // ln p(g) = ln Dir(x | α) + ln Gamma(S | A, 1) − (K − 1)·ln S.
            let s: f64 = g.iter().sum();
            let decomposed = dirichlet_log_pdf(&alpha, &x).unwrap()
                + Gamma::new(a_total, 1.0).unwrap().log_prob(&s)
                - 3.0 * s.ln();
            close_rel(trace.log_prior, decomposed, 1e-11);
        }
    }

    #[test]
    fn sample_dirichlet_rescoring_reproduces_the_trace() {
        let alpha = [0.7, 2.0, 5.0];
        let model = || sample_dirichlet("w", &alpha).unwrap();
        let (x, trace) = prior_run(5, model());
        let (x2, rescored) = run(
            ScoreGivenTrace {
                base: trace.clone(),
                trace: Trace::default(),
            },
            model(),
        );
        assert_eq!(x, x2);
        assert_eq!(rescored.log_prior, trace.log_prior);
        assert_eq!(rescored.total_log_weight(), trace.total_log_weight());
    }

    #[test]
    fn sample_dirichlet_is_dirichlet_distributed() {
        let alpha = [0.5, 1.5, 3.0];
        let a_total: f64 = alpha.iter().sum();
        let mut rng = StdRng::seed_from_u64(77);
        let n = 20_000;
        let mut mean = [0.0; 3];
        for _ in 0..n {
            let (x, trace) = run(
                PriorHandler {
                    rng: &mut rng,
                    trace: Trace::default(),
                },
                sample_dirichlet("w", &alpha).unwrap(),
            );
            assert!(trace.log_prior.is_finite());
            for i in 0..3 {
                mean[i] += x[i] / n as f64;
            }
        }
        for i in 0..3 {
            let m = alpha[i] / a_total;
            let v = m * (1.0 - m) / (a_total + 1.0);
            close(mean[i], m, 5.0 * (v / n as f64).sqrt());
        }
    }

    #[test]
    fn sample_dirichlet_underflow_is_a_zero_weight_uniform_vector() {
        // Every Gamma(1e-300, 1) draw underflows to 0 (the probability that one
        // does not is about 7e-298): the proportions are undefined.
        let (x, trace) = prior_run(3, sample_dirichlet("w", &[1e-300; 3]).unwrap());
        assert_eq!(x, vec![1.0 / 3.0; 3]);
        assert_eq!(trace.log_factors, NEG_INF);
        assert_eq!(trace.total_log_weight(), NEG_INF);

        // Replayed values that do not define a simplex point take the same path.
        let alpha = [2.0, 3.0];
        for bad in [
            [0.0, 0.0],
            [1.0, -0.5],
            [f64::NAN, 1.0],
            [f64::INFINITY, 1.0],
        ] {
            let mut base = Trace::default();
            for (i, &g) in bad.iter().enumerate() {
                base.insert_choice(addr!("w", i), ChoiceValue::F64(g), 0.0);
            }
            let (x, t) = run(
                ScoreGivenTrace {
                    base,
                    trace: Trace::default(),
                },
                sample_dirichlet("w", &alpha).unwrap(),
            );
            assert_eq!(x, vec![0.5, 0.5], "{bad:?}");
            assert_eq!(t.log_factors, NEG_INF, "{bad:?}");
        }

        // A single underflowed draw is still a simplex point, but the Gamma
        // density at 0 is zero, so the trace has zero weight through log_prior.
        let mut base = Trace::default();
        base.insert_choice(addr!("w", 0), ChoiceValue::F64(0.0), 0.0);
        base.insert_choice(addr!("w", 1), ChoiceValue::F64(2.0), 0.0);
        let (x, t) = run(
            ScoreGivenTrace {
                base,
                trace: Trace::default(),
            },
            sample_dirichlet("w", &[0.5, 3.0]).unwrap(),
        );
        assert_eq!(x, vec![0.0, 1.0]);
        assert_eq!(t.log_prior, NEG_INF);
        assert_eq!(t.log_factors, 0.0);
    }

    // -------------------------------------------------------------------------
    // Inference end to end
    // -------------------------------------------------------------------------

    // MH over the Gamma sites, with categorical observations of the returned
    // weights, must recover the conjugate posterior Dir(α + n): this checks
    // that the trace's Gamma bookkeeping makes the acceptance ratios right.
    #[test]
    fn mh_over_gamma_sites_recovers_the_conjugate_posterior() {
        let alpha = [2.0, 1.0, 1.0];
        let data = [0usize, 0, 1, 0, 2, 0, 1, 0]; // counts (5, 2, 1)
        let model = move || {
            sample_dirichlet("w", &alpha).unwrap().bind(move |w| {
                let weights = w.clone();
                let obs: Vec<(usize, usize)> = data.iter().copied().enumerate().collect();
                traverse_vec(obs, move |(t, y)| {
                    observe(addr!("y", t), Categorical::new(weights.clone()).unwrap(), y)
                })
                .map(move |_| w)
            })
        };
        let mut rng = StdRng::seed_from_u64(61);
        let draws = adaptive_mcmc_chain(&mut rng, model, 6_000, 2_000);
        let posterior_mean = dirichlet_predictive(&alpha, &[5, 2, 1]).unwrap();
        for (i, &want) in posterior_mean.iter().enumerate() {
            let mean = draws.iter().map(|(w, _)| w[i]).sum::<f64>() / draws.len() as f64;
            close(mean, want, 0.04);
        }
    }

    // The pattern the issue describes for the back-off concentration: a
    // Normal prior on ln α, the Dirichlet integrated out in closed form and
    // entered with `factor`. The MH posterior mean of ln α must match a
    // quadrature of the same posterior.
    #[test]
    fn concentration_posterior_via_factor_matches_quadrature() {
        let counts = vec![40u64, 3, 1, 0];
        let prior = Normal::new(0.0, 2.0).unwrap();
        let log_post = |la: f64| {
            prior.log_prob(&la)
                + dirichlet_categorical_log_marginal(&[la.exp(); 4], &counts).unwrap()
        };
        let grid: Vec<f64> = (0..=8_000)
            .map(|i| -16.0 + 24.0 * i as f64 / 8_000.0)
            .collect();
        let lps: Vec<f64> = grid.iter().map(|&la| log_post(la)).collect();
        let z = log_sum_exp(&lps);
        let grid_mean: f64 = grid
            .iter()
            .zip(&lps)
            .map(|(&la, &lp)| la * (lp - z).exp())
            .sum();

        let model = move || {
            let counts = counts.clone();
            sample(addr!("log_alpha"), Normal::new(0.0, 2.0).unwrap()).bind(move |log_alpha| {
                let alpha = vec![log_alpha.exp(); counts.len()];
                let loglik = dirichlet_categorical_log_marginal(&alpha, &counts).unwrap_or(NEG_INF);
                factor(loglik).map(move |_| log_alpha)
            })
        };
        let mut rng = StdRng::seed_from_u64(68);
        let draws = adaptive_mcmc_chain(&mut rng, model, 8_000, 2_000);
        let mh_mean = draws.iter().map(|(la, _)| la).sum::<f64>() / draws.len() as f64;
        close(mh_mean, grid_mean, 0.15);
    }
}
