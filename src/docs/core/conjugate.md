# Conjugate helpers: Dirichlet, Multinomial and their marginals

Fugue's sites are scalar (`f64`, `bool`, `u64`, `usize`, `i64`), so a Dirichlet
over category probabilities cannot be a site. This module covers the models
that need one without changing that:

- **Closed-form conjugate helpers**, all in log space: Dirichlet and
  multinomial log-densities, the Dirichlet–multinomial and Beta–binomial
  marginal likelihoods (with the probabilities integrated out), posterior
  updates and posterior predictives. A marginal likelihood enters a model
  through [`factor`](crate::factor).
- **`Dirichlet` and `Multinomial` distribution types** over `Vec<f64>` and
  `Vec<u64>`, for sampling and scoring outside a model.
- **[`sample_dirichlet`]**, which draws a Dirichlet vector *inside* a model from
  `K` scalar `Gamma(αᵢ, 1)` sites at `addr!(name, i)` and normalizes them.

Vector-valued sites (new `Model` variants and `Handler` methods for simplex and
count vectors) were deliberately not added: every `Handler` implementor, in
this crate and downstream, would need new methods, and every inference kernel
would need proposals on the simplex. The helpers here are enough for a
Dirichlet–categorical model with a sampled concentration.

## What is here

| Item | Computes |
| ---- | -------- |
| [`ln_multivariate_beta`] | `ln B(α) = Σᵢ lnΓ(αᵢ) − lnΓ(Σᵢ αᵢ)` |
| [`ln_multinomial_coefficient`] | `ln(N! / Πᵢ nᵢ!)` |
| [`dirichlet_log_pdf`] | `ln Dir(x \| α) = Σᵢ (αᵢ − 1)·ln xᵢ − ln B(α)` |
| [`multinomial_log_pmf`] | `ln Mult(n \| N, p) = ln(N! / Πᵢ nᵢ!) + Σᵢ nᵢ·ln pᵢ` |
| [`dirichlet_multinomial_log_marginal`] | `ln P(n \| α)` for a **count vector** |
| [`dirichlet_categorical_log_marginal`] | `ln P(x₁…x_N \| α)` for **one sequence** |
| [`beta_binomial_log_marginal`] | `ln P(k \| n, a, b)`, the `K = 2` count version |
| [`beta_bernoulli_log_marginal`] | `ln P(sequence \| a, b)`, the `K = 2` sequence version |
| [`dirichlet_posterior`], [`beta_posterior`] | `α + n`, `(a + s, b + f)` |
| [`dirichlet_predictive`], [`dirichlet_log_predictive`] | `(αᵢ + nᵢ) / (A + N)` and its log |
| [`Dirichlet`], [`Multinomial`] | standalone `Distribution<Vec<f64>>` / `Distribution<Vec<u64>>` |
| [`sample_dirichlet`] | a `Dir(α)` draw inside a model, from `K` Gamma sites |

Throughout, `A = Σᵢ αᵢ` and `N = Σᵢ nᵢ`.

## A sequence or a count vector?

The two Dirichlet marginals answer different questions about the same data:

- [`dirichlet_categorical_log_marginal`] is the probability of **one
  particular sequence** of `N` draws with counts `n`:
  `ln B(α + n) − ln B(α)`. It is the product of the sequential predictive
  probabilities (the Pólya urn), `Πₜ (α_{xₜ} + n_{xₜ}⁽ᵗ⁾) / (A + t − 1)`, which is
  the prequential likelihood a sequential predictor accumulates.
- [`dirichlet_multinomial_log_marginal`] is the probability of the **count
  vector** itself, summed over all `N! / Πᵢ nᵢ!` sequences that have it; it sums
  to 1 over count vectors with total `N`.

They differ by [`ln_multinomial_coefficient`], which does not depend on `α`.
Both give the same posterior over `α` and the same Bayes factors between
concentrations; use the sequence version when the data are a sequence, the
count version when only counts were recorded. The Beta–Bernoulli and
Beta–binomial marginals are the same pair for `K = 2`.

```rust
# use fugue::*;
let (alpha, counts) = ([0.5, 1.5, 0.25], [3u64, 1, 2]);
let seq = dirichlet_categorical_log_marginal(&alpha, &counts).unwrap();
let vec = dirichlet_multinomial_log_marginal(&alpha, &counts).unwrap();
let coef = ln_multinomial_coefficient(&counts).unwrap();
assert!((vec - (seq + coef)).abs() < 1e-12);

// The sequence probability is the Pólya urn, draw by draw, for any order.
let sequence = [0usize, 2, 0, 1, 2, 0];
let mut seen = [0u64; 3];
let mut urn = 0.0;
for (t, &c) in sequence.iter().enumerate() {
    let a_total: f64 = alpha.iter().sum();
    urn += ((alpha[c] + seen[c] as f64) / (a_total + t as f64)).ln();
    seen[c] += 1;
}
assert!((urn - seq).abs() < 1e-12);
```

## Conjugate updates

```rust
# use fugue::*;
let alpha = [0.5, 0.5, 0.5];
let counts = [12, 3, 0];
// Dir(α) prior and counts n: the posterior is Dir(α + n) ...
let post = dirichlet_posterior(&alpha, &counts).unwrap();
assert_eq!(post, vec![12.5, 3.5, 0.5]);
// ... and the predictive for the next draw is its mean.
let p = dirichlet_predictive(&alpha, &counts).unwrap();
assert!((p[0] - 12.5 / 16.5).abs() < 1e-12);
```

## Inferring a concentration with `factor`

A concentration is a scalar, so it can be a site; the Dirichlet itself is
integrated out in closed form and its marginal likelihood enters as a factor.
Here the `log α` site gets a `Normal` prior and adaptive Metropolis–Hastings
samples its posterior:

```rust
# use fugue::*;
# use rand::rngs::StdRng;
# use rand::SeedableRng;
// Counts of the next action in one context: how much should they be smoothed?
let counts: Vec<u64> = vec![40, 3, 1, 0];
let model = move || {
    let counts = counts.clone();
    sample(addr!("log_alpha"), Normal::new(0.0, 2.0).unwrap()).bind(move |log_alpha| {
        let alpha = vec![log_alpha.exp(); counts.len()];
        // exp() can overflow or underflow at an extreme proposal; treat an
        // invalid α as probability zero rather than panicking.
        let loglik = dirichlet_categorical_log_marginal(&alpha, &counts)
            .unwrap_or(f64::NEG_INFINITY);
        factor(loglik).map(move |_| log_alpha)
    })
};
let mut rng = StdRng::seed_from_u64(7);
let draws = adaptive_mcmc_chain(&mut rng, model, 500, 200);
let mean = draws.iter().map(|(la, _)| la).sum::<f64>() / draws.len() as f64;
assert!(mean.is_finite());
```

## A Dirichlet inside a model

[`sample_dirichlet`] builds the vector from `K` scalar Gamma sites, so it
composes with everything else. Here the weights are conditioned on categorical
observations; the posterior is `Dir(α + n)`, which inference over the Gamma
sites recovers:

```rust
# use fugue::*;
let data = vec![0usize, 2, 2, 1, 2];
let model = sample_dirichlet("w", &[1.0, 1.0, 1.0]).unwrap().bind(move |w| {
    let weights = w.clone();
    let observations: Vec<(usize, usize)> = data.into_iter().enumerate().collect();
    traverse_vec(observations, move |(t, y)| {
        observe(addr!("y", t), Categorical::new(weights.clone()).unwrap(), y)
    })
    .map(move |_| w)
});
```

The trace holds the Gamma draws at `addr!("w", i)` and their joint Gamma
density as `log_prior`, which differs from the Dirichlet density of the
returned vector by the density of their total and a Jacobian; see
[`sample_dirichlet`] for the exact statement, and for the underflow behaviour
at concentrations below about 0.05.

## Numerical notes

- **Log space throughout.** No density or marginal is formed in linear space.
- **No large `lnΓ` differences.** A marginal likelihood is a sum of `lnΓ`
  differences, and at large counts the two values in a difference are huge and
  nearly equal (`lnΓ(10⁹) ≈ 2·10¹⁰`). They are never subtracted directly:
  `lnΓ(z + a) − lnΓ(z + b)` is evaluated as a Stirling-series difference, and
  the marginals are assembled from log Pólya coefficients `ln[(αᵢ)ₙᵢ / nᵢ!]`.
  So `beta_binomial_log_marginal(1.0, 1.0, k, 10¹²)` is `−ln(10¹² + 1)` to full
  precision, where a plain `lnΓ` difference is off by about 2·10⁻³. Tiny
  concentrations (down to the smallest positive `f64`) and large counts are the
  regime this is built for. The one case with a residual error is
  concentrations *and* counts both large and comparable (≳ 10⁷), where terms
  of size `N·ln 2` cancel to a small result; the absolute error there is about
  `10⁻¹⁶·N·ln N`.
- **Counts** are `u64` and are converted to `f64`, exactly up to `2⁵³`.
- **Invalid parameters are errors** (`FugueError::InvalidParameters`): empty
  vectors and mismatched lengths (`InvalidCount`), a total count that overflows
  `u64` (`InvalidCount`), a concentration that is not positive and finite or a
  total concentration past about `2.5·10³⁰⁵`, where `lnΓ` overflows
  (`InvalidShape`), and a probability vector that is not one
  (`InvalidProbability`, with [`Categorical`](crate::Categorical)'s rules).
- **Values outside the support are `-∞`**, never an error: an `x` off the
  simplex by more than [`SIMPLEX_TOLERANCE`], `k > n`, or a count vector whose
  total is not the `Multinomial`'s `n`.
