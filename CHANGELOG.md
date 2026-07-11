# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
(pre-1.0: see the API-stability note in `README.md`'s Roadmap section).

For the initial 0.1.0 release notes, see `.github/CHANGELOG.md`.

## [Unreleased]

The entries below summarize a full-crate audit remediation (170 findings,
tracked as `FG-01` .. `FG-64` in the project's audit record) organized by
area. Each bullet range names the finding IDs addressed in that area; a later
pass appends the individual per-finding change lines under each heading.

### Correctness — MCMC / Metropolis-Hastings (FG-01, FG-02, FG-10 – FG-12, FG-35 – FG-42, FG-57)

Proposal-distribution corrections, normalized and multi-chain effective
sample size, split-R-hat, autocorrelation/Geweke diagnostics, and removal of
redundant recomputation in the adaptive MH sampler.

### Correctness — Sequential Monte Carlo (FG-03, FG-13, FG-43, FG-58, FG-59)

Prior-cancelled (not prior-squared) importance weights, weight-preserving
rejuvenation, no terminal resample, an unbiased log-evidence estimate, and a
move-not-clone particle construction path.

### Correctness — Approximate Bayesian Computation (FG-09, FG-34)

Importance-weighted ABC-SMC (replacing a biased prior-replacement heuristic)
with bounded, typed-error attempt budgets instead of unbounded loops or
panics on an empty population.

### Correctness — Variational Inference (FG-04, FG-16, FG-17, FG-18, FG-44, FG-46, FG-60)

Support-matched guide families (Normal/LogNormal/Beta) instead of a
one-size-fits-all Normal, both location *and* scale optimized via
common-random-numbers finite-difference gradients, an ELBO-plateau
convergence test, a corrected (non-double-counted) prior-baseline ELBO, and
exact (not moment-matched) Beta sampling.

### New — Hamiltonian Monte Carlo (FG-31) and expanded distribution coverage

A new gradient-based (finite-difference force, exact Metropolis correction)
HMC kernel, plus seven new distributions (StudentT, Cauchy, Laplace, Weibull,
ChiSquared, InverseGamma, DiscreteUniform) bringing the total to 17.

### Runtime / handler correctness (FG-47 and related)

Duplicate-address and structure-mismatch detection in the replay/scoring
interpreters now returns a typed `FugueError` (`AddressConflict`,
`UnexpectedModelStructure`) instead of panicking.

### Performance (FG-05, FG-22, FG-24, FG-62 – FG-64)

`Arc<str>` addressing, removal of a dead memory-pooling subsystem, and
realistic end-to-end benchmarks in place of micro-benchmarks that didn't
reflect actual usage.

### Documentation, examples, and API surface hygiene (FG-23, FG-25, FG-33, FG-50, FG-51)

- **FG-23**: Replaced the "production-ready" tagline (README, mdBook home
  page, and a stale duplicate landing page) with accurate positioning:
  type-safe, monadic, pre-1.0, actively developed. Added an explicit
  pre-1.0 SemVer policy note.
- **FG-25**: Added `examples/smc_inference.rs`, `examples/abc_inference.rs`,
  and `examples/vi_inference.rs` — the first examples anywhere in the crate
  (README, `examples/`, or mdBook) to exercise `adaptive_smc`,
  `abc_smc_weighted`, and `optimize_meanfield_vi_with_config`, each checked
  against a closed-form posterior. Wired into a new mdBook "Advanced
  Inference" tutorial section. Added `hmc_chain` to the README's example
  index.
- **FG-33**: Removed 11 of 22 `ErrorCode` variants (and the `FugueError`
  variants/constructors/macro that existed only to hold them) that no code
  path in the crate ever constructed: `NumericalOverflow`,
  `NumericalUnderflow`, `NumericalInstability`, `InvalidLogDensity`,
  `ModelExecutionFailed`, `InferenceConvergenceFailed`,
  `InsufficientSamples`, `InvalidInferenceConfig`, `TraceCorrupted`,
  `TraceReplayFailed`, `UnsupportedType`. The 11 surviving codes are each
  verified live (grepped construction sites) in `src/error.rs`'s module
  docs. ABC and VI's own failure modes (`ABCError`, `GuideError`) keep their
  dedicated, more precise error types rather than being folded into this
  general enum.
- **FG-50**: README/mdBook now state the exact distribution count (17,
  enumerated) instead of the ambiguous "10+".
- **FG-51**: The README's unverified "1.70+" claim was wrong: real
  `rustc 1.70.0` fails to build the crate (an `E0659` ambiguous-name error on
  `pub mod core` vs. the `core` extern-prelude crate, and
  `usize::is_multiple_of`, stable only since 1.87.0). Pinned the verified
  floor, `rust-version = "1.87"`, in `Cargo.toml`, corrected the README/mdBook
  badges accordingly, and added a dedicated MSRV job to
  `.github/workflows/ci.yml` that actually builds against `rustc 1.87.0`.

[Unreleased]: https://github.com/alexnodeland/fugue/compare/v0.1.0...HEAD
