# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
(pre-1.0: see the API-stability note in `README.md`'s Roadmap section).

For the initial 0.1.0 release notes, see `.github/CHANGELOG.md`.

## [Unreleased]

### Added

- **Retention thinning on the adaptive MCMC chain drivers**:
  `adaptive_mcmc_chain_thinned` and
  `adaptive_mcmc_chain_with_overrides_thinned` take a `thin: usize` and push
  only every `thin`-th draw. `adaptive_mcmc_chain` and
  `adaptive_mcmc_chain_with_overrides` are now `thin = 1` wrappers with their
  signatures and behaviour unchanged.

  Purely a **memory** change with no statistical content. The existing drivers
  materialize every iteration — an `(A, Trace)` per step, pushed into the `Vec`
  returned by value — so a caller that wanted a thinned subsequence (the common
  case for autocorrelated single-site draws) had to hold the entire chain live
  before discarding most of it. For a structure-varying model with ~140 sites
  over a 10 000-step chain that is ~10 000 `Trace` clones of ~140 `BTreeMap`
  entries resident at once to keep 500, which on a 32-bit wasm heap is a
  plausible OOM rather than mere waste — and the caller's only lever was to
  shorten the chain, paying in statistics for a memory problem.

  **The retained draws are bit-identical to thinning the full chain.** `thin`
  gates the `push` and nothing else: every transition still runs, so the RNG is
  consumed in the same order and quantity and the kept draws are the *same*
  draws, not merely draws from the same distribution. Retained indices are
  `0, thin, 2·thin, …`, matching `Iterator::step_by`; `thin = 0` normalizes to
  `1`. Pinned by an equality test against `step_by` at three strides over a
  multi-site model, asserting both values and trace weights, plus the two edge
  cases.

## [0.2.1] - 2026-07-28

EA-as-PPL upstream primitives (F1-F6 of the cross-repo plan, tracking issue
[#44](https://github.com/alexnodeland/fugue/issues/44)): the additive machinery
that lets an evolutionary-computation layer run genuinely PPL-native inference
- population-coupled SMC moves, block regeneration, and trace surgery.

### Added

- **Population-coupled SMC kernels (F4)**: object-safe `PopulationKernel<A>`
  trait with the pi_beta-product invariance contract (W/T/S/E) documented on
  the trait; `NoKernel` identity kernel; generic mask-driven `CrossoverKernel`
  (pairwise block-swap Metropolis move on the product target - a symmetric
  involution, Hastings ratio 1); and `adaptive_smc_with_kernel`, which invokes
  the kernel after each intermediate resample + rejuvenation and never at the
  terminal beta = 1 step (FG-43). `adaptive_smc` is now defined as
  `adaptive_smc_with_kernel(.., &mut NoKernel)` with its signature unchanged.
  Pinned by product-invariance, weight-preservation (bit-identical across a
  sweep), log-evidence non-corruption vs analytic marginal likelihood (FG-58),
  and joint-support truncation tests.
- **Block-regeneration MH (F2)**: `block_regeneration_mh` - delete the choices
  at an address set S, replay the model via `score_given_trace_reconciled`
  (fresh prior draws at S), and accept with the prior-cancelling ratio
  including fresh/vanished RJMCMC bookkeeping. For fixed structure the ratio
  collapses to `beta * delta-loglik`. Note: unlike the single-site kernels
  there is deliberately **no dimension-selection term** - the block is fixed by
  the caller, not selected uniformly from a state-dependent site set. Pinned by
  conjugate Beta-Bernoulli validation, product-Normal analytic posterior,
  trans-dimensional switch-gated-branch analytic posterior, and fresh-rescore
  equality (FG-48) tests.
- **Trace subtree surgery (F3)**: boundary-aware `Address::has_prefix`
  (recognizes `#`, `::`, `/` segment separators; does not treat `"gene"` as a
  prefix of `"generation"`) and `Trace::{extract_prefix, truncate_prefix,
  graft_prefix}`. All three deliberately leave the flat, non-address-keyed log
  accumulators zeroed/stale - the only correct recomputation is a model
  re-score, which the F2/F4 kernels perform. Pinned by boundary, round-trip,
  graft-then-rescore-equality, and accumulator-zeroing tests.
- **Decode-replay helpers (F5)**: `decode_particle` / `decode_particles`
  recover the model return value (e.g. a decoded genome) from a particle trace
  by replaying under `ScoreGivenTrace`; fallible `try_decode_particle` (backed
  by `SafeScoreGivenTrace`) for traces of uncertain provenance. `Particle`
  deliberately does not cache the return value (would force `A: Clone` through
  resampling and break the FG-59 move-not-clone construction).
- **Crate-root re-export widening (F6)**: `SMCResult` (previously returned by
  `adaptive_smc` yet unreachable by name from the root), `smc_prior_particles`,
  `normalize_particles`, `resample_particles`, `rejuvenate_particles`,
  `systematic_resample`, `stratified_resample`, `multinomial_resample`,
  `score_given_trace_reconciled`, `ReconcileReport`, plus all new F2/F4/F5
  items. Pinned by a root-import integration test.

### Fixed

- **SMC rejuvenation now moves non-F64 sites (F1)**: `tempered_single_site_mh`
  previously collected only `ChoiceValue::F64` sites and returned
  `current.clone()` otherwise, so a population of pure Bool/U64/Usize traces
  (bit-string or permutation genomes) was silently frozen during rejuvenation.
  It now picks the target uniformly over all sites and dispatches by value type
  through the same typed proposal machinery as `adaptive_single_site_mh`
  (flip for Bool, reflected discrete walk for U64, prior-resample for Usize,
  integer walk for I64, Gaussian/log-space walk for F64), with the tempered
  trans-dimensional acceptance
  `delta-log-prior + beta * delta-loglik + (log q_rev - log q_fwd) + dim_term`.
  Pinned by a Bool-population movement regression and an independent-Bernoulli
  analytic-posterior test.

### Added (carried from the pre-0.2.1 unreleased queue)

- **Incremental HMC session API**: `HmcSession` exposes the HMC kernel one
  transition at a time (`step`, `step_recorded` with full leapfrog
  trajectories and per-point Hamiltonians, `set_step_size`/`set_n_leapfrog`
  live retuning). `hmc_chain` is now a thin wrapper over it; same-seed
  equivalence is pinned by test. New public types: `HmcSession`,
  `HmcStepInfo`, `LeapfrogPoint`.
- **`crates/fugue-wasm`** (unpublished workspace member): wasm-bindgen
  bindings that run the real crate in the browser — a `prob!`-subset DSL
  interpreted into actual `Model` combinators, incremental multi-chain MH
  (`WasmMh`), incremental HMC (`WasmHmc`), a bootstrap particle filter over
  fugue's SMC primitives (`WasmParticleFilter`), one-shot adaptive tempered
  SMC with log-evidence (`wasm_smc_run`), and log-joint grid evaluation for
  posterior heatmaps (`log_joint_grid`). Powers the docs' Playground page
  and the WASM-backed explorable widgets (mirrored-JS math remains only as
  a fallback).

## [0.2.0] - 2026-07-13

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
