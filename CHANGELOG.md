# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
(pre-1.0: see the API-stability note in `README.md`'s Roadmap section).

For the initial 0.1.0 release notes, see `.github/CHANGELOG.md`.

## [Unreleased]

### Fixed

- **`f64` proposal selection is now a function of the site's distribution,
  never of its current value (FG-N1)**. `Distribution<T>` gains a
  `support(&self) -> Support` method (default `Support::Real`; `Uniform` and
  `Beta` report `Bounded { lower, upper }`, `LogNormal`, `Exponential`,
  `Gamma`, `Weibull`, `ChiSquared` and `InverseGamma` report `Positive`), and
  the single-site kernels pick the proposal from it via
  `proposal_kind_for_support`: `Real` -> Gaussian walk, `Positive` -> log-space
  walk (with the FG-02 Jacobian), `Bounded` -> Gaussian walk reflected at both
  bounds. Explicit `SiteProposal` overrides still win.

  The previous selector chose the log-space walk iff `current > 0` **and** the
  prior density at a probe value of `-1.0` was `-inf`. For `Uniform(-0.5, 0.5)`
  the probe is `-inf` (the support does not contain `-1`), so a positive first
  draw put the site on a walk that can never propose a negative value, and the
  kind was cached per address, so the whole chain inherited the sign of its
  first state (`P(x > 0) = 1.0` against a truth of `0.5`, seed-dependent). In
  SMC rejuvenation the kind was re-chosen per call from the current state, so
  the move was not pi_beta-invariant: a Gaussian step from `x < 0` to `x' > 0`
  has a reverse density of zero that the code treated as symmetric, and mass
  leaked one way until the population sat entirely in the positive half. A
  kernel whose shape depends on the current value is not invariant for the
  target; the support is a property of the site and is safe to select on. The
  probe and the per-address kind cache are gone: nothing about the kind is
  cached across executions, so a distribution whose bounds depend on another
  site (`Uniform(0, sigma)`) gets the bounds of the execution being proposed.
  Foreign distributions that leave the default `Real` get the Gaussian walk,
  which is always correct (out-of-support proposals score `-inf` and reject),
  merely less efficient than a declared support.

  Pinned by `fgn1_bounded_prior_containing_negatives_visits_both_signs` (six
  seeds, `P(x > 0) ~ 0.5`), `fgn1_bounded_prior_matches_analytic_mean_under_factor`
  (`rho ∝ e^{2x}` on `[-1/2, 1/2]`: mean `0.1565`, not the confined `0.291`),
  `fgn1_proposal_kind_is_a_function_of_the_support`, and
  `fgn1_smc_rejuvenation_is_invariant_on_bounded_prior_with_negatives` (mean,
  sign mass, and analytic log-evidence through ten rejuvenation steps per
  tempering stage). Both chain tests fail on the pre-fix selector. The FG-02
  and FG-42 regressions are unchanged and still pass.
- **`NaN` log-weights are sanitised to `-inf` and MH escapes a non-finite
  state (FG-N2)**. A `NaN` log-density is now read as "probability zero" at
  every accumulation point - `factor()` at construction, the `on_factor` and
  `on_observe_*` methods of every shipped handler - and `log_sum_exp` /
  `weighted_log_sum_exp` treat `NaN` terms as `-inf` (new helper
  `core::numerical::nan_to_neg_inf`). The shared Metropolis decision
  (`mh_accept`, used by the single-site kernels, `block_regeneration_mh` and
  SMC rejuvenation) accepts any proposal with a finite density when the
  *current* state's density is not finite, and stays otherwise.

  Previously one `factor(NaN)` made the trace weight `NaN`; in SMC the
  population normalizer became `NaN`, the ESS non-finite, and the tempering
  ladder silently jumped to `beta = 1` with uniform prior weights and a `NaN`
  evidence, returned without error; in MH `NaN >= 0.0` and `u < exp(NaN)` are
  both false, so a chain initialised on a `NaN` state rejected every proposal
  forever. Every consumer had been guarding this itself. There is no other
  reading of an invalid weight under which the trace weight, the SMC
  normalizer and the acceptance ratio all stay well-defined.

  Pinned by `fgn2_factor_nan_accumulates_as_neg_inf`,
  `fgn2_mh_chain_escapes_a_nan_start_and_targets_the_half_normal` (six seeds,
  driven from the exact prior state through `adaptive_single_site_mh`),
  `fgn2_mh_full_chain_driver_survives_nan_factor`,
  `fgn2_smc_does_not_collapse_to_the_prior_on_nan_factor` (with and without
  rejuvenation: log-evidence `ln 1/2`, zero weight on the `NaN` region,
  half-normal mean `0.798` rather than the prior mean `0`), and
  `test_log_sum_exp_treats_nan_as_neg_inf`.
- **Reverse-move densities come from the re-scored current trace, not the
  caller's stored `logp` (FG-N6)**. `block_regeneration_mh` now reads the
  block and vanished-site `logp` terms (and builds its block-deleted base)
  from the `ScoreGivenTrace` re-score it already performs; the SMC
  rejuvenation kernel scores first and proposes from the re-scored trace; and
  `adaptive_single_site_mh` proposes from the re-scored trace (see X-5 below).
  Every re-scoring entry point already paid for `cur_scored` and then ignored
  it. A trace assembled with `insert_choice(.., 0.0)` - every fugue-evo
  `to_trace` / `trace_of` - stores `logp = 0` at each site, so the
  reverse-birth term of a block move or a branch-closing proposal was summed
  as zero and `log alpha` inflated by exactly the missing prior density on the
  very transition meant to correct for the dimension change. `ScoreGivenTrace`
  consumes no randomness, so for handler-produced traces nothing changes: the
  draws are bit-identical.

  Pinned by three exact-equivalence tests
  (`fgn6_{single_site_mh,block_regeneration,smc_rejuvenation}_from_zero_logp_*`):
  from the same seed a step started on the zero-`logp` trace must make the
  same decision and land on the same fully scored state as one started on the
  scored trace, over hundreds of seeds that include branch-closing moves. The
  block and rejuvenation tests fail on the pre-fix code.
- **`adaptive_smc_with_kernel` no longer ignores the kernel when
  `rejuvenation_steps == 0`, and particles enter every sweep with uniform
  weights (FG-N3, FG-N7)**. `PopulationKernel` gains `is_identity(&self) ->
  bool` (default `false`; `NoKernel` returns `true`), and the FG-43 shortcut -
  a single prior-importance reweight with no tempering ladder - is taken only
  when there is nothing that moves particles: no rejuvenation *and* an
  identity kernel. A `CrossoverKernel` with `rejuvenation_steps == 0` (the
  configuration fugue-evo's grammar SMC ships by default) used to be
  silently skipped because `kernel.sweep` lives inside the ladder the shortcut
  bypasses. Resampled clones now carry `weight = 1/n`, `log_weight = -ln n`
  instead of the beta-stale importance weights of the particles they were
  copied from, matching what the `PopulationKernel` contract (W) already told
  kernel authors to expect. Pinned by
  `fgn3_non_identity_kernel_is_applied_with_zero_rejuvenation_steps` (a
  recording kernel that asserts the weight contract on entry and is swept only
  at intermediate `beta`) and
  `fgn3_crossover_kernel_without_rejuvenation_is_applied_and_invariant`
  (analytic posterior means and log-evidence through the crossover sweeps).
- **`sequence_vec` / `traverse_vec` / `plate!` no longer recurse at
  construction over runs of `Pure` (FG-N5)**. `bind` on a `Pure` calls its
  continuation immediately and that continuation tail-called the next
  element's, so `k` consecutive `pure`s were `k` nested frames and
  `plate!(i in 0..100_000 => pure(i))` overflowed before `run` was reached
  (the FG-19 fix covered effectful elements only). Consecutive `Pure` values
  are now batched into one closure that extends the accumulator and calls the
  following continuation once; construction and interpretation are O(1) in
  stack depth for any mix of `pure` and effects, and input order is preserved
  across batch boundaries. Pinned by
  `fgn5_plate_over_pure_is_stack_safe_at_construction` (100 000 `pure`s on a
  512 KiB stack; overflows pre-fix) and
  `fgn5_sequence_vec_mixed_pure_and_effects_preserves_order_on_small_stack`.
- **Left-nested `bind` chains are documented as O(N) stack / O(N²) time, with
  the stack-safe shapes spelled out (FG-N4)**. `let mut m = ..; for .. { m =
  m.bind(..) }` wraps the first node's continuation once per iteration; the
  FG-19 trampoline removes recursion *across* nodes, not inside one node's
  continuation tower, and this is inherent to the CPS encoding (a
  Codensity-style `Model` would remove it and is out of scope for a patch
  release). Measured envelope in a debug build: ~5 000 observes on a 2 MiB
  thread stack (10 000 overflows), ~20 000 on 8 MiB (40 000 overflows), with
  20 000 nodes taking 20 s. `ModelExt::bind` now says so and names the
  alternatives - `traverse_vec` / `plate!` for independent sites, a
  build-from-the-back fold whose continuations *return* the rest of the chain
  for sequentially dependent ones - and the `monad` and `smc` explorables,
  which taught the left-nested loop, carry a warning with the right-nested
  version. Pinned by `fgn4_left_nested_bind_chain_of_a_few_thousand_observes_runs`
  (3 000 observes on 2 MiB: the documented envelope stays true) and
  `fgn4_traverse_vec_and_right_nested_fold_are_stack_safe_for_100k_observes`
  (both recommended shapes at 100 000 observes on 512 KiB).

### Changed

- **Maintainer binaries are no longer `[dev-dependencies]` (FG-N8)**.
  `cargo-llvm-cov`, `mdbook`, `mdbook-mermaid`, `mdbook-katex`,
  `mdbook-admonish`, `mdbook-linkcheck` and `mdbook-toc` are tools the
  maintainer runs, not libraries this crate links; declaring them compiled
  their entire dependency trees (`reqwest`, `tokio`, ...) into every
  `cargo test` and `cargo clippy --all-targets`. They are `cargo install`ed
  instead (`make install-dev-tools`; the coverage and docs workflows already
  did this). `clap` and `proptest`, declared but referenced by no target,
  went at the same time. `Cargo.lock` shrinks from 461 to 99 packages; the
  published library's `[dependencies]` are untouched. The MSRV CI job keeps
  its manifest-trimming step as a guard for the one remaining dev-dependency
  (`criterion`).

### Added

- **Single-step MH with overrides and a no-rescore variant (X-5)**.
  `adaptive_single_site_mh_with_overrides(rng, model_fn, current, adaptation,
  &overrides)` is the one-transition counterpart of
  `adaptive_mcmc_chain_with_overrides`: a caller driving a chain incrementally
  can now apply a `SiteProposal::Reflect { .. }` (or any other override) per
  address, which previously only the batch driver honoured.
  `adaptive_single_site_mh_cached(rng, model_fn, current, adaptation,
  &overrides, adapt)` is the transition the chain drivers run internally,
  exposed: it takes an **already-scored** `current` (every trace the other MH
  entry points return is one), reads the current log-density from its
  accumulators, executes the model exactly **once** (the proposal), and returns
  `Some((result, scored_trace, log_weight))` on acceptance or `None` on
  rejection - half the cost of the re-scoring variants, and pinned bit-for-bit
  against `adaptive_mcmc_chain` from the same seed. `adapt` selects
  adapt-vs-frozen scales (FG-57). `adaptive_single_site_mh` is now a thin
  wrapper over the `_with_overrides` variant with its signature and its RNG
  consumption unchanged. `proposal_kind_for_support` is exported at the root.

  Contract on `current` for the cached variant, stated on the function: its
  accumulators and per-choice `logp` are trusted, so a hand-assembled trace
  (`insert_choice(.., 0.0)`) must go through a re-scoring entry point first.
  The re-scoring variants take care of this themselves - see FG-N6 below - and
  on rejection now return the **re-scored** current trace rather than a clone
  of the caller's input, so every trace they return is a valid cached-step
  input (FG-40).

## [0.2.2] - 2026-08-05

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
