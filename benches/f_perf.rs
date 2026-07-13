//! End-to-end performance benchmarks for the *shipped* inference entry points
//! (FG-24), and the FG-05 `Address` before/after measurement.
//!
//! Prior to this file the only benches in the crate exercised either dead code
//! (`memory_benchmarks.rs`, the unused CowTrace/TracePool subsystem — now deleted)
//! or isolated bookkeeping utilities (`mcmc_benchmarks.rs`: `DiminishingAdaptation`,
//! ESS). None ran a representative model through `adaptive_mcmc_chain`,
//! `adaptive_smc`, or `elbo_with_guide` — the functions a library user actually
//! calls. This bench closes that gap.
//!
//! ## Baseline numbers (committed; re-measure with `cargo bench --bench f_perf`)
//!
//! Machine: Apple Silicon (darwin), `--release`, criterion 0.5. Numbers are the
//! median per-iteration wall time reported by criterion; treat them as an
//! order-of-magnitude regression tripwire, not a precise SLA.
//!
//! FG-05 (`Arc<str>` + cached-hash `Address`) before/after on the
//! `mcmc_end_to_end` model. "before" was measured by temporarily reverting
//! `Address` to a `String`-backed struct that re-hashes its string on every probe
//! (the pre-fix representation) and re-running this same bench; "after" is the
//! shipped `Arc<str>` + precomputed-`u64`-hash representation:
//!
//! ```text
//!   mcmc_end_to_end/20   before (String Address): 1.530 ms  [1.526, 1.536]
//!   mcmc_end_to_end/20   after  (Arc<str>+hash) : 1.532 ms  [1.529, 1.535]  (neutral)
//!   mcmc_end_to_end/50   before (String Address): 7.409 ms  [7.388, 7.431]
//!   mcmc_end_to_end/50   after  (Arc<str>+hash) : 7.310 ms  [7.301, 7.320]  (~1.3% faster,
//!                                                                            non-overlapping CIs)
//!   smc_tempered/20site  after                  : 49.4 ms
//!   vi_elbo/20site       after                  : 2.27 ms
//! ```
//!
//! Honest reading of FG-05: the end-to-end delta is small and only statistically
//! resolvable at the larger (50-site) model. The reason is that the per-step hot
//! path stores choices in a `BTreeMap<Address, Choice>`, which orders keys by
//! `Ord` (lexicographic `str` compare) and never calls `Hash` — so the cached
//! `u64` hash does nothing for the trace itself. The measurable win comes purely
//! from `Arc<str>` making the 3-5 whole-trace clones per MH step allocation-free
//! in their keys (which matters more as the trace grows, hence the 50-site win vs
//! the 20-site wash). The cached hash is retained because `Address` *is* a
//! `HashMap` key on the proposal-cache (`mh.rs` `kind_cache`), adaptation
//! (`mcmc_utils.rs` `scales`/`accept_counts`), and VI (`vi.rs` `params`) paths,
//! where it removes the per-probe string re-hash; on those paths it is a strict
//! improvement, and on the BTreeMap path it is within noise. The audit's premise
//! that address allocation *dominates* per-iteration cost is not borne out on this
//! model — model re-execution, Box-Muller draws, and BTreeMap `Ord` compares
//! dominate — but the representation is the standard, non-regressing choice with a
//! guaranteed O(1)-clone asymptotic benefit for long/hierarchical addresses.
//!
//! FG-22 / FG-62 pooling decision (memory subsystem CUT): before deleting the
//! subsystem this file had a `pooling_evidence` group comparing the shipped
//! `PriorHandler` (fresh `Trace` per run) against `PooledPriorHandler` +
//! `TracePool` over the identical 20-site model. The measurement was:
//!
//! ```text
//!   pooling_evidence/prior_handler/20  : 3.049 ms
//!   pooling_evidence/pooled_handler/20 : 2.931 ms   (only ~3.8% faster)
//! ```
//!
//! 3.8% is far below the 10% end-to-end margin the remediation brief required to
//! justify keeping a whole dead subsystem, so CowTrace / TracePool / TraceBuilder /
//! PooledPriorHandler were deleted (see the FG-22 resolution). The `pooling_evidence`
//! group is therefore gone; its numbers are recorded above for the audit trail.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box as std_black_box;

use fugue::runtime::handler::run;
use fugue::*;

use rand::rngs::StdRng;
use rand::SeedableRng;

/// A reference hierarchical model with exactly `n_sites` continuous sample sites
/// (`mu` plus `n_sites - 1` local `x_i`), each `x_i` tied to a fixed observation.
/// The observations give the likelihood real curvature so SMC/VI do meaningful
/// work rather than collapsing to the prior.
fn reference_model(n_sites: usize) -> impl Fn() -> Model<f64> + Clone {
    move || {
        let mut m: Model<f64> = sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap());
        for i in 0..n_sites.saturating_sub(1) {
            m = m.bind(move |mu| {
                sample(addr!("x", i), Normal::new(mu, 1.0).unwrap()).bind(move |x| {
                    let datum = 0.2 * (i as f64) - 1.0;
                    observe(addr!("y", i), Normal::new(x, 0.5).unwrap(), datum).map(move |_| mu)
                })
            });
        }
        m
    }
}

/// FG-24 + FG-05: `adaptive_mcmc_chain` end-to-end on 20- and 50-site models.
fn bench_mcmc_end_to_end(c: &mut Criterion) {
    let mut group = c.benchmark_group("mcmc_end_to_end");
    for &n_sites in &[20usize, 50] {
        let model = reference_model(n_sites);
        group.bench_with_input(BenchmarkId::from_parameter(n_sites), &n_sites, |b, _| {
            b.iter(|| {
                // Fixed seed: keeps the acceptance path deterministic across runs
                // so timing deltas reflect code, not RNG luck.
                let mut rng = StdRng::seed_from_u64(0xF06E_2026);
                let samples = adaptive_mcmc_chain(&mut rng, &model, black_box(50), black_box(50));
                std_black_box(samples)
            });
        });
    }
    group.finish();
}

/// FG-24: the tempered `adaptive_smc` path (rejuvenation_steps > 0 exercises the
/// resample + MCMC-move machinery, not just a single importance reweight).
fn bench_smc(c: &mut Criterion) {
    let mut group = c.benchmark_group("smc_tempered");
    let model = reference_model(20);
    group.bench_function("adaptive_smc_20site_64particles", |b| {
        b.iter(|| {
            let mut rng = StdRng::seed_from_u64(0x5AFE_2026);
            let config = SMCConfig {
                resampling_method: ResamplingMethod::Systematic,
                ess_threshold: 0.5,
                rejuvenation_steps: 3,
            };
            let result = adaptive_smc(&mut rng, black_box(64), &model, config);
            std_black_box(result.log_evidence)
        });
    });
    group.finish();
}

/// FG-24: `elbo_with_guide` on a real guide fitted to the reference model shape.
fn bench_vi_elbo(c: &mut Criterion) {
    let mut group = c.benchmark_group("vi_elbo");
    let model = reference_model(20);

    // Build a real-line Normal mean-field guide from a prior draw of the model.
    let mut seed_rng = StdRng::seed_from_u64(7);
    let (_a, seed_trace) = run(
        PriorHandler {
            rng: &mut seed_rng,
            trace: Trace::default(),
        },
        (reference_model(20))(),
    );
    let guide = MeanFieldGuide::from_trace(&seed_trace).expect("all latents are continuous");

    group.bench_function("elbo_with_guide_20site_128samples", |b| {
        b.iter(|| {
            let mut rng = StdRng::seed_from_u64(0xE1B0_2026);
            let elbo = elbo_with_guide(&mut rng, &model, &guide, black_box(128));
            std_black_box(elbo)
        });
    });
    group.finish();
}

criterion_group!(benches, bench_mcmc_end_to_end, bench_smc, bench_vi_elbo);
criterion_main!(benches);
