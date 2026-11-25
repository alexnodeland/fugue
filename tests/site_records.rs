//! Addressed factors (`factor_at`) and per-site weight records
//! (`RecordSites`, `SiteWeights`, `run_recorded`).
//!
//! What is pinned here:
//!
//! - `factor_at(addr, w)` and `factor(w)` give the same trace, bit for bit,
//!   under every built-in interpreter, synchronously and through `run_async`,
//!   and the same inference output from the same seed;
//! - a handler written before `on_factor_at` existed still receives every
//!   factor weight through `on_factor`;
//! - a recorder does not disturb the handler it wraps;
//! - the per-site sums equal the trace's flat accumulators;
//! - the forwarding rules through `Delegate`.

use fugue::core::model::AddressedFactor;
use fugue::runtime::handler::run;
use fugue::runtime::record::{run_recorded, RecordSites, SiteWeights};
use fugue::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// A model with sample and observe sites of all five value types and three
/// factors. `addressed` chooses between `factor_at` and `factor` for them.
fn all_sites(addressed: bool) -> Model<f64> {
    let fac = move |name: &'static str, w: f64| {
        if addressed {
            factor_at(Address::new(name), w)
        } else {
            factor(w)
        }
    };
    prob! {
        let x <- sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
        let b <- sample(addr!("b"), Bernoulli::new(0.4).unwrap());
        let n <- sample(addr!("n"), Poisson::new(3.0).unwrap());
        let c <- sample(addr!("c"), Categorical::new(vec![0.2, 0.3, 0.5]).unwrap());
        let i <- sample(addr!("i"), DiscreteUniform::new(-3, 3).unwrap());
        observe(addr!("obs/f64"), Normal::new(x, 0.7).unwrap(), 0.25);
        fac("fac/first", -0.5 * x * x);
        observe(addr!("obs/bool"), Bernoulli::new(0.3).unwrap(), b);
        observe(addr!("obs/u64"), Poisson::new(2.0 + n as f64).unwrap(), 4u64);
        fac("fac/second", -0.1 * (c as f64));
        observe(addr!("obs/usize"), Categorical::new(vec![0.6, 0.3, 0.1]).unwrap(), 1usize);
        observe(addr!("obs/i64"), DiscreteUniform::new(-5, 5).unwrap(), i);
        fac("fac/third", 0.75);
        pure(x)
    }
}

fn assert_same_trace(a: &Trace, b: &Trace) {
    assert_eq!(a.log_prior.to_bits(), b.log_prior.to_bits());
    assert_eq!(a.log_likelihood.to_bits(), b.log_likelihood.to_bits());
    assert_eq!(a.log_factors.to_bits(), b.log_factors.to_bits());
    assert_eq!(a.choices.len(), b.choices.len());
    for (ka, ca) in &a.choices {
        let cb = &b.choices[ka];
        assert_eq!(ca.value, cb.value);
        assert_eq!(ca.logp.to_bits(), cb.logp.to_bits());
    }
}

fn prior_run(seed: u64, addressed: bool) -> (f64, Trace) {
    let mut rng = StdRng::seed_from_u64(seed);
    run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        all_sites(addressed),
    )
}

// ---------------------------------------------------------------------------
// factor_at is factor, with a name
// ---------------------------------------------------------------------------

#[test]
fn factor_at_scores_as_factor_under_every_builtin_interpreter() {
    for seed in 0..25 {
        // PriorHandler
        let (xa, addressed) = prior_run(seed, true);
        let (xp, plain) = prior_run(seed, false);
        assert_eq!(xa, xp);
        assert_same_trace(&addressed, &plain);
        // The addressed factors left no choice and no likelihood behind.
        assert_eq!(addressed.choices.len(), 5);

        // ReplayHandler
        let replay = |addressed_model: bool| {
            let mut rng = StdRng::seed_from_u64(seed + 1000);
            run(
                ReplayHandler {
                    rng: &mut rng,
                    base: plain.clone(),
                    trace: Trace::default(),
                },
                all_sites(addressed_model),
            )
            .1
        };
        assert_same_trace(&replay(true), &replay(false));
        assert_same_trace(&replay(true), &plain);

        // ScoreGivenTrace and its safe sibling
        let score = |addressed_model: bool| {
            run(
                ScoreGivenTrace {
                    base: plain.clone(),
                    trace: Trace::default(),
                },
                all_sites(addressed_model),
            )
            .1
        };
        assert_same_trace(&score(true), &score(false));
        let safe = |addressed_model: bool| {
            run(
                SafeScoreGivenTrace {
                    base: plain.clone(),
                    trace: Trace::default(),
                    warn_on_error: false,
                },
                all_sites(addressed_model),
            )
            .1
        };
        assert_same_trace(&safe(true), &safe(false));

        // The reconciling scorer
        let reconciled = |addressed_model: bool| {
            let mut rng = StdRng::seed_from_u64(seed);
            score_given_trace_reconciled(plain.clone(), &mut rng, all_sites(addressed_model))
                .unwrap()
                .1
        };
        assert_same_trace(&reconciled(true), &reconciled(false));
    }
}

#[test]
fn factor_at_scores_as_factor_through_run_async() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    for seed in 0..10 {
        let (_, sync_trace) = prior_run(seed, false);
        let mut rng = StdRng::seed_from_u64(seed);
        let (_, async_trace) = rt.block_on(run_async(
            FromSync(PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            }),
            all_sites(true),
        ));
        assert_same_trace(&async_trace, &sync_trace);

        // FromSync keeps the address for a synchronous recorder underneath.
        let mut sites = SiteWeights::new();
        let mut rng = StdRng::seed_from_u64(seed);
        let (_, recorded) = rt.block_on(run_async(
            FromSync(RecordSites {
                inner: PriorHandler {
                    rng: &mut rng,
                    trace: Trace::default(),
                },
                sites: &mut sites,
            }),
            all_sites(true),
        ));
        assert_same_trace(&recorded, &sync_trace);
        assert_eq!(sites.factors().len(), 3);
        assert_eq!(sites.unaddressed_factors(), 0.0);
    }
}

/// An async handler that implements only the required methods: an addressed
/// factor reaches its `on_factor` through the default `on_factor_at`.
#[test]
fn async_handler_without_on_factor_at_still_receives_the_weight() {
    struct Factors(Vec<f64>);
    impl AsyncHandler for Factors {
        async fn on_sample_f64(&mut self, _: &Address, _: &dyn Distribution<f64>) -> f64 {
            0.0
        }
        async fn on_sample_bool(&mut self, _: &Address, _: &dyn Distribution<bool>) -> bool {
            false
        }
        async fn on_sample_u64(&mut self, _: &Address, _: &dyn Distribution<u64>) -> u64 {
            0
        }
        async fn on_sample_usize(&mut self, _: &Address, _: &dyn Distribution<usize>) -> usize {
            0
        }
        async fn on_observe_f64(&mut self, _: &Address, _: &dyn Distribution<f64>, _: f64) {
            panic!("an addressed factor must not arrive as an observation");
        }
        async fn on_observe_bool(&mut self, _: &Address, _: &dyn Distribution<bool>, _: bool) {}
        async fn on_observe_u64(&mut self, _: &Address, _: &dyn Distribution<u64>, _: u64) {}
        async fn on_observe_usize(&mut self, _: &Address, _: &dyn Distribution<usize>, _: usize) {}
        async fn on_factor(&mut self, logw: f64) {
            self.0.push(logw);
        }
        async fn finish(self) -> Trace {
            Trace {
                log_factors: self.0.iter().sum(),
                ..Default::default()
            }
        }
    }

    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let model = factor_at(addr!("a"), -1.0).bind(|_| factor(-2.0));
    let (_, trace) = rt.block_on(run_async(Factors(Vec::new()), model));
    assert_eq!(trace.log_factors, -3.0);
}

/// A handler that predates `on_factor_at` (it implements only the required
/// methods) sees an addressed factor as a factor, never as an observation.
#[test]
fn handler_without_on_factor_at_still_receives_the_weight() {
    #[derive(Default)]
    struct Legacy {
        factors: Vec<f64>,
        f64_observations: usize,
    }
    impl Handler for Legacy {
        fn on_sample_f64(&mut self, _: &Address, _: &dyn Distribution<f64>) -> f64 {
            0.5
        }
        fn on_sample_bool(&mut self, _: &Address, _: &dyn Distribution<bool>) -> bool {
            true
        }
        fn on_sample_u64(&mut self, _: &Address, _: &dyn Distribution<u64>) -> u64 {
            1
        }
        fn on_sample_usize(&mut self, _: &Address, _: &dyn Distribution<usize>) -> usize {
            0
        }
        fn on_sample_i64(&mut self, _: &Address, _: &dyn Distribution<i64>) -> i64 {
            0
        }
        fn on_observe_f64(&mut self, _: &Address, _: &dyn Distribution<f64>, _: f64) {
            self.f64_observations += 1;
        }
        fn on_observe_bool(&mut self, _: &Address, _: &dyn Distribution<bool>, _: bool) {}
        fn on_observe_u64(&mut self, _: &Address, _: &dyn Distribution<u64>, _: u64) {}
        fn on_observe_usize(&mut self, _: &Address, _: &dyn Distribution<usize>, _: usize) {}
        fn on_observe_i64(&mut self, _: &Address, _: &dyn Distribution<i64>, _: i64) {}
        fn on_factor(&mut self, logw: f64) {
            self.factors.push(logw);
        }
        fn finish(self) -> Trace {
            Trace {
                log_factors: self.factors.iter().sum(),
                // Smuggle the observation count out for the assertion.
                log_likelihood: self.f64_observations as f64,
                ..Default::default()
            }
        }
    }

    let (_, trace) = run(Legacy::default(), all_sites(true));
    // x = 0.5, c = 0: -0.5 * 0.25 - 0.0 + 0.75
    assert_eq!(trace.log_factors, -0.125 + 0.0 + 0.75);
    // Only the one real f64 observation arrived as an observation.
    assert_eq!(trace.log_likelihood, 1.0);
}

#[test]
fn factor_at_maps_nan_to_neg_inf_like_factor() {
    let (_, t) = prior_run_model(factor_at(addr!("bad"), f64::NAN));
    assert_eq!(t.log_factors, f64::NEG_INFINITY);
    let (_, t) = prior_run_model(factor(f64::NAN));
    assert_eq!(t.log_factors, f64::NEG_INFINITY);
    assert_eq!(
        AddressedFactor::new(f64::NAN).log_weight(),
        f64::NEG_INFINITY
    );
    // +inf and -inf pass through, as for `factor`.
    let (_, t) = prior_run_model(factor_at(addr!("reject"), f64::NEG_INFINITY));
    assert_eq!(t.log_factors, f64::NEG_INFINITY);
}

fn prior_run_model<A>(model: Model<A>) -> (A, Trace) {
    let mut rng = StdRng::seed_from_u64(0);
    run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        model,
    )
}

/// The node `factor_at` builds is an f64 observation of an `AddressedFactor`,
/// as documented for code that walks a `Model` itself, and it survives `bind`.
#[test]
fn factor_at_node_is_recognisable_by_a_custom_interpreter() {
    let model = factor_at(addr!("w"), -2.5).bind(|_| pure(7));
    match model {
        Model::ObserveF64 {
            addr, dist, value, ..
        } => {
            assert_eq!(addr, addr!("w"));
            assert_eq!(value, 0.0);
            assert_eq!(AddressedFactor::weight_of(&*dist), Some(-2.5));
            assert_eq!(dist.log_prob(&123.0), -2.5);
            let copy = dist.clone_box();
            assert_eq!(AddressedFactor::weight_of(&*copy), Some(-2.5));
        }
        _ => panic!("factor_at should build an ObserveF64 node"),
    }
}

/// Inference does not tell the two kinds of factor apart: same seed, same
/// output, for SMC (with rejuvenation) and for single-site MH.
#[test]
fn inference_is_identical_for_factor_and_factor_at() {
    let model = |addressed: bool| {
        move || {
            sample(addr!("mu"), Normal::new(0.0, 2.0).unwrap()).bind(move |mu| {
                let w = Normal::new(mu, 1.0).unwrap().log_prob(&1.3)
                    + Normal::new(mu, 1.0).unwrap().log_prob(&0.4);
                let f = if addressed {
                    factor_at(addr!("collapsed"), w)
                } else {
                    factor(w)
                };
                f.map(move |_| mu)
            })
        }
    };
    let smc = |addressed: bool| {
        let mut rng = StdRng::seed_from_u64(99);
        adaptive_smc(
            &mut rng,
            200,
            model(addressed),
            SMCConfig {
                rejuvenation_steps: 2,
                ..Default::default()
            },
        )
    };
    let (a, p) = (smc(true), smc(false));
    assert_eq!(a.log_evidence.to_bits(), p.log_evidence.to_bits());
    for (pa, pp) in a.particles.iter().zip(&p.particles) {
        assert_same_trace(&pa.trace, &pp.trace);
        assert_eq!(pa.log_weight.to_bits(), pp.log_weight.to_bits());
    }

    let mh = |addressed: bool| {
        let mut rng = StdRng::seed_from_u64(5);
        adaptive_mcmc_chain(&mut rng, model(addressed), 200, 50)
    };
    let (a, p) = (mh(true), mh(false));
    assert_eq!(a.len(), p.len());
    for ((va, ta), (vp, tp)) in a.iter().zip(&p) {
        assert_eq!(va, vp);
        assert_same_trace(ta, tp);
    }
}

// ---------------------------------------------------------------------------
// Recording
// ---------------------------------------------------------------------------

#[test]
fn recorder_does_not_disturb_the_inner_handler() {
    for seed in 0..25 {
        for addressed in [true, false] {
            let (x_plain, plain) = prior_run(seed, addressed);
            let mut rng = StdRng::seed_from_u64(seed);
            let (x_rec, recorded, _) = run_recorded(
                PriorHandler {
                    rng: &mut rng,
                    trace: Trace::default(),
                },
                all_sites(addressed),
            );
            assert_eq!(x_plain, x_rec);
            assert_same_trace(&plain, &recorded);
            // The RNG is where it would have been without the recorder.
            let mut reference = StdRng::seed_from_u64(seed);
            let _ = run(
                PriorHandler {
                    rng: &mut reference,
                    trace: Trace::default(),
                },
                all_sites(addressed),
            );
            assert_eq!(rng.gen::<u64>(), reference.gen::<u64>());
        }
    }
}

#[test]
fn records_every_observation_and_factor_by_address() {
    let mut rng = StdRng::seed_from_u64(4);
    let (x, trace, sites) = run_recorded(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        all_sites(true),
    );
    let names: Vec<&str> = sites.observations().keys().map(|a| a.as_str()).collect();
    assert_eq!(
        names,
        vec!["obs/bool", "obs/f64", "obs/i64", "obs/u64", "obs/usize"]
    );
    let names: Vec<&str> = sites.factors().keys().map(|a| a.as_str()).collect();
    assert_eq!(names, vec!["fac/first", "fac/second", "fac/third"]);

    // Each record is the site's own log-density.
    assert_eq!(
        sites.observation(&addr!("obs/f64")),
        Some(Normal::new(x, 0.7).unwrap().log_prob(&0.25))
    );
    assert_eq!(sites.observation(&addr!("obs/usize")), Some(0.3f64.ln()));
    assert_eq!(sites.observation(&addr!("obs/i64")), Some(-(11f64.ln())));
    assert_eq!(sites.factor(&addr!("fac/first")), Some(-0.5 * x * x));
    assert_eq!(sites.factor(&addr!("fac/third")), Some(0.75));
    assert_eq!(sites.observation(&addr!("fac/first")), None);
    assert_eq!(sites.factor(&addr!("obs/f64")), None);
    assert_eq!(sites.unaddressed_factors(), 0.0);
    assert!(!sites.is_empty());

    // Prefix sums partition the totals.
    assert_eq!(sites.observations_under("obs"), sites.log_likelihood());
    assert_eq!(sites.factors_under("fac"), sites.log_factors());
    assert_eq!(sites.under("obs"), sites.log_likelihood());
    assert_eq!(sites.observations_under("fac"), 0.0);
    assert_eq!(sites.under("nothing"), 0.0);
    // `has_prefix` boundaries: "ob" is not a prefix of "obs/...".
    assert_eq!(sites.observations_under("ob"), 0.0);
    assert!((sites.total() - (trace.log_likelihood + trace.log_factors)).abs() < 1e-12);

    // Anonymous factors go to the one unaddressed sum.
    let mut rng = StdRng::seed_from_u64(4);
    let (x, trace, sites) = run_recorded(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        all_sites(false),
    );
    assert!(sites.factors().is_empty());
    let c = trace.get_usize(&addr!("c")).unwrap() as f64;
    assert_eq!(sites.unaddressed_factors(), -0.5 * x * x + -0.1 * c + 0.75);
    assert_eq!(sites.log_factors(), sites.unaddressed_factors());
    assert_eq!(sites.log_factors(), trace.log_factors);
}

/// The headline invariant: per-site sums equal the flat accumulators, for
/// many random executions, under the prior handler and under a scorer.
#[test]
fn per_site_sums_agree_with_the_flat_accumulators() {
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-12 * (1.0 + a.abs().max(b.abs()));
    for seed in 0..500 {
        for addressed in [true, false] {
            let mut rng = StdRng::seed_from_u64(seed);
            let (_, trace, sites) = run_recorded(
                PriorHandler {
                    rng: &mut rng,
                    trace: Trace::default(),
                },
                all_sites(addressed),
            );
            assert!(close(sites.log_likelihood(), trace.log_likelihood));
            assert!(close(sites.log_factors(), trace.log_factors));
            assert!(close(
                sites.total() + trace.log_prior,
                trace.total_log_weight()
            ));

            let (_, scored, rescored_sites) = run_recorded(
                ScoreGivenTrace {
                    base: trace.clone(),
                    trace: Trace::default(),
                },
                all_sites(addressed),
            );
            assert_same_trace(&scored, &trace);
            assert_eq!(rescored_sites, sites);
        }
    }
}

/// With one observation and one factor there is no summation order to differ
/// in, so the agreement is exact; and a `-inf` site makes both sides `-inf`.
#[test]
fn single_site_totals_are_exact_and_infinities_agree() {
    let model = || {
        sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap()).bind(|mu| {
            observe(addr!("y"), Normal::new(mu, 0.3).unwrap(), 0.9)
                .bind(move |_| factor_at(addr!("f"), -mu.abs()))
        })
    };
    for seed in 0..100 {
        let mut rng = StdRng::seed_from_u64(seed);
        let (_, trace, sites) = run_recorded(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            model(),
        );
        assert_eq!(
            sites.log_likelihood().to_bits(),
            trace.log_likelihood.to_bits()
        );
        assert_eq!(sites.log_factors().to_bits(), trace.log_factors.to_bits());
    }

    let impossible = observe(addr!("in"), Uniform::new(0.0, 1.0).unwrap(), 0.5)
        .bind(|_| observe(addr!("out"), Uniform::new(0.0, 1.0).unwrap(), 2.0))
        .bind(|_| factor_at(addr!("nan"), f64::NAN));
    let (_, trace, sites) = prior_recorded(impossible);
    assert_eq!(sites.observation(&addr!("in")), Some(0.0));
    assert_eq!(sites.observation(&addr!("out")), Some(f64::NEG_INFINITY));
    assert_eq!(sites.factor(&addr!("nan")), Some(f64::NEG_INFINITY));
    assert_eq!(trace.log_likelihood, f64::NEG_INFINITY);
    assert_eq!(sites.log_likelihood(), f64::NEG_INFINITY);
    assert_eq!(sites.log_factors(), trace.log_factors);
}

fn prior_recorded<A>(model: Model<A>) -> (A, Trace, SiteWeights) {
    let mut rng = StdRng::seed_from_u64(0);
    run_recorded(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        model,
    )
}

#[test]
fn repeated_addresses_accumulate() {
    let model = observe(addr!("y"), Normal::new(0.0, 1.0).unwrap(), 0.0)
        .bind(|_| observe(addr!("y"), Normal::new(0.0, 1.0).unwrap(), 1.0))
        .bind(|_| factor_at(addr!("f"), -1.0))
        .bind(|_| factor_at(addr!("f"), -2.0))
        .bind(|_| factor(-4.0))
        .bind(|_| factor(-8.0));
    let (_, trace, sites) = prior_recorded(model);
    let n = Normal::new(0.0, 1.0).unwrap();
    assert_eq!(sites.observations().len(), 1);
    assert_eq!(
        sites.observation(&addr!("y")),
        Some(n.log_prob(&0.0) + n.log_prob(&1.0))
    );
    assert_eq!(sites.factor(&addr!("f")), Some(-3.0));
    assert_eq!(sites.unaddressed_factors(), -12.0);
    assert_eq!(sites.log_factors(), trace.log_factors);
    assert_eq!(sites.log_likelihood(), trace.log_likelihood);
}

#[test]
fn site_weights_manual_api() {
    let mut sites = SiteWeights::new();
    assert!(sites.is_empty());
    assert_eq!(sites.total(), 0.0);
    sites.record_observation(&addr!("a"), f64::NAN);
    assert_eq!(sites.observation(&addr!("a")), Some(f64::NEG_INFINITY));
    sites.record_factor(&addr!("b"), -1.0);
    sites.record_unaddressed_factor(-2.0);
    assert!(!sites.is_empty());
    let copy = sites.clone();
    assert_eq!(copy, sites);
    sites.clear();
    assert!(sites.is_empty());
    assert_eq!(sites, SiteWeights::default());
}

// ---------------------------------------------------------------------------
// Composition with Delegate
// ---------------------------------------------------------------------------

/// `Delegate::new` is transparent: a recorder underneath keeps factor addresses.
#[test]
fn delegate_new_keeps_factor_addresses() {
    let mut sites = SiteWeights::new();
    let mut rng = StdRng::seed_from_u64(8);
    let (_, trace) = run(
        Delegate::new(RecordSites {
            inner: PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            sites: &mut sites,
        }),
        all_sites(true),
    );
    assert_eq!(sites.factors().len(), 3);
    assert_eq!(sites.unaddressed_factors(), 0.0);
    assert_same_trace(&trace, &prior_run(8, true).1);
}

/// An `on_factor` override written before addressed factors existed sees every
/// factor weight, addressed or not.
#[test]
fn an_on_factor_override_sees_addressed_factors() {
    struct Halve;
    impl<H: Handler> Overrides<H> for Halve {
        fn on_factor(&mut self, inner: &mut H, logw: f64) {
            inner.on_factor(0.5 * logw)
        }
    }
    let run_halved = |addressed: bool| {
        let mut rng = StdRng::seed_from_u64(8);
        run(
            Delegate::with(
                PriorHandler {
                    rng: &mut rng,
                    trace: Trace::default(),
                },
                Halve,
            ),
            all_sites(addressed),
        )
        .1
    };
    let (a, p) = (run_halved(true), run_halved(false));
    assert_same_trace(&a, &p);
    let (_, full) = prior_run(8, true);
    assert!((a.log_factors - 0.5 * full.log_factors).abs() < 1e-12);
}

/// The default hook drops the address (documented); forwarding `on_factor_at`
/// keeps it, and a recorder on the outside always sees it.
#[test]
fn overrides_and_factor_addresses() {
    struct Plain;
    impl<H: Handler> Overrides<H> for Plain {}

    struct Forwarding;
    impl<H: Handler> Overrides<H> for Forwarding {
        fn on_factor_at(&mut self, inner: &mut H, addr: &Address, logw: f64) {
            inner.on_factor_at(addr, logw)
        }
    }

    fn inner_recorded<O>(overrides: O) -> SiteWeights
    where
        O: for<'a, 'b> Overrides<RecordSites<'a, PriorHandler<'b, StdRng>>>,
    {
        let mut sites = SiteWeights::new();
        let mut rng = StdRng::seed_from_u64(8);
        let _ = run(
            Delegate::with(
                RecordSites {
                    inner: PriorHandler {
                        rng: &mut rng,
                        trace: Trace::default(),
                    },
                    sites: &mut sites,
                },
                overrides,
            ),
            all_sites(true),
        );
        sites
    }

    let dropped = inner_recorded(Plain);
    assert!(dropped.factors().is_empty());
    let kept = inner_recorded(Forwarding);
    assert_eq!(kept.factors().len(), 3);
    // Either way no weight is lost.
    assert!((dropped.log_factors() - kept.log_factors()).abs() < 1e-12);
    assert_eq!(dropped.observations(), kept.observations());

    // Recorder outside the delegate: addresses arrive regardless.
    let mut rng = StdRng::seed_from_u64(8);
    let (_, _, outside) = run_recorded(
        Delegate::with(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            Plain,
        ),
        all_sites(true),
    );
    assert_eq!(outside, kept);
}

/// Recorders nest: both see the same sites.
#[test]
fn recorders_nest() {
    let mut inner_sites = SiteWeights::new();
    let mut rng = StdRng::seed_from_u64(2);
    let (_, _, outer_sites) = run_recorded(
        RecordSites {
            inner: PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            sites: &mut inner_sites,
        },
        all_sites(true),
    );
    assert_eq!(inner_sites, outer_sites);
    assert_eq!(inner_sites.factors().len(), 3);
}
