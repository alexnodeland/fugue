#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/docs/runtime/delegate.md"))]

use crate::core::address::Address;
use crate::core::distribution::Distribution;
use crate::runtime::handler::Handler;
use crate::runtime::trace::Trace;

/// What a [`Delegate`] does at each kind of site.
///
/// Every method defaults to forwarding to the inner handler, so an
/// implementation overrides only the sites it cares about. The hooks mirror
/// [`Handler`]'s methods, with the same names and argument order, and take the
/// inner handler as an extra argument after `self`. To forward a site, call the
/// same method on `inner`, before or after your own work; a hook that does not
/// forward answers the site itself, and the inner handler never sees it.
///
/// Implement it for every `H: Handler` to get a decorator that wraps any
/// handler, or for one concrete handler to reach its fields, such as a
/// [`PriorHandler`](crate::runtime::interpreters::PriorHandler)'s `trace`.
/// `()` implements it and overrides nothing.
///
/// Example (count the `f64` sample sites and forward every site):
/// ```rust
/// # use fugue::*;
/// # use fugue::runtime::interpreters::PriorHandler;
/// # use rand::rngs::StdRng;
/// # use rand::SeedableRng;
/// /// Counts the f64 sample sites. Every other hook keeps its default: forward.
/// struct CountF64<'a>(&'a mut usize);
///
/// impl<H: Handler> Overrides<H> for CountF64<'_> {
///     fn on_sample_f64(
///         &mut self,
///         inner: &mut H,
///         addr: &Address,
///         dist: &dyn Distribution<f64>,
///     ) -> f64 {
///         *self.0 += 1;
///         inner.on_sample_f64(addr, dist)
///     }
/// }
///
/// let model = prob! {
///     let mu <- sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap());
///     let heads <- sample(addr!("coin"), Bernoulli::new(0.5).unwrap());
///     observe(addr!("y"), Normal::new(mu, 1.0).unwrap(), 0.3);
///     pure(heads)
/// };
///
/// let mut count = 0;
/// let mut rng = StdRng::seed_from_u64(42);
/// let handler = Delegate::with(
///     PriorHandler { rng: &mut rng, trace: Trace::default() },
///     CountF64(&mut count),
/// );
/// let (_heads, trace) = runtime::handler::run(handler, model);
///
/// assert_eq!(count, 1); // "mu": the bool site and the observation were only forwarded
/// assert_eq!(trace.choices.len(), 2); // PriorHandler recorded both samples
/// ```
pub trait Overrides<H: Handler> {
    /// Hook for an `f64` sample site. Defaults to `inner.on_sample_f64(addr, dist)`.
    fn on_sample_f64(
        &mut self,
        inner: &mut H,
        addr: &Address,
        dist: &dyn Distribution<f64>,
    ) -> f64 {
        inner.on_sample_f64(addr, dist)
    }

    /// Hook for a `bool` sample site. Defaults to `inner.on_sample_bool(addr, dist)`.
    fn on_sample_bool(
        &mut self,
        inner: &mut H,
        addr: &Address,
        dist: &dyn Distribution<bool>,
    ) -> bool {
        inner.on_sample_bool(addr, dist)
    }

    /// Hook for a `u64` sample site. Defaults to `inner.on_sample_u64(addr, dist)`.
    fn on_sample_u64(
        &mut self,
        inner: &mut H,
        addr: &Address,
        dist: &dyn Distribution<u64>,
    ) -> u64 {
        inner.on_sample_u64(addr, dist)
    }

    /// Hook for a `usize` sample site. Defaults to `inner.on_sample_usize(addr, dist)`.
    fn on_sample_usize(
        &mut self,
        inner: &mut H,
        addr: &Address,
        dist: &dyn Distribution<usize>,
    ) -> usize {
        inner.on_sample_usize(addr, dist)
    }

    /// Hook for an `i64` sample site. Defaults to `inner.on_sample_i64(addr, dist)`,
    /// so a delegate panics at an `i64` site only if its inner handler does
    /// (see [`Handler::on_sample_i64`]).
    fn on_sample_i64(
        &mut self,
        inner: &mut H,
        addr: &Address,
        dist: &dyn Distribution<i64>,
    ) -> i64 {
        inner.on_sample_i64(addr, dist)
    }

    /// Hook for an `f64` observation. Defaults to `inner.on_observe_f64(addr, dist, value)`.
    fn on_observe_f64(
        &mut self,
        inner: &mut H,
        addr: &Address,
        dist: &dyn Distribution<f64>,
        value: f64,
    ) {
        inner.on_observe_f64(addr, dist, value)
    }

    /// Hook for a `bool` observation. Defaults to `inner.on_observe_bool(addr, dist, value)`.
    fn on_observe_bool(
        &mut self,
        inner: &mut H,
        addr: &Address,
        dist: &dyn Distribution<bool>,
        value: bool,
    ) {
        inner.on_observe_bool(addr, dist, value)
    }

    /// Hook for a `u64` observation. Defaults to `inner.on_observe_u64(addr, dist, value)`.
    fn on_observe_u64(
        &mut self,
        inner: &mut H,
        addr: &Address,
        dist: &dyn Distribution<u64>,
        value: u64,
    ) {
        inner.on_observe_u64(addr, dist, value)
    }

    /// Hook for a `usize` observation. Defaults to
    /// `inner.on_observe_usize(addr, dist, value)`.
    fn on_observe_usize(
        &mut self,
        inner: &mut H,
        addr: &Address,
        dist: &dyn Distribution<usize>,
        value: usize,
    ) {
        inner.on_observe_usize(addr, dist, value)
    }

    /// Hook for an `i64` observation. Defaults to
    /// `inner.on_observe_i64(addr, dist, value)`, so, as for `i64` samples, a
    /// delegate panics only if its inner handler does.
    fn on_observe_i64(
        &mut self,
        inner: &mut H,
        addr: &Address,
        dist: &dyn Distribution<i64>,
        value: i64,
    ) {
        inner.on_observe_i64(addr, dist, value)
    }

    /// Hook for a factor. Defaults to `inner.on_factor(logw)`.
    fn on_factor(&mut self, inner: &mut H, logw: f64) {
        inner.on_factor(logw)
    }

    /// Hook for the end of the run: takes the overrides and the inner handler by
    /// value and returns the trace. Defaults to `inner.finish()`.
    fn finish(self, inner: H) -> Trace
    where
        Self: Sized,
    {
        inner.finish()
    }
}

/// `()` overrides nothing: `Delegate<H, ()>`, what [`Delegate::new`] builds,
/// forwards every site to `H`.
impl<H: Handler> Overrides<H> for () {}

/// A [`Handler`] that forwards every site to an inner handler, through the
/// hooks of an [`Overrides`] value.
///
/// Each site goes to the matching hook of `overrides`, with `inner` as its
/// extra argument, and each hook defaults to forwarding the site to `inner`. So
/// `Delegate::new(h)` (overrides `()`) behaves exactly like `h`, and
/// `Delegate::with(h, o)` behaves like `h` except at the sites `o` overrides.
///
/// Delegates nest. In `Delegate<Delegate<H, A>, B>` the hooks of `B` see each
/// site first, and their `inner` is the `Delegate<H, A>`, which passes the site
/// on through `A` to `H`. At the end of the run `B::finish` runs first, and
/// `A::finish` runs when `B::finish` finishes its inner handler.
///
/// A `Delegate` is a plain struct of its two fields: no boxing, and every hook
/// call is statically dispatched.
///
/// Example (two decorators nested around a `PriorHandler`):
/// ```rust
/// # use fugue::*;
/// # use fugue::runtime::interpreters::PriorHandler;
/// # use rand::rngs::StdRng;
/// # use rand::SeedableRng;
/// /// Logs the f64 sample sites it sees, under a name, and forwards them.
/// struct Log<'a>(&'static str, &'a std::cell::RefCell<Vec<String>>);
///
/// impl<H: Handler> Overrides<H> for Log<'_> {
///     fn on_sample_f64(
///         &mut self,
///         inner: &mut H,
///         addr: &Address,
///         dist: &dyn Distribution<f64>,
///     ) -> f64 {
///         self.1.borrow_mut().push(format!("{} saw {}", self.0, addr));
///         inner.on_sample_f64(addr, dist)
///     }
/// }
///
/// let log = std::cell::RefCell::new(Vec::new());
/// let mut rng = StdRng::seed_from_u64(1);
/// let prior = PriorHandler { rng: &mut rng, trace: Trace::default() };
/// let handler = Delegate::with(Delegate::with(prior, Log("inner", &log)), Log("outer", &log));
///
/// let (_x, trace) =
///     runtime::handler::run(handler, sample(addr!("x"), Normal::new(0.0, 1.0).unwrap()));
///
/// assert_eq!(log.into_inner(), ["outer saw x", "inner saw x"]);
/// assert!(trace.get_f64(&addr!("x")).is_some()); // recorded by the PriorHandler
/// ```
#[derive(Clone, Debug)]
pub struct Delegate<H, O = ()> {
    /// The handler that every site not overridden is forwarded to.
    pub inner: H,
    /// The hooks, and whatever state they keep.
    pub overrides: O,
}

impl<H> Delegate<H> {
    /// Wrap `inner` with no overrides, so every site is forwarded to it.
    ///
    /// The result behaves exactly like `inner`: same values, same trace.
    ///
    /// Example:
    /// ```rust
    /// # use fugue::*;
    /// # use fugue::runtime::interpreters::PriorHandler;
    /// # use rand::rngs::StdRng;
    /// # use rand::SeedableRng;
    /// let model = || sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
    ///
    /// let mut rng = StdRng::seed_from_u64(7);
    /// let (x, _) = runtime::handler::run(
    ///     PriorHandler { rng: &mut rng, trace: Trace::default() },
    ///     model(),
    /// );
    ///
    /// let mut rng = StdRng::seed_from_u64(7);
    /// let delegate = Delegate::new(PriorHandler { rng: &mut rng, trace: Trace::default() });
    /// let (y, trace) = runtime::handler::run(delegate, model());
    ///
    /// assert_eq!(x, y); // same seed, same draw
    /// assert_eq!(trace.get_f64(&addr!("x")), Some(x));
    /// ```
    pub fn new(inner: H) -> Self {
        Delegate {
            inner,
            overrides: (),
        }
    }
}

impl<H, O> Delegate<H, O> {
    /// Wrap `inner`, sending each site through the hooks of `overrides` first.
    ///
    /// Example (clamp every `f64` sample to `[-1, 1]` before the model sees it;
    /// the trace keeps the value the inner handler drew):
    /// ```rust
    /// # use fugue::*;
    /// # use fugue::runtime::interpreters::PriorHandler;
    /// # use rand::rngs::StdRng;
    /// # use rand::SeedableRng;
    /// struct Clamp;
    ///
    /// impl<H: Handler> Overrides<H> for Clamp {
    ///     fn on_sample_f64(
    ///         &mut self,
    ///         inner: &mut H,
    ///         addr: &Address,
    ///         dist: &dyn Distribution<f64>,
    ///     ) -> f64 {
    ///         inner.on_sample_f64(addr, dist).clamp(-1.0, 1.0)
    ///     }
    /// }
    ///
    /// let mut rng = StdRng::seed_from_u64(3);
    /// let handler = Delegate::with(PriorHandler { rng: &mut rng, trace: Trace::default() }, Clamp);
    /// let (x, _trace) =
    ///     runtime::handler::run(handler, sample(addr!("x"), Normal::new(0.0, 10.0).unwrap()));
    ///
    /// assert!((-1.0..=1.0).contains(&x));
    /// ```
    pub fn with(inner: H, overrides: O) -> Self {
        Delegate { inner, overrides }
    }
}

/// Every method calls the hook of the same name on `overrides`, passing `inner`.
impl<H: Handler, O: Overrides<H>> Handler for Delegate<H, O> {
    fn on_sample_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>) -> f64 {
        self.overrides.on_sample_f64(&mut self.inner, addr, dist)
    }

    fn on_sample_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>) -> bool {
        self.overrides.on_sample_bool(&mut self.inner, addr, dist)
    }

    fn on_sample_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>) -> u64 {
        self.overrides.on_sample_u64(&mut self.inner, addr, dist)
    }

    fn on_sample_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>) -> usize {
        self.overrides.on_sample_usize(&mut self.inner, addr, dist)
    }

    fn on_sample_i64(&mut self, addr: &Address, dist: &dyn Distribution<i64>) -> i64 {
        self.overrides.on_sample_i64(&mut self.inner, addr, dist)
    }

    fn on_observe_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>, value: f64) {
        self.overrides
            .on_observe_f64(&mut self.inner, addr, dist, value)
    }

    fn on_observe_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>, value: bool) {
        self.overrides
            .on_observe_bool(&mut self.inner, addr, dist, value)
    }

    fn on_observe_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>, value: u64) {
        self.overrides
            .on_observe_u64(&mut self.inner, addr, dist, value)
    }

    fn on_observe_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>, value: usize) {
        self.overrides
            .on_observe_usize(&mut self.inner, addr, dist, value)
    }

    fn on_observe_i64(&mut self, addr: &Address, dist: &dyn Distribution<i64>, value: i64) {
        self.overrides
            .on_observe_i64(&mut self.inner, addr, dist, value)
    }

    fn on_factor(&mut self, logw: f64) {
        self.overrides.on_factor(&mut self.inner, logw)
    }

    fn finish(self) -> Trace {
        self.overrides.finish(self.inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::addr;
    use crate::core::distribution::*;
    use crate::core::model::{factor, observe, pure, sample, Model};
    use crate::runtime::handler::run;
    use crate::runtime::interpreters::{PriorHandler, ReplayHandler, ScoreGivenTrace};
    use crate::runtime::trace::ChoiceValue;
    use rand::rngs::StdRng;
    use rand::{RngCore, SeedableRng};
    use std::cell::RefCell;

    type Sites = (f64, bool, u64, usize, i64);

    /// A sample site and an observation of each of the five value types, then a
    /// factor. The observations and the factor depend on the sampled values, so
    /// every accumulator depends on what the handler did at the sample sites.
    fn all_sites() -> Model<Sites> {
        crate::prob! {
            let f <- sample(addr!("f"), Normal::new(0.0, 1.0).unwrap());
            let b <- sample(addr!("b"), Bernoulli::new(0.3).unwrap());
            let u <- sample(addr!("u"), Poisson::new(2.5).unwrap());
            let z <- sample(addr!("z"), Categorical::new(vec![0.2, 0.5, 0.3]).unwrap());
            let i <- sample(addr!("i"), DiscreteUniform::new(-3, 3).unwrap());
            observe(addr!("f_obs"), Normal::new(f, 1.0).unwrap(), 0.4);
            observe(addr!("b_obs"), Bernoulli::new(if b { 0.8 } else { 0.2 }).unwrap(), true);
            observe(addr!("u_obs"), Poisson::new(1.0 + u as f64).unwrap(), 3);
            observe(addr!("z_obs"), Categorical::new(vec![0.6, 0.3, 0.1]).unwrap(), z);
            observe(addr!("i_obs"), DiscreteUniform::new(i - 2, i + 4).unwrap(), i + 1);
            factor(-0.5 * f * f);
            pure((f, b, u, z, i))
        }
    }

    /// The same trace, bit for bit: the same choices (address, value, logp) and
    /// the same three accumulators.
    fn assert_same_trace(a: &Trace, b: &Trace) {
        let keys = |t: &Trace| t.choices.keys().cloned().collect::<Vec<_>>();
        assert_eq!(keys(a), keys(b));
        for (addr, ca) in &a.choices {
            let cb = &b.choices[addr];
            assert_eq!(ca.addr, cb.addr);
            assert_eq!(ca.value, cb.value, "value at {}", addr);
            assert_eq!(ca.logp.to_bits(), cb.logp.to_bits(), "logp at {}", addr);
        }
        assert_eq!(a.log_prior.to_bits(), b.log_prior.to_bits());
        assert_eq!(a.log_likelihood.to_bits(), b.log_likelihood.to_bits());
        assert_eq!(a.log_factors.to_bits(), b.log_factors.to_bits());
    }

    /// Guards the comparisons against passing vacuously: every site was
    /// visited and every accumulator is a finite, nonzero number.
    fn assert_every_site_recorded(t: &Trace) {
        assert_eq!(t.choices.len(), 5);
        assert!(t.get_f64(&addr!("f")).is_some());
        assert!(t.get_bool(&addr!("b")).is_some());
        assert!(t.get_u64(&addr!("u")).is_some());
        assert!(t.get_usize(&addr!("z")).is_some());
        assert!(t.get_i64(&addr!("i")).is_some());
        for w in [t.log_prior, t.log_likelihood, t.log_factors] {
            assert!(w.is_finite() && w != 0.0, "accumulator {}", w);
        }
    }

    fn prior_run(seed: u64) -> (Sites, Trace) {
        let mut rng = StdRng::seed_from_u64(seed);
        run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            all_sites(),
        )
    }

    #[test]
    fn delegate_new_matches_prior_handler() {
        let mut rng_a = StdRng::seed_from_u64(11);
        let (va, ta) = run(
            PriorHandler {
                rng: &mut rng_a,
                trace: Trace::default(),
            },
            all_sites(),
        );
        let mut rng_b = StdRng::seed_from_u64(11);
        let (vb, tb) = run(
            Delegate::new(PriorHandler {
                rng: &mut rng_b,
                trace: Trace::default(),
            }),
            all_sites(),
        );

        assert_every_site_recorded(&ta);
        assert_eq!(va, vb);
        assert_same_trace(&ta, &tb);
        // The delegate consumed exactly the randomness the handler did.
        assert_eq!(rng_a.next_u64(), rng_b.next_u64());
    }

    #[test]
    fn delegate_new_matches_replay_handler() {
        // A base trace missing two sites, so the replay reuses three values and
        // samples two fresh from the RNG.
        let (_, mut base) = prior_run(12);
        base.choices.remove(&addr!("u"));
        base.choices.remove(&addr!("i"));

        let mut rng_a = StdRng::seed_from_u64(13);
        let (va, ta) = run(
            ReplayHandler {
                rng: &mut rng_a,
                base: base.clone(),
                trace: Trace::default(),
            },
            all_sites(),
        );
        let mut rng_b = StdRng::seed_from_u64(13);
        let (vb, tb) = run(
            Delegate::new(ReplayHandler {
                rng: &mut rng_b,
                base,
                trace: Trace::default(),
            }),
            all_sites(),
        );

        assert_every_site_recorded(&ta);
        assert_eq!(va, vb);
        assert_same_trace(&ta, &tb);
        assert_eq!(rng_a.next_u64(), rng_b.next_u64());
    }

    #[test]
    fn delegate_new_matches_score_given_trace() {
        let (_, base) = prior_run(14);

        let (va, ta) = run(
            ScoreGivenTrace {
                base: base.clone(),
                trace: Trace::default(),
            },
            all_sites(),
        );
        let (vb, tb) = run(
            Delegate::new(ScoreGivenTrace {
                base,
                trace: Trace::default(),
            }),
            all_sites(),
        );

        assert_every_site_recorded(&ta);
        assert_eq!(va, vb);
        assert_same_trace(&ta, &tb);
    }

    /// Logs the `usize` sites it sees, samples and observations, and forwards them.
    struct UsizeSites<'a>(&'a mut Vec<String>);

    impl<H: Handler> Overrides<H> for UsizeSites<'_> {
        fn on_sample_usize(
            &mut self,
            inner: &mut H,
            addr: &Address,
            dist: &dyn Distribution<usize>,
        ) -> usize {
            let value = inner.on_sample_usize(addr, dist);
            self.0.push(format!("sample {} = {}", addr, value));
            value
        }

        fn on_observe_usize(
            &mut self,
            inner: &mut H,
            addr: &Address,
            dist: &dyn Distribution<usize>,
            value: usize,
        ) {
            self.0.push(format!("observe {} = {}", addr, value));
            inner.on_observe_usize(addr, dist, value)
        }
    }

    #[test]
    fn overrides_see_their_own_sites_and_forward_the_rest() {
        let mut seen = Vec::new();
        let mut rng = StdRng::seed_from_u64(21);
        let (v, t) = run(
            Delegate::with(
                PriorHandler {
                    rng: &mut rng,
                    trace: Trace::default(),
                },
                UsizeSites(&mut seen),
            ),
            all_sites(),
        );

        let z = v.3;
        assert_eq!(
            seen,
            [
                format!("sample z = {}", z),
                format!("observe z_obs = {}", z)
            ]
        );
        // Every site, overridden or not, reached the PriorHandler as it would
        // have without the delegate.
        let (pv, pt) = prior_run(21);
        assert_eq!(v, pv);
        assert_same_trace(&t, &pt);
    }

    /// Resolves the `usize` sample sites itself, recording the choice in the
    /// `PriorHandler`'s trace the way `PriorHandler` records a draw.
    struct Pick(usize);

    impl<R: RngCore> Overrides<PriorHandler<'_, R>> for Pick {
        fn on_sample_usize(
            &mut self,
            inner: &mut PriorHandler<'_, R>,
            addr: &Address,
            dist: &dyn Distribution<usize>,
        ) -> usize {
            let logp = dist.log_prob(&self.0);
            inner.trace.log_prior += logp;
            inner
                .trace
                .insert_choice(addr.clone(), ChoiceValue::Usize(self.0), logp);
            self.0
        }
    }

    #[test]
    fn an_override_can_answer_a_site_itself() {
        for pick in 0..3 {
            let mut rng = StdRng::seed_from_u64(31);
            let (v, t) = run(
                Delegate::with(
                    PriorHandler {
                        rng: &mut rng,
                        trace: Trace::default(),
                    },
                    Pick(pick),
                ),
                all_sites(),
            );

            // The model and the trace both see the forced value, scored under
            // the site's own distribution, and the observation of it is scored
            // at it too.
            assert_eq!(v.3, pick);
            assert_eq!(t.get_usize(&addr!("z")), Some(pick));
            let prior = [0.2_f64, 0.5, 0.3][pick].ln();
            assert!((t.choices[&addr!("z")].logp - prior).abs() < 1e-12);
            assert_every_site_recorded(&t);

            // Re-scoring the trace from scratch gives the same weights.
            let (_, rescored) = run(
                ScoreGivenTrace {
                    base: t.clone(),
                    trace: Trace::default(),
                },
                all_sites(),
            );
            assert!((rescored.log_prior - t.log_prior).abs() < 1e-12);
            assert!((rescored.log_likelihood - t.log_likelihood).abs() < 1e-12);
        }
    }

    /// Logs each `f64` sample and the end of the run under a name, then forwards.
    struct Tap<'a> {
        name: &'static str,
        log: &'a RefCell<Vec<String>>,
    }

    impl<H: Handler> Overrides<H> for Tap<'_> {
        fn on_sample_f64(
            &mut self,
            inner: &mut H,
            addr: &Address,
            dist: &dyn Distribution<f64>,
        ) -> f64 {
            self.log
                .borrow_mut()
                .push(format!("{}: sample {}", self.name, addr));
            inner.on_sample_f64(addr, dist)
        }

        fn finish(self, inner: H) -> Trace {
            self.log.borrow_mut().push(format!("{}: finish", self.name));
            inner.finish()
        }
    }

    #[test]
    fn nested_delegates_compose() {
        let log = RefCell::new(Vec::new());
        let mut seen = Vec::new();
        let mut rng = StdRng::seed_from_u64(41);
        let prior = PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        };
        // Three layers around the PriorHandler: `UsizeSites` innermost, then
        // two taps on the f64 sites.
        let handler = Delegate::with(
            Delegate::with(
                Delegate::with(prior, UsizeSites(&mut seen)),
                Tap {
                    name: "inner",
                    log: &log,
                },
            ),
            Tap {
                name: "outer",
                log: &log,
            },
        );
        let (v, t) = run(handler, all_sites());

        assert_eq!(
            log.into_inner(),
            [
                "outer: sample f",
                "inner: sample f",
                "outer: finish",
                "inner: finish"
            ]
        );
        assert_eq!(seen.len(), 2);
        let (pv, pt) = prior_run(41);
        assert_eq!(v, pv);
        assert_same_trace(&t, &pt);
    }

    /// Adds a fixed log-weight to the trace at the end of the run.
    struct Bonus(f64);

    impl<H: Handler> Overrides<H> for Bonus {
        fn finish(self, inner: H) -> Trace {
            let mut trace = inner.finish();
            trace.log_factors += self.0;
            trace
        }
    }

    #[test]
    fn a_finish_override_runs() {
        let mut rng = StdRng::seed_from_u64(51);
        let (_, t) = run(
            Delegate::with(
                PriorHandler {
                    rng: &mut rng,
                    trace: Trace::default(),
                },
                Bonus(2.0),
            ),
            all_sites(),
        );

        let (_, pt) = prior_run(51);
        assert_eq!(t.log_factors, pt.log_factors + 2.0);
        // Everything else is the inner handler's trace, untouched.
        assert_eq!(t.log_prior.to_bits(), pt.log_prior.to_bits());
        assert_eq!(t.log_likelihood.to_bits(), pt.log_likelihood.to_bits());
        assert_eq!(t.choices.len(), pt.choices.len());
    }
}
