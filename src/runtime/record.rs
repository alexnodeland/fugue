#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/docs/runtime/record.md"))]

use crate::core::address::Address;
use crate::core::distribution::Distribution;
use crate::core::model::Model;
use crate::core::numerical::nan_to_neg_inf;
use crate::runtime::handler::{run, Handler};
use crate::runtime::trace::Trace;
use std::collections::BTreeMap;

/// Log-weights of one model execution, by site.
///
/// Filled by [`RecordSites`] (or by a handler of your own through the
/// `record_*` methods). It holds one log-weight per observation address, one
/// per addressed factor ([`factor_at`](crate::core::model::factor_at)), and
/// the sum of the anonymous factors ([`factor`](crate::core::model::factor)),
/// which have no address to file them under. Weights recorded at the same
/// address add up.
///
/// The totals reproduce a trace's flat accumulators:
/// [`log_likelihood`](Self::log_likelihood) is `Trace::log_likelihood` and
/// [`log_factors`](Self::log_factors) is `Trace::log_factors`, up to the
/// rounding of summing in a different order.
///
/// Example:
/// ```rust
/// # use fugue::*;
/// # use fugue::runtime::record::SiteWeights;
/// let mut sites = SiteWeights::new();
/// sites.record_observation(&Address::new("m/a/x/0"), -0.5);
/// sites.record_observation(&Address::new("m/a/x/1"), -1.5);
/// sites.record_observation(&Address::new("m/b/x/0"), -3.0);
/// sites.record_factor(&Address::new("m/a/penalty"), -0.25);
/// sites.record_unaddressed_factor(-1.0);
///
/// assert_eq!(sites.observation(&Address::new("m/a/x/1")), Some(-1.5));
/// assert_eq!(sites.log_likelihood(), -5.0);
/// assert_eq!(sites.log_factors(), -1.25);
/// assert_eq!(sites.total(), -6.25);
///
/// // Attribution by address prefix (the `Address::has_prefix` rule).
/// assert_eq!(sites.observations_under("m/a"), -2.0);
/// assert_eq!(sites.factors_under("m/a"), -0.25);
/// assert_eq!(sites.under("m/a"), -2.25);
/// ```
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SiteWeights {
    observations: BTreeMap<Address, f64>,
    factors: BTreeMap<Address, f64>,
    unaddressed_factors: f64,
}

impl SiteWeights {
    /// An empty record.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add `logw` to the observation record at `addr`. `NaN` counts as `-∞`,
    /// the rule the interpreters apply when they accumulate (FG-N2).
    pub fn record_observation(&mut self, addr: &Address, logw: f64) {
        *self.observations.entry(addr.clone()).or_insert(0.0) += nan_to_neg_inf(logw);
    }

    /// Add `logw` to the addressed-factor record at `addr`. `NaN` counts as `-∞`.
    pub fn record_factor(&mut self, addr: &Address, logw: f64) {
        *self.factors.entry(addr.clone()).or_insert(0.0) += nan_to_neg_inf(logw);
    }

    /// Add `logw` to the sum of anonymous factors. `NaN` counts as `-∞`.
    pub fn record_unaddressed_factor(&mut self, logw: f64) {
        self.unaddressed_factors += nan_to_neg_inf(logw);
    }

    /// Log-likelihood recorded per observation address.
    pub fn observations(&self) -> &BTreeMap<Address, f64> {
        &self.observations
    }

    /// Log-weight recorded per addressed factor.
    pub fn factors(&self) -> &BTreeMap<Address, f64> {
        &self.factors
    }

    /// The log-likelihood recorded at `addr`, or `None` if no observation was
    /// recorded there.
    pub fn observation(&self, addr: &Address) -> Option<f64> {
        self.observations.get(addr).copied()
    }

    /// The log-weight recorded for the addressed factor at `addr`, or `None`
    /// if there was none.
    pub fn factor(&self, addr: &Address) -> Option<f64> {
        self.factors.get(addr).copied()
    }

    /// Sum of the log-weights of the anonymous factors (`0.0` if there were none).
    pub fn unaddressed_factors(&self) -> f64 {
        self.unaddressed_factors
    }

    /// Sum over all observation records: the trace's `log_likelihood`.
    pub fn log_likelihood(&self) -> f64 {
        self.observations.values().sum()
    }

    /// Sum over all factor records, addressed and anonymous: the trace's
    /// `log_factors`.
    pub fn log_factors(&self) -> f64 {
        self.factors.values().sum::<f64>() + self.unaddressed_factors
    }

    /// [`log_likelihood`](Self::log_likelihood) plus
    /// [`log_factors`](Self::log_factors): everything in the trace's total
    /// log-weight except the prior.
    pub fn total(&self) -> f64 {
        self.log_likelihood() + self.log_factors()
    }

    /// Sum of the observation records whose address is under `prefix` (per
    /// [`Address::has_prefix`]); `0.0` if there are none.
    pub fn observations_under(&self, prefix: &str) -> f64 {
        sum_under(&self.observations, prefix)
    }

    /// Sum of the addressed-factor records whose address is under `prefix`
    /// (per [`Address::has_prefix`]); `0.0` if there are none. Anonymous
    /// factors have no address and are never included.
    pub fn factors_under(&self, prefix: &str) -> f64 {
        sum_under(&self.factors, prefix)
    }

    /// [`observations_under`](Self::observations_under) plus
    /// [`factors_under`](Self::factors_under).
    pub fn under(&self, prefix: &str) -> f64 {
        self.observations_under(prefix) + self.factors_under(prefix)
    }

    /// True if nothing has been recorded.
    pub fn is_empty(&self) -> bool {
        self.observations.is_empty() && self.factors.is_empty() && self.unaddressed_factors == 0.0
    }

    /// Forget everything, so the record can be reused for another run.
    pub fn clear(&mut self) {
        self.observations.clear();
        self.factors.clear();
        self.unaddressed_factors = 0.0;
    }
}

fn sum_under(map: &BTreeMap<Address, f64>, prefix: &str) -> f64 {
    // As in `Trace::prefix_addresses`: keys sharing a raw string prefix are
    // contiguous, and `has_prefix` filters that range.
    map.range(Address::new(prefix.to_string())..)
        .take_while(|(a, _)| a.as_str().starts_with(prefix))
        .filter(|(a, _)| a.has_prefix(prefix))
        .map(|(_, w)| *w)
        .sum()
}

/// A [`Handler`] that records per-site observation and factor log-weights
/// into a [`SiteWeights`] and forwards every site to an inner handler.
///
/// The inner handler sees exactly the calls it would see without the wrapper,
/// so its trace, its return values and its RNG draws are unchanged; recording
/// is a side channel. At an observation the record is
/// `dist.log_prob(&value)`, at a factor it is `logw` (both with `NaN` as
/// `-∞`), which is what the built-in interpreters accumulate. See the
/// [module documentation](crate::runtime::record) for the exact agreement
/// with the trace's accumulators.
///
/// The record is borrowed, so it outlives the handler, which
/// [`run`] consumes. [`run_recorded`] wraps the
/// borrow for the common case.
///
/// Example:
/// ```rust
/// # use fugue::*;
/// # use fugue::runtime::record::{RecordSites, SiteWeights};
/// # use rand::rngs::StdRng;
/// # use rand::SeedableRng;
/// let model = sample(addr!("p"), Beta::new(2.0, 2.0).unwrap()).bind(|p| {
///     observe(addr!("flip", 0), Bernoulli::new(p).unwrap(), true)
///         .bind(move |_| observe(addr!("flip", 1), Bernoulli::new(p).unwrap(), false))
///         .map(move |_| p)
/// });
///
/// let mut sites = SiteWeights::new();
/// let mut rng = StdRng::seed_from_u64(5);
/// let (p, trace) = runtime::handler::run(
///     RecordSites {
///         inner: PriorHandler { rng: &mut rng, trace: Trace::default() },
///         sites: &mut sites,
///     },
///     model,
/// );
///
/// assert_eq!(sites.observation(&addr!("flip", 0)), Some(p.ln()));
/// assert_eq!(sites.observation(&addr!("flip", 1)), Some((1.0 - p).ln()));
/// assert!((sites.log_likelihood() - trace.log_likelihood).abs() < 1e-12);
/// ```
pub struct RecordSites<'s, H> {
    /// The handler that executes the model.
    pub inner: H,
    /// Where the per-site log-weights are recorded.
    pub sites: &'s mut SiteWeights,
}

macro_rules! impl_recording_methods {
    ($(($sample:ident, $observe:ident, $ty:ty)),* $(,)?) => {
        $(
            fn $sample(&mut self, addr: &Address, dist: &dyn Distribution<$ty>) -> $ty {
                self.inner.$sample(addr, dist)
            }

            fn $observe(&mut self, addr: &Address, dist: &dyn Distribution<$ty>, value: $ty) {
                self.sites.record_observation(addr, dist.log_prob(&value));
                self.inner.$observe(addr, dist, value)
            }
        )*
    };
}

impl<H: Handler> Handler for RecordSites<'_, H> {
    impl_recording_methods! {
        (on_sample_f64,   on_observe_f64,   f64),
        (on_sample_bool,  on_observe_bool,  bool),
        (on_sample_u64,   on_observe_u64,   u64),
        (on_sample_usize, on_observe_usize, usize),
        (on_sample_i64,   on_observe_i64,   i64),
    }

    fn on_factor(&mut self, logw: f64) {
        self.sites.record_unaddressed_factor(logw);
        self.inner.on_factor(logw)
    }

    fn on_factor_at(&mut self, addr: &Address, logw: f64) {
        self.sites.record_factor(addr, logw);
        self.inner.on_factor_at(addr, logw)
    }

    fn finish(self) -> Trace {
        self.inner.finish()
    }
}

/// Run `model` under `handler` and return its value, the handler's trace, and
/// the per-site record of observation and factor log-weights.
///
/// Equivalent to [`run`] with `handler` wrapped
/// in a [`RecordSites`]: the value and the trace are exactly what
/// `run(handler, model)` returns.
///
/// Example:
/// ```rust
/// # use fugue::*;
/// # use fugue::runtime::record::run_recorded;
/// // Score a known trace and see where its likelihood comes from.
/// let mut base = Trace::default();
/// base.insert_choice(addr!("mu"), ChoiceValue::F64(1.0), 0.0);
///
/// let model = sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap()).bind(|mu| {
///     observe(addr!("near"), Normal::new(mu, 1.0).unwrap(), 1.0)
///         .bind(move |_| observe(addr!("far"), Normal::new(mu, 1.0).unwrap(), 4.0))
/// });
/// let (_, trace, sites) = run_recorded(
///     ScoreGivenTrace { base, trace: Trace::default() },
///     model,
/// );
///
/// let near = sites.observation(&addr!("near")).unwrap();
/// let far = sites.observation(&addr!("far")).unwrap();
/// assert!((near - far - 4.5).abs() < 1e-12); // 3 sigma further: 3^2 / 2
/// assert!((near + far - trace.log_likelihood).abs() < 1e-12);
/// ```
pub fn run_recorded<A>(handler: impl Handler, model: Model<A>) -> (A, Trace, SiteWeights) {
    let mut sites = SiteWeights::new();
    let (a, trace) = run(
        RecordSites {
            inner: handler,
            sites: &mut sites,
        },
        model,
    );
    (a, trace, sites)
}
