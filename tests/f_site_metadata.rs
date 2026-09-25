//! #63: site metadata for handlers.
//!
//! A handler sees only `(addr, dist)` at a site. `Distribution::as_any` lets
//! it recognise the distribution by type (`dist.downcast_ref::<D>()`), and
//! `WithMeta<D, M>` carries metadata along with a distribution. This file pins
//! the contract end to end:
//!
//! - a handler that knows `WithMeta<Categorical, Question>` reads the question,
//!   the option names and the probabilities at the site, answers by name, and
//!   falls back to prior sampling wherever it can't answer;
//! - `PriorHandler`, `ScoreGivenTrace` and the adaptive MH chain record the same
//!   traces, bit for bit, whether the sites' distributions are wrapped or bare,
//!   at sample and observe sites of every value type;
//! - a distribution defined outside the crate that implements only the
//!   required methods is never recognised;
//! - `clone_box` of a `WithMeta` keeps the metadata downcastable.
//!
//! The unit tests in `src/core/distribution.rs` cover `as_any` on every
//! built-in distribution.

use fugue::runtime::handler::run;
use fugue::*;
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};

// -----------------------------------------------------------------------------
// A handler that reads site metadata.
// -----------------------------------------------------------------------------

/// What a decision site asks, and the names of its options in index order.
#[derive(Clone, Debug)]
struct Question {
    text: &'static str,
    options: Vec<&'static str>,
}

fn tool_question() -> Question {
    Question {
        text: "Which tool next?",
        options: vec!["search", "edit", "test"],
    }
}

/// What `AnswerByName` read at a site it answered.
#[derive(Debug, PartialEq)]
struct Answered {
    addr: Address,
    question: &'static str,
    options: Vec<&'static str>,
    probs: Vec<f64>,
}

/// Everything `AnswerByName` read during a run.
#[derive(Default)]
struct Seen {
    answered: Vec<Answered>,
    /// Address and probabilities of each bare `Categorical` it sampled.
    bare_categoricals: Vec<(Address, Vec<f64>)>,
}

/// Answers decision sites by option name.
///
/// At a `usize` site it downcasts the distribution to
/// `WithMeta<Categorical, Question>`. If that works and the question lists
/// `answer`, it notes what it read and takes that option, scoring it under the
/// site's own distribution. Everything else goes to an inner `PriorHandler`,
/// including a `usize` site whose downcast fails; at a bare `Categorical` it
/// also notes the probabilities, which it reads directly rather than probing
/// `log_prob`.
struct AnswerByName<'r> {
    answer: &'static str,
    seen: &'r mut Seen,
    prior: PriorHandler<'r, StdRng>,
}

impl<'r> AnswerByName<'r> {
    fn new(answer: &'static str, seen: &'r mut Seen, rng: &'r mut StdRng) -> Self {
        AnswerByName {
            answer,
            seen,
            prior: PriorHandler {
                rng,
                trace: Trace::default(),
            },
        }
    }
}

impl Handler for AnswerByName<'_> {
    fn on_sample_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>) -> usize {
        if let Some(site) = dist.downcast_ref::<WithMeta<Categorical, Question>>() {
            let question = site.meta();
            if let Some(choice) = question.options.iter().position(|&o| o == self.answer) {
                self.seen.answered.push(Answered {
                    addr: addr.clone(),
                    question: question.text,
                    options: question.options.clone(),
                    probs: site.dist().probs().to_vec(),
                });
                let logp = dist.log_prob(&choice);
                self.prior.trace.log_prior += logp;
                self.prior
                    .trace
                    .insert_choice(addr.clone(), ChoiceValue::Usize(choice), logp);
                return choice;
            }
        } else if let Some(categorical) = dist.downcast_ref::<Categorical>() {
            self.seen
                .bare_categoricals
                .push((addr.clone(), categorical.probs().to_vec()));
        }
        // Not a question this handler can answer: behave as the prior does.
        self.prior.on_sample_usize(addr, dist)
    }

    fn on_sample_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>) -> f64 {
        self.prior.on_sample_f64(addr, dist)
    }
    fn on_sample_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>) -> bool {
        self.prior.on_sample_bool(addr, dist)
    }
    fn on_sample_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>) -> u64 {
        self.prior.on_sample_u64(addr, dist)
    }
    fn on_sample_i64(&mut self, addr: &Address, dist: &dyn Distribution<i64>) -> i64 {
        self.prior.on_sample_i64(addr, dist)
    }
    fn on_observe_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>, value: f64) {
        self.prior.on_observe_f64(addr, dist, value)
    }
    fn on_observe_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>, value: bool) {
        self.prior.on_observe_bool(addr, dist, value)
    }
    fn on_observe_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>, value: u64) {
        self.prior.on_observe_u64(addr, dist, value)
    }
    fn on_observe_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>, value: usize) {
        self.prior.on_observe_usize(addr, dist, value)
    }
    fn on_observe_i64(&mut self, addr: &Address, dist: &dyn Distribution<i64>, value: i64) {
        self.prior.on_observe_i64(addr, dist, value)
    }
    fn on_factor(&mut self, logw: f64) {
        self.prior.on_factor(logw)
    }
    fn finish(self) -> Trace {
        self.prior.finish()
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

/// Assert two traces record the same execution, bit for bit: the same
/// addresses with the same values and per-choice log-probabilities, and the
/// same accumulated prior, likelihood and factor weights.
#[track_caller]
fn assert_same_trace(left: &Trace, right: &Trace) {
    let weights = |t: &Trace| {
        (
            t.log_prior.to_bits(),
            t.log_likelihood.to_bits(),
            t.log_factors.to_bits(),
        )
    };
    let choices = |t: &Trace| {
        t.choices
            .iter()
            .map(|(addr, c)| (addr.clone(), c.value.clone(), c.logp.to_bits()))
            .collect::<Vec<_>>()
    };
    assert_eq!(weights(left), weights(right), "accumulated weights differ");
    assert_eq!(choices(left), choices(right), "choices differ");
}

/// A decision site that carries its question, then an outcome that depends on
/// the decision.
fn decision_model() -> Model<(usize, bool)> {
    prob! {
        let tool <- sample(
            addr!("tool"),
            WithMeta::new(Categorical::new(vec![0.6, 0.3, 0.1]).unwrap(), tool_question()),
        );
        let ok <- sample(addr!("ok"), Bernoulli::new([0.9, 0.7, 0.5][tool]).unwrap());
        pure((tool, ok))
    }
}

#[test]
fn a_handler_answers_a_with_meta_site_by_name() {
    let mut rng = StdRng::seed_from_u64(63);
    let mut seen = Seen::default();
    let handler = AnswerByName::new("edit", &mut seen, &mut rng);
    let ((tool, _ok), trace) = run(handler, decision_model());

    // It read the question, the option names and the probabilities at the site...
    assert_eq!(
        seen.answered,
        vec![Answered {
            addr: addr!("tool"),
            question: "Which tool next?",
            options: vec!["search", "edit", "test"],
            probs: vec![0.6, 0.3, 0.1],
        }]
    );
    assert!(seen.bare_categoricals.is_empty());

    // ...and took the option it was asked for, scored under the site's own
    // distribution.
    assert_eq!(tool, 1);
    let choice = &trace.choices[&addr!("tool")];
    assert_eq!(choice.value, ChoiceValue::Usize(1));
    assert_eq!(choice.logp.to_bits(), 0.3f64.ln().to_bits());

    // The rest of the model ran as usual, and the result is an ordinary trace
    // of the model: ScoreGivenTrace re-scores it to the same weights.
    assert!(trace.get_bool(&addr!("ok")).is_some());
    let (_, rescored) = run(
        ScoreGivenTrace {
            base: trace.clone(),
            trace: Trace::default(),
        },
        decision_model(),
    );
    assert_same_trace(&rescored, &trace);
}

/// `usize` sites the handler can't answer: a bare `Categorical`, a `WithMeta`
/// with another metadata type, and a question that doesn't list the answer.
fn unanswerable_model() -> Model<(usize, usize, usize, f64)> {
    let yes_or_no = Question {
        text: "Continue?",
        options: vec!["yes", "no"],
    };
    prob! {
        let a <- sample(addr!("bare"), Categorical::new(vec![0.5, 0.3, 0.2]).unwrap());
        let b <- sample(
            addr!("other_meta"),
            WithMeta::new(Categorical::new(vec![0.1, 0.9]).unwrap(), "not a Question"),
        );
        let c <- sample(
            addr!("no_such_option"),
            WithMeta::new(Categorical::uniform(2).unwrap(), yes_or_no),
        );
        let x <- sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
        observe(
            addr!("picked"),
            WithMeta::new(Categorical::new(vec![0.2, 0.8]).unwrap(), tool_question()),
            1usize,
        );
        pure((a, b, c, x))
    }
}

#[test]
fn the_handler_falls_back_to_the_prior_when_it_cannot_answer() {
    for seed in 0..8 {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut seen = Seen::default();
        let handler = AnswerByName::new("edit", &mut seen, &mut rng);
        let (value, trace) = run(handler, unanswerable_model());
        let (prior_value, prior_trace) = prior_run(seed, unanswerable_model());

        // Exactly what PriorHandler does, draw for draw.
        assert_eq!(value, prior_value);
        assert_same_trace(&trace, &prior_trace);

        // Nothing was answered. Only the bare Categorical was recognised as
        // one: a `WithMeta<Categorical, _>` is not a `Categorical`.
        assert!(seen.answered.is_empty());
        assert_eq!(
            seen.bare_categoricals,
            vec![(addr!("bare"), vec![0.5, 0.3, 0.2])]
        );
    }
}

// -----------------------------------------------------------------------------
// Wrapped and bare sites behave identically under the built-in handlers.
// -----------------------------------------------------------------------------

/// Sample `dist` at `addr`, wrapped in `WithMeta` with `meta` if `wrap` is set.
fn sample_at<T, D, M>(wrap: bool, addr: Address, dist: D, meta: M) -> Model<T>
where
    T: SampleType,
    D: Distribution<T> + Clone + 'static,
    M: Clone + Send + Sync + 'static,
{
    if wrap {
        sample(addr, WithMeta::new(dist, meta))
    } else {
        sample(addr, dist)
    }
}

/// Observe `value` at `addr` under `dist`, wrapped in `WithMeta` with `meta`
/// if `wrap` is set.
fn observe_at<T, D, M>(wrap: bool, addr: Address, dist: D, meta: M, value: T) -> Model<()>
where
    T: SampleType,
    D: Distribution<T> + Clone + 'static,
    M: Clone + Send + Sync + 'static,
{
    if wrap {
        observe(addr, WithMeta::new(dist, meta), value)
    } else {
        observe(addr, dist, value)
    }
}

/// The values of `every_kind_of_site`'s sample sites.
type Draws = (f64, f64, f64, f64, bool, u64, u64, usize, i64);

/// Sample and observe sites of every value type, with real, positive and
/// bounded continuous supports. With `wrap`, each site's distribution is a
/// `WithMeta` (with assorted metadata types); without it, the bare
/// distribution.
fn every_kind_of_site(wrap: bool) -> Model<Draws> {
    prob! {
        let mu <- sample_at(wrap, addr!("mu"), Normal::new(0.0, 1.0).unwrap(), "location");
        let sigma <- sample_at(wrap, addr!("sigma"), Gamma::new(3.0, 3.0).unwrap(), 1u8);
        let p <- sample_at(wrap, addr!("p"), Beta::new(2.0, 3.0).unwrap(), tool_question());
        let u <- sample_at(wrap, addr!("u"), Uniform::new(-0.5, 0.5).unwrap(), ());
        let flag <- sample_at(wrap, addr!("flag"), Bernoulli::new(p).unwrap(), "flag");
        let count <- sample_at(wrap, addr!("count"), Poisson::new(3.0).unwrap(), vec![0.5]);
        let hits <- sample_at(wrap, addr!("hits"), Binomial::new(10, p).unwrap(), "hits");
        let tool <- sample_at(
            wrap,
            addr!("tool"),
            Categorical::new(vec![0.6, 0.3, 0.1]).unwrap(),
            tool_question(),
        );
        let die <- sample_at(wrap, addr!("die"), DiscreteUniform::new(1, 6).unwrap(), "d6");
        observe_at(wrap, addr!("y"), Normal::new(mu + u, sigma).unwrap(), "y", 0.4);
        observe_at(wrap, addr!("seen"), Bernoulli::new(p).unwrap(), "seen", true);
        observe_at(wrap, addr!("arrivals"), Poisson::new(2.5).unwrap(), "arrivals", 4u64);
        observe_at(
            wrap,
            addr!("picked"),
            Categorical::new(vec![0.2, 0.8]).unwrap(),
            tool_question(),
            1usize,
        );
        observe_at(wrap, addr!("roll"), DiscreteUniform::new(1, 6).unwrap(), "roll", 3i64);
        factor(-0.25);
        pure((mu, sigma, p, u, flag, count, hits, tool, die))
    }
}

#[test]
fn prior_handler_traces_are_identical_for_wrapped_and_bare_sites() {
    for seed in 0..16 {
        let (wrapped_value, wrapped) = prior_run(seed, every_kind_of_site(true));
        let (bare_value, bare) = prior_run(seed, every_kind_of_site(false));
        assert_eq!(wrapped_value, bare_value);
        assert_same_trace(&wrapped, &bare);
        assert_eq!(bare.choices.len(), 9);
        assert!(bare.log_prior.is_finite() && bare.log_likelihood.is_finite());
    }
}

#[test]
fn score_given_trace_is_identical_for_wrapped_and_bare_sites() {
    for seed in 0..16 {
        // A complete trace of the model, drawn from the prior.
        let (_, base) = prior_run(seed, every_kind_of_site(false));

        let score = |wrap| {
            run(
                ScoreGivenTrace {
                    base: base.clone(),
                    trace: Trace::default(),
                },
                every_kind_of_site(wrap),
            )
        };
        let (wrapped_value, wrapped) = score(true);
        let (bare_value, bare) = score(false);
        assert_eq!(wrapped_value, bare_value);
        assert_same_trace(&wrapped, &bare);
        // Both reproduce the prior run's own weights.
        assert_same_trace(&bare, &base);
    }
}

#[test]
fn adaptive_mh_chain_is_identical_for_wrapped_and_bare_sites() {
    // MH picks each f64 site's proposal from `support()`: a Gaussian walk for
    // `mu`, a log-space walk for `sigma`, and reflected walks for `p` and `u`.
    // So identical chains also pin that `WithMeta` forwards the support. (If
    // it didn't, the wrapped chain would walk `p` with a plain Gaussian, and
    // the first proposal outside [0, 1] would fail this test inside the model,
    // at `Bernoulli::new(p).unwrap()`.)
    let chain = |wrap: bool| {
        let mut rng = StdRng::seed_from_u64(63);
        adaptive_mcmc_chain(&mut rng, move || every_kind_of_site(wrap), 150, 50)
    };
    let wrapped = chain(true);
    let bare = chain(false);
    assert_eq!(wrapped.len(), bare.len());
    for ((wrapped_value, wrapped_trace), (bare_value, bare_trace)) in wrapped.iter().zip(&bare) {
        assert_eq!(wrapped_value, bare_value);
        assert_same_trace(wrapped_trace, bare_trace);
    }
    // The chains moved, so the comparison is not between two frozen chains.
    let first = bare[0].0;
    assert!(bare.iter().any(|(draws, _)| draws.0 != first.0));
    assert!(bare.iter().any(|(draws, _)| draws.1 != first.1));
    assert!(bare.iter().any(|(draws, _)| draws.2 != first.2));
}

// -----------------------------------------------------------------------------
// Foreign distributions and clone_box.
// -----------------------------------------------------------------------------

/// A distribution defined outside the crate that implements only the required
/// methods, so it keeps the default `as_any`.
#[derive(Clone)]
struct ThreeSided;

impl Distribution<usize> for ThreeSided {
    fn sample(&self, rng: &mut dyn RngCore) -> usize {
        (rng.next_u32() % 3) as usize
    }
    fn log_prob(&self, x: &usize) -> f64 {
        if *x < 3 {
            -(3f64.ln())
        } else {
            f64::NEG_INFINITY
        }
    }
    fn clone_box(&self) -> Box<dyn Distribution<usize>> {
        Box::new(self.clone())
    }
}

#[test]
fn a_foreign_distribution_keeps_the_default_and_is_not_recognised() {
    let dist: &dyn Distribution<usize> = &ThreeSided;
    assert!(dist.as_any().is_none());
    assert!(dist.downcast_ref::<ThreeSided>().is_none());
    assert!(dist.downcast_ref::<Categorical>().is_none());

    // Wrapping it opts the site in: the wrapper is recognised, even though
    // the distribution inside it isn't.
    let site = WithMeta::new(ThreeSided, tool_question());
    let dist: &dyn Distribution<usize> = &site;
    let found = dist
        .downcast_ref::<WithMeta<ThreeSided, Question>>()
        .expect("WithMeta returns Some from as_any");
    assert_eq!(found.meta().text, "Which tool next?");
    assert!(found.dist().as_any().is_none());

    // A handler looking for `WithMeta<Categorical, Question>` matches neither
    // site and samples both from the prior.
    let model = || {
        sample(addr!("plain"), ThreeSided).bind(|a| {
            sample(addr!("wrapped"), WithMeta::new(ThreeSided, tool_question()))
                .map(move |b| (a, b))
        })
    };
    let mut rng = StdRng::seed_from_u64(3);
    let mut seen = Seen::default();
    let (value, trace) = run(AnswerByName::new("edit", &mut seen, &mut rng), model());
    let (prior_value, prior_trace) = prior_run(3, model());
    assert_eq!(value, prior_value);
    assert_same_trace(&trace, &prior_trace);
    assert!(seen.answered.is_empty() && seen.bare_categoricals.is_empty());
}

#[test]
fn clone_box_keeps_the_metadata_downcastable() {
    let site = WithMeta::new(
        Categorical::new(vec![0.6, 0.3, 0.1]).unwrap(),
        tool_question(),
    );
    let boxed: Box<dyn Distribution<usize>> = site.clone_box();
    let copy = boxed
        .downcast_ref::<WithMeta<Categorical, Question>>()
        .expect("clone_box keeps the wrapper");
    assert_eq!(copy.meta().options, vec!["search", "edit", "test"]);
    assert_eq!(copy.dist().probs(), &[0.6, 0.3, 0.1]);

    // A clone of the clone, too, and it still scores as the categorical.
    let again = boxed.clone_box();
    assert_eq!(
        again
            .downcast_ref::<WithMeta<Categorical, Question>>()
            .map(|w| w.meta().text),
        Some("Which tool next?")
    );
    assert_eq!(
        again.log_prob(&1).to_bits(),
        site.dist().log_prob(&1).to_bits()
    );
}
