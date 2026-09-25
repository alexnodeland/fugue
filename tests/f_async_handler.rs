//! Async interpretation (#62): `run_async` under a real executor.
//!
//! `RemoteOracle` stands in for a handler whose sites are resolved over the
//! network (a tool call, an LLM): every sample and observe site waits out a
//! timer, and each sample site takes its value from a task spawned on the
//! runtime, which the multi-threaded scheduler may run on another worker.
//!
//! The first test spawns runs with `tokio::spawn`, which requires the future to
//! be `Send + 'static`: this file compiling is the check that `run_async` over a
//! concrete handler whose futures are `Send` is itself `Send`, the claim in
//! `AsyncHandler`'s documentation. The second runs a handler that is not `Send`
//! on a current-thread runtime, which the trait deliberately allows.

use fugue::runtime::handler::run;
use fugue::*;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Latency of each stand-in remote call.
const LATENCY: Duration = Duration::from_millis(1);

type AllTypes = (f64, bool, u64, usize, i64);

/// Samples and observes each of the five value types and ends with a factor.
fn all_effects_model() -> Model<AllTypes> {
    prob! {
        let x <- sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
        let b <- sample(addr!("b"), Bernoulli::new(0.3).unwrap());
        let n <- sample(addr!("n"), Poisson::new(4.0).unwrap());
        let c <- sample(addr!("c"), Categorical::new(vec![0.2, 0.5, 0.3]).unwrap());
        let z <- sample(addr!("z"), DiscreteUniform::new(-3, 3).unwrap());
        observe(addr!("obs_f64"), Normal::new(x, 0.5).unwrap(), 0.4);
        observe(addr!("obs_bool"), Bernoulli::new(if b { 0.9 } else { 0.2 }).unwrap(), true);
        observe(addr!("obs_u64"), Poisson::new(n as f64 + 1.0).unwrap(), 3u64);
        observe(addr!("obs_usize"), Categorical::uniform(3).unwrap(), c);
        observe(addr!("obs_i64"), DiscreteUniform::new(z - 1, z + 1).unwrap(), z);
        factor(-0.5 * x * x);
        pure((x, b, n, c, z))
    }
}

/// The sites of `all_effects_model`, in program order.
fn program_order() -> Vec<Address> {
    ["x", "b", "n", "c", "z"]
        .into_iter()
        .chain(["obs_f64", "obs_bool", "obs_u64", "obs_usize", "obs_i64"])
        .map(|name| addr!(name))
        .collect()
}

/// The oracle's answers for run `j`, as a trace of sample-site values.
fn answer_sheet(j: usize) -> Trace {
    let mut answers = Trace::default();
    let values = [
        ("x", ChoiceValue::F64(0.25 * j as f64 - 1.0)),
        ("b", ChoiceValue::Bool(j.is_multiple_of(2))),
        ("n", ChoiceValue::U64(j as u64)),
        ("c", ChoiceValue::Usize(j % 3)),
        ("z", ChoiceValue::I64((j % 7) as i64 - 3)),
    ];
    for (name, value) in values {
        answers.insert_choice(addr!(name), value, 0.0);
    }
    answers
}

/// Where the handler records each site it is asked about, shared with the
/// test. `Arc<Mutex<_>>` keeps the handler `Send`; `Rc<RefCell<_>>` does not.
trait VisitLog {
    fn push(&self, addr: &Address);
    fn sites(&self) -> Vec<Address>;
}

impl VisitLog for Arc<Mutex<Vec<Address>>> {
    fn push(&self, addr: &Address) {
        self.lock().unwrap().push(addr.clone());
    }
    fn sites(&self) -> Vec<Address> {
        self.lock().unwrap().clone()
    }
}

impl VisitLog for Rc<RefCell<Vec<Address>>> {
    fn push(&self, addr: &Address) {
        self.borrow_mut().push(addr.clone());
    }
    fn sites(&self) -> Vec<Address> {
        self.borrow().clone()
    }
}

/// Resolves each sample site with the answer from its sheet, received after a
/// timer from another task, and scores it under the site's prior. Each
/// observation also waits before it is scored. Values and scores follow
/// `ScoreGivenTrace` over the answer sheet, so that is the expected trace.
struct RemoteOracle<L> {
    answers: Trace,
    visits: L,
    trace: Trace,
}

impl<L: VisitLog> RemoteOracle<L> {
    fn new(answers: Trace, visits: L) -> Self {
        RemoteOracle {
            answers,
            visits,
            trace: Trace::default(),
        }
    }

    /// One round trip: wait out the latency, then receive the answer from a
    /// spawned task.
    async fn ask(&mut self, addr: &Address) -> ChoiceValue {
        self.visits.push(addr);
        tokio::time::sleep(LATENCY).await;
        let answer = self.answers.choices[addr].value.clone();
        tokio::spawn(async move { answer })
            .await
            .expect("oracle task")
    }

    /// Record a sampled value. Callers compute `logp` after the `.await`, from
    /// the `dist` borrowed across it.
    fn record(&mut self, addr: &Address, value: ChoiceValue, logp: f64) {
        self.trace.log_prior += logp;
        self.trace.insert_choice(addr.clone(), value, logp);
    }

    /// An observation is reported remotely before it is scored.
    async fn report(&mut self, addr: &Address) {
        self.visits.push(addr);
        tokio::time::sleep(LATENCY).await;
    }
}

impl<L: VisitLog> AsyncHandler for RemoteOracle<L> {
    async fn on_sample_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>) -> f64 {
        let x = self.ask(addr).await.as_f64().expect("f64 answer");
        self.record(addr, ChoiceValue::F64(x), dist.log_prob(&x));
        x
    }

    async fn on_sample_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>) -> bool {
        let x = self.ask(addr).await.as_bool().expect("bool answer");
        self.record(addr, ChoiceValue::Bool(x), dist.log_prob(&x));
        x
    }

    async fn on_sample_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>) -> u64 {
        let x = self.ask(addr).await.as_u64().expect("u64 answer");
        self.record(addr, ChoiceValue::U64(x), dist.log_prob(&x));
        x
    }

    async fn on_sample_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>) -> usize {
        let x = self.ask(addr).await.as_usize().expect("usize answer");
        self.record(addr, ChoiceValue::Usize(x), dist.log_prob(&x));
        x
    }

    async fn on_sample_i64(&mut self, addr: &Address, dist: &dyn Distribution<i64>) -> i64 {
        let x = self.ask(addr).await.as_i64().expect("i64 answer");
        self.record(addr, ChoiceValue::I64(x), dist.log_prob(&x));
        x
    }

    async fn on_observe_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>, value: f64) {
        self.report(addr).await;
        self.trace.log_likelihood += dist.log_prob(&value);
    }

    async fn on_observe_bool(
        &mut self,
        addr: &Address,
        dist: &dyn Distribution<bool>,
        value: bool,
    ) {
        self.report(addr).await;
        self.trace.log_likelihood += dist.log_prob(&value);
    }

    async fn on_observe_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>, value: u64) {
        self.report(addr).await;
        self.trace.log_likelihood += dist.log_prob(&value);
    }

    async fn on_observe_usize(
        &mut self,
        addr: &Address,
        dist: &dyn Distribution<usize>,
        value: usize,
    ) {
        self.report(addr).await;
        self.trace.log_likelihood += dist.log_prob(&value);
    }

    async fn on_observe_i64(&mut self, addr: &Address, dist: &dyn Distribution<i64>, value: i64) {
        self.report(addr).await;
        self.trace.log_likelihood += dist.log_prob(&value);
    }

    async fn on_factor(&mut self, logw: f64) {
        self.trace.log_factors += logw;
    }

    async fn finish(self) -> Trace {
        self.trace
    }
}

/// The run must return the oracle's answers, leave exactly the trace
/// `ScoreGivenTrace` computes over the answer sheet (bit for bit), and have
/// visited every site once, in program order.
fn check_run(j: usize, (value, trace): &(AllTypes, Trace), visits: Vec<Address>) {
    let answers = answer_sheet(j);
    let expected_value = (
        answers.get_f64(&addr!("x")).unwrap(),
        answers.get_bool(&addr!("b")).unwrap(),
        answers.get_u64(&addr!("n")).unwrap(),
        answers.get_usize(&addr!("c")).unwrap(),
        answers.get_i64(&addr!("z")).unwrap(),
    );
    assert_eq!(*value, expected_value, "run {j}: value");

    let (_, expected) = run(
        ScoreGivenTrace {
            base: answers,
            trace: Trace::default(),
        },
        all_effects_model(),
    );
    assert_eq!(trace.choices.len(), expected.choices.len(), "run {j}");
    for (addr, choice) in &expected.choices {
        let got = &trace.choices[addr];
        assert_eq!(got.value, choice.value, "run {j}: value at {addr}");
        assert_eq!(
            got.logp.to_bits(),
            choice.logp.to_bits(),
            "run {j}: logp at {addr}"
        );
    }
    assert_eq!(trace.log_prior.to_bits(), expected.log_prior.to_bits());
    assert_eq!(
        trace.log_likelihood.to_bits(),
        expected.log_likelihood.to_bits()
    );
    assert_eq!(trace.log_factors.to_bits(), expected.log_factors.to_bits());
    assert!(trace.total_log_weight().is_finite(), "run {j}");

    assert_eq!(visits, program_order(), "run {j}: sites awaited in order");
}

// #62: a handler that awaits a timer at every site and takes its sample values
// from other tasks, run concurrently eight times, each run spawned on tokio's
// multi-threaded runtime. Each run is sequential and exact; the concurrency is
// across runs.
#[test]
fn send_handler_runs_spawned_on_a_multi_threaded_runtime() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_time()
        .build()
        .expect("tokio runtime");

    let runs: Vec<_> = (0..8)
        .map(|j| {
            let visits = Arc::new(Mutex::new(Vec::new()));
            let handler = RemoteOracle::new(answer_sheet(j), Arc::clone(&visits));
            // `spawn` takes only `Send + 'static` futures: this line compiles
            // because `RemoteOracle<Arc<Mutex<_>>>`'s futures are `Send`.
            let task = rt.spawn(run_async(handler, all_effects_model()));
            (task, visits)
        })
        .collect();

    rt.block_on(async {
        for (j, (task, visits)) in runs.into_iter().enumerate() {
            let result = task.await.expect("run panicked");
            check_run(j, &result, visits.sites());
        }
    });
}

// #62: the trait does not require `Send`. A handler holding an `Rc` is not
// `Send`, so its run cannot be spawned on a multi-threaded runtime, but it runs
// on a current-thread one, with the same result.
#[test]
fn non_send_handler_runs_on_a_current_thread_runtime() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .expect("tokio runtime");

    let j = 3;
    let visits = Rc::new(RefCell::new(Vec::new()));
    let handler = RemoteOracle::new(answer_sheet(j), Rc::clone(&visits));
    let result = rt.block_on(run_async(handler, all_effects_model()));
    check_run(j, &result, visits.sites());
}
