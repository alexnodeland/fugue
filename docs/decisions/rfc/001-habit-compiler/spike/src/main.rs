//! Spike: an agent "flow" as a fugue program, interpreted three ways.
//!
//! - simulate: `PriorHandler`: decisions from the learned world-model policy,
//!   outcomes from the learned dynamics ("dreaming").
//! - execute:  `HarnessHandler`: decisions from a Jev-like calibrated oracle
//!   pooled with the world-model prior (escalating to an "LLM" when unsure),
//!   outcomes from the real environment, scored as observations.
//! - audit:    `ScoreGivenTrace` / `score_given_trace_reconciled`: score a
//!   recorded agent episode against the flow (conformance, surprise, and
//!   off-policy evaluation from logged propensities).
//!
//! The environment, the "LLM agent" and the "Jev" oracle are all mocks. The
//! numbers this prints are about API feasibility, not evidence of real-world
//! benefit.

use std::collections::HashMap;
use std::sync::Arc;

use fugue::runtime::handler::run;
use fugue::runtime::interpreters::{score_given_trace_reconciled, PriorHandler, ScoreGivenTrace};
use fugue::*;
use rand::rngs::StdRng;
use rand::{Rng, RngCore, SeedableRng};

// ---------------------------------------------------------------------------
// Action space: the tools the agent is allowed to call.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Tool {
    ReadLogs,
    RunTests,
    EditFile,
    Revert,
    Finish,
}
const TOOLS: [Tool; 5] = [
    Tool::ReadLogs,
    Tool::RunTests,
    Tool::EditFile,
    Tool::Revert,
    Tool::Finish,
];
const K: usize = TOOLS.len();

fn idx(t: Tool) -> usize {
    TOOLS.iter().position(|&x| x == t).unwrap()
}

/// The abstract state the world model conditions on: last tool and whether it
/// "worked". Deliberately coarse; the real environment has hidden state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Ctx {
    Start,
    After(Tool, bool),
}

// ---------------------------------------------------------------------------
// World model: Dirichlet-categorical policy and Beta-Bernoulli outcomes, with
// the action support restricted to the tools currently exposed.
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct WorldModel {
    allowed: [bool; K],
    alpha: f64,
    next: HashMap<Ctx, [f64; K]>,
    ok: HashMap<(Ctx, Tool), (f64, f64)>,
}

impl WorldModel {
    fn new(allowed: [bool; K]) -> Self {
        Self {
            allowed,
            alpha: 0.5,
            next: HashMap::new(),
            ok: HashMap::new(),
        }
    }

    fn learn(&mut self, episode: &[(Tool, bool)]) {
        let mut ctx = Ctx::Start;
        for &(tool, ok) in episode {
            self.next.entry(ctx).or_insert([0.0; K])[idx(tool)] += 1.0;
            if tool == Tool::Finish {
                break;
            }
            let e = self.ok.entry((ctx, tool)).or_insert((0.0, 0.0));
            if ok {
                e.0 += 1.0
            } else {
                e.1 += 1.0
            }
            ctx = Ctx::After(tool, ok);
        }
    }

    /// Posterior predictive over the next tool; zero mass on unexposed tools.
    fn policy(&self, ctx: Ctx) -> Vec<f64> {
        let counts = self.next.get(&ctx).copied().unwrap_or([0.0; K]);
        let w: Vec<f64> = (0..K)
            .map(|i| {
                if self.allowed[i] {
                    counts[i] + self.alpha
                } else {
                    0.0
                }
            })
            .collect();
        let z: f64 = w.iter().sum();
        w.iter().map(|x| x / z).collect()
    }

    /// Posterior predictive P(ok | ctx, tool) under a Beta(1, 1) prior.
    fn p_ok(&self, ctx: Ctx, tool: Tool) -> f64 {
        let (a, b) = self.ok.get(&(ctx, tool)).copied().unwrap_or((0.0, 0.0));
        (a + 1.0) / (a + b + 2.0)
    }
}

// ---------------------------------------------------------------------------
// The flow. `decide#i` sites are tool choices, `outcome#i` sites are tool
// results. The program does not say who decides or where outcomes come from;
// that is the handler's job.
// ---------------------------------------------------------------------------

type Episode = Vec<(Tool, bool)>;

fn flow(wm: Arc<WorldModel>, max_steps: usize) -> Model<Episode> {
    flow_with(wm, max_steps, false)
}

/// `greedy` compiles each decision to a point mass on the world model's top
/// choice: the deterministic "habit" as a program of its own.
fn flow_with(wm: Arc<WorldModel>, max_steps: usize, greedy: bool) -> Model<Episode> {
    step(wm, 0, max_steps, Ctx::Start, Vec::new(), greedy)
}

fn one_hot_argmax(p: &[f64]) -> Vec<f64> {
    let best = argmax(p);
    (0..p.len())
        .map(|i| if i == best { 1.0 } else { 0.0 })
        .collect()
}

fn argmax(p: &[f64]) -> usize {
    (0..p.len()).fold(0, |b, i| if p[i] > p[b] { i } else { b })
}

fn step(
    wm: Arc<WorldModel>,
    i: usize,
    max_steps: usize,
    ctx: Ctx,
    hist: Episode,
    greedy: bool,
) -> Model<Episode> {
    if i == max_steps {
        return pure(hist);
    }
    let p = wm.policy(ctx);
    let policy = Categorical::new(if greedy { one_hot_argmax(&p) } else { p }).unwrap();
    sample(addr!("decide", i), policy).bind(move |a| {
        let mut hist = hist;
        let tool = TOOLS[a];
        if tool == Tool::Finish {
            hist.push((tool, true));
            return pure(hist);
        }
        let p = wm.p_ok(ctx, tool);
        sample(addr!("outcome", i), Bernoulli::new(p).unwrap()).bind(move |ok| {
            let mut hist = hist;
            hist.push((tool, ok));
            step(wm, i + 1, max_steps, Ctx::After(tool, ok), hist, greedy)
        })
    })
}

/// Success as the world model can see it: finished, and the last thing that
/// happened before finishing was a passing test run.
fn believed_success(ep: &[(Tool, bool)]) -> bool {
    if ep.last().map(|s| s.0) != Some(Tool::Finish) {
        return false;
    }
    for &(t, ok) in ep.iter().rev().skip(1) {
        match t {
            Tool::RunTests => return ok,
            Tool::EditFile | Tool::Revert => return false,
            _ => {}
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Mocks: environment, LLM agent, and a calibrated System-One oracle.
// ---------------------------------------------------------------------------

#[derive(Clone, Default)]
struct Env {
    located: bool,
    fixed: bool,
    edited_since_test: bool,
    verified: bool,
}

impl Env {
    fn call(&mut self, tool: Tool, rng: &mut dyn RngCore) -> bool {
        match tool {
            Tool::ReadLogs => {
                if rng.gen_bool(0.9) {
                    self.located = true;
                }
                true
            }
            Tool::EditFile => {
                self.edited_since_test = true;
                self.verified = false;
                let p = if self.located { 0.8 } else { 0.1 };
                if rng.gen_bool(p) {
                    self.fixed = true;
                }
                true
            }
            Tool::RunTests => {
                self.edited_since_test = false;
                let pass = self.fixed && rng.gen_bool(0.95);
                self.verified = pass;
                pass
            }
            Tool::Revert => {
                self.fixed = false;
                self.verified = false;
                true
            }
            Tool::Finish => true,
        }
    }

    fn success(&self) -> bool {
        self.fixed && self.verified
    }

    /// What a competent agent would do next; ground truth for the mocks.
    fn best_next(&self) -> Tool {
        if self.verified {
            Tool::Finish
        } else if self.edited_since_test {
            Tool::RunTests
        } else if !self.located {
            Tool::ReadLogs
        } else {
            Tool::EditFile
        }
    }
}

fn random_allowed(allowed: &[bool; K], rng: &mut dyn RngCore) -> Tool {
    let opts: Vec<Tool> = (0..K).filter(|&i| allowed[i]).map(|i| TOOLS[i]).collect();
    opts[rng.gen_range(0..opts.len())]
}

/// The mock LLM: with probability `skill` it takes the competent next step,
/// otherwise a random exposed tool. It never calls a tool the manifest does
/// not expose; when the competent step is unexposed it picks at random.
fn llm_choice(env: &Env, allowed: &[bool; K], skill: f64, rng: &mut dyn RngCore) -> Tool {
    if rng.gen_bool(skill) {
        let best = env.best_next();
        if allowed[idx(best)] {
            best
        } else {
            random_allowed(allowed, rng)
        }
    } else {
        random_allowed(allowed, rng)
    }
}

/// The probability that [`llm_choice`] returns `tool`: the behavior-policy
/// propensity logged for an escalated decision.
fn llm_prob(env: &Env, allowed: &[bool; K], skill: f64, tool: Tool) -> f64 {
    if !allowed[idx(tool)] {
        return 0.0;
    }
    let n = allowed.iter().filter(|&&a| a).count() as f64;
    let best = env.best_next();
    let skilled = if !allowed[idx(best)] {
        1.0 / n
    } else if tool == best {
        1.0
    } else {
        0.0
    };
    skill * skilled + (1.0 - skill) / n
}

fn llm_episode(
    rng: &mut dyn RngCore,
    skill: f64,
    max_steps: usize,
    allowed: &[bool; K],
) -> (Episode, bool) {
    let mut env = Env::default();
    let mut ep = Vec::new();
    for _ in 0..max_steps {
        let t = llm_choice(&env, allowed, skill, rng);
        let ok = env.call(t, rng);
        ep.push((t, ok));
        if t == Tool::Finish {
            break;
        }
    }
    (ep, env.success())
}

/// Stand-in for a System-One `Choice` over the exposed tools: the top option
/// is right with probability equal to its stated probability (calibrated by
/// construction), and the rest of the mass is spread over the other options.
fn oracle_choice(env: &Env, allowed: &[bool; K], rng: &mut dyn RngCore) -> Vec<f64> {
    let conf: f64 = rng.gen_range(0.35..0.99);
    let truth = env.best_next();
    let top = if rng.gen_bool(conf) {
        truth
    } else {
        loop {
            let t = random_allowed(allowed, rng);
            if t != truth {
                break t;
            }
        }
    };
    let n = allowed.iter().filter(|&&a| a).count() as f64;
    let rest = (1.0 - conf) / (n - 1.0);
    (0..K)
        .map(|i| {
            if !allowed[i] {
                0.0
            } else if TOOLS[i] == top {
                conf
            } else {
                rest
            }
        })
        .collect()
}

/// Log-linear pooling of the world-model prior with the oracle's answer.
fn pool(prior: &[f64], oracle: &[f64], w_prior: f64) -> Vec<f64> {
    let logits: Vec<f64> = prior
        .iter()
        .zip(oracle)
        .map(|(&p, &q)| {
            if p == 0.0 || q == 0.0 {
                f64::NEG_INFINITY
            } else {
                w_prior * p.ln() + q.ln()
            }
        })
        .collect();
    let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let w: Vec<f64> = logits.iter().map(|l| (l - m).exp()).collect();
    let z: f64 = w.iter().sum();
    w.iter().map(|x| x / z).collect()
}

// ---------------------------------------------------------------------------
// The harness as a fugue handler.
// ---------------------------------------------------------------------------

#[derive(Default, Debug)]
struct Stats {
    oracle_calls: usize,
    escalations: usize,
    tool_calls: usize,
    surprise: f64,
}

struct HarnessHandler<'a> {
    rng: &'a mut StdRng,
    env: &'a mut Env,
    stats: &'a mut Stats,
    allowed: [bool; K],
    use_oracle: bool,
    prior_weight: f64,
    escalate_below: f64,
    greedy: bool,
    consult_below: f64,
    llm_skill: f64,
    last_tool: Option<Tool>,
    trace: Trace,
}

impl Handler for HarnessHandler<'_> {
    /// A decision site: pool the world-model prior (the site's own
    /// distribution) with the oracle, and escalate when the result is flat.
    fn on_sample_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>) -> usize {
        let prior: Vec<f64> = (0..K).map(|i| dist.log_prob(&i).exp()).collect();
        let habit_conf = prior.iter().cloned().fold(0.0, f64::max);
        let probs = if self.use_oracle && habit_conf < self.consult_below {
            self.stats.oracle_calls += 1;
            let q = oracle_choice(self.env, &self.allowed, self.rng);
            pool(&prior, &q, self.prior_weight)
        } else {
            prior
        };
        let conf = probs.iter().cloned().fold(0.0, f64::max);
        let (a, logp) = if conf < self.escalate_below {
            // Escalate. The mock LLM's policy is known, so log its actual
            // probability of the choice; a real LLM's propensity would have to
            // be estimated, or the decision excluded from off-policy
            // evaluation.
            self.stats.escalations += 1;
            let tool = llm_choice(self.env, &self.allowed, self.llm_skill, self.rng);
            let p = llm_prob(self.env, &self.allowed, self.llm_skill, tool);
            (idx(tool), p.ln())
        } else if self.greedy {
            let a = argmax(&probs);
            (a, 0.0) // deterministic: propensity 1
        } else {
            let a = Categorical::new(probs.clone()).unwrap().sample(self.rng);
            (a, probs[a].ln())
        };
        self.last_tool = Some(TOOLS[a]);
        self.trace
            .insert_choice(addr.clone(), ChoiceValue::Usize(a), logp);
        self.trace.log_prior += logp;
        a
    }

    /// An outcome site: run the tool for real and score what came back as an
    /// observation under the world model (-logp is the step's surprise).
    fn on_sample_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>) -> bool {
        let tool = self.last_tool.expect("outcome site without a decision");
        let ok = self.env.call(tool, self.rng);
        let logp = dist.log_prob(&ok);
        self.stats.tool_calls += 1;
        self.stats.surprise -= logp;
        self.trace
            .insert_choice(addr.clone(), ChoiceValue::Bool(ok), logp);
        self.trace.log_likelihood += logp;
        ok
    }

    fn on_sample_f64(&mut self, addr: &Address, _: &dyn Distribution<f64>) -> f64 {
        unreachable!("no f64 sites in the flow ({addr})")
    }
    fn on_sample_u64(&mut self, addr: &Address, _: &dyn Distribution<u64>) -> u64 {
        unreachable!("no u64 sites in the flow ({addr})")
    }
    fn on_observe_f64(&mut self, _: &Address, d: &dyn Distribution<f64>, v: f64) {
        self.trace.log_likelihood += d.log_prob(&v);
    }
    fn on_observe_bool(&mut self, _: &Address, d: &dyn Distribution<bool>, v: bool) {
        self.trace.log_likelihood += d.log_prob(&v);
    }
    fn on_observe_u64(&mut self, _: &Address, d: &dyn Distribution<u64>, v: u64) {
        self.trace.log_likelihood += d.log_prob(&v);
    }
    fn on_observe_usize(&mut self, _: &Address, d: &dyn Distribution<usize>, v: usize) {
        self.trace.log_likelihood += d.log_prob(&v);
    }
    fn on_factor(&mut self, logw: f64) {
        self.trace.log_factors += logw;
    }
    fn finish(self) -> Trace {
        self.trace
    }
}

struct Config {
    name: &'static str,
    use_oracle: bool,
    prior_weight: f64,
    escalate_below: f64,
    greedy: bool,
    consult_below: f64,
}

struct RunOut {
    success: bool,
    steps: usize,
    trace: Trace,
    stats: Stats,
}

const LLM_SKILL: f64 = 0.85;

fn execute(rng: &mut StdRng, wm: &Arc<WorldModel>, cfg: &Config, max_steps: usize) -> RunOut {
    let mut env = Env::default();
    let mut stats = Stats::default();
    let handler = HarnessHandler {
        rng,
        env: &mut env,
        stats: &mut stats,
        allowed: wm.allowed,
        use_oracle: cfg.use_oracle,
        prior_weight: cfg.prior_weight,
        escalate_below: cfg.escalate_below,
        greedy: cfg.greedy,
        consult_below: cfg.consult_below,
        llm_skill: LLM_SKILL,
        last_tool: None,
        trace: Trace::default(),
    };
    let (ep, trace) = run(handler, flow(wm.clone(), max_steps));
    RunOut {
        success: env.success(),
        steps: ep.len(),
        trace,
        stats,
    }
}

/// Encode a raw agent episode as a fugue trace at the flow's addresses.
fn episode_trace(ep: &[(Tool, bool)]) -> Trace {
    let mut t = Trace::default();
    for (i, &(tool, ok)) in ep.iter().enumerate() {
        t.insert_choice(addr!("decide", i), ChoiceValue::Usize(idx(tool)), 0.0);
        if tool != Tool::Finish {
            t.insert_choice(addr!("outcome", i), ChoiceValue::Bool(ok), 0.0);
        }
    }
    t
}

fn decision_logp(t: &Trace) -> f64 {
    t.choices
        .values()
        .filter(|c| c.addr.as_str().starts_with("decide"))
        .map(|c| c.logp)
        .sum()
}

fn fmt_policy(p: &[f64]) -> String {
    TOOLS
        .iter()
        .zip(p)
        .filter(|(_, &x)| x > 0.0)
        .map(|(t, x)| format!("{t:?}={x:.2}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn main() {
    let mut rng = StdRng::seed_from_u64(7);
    let all = [true; K];
    let max_steps = 12;

    // 1. Watch an LLM agent; learn the world model from its traces.
    let mut wm = WorldModel::new(all);
    let (mut llm_ok, mut llm_steps) = (0usize, 0usize);
    let n_train = 300;
    for _ in 0..n_train {
        let (ep, ok) = llm_episode(&mut rng, LLM_SKILL, max_steps, &all);
        llm_steps += ep.len();
        llm_ok += ok as usize;
        wm.learn(&ep);
    }
    let wm = Arc::new(wm);
    println!("== 1. world model learned from {n_train} LLM-agent episodes");
    println!(
        "   LLM agent: success {:.3}, {:.1} steps (= LLM decisions) per episode",
        llm_ok as f64 / n_train as f64,
        llm_steps as f64 / n_train as f64
    );
    for ctx in [
        Ctx::Start,
        Ctx::After(Tool::ReadLogs, true),
        Ctx::After(Tool::EditFile, true),
        Ctx::After(Tool::RunTests, false),
        Ctx::After(Tool::RunTests, true),
    ] {
        println!("   P(next | {ctx:?}) = {}", fmt_policy(&wm.policy(ctx)));
    }

    // 2. Simulate: the same program under PriorHandler is the world model
    //    "dreaming" the flow.
    let n = 4000;
    let mut believed = 0usize;
    for _ in 0..n {
        let (ep, _) = run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            flow(wm.clone(), max_steps),
        );
        believed += believed_success(&ep) as usize;
    }
    let mut believed_greedy = 0usize;
    for _ in 0..n {
        let (ep, _) = run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            flow_with(wm.clone(), max_steps, true),
        );
        believed_greedy += believed_success(&ep) as usize;
    }
    println!("\n== 2. simulate (PriorHandler)");
    println!(
        "   P(success) under the world model's own belief: habit sampled {:.3}, habit greedy {:.3}",
        believed as f64 / n as f64,
        believed_greedy as f64 / n as f64
    );

    // 3. Execute: the same program under the harness handler.
    println!("\n== 3. execute (HarnessHandler), {n} episodes each");
    let configs = [
        Config {
            name: "habit, sampled",
            use_oracle: false,
            prior_weight: 1.0,
            escalate_below: 0.0,
            greedy: false,
            consult_below: 1.01,
        },
        Config {
            name: "habit, greedy",
            use_oracle: false,
            prior_weight: 1.0,
            escalate_below: 0.0,
            greedy: true,
            consult_below: 1.01,
        },
        Config {
            name: "oracle, greedy",
            use_oracle: true,
            prior_weight: 0.0,
            escalate_below: 0.0,
            greedy: true,
            consult_below: 1.01,
        },
        Config {
            name: "habit x oracle, greedy",
            use_oracle: true,
            prior_weight: 0.5,
            escalate_below: 0.0,
            greedy: true,
            consult_below: 1.01,
        },
        Config {
            name: "  + escalate below 0.60",
            use_oracle: true,
            prior_weight: 0.5,
            escalate_below: 0.60,
            greedy: true,
            consult_below: 1.01,
        },
        Config {
            name: "  + escalate below 0.80",
            use_oracle: true,
            prior_weight: 0.5,
            escalate_below: 0.80,
            greedy: true,
            consult_below: 1.01,
        },
        Config {
            name: "habit-first, gate 0.80",
            use_oracle: true,
            prior_weight: 0.5,
            escalate_below: 0.0,
            greedy: true,
            consult_below: 0.80,
        },
        Config {
            name: "  + escalate below 0.60",
            use_oracle: true,
            prior_weight: 0.5,
            escalate_below: 0.60,
            greedy: true,
            consult_below: 0.80,
        },
        Config {
            name: "habit x oracle, sampled",
            use_oracle: true,
            prior_weight: 0.5,
            escalate_below: 0.0,
            greedy: false,
            consult_below: 1.01,
        },
    ];
    println!(
        "   {:<26} success {:.3}  steps {:>4.1}  LLM calls/ep {:.2}",
        "LLM agent alone",
        llm_ok as f64 / n_train as f64,
        llm_steps as f64 / n_train as f64,
        llm_steps as f64 / n_train as f64
    );
    let mut logs: Vec<RunOut> = Vec::new();
    let mut truth = [0.0f64; 2];
    for (ci, cfg) in configs.iter().enumerate() {
        let (mut ok, mut steps, mut oc, mut esc, mut surprise) = (0, 0, 0, 0, 0.0);
        for _ in 0..n {
            let r = execute(&mut rng, &wm, cfg, max_steps);
            ok += r.success as usize;
            steps += r.steps;
            oc += r.stats.oracle_calls;
            esc += r.stats.escalations;
            surprise += r.stats.surprise / r.stats.tool_calls.max(1) as f64;
            if ci == configs.len() - 1 {
                logs.push(r);
            }
        }
        let nf = n as f64;
        let rate = ok as f64 / nf;
        if ci < 2 {
            truth[ci] = rate;
        }
        println!(
            "   {:<26} success {:.3}  steps {:>4.1}  LLM calls/ep {:.2}  oracle calls/ep {:.2}  surprise/step {:.2} nats",
            cfg.name,
            rate,
            steps as f64 / nf,
            esc as f64 / nf,
            oc as f64 / nf,
            surprise / nf
        );
    }

    // 4. Audit: score recorded agent episodes against the flow.
    println!("\n== 4. audit (ScoreGivenTrace / score_given_trace_reconciled)");
    let score = |ep: &Episode| {
        let (_, t) = run(
            ScoreGivenTrace {
                base: episode_trace(ep),
                trace: Trace::default(),
            },
            flow(wm.clone(), max_steps),
        );
        -t.total_log_weight() / ep.len() as f64
    };
    let mut held_out: Vec<f64> = (0..1000)
        .map(|_| score(&llm_episode(&mut rng, LLM_SKILL, max_steps, &all).0))
        .collect();
    held_out.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let odd: Episode = vec![
        (Tool::EditFile, true),
        (Tool::Revert, true),
        (Tool::EditFile, true),
        (Tool::Finish, true),
    ];
    let odd_score = score(&odd);
    let rank = held_out.iter().filter(|&&x| x < odd_score).count() as f64 / 10.0;
    println!(
        "   1000 held-out LLM episodes: surprise median {:.2}, p99 {:.2} nats/step",
        held_out[500], held_out[990]
    );
    println!(
        "   edit/revert/edit/finish:    surprise {:.2} nats/step (percentile {:.1})",
        odd_score, rank
    );
    // Action-space drift: the Revert tool is unexposed; the flow compiled
    // against the new manifest gives the old episode zero probability.
    let mut no_revert = (*wm).clone();
    no_revert.allowed[idx(Tool::Revert)] = false;
    let (_, t, report) = score_given_trace_reconciled(
        episode_trace(&odd),
        &mut rng,
        flow(Arc::new(no_revert), max_steps),
    )
    .unwrap();
    println!(
        "   after unexposing Revert: log p = {} (fresh {:?}, vanished {:?})",
        t.total_log_weight(),
        report.fresh_addresses,
        report.vanished_addresses
    );
    // Structural deviation: an episode that stops early leaves flow sites
    // unvisited; the reconciling scorer names them.
    let truncated: Episode = vec![(Tool::ReadLogs, true), (Tool::EditFile, true)];
    let (_, _, report) = score_given_trace_reconciled(
        episode_trace(&truncated),
        &mut rng,
        flow(wm.clone(), max_steps),
    )
    .unwrap();
    println!(
        "   truncated episode: flow ran past its end; first site it never reached: {}",
        report
            .fresh_addresses
            .first()
            .map(|a| a.to_string())
            .unwrap_or_default()
    );

    // 5. Off-policy evaluation. The sampled pooled runs logged a propensity
    //    at every decision site. Re-scoring those traces under a target flow
    //    gives the target's log-probability of the same decisions; the
    //    difference is the importance weight. The target is just another
    //    fugue program: here the sampled habit and the greedy habit.
    println!(
        "\n== 5. off-policy evaluation from the sampled habit x oracle logs ({} episodes)",
        logs.len()
    );
    for (ti, (name, greedy)) in [("habit, sampled", false), ("habit, greedy", true)]
        .into_iter()
        .enumerate()
    {
        let (mut w_sum, mut w2_sum, mut ws_sum) = (0.0, 0.0, 0.0);
        for r in &logs {
            let (_, rescored) = run(
                ScoreGivenTrace {
                    base: r.trace.clone(),
                    trace: Trace::default(),
                },
                flow_with(wm.clone(), max_steps, greedy),
            );
            let w = (decision_logp(&rescored) - decision_logp(&r.trace)).exp();
            w_sum += w;
            w2_sum += w * w;
            ws_sum += w * r.success as u8 as f64;
        }
        let nl = logs.len() as f64;
        println!(
            "   {:<16} IPS {:.3}  self-normalised {:.3}  on-policy truth {:.3}  ESS {:.0}",
            name,
            ws_sum / nl,
            ws_sum / w_sum,
            truth[ti],
            w_sum * w_sum / w2_sum
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With the competent step unexposed (reading logs, at the start), the
    /// mock LLM still only calls exposed tools, and its logged propensities
    /// are a distribution over exactly those tools.
    #[test]
    fn escalation_respects_the_manifest_and_logs_true_propensities() {
        let env = Env::default();
        assert_eq!(env.best_next(), Tool::ReadLogs);
        let mut allowed = [true; K];
        allowed[idx(Tool::ReadLogs)] = false;
        let mut rng = StdRng::seed_from_u64(1);
        for _ in 0..2000 {
            let t = llm_choice(&env, &allowed, LLM_SKILL, &mut rng);
            assert!(allowed[idx(t)], "called unexposed {t:?}");
        }
        let total: f64 = TOOLS
            .iter()
            .map(|&t| llm_prob(&env, &allowed, LLM_SKILL, t))
            .sum();
        assert!((total - 1.0).abs() < 1e-12, "{total}");
        assert_eq!(llm_prob(&env, &allowed, LLM_SKILL, Tool::ReadLogs), 0.0);
    }

    /// With every tool exposed, the competent step gets the skill plus its
    /// share of the random choice.
    #[test]
    fn the_competent_step_carries_the_skill() {
        let env = Env::default();
        let allowed = [true; K];
        let p = llm_prob(&env, &allowed, LLM_SKILL, Tool::ReadLogs);
        assert!((p - (LLM_SKILL + (1.0 - LLM_SKILL) / K as f64)).abs() < 1e-12);
    }
}
