//! Sample-or-observe sites: one program, three interpreters.
//!
//! A tool-use loop is written once, as a fugue program with a decision site
//! and an outcome site per step. `PriorHandler` simulates it; a handler that
//! makes the calls executes it, scoring what the world returns like an
//! observation; `ScoreGivenTrace` audits a recorded run. See
//! `docs/src/how-to/sample-or-observe-sites.md`.

use fugue::runtime::handler::run;
use fugue::runtime::interpreters::{PriorHandler, ScoreGivenTrace};
use fugue::runtime::trace::{ChoiceValue, Trace};
use fugue::*;
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};

// ANCHOR: program
/// The tools the agent can call. Calling `RESPOND` ends the episode.
const TOOLS: [&str; 3] = ["lookup", "update", "respond"];
const RESPOND: usize = 2;

/// The agent's habit at a step: its probability of calling each tool.
fn habit(step: usize) -> Vec<f64> {
    if step == 0 {
        vec![0.8, 0.1, 0.1]
    } else {
        vec![0.3, 0.3, 0.4]
    }
}

/// How often each tool succeeds, as far as the model knows.
fn reliability(tool: usize) -> f64 {
    [0.95, 0.7, 1.0][tool]
}

/// One episode, from `step` on: a decision site picks a tool, and unless it
/// ends the episode, an outcome site says whether the call succeeded. The
/// outcome's distribution carries its tool as metadata, so a handler that
/// executes the call knows which call to make.
fn episode(step: usize, max_steps: usize) -> Model<Vec<(usize, bool)>> {
    if step == max_steps {
        return pure(Vec::new());
    }
    let decide = Categorical::new(habit(step)).unwrap();
    sample(addr!("decide", step), decide).bind(move |tool| {
        if tool == RESPOND {
            return pure(Vec::new());
        }
        let outcome = WithMeta::new(Bernoulli::new(reliability(tool)).unwrap(), tool);
        sample(addr!("outcome", step), outcome).bind(move |ok| {
            episode(step + 1, max_steps).map(move |rest| {
                let mut steps = vec![(tool, ok)];
                steps.extend(rest);
                steps
            })
        })
    })
}
// ANCHOR_END: program

// ANCHOR: world
/// A stand-in for the real world: each call succeeds or fails as scripted.
struct World {
    script: Vec<bool>,
    calls: Vec<usize>,
}

impl World {
    fn call(&mut self, tool: usize) -> bool {
        let ok = self.script[self.calls.len() % self.script.len()];
        self.calls.push(tool);
        ok
    }
}
// ANCHOR_END: world

// ANCHOR: execute
/// Executes a program's outcome sites against the world. Decision sites are
/// left to the inner `PriorHandler`, which draws them from the habit.
struct Execute<'a> {
    world: &'a mut World,
    /// Each executed step's surprise, `-ln p(outcome)` under the model.
    surprise: &'a mut Vec<f64>,
}

impl<'r, R: RngCore> Overrides<PriorHandler<'r, R>> for Execute<'_> {
    fn on_sample_bool(
        &mut self,
        inner: &mut PriorHandler<'r, R>,
        addr: &Address,
        dist: &dyn Distribution<bool>,
    ) -> bool {
        let tool = *dist
            .downcast_ref::<WithMeta<Bernoulli, usize>>()
            .expect("outcome sites carry their tool")
            .meta();
        // The value comes from outside, so it is scored like an observation:
        // into the likelihood, and recorded so that the run can be replayed
        // and audited.
        let ok = self.world.call(tool);
        let logp = dist.log_prob(&ok);
        inner.trace.log_likelihood += logp;
        inner
            .trace
            .insert_choice(addr.clone(), ChoiceValue::Bool(ok), logp);
        self.surprise.push(-logp);
        ok
    }
}
// ANCHOR_END: execute

/// An episode's steps, ending with the response, or with a note that it
/// reached `max_steps` first.
fn show(steps: &[(usize, bool)], max_steps: usize) -> String {
    let end = if steps.len() < max_steps {
        TOOLS[RESPOND]
    } else {
        "(out of steps)"
    };
    steps
        .iter()
        .map(|&(tool, ok)| format!("{}({})", TOOLS[tool], if ok { "ok" } else { "error" }))
        .chain(std::iter::once(end.to_string()))
        .collect::<Vec<_>>()
        .join(" → ")
}

fn main() {
    let max_steps = 6;

    // ANCHOR: simulate
    // Simulate: the model draws every site, decisions and outcomes alike.
    let mut rng = StdRng::seed_from_u64(7);
    let (dreamt, simulated) = run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        episode(0, max_steps),
    );
    println!("simulated: {}", show(&dreamt, max_steps));
    println!(
        "  log prior {:.3}, log likelihood {:.3}",
        simulated.log_prior, simulated.log_likelihood
    );
    // ANCHOR_END: simulate

    // ANCHOR: run_execute
    // Execute: decisions still come from the habit, but each outcome is the
    // world's answer to a real call.
    let mut world = World {
        script: vec![true, false, true],
        calls: Vec::new(),
    };
    let mut surprise = Vec::new();
    let mut rng = StdRng::seed_from_u64(7);
    let handler = Delegate::with(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        Execute {
            world: &mut world,
            surprise: &mut surprise,
        },
    );
    let (done, executed) = run(handler, episode(0, max_steps));
    println!("executed:  {}", show(&done, max_steps));
    println!(
        "  log prior {:.3} (the decisions), log likelihood {:.3} (the world's answers)",
        executed.log_prior, executed.log_likelihood
    );
    let per_step: Vec<String> = surprise.iter().map(|s| format!("{s:.2}")).collect();
    println!("  surprise per call, in nats: {}", per_step.join(", "));
    // ANCHOR_END: run_execute

    // ANCHOR: audit
    // Audit: score the recorded run under the same program.
    let (_, audited) = run(
        ScoreGivenTrace {
            base: executed.clone(),
            trace: Trace::default(),
        },
        episode(0, max_steps),
    );
    for (addr, choice) in &executed.choices {
        assert!((audited.choices[addr].logp - choice.logp).abs() < 1e-12);
    }
    assert!((audited.total_log_weight() - executed.total_log_weight()).abs() < 1e-9);
    println!(
        "audited:   same {} sites, total log weight {:.3}; ScoreGivenTrace puts all of it in the prior ({:.3})",
        audited.choices.len(),
        audited.total_log_weight(),
        audited.log_prior
    );
    // ANCHOR_END: audit
}

#[cfg(test)]
mod tests {
    use super::*;

    fn execute(script: Vec<bool>, seed: u64) -> (Vec<(usize, bool)>, Trace, Vec<f64>, Vec<usize>) {
        let mut world = World {
            script,
            calls: Vec::new(),
        };
        let mut surprise = Vec::new();
        let mut rng = StdRng::seed_from_u64(seed);
        let handler = Delegate::with(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            Execute {
                world: &mut world,
                surprise: &mut surprise,
            },
        );
        let (steps, trace) = run(handler, episode(0, 6));
        (steps, trace, surprise, world.calls)
    }

    #[test]
    fn executing_takes_the_worlds_answers_and_scores_them_as_likelihood() {
        let (steps, trace, surprise, calls) = execute(vec![false], 3);
        // Every call failed, because the world said so.
        assert!(steps.iter().all(|&(_, ok)| !ok));
        assert_eq!(calls, steps.iter().map(|&(t, _)| t).collect::<Vec<_>>());
        // The outcomes' scores are the likelihood; each is a surprise.
        let expected: f64 = steps
            .iter()
            .map(|&(tool, _)| (1.0 - reliability(tool)).ln())
            .sum();
        assert!((trace.log_likelihood - expected).abs() < 1e-12);
        assert!((surprise.iter().sum::<f64>() + expected).abs() < 1e-12);
    }

    #[test]
    fn the_audit_scores_each_site_as_the_run_did() {
        let (_, executed, _, _) = execute(vec![true, false], 11);
        let (_, audited) = run(
            ScoreGivenTrace {
                base: executed.clone(),
                trace: Trace::default(),
            },
            episode(0, 6),
        );
        assert_eq!(audited.choices.len(), executed.choices.len());
        for (addr, choice) in &executed.choices {
            assert!((audited.choices[addr].logp - choice.logp).abs() < 1e-12);
        }
        assert!((audited.total_log_weight() - executed.total_log_weight()).abs() < 1e-9);
    }

    #[test]
    fn addresses_depend_on_the_step_alone() {
        // Whoever supplied a value, the sites are `decide#i` and `outcome#i`
        // for each step taken, and one more `decide` if the episode ended by
        // responding, so a recorded run can be audited site by site.
        let (steps, executed, _, _) = execute(vec![true], 5);
        let mut names: Vec<String> = executed.choices.keys().map(|a| a.to_string()).collect();
        names.sort();
        let mut expected = Vec::new();
        for i in 0..steps.len() {
            expected.push(format!("decide#{i}"));
            expected.push(format!("outcome#{i}"));
        }
        if steps.len() < 6 {
            expected.push(format!("decide#{}", steps.len()));
        }
        expected.sort();
        assert_eq!(names, expected);
    }
}
