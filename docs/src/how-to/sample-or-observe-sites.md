# Sample-or-Observe Sites

```admonish info title="Contents"
<!-- toc -->
```

Some programs describe a process that also runs for real: an agent calling tools, a controller acting on a plant, a policy in an environment. Such a program has sites whose values the model draws when it **simulates**, and the world supplies when the program is **executed**. There, the handler takes the world's value and scores it like an observation.

This page shows the pattern, where each score belongs, and what to watch for. It is a legitimate choice for a handler to make, not a misuse of `sample`: a `sample` site says *this value is random under the model*, and the handler decides where the value comes from.

## One program, three interpreters

| Interpreter | Decision sites | Outcome sites | Scores go to |
|---|---|---|---|
| **Simulate** with `PriorHandler` | drawn from the model | drawn from the model | `log_prior` |
| **Execute** with a handler of your own | drawn from the model (or chosen by a policy) | **supplied by the world**, then scored | decisions to `log_prior`, the world's answers to `log_likelihood` |
| **Audit** with `ScoreGivenTrace` | read from a recorded trace | read from a recorded trace | `log_prior` (every sample site) |

The program is the same in all three. Only the handler changes.

## The program

A tool-use loop. At each step a decision site picks a tool from the agent's habit. Unless the tool ends the episode, an outcome site says whether the call succeeded:

```rust,ignore
{{#include ../../../examples/sample_or_observe.rs:program}}
```

The outcome's distribution is wrapped in `WithMeta`, which carries the tool as metadata. A handler that executes the call reads it at the site, so it needs no state of its own saying which decision the outcome belongs to.

## Simulate

`PriorHandler` draws every site, decisions and outcomes alike. This is the model "dreaming" an episode:

```rust,ignore
{{#include ../../../examples/sample_or_observe.rs:simulate}}
```

## Execute

To execute, a handler makes the call and takes the world's answer. With `Delegate`, it overrides only the outcome sites and leaves the decisions to `PriorHandler`:

```rust,ignore
{{#include ../../../examples/sample_or_observe.rs:world}}

{{#include ../../../examples/sample_or_observe.rs:execute}}
```

```rust,ignore
{{#include ../../../examples/sample_or_observe.rs:run_execute}}
```

The world's answer is recorded in the trace with its log-probability, like any choice, so the run can be replayed and audited. Its score goes to `log_likelihood`, because the model did not draw it: it is evidence about the model.

**Surprise.** `-ln p(outcome)` is the step's surprise, in nats: how unexpected the world's answer was under the model. A call the model thought would succeed 95% of the time costs 0.05 nats when it does, and 3.0 when it fails. Summed over an episode, surprise measures how well the model predicts the world; a run of high-surprise steps says the model no longer fits.

```admonish tip title="Calls over the network"
When the world is reached over the network (a tool call, a model API), implement the handler as an `AsyncHandler` and interpret the program with `run_async`, so a site waits for its answer without blocking a thread. The accounting on this page is the same.
```

## Audit

`ScoreGivenTrace` re-scores a recorded trace under the same program: here the run just executed, but it could be a log from production:

```rust,ignore
{{#include ../../../examples/sample_or_observe.rs:audit}}
```

Each site's `Choice::logp` and the total log weight match the executed run. The split does not: `ScoreGivenTrace` treats every sample site alike and puts all of it in `log_prior`. Compare per-site scores or totals across interpreters, not the prior and likelihood separately.

## The accounting: prior or likelihood?

The rule: **a value the handler draws from the site's distribution is scored in `log_prior`; a value the world supplies is scored in `log_likelihood`.**

It matters wherever the two are used apart:

- **Importance weights.** With the prior as the proposal, a run's weight is its likelihood (plus factors): the prior terms cancel against the proposal. If the world's answers were put in `log_prior`, every executed run would get the same weight whatever the world said.
- **Model evidence.** Averaging the likelihood over runs drawn from the prior estimates the evidence, `p(data)`. The world's answers are the data.
- **Decisions.** A decision the handler drew from the habit is part of the proposal, and its log-probability is the **propensity** of the action taken. Logged, it is what off-policy evaluation reweights by. A decision made outside the model, such as an LLM's choice, is a value from the world like an outcome. Score it as likelihood to measure how well the habit predicts the agent.

## When to use `observe` instead

Use `observe` when the value is known **before** the run: data that the program is conditioned on, fixed when the model is built.

```rust,ignore
observe(addr!("y", i), Normal::new(mu, sigma).unwrap(), data[i])
```

A sample-or-observe site is for values that exist only **during** the run, because they depend on what was decided earlier. The model cannot build an `observe` for an outcome of a call it has not made yet.

## Pitfalls

- **Values outside the support.** The world can return a value the model thought impossible. A tool whose reliability is 1.0 that fails scores `-inf`, and the run's weight with it. Keep probabilities off 0 and 1 (clamp to `[1e-6, 1 - 1e-6]`) where the world can surprise the model, or treat `-inf` as a signal to stop trusting the model.
- **Addresses must not depend on who supplied a value.** A site's address must be the same whether it is simulated, executed or audited, or `ScoreGivenTrace` will not find the recorded value. Build addresses from the program's own state (the step index), never from where the value came from.
- **State between sites belongs in the program.** If an outcome site needs to know which decision it belongs to, put that in the site: in its address or its distribution's metadata (`WithMeta`), as above. A variable kept in the handler (`last_tool`) breaks as soon as a site is skipped or replayed.
- **Record what you score.** A handler that supplies a value must insert the choice into its trace with its log-probability. Otherwise replay samples a fresh value at that site, and an audit panics on the missing value.

## See also

- [Custom Handlers](./custom-handlers.md): writing handlers, and `Delegate` for overriding a few sites.
- RFC-001's [spike](https://github.com/alexnodeland/fugue/tree/main/docs/decisions/rfc/001-habit-compiler/spike), whose harness handler executes outcome sites this way, and [stretto](https://github.com/alexnodeland/stretto)'s audit, which scores an agent's recorded sessions under a flow with `ScoreGivenTrace`.
