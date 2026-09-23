# RFC-001: Habit compiler — compiling agent behavior into System-One flows

- **Status:** Draft. The design was iterated with @alexnodeland on 2026-09-23; the decisions are in §3.11.
- **Authors:** @alexnodeland (drafted with Claude Code)
- **Created:** 2026-09-23
- **Updated:** 2026-09-23
- **Supersedes / Related:**
  - runnable spike in [`001-habit-compiler/spike/`](001-habit-compiler/spike/);
  - implementation in a new repo, [**stretto**](https://github.com/alexnodeland/stretto), with first results in [`docs/results/phase0-2026-09-23.md`](https://github.com/alexnodeland/stretto/blob/main/docs/results/phase0-2026-09-23.md);
  - TypeSafe AI's Jev (released 2026-09-15).

---

## 1. Summary

LLM agents make every decision with a full model call, even a choice they have made the same way a thousand times before. TypeSafe's Jev makes a typed decision in about 100 ms for a tiny fraction of a cent, but only inside a program someone has already written. TypeSafe's own guidance is that "code handles deterministic work and owns the control flow". Nobody writes those programs from what agents actually do.

This RFC proposes a harness plugin to close that gap. The plugin:

- watches an agent's tool calls (MCP or function calls) as discrete actions;
- keeps a Bayesian model of the agent's behavior over the tools currently exposed;
- compiles the predictable parts into typed probabilistic programs ("flows").

At each branch point a flow asks the cheapest resolver the posterior says is good enough. In order, those are the learned habit, a System-One model, the LLM, or a person.

Fugue supplies the representation. A flow is a `Model`, its branch points are addressed sites, and the harness is a `Handler`. So one flow can be simulated, executed, audited against recorded traces and evaluated counterfactually, just by swapping the interpreter. A spike ([Appendix A](#appendix-a-the-spike)) does all four on fugue 0.2.3 with no changes to the library, using mocks for the tools, the LLM and Jev.

The first implementation, **stretto**, is a Rust MCP proxy. It serves compiled flows to the agent as `plan_*`/`commit_*` macro-tools, and its first evaluation is on τ²-bench (§3.11).

---

## 2. Context / Motivation

### 2.1 What Jev is, and what it is not

- **A "System One" model.** It takes *state* plus typed questions and returns typed answers with probabilities. It never generates text. There are three primitives:
  - `Choice`: up to 255 options; returns per-option probabilities and a confidence.
  - `Score`: 2–10 ordered levels.
  - `Noul`: the probability that a statement is true.
- **The API.** One call is `POST https://api.typesafe.ai/v1/systemone` with a bearer token. The body is `state` plus a map of questions; each question has a `type`, `instructions` and `criteria`.
- **Questions are independent.** All questions in a call are evaluated "in parallel and in isolation". Adding questions barely changes latency, so TypeSafe recommends speculative fan-out.
- **Speed, price and context.**
  - 70–500 ms end to end.
  - $0.042 per million input tokens; output is free.
  - About 64k tokens of shared context. It is also on Cloudflare Workers AI as `typesafe/jev` with a 32k window.
- **Calibration.** It is trained with "Reinforcement Learning for Calibrated Decisions". The reported *confidence* is a statistic of the shape of the returned distribution, not a separate guarantee. `Noul` has no confidence at all.
- **Known limits** (jev-1.13 "jaggedness" page):
  - literal reading of the question;
  - no counting, arithmetic or date comparison;
  - accuracy falls as irrelevant state grows;
  - no adversarial robustness: "injected instructions or misleading framing can steer answers";
  - no structural invariants across questions: P(noul) ≠ 1 − P(not noul).
- **No customization.** No fine-tuning and no few-shot conditioning are documented. Community write-ups so far measure it against frontier LLMs, not against human labels.

### 2.2 The gap

Three facts line up:

1. **Agent traces contain implicit programs.** Tool sequences are far more predictable than their arguments (0.87 against 0.69 similarity across runs [17]). Mapped to a small harness-level alphabet, they cost under one bit per step [8].
2. **System-One models make branch decisions cheap,** but only inside a program with typed branch points.
3. **Nobody derives those programs from behavior and resolves their branch points with calibrated decisions.** The nearest work does something else:
   - it compiles traces into workflows, but leaves the branches to the LLM or to hand-written rules [24];
   - it skips the LLM one step at a time on an uncalibrated score [13];
   - or it learns models of agents only in order to monitor them [1–4].

The missing piece is a compiler from (1) to (2), and a probabilistic programming language (PPL) is the natural home for what it produces.

### 2.3 Related work

Every piece of this exists, and most of it appeared in the last twelve months. Nothing we found combines the pieces. The scan was run on 2026-09-23; numbers in brackets refer to Appendix B.

| Strand | Closest work | What it does | What it leaves open |
|---|---|---|---|
| Learned models of agents, for assurance | AgentGuard [1], TriCEGAR [2], ProbGuard [3], TraceToChain [4], ATLAS [9], PrefixGuard [10] | Learn an MDP, DTMC or automaton from traces (online in [1, 2]); model-check it; flag, re-prompt or halt the agent | Only monitors. Mostly point estimates; [4] is the Bayesian exception, and it runs offline. Abstractions are hand-made [1], derived from specs [3] or from an LLM [9], or refined from counterexamples [2] |
| How predictable tool sequences are | Automata from Agent Traces [8], AutoTool [13], How Consistent Are LLM Agents? [17] | 0.93 bits/step on harness-level alphabets, with structure "shaped more by the harness than by the LLM" [8]. Next-tool entropy drops from 3.50 to 1.93 bits with 2nd-order context [13] | This is both our motivation and our warning: arguments vary much more than tool choice [17] |
| Compiling traces into workflows | TraceCompiler [24], Speculative Macro Commit [16], Act While Thinking [14], AWM / plan caching / AgentRR [25] | Mine argument dataflow and recurring macros, and compile them into mostly deterministic workflows (34 calls down to 11 on one task [24]) | Branch points go to the LLM or to hand-written rules [24]. Speculation keeps the LLM confirming every step [14–16] |
| Skipping the LLM | AutoTool [13] | Executes the predicted next tool without the LLM when a score clears a threshold and the arguments can be filled; capped at 30% of steps | One step at a time, on a heuristic, uncalibrated score, with no outcome model |
| Calibrated escalation | R2V Agent, ReDAct [23] | A calibrated router decides when a small model should hand a step to a large one. Escalating about 15% of steps can match running the large model throughout | The fast path is a generative small model, not a typed choice among a flow's options |
| World models of tool environments | ToolEmu [18], StableToolBench / MirrorAPI / GTM [19], WMA / WebDreamer [20], MCP-Cosmos [21] | Simulate tool responses for testing, training or planning | Never used to authorize execution. LLM simulators are unreliable [18, 20] |
| Secure plan-then-execute; PPL + LLM | _pending: literature scan still running_ | | |

**What is new here.** Stated narrowly:

1. **Calibrated, typed decisions at branch points.** These are the points TraceCompiler leaves to the LLM and AutoTool approximates with a heuristic score. Each one is resolved by the habit, a System-One model, the LLM or a person, chosen by expected loss.
2. **Two continuously updated Bayesian models over the live tool manifest:**
   - one of the agent, P(action | abstract state), used to find flows;
   - one of the environment, P(outcome | state, action), used to check them.

   Prior work either models only the environment [1, 2] or merges the two into one chain [3, 4, 9].
3. **The model authorizes execution.** Simulation and model checking decide whether a flow is promoted, rather than only raising alerts.
4. **One artifact, four uses.** A flow is a fugue program. Simulating, executing, auditing and evaluating it counterfactually are four handlers over the same object, and logged propensities make the last one possible.
5. **Semantic predicates in TriCEGAR-style refinement.** Predicates are System-One questions about the raw state, and one is kept when it raises the marginal likelihood of the traces.

A skeptical reviewer could describe this as "AutoTool + TraceCompiler + TriCEGAR + R2V". That is roughly right, and it is the argument for building it: each of those pieces lacks something another one supplies.

### 2.4 Goals and non-goals

**Goals**

- **Measure before changing anything (shadow mode).** For each decision context, measure how predictable the agent is and how well a System-One model agrees with it.
- **Learn continuously.** Learn a Bayesian world model over the exposed action space from traces, as they arrive.
- **Compile and run.** Compile predictable sub-flows into typed programs with explicit decision sites. Run them with uncertainty-gated arbitration and escalation.
- **Audit and evaluate.** Check conformance, surprise and drift, and evaluate counterfactually from logged propensities.
- **Stay oracle-agnostic.** Jev is the first implementation of an `Oracle` trait, not a dependency of fugue core.

**Non-goals**

- Replacing the LLM for open-ended work, or generating free text with a classifier.
- Training or fine-tuning models.
- Multi-agent coordination, at least initially.

---

## 3. Proposed solution

### 3.1 Architecture

The agent keeps its own loop. The proxy sits between the agent and its tools: it records every call, and it serves compiled flows as extra tools.

```text
agent (any MCP client)
  │ tools/call: real tools, plus plan_<flow> / commit_<flow>
  ▼
stretto proxy (Rust) ──record──► trace store ──► world model ──► compiler
  │   ▲                                               ▲             │
  │   └── flow runtime (fugue Handler) ◄── flows ◄────┼─────────────┘
  │          │ at each decision site: habit │ Jev │ LLM hand-back
  ▼          ▼                              outcomes
MCP servers (real tools)
```

### 3.2 The mapping onto fugue

| Agent-harness concept | Fugue construct | Notes |
|---|---|---|
| Next tool choice | `sample(addr!("decide", i), Categorical::new(..))` | Support is the set of exposed tools |
| Tool result (abstracted) | outcome site `sample(addr!("outcome", i), ..)` | Sampled when simulating; scored as an observation when executing |
| Recorded episode | `Trace` | `Choice::logp` at decision sites is the logged propensity |
| Harness runtime | `impl Handler` | Decides who resolves each site |
| Simulate a flow | `PriorHandler` | The world model "dreaming" |
| Conformance and surprise | `ScoreGivenTrace`, `score_given_trace_reconciled` | `fresh`/`vanished` addresses are structural deviations |
| Counterfactual evaluation | Re-score a logged trace under a target flow | Log-ratio at decision sites is the importance weight |
| `commit_*` after `plan_*` | Replay the planned trace (`ReplayHandler`) up to the confirmation site, then continue into the write sites | Every decision resolves exactly as it did at plan time |
| Sub-flow extraction and splicing | `Trace::extract_prefix` / `graft_prefix` | From the F3 trace-surgery work |
| Flow structure search | `block_regeneration_mh`, `PopulationKernel`, fugue-evo | From the EA-as-PPL work |
| Online belief over latent task phase | SMC / particle filter | |
| Choosing a state abstraction | Log-evidence | Closed form for Dirichlet–multinomial; SMC otherwise |

### 3.3 World model

- **Abstract action.** α = (tool, closed-set arguments). Enum and boolean arguments from the tool's JSON Schema are part of the action. Free-text arguments become typed slots (§3.5).
- **Abstract observation.** φ = MCP `isError` plus a vector of predicate answers (§3.4).
- **Context.**
  - The last *k* (α, φ) pairs, with hierarchical back-off to shorter contexts. That means Dirichlet smoothing, or a Pitman–Yor / sequence-memoizer model once there is enough data.
  - Optionally, a latent task phase *z* (an HMM) tracked by SMC.
- **Parameters.**
  - A Dirichlet over the next action for each context.
  - A Beta or Dirichlet over outcomes for each (context, action).
  - Both are conjugate, so each event is an O(1) streaming update.
- **Support is the current manifest** (`tools/list`).
  - When a tool disappears, every flow that uses it scores −∞ and is invalidated. The spike shows this.
  - When a tool appears, its prior comes from back-off, plus optionally Jev-judged similarity to the descriptions of existing tools.
- **Non-stationarity.**
  - Exponential forgetting on counts.
  - Change-point alarms on surprise (Bayesian online change-point detection) for when the agent's model, its prompts or its tools change.
- **Posteriors, not frequencies.** Every quantity used downstream is a posterior. Observing 3/3 and 300/300 gives the same frequency but should give very different arbitration decisions.

### 3.4 State abstraction by predicate refinement

The world model is only as good as its state abstraction, and every prior system reports this as its hardest part [1–4, 9]. In the spike, a single hidden bit (whether the logs located the bug) made the model believe the greedy habit succeeds 99.9% of the time. In the environment it succeeds 94.5% of the time.

TriCEGAR [2] already refines abstractions of agent traces with counterexamples, using predicate trees over typed lifecycle events. We add three things:

- predicates about the *raw* state, evaluated by a System-One model;
- selection by marginal likelihood;
- a refined model that goes on to authorize execution, not only monitoring.

The loop:

1. **Detect aliasing.** Look for:
   - contexts whose next-action distribution stays high-entropy;
   - transitions with high surprise;
   - gaps between simulated and real outcomes on executed flows.
2. **Propose predicates.** An LLM reads examples from the aliased context and proposes `Noul`/`Choice` questions about the raw state, for example "Do the logs name a specific file?". This is TypeSafe's own "autoresearch feature discovery" cookbook pattern, pointed at the world model instead of at a regressor.
3. **Evaluate cheaply.** Jev answers each candidate question over the stored traces. At Jev's pricing that costs cents per thousand traces.
4. **Keep what explains the data.** Keep a predicate if it raises the marginal likelihood of the trace corpus under the world model. This is closed form for the Dirichlet model and SMC evidence for latent-variable variants. Bayesian model selection supplies the Occam penalty.

### 3.5 Compiling flows

- **Traces first; the policy checks.**
  - Structure, argument dataflow and branch probabilities are mined from traces.
  - A written policy, where one exists, is used only to name flows and to check the rule guards. It is never used to invent structure.
- **Candidate regions** are sub-graphs of the world model with:
  - high visitation;
  - low conditional entropy given the available predicates;
  - high downstream success;
  - bounded stakes.
- **Decision sites** are the points where a region branches. Each one gets a question spec made of:
  - the options, which are the successor abstract actions;
  - instructions derived from the LLM's own rationales in the traces;
  - a *state slice*: the minimal fields that predicted the branch. Jev's accuracy drops with irrelevant state, so the slice matters.
- **"The LLM names it, Jev finds it."** When the agent calls a macro-tool it has already read the conversation, so intent, descriptions and the user's stated reasons cost nothing to pass as arguments. Jev handles the decisions that arise *mid-flow*, over data the LLM has not seen:
  - matching descriptions to fetched records ("the Boston trip next week" → one of N reservations);
  - classifying stated reasons against the policy's categories;
  - judging tool outputs (error, retry, alternative path, or hand back);
  - picking the next sub-flow when the rule guards do not settle it.
- **Rule guards (dates, amounts, eligibility) are code, never Jev.** Jev cannot compare dates or do arithmetic.
  - An LLM compiles the domain policy into typed guard predicates once, offline.
  - Each guard is tested against the successful traces.
  - A person reviews any disagreement.
- **Arguments**, handled according to the tool's JSON Schema:
  - `enum` → `Choice`; `boolean` → `Noul`; an optional argument → a "was it stated?" `Noul`. TypeSafe's function-calling cookbook does exactly this.
  - A free-text argument becomes a dataflow binding: from a macro-tool input, or from an earlier output where the traces show the value appearing verbatim (TraceCompiler's provenance classes [24]).
  - Otherwise the region is not compiled.
- **Writes go through plan/commit pairs.**
  - `plan_<flow>` runs the lookups, the guards and the Jev decisions without writing anything. It returns the exact proposed write calls plus a token.
  - The agent shows the proposal to the user and obtains an explicit "yes", as τ²-bench's policies require.
  - `commit_<flow>(token)` then executes exactly what was planned, by replaying the planned trace up to the confirmation site and continuing into the write sites.
- **Output is a serializable flow IR** (sites, questions, bindings, guards). It is interpreted into a fugue `Model` at load time. fugue-wasm's `dsl.rs` already interprets a `prob!` subset into real `Model`s at runtime; the flow IR generalizes that.
- **Macro-tools as options.** The proxy serves each flow as a pair of MCP tools. In options-framework terms:
  - the initiation set is the applicability predicate;
  - the intra-option policy is the flow;
  - termination is completion, or a hand-back that names the unresolved site.

### 3.6 Execution: arbitration, not pooling

At each decision site the runtime picks the cheapest resolver whose expected loss is acceptable:

1. **Habit.** The world model's posterior predictive. Free and instant. Used when its top option is confident *and* is backed by enough evidence.
2. **System-One oracle (Jev).** Consulted only where the habit is unsure. Its answer is treated as an observation from a sensor with a per-site confusion matrix learned from outcomes (Dawid–Skene style). That also repairs the fact that Jev's per-question probabilities are not jointly coherent.
3. **LLM.** Inside a macro-tool this means a hand-back: the flow returns the unresolved site, its options and the evidence gathered so far, and the agent decides.
4. **Human.** When the action is irreversible and confidence is below the site's bar.

Stakes come from MCP tool annotations (`readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`). These are untrusted hints with pessimistic defaults, so the harness cross-checks them against side effects observed in the traces.

The spike shows why arbitration beats pooling:

| Configuration | Success | LLM calls per episode |
|---|---|---|
| LLM alone | 0.870 | 4.96 |
| Habit, greedy (pure habit) | 0.945 | 0 |
| Habit pooled with the oracle at every step | 0.803 | 0 |
| Oracle consulted only when the habit's top option is below 0.8; escalate below 0.6 | 0.963 | 0.41 |

These numbers come from mocks and say nothing about real workloads; the mechanism is what matters. It is the uncertainty-based arbitration between habitual and deliberative control described by Daw, Niv & Dayan (2005), made explicit.

The runtime also:

- **Records every resolved decision** with its propensity (`Choice::logp`), its resolver, its question and a hash of its state slice.
- **Watches surprise.** It tracks surprise per step. When surprise exceeds the site's threshold it ends the flow with a hand-back.
- **Speculates safely.** It pre-executes the most likely next call only if that call is read-only and idempotent.

### 3.7 Evaluation and assurance

- **Shadow, then canary, then promote.** A site is automated only after:
  - shadow mode shows the required agreement and calibration on real traffic;
  - a canary shows no drop in downstream success.
- **Per-site calibration.** TypeSafe claims calibration in general. The harness measures it per site (reliability curves, ECE), because the traces contain the labels: what the LLM did, and whether the episode succeeded.
- **Counterfactual evaluation.** Every stochastic decision logs a propensity, so a candidate flow can be evaluated on old logs by re-scoring them. In the spike this estimated a sampled target at 0.768 (inverse propensity scoring, IPS) against an on-policy 0.777. But:
  - Whole-trajectory importance weights degenerate quickly. The same run had an effective sample size (ESS) of 20 out of 4000.
  - A greedy target was overestimated at 0.996, against a true 0.945.
  - So prefer per-site (contextual-bandit) estimates and doubly-robust estimators that use the world model as the direct method.
  - Counterfactual evaluation needs exploration. Keep a small sampling rate on reversible, low-stakes sites only.
- **Simulation ranks; it does not certify.** Simulated success is optimistic whenever the abstraction aliases hidden state: in the spike, 0.999 believed against 0.945 real. Use simulation to:
  - prune candidate flows;
  - run statistical model checking of safety properties, for example "P(destructive call without a preceding passing check) < 10⁻³".

  Then confirm on canary traffic.
- **Conformance.** `score_given_trace_reconciled` names the sites where a recorded episode:
  - left the flow (`fresh`);
  - did things the flow does not model (`vanished`).

  Per-step surprise ranks episodes for review. In the spike a deliberately odd episode ranked at the 96th percentile of held-out episodes, not beyond the 99th. A noisy agent produces odd episodes regularly, so surprise is a triage signal, not a detector.

### 3.8 Security properties

Compiled flows are the "plan-then-execute" and "action-selector" patterns from the prompt-injection literature. The difference is that the plan comes from observed behavior rather than from an up-front LLM plan:

- control flow is fixed before any untrusted tool output is read;
- the System-One model can only choose among enumerated options, so injected content can at worst pick a different *allowed* branch.

That bounds the blast radius but is not immunity. Jev "does not treat state as hostile by default", and calibration measured on benign traffic need not survive an attack [Ray 2026]. So:

- keep untrusted tool output out of the state slices of high-stakes sites whenever trusted fields can decide the branch;
- attach provenance to state slices, and raise the confidence bar when untrusted content is present;
- never let a flow call a tool outside its compiled set.

### 3.9 Where it lives

The implementation lives in a new repo, **stretto**, like fugue-evo does. In a fugue, a stretto is where entries of the subject overlap and compress: here, many traces compress into one flow.

The proxy brings dependencies fugue core should not carry: an async runtime, an MCP SDK, HTTP clients and benchmark glue.

| Crate | Contents |
|---|---|
| `stretto-trace` | Canonical episode schema. Ingest from τ²-bench trajectory logs and proxy logs, and later from OpenTelemetry GenAI spans |
| `stretto-model` | Abstraction, Dirichlet/Beta world model with back-off, forgetting and evidence, on fugue |
| `stretto-oracle` | `Oracle` trait; Jev HTTP client; mock oracle; a replay cache keyed by content, so every Jev answer an experiment uses is paid for once and is reproducible |
| `stretto-compile` | Flow mining, dataflow provenance, policy-guard checking, flow IR → fugue `Model` |
| `stretto-proxy` | MCP proxy serving the real tools plus `plan_*`/`commit_*` macro-tools; runtime handler with arbitration and propensity logging |
| `stretto-report` | Phase 0 and experiment reports |
| `bench/tau2` (Python) | τ²-bench tools exposed as an MCP server bound to each task's environment; an agent that is an MCP client |

The spike surfaced six changes to `fugue-ppl`. All are additive, and they will go to this repo as their own PRs:

1. **Async interpretation.**
   - Today `run` is a synchronous trampoline, but tool calls and Jev calls are network I/O.
   - Add `run_async` over the same `Model`, driven by an `AsyncHandler`.
   - The model is data, so this is a second trampoline, not a rewrite.
2. **Site metadata for handlers.**
   - A handler sees only `(addr, dist)`. The spike's handler recovers the prior from `dist.log_prob`, but a Jev question also needs the question spec and a state slice.
   - Options: an address-keyed registry, or an optional `fn as_any(&self) -> Option<&dyn Any>` on `Distribution` that defaults to `None`.
3. **Less handler boilerplate.**
   - A handler must implement every `on_sample_*` and `on_observe_*` method.
   - A delegating adapter would fix this: override the sites you care about and defer the rest to an inner handler.
4. **Flow IR.** Generalize fugue-wasm's `dsl.rs` interpreter into a serializable program format that builds `Model`s at runtime.
5. **Vector-valued sites.**
   - There is no `Dirichlet` or `Multinomial`, and sites are scalar.
   - Conjugate helpers are enough for the world model.
   - A Gamma-normalization helper, or vector sites, would let the Bayesian model live inside fugue proper.
6. **Documented pattern: sample-or-observe sites.**
   - The same site is sampled when simulating and scored as an observation when executing.
   - This is a legitimate choice for a handler to make, and the spike does it.
   - It deserves a how-to page.

### 3.10 Phased plan

| Phase | Build | Gate to start |
|---|---|---|
| 0. Measure | Offline, on τ²-bench trajectories: the world model, plus Jev asked retrospectively at every recorded LLM decision ("replayed shadow mode") | None |
| 1. Audit and proxy | Recording proxy; conformance, surprise and drift; guard compilation and checking against traces | Phase 0 numbers are in |
| 2. Compile and run | Flow IR, plan/commit macro-tools, arbitration runtime, the experiment arms in §3.11 | Held-out projections show ≥ 20% fewer LLM turns at ≤ 1 point of pass^1 lost |
| 3. Learn | Predicate refinement, per-site counterfactual evaluation, flow search with fugue-evo, big-to-small transfer | Phase 2 results on airline and retail |

Replayed shadow mode is equivalent to live shadow mode, because Jev's answer depends only on the state we send it. It lets Phase 0 run on recorded trajectories before the proxy exists.

For each decision context, the Phase 0 report gives:

- the number of visits;
- next-tool entropy;
- the habit's top-option posterior, with a credible interval;
- Jev's agreement with the LLM for each of the four roles in §3.5, and its calibration;
- downstream success;
- argument provenance: closed-set, bound from input, copied from an earlier output, or generated;
- the share of LLM turns avoidable at a target error rate.

### 3.11 First experiment: τ²-bench

These decisions were made during the 2026-09-23 design iteration.

**Phase 0a results.** stretto measured τ²-bench's published airline and retail trajectories: four models, 4 trials per task, habit trained on the official train split and tested on the held-out split. No API keys were needed.

- **Macro-tool headroom.** 24–25% of LLM turns sit inside runs of tool calls that a macro-tool could perform in one call.
- **Argument binding.** Identifiers, items, payment methods and flights in write calls are almost always copied from earlier outputs or user messages. What agents generate is mostly closed-set choices and arithmetic.
- **Where a content-blind habit fails.** A habit that sees only the action sequence can act on just 6–8% of the decisions made right after a tool returns (at τ = 0.8). Continuing a run depends on what the tool returned, which is where a System-One model is needed.
- **Transfer.** A habit learned from one model predicts another within 3–7 points of top-1.
- **Code features and a named intent.**
  - Code features were read from tool outputs and selected by cross-validation grouped by task. The grouping is needed because task-identifying fields fool in-sample evidence.
  - Adding them, plus the intent a macro-tool call names, raises the share of retail decisions the habit could take at τ = 0.8 from 15% to 24%. Airline stays near 12%.
  - About three quarters of decisions still need content-aware judgment. That residual is what Jev is measured on in Phase 0b.
- **Results page:** [Stretto Phase 0](https://claude.ai/artifact/FWzzt74xNUWoaeq5seua6t). It is private by default; the owner shares it from the page.

| Question | Decision |
|---|---|
| What v1 is for | Compile and run flows, measured first |
| Workload | τ²-bench [26]: airline and retail first; telecom's dual-control domain later |
| Where the harness plugs in | A Rust MCP proxy; flows served as macro-tools |
| Jev | API access available. Jev owns the four mid-flow roles in §3.5 |
| Writes | Plan/commit pairs; the agent obtains the user's explicit "yes" between them |
| Branches nothing in the flow can settle | Resumable: the flow pauses and returns a token; `resume_<flow>(token, choice)` continues it. Plan/commit is the same mechanism, paused at the confirmation site |
| Rule guards | Compiled from the policy by an LLM, tested against traces, reviewed by a person |
| Flow discovery | Traces first; the policy only names flows and checks guards |
| Models | Transfer first. Flows are compiled from published frontier-model trajectories (free to us) and run by GLM (Z.ai) and MiniMax on their subscription keys. American frontier models come later. Cost is kept to a minimum |
| Phase 0 data | τ²-bench's published trajectories |
| Checking raw writes | A separate experimental arm, so the gains from flows and from checking alone stay separable |
| Win conditions | All four: fewer LLM calls, tokens and dollars; higher pass^k; Jev agreeing with the frontier model at branches, and well calibrated; fewer policy violations |
| Code | New public repo, [stretto](https://github.com/alexnodeland/stretto) (MIT); fugue changes go upstream as their own PRs |
| Phase 2 gate | ≥ 20% fewer LLM turns at ≤ 1 point of pass^1 lost, on held-out tasks |
| Keys | TypeSafe, GLM and MiniMax keys are added later; until then stretto runs offline with mock and replay oracles |

**Experimental arms.** Each arm runs on held-out tasks, with k trials per task:

| Arm | Agent sees | Branches resolved by |
|---|---|---|
| A | Raw tools | The LLM (baseline) |
| B | Raw tools, with guard checks on writes | The LLM |
| C | Raw tools + macro-tools | The LLM, via hand-back at every branch (TraceCompiler-like) |
| D | Raw tools + macro-tools | Habit → Jev → LLM, by arbitration (ours) |
| E | As D, plus guard checks on raw writes | As D |
| A-small, D-small | The same arms, run by a small model with flows compiled from the frontier model's traces | |

**Metrics**

- LLM calls, tokens and dollars per task.
- pass^1 … pass^k.
- For each Jev role: agreement with the frontier model on held-out tasks, and calibration (ECE).
- Policy violations:
  - writes without a preceding confirmation;
  - writes that fail the policy's guards;
  - calls outside a flow's tool set.
- Macro-tool adoption rate: agents given extra tools may not use them (see §4).
- End-to-end latency.

---

## 4. Drawbacks

- **Predictability has a ceiling.**
  - For general agents, top-1 next-tool accuracy is 27.8% [14] and at most 55% [15]. Even with joint RL it reaches only 61–66%.
  - Low entropy appears only after traces are mapped to small, harness-level alphabets [8].
  - Coding agents on novel tasks may be mostly long tail. Phase 0 exists to measure all this cheaply.
- **Arguments are the hard part.**
  - They vary much more than tool choice [17].
  - A region whose free-text arguments cannot be bound from macro-tool inputs or earlier outputs stays with the LLM.
- **The harness changes its own data.**
  - Once flows execute, the traces are produced by agent and harness together, so statistics learned from ungated traces need not describe the gated system [Ray 2026].
  - Logged propensities and a little exploration mitigate this; they do not remove it.
- **Macro-tools might go unused.** Agents given a world model as a tool used it less than 1% of the time [Qian 2026]. Adoption is a measured outcome, not an assumption. If it is low, we add a reference agent loop for the experiments.
- **Vendor risk.** Jev is proprietary and in early access. It offers no fine-tuning, and there are no public accuracy benchmarks against human labels. Hence the `Oracle` trait and the replay cache.
- **Evaluation is hard (§3.7).**
  - Simulation is optimistic and counterfactual estimates are high-variance.
  - Guarantees in the style of ProbGuard's PAC bounds call for 530 to 10⁵ traces [3].
  - Canaries cost traffic.
- **Drift.** A new LLM version, prompt or tool changes behavior. Compiled flows must be re-validated, not trusted forever.
- **Scope.** This is a new product surface beside a PPL that is still pre-1.0 and has a single maintainer.

---

## 5. Alternatives considered

- **Use Jev only as middleware** (LangChain's `ModelRouterMiddleware` and `AutoModeMiddleware`).
  - It is cheap, useful and orthogonal: it routes calls and gates risk.
  - But it learns nothing and compiles nothing.
- **Skip single steps the way AutoTool does** [13].
  - It is proven to cut LLM calls by up to 30%.
  - But it works one step at a time on an uncalibrated score, with no outcome model and no way to verify anything. We use it as a comparison point, not as a design.
- **Compile deterministic workflows the way TraceCompiler does** [24]. This is arm C. It shows how much of the gain comes from structure alone, before any calibrated branch resolution.
- **Distill the agent into a small policy model, or use R2V-style escalation** [23].
  - These are strong baselines for cost.
  - But they give no typed decision points and no per-decision probabilities to audit.
  - They have no structure to verify, and they need retraining on every drift.
- **Plain counting instead of a PPL.**
  - Counting is fine for Phase 0, and we should use closed forms wherever they exist.
  - It stops being enough once we need any of these:
    - per-site oracle reliability when the true answer is latent;
    - latent task phases;
    - abstraction selection by evidence;
    - decision-theoretic escalation.

---

## 6. Unresolved questions

1. **Phase 0 data.** Fresh τ²-bench runs (API spend), or published trajectories where they exist and are usable?
2. **Models and budget.** Which frontier and small models, and how much can we spend?
3. **The repo.** Visibility and license for stretto.
4. **How are compiled flows reviewed?** Probably as code, with the flow IR diffed in pull requests.
5. **Privacy for non-benchmark workloads.** Traces contain user data. The store should keep hashes and state slices, with retention limits.

---

## 7. Decision

(To be filled in once reviewed)

- Outcome: Accepted | Rejected | Deferred
- Notes from discussion.

---

## Appendix A: the spike

[`001-habit-compiler/spike/src/main.rs`](001-habit-compiler/spike/src/main.rs) is about 800 lines. It runs against fugue 0.2.3 with no changes to the library:

```bash
cargo run --release --manifest-path docs/decisions/rfc/001-habit-compiler/spike/Cargo.toml
```

**Setup**

- **The task.** A CI-repair task over five tools: `ReadLogs`, `RunTests`, `EditFile`, `Revert`, `Finish`.
  - A hidden bit records whether the logs located the bug.
  - A mock "LLM agent" takes the right action 85% of the time and a random action otherwise.
- **The world model.** A Dirichlet–categorical model over the next tool, given (last tool, did it work), with Beta–Bernoulli outcomes. It is learned from 300 of the agent's episodes.
- **The flow.** One fugue program with `decide#i` and `outcome#i` sites, run under different handlers.
- **The mock oracle.** It is calibrated by construction: its top option is right with exactly its stated probability, and that probability is drawn uniformly from [0.35, 0.99].

**Results** (seed 7; 4000 episodes per configuration)

| What | Result |
|---|---|
| LLM agent alone | success 0.870, 4.96 LLM calls per episode |
| Simulated success (`PriorHandler`), sampled and greedy habit | 0.811 and 0.999 (believed) |
| Habit, sampled / greedy (`HarnessHandler`, real environment) | 0.777 / 0.945, no LLM calls |
| Oracle only, greedy | 0.580 |
| Habit pooled with the oracle at every step | 0.803 |
| Pooled, escalating below 0.6 / 0.8 | 0.910 at 1.41 LLM calls / 0.938 at 1.96 |
| Habit first, oracle only below 0.8 | 0.916 at 1.62 oracle calls, no LLM calls |
| …and escalate below 0.6 | **0.963 at 0.41 LLM calls and 1.50 oracle calls** |
| Surprise of held-out LLM episodes | median 0.58, p99 3.46 nats/step; the odd episode is 2.95 (96th percentile) |
| Tool unexposed after recording | the old episode scores `-inf` |
| Episode that stops early | the reconciling scorer names `decide#2` as the first unreached site |
| Counterfactual estimate, sampled habit, from pooled logs | IPS 0.768 against a true 0.777; ESS 20 of 4000 |
| Counterfactual estimate, greedy habit, from pooled logs | IPS 0.996 against a true 0.945; ESS 347 |

**What the spike shows**

- The mapping in §3.2 works with today's API.
- One program serves as simulator, executor, auditor and counterfactual target.
- Logged `Choice::logp` values are usable as propensities.
- The action-space constraint behaves like a type check.

**What it does not show**

- **It is not evidence of benefit on real workloads.** Every component is a mock.
- **The greedy habit beating its teacher is an artifact.** It follows from the mock agent's errors being independent and random ("the mode denoises the teacher"). Real LLM errors are correlated with state and will not wash out this way.
- **The weak oracle is an assumption.** The mock oracle is deliberately worse than the learned habit on routine states, and that is what makes naive pooling look bad. How good Jev is on each site is exactly what Phase 0 measures.

---

## Appendix B: references

Papers marked "preprint" had no listed venue on the date of the scan (2026-09-23).

**Learned models of agents, for assurance**

1. R. Koohestani. *AgentGuard: Runtime Verification of AI Agents.* ASE 2025 AgenticSE workshop. <https://arxiv.org/abs/2509.23864>
2. R. Koohestani et al. *TriCEGAR: A Trace-Driven Abstraction Mechanism for Agentic AI.* Preprint, 2026. <https://arxiv.org/abs/2601.22997>
3. H. Wang, C. M. Poskitt, J. Wei, J. Sun. *ProbGuard: Proactive Runtime Monitoring for LLM Agent Safety via Probabilistic Prediction* (v1 title: *Pro2Guard*). ASE 2026. <https://arxiv.org/abs/2508.00500>
4. P. T. Tran-Truong, X.-B. Le. *Measuring the Unmeasurable: Markov Chain Reliability for LLM Agents.* Preprint, 2026. <https://arxiv.org/abs/2604.24579>
5. Z. Chen, M. Kang, B. Li. *ShieldAgent: Shielding Agents via Verifiable Safety Policy Reasoning.* ICML 2025. <https://arxiv.org/abs/2503.22738>
   Also: H. Wang et al. *AgentSpec.* ICSE 2026. <https://arxiv.org/abs/2503.18666>
6. F. Fournier, L. Limonad, Y. David. *Agentic AI Process Observability: Discovering Behavioral Variability.* PMAI 2025. <https://arxiv.org/abs/2505.20127>
7. L. Lin et al. *Mining Workflow Graphs for Black-Box Boundary Testing of Conversational LLM Agents.* Preprint, 2026. <https://arxiv.org/abs/2607.06873>
8. S. Cho et al. *Automata from Agent Traces: Failure and Next-Step Prediction.* Preprint, 2026. <https://arxiv.org/abs/2608.23670>
9. I. D. Lopez-Miguel et al. *ATLAS: Discovering Agent Strategies through LLM-Guided Abstraction and Automata Learning.* MODELS 2026. <https://arxiv.org/abs/2608.14352>
10. X. Huang et al. *PrefixGuard: From LLM-Agent Traces to Online Failure-Warning Monitors.* Preprint, 2026. <https://arxiv.org/abs/2605.06455>
11. M. Tappler et al. *Automata Learning meets Shielding.* ISoLA 2022. <https://arxiv.org/abs/2212.01838>

**Tool-sequence models, speculation and compilation**

12. X. Liu et al. *ToolNet: Connecting Large Language Models with Massive Tools via Tool Graph.* Preprint, 2024. <https://arxiv.org/abs/2403.00839>
13. J. Jia, Q. Li. *AutoTool: Efficient Tool Selection for Large Language Model Agents.* AAAI 2026. <https://arxiv.org/abs/2511.14650>
14. Y. Sui et al. *Act While Thinking* (now *Parallelizing Tool Execution and LLM Generation for Low-Latency Agent Serving*). Preprint, 2026. <https://arxiv.org/abs/2603.18897>
15. N. Ye et al. *Speculative Actions: A Lossless Framework for Faster AI Agents.* ICLR 2026. <https://arxiv.org/abs/2510.04371>
16. Z. Liu, S. Kundu, P. A. Beerel. *Speculative Macro Commit for Faster Tool-Using Agents.* MLSP 2026. <https://arxiv.org/abs/2609.03236>
17. A. Yagubyan. *How Consistent Are LLM Agents? Measuring Behavioral Reproducibility in Multi-Step Tool-Calling Pipelines.* Preprint, 2026. <https://arxiv.org/abs/2605.28840>

**World models of tool environments**

18. Y. Ruan et al. *Identifying the Risks of LM Agents with an LM-Emulated Sandbox* (ToolEmu). ICLR 2024. <https://arxiv.org/abs/2309.15817>
19. Z. Guo et al. *StableToolBench* (<https://arxiv.org/abs/2403.07714>) and *MirrorAPI* (<https://arxiv.org/abs/2503.20527>); Z. Ren et al. *GTM* (<https://arxiv.org/abs/2512.04535>).
20. H. Chae et al. *Web Agents with World Models.* ICLR 2025. <https://arxiv.org/abs/2410.13232>
    Y. Gu et al. *Is Your LLM Secretly a World Model of the Internet?* <https://arxiv.org/abs/2411.06559>
21. G. Ganapavarapu, D. Patel. *MCP-Cosmos: World Model-Augmented Agents for Complex Task Execution in MCP Environments.* Preprint, 2026. <https://arxiv.org/abs/2605.09131>
22. Y. Zuo et al. *Qwen-AgentWorld.* Preprint, 2026. <https://arxiv.org/abs/2606.24597>
    Z. Wang et al. *Agent World Model.* ICML 2026. <https://arxiv.org/abs/2602.10090>

**Escalation and flow reuse**

23. R. V. Hemadri et al. *R2V Agent: Teaching SLMs When to Ask for Help.* Preprint, 2026. <https://arxiv.org/abs/2605.16604>
    D. Piatrashyn et al. *ReDAct.* Preprint, 2026. <https://arxiv.org/abs/2604.07036>
24. S. El Yadouni, G. Li. *TraceCompiler: Skill-Guided Mining and Compilation of LLM Agent Traces into Mostly Deterministic Workflows.* Preprint, 2026. <https://arxiv.org/abs/2608.02680>
25. Z. Z. Wang et al. *Agent Workflow Memory.* ICML 2025 (<https://arxiv.org/abs/2409.07429>). Q. Zhang et al. *Agentic Plan Caching.* NeurIPS 2025 (<https://arxiv.org/abs/2506.14852>). E. Feng et al. *AgentRR* (<https://arxiv.org/abs/2505.17716>).

**Benchmark**

26. V. Barrès, H. Dong, S. Ray, X. Si, K. Narasimhan. *τ²-Bench: Evaluating Conversational Agents in a Dual-Control Environment.* 2025. <https://arxiv.org/abs/2506.07982>; code at <https://github.com/sierra-research/tau2-bench>

**Pitfalls cited in §3.8 and §4**

- S. Ray. *What Can Be Enforced? A Theory of Certified Runtime Safety for Tool-Using Agents.* Preprint, 2026. <https://arxiv.org/abs/2607.22868>
- C. Qian et al. *Current Agents Fail to Leverage World Model as Tool for Foresight.* Preprint, 2026. <https://arxiv.org/abs/2601.03905>

**Foundations**

- N. D. Daw, Y. Niv, P. Dayan. *Uncertainty-based competition between prefrontal and dorsolateral striatal systems for behavioral control.* Nature Neuroscience, 2005.
- E. Clarke et al. *Counterexample-Guided Abstraction Refinement.* CAV 2000.
- A. P. Dawid, A. M. Skene. *Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm.* JRSS C, 1979.
- R. S. Sutton, D. Precup, S. Singh. *Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning.* Artificial Intelligence, 1999.
- R. P. Adams, D. J. C. MacKay. *Bayesian Online Changepoint Detection.* 2007.
- M. Dudík, J. Langford, L. Li. *Doubly Robust Policy Evaluation and Learning.* ICML 2011.

**TypeSafe and MCP**

- TypeSafe AI. *Introducing System One Models & Jev* (<https://typesafe.ai/blog/introducing-system-one-models-and-jev>). Docs, including the API reference, confidence, jev-1.13 jaggedness and cookbooks: <https://docs.typesafe.ai/llms.txt>
- LangChain. *Building a harness with Jev.* <https://www.langchain.com/blog/building-a-harness-with-jev>
- Model Context Protocol blog. *Tool Annotations as Risk Vocabulary: What Hints Can and Can't Do.* 2026-03-16. <https://blog.modelcontextprotocol.io/posts/2026-03-16-tool-annotations/>
