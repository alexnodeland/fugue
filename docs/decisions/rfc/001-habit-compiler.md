# RFC-001: Habit compiler — compiling agent behavior into System-One flows

- **Status:** Accepted (2026-09-23, [#51](https://github.com/alexnodeland/fugue/pull/51)). Amended the same day with what Phase 0 found (§3.12, [#52](https://github.com/alexnodeland/fugue/pull/52)) and with Phase 0b v2's read-only flows and arbitration (§3.13, [#53](https://github.com/alexnodeland/fugue/pull/53)), and on 2026-09-24 with the first live form and pilot (§3.14, [#54](https://github.com/alexnodeland/fugue/pull/54)), with the built implementation, the airline pilot, policy guards and the flow audit (§3.15, [#55](https://github.com/alexnodeland/fugue/pull/55)), with the arms measured before building macro-tools (§3.16, [#56](https://github.com/alexnodeland/fugue/pull/56)), with the habit alone live, fewer traces and Jev as a confirmation judge (§3.17, [#58](https://github.com/alexnodeland/fugue/pull/58)), and with a cold start from an agent's own sessions, options from the manifest, a second confirmation question and matching descriptions to records (§3.18, [#59](https://github.com/alexnodeland/fugue/pull/59)). The design was iterated with @alexnodeland on 2026-09-23; the decisions are in §3.11.
- **Authors:** @alexnodeland (drafted with Claude Code)
- **Created:** 2026-09-23
- **Updated:** 2026-09-24
- **Supersedes / Related:**
  - runnable spike in [`001-habit-compiler/spike/`](001-habit-compiler/spike/);
  - implementation in a new repo, [**stretto**](https://github.com/alexnodeland/stretto), with results in:
    - [`docs/results/phase0-2026-09-23.md`](https://github.com/alexnodeland/stretto/blob/main/docs/results/phase0-2026-09-23.md) (Phase 0a);
    - [`docs/results/phase0b-2026-09-23-summary.md`](https://github.com/alexnodeland/stretto/blob/main/docs/results/phase0b-2026-09-23-summary.md) (Phase 0b, Jev);
    - [`docs/results/phase0b-v2-2026-09-23-summary.md`](https://github.com/alexnodeland/stretto/blob/main/docs/results/phase0b-v2-2026-09-23-summary.md) (Phase 0b v2);
    - [`docs/results/phase0b-v2-goal-free-2026-09-24-summary.md`](https://github.com/alexnodeland/stretto/blob/main/docs/results/phase0b-v2-goal-free-2026-09-24-summary.md) (v2 without a goal);
    - [`pilot/README.md`](https://github.com/alexnodeland/stretto/blob/main/pilot/README.md#results-so-far) (the live pilot);
    - [`docs/results/pilot-airline-2026-09-24.md`](https://github.com/alexnodeland/stretto/blob/main/docs/results/pilot-airline-2026-09-24.md) (the airline pilot);
    - [`docs/results/guards-2026-09-24.md`](https://github.com/alexnodeland/stretto/blob/main/docs/results/guards-2026-09-24.md) (policy guards against published trajectories);
    - [`docs/results/pilot-guards-2026-09-24.md`](https://github.com/alexnodeland/stretto/blob/main/docs/results/pilot-guards-2026-09-24.md) (the guards, live);
    - [`docs/results/audit-2026-09-24.md`](https://github.com/alexnodeland/stretto/blob/main/docs/results/audit-2026-09-24.md) (a flow audited as a fugue program);
    - [`docs/results/arms-2026-09-24.md`](https://github.com/alexnodeland/stretto/blob/main/docs/results/arms-2026-09-24.md) (arm C, the habit alone, and what naming a flow would add);
    - [`docs/results/pilot-habit-2026-09-24.md`](https://github.com/alexnodeland/stretto/blob/main/docs/results/pilot-habit-2026-09-24.md) (the habit alone, live);
    - [`docs/results/sweep-2026-09-24.md`](https://github.com/alexnodeland/stretto/blob/main/docs/results/sweep-2026-09-24.md) (fewer traces);
    - [`docs/results/confirm-2026-09-24.md`](https://github.com/alexnodeland/stretto/blob/main/docs/results/confirm-2026-09-24.md) (Jev as a confirmation judge);
    - [`docs/results/cold-start-2026-09-24.md`](https://github.com/alexnodeland/stretto/blob/main/docs/results/cold-start-2026-09-24.md) (a cold start from an agent's own sessions);
    - [`docs/results/manifest-options-2026-09-24.md`](https://github.com/alexnodeland/stretto/blob/main/docs/results/manifest-options-2026-09-24.md) (options from the tool manifest);
    - [`docs/results/confirm-second-2026-09-24.md`](https://github.com/alexnodeland/stretto/blob/main/docs/results/confirm-second-2026-09-24.md) (a second confirmation question);
    - [`docs/results/matching-2026-09-24.md`](https://github.com/alexnodeland/stretto/blob/main/docs/results/matching-2026-09-24.md) (matching descriptions to records);
    - the working paper, [alexnodeland.github.io/stretto](https://alexnodeland.github.io/stretto/);
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

The first implementation, **stretto**, is a Rust MCP proxy. It serves compiled flows to the agent as `plan_*`/`commit_*` macro-tools, and its first evaluation is on τ²-bench (§3.11). *Amended (§3.14):* its first live form adds no tools. A read-only flow runs behind the agent's own calls and returns its lookups with each result. *Amended (§3.15):* the proxy serves flows, policy guards and a commit tool for any MCP server, and a flow is audited against new sessions as a fugue program.

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
| `commit_*` after `plan_*` | Replay the planned trace (`ReplayHandler`) up to the confirmation site, then continue into the write sites | Every decision resolves exactly as it did at plan time. *Amended (§3.13):* the writes are the ones the LLM specified; the flow decides none |
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
- **"The LLM names it, Jev finds it."** When the agent calls a macro-tool it has already read the conversation, so intent, descriptions and the user's stated reasons cost nothing to pass as arguments. *Amended (§3.14):* a read-only flow needs no name; measured without the goal, it agrees and saves as much. Jev handles the decisions that arise *mid-flow*, over data the LLM has not seen:
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
  - *Amended (§3.13):* flows propose no writes. `plan_*` only reads. `commit_*` executes the write calls the LLM specified after the user's confirmation, and decides nothing itself.
- **Output is a serializable flow IR** (sites, questions, bindings, guards). It is interpreted into a fugue `Model` at load time. fugue-wasm's `dsl.rs` already interprets a `prob!` subset into real `Model`s at runtime; the flow IR generalizes that.
- **Macro-tools as options.** The proxy serves each flow as a pair of MCP tools. In options-framework terms:
  - the initiation set is the applicability predicate;
  - the intra-option policy is the flow;
  - termination is completion, or a hand-back that names the unresolved site.
  - *Amended (§3.16):* not built. A name would add only the lookups that need a value from the conversation: at most 1.9% of LLM turns in retail and 7.9% in airline.

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
- **Speculates safely.** It pre-executes the most likely next call only if that call is read-only and idempotent. *Amended (§3.14):* the first live form is this, run as a flow, with the results returned to the agent.

*Amended (§3.16):* for read-only flows, the habit alone saves as many turns as the arbitration above. Replayed on GLM-5's and Claude 3.7 Sonnet's test episodes, weighing Jev's answers cut the detours in airline by half to two-thirds, and in retail by up to a quarter. So arbitration is a precision setting, and a flow can run without a System-One model.

*Amended (§3.17):* only once there are enough traces. With the habit trained on three retail tasks, the arbitration above saves 20.4% of turns and the habit alone 1.8%. A deployment starts on the arbiter, and the habit takes over the savings as its traces accumulate.

*Amended (§3.18):* not for a deployment's own first sessions. §3.17's samples were clustered, and its arbiter was fitted on thousands of other agents' decisions. From a random handful of an agent's own sessions, the habit alone saved more than an arbiter fitted on those sessions until about twenty of them. A deployment starts on the habit alone, can carry an arbiter fitted elsewhere as insurance against a narrow start, and fits its own once it has enough sessions.

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
| Where the harness plugs in | A Rust MCP proxy; flows served as macro-tools. *Amended:* the first live form runs behind the agent's own tool calls, with no new tools (arm D0, §3.14). *Amended again:* that is the form; named macro-tools are not built (§3.16) |
| Jev | API access available. Jev owns the four mid-flow roles in §3.5 |
| Writes | Plan/commit pairs; the agent obtains the user's explicit "yes" between them. *Amended:* between LLM turns flows only read; every write goes back to the LLM, so a wrong pick is a detour rather than a risk (§3.13) |
| Branches nothing in the flow can settle | Resumable: the flow pauses and returns a token; `resume_<flow>(token, choice)` continues it. Plan/commit is the same mechanism, paused at the confirmation site |
| Rule guards | Compiled from the policy by an LLM, tested against traces, reviewed by a person. *Amended:* built, 12 retail and 10 airline rules; the audit against published trajectories decides which are enforced (§3.15). *Amended again:* the explicit yes before a write is judged by Jev, not a word list; logged, not yet enforced (§3.17) |
| Flow discovery | Traces first; the policy only names flows and checks guards |
| Models | Transfer first. Flows are compiled from published frontier-model trajectories (free to us) and run by GLM (Z.ai) and MiniMax on their subscription keys. American frontier models come later. Cost is kept to a minimum. *Amended:* compile from the 2026 frontier runs on Sierra's leaderboard, which transfer to GLM-5 better than the 2025 baselines (§3.12) |
| Phase 0 data | τ²-bench's published trajectories |
| Checking raw writes | A separate experimental arm, so the gains from flows and from checking alone stay separable |
| Win conditions | All four: fewer LLM calls, tokens and dollars; higher pass^k; Jev agreeing with the frontier model at branches, and well calibrated; fewer policy violations |
| Code | New public repo, [stretto](https://github.com/alexnodeland/stretto) (MIT); fugue changes go upstream as their own PRs |
| Phase 2 gate | ≥ 20% fewer LLM turns at ≤ 1 point of pass^1 lost, on held-out tasks, *judged per agent model and domain* (amended: the savings depend on how the agent calls tools, §3.12; *and harness*, §3.15). Tokens and dollars are projected alongside turns, because a flow also keeps intermediate tool outputs out of the LLM's context. *Amended again:* read-only flows take no risky decisions, so offline the gate is turns saved. Detours are counted and charged in tokens, and the live pilot must show they cost no pass^1 (§3.13) |
| Arbitration | The habit acts only in contexts where its held-out agreement is at least 99%, validated per context rather than by one global threshold. Jev decides everywhere else. *Amended:* validation needs at least 20 decisions from at least 10 distinct tasks; what survives is hand-backs only, so the rule is revisited with the Phase 0b data (§3.12). *Amended again:* Jev's answers are no longer taken at their word. A conditional logit combines them with the habit's prior, Jev's record at the site and state predicates, fitted by cross-validation over tasks (§3.13). *And again:* for read-only flows, optional. The habit alone saves as many turns, and the arbiter makes fewer detours (§3.16). *And again:* optional only with enough traces. With few, the arbiter carries the savings (§3.17). *And again:* not with a deployment's own few sessions, where the habit alone did better until about twenty (§3.18) |
| Keys | The TypeSafe key is in the environment (Phase 0b ran on 2026-09-23); GLM and MiniMax keys come with Phase 2. Offline work uses mock and replay oracles |

**Experimental arms.** Each arm runs on held-out tasks, with k trials per task:

| Arm | Agent sees | Branches resolved by |
|---|---|---|
| A | Raw tools | The LLM (baseline) |
| B | Raw tools, with guard checks on writes (built and run live, §3.15) | The LLM |
| C | Raw tools + macro-tools | The LLM, via hand-back at every branch (TraceCompiler-like). Replayed as a flow behind the tools: 0–1.7% of turns saved (§3.16) |
| D | Raw tools + macro-tools | Habit → Jev → LLM, by arbitration (ours). Not built: naming adds at most 1.9% of turns in retail and 7.9% in airline over D0 (§3.16) |
| E | As D, plus guard checks on raw writes | As D |
| D0 | Raw tools; each result may carry a read-only flow's lookups (§3.14), served by `stretto-proxy` for any MCP server (§3.15) | Habit and Jev by arbitration, or the habit alone (§3.16; live in §3.17), for lookups only; the LLM for everything else |
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

### 3.12 Amendment 1: what Phase 0 found (2026-09-23)

Phase 0 finished on τ²-bench's published trajectories: the four 2025 baselines, plus nine current models from Sierra's leaderboard used only as transfer targets. Phase 0b then asked Jev at every held-out decision a flow would hand it. Four findings changed the plan; §3.11's decisions table is updated to match.

**1. The gate depends on how the agent calls tools, so it is judged per model and domain.**

- Held-out episodes were replayed through `plan_*` flows, with the habit acting only where validated and a System-One model that always agrees with the agent deciding the rest.
  - Pooled over the 2025 baselines, flows save 22.3% of LLM turns in retail and 20.6% in airline.
  - They save 27–28% of input tokens, because a collapsed run also stops carrying its intermediate tool outputs.
- Per model the spread is wide:

  | Agent model | Parallel tool turns (retail / airline) | Turns saved, retail | Turns saved, airline (ceiling) |
  |---|---|---|---|
  | Qwen3.5-397B | 0% / 0% | 43.5% | 44.1% (53.1%) |
  | Claude Sonnet 4.5 | 11% / 4% | 32.1% | 33.2% (42.5%) |
  | GPT-5.2, reasoning high | 19% / 37% | 28.6% | 20.8% (26.9%) |
  | GLM-5 | 27% / 45% | 29.1% | 15.4% (18.9%) |
  | Claude Opus 4.5 | 28% / 41% | 28.6% | 13.4% (18.2%) |

- Agents that call tools one at a time leave long runs to collapse. Agents that batch independent calls have already done part of that work.
- For GLM-5 and Claude Opus 4.5 in airline, even collapsing every run falls short of 20%.
- So the gate is judged for each (agent model, domain) pair, and GLM's and MiniMax's parallel-call rates are the first thing Phase 2 measures.

**2. Validation must count tasks, and what it validates is a stopping rule.**

- Every task repeats across trials and agent models, so its decisions are near-copies.
  - Trained on the 2026 frontier runs, one airline context validated at 100% of 28 decisions drawn from a few tasks.
  - On held-out tasks it held 43%, and put a risky call in 10% of episodes.
- Validation now also requires 10 distinct tasks.
- With that rule, each domain keeps one validated context, and both are hand-backs to the LLM:
  - The habit adds safety but no automation.
  - Every in-flow decision falls to the System-One model: about 7.5 per episode.

**3. Compile from current frontier runs.** Here is how well habits learned from other models predict GLM-5's held-out actions (top-1):

- **Retail:** Claude Sonnet 4.5 67%, Claude Opus 4.5 66%, Gemini 3 Flash 66%. GLM-5's own habit gets 67%, and the 2025 baselines 56–63%.
- **Airline:** Claude Opus 4.5 62%, Gemini 3 Pro 62%, Gemini 3 Flash 62%. GLM-5's own habit gets only 57%, and the 2025 baselines 48–57%.

**4. Zero-shot Jev is well calibrated, but not accurate enough to carry a flow.** jev-1.13.0 answered all 8,946 questions (28M input tokens, $1.19).

- **Next step** (which tool comes next, or hand back):
  - it agrees with the agent 72% of the time in both domains;
  - it is right about stopping versus going on 76% of the time;
  - its expected calibration error is 0.06;
  - at p ≥ 0.9 it covers a third of decisions, at 92–93% agreement.
- **Closed-set arguments:**
  - 98.5% agreement in retail, where the values are things like a cancellation reason;
  - 58% in airline. Two things drive that:
    - baggage counts (33–45%), which need arithmetic, and §3.5 already assigns to code;
    - composite arguments such as `flights`, which the v1 question builder wrongly offered as closed sets of other episodes' values. That is fixed.
- **Projected with the validated habit:**
  - Trusting Jev at p ≥ 0.9, flows save 3.9% of LLM turns in retail and 4.7% in airline, with a risky call in 6.2% and 13.1% of episodes.
  - Lower thresholds save more (15% and 12% at p ≥ 0.5), but put a risky call in half to two thirds of episodes.
  - A two-key rule, where Jev's pick counts only when it is also the habit's top option, cuts risk only a little. At p ≥ 0.9 it saves 3.0% with 4.4% risky in retail, and 2.9% with 8.4% in airline.
  - No agent model passes the gate.
- **The v1 questions were deliberately naive.** Every tool was an option, and the state was the whole recent transcript.
  - §3.5 specifies narrower questions: options limited to the successors seen at the site, a state slice, and instructions drawn from the traces.
  - §3.6 combines Jev with the habit through a per-site confusion matrix, instead of taking Jev at its word.
  - Those are the next Phase 0b iterations. They are measured offline in the same way, for about $1 per full pass.
- Agreement with the agent is a conservative proxy. A different but equally valid next step counts as a disagreement, so the live runs in Phase 2 remain the real test.

**Also done.** Phase 1's recording proxy ([`stretto-proxy`](https://github.com/alexnodeland/stretto/tree/main/crates/stretto-proxy)):

- It forwards every MCP line unchanged and records sessions, which `stretto-trace` reads into episodes.
- Serving flows as macro-tools and checking writes against compiled guards come next.

### 3.13 Amendment 2: read-only flows and arbitration (2026-09-23)

Phase 0b v2 built the question design of §3.5 and the arbitration of §3.6, and measured them the same way as v1:

- **Answers:** jev-1.13.0 answered 32,116 questions, 80.7M input tokens, for $3.39.
- **Agent models:** the four 2025 baselines as source models, and the nine leaderboard models as transfer targets.
- **Details:** in stretto's [v2 summary](https://github.com/alexnodeland/stretto/blob/main/docs/results/phase0b-v2-2026-09-23-summary.md).

Five findings; §3.11's decisions table and §6 are updated to match.

**1. Flows only read between LLM turns, so a wrong pick is a detour, not a risk.**

- A flow decides no writes. Writes, and any tool not marked read-only, go back to the LLM. v1's projection had let flows execute writes mid-run.
- Plan/commit (§3.5) is amended to match:
  - `plan_*` only reads, and proposes no writes of its own.
  - `commit_*` executes, in one call, the write calls the LLM specified after the user's "yes". It decides nothing.
  - The projection is conservative here: it hands every write back to the LLM, one turn each, instead of batching confirmed writes into one `commit_*` call.
- A flow that picks the wrong next step makes an extra lookup and then hands back. That costs a pause, and the extra output rides along in later prompts (the projection charges it). Nothing in the environment changes, so offline no decision is risky.
- The ceiling barely moves. With a perfect System-One model:
  - GLM-5 goes from 29.1% to 28.2% of LLM turns in retail, and from 15.4% to 14.9% in airline;
  - the 2025 baselines pooled go from 22.3% to 21.0% in retail, and from 20.6% to 17.7% in airline.

**2. Narrower questions did not make Jev more accurate.** Scored the same way, on the same 300-per-domain sample of decisions:

| Domain | v1 question | v2 question |
|---|---|---|
| Retail | 77.7% | 73.1% |
| Airline | 73.4% | 73.4% |

- The v2 question offers only the lookups seen at the site, over a state slice.
- The stop question asked on its own does worse still.

**3. Arbitration and state predicates help; describing the tools does not.**

- **The arbiter.** A conditional logit, fitted by cross-validation over tasks on the source models only, weighs:
  - the habit's prior;
  - both question designs;
  - Jev's record at the site on other tasks (a one-coin Dawid–Skene sensor);
  - the answers to state predicates.
- **The predicates.** Three domain-agnostic ones were proposed as §3.4 describes:
  - records in a list still unchecked;
  - options not yet looked up;
  - something only the customer can give.
- **Agreement with the agent's next step:**

  | Configuration | Retail | Airline | GLM-5, retail | GLM-5, airline |
  |---|---|---|---|---|
  | One question (Jev alone) | 76.2% | 72.4% | 77.9% | 63.3% |
  | Combined | 77.9% | 74.3% | 81.7% | 69.5% |
  | Combined, with predicates | 80.5% | 78.9% | 85.7% | 75.8% |

  - Where the combination is at least 0.99 sure, it agrees 99.2% of the time in retail (on 10.6% of decisions) and 97.0% in airline (on 25.2%).
  - "Records still unchecked" carries the most weight.
- **Dataflow hints** describe each lookup by the write arguments its results supplied in training. They fixed the sites they targeted, but did not raise agreement overall.

**4. Because detours are harmless, flows can act on weaker picks.**

- At p ≥ 0.9 read-only flows save 2.5–5.9% of LLM turns.
- Acting at p ≥ 0.5 saves 15–16% in retail and 10–11% in airline, pooled over the 2025 baselines.
- Taking the likeliest lookup whenever it is at least 0.3 likely (*lookup first*) saves 17% and 13%.
  - Handing back costs a turn and a wrong lookup does not, so this is the cost-optimal rule for read-only flows.
  - Detours then show up in 57–70% of episodes.
  - Dollars fall faster than turns even after charging detours, because a typical lookup returns only 200–250 tokens.

**5. Sequential callers clear the gate offline.** Read-only flows, combined answers trusted at p ≥ 0.5, on the transfer targets:

| Agent model | Retail | Airline | Offline gate |
|---|---|---|---|
| Qwen3.5 | 29.7% | 22.5% | Both domains |
| Qwen3-Max | 24.2% | 19.9% (26.3% lookup first) | Both domains (airline with lookup first) |
| Gemini 3 Flash | 24.1% | 10.8% | Retail |
| Gemini 3 Pro | 23.1% | 10.6% | Retail |
| Claude Sonnet 4.5 | 21.1% | 15.2% | Retail |
| GPT-5.2, reasoning off | 18.1% (22.8% lookup first) | 2.4% | Retail (lookup first) |
| GLM-5 | 14.7% (20.5% with predicates, lookup first) | 2.7% | Retail (predicates, lookup first) |
| Claude Opus 4.5 | 14.7% | 2.8% | Neither |
| GPT-5.2, reasoning high | 13.9% | 2.8% | Neither |

- In airline the heavy parallel callers can't pass even in principle: their read-only ceilings are 12–17% (§3.12).
- The detour rate is the price of each pass: from 9% to 75% of episodes.

What this changes:

- **The offline gate is now turns saved.** Detours are counted and charged in tokens. The live pilot must show that they do not cost pass^1 before anything is deployed.
- **The pilot model is chosen by the projection.** Offline, Qwen3.5 is the strongest candidate. GLM-5, whose key is at hand, clears the gate only in retail, and only with the predicates.

### 3.14 Amendment 3: the first live form (2026-09-24)

stretto ran flows live for the first time, in τ²-bench retail:

- **Agent:** GLM-5.3, in Claude Code on Z.ai's GLM Coding Plan (§6, question 8). GLM-5.3 also plays the customer, from τ²-bench's user-simulator prompt. The reward is τ²-bench's database check; its natural-language assertions need an LLM judge and were left out.
- **Tools:** τ²-bench's, served over MCP behind stretto's recording proxy.
- **Flow:** compiled from the 2025 baselines without a goal, and served by `stretto flow-serve`. It asks Jev live, through the replay cache.
- **Details:** stretto's [pilot results](https://github.com/alexnodeland/stretto/blob/main/pilot/README.md#results-so-far) and [goal-free summary](https://github.com/alexnodeland/stretto/blob/main/docs/results/phase0b-v2-goal-free-2026-09-24-summary.md).

Five findings; §3.11, §4 and §6 are updated to match.

**1. The goal adds nothing to read-only flows.** Every v2 run gave flows the episode's goal (the writes it goes on to make), as a macro-tool call would name it (§3.5). Asked again without it, 8,092 questions for $0.93, pooled over the 2025 baselines:

| | Retail, with goal | Retail, without | Airline, with goal | Airline, without |
|---|---|---|---|---|
| Next step, combined answer | 80.5% | 80.5% | 78.9% | 78.4% |
| Turns saved, lookup first, p ≥ 0.3 | 17.4% | 17.3% | 12.8% | 12.6% |
| Episodes with a detour | 67.3% | 62.3% | 57.2% | 60.0% |

- GLM-5 still clears the gate in retail: 20.3% without the goal, 20.5% with it.
- This fits what read-only flows do. They hand every write back. What they decide, which record to look up next and whether there is enough, is in the lookups already made and in what the customer said.

**2. So the first live form needs no macro-tool.** In arm D0, after each of the agent's own tool calls, the flow asks its questions, makes the lookups it is sure enough of, and returns them in the same tool response, until it hands back.

- It is §3.6's safe speculation, run as a flow, with the results given to the agent.
- It adds no tools and changes no prompt, which sidesteps the adoption risk in §4.
- Flows that end in writes are still named by the LLM, as `plan_*` and `commit_*` (§3.5).

**3. Arguments: act on the probability of the whole call.** Offline, an argument counted as bindable when its value appeared earlier in the episode (§3.12). Live, the flow has to pick one.

- For each lookup argument, it learns from training where the values came from: a tool and a path in its result, such as `get_user_details` at `$.orders[*]`.
- It takes the first value there that has not been looked up yet, preferring one the customer mentioned.
- It acts when the tool's probability times the binding's agreement is at least 0.3. The agreement is measured per call: how often the arguments it would have bound, all of them at once, matched the agent's own call in training. That is 92% for orders and 60% for products the customer did not mention (95% and 76% when mentioned). A lookup with several arguments, such as airline's flight status (a flight number and a date), counts as agreeing only when every argument matches.
- On one trial of GLM-5's recorded test episodes (40 episodes; see finding 4), the rule cut detours from 42 to 18 without losing a saved turn. 89% of the flow's lookups were then the agent's own, up from 78%.

**4. The live flow reproduces the projection.** GLM-5's recorded retail test episodes were replayed through the live flow, with real Jev answers and no LLM; a recorded call the flow had already made was skipped.

- Over all four trials, 160 episodes and 1,384 LLM turns, it saved 294 turns (21.2%). The offline projection for the same episodes saves 281 (20.3%).
- Detours came up in 27 episodes, against 45 projected, and 91% of the flow's lookups were the agent's own.
- Where the flow follows the agent's path, it asks byte-identical questions to the offline run, so they are served from the replay cache.

**5. Live, the agent uses the flow's lookups.** Ten retail test tasks, drawn at random, ran once in each arm:

| | Without the flow | With the flow |
|---|---|---|
| LLM turns | 110 | 84 (24% fewer) |
| Agent input tokens | 723,926 | 571,528 (21% fewer) |
| Passed the database check | 8 of 10 | 8 of 10 |

- Per episode the flow saved 2.6 LLM turns (95% interval 1.0 to 4.2), with fewer turns in 9 of the 10 pairs.
- The flow acted on 9 of the 10 tasks, with 26 lookups, and the agent repeated 3 of them. On those tasks, turns fell by 26%. The one task where it never acted took a turn more, from the simulated customer.
- The two failures were the same agent error in both arms: a return processed before the exchange it blocked, and a wrong variant.
- Ten pairs cannot bound a one-point loss of pass^1. They show that the mechanism works live and give a first effect size.
- The pilot cost 236 Z.ai credits at the off-peak rate, and the arm with the flow used 16% fewer. That is about 12 per episode, where Z.ai's Lite plan allows 2,000 per 5 hours.

What this changes:

- **Arm D0** joins §3.11's arms: raw tools, with a read-only flow behind them.
- **The pilot model** is GLM-5.3, the model the Z.ai key serves (§6, question 8).
- **Next:** a paired run large enough to bound pass^1 and the cost of detours; airline; and the sequential callers the projection favors, once a key for them is at hand.

### 3.15 Amendment 4: the first design, built; airline, guards and the audit (2026-09-24)

stretto now implements the first design end to end. It measured three more things: the live form in airline, policy guards against published trajectories, and a flow audited as a fugue program.

- **Details:** stretto's [implementation status](https://github.com/alexnodeland/stretto/blob/main/docs/design.md#implementation-status-2026-09-24), [airline pilot](https://github.com/alexnodeland/stretto/blob/main/docs/results/pilot-airline-2026-09-24.md), [guard audit](https://github.com/alexnodeland/stretto/blob/main/docs/results/guards-2026-09-24.md) and [flow audit](https://github.com/alexnodeland/stretto/blob/main/docs/results/audit-2026-09-24.md), and the [working paper](https://alexnodeland.github.io/stretto/), which collects every result so far.

Five findings; §3.11, §4, §6 and §7 are updated to match.

**1. The first design is built.**

- **The flow IR.** A flow compiles to versioned JSON: the habit, the sites, the fitted arbiter folds, the argument bindings and its provenance. `stretto compile` builds it from τ²-bench results, and `stretto learn` from sessions the proxy recorded. It reloads exactly.
- **The proxy.** `stretto-proxy` serves the flow for any stdio MCP server. After each of the agent's calls it makes the flow's lookups and appends them to the result (arm D0). It also:
  - checks the agent's calls against policy guards (arms B and E);
  - adds `stretto_commit`, which makes the writes the user confirmed in one call (§3.5's commit), each guarded;
  - logs the conversation a host hands it, which MCP never carries.
- **The loop.** A test records forty sessions through the proxy, learns a flow from their logs, and serves it to a new session.
- **Not built yet:**
  - `plan_*` / `resume_*` macro-tools that the LLM names (arms C and D);
  - counterfactual evaluation from logged propensities;
  - predicate refinement (§3.4).

**2. Airline, live: 17% fewer LLM turns.** Ten airline test tasks, set up as the retail pilot (§3.14):

| | Without the flow | With the flow |
|---|---|---|
| LLM turns | 127 | 105 (17.3% fewer) |
| Agent input tokens | 996,580 | 849,367 (14.8% fewer) |
| Passed the database check | 8 of 10 | 9 of 10 |

- The flow saved 2.2 turns per episode (95% interval 0.5 to 3.9), with fewer turns in 7 of 10 pairs.
- It acted on every task, with 24 lookups the agent never repeated.
- None of the three failures came from a flow decision. In the one in the flows arm, the customer would pay under $100 for a change. The agent upgraded the basic-economy reservation (a $301 charge), changed its flights (a $220 refund), and quoted the net, $81. The task counts the upgrade's cost and expects no change.
- The pilot cost 334 Z.ai credits, 11% less in the flows arm.

**3. The savings follow the harness as well as the model.**

- Offline, flows save GLM-5 under 5% of its airline turns (§3.13). GLM-5 in τ²-bench's own harness makes parallel calls in 45% of its airline tool turns.
- GLM-5.3 in Claude Code made them in 3.8% (3.0% in retail), like the sequential callers the projection favors, and saved 17%.
- The live agent differs from the leaderboard run in model version as well as harness. Either way, the calling style that decides the savings belongs to the agent in its harness, as [8] found for trace structure in general, and it can be measured from a few recorded sessions.

**4. Guards pay off where the API does not check the policy.** Every write in τ²-bench's published trajectories was checked against what came before it, as the proxy checks it, and sorted by whether the tool accepted it:

- **Airline.** The enforced rules refuse an accepted write in 141 of 369 failed episodes (38%) and in 5 of 431 successful ones. τ²-bench's own task notes forbid each of the five. The cancellation-eligibility rule alone fires in 116 failed episodes. The airline API does not check eligibility, and its policy says so.
- **Retail.** 1 of 500 failed and 28 of 1,324 successful episodes. Each is a break of the written policy that passed the database check. Retail's tools already enforce most of their own policy.
- **Confirmation.** The confirmation rule, a word list, misses as often in successful episodes as in failed ones, so it is logged, not enforced. Judging a "yes" is a System-One question.
- **Three-valued.** A rule without the facts to decide (a record never looked up) does not refuse.
- **A correction, in time.** After the pilot, the basic-economy rule was changed to refuse finding 2's upgrade path. τ²-bench's own task 32 expects that very path (upgrade a basic-economy reservation, then change its flights), so the change was reverted before the guards ran live.
- **Live, GLM-5.3 gave them nothing to refuse** ([details](https://github.com/alexnodeland/stretto/blob/main/docs/results/pilot-guards-2026-09-24.md)). Arm B ran on the four airline test tasks where the 2025 agents' refused writes concentrate (a guard would have refused a write in 35 of their 50 failed episodes), plus four harm checks:
  - GLM-5.3 passed all four in both arms. On the two forbidden cancellations it declined on its own.
  - With the guards on, 7 of 8 episodes passed. The failure repeated finding 2's cost error, with nothing refused.
  - The proxy checked 9 writes live and passed them all, among them task 32's upgrade-then-change.
  - So guards are insurance whose value depends on the agent, at no cost when they do not fire. 223 Z.ai credits.

**5. A flow is audited as a fugue program.**

- **The program.** A flow's decisions over an episode are a categorical choice per decision site. `stretto audit` builds them as one fugue `Model` and runs it under two handlers:
  - `ScoreGivenTrace` scores the agent's own steps: the log-probability of its path under the flow;
  - `PriorHandler` runs the flow as a stochastic policy.
- **Results.** On GLM-5's published test episodes, answered from the replay cache at no cost:

| | Retail | Airline |
|---|---|---|
| Decisions scored | 724 | 228 |
| The flow's likeliest option was the agent's step | 85.9% | 64.5% |
| Surprise, nats per decision | 0.42 | 1.02 |
| Calibration | underconfident: a top option at 0.8–0.9 is right 96% of the time | overconfident: 74% |

- **Reading them.** The retail figure matches the offline agreement (85.7%, §3.13). The airline one flags finding 3's parallel reads: after a reservation read, the flow agrees 42% of the time. This is the re-validation that §4 asks for when behavior drifts.

What this changes:

- **The Phase 2 gate is judged per agent model, harness and domain** (§3.11). A projection made in another harness can be wrong in either direction.
- **Arms B and E can run, and B has.** The guards exist, are audited, and ran live; they matter for agents that make the writes they refuse, such as the cheaper models §3.11 plans to run on flows compiled from frontier traces. Arm D0 runs through the proxy for any MCP server.
- **Drift has a check (§4):** audit new sessions before trusting a flow compiled from older ones.
- **Next:**
  - a paired run large enough to bound pass^1;
  - `plan_*` macro-tools;
  - counterfactual evaluation from the propensities the proxy logs.


### 3.16 Amendment 5: arm C, the habit alone, and what naming would add (2026-09-24)

Before building `plan_*` (arms C and D), stretto measured what naming a flow could add. It also replayed arm C and a habit-only flow against D0. No LLM ran. The only new spend was $0.078 of Jev questions for the replays, plus $1.11 to ask the v1 questions a second time.

- **Details:** stretto's [arms results](https://github.com/alexnodeland/stretto/blob/main/docs/results/arms-2026-09-24.md) and the [working paper](https://alexnodeland.github.io/stretto/)'s §5.9.

Four findings; §3.5, §3.6, §3.11, §4, §5, §6 and §7 are updated to match.

**1. Naming adds little, so `plan_*` is not built.**

- A flow behind the tools binds every lookup argument that came from an earlier output. A named macro-tool could also make a lookup that needs a value from the conversation, such as a city the customer named or a date the agent works out, if the LLM passed that value.
- Phase 0 now splits the lookups inside runs by where their arguments came from. Across the thirteen agent models:
  - lookups that need the conversation are 0.0–1.9% of LLM turns in retail and 0.9–7.9% in airline (GLM-5: 1.6% and 2.9%);
  - lookups a flow can bind are 6–38%.
- These are upper bounds. They assume the agent adopts the tool (§4) and passes the right values before it has seen the first results.
- The top of the airline range is flight searches, by agents that search many dates: Gemini 3 Flash (7.9%) and Qwen3.5 (7.3%).

**2. Arm C saves almost nothing.**

- GLM-5's published test episodes were replayed through τ²-bench's tools with a flow behind them: all four trials, 160 in retail and 80 in airline.
- One flow goes on only where the habit is at least 0.9 sure of the lookup and its arguments. It saves 1.7% of turns in retail and none in airline. At 0.99 it saves none in either.
- Branches are everywhere. The gain comes from acting on probabilities, which read-only flows make safe (§3.13).

**3. The habit alone saves as many turns as the arbiter.** Same replay and same rule (lookup first at 0.3), but the flow decides by the habit's prediction alone and never asks Jev:

| | Retail | Airline |
|---|---|---|
| D0, the arbiter: turns saved | 21.2% | 6.5% |
| The habit alone: turns saved | 22.5% | 8.7% |
| Difference, with a 95% interval bootstrapped over tasks | +1.2 (−0.5 to +3.0) | +2.2 (+0.6 to +4.2) |
| Detours: arbiter / habit alone | 58 / 58 | 26 / 50 |

- Weighing Jev's answers buys precision, not savings. On GLM-5's episodes it halves the detours in airline and changes nothing in retail.
- Claude 3.7 Sonnet's test episodes, one of the four source agents, show the same pattern:
  - the habit alone saves 1.2 points more in retail (−0.6 to +2.9) and 2.6 more in airline (+1.1 to +4.5);
  - it makes 101 detours against 73 in retail, and 95 against 29 in airline.
- §7's early note that "the habit alone saves 0–2%" held for a habit that acts only where its held-out agreement is 99%. On read-only lookups at 0.3, the habit carries the savings.
- `stretto-proxy --flow-decider habit` serves a flow with no System-One model: no key, and no cost or latency per question.
- D0's retail replay reproduces the one the live pilot was built on (294 turns, 21.2%). Airline saves less than live (17.3%), because GLM-5 in τ²-bench's harness makes several calls per turn (§3.15, finding 3).

**4. The v1 answers are published.**

- The session that ran the first Phase 0b run published its answers, 8,946 of them. They reproduce the v1 report line for line.
- The same questions, asked again of the same model version, got the same pick 97.0% of the time. The report's headline figures moved by at most 0.3 points, and every gate verdict stayed the same. Both sets are published.
- Jev does not answer identically twice. A replay cache, not re-asking, is what makes a Phase 0b result reproducible.

What this changes:

- **The form is D0.** §3.5's named macro-tools are not built. `commit_*` remains, as `stretto_commit`.
- **Arbitration is a precision setting (§3.6).** Read-only flows save turns on the habit alone, and the arbiter trades some of those savings for fewer detours.
- **Jev's place narrows** to what the habit cannot see: content decisions, and deployments with few traces (§6, question 9).
- **Next:**
  - a paired run large enough to bound pass^1;
  - the habit alone, live;
  - where a System-One model earns its place;
  - counterfactual evaluation from the propensities the proxy logs.

### 3.17 Amendment 6: the habit alone live, fewer traces, and judging a confirmation (2026-09-24)

Three follow-ups to §3.16 test where a System-One model earns its place (§6, question 9). The habit-only flow ran live, the habit was trained on fewer traces, and Jev was asked to judge confirmations. The live run cost 109.1 Z.ai credits. Jev's new answers cost $1.24: $1.12 for the fewer-traces sweep and $0.12 for the confirmations. They are published with the arms replays' answers, so every result here replays without a key.

- **Details:** stretto's [habit pilot](https://github.com/alexnodeland/stretto/blob/main/docs/results/pilot-habit-2026-09-24.md), [fewer traces](https://github.com/alexnodeland/stretto/blob/main/docs/results/sweep-2026-09-24.md), [confirmations](https://github.com/alexnodeland/stretto/blob/main/docs/results/confirm-2026-09-24.md) and [answer bundle](https://github.com/alexnodeland/stretto/blob/main/docs/results/answers-2026-09-24-arms-sweep-confirm.md), and the [working paper](https://alexnodeland.github.io/stretto/)'s §5.9–5.11.

Three findings; §3.6, §3.11, §4, §6 and §7 are updated to match.

**1. Live, the habit alone did what D0 did.** GLM-5.3 in Claude Code ran the retail pilot's ten tasks again (§3.14), with the flow deciding on the habit alone:

| | No flow | D0: the arbiter | The habit alone |
|---|---|---|---|
| LLM turns | 110 | 84 (−23.6%) | 79 (−28.2%) |
| Flow lookups (repeated by the agent) | | 26 (3) | 37 (0) |
| Passed the database check | 8 of 10 | 8 of 10 | 9 of 10 |
| Z.ai credits | 128.5 | 108.0 | 109.1 |

- Against D0, the habit alone changed the turns per episode by −0.5 (95% interval −1.4 to +0.4). It asked Jev nothing.
- It ran later the same day than the other two arms, so model drift cannot be ruled out, and ten tasks cannot bound a pass-rate effect.

**2. With few traces, Jev's answers carry the savings.** `--train-fraction` trains the habit, the sites and the bindings on a nested sample of the training tasks. Each flow replayed GLM-5's test episodes as in §3.16, with both deciders at 0.3. The sizes in between are in the results:

| Domain | Training tasks | Successful episodes | The arbiter: turns saved (detours) | The habit alone: turns saved (detours) |
|---|---|---|---|---|
| Retail | 1 | 3 | 10.0% (28) | 4.9% (260) |
| Retail | 3 | 21 | 20.4% (26) | 1.8% (23) |
| Retail | 7 | 67 | 20.2% (38) | 22.4% (63) |
| Retail | 74, all | 831 | 21.2% (58) | 22.5% (58) |
| Airline | 1 | 15 | 0 (0) | 0 (0) |
| Airline | 3 | 28 | 5.6% (1) | 4.1% (37) |
| Airline | 8 | 75 | 6.0% (11) | 8.7% (76) |
| Airline | 30, all | 246 | 6.5% (26) | 8.7% (50) |

- From 7 retail tasks and 8 airline tasks, the habit alone saves what it saves with every task, 1–3 points more than the arbiter. §3.16's finding holds there.
- Below that, the arbiter carries the savings. With 3 retail tasks the difference is −18.6 points (95% interval −21.1 to −16.0). The habit is unsure where agents read the next order and never reads one, where the arbiter reads 394. With one retail task, the habit copies that task's path and makes 260 detours.
- The traces set a flow's reach, because its sites and options come only from training. With one airline task the flow knows two sites, and neither decider saves a turn.
- Two cautions. Only the habit's side shrank: at every size the arbiter was fitted on at least 2,344 held-out retail decisions and 992 airline ones, where a deployment with few sessions has few. And each size is one draw of tasks, so the sweep locates the crossover, between 3 and 7 retail tasks, rather than tracing a curve.

**3. Jev judges a confirmation better than a word list.**

- `stretto confirm` asks Jev one yes/no question per write in τ²-bench's published trajectories. It shows what the agent said last before the customer's last message, that message, and the call about to be made, and asks whether the customer explicitly agreed to this change. 3,863 questions cost $0.12.
- The two judges agree on 94.9% of the 3,484 accepted writes. Of the 178 disagreements, a random 40 (20 of each kind) were labelled blind to both judges. Jev was right on 30, the word list on 10. Weighted by how often each kind occurs, Jev is right on 78%. One annotator labelled them: Claude, which also ran the analysis.
- Jev catches mismatches that no word list can see: the customer agrees to one change, and the agent makes another, such as a refund to a different card or a cancellation instead of a return.
- Most of its errors are too easy a yes, to a change the agent never described first. A second question could catch those.
- Enforced, it would refuse 5–11% of accepted writes in successful episodes. It stays logged until a live run measures what refusing costs.

What this changes:

- **A System-One model's place in read-only flows is the cold start (§3.6).** A deployment compiles its first flows with the arbiter, from a few sessions. The habit takes over the savings as traces accumulate, and the arbiter then buys precision.
- **The explicit yes before a write is a System-One judgment (§3.5, §3.15).** Jev's judgment replaces the word list as the confirmation check to take live. It is logged, not enforced.
- **Question 9 (§6) is answered in part.** Matching a description to a record is still untested, and so is a cold start run live with the arbiter fitted on a deployment's own few sessions.
- **Next:**
  - a paired run large enough to bound pass^1;
  - a flow learned from a few sessions, run live;
  - options from the tool manifest as well as from traces, so Jev can act where the traces are silent;
  - the confirmation judge enforced, with a second question;
  - counterfactual evaluation from the propensities the proxy logs.

---

### 3.18 Amendment 7: a cold start from an agent's own sessions, options from the manifest, and two content roles (2026-09-24)

Four follow-ups to §3.17, each on §6's question 9: where does a System-One model earn its place? A flow was learned from an agent's own first sessions, offline and live. Flows were offered every read-only tool in the manifest. The confirmation judge got a second question. And Jev was asked to match descriptions to records. The live runs cost 182 Z.ai credits of the 200 approved. Jev's 25,981 new answers cost $2.33: $0.62 for the cold start, $1.17 for options from the manifest, $0.25 for the confirmation's second question and $0.29 for matching. They are published, so every result here replays without a key.

- **Details:** stretto's [cold start](https://github.com/alexnodeland/stretto/blob/main/docs/results/cold-start-2026-09-24.md), [options from the manifest](https://github.com/alexnodeland/stretto/blob/main/docs/results/manifest-options-2026-09-24.md), [second confirmation question](https://github.com/alexnodeland/stretto/blob/main/docs/results/confirm-second-2026-09-24.md), [matching](https://github.com/alexnodeland/stretto/blob/main/docs/results/matching-2026-09-24.md) and [answer bundle](https://github.com/alexnodeland/stretto/blob/main/docs/results/answers-2026-09-24-cold-manifest-match.md), and the [working paper](https://alexnodeland.github.io/stretto/)'s §5.10–5.14.

Five findings, one of them a correction; §3.6, §3.11, §4, §6 and §7 are updated to match.

**1. A cold start needs no arbiter of its own.** `stretto learn` holds out 30% of a deployment's sessions, by task, and fits one arbiter on Jev's answers there. Offline, flows learned from GLM-5's own published sessions (trial 0 of each training task drawn) replayed its test episodes as in §3.17:

| Domain | Sessions | The arbiter flow: turns saved (detours) | The habit alone: turns saved (detours) |
|---|---|---|---|
| Retail | 5, four random draws | 6.5–16.5% | 11.8–18.9% |
| Retail | 5, clustered by task id | 7.9% (16) | 2.7% (16) |
| Retail | 10 | 11.6% (25) | 20.9% (51) |
| Retail | 20 | 22.6% (31) | 20.4% (34) |
| Retail | 40 | 23.9% (160) | 22.4% (58) |
| Retail | 74, all | 22.3% (47) | 22.4% (51) |
| Airline | 5, 10 and 20 | 4.3–6.3% | 8.7% |
| Airline | 30, all | 6.2% (39) | 8.7% (85) |

- The arbiter flow trails the habit alone until about twenty sessions, and in airline at every size tried. The split takes 30% of the sessions from the habit, the sites and the bindings, and eight weights fitted on 3–11 decisions land anywhere: across the five-session draws, the weight on the habit ran from −0.32 to 0.30.
- Which sessions a flow starts from matters more than how it decides. From five retail sessions the habit alone saved 11.8–18.9% on the four random draws, and 2.7% on one clustered by task id, like §3.17's samples.
- Two repairs were tried. Refitting the habit on every session once the arbiter is fitted (`--refit-habit`) was no remedy: its arbiter met sites it never saw fitted, and on one draw made 355 detours. Taking the arbiter of the all-task flow, fitted on four other agents' 4,513 decisions (`--arbiter-from`), lifted the clustered draw from 2.7% to 11.1%, and changed the random ones by −1.9 to +0.1 points.
- Live, GLM-5.3 in Claude Code recorded five retail training sessions through the proxy, and `stretto learn --sessions` learned a flow from them. Replayed on GLM-5's 40 trial-0 test episodes, it saved 19.0% of turns, against D0's 22.2%. On three of the pilot's tasks it took 12 LLM turns where the agent alone took 19, 14 where it took 16, and 26 where it took 14: on that task the agent failed as before, and the simulated customer kept talking after a transfer. The harness now ends an episode at a transfer, as τ²-bench's customer is told to. Three tasks show that such a flow runs live, not what it saves.

**2. Correction to §3.17: its samples were clustered.** `--train-fraction` ranked tasks by an FNV-1a hash that barely mixes an id's last characters, so neighbouring ids sorted together. The three retail tasks were 104, 105 and 109, and the seven were 103–107, 109 and 113. Neighbouring τ²-bench tasks often share a customer and a kind of request, so those samples are narrower than random ones. §3.17's numbers hold for the tasks drawn, but its crossover rests on one clustered draw. The sampler now mixes its order first, and `--train-tasks` names a sample exactly, so each of the sweep's flows can be rebuilt.

**3. Options from the manifest add nothing; bindings are the limit.** `--manifest-options` offers every read-only tool at every site, so Jev can choose a lookup no trace showed. The flow binds such a lookup by argument name. On the sweep's one- and three-task samples:

- In retail it saves what it saved before (10.0% and 20.3% of turns), with up to twice the detours. The arbiter keeps to the lookups the habit knows.
- In airline the fitted arbiter gives the habit almost no weight, so Jev's picks bring in flight searches no trace showed. Bound by name, a search takes its origin, destination and date from the flights just returned or from the reservation, not from the trip the customer wants. Nearly every one is a detour: 83 and 91, against 0 and 1 without. With one task the flow saves 1.1% of turns where it saved nothing; with three, 3.7% against 5.6%.
- Only traces teach a flow how to fill a lookup's arguments. The option stays off by default.

**4. A second confirmation question catches the costly lapses, and half its flags are false.** Asked on its own about the same three fields, a second yes/no question fails a write unless both answers are yes.

- The first wording, "did the agent's message describe this exact change?", was too literal. It failed 18% of the accepted writes in successful retail episodes, and 3 of 30 random writes it newly failed were lapses.
- The second, "had the agent proposed this change?", fails 113 more accepted writes: 6.6% in successful retail episodes, 13.4% in failed ones. Of 30 labelled blind, 16 were lapses (53%, 95% interval 36–70%).
- Seven of those are calls that differ from what the customer agreed to, such as a whole order cancelled where three items were to go, or a refund to a gift card instead of the Mastercard the customer named.
- It is logged, not enforced. The labels come from one annotator, the model that ran the analysis, and a write labelled confirmed in §3.17 was labelled a lapse this time, so both readings are reported.

**5. Matching a description to a record is not a check.** `stretto match` asks Jev which item, variant, payment method or reservation the customer means, from the customer's words and the candidates alone. Over 3,978 such choices, Jev picked the record the task expects 81.6% of the time, against 89.9% for the agents. Where the two differed, the agent was wrong only 19% of the time, and Jev's confident disagreements catch 25 of the agents' 402 wrong picks. The customer's words alone are often ambiguous, and showing Jev the agent's messages would let it copy the agent's pick.

What this changes:

- **A System-One model's place is not a deployment's own cold start (§3.6).** §3.17's reading held for clustered samples scored by an arbiter fitted on thousands of other agents' decisions. A deployment starts on the habit alone, can carry an arbiter fitted elsewhere as insurance against a narrow start, and fits its own once it has some twenty sessions.
- **Its clearest role so far is judging what the habit cannot see.** The confirmation judge, with the second question logged, is the one to take live. Matching a description to a record is not a check the agents need.
- **Options stay with the traces.** A lookup no trace showed needs bindings, and only traces teach those.
- **Question 9 (§6) is answered for now,** except for enforcement: what refusing an unconfirmed write costs is for a live run.
- **Next:**
  - a paired run large enough to bound pass^1;
  - the cold start live on more tasks, with an arbiter fitted on public traces shipped with the compiler;
  - the confirmation judge enforced;
  - counterfactual evaluation from the propensities the proxy logs.

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
  - Logged propensities and a little exploration mitigate this; they do not remove it. *Amended (§3.15):* the proxy logs every live decision's probabilities.
- **Macro-tools might go unused.** Agents given a world model as a tool used it less than 1% of the time [Qian 2026]. Adoption is a measured outcome, not an assumption. If it is low, we add a reference agent loop for the experiments. *Amended (§3.14):* the first live form adds no tools. What must be measured instead is whether the agent repeats the flow's lookups. *Amended (§3.16):* named macro-tools are not built, so the risk does not arise.
- **Vendor risk.** Jev is proprietary and in early access. It offers no fine-tuning, and there are no public accuracy benchmarks against human labels. Hence the `Oracle` trait and the replay cache. *Amended (§3.16):* reduced. Read-only flows save as many turns without a System-One model. *Amended (§3.17):* once there are enough traces. A cold start still needs one. *Amended (§3.18):* a deployment's own cold start does not either; an arbiter fitted elsewhere is insurance against a narrow one.
- **Evaluation is hard (§3.7).**
  - Simulation is optimistic and counterfactual estimates are high-variance.
  - Guarantees in the style of ProbGuard's PAC bounds call for 530 to 10⁵ traces [3].
  - Canaries cost traffic.
- **Drift.** A new LLM version, prompt or tool changes behavior. Compiled flows must be re-validated, not trusted forever. *Amended (§3.15):* `stretto audit` scores new sessions under a flow, per site and per episode. So does a new harness: the same model family called tools very differently in Claude Code and in τ²-bench's harness.
- **Scope.** This is a new product surface beside a PPL that is still pre-1.0 and has a single maintainer.

---

## 5. Alternatives considered

- **Use Jev only as middleware** (LangChain's `ModelRouterMiddleware` and `AutoModeMiddleware`).
  - It is cheap, useful and orthogonal: it routes calls and gates risk.
  - But it learns nothing and compiles nothing.
- **Skip single steps the way AutoTool does** [13].
  - It is proven to cut LLM calls by up to 30%.
  - But it works one step at a time on an uncalibrated score, with no outcome model and no way to verify anything. We use it as a comparison point, not as a design.
- **Compile deterministic workflows the way TraceCompiler does** [24]. This is arm C. It shows how much of the gain comes from structure alone, before any calibrated branch resolution. *Measured (§3.16):* replayed as a flow that hands every branch back, it saves 0–1.7% of turns, where acting on probabilities saves 21–22% in retail.
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

1. ~~**Phase 0 data.**~~ Resolved: published trajectories, including Sierra's leaderboard runs.
2. ~~**Models and budget.**~~ Resolved: transfer first, run by GLM and MiniMax, at minimum cost (§3.11).
3. ~~**The repo.**~~ Resolved: [stretto](https://github.com/alexnodeland/stretto), public, MIT.
4. **How are compiled flows reviewed?** Probably as code, with the flow IR diffed in pull requests. Partly answered (§3.15): the IR is versioned JSON that records its provenance, and `stretto audit` scores a flow on new sessions before it is trusted.
5. **Privacy for non-benchmark workloads.** Traces contain user data. The store should keep hashes and state slices, with retention limits.
6. **What should a System-One decision be scored against?**
   - Agreement with the agent undercounts valid alternatives, such as looking up two orders in either order.
   - Replayed outcomes can't settle this, so only live pass^1 can.
   - With read-only flows the question narrows: does a detour (an extra lookup before handing back) ever cost pass^1? (§3.13)
   - First live evidence (§3.14): 8 of 10 episodes passed the database check without the flow and 8 of 10 with it. That is too few to bound a one-point loss.
   - Airline (§3.15): 8 of 10 without the flow and 9 of 10 with it, and no failure came from a flow decision. Twenty pairs are still too few.
7. ~~**Can System-One questions reach roughly 99% agreement on the decisions flows need?**~~ Answered for now (§3.13):
   - No. The best combination reaches 79–81%, and at least 0.99 sure only on 11–25% of decisions.
   - Read-only flows don't need it: acting on weaker picks costs detours, not risk.
8. ~~**Which agent model runs the first live pilot?**~~ Resolved (§3.14): GLM-5.3, the model the Z.ai coding-plan key serves.
   - Offline, Qwen3.5 clears the gate in both domains, and Qwen3-Max does too (airline with lookup first). They are next once a key for them is at hand.
   - GLM-5 clears it in retail only, with the state predicates, with or without the goal.
9. **Where does a System-One model earn its place?** (§3.16, §3.17, §3.18)
   - For read-only flows with enough traces, the habit alone saves as many turns, live too. Jev's weighed answers buy precision: half to two-thirds fewer detours in airline.
   - With few traces, on §3.17's clustered samples and an arbiter fitted on thousands of other agents' decisions, they carried the savings. From a deployment's own first sessions they did not: the habit alone saved more than an arbiter fitted on those sessions until about twenty (§3.18). An arbiter fitted elsewhere lifted a narrow start.
   - Judging a confirmation before a write: where Jev and the word list disagree, hand labels side with Jev on 30 of 40. A second question catches calls that differ from what the customer agreed to, and half its flags are false (§3.18). Enforcing it is untested.
   - Matching a description to a record: no. Jev picked the expected record less often than the agents did (§3.18).

---

## 7. Decision

- **Outcome:** Accepted on 2026-09-23 by @alexnodeland (merged in [#51](https://github.com/alexnodeland/fugue/pull/51)).
- **Notes:**
  - Implementation proceeds in [stretto](https://github.com/alexnodeland/stretto).
  - The fugue changes in §3.9 land as their own PRs.
  - Phase 0 results so far:
    - macro-tool headroom is 23–25% of LLM turns;
    - the habit alone saves 0–2%;
    - a System-One model that always agrees with the agent would save 20.5–22%.

    So the next step, Phase 0b, measures Jev's actual agreement and calibration inside flows, and projects tokens and dollars alongside turns.
  - Amended the same day (§3.12), after Phase 0b:
    - the gate is judged per agent model and domain;
    - context validation counts distinct tasks;
    - flows are compiled from the 2026 frontier runs;
    - zero-shot Jev, as asked in v1, saves only 4–5% of turns at acceptable risk. The next iterations target the question design and arbitration that §3.5–3.6 specify.
  - Amended again the same day (§3.13), after Phase 0b v2:
    - flows only read between LLM turns, so offline the gate is turns saved, and detours are for the live pilot to clear;
    - Jev's answers are combined with the habit, its per-site record and state predicates;
    - offline, Qwen3.5 clears the gate in both domains (Qwen3-Max too, with lookup first in airline); GLM-5 only in retail.
  - Amended on 2026-09-24 (§3.14), after the first live runs:
    - the goal adds nothing to read-only flows, so the first live form runs behind the agent's own calls, with no new tools (arm D0);
    - live lookups act on the tool's probability times the binding's agreement in training;
    - replayed through the live flow, GLM-5's recorded test episodes save 21.2% of LLM turns (20.3% projected);
    - live, on ten paired retail tasks, GLM-5.3 took 24% fewer LLM turns with the flow (8 and 8 of 10 passed the database check).
  - Amended later on 2026-09-24 (§3.15):
    - the first design is built: a flow IR; the proxy serving flows, guards and a commit tool for any MCP server; learning from recorded sessions; and an audit that runs a flow as a fugue program;
    - live, on ten paired airline tasks, GLM-5.3 took 17% fewer LLM turns with the flow (8 and 9 of 10 passed);
    - the savings depend on the agent's harness as well as its model, so the gate is judged per model, harness and domain;
    - guards refuse a policy-breaking write in 38% of failed airline episodes and 1% of successful ones, each of those five a violation τ²-bench's own notes name; live, GLM-5.3 gave them nothing to refuse, and they did no harm.
  - Amended again on 2026-09-24 (§3.16), before building macro-tools:
    - naming a flow could add at most 1.9% of LLM turns in retail and 7.9% in airline over D0, so `plan_*` is not built;
    - arm C, a flow that hands every branch back, saves 0–1.7% of turns;
    - the habit alone saves as many turns as the arbiter, and the arbiter cuts detours in airline by half to two-thirds, so arbitration is a precision setting and a flow can run without Jev;
    - the v1 answers are published and reproduce the v1 report exactly; asked a second time, Jev keeps the same pick 97% of the time.
  - Amended again on 2026-09-24 (§3.17), testing where a System-One model earns its place:
    - live, on the retail pilot's ten tasks, the habit-only flow took 79 LLM turns, against D0's 84 and 110 with no flow, and passed 9 of 10;
    - with few traces, Jev's answers carry the savings: trained on three retail tasks, the flow saves 20.4% of turns with the arbiter and 1.8% on the habit alone, and from seven tasks on the habit alone saves as many;
    - Jev judges a customer's confirmation better than the guards' word list, 30 of 40 on hand-labelled disagreements; it is logged, not enforced.
  - Amended again on 2026-09-24 (§3.18), from a deployment's own first sessions and in two content roles:
    - from five of an agent's own sessions, the habit alone saved more than an arbiter fitted on them, which pays from about twenty sessions; so a deployment starts on the habit alone, and §3.17's cold-start finding holds only for its clustered samples;
    - a flow learned from five sessions GLM-5.3 recorded through the proxy ran live on three pilot tasks;
    - options from the tool manifest add nothing, because a lookup no trace showed cannot be bound;
    - a second confirmation question catches the costliest lapses with half its flags false, and matching descriptions to records is no check; both stay logged.

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
