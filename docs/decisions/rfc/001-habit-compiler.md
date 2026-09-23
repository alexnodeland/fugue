# RFC-001: Habit compiler — compiling agent behavior into System-One flows

- **Status:** Draft
- **Authors:** @alexnodeland (drafted with Claude Code)
- **Created:** 2026-09-23
- **Updated:** 2026-09-23
- **Supersedes / Related:** runnable spike in [`001-habit-compiler/spike/`](001-habit-compiler/spike/); TypeSafe AI's Jev (released 2026-09-15)

---

## 1. Summary

LLM agents make every decision with a full model call, even a choice they have made the same way a thousand times before. TypeSafe's Jev makes a typed decision in about 100 ms for a tiny fraction of a cent, but only inside a program someone has already written. TypeSafe's own guidance is that "code handles deterministic work and owns the control flow". Nobody writes those programs from what agents actually do.

This RFC proposes a harness plugin to close that gap. The plugin:

- watches an agent's tool calls (MCP or function calls) as discrete actions;
- keeps a Bayesian model of the agent's behavior over the tools currently exposed;
- compiles the predictable parts into typed probabilistic programs ("flows").

At each branch point a flow asks the cheapest resolver the posterior says is good enough. In order, those are the learned habit, a System-One model, the LLM, or a person.

Fugue supplies the representation. A flow is a `Model`, its branch points are addressed sites, and the harness is a `Handler`. So one flow can be simulated, executed, audited against recorded traces and evaluated counterfactually, just by swapping the interpreter. A spike ([Appendix A](#appendix-a-the-spike)) does all four on fugue 0.2.3 with no changes to the library, using mocks for the tools, the LLM and Jev.

---

## 2. Context / Motivation

### 2.1 What Jev is, and what it is not

- **A "System One" model.** It takes *state* plus typed questions and returns typed answers with probabilities. It never generates text. There are three primitives:
  - `Choice`: up to 255 options; returns per-option probabilities and a confidence.
  - `Score`: 2–10 ordered levels.
  - `Noul`: the probability that a statement is true.
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

1. **Agent traces contain implicit programs.** Many production agent workloads (triage, runbooks, CI repair, ticket handling) walk the same few paths through their tool space.
2. **System-One models make branch decisions cheap,** but only inside a program with typed branch points.
3. **Nobody derives those programs from behavior.** Nearby work does something else with the traces (§2.3):
   - workflow induction extracts routines for an LLM to reuse;
   - process mining describes flows;
   - runtime verification learns models of agents in order to monitor them.

The missing piece is a compiler from (1) to (2), and a probabilistic programming language (PPL) is the natural home for what it produces.

### 2.3 Related work

_Filled in from the literature scan; see Appendix B for links._

<!-- RELATED-WORK -->

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

The agent's tool calls pass through the harness, which records them and executes them. Recorded traces feed the world model and the compiler. Compiled flows run in the flow runtime, which resolves each decision site with the habit, Jev, the LLM or a person.

```text
agent ──tool calls──► harness (MCP proxy / hooks) ──────────────► tools
                         │ record                      ▲ execute
                         ▼                             │
                   trace store ──► world model ──► compiler ──► flows (fugue Models)
                                        ▲                          │
                                        └──── outcomes ◄── runtime (fugue Handler)
                                                                   │ each decision site
                                                                   ▼
                                                   habit │ Jev │ LLM │ human
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

The world model is only as good as its state abstraction. In the spike, a single hidden bit (whether the logs located the bug) made the model believe the greedy habit succeeds 99.9% of the time. In the environment it succeeds 94.5% of the time. The refinement loop:

1. **Detect aliasing.** Look for:
   - contexts whose next-action distribution stays high-entropy;
   - transitions with high surprise;
   - gaps between simulated and real outcomes on executed flows.
2. **Propose predicates.** An LLM reads examples from the aliased context and proposes `Noul`/`Choice` questions about the raw state, for example "Do the logs name a specific file?". This is TypeSafe's own "autoresearch feature discovery" cookbook pattern, pointed at the world model instead of at a regressor.
3. **Evaluate cheaply.** Jev answers each candidate question over the stored traces. At Jev's pricing that costs cents per thousand traces.
4. **Keep what explains the data.** Keep a predicate if it raises the marginal likelihood of the trace corpus under the world model. This is closed form for the Dirichlet model and SMC evidence for latent-variable variants. Bayesian model selection supplies the Occam penalty.

This is counterexample-guided abstraction refinement (CEGAR), with a System-One model as the predicate evaluator. It is the part of the system that learns.

### 3.5 Compiling flows

- **Candidate regions** are sub-graphs of the world model with:
  - high visitation;
  - low conditional entropy given the available predicates;
  - high downstream success;
  - bounded stakes.
- **Decision sites** are the points where a region branches. Each one gets a question spec made of:
  - the options, which are the successor abstract actions;
  - instructions derived from the LLM's own rationales in the traces;
  - a *state slice*: the minimal fields that predicted the branch. Jev's accuracy drops with irrelevant state, so the slice matters.
- **Arguments** are handled according to the tool's JSON Schema:
  - `enum` → `Choice`; `boolean` → `Noul`; an optional argument → a "was it stated?" `Noul`. TypeSafe's function-calling cookbook does exactly this.
  - A free-text argument becomes a dataflow binding from an earlier output, when the traces show the value the LLM used appearing verbatim in a prior observation.
  - Otherwise it becomes a narrow LLM slot. If neither works, the region is not compiled.
- **Output is a serializable flow IR** (sites, questions, bindings, guards). It is interpreted into a fugue `Model` at load time. fugue-wasm's `dsl.rs` already interprets a `prob!` subset into real `Model`s at runtime; the flow IR generalizes that.
- **Macro-tools (optional).** A compiled flow can be re-exposed to the agent as a new MCP tool, so the agent calls a twelve-step routine as one action. In options-framework terms:
  - the initiation set is the applicability predicate;
  - the intra-option policy is the flow;
  - termination is completion or escalation.

### 3.6 Execution: arbitration, not pooling

At each decision site the runtime picks the cheapest resolver whose expected loss is acceptable:

1. **Habit.** The world model's posterior predictive. Free and instant. Used when its top option is confident *and* is backed by enough evidence.
2. **System-One oracle (Jev).** Consulted only where the habit is unsure. Its answer is treated as an observation from a sensor with a per-site confusion matrix learned from outcomes (Dawid–Skene style). That also repairs the fact that Jev's per-question probabilities are not jointly coherent.
3. **LLM.** When both of the above are unsure, or when the site's stakes are high.
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
- **Watches surprise.** It tracks surprise per step. When surprise exceeds the site's threshold it ends the flow and hands the partial trace back to the LLM as context.
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

That bounds the blast radius but is not immunity, because Jev "does not treat state as hostile by default". So:

- keep untrusted tool output out of the state slices of high-stakes sites whenever trusted fields can decide the branch;
- attach provenance to state slices, and raise the confidence bar when untrusted content is present;
- never let a flow call a tool outside its compiled set.

### 3.9 Where it lives

A new crate, `fugue-flow`, depending on `fugue-ppl`:

| Module | Contents |
|---|---|
| `ingest` | Trace readers: MCP JSON-RPC (proxy), Claude Code hook events, OpenTelemetry GenAI spans |
| `model` | Abstraction, Dirichlet/Beta world model, back-off, forgetting, evidence |
| `refine` | Predicate proposal and evaluation loop |
| `compile` | Region mining, question specs from JSON Schema, flow IR, IR → `Model` |
| `runtime` | Async `Handler`, arbitration, escalation, propensity logging |
| `oracle` | `Oracle` trait; `JevOracle` behind a feature flag; mock and replay oracles |
| `audit` | Conformance, surprise, per-site calibration, counterfactual evaluation |

The spike surfaced six changes to `fugue-ppl`, all additive:

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
| 0. Measure | Recorder; shadow-mode Jev at every LLM tool choice; Dirichlet world model; per-context report | None |
| 1. Audit | Conformance, surprise, drift, process map (no behavior change) | Useful on its own |
| 2. Compile and run | Flow IR, arbitration runtime, canary promotion | Phase 0 shows ≥ X% of LLM decisions compilable at ≤ Y pp success loss on a real workload |
| 3. Learn | Predicate refinement, per-site counterfactual evaluation, macro-tools, flow search with fugue-evo | Phase 2 is live on one workload |

The Phase 0 report gives, for each decision context:

- the number of visits;
- the habit's top-option posterior, with a credible interval;
- Jev's agreement with the LLM, and its calibration;
- downstream success;
- the share of LLM calls avoidable at a target error rate;
- projected cost and latency.

---

## 4. Drawbacks

- **Little to compile in open-ended work.** Coding agents on novel tasks may be mostly long tail. Phase 0 exists to find this out cheaply.
- **Arguments are the hard part.** Most tool arguments are free text. When bindings cannot be recovered from earlier outputs, the region stays with the LLM.
- **Vendor risk.** Jev is proprietary and in early access. It offers no fine-tuning, and there are no public accuracy benchmarks against human labels. Hence the `Oracle` trait.
- **Evaluation is hard (§3.7).** Simulation is optimistic, counterfactual estimates are high-variance, and canaries cost traffic.
- **Drift.** A new LLM version, prompt or tool changes behavior. Compiled flows must be re-validated, not trusted forever.
- **Scope.** This is a new product surface beside a PPL that is still pre-1.0 and has a single maintainer.

---

## 5. Alternatives considered

- **Use Jev only as middleware** (LangChain's `ModelRouterMiddleware` and `AutoModeMiddleware`).
  - It is cheap, useful and orthogonal: it routes calls and gates risk.
  - But it learns nothing and compiles nothing.
- **Distill the agent into a small policy model.** This is a strong baseline for cost, but:
  - it gives no typed decision points and no per-decision probabilities to audit;
  - it has no structure to verify;
  - it needs retraining on every drift.
- **LLM-based workflow induction.**
  - It extracts routines as text or code for the LLM to reuse.
  - Decisions stay with the LLM, and there is no calibrated fallback.
- **Process mining plus hand-written flows.** This is descriptive: a person still writes the program, and there is no uncertainty.
- **Plain counting instead of a PPL.**
  - Counting is fine for Phase 0, and we should use closed forms wherever they exist.
  - It stops being enough once we need any of these:
    - per-site oracle reliability when the true answer is latent;
    - latent task phases;
    - abstraction selection by evidence;
    - decision-theoretic escalation.

---

## 6. Unresolved questions

1. **Which workload for Phase 0?** It needs repetition, real stakes and accessible traces.
2. **Where should the harness sit first?** The options:
   - Claude Code hooks: easy capture, limited control.
   - An MCP proxy: framework-agnostic, full control.
   - LangChain or Vercel middleware: where Jev already integrates.
3. **How are compiled flows reviewed?** Probably as code, with the flow IR diffed in pull requests.
4. **Where does the code live?** Should `fugue-flow` be in this repo, or in its own like fugue-evo?
5. **Privacy.** Traces contain user data. The store should keep hashes and state slices, with retention limits.
6. **What are X and Y** in the Phase 2 gate?

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

<!-- REFERENCES -->
