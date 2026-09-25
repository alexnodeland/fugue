# RFC-001: Habit compiler — compiling agent behavior into System-One flows

- **Status:** Accepted (2026-09-23, [#51](https://github.com/alexnodeland/fugue/pull/51)). Amended eight times on 2026-09-23 and 2026-09-24 as stretto's results came in (§3.12–§3.19, [#52](https://github.com/alexnodeland/fugue/pull/52)–[#60](https://github.com/alexnodeland/fugue/pull/60)). On 2026-09-25 the design and its amendments moved to stretto, as [stretto's RFC-001](https://github.com/alexnodeland/stretto/blob/main/docs/rfc/001-habit-compiler.md), where it is amended from now on ([#68](https://github.com/alexnodeland/fugue/pull/68)). This page keeps fugue's side of it: why fugue fits (§3.2), what fugue decided about its own scope and the six changes to `fugue-ppl` (§3.9), the spike (Appendix A) and the decision record (§7). Section numbers follow the full RFC.
- **Authors:** @alexnodeland (drafted with Claude Code)
- **Created:** 2026-09-23
- **Updated:** 2026-09-25
- **Supersedes / Related:**
  - the full RFC, with the design, the phased plan, the τ²-bench experiment and all eight amendments: [stretto's `docs/rfc/001-habit-compiler.md`](https://github.com/alexnodeland/stretto/blob/main/docs/rfc/001-habit-compiler.md);
  - the implementation, [**stretto**](https://github.com/alexnodeland/stretto), with every result indexed in [`docs/results/`](https://github.com/alexnodeland/stretto/blob/main/docs/results/README.md);
  - runnable spike in [`001-habit-compiler/spike/`](001-habit-compiler/spike/);
  - the six changes to `fugue-ppl`: [#61](https://github.com/alexnodeland/fugue/issues/61).

---

## 1. Summary

LLM agents make every decision with a full model call, even a choice they have made the same way a thousand times before. The RFC proposes a harness plugin that watches an agent's tool calls, keeps a Bayesian model of its behavior, and compiles the predictable parts into typed probabilistic programs ("flows"). At each branch point a flow asks the cheapest resolver the posterior says is good enough: the learned habit, a System-One model such as TypeSafe's Jev, the LLM, or a person.

Fugue supplies the representation. A flow is a `Model`, its branch points are addressed sites, and the harness is a `Handler`. So one flow can be simulated, executed, audited against recorded traces and evaluated counterfactually, just by swapping the interpreter. A spike ([Appendix A](#appendix-a-the-spike)) does all four on fugue 0.2.3 with no changes to the library, using mocks for the tools, the LLM and Jev.

The implementation is stretto, a separate repo, as fugue-evo is. Its design, experiments and results are in [stretto's copy of this RFC](https://github.com/alexnodeland/stretto/blob/main/docs/rfc/001-habit-compiler.md).

---

## 3. Proposed solution: fugue's side

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

### 3.9 Where it lives

The implementation lives in a new repo, **stretto**, like fugue-evo does. In a fugue, a stretto is where entries of the subject overlap and compress: here, many traces compress into one flow.

The proxy brings dependencies fugue core should not carry: an async runtime, an MCP SDK, HTTP clients and benchmark glue.

Fugue core also stays oracle-agnostic (the full RFC's §2.4): Jev is the first implementation of an `Oracle` trait, in stretto, not a dependency of fugue core. stretto's crates are listed in [its copy of this section](https://github.com/alexnodeland/stretto/blob/main/docs/rfc/001-habit-compiler.md#39-where-it-lives).

The spike surfaced six changes to `fugue-ppl`. All are additive, and they will go to this repo as their own PRs:

1. **Async interpretation** ([#62](https://github.com/alexnodeland/fugue/issues/62)).
   - Today `run` is a synchronous trampoline, but tool calls and Jev calls are network I/O.
   - Add `run_async` over the same `Model`, driven by an `AsyncHandler`.
   - The model is data, so this is a second trampoline, not a rewrite.
2. **Site metadata for handlers** ([#63](https://github.com/alexnodeland/fugue/issues/63)).
   - A handler sees only `(addr, dist)`. The spike's handler recovers the prior from `dist.log_prob`, but a Jev question also needs the question spec and a state slice.
   - Options: an address-keyed registry, or an optional `fn as_any(&self) -> Option<&dyn Any>` on `Distribution` that defaults to `None`.
3. **Less handler boilerplate** ([#64](https://github.com/alexnodeland/fugue/issues/64)).
   - A handler must implement every `on_sample_*` and `on_observe_*` method.
   - A delegating adapter would fix this: override the sites you care about and defer the rest to an inner handler.
4. **Flow IR** ([#65](https://github.com/alexnodeland/fugue/issues/65)). Generalize fugue-wasm's `dsl.rs` interpreter into a serializable program format that builds `Model`s at runtime.
5. **Vector-valued sites** ([#66](https://github.com/alexnodeland/fugue/issues/66)).
   - There is no `Dirichlet` or `Multinomial`, and sites are scalar.
   - Conjugate helpers are enough for the world model.
   - A Gamma-normalization helper, or vector sites, would let the Bayesian model live inside fugue proper.
6. **Documented pattern: sample-or-observe sites** ([#67](https://github.com/alexnodeland/fugue/issues/67)).
   - The same site is sampled when simulating and scored as an observation when executing.
   - This is a legitimate choice for a handler to make, and the spike does it.
   - It deserves a how-to page.

*Status (2026-09-25):* all six are built, each in its own PR, tracked in [#61](https://github.com/alexnodeland/fugue/issues/61):

1. async interpretation, [#71](https://github.com/alexnodeland/fugue/pull/71);
2. site metadata, [#70](https://github.com/alexnodeland/fugue/pull/70);
3. the delegating adapter, [#69](https://github.com/alexnodeland/fugue/pull/69);
4. the program format, [#74](https://github.com/alexnodeland/fugue/pull/74);
5. the conjugate helpers, [#72](https://github.com/alexnodeland/fugue/pull/72);
6. the how-to, [#73](https://github.com/alexnodeland/fugue/pull/73).

A live flow is now a fugue program, as the full RFC's §3.5 intends ([stretto 449c6b3](https://github.com/alexnodeland/stretto/commit/449c6b3c33befdd533eedd52c263e1a86b5cf51d)):

- **Stored.** A flow's file holds its run in the program format (item 4). stretto registers the flow's own distributions for it, and they carry their site as metadata (item 2).
- **Executed.** `stretto-proxy` interprets the program with `run_async` (item 1).
- **Audited.** The proxy logs each run's trace, which `ScoreGivenTrace` scores again under the same program.

Offline, stretto uses fugue as before. `stretto audit` simulates a flow with `PriorHandler` and scores recorded episodes with `ScoreGivenTrace`. The habit's back-off concentration is sampled with adaptive Metropolis–Hastings.

---

## 7. Decision

- **Outcome:** Accepted on 2026-09-23 by @alexnodeland (merged in [#51](https://github.com/alexnodeland/fugue/pull/51)).
- **Notes:**
  - Implementation proceeds in [stretto](https://github.com/alexnodeland/stretto).
  - The fugue changes in §3.9 land as their own PRs, tracked in [#61](https://github.com/alexnodeland/fugue/issues/61). All six landed on 2026-09-25, in [#69](https://github.com/alexnodeland/fugue/pull/69)–[#74](https://github.com/alexnodeland/fugue/pull/74).
  - Amended eight times as stretto's results came in (§3.12–§3.19, [#52](https://github.com/alexnodeland/fugue/pull/52)–[#60](https://github.com/alexnodeland/fugue/pull/60)). The amendments' findings are summarized in [stretto's copy](https://github.com/alexnodeland/stretto/blob/main/docs/rfc/001-habit-compiler.md#7-decision).
  - Moved on 2026-09-25 ([#68](https://github.com/alexnodeland/fugue/pull/68)): the design and its amendments went to stretto, which is where the RFC is amended from now on. This page keeps fugue's side.

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

References for the full RFC are in [stretto's copy](https://github.com/alexnodeland/stretto/blob/main/docs/rfc/001-habit-compiler.md#appendix-b-references).
