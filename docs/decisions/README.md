# 🚦 Proposal → Decision Workflow

## 🔍 Process Overview

### 1. **Stage 1: Idea / Proposal (RFC)**

* Contributor writes an RFC in `docs/rfcs/`.
* Open a Pull Request titled:

```text
RFC-XXX: <short descriptive title>
```

* Add label: `rfc/draft`
* PR is used for discussion and iteration.
* When the team accepts it → merge PR with label `rfc/accepted`.

**Outcome:**
RFC file lives in `/docs/rfcs/`, marked as **Accepted**.

---

### 2. **Stage 2: Implementation Tracking (Issue)**

* Create a GitHub Issue for implementing the accepted RFC.
* Title:

```text
Implement RFC-XXX: <title>
```

* Link RFC in description.
* Labels: `implementation`, `rfc/accepted`.
* Assign owners, set milestones, etc.

**Outcome:**
Code changes are tracked in Issues and PRs (linked back to RFC).

---

### 3. **Stage 3: Decision Record (ADR)**

* Once implementation is complete:

  * Create a new ADR in `docs/adrs/` summarizing the final decision.
  * Mark the originating RFC as `Implemented`.
* ADR should be concise, link back to RFC and Issues.

**Outcome:**
You now have:

* RFC (exploration + rationale).
* Issue(s) (execution tracking).
* ADR (permanent decision log).

---

## 🏷️ GitHub Labels

* `rfc/draft` → new proposal under discussion.
* `rfc/accepted` → proposal approved, ready for implementation.
* `rfc/rejected` → proposal not moving forward.
* `implementation` → issues/PRs related to building an accepted RFC.
* `adr` → final records of architectural decisions (optional label for clarity).

---

## 🔄 PR Workflow

1. **Contributor** opens PR with RFC (label `rfc/draft`).
2. **Discussion** happens in PR comments (like a mini design review).
3. **Core team** decides → update status in RFC, change label (`rfc/accepted` or `rfc/rejected`).
4. **Merge PR** → RFC lives in repo.
5. **Implementation Issue** is created/linked.
6. **Final ADR** written (via new PR) when implementation stabilizes.

---

## 📂 Repo Layout Example

```text
/docs
  /rfcs
    001-initial-architecture.md
    002-new-database-strategy.md
    003-template.md
  /adrs
    001-use-postgresql.md
    002-adopt-open-telemetry.md
```

---

✨ This gives you:

* **Visibility** → RFC PRs for discussion.
* **Actionability** → Issues for execution.
* **History** → ADRs for long-term architectural record.
