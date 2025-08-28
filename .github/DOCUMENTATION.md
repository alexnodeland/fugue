# Documentation System Specification

This document defines the standards and workflows for writing, organizing, and testing documentation in this repository. It ensures a consistent developer experience, maintains correctness via CI, and provides both reference documentation and learning materials.

## 1. Documentation Structure

### 1.1 Source Layout

```text
your-crate/
├─ src/ # Rust source code
│ ├─ lib.rs
│ └─ foo.rs # modules
├─ examples/ # runnable examples (compiled with cargo run --example)
│ └─ widget_basic.rs
├─ docs/
│ ├─ src/ # mdBook sources (how-tos, tutorials, guides)
│ │ └─ ...
│ └─ api/ # long-form API docs (included into rustdoc)
│ ├─ \_prelude.md
│ ├─ foo.md
│ └─ ...
├─ tests/
│ └─ doctests.rs # optional: standalone doctests via doc-comment
└─ Cargo.toml
```

### 1.2 Documentation Types

- **API Docs** (docs/api/_.md)
  - Long-form reference documentation attached to modules/items with #[doc = include_str!(...)]. Includes extended examples, design notes, and error-handling guidance.
- **Inline Docs** (/// and //!)
  - Concise explanations and small, copy-pasteable examples (≤ 25 lines, ≤ 2 imports) directly in the source. Show up in IDEs and rustdoc item pages.
- **mdBook Docs** (docs/src)
  - Guides, how-tos, and tutorials. Include anchored snippets from examples/ to keep docs and code in sync. Tested with mdbook test.
- **Examples** (examples/_.rs)
  - Full, runnable programs demonstrating end-to-end workflows. Serve as canonical sources for snippets (via anchors) and are part of CI.

## 2. Writing Documentation

### 2.1 Inline API Documentation

- Keep short usage examples next to the code:

```rust
/// Creates a new widget.
///
/// `/// use your_crate::Widget;
/// let w = Widget::new();
///`
pub fn new() -> Self { ... }
```

- Use doctest fences:
  - ```rust``` — compile & run
  - ```rust,no_run``` — compile only
  - ```rust,ignore``` — shown but not tested
  - ```rust,compile_fail``` — must fail

### 2.2 API Markdown Documents (docs/api/\*.md)

Each module’s long-form doc should include: 1. Overview — purpose and conceptual model. 2. Usage Examples — extended, doctested code with hidden imports. 3. Design & Evolution — rationale, invariants, feature flags, and proposal workflow. 4. Error Handling — common failure modes and best practices. 5. Integration Notes — relation to other modules and external crates. 6. Reference Links — cross-links to API items, mdBook tutorials, and examples.

#### Example Template

````markdown
# Module: Foo

## Overview

Explain the problem this module solves and the abstractions it provides.

## Usage Examples

```rust
# use crate::foo::*;
let w = Widget::new();
w.do_it();
```
````

#### Design & Evolution

- Status: Stable since vX.Y.
- Feature flags: foo_async requires feature="async".
- Invariants: Cheap to clone; idempotent builder.

#### Proposal Workflow

- Open a Design Proposal (DP) issue.
- Discuss API surface and alternatives.
- Land behind feature flag → stabilize on minor release.

#### Error Handling

- FooError::Io when file access fails.
- Always check return values.

#### Integration Notes

- Works with serde if feature="serde".
- Compatible with async runtimes.

#### Reference Links

- Widget
- Tutorial: Foo in practice

### 2.3 mdBook Documentation

- Use **real code from `examples/`** via anchors:

```rust,no_run
{{#include ../../examples/widget_basic.rs:snippet_simple}}
```

- Write narrative, tutorials, and workflows.
- Ensure all code blocks compile with mdbook test -L target/debug/deps.

### 2.4 Examples (examples/*.rs)

- Contain runnable, end-to-end code.
- Mark snippet regions with // ANCHOR: name and // ANCHOR_END: name for inclusion in mdBook and API docs.

Example:

````rust
// examples/widget_basic.rs

// ANCHOR: snippet_simple
use your_crate::Widget;

fn main() {
    let w = Widget::new();
    w.do_it();
}
// ANCHOR_END: snippet_simple
````

## 3. Testing Documentation

### 3.1 Cargo Doctests

Run doctests embedded in code and included API Markdown:

```bash
cargo test --doc --all-features
```

### 3.2 mdBook Tests

Run doctests for all code snippets in guides/tutorials:

```bash
cargo build --all-features
mdbook test docs/src -L target/debug/deps
```

### 3.3 Standalone Markdown Tests (Optional)

If testing Markdown files not attached to modules (e.g., README.md):

```rust
// tests/doctests.rs
doc_comment::doctest!("../README.md");
doc_comment::doctest!("../docs/api/foo.md");
```

## 4. Continuous Integration

### 4.1 Lints & Flags

- Deny broken links and missing docs:

```rust
#![deny(rustdoc::broken_intra_doc_links)]
#![warn(missing_docs)]
```

### 4.2 GitHub Actions Workflow

```yaml
name: Docs & Examples
on: [push, pull_request]

jobs:
  docs_examples:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo build --all-features
      - run: cargo test --all-features
      - run: cargo test --doc --all-features
      - name: Install mdBook
        run: cargo install mdbook --locked
      - name: Test mdBook docs
        run: mdbook test docs/src -L target/debug/deps
```

## 5. Responsibilities & Best Practices

- Inline docs = concise, always present.
- API Markdown = extended reference; only when inline would be too long.
- Examples/ = runnable, anchored snippets; the single source of truth.
- mdBook = tutorials & how-tos, always include snippets from examples/.
- CI = ensures every snippet compiles/tests, preventing drift.

## 6. Proposal / Evolution Process

1. Proposals filed as DP issues (or rfcs/NNNN-title.md).
2. Discuss API changes with design notes and alternatives.
3. Implement behind feature flag.
4. Stabilize in next minor release after usage feedback.
5. Update API docs + mdBook + examples simultaneously.

## ✅ Outcome

- Developers learn with mdBook tutorials.
- They confirm details in API reference.
- All examples stay executable and CI-covered.
- Evolution is traceable and standardized.
