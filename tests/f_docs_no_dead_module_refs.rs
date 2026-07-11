//! FG-22 guard: user-facing docs must not reference the removed
//! `fugue::runtime::memory` module or its public types (`CowTrace`, `TracePool`,
//! `TraceBuilder`, `PooledPriorHandler`).
//!
//! Those references live inside ```rust,ignore``` doc code blocks, which are
//! compiled by neither `cargo test` (they are markdown, not doctests on public
//! items) nor `mdbook test` (which skips `ignore`), so a broken module path could
//! silently reappear. This test greps the rendered docs tree and the changelog so
//! any reintroduced reference fails CI.

use std::fs;
use std::path::{Path, PathBuf};

fn collect_md(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_md(&path, out);
        } else if path.extension().is_some_and(|e| e == "md") {
            out.push(path);
        }
    }
}

#[test]
fn docs_do_not_reference_removed_memory_module() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));

    let mut files = Vec::new();
    collect_md(&root.join("docs/src"), &mut files);
    // The changelog documents user-facing surface too.
    files.push(root.join(".github/CHANGELOG.md"));

    // The removed module path and its public types (FG-22). None of these resolve
    // any longer, so any occurrence in user-facing docs is a broken reference.
    let forbidden = [
        "runtime::memory",
        "TracePool",
        "CowTrace",
        "TraceBuilder",
        "PooledPriorHandler",
    ];

    let mut offenders = Vec::new();
    for f in &files {
        let Ok(text) = fs::read_to_string(f) else {
            continue;
        };
        for (i, line) in text.lines().enumerate() {
            for pat in &forbidden {
                if line.contains(pat) {
                    offenders.push(format!("{}:{}: {}", f.display(), i + 1, line.trim()));
                }
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "docs still reference the removed fugue::runtime::memory subsystem (FG-22):\n{}",
        offenders.join("\n")
    );
}
