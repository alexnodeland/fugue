#!/usr/bin/env bash
set -euo pipefail

# Rust components
rustup component add rustfmt clippy

# Dev tooling
cargo install mdbook mdbook-linkcheck mdbook-admonish mdbook-mermaid --locked || true
cargo install cargo-deny cargo-nextest --locked || true

# Python-less installer for pre-commit if you prefer pipx; otherwise fallback to pip
if command -v pipx >/dev/null 2>&1; then
  pipx install pre-commit || true
else
  pip install --user pre-commit || true
fi

# Optional: TOML formatter / linter
if ! command -v taplo >/dev/null 2>&1; then
  curl -sSfL https://github.com/tamasfe/taplo/releases/latest/download/taplo-full-linux-x86_64.gz \
    | gunzip > /usr/local/bin/taplo && chmod +x /usr/local/bin/taplo || true
fi

# Install git hooks via pre-commit (if config exists)
if [ -f ".pre-commit-config.yaml" ]; then
  pre-commit install --hook-type pre-commit --hook-type commit-msg --hook-type pre-push || true
fi

# Run your project setup (Makefile target)
if command -v make >/dev/null 2>&1; then
  make setup || true
fi