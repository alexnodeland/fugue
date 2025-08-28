.PHONY: help test coverage clean lint fmt check all

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*##"; printf "\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  %-15s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

test: ## Run all tests
	cargo test --all-features --workspace

coverage: ## Generate coverage report (requires cargo-llvm-cov)
	cargo llvm-cov --all-features --fail-under-lines 80 --html --open

clean: ## Clean build artifacts and coverage reports
	cargo clean

lint: ## Run clippy linter
	cargo clippy --all-targets --all-features -- -D warnings

fmt: ## Format code
	cargo fmt --all

check: ## Check code formatting
	cargo fmt --all -- --check

bench: ## Run benchmarks
	cargo bench

doc: ## Generate and open documentation
	cargo doc --all-features --no-deps --open

mdbook: ## Build mdbook documentation
	mdbook build docs

install-tools: ## Install development tools
	cargo install cargo-llvm-cov
	cargo install cargo-watch
	cargo install cargo-edit

watch: ## Watch for changes and run tests
	cargo watch -x test

all: fmt lint test coverage ## Run all checks (format, lint, test, coverage)

docs-all: doc mdbook ## Build all documentation (rustdoc + mdbook)
