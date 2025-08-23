.PHONY: help test coverage coverage-html clean lint fmt check all

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*##"; printf "\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  %-15s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

test: ## Run all tests
	cargo test --all-features --workspace

coverage: ## Generate coverage report (requires cargo-tarpaulin)
	cargo tarpaulin --verbose --all-features --workspace --timeout 120 --out Xml --out Html --output-dir coverage

coverage-html: coverage ## Generate and open HTML coverage report
	open coverage/tarpaulin-report.html || xdg-open coverage/tarpaulin-report.html

clean: ## Clean build artifacts and coverage reports
	cargo clean
	rm -rf coverage/
	rm -f cobertura.xml tarpaulin-report.html

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

install-tools: ## Install development tools
	cargo install cargo-tarpaulin
	cargo install cargo-watch
	cargo install cargo-edit

watch: ## Watch for changes and run tests
	cargo watch -x test

all: fmt lint test coverage ## Run all checks (format, lint, test, coverage)
