# Headroom Rust build targets. `just` is not installed on dev boxes; this
# Makefile is the source of truth and is mirrored by .github/workflows/rust.yml.

SHELL := /bin/bash
CARGO ?= cargo
MATURIN ?= maturin
PYTHON ?= python3
FIXTURES ?= tests/parity/fixtures

.PHONY: help test test-parity bench build-proxy build-wheel fmt fmt-check lint clippy clean

help:
	@echo "Headroom Rust targets:"
	@echo "  make test         - cargo test --workspace"
	@echo "  make test-parity  - maturin develop + parity-run against fixtures"
	@echo "  make bench        - cargo bench --workspace"
	@echo "  make build-proxy  - release build + strip headroom-proxy, print size"
	@echo "  make build-wheel  - release wheel for headroom-py"
	@echo "  make fmt          - cargo fmt --all"
	@echo "  make fmt-check    - cargo fmt --all -- --check"
	@echo "  make lint         - cargo clippy --workspace -- -D warnings"
	@echo "  make clean        - cargo clean"

test:
	$(CARGO) test --workspace

test-parity:
	@if [ -z "$$VIRTUAL_ENV" ]; then \
		echo "error: activate a venv first (e.g. source .venv/bin/activate)"; \
		exit 1; \
	fi
	$(MATURIN) develop -m crates/headroom-py/Cargo.toml
	$(CARGO) run -p headroom-parity -- run --fixtures $(FIXTURES)

bench:
	$(CARGO) bench --workspace

build-proxy:
	$(CARGO) build --release -p headroom-proxy
	@BIN=target/release/headroom-proxy; \
	if command -v strip >/dev/null 2>&1; then strip "$$BIN" || true; fi; \
	SIZE=$$(wc -c < "$$BIN"); \
	printf 'headroom-proxy: %s bytes (%.1f MiB)\n' "$$SIZE" "$$(echo "$$SIZE / 1048576" | bc -l)"

build-wheel:
	$(MATURIN) build --release -m crates/headroom-py/Cargo.toml

fmt:
	$(CARGO) fmt --all

fmt-check:
	$(CARGO) fmt --all -- --check

clippy lint:
	$(CARGO) clippy --workspace -- -D warnings

clean:
	$(CARGO) clean
