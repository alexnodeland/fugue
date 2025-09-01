#!/usr/bin/env bash
set -euo pipefail

echo "Starting post-create setup..."

# Rust components
echo "Installing Rust components..."
rustup component add rustfmt clippy

echo "Installing taplo..."
if ! command -v taplo >/dev/null 2>&1; then
  ARCH=$(uname -m)
  case $ARCH in
    x86_64)
      TAPLO_ARCH="linux-x86_64"
      ;;
    aarch64|arm64)
      TAPLO_ARCH="linux-aarch64"
      ;;
    *)
      echo "Unsupported architecture: $ARCH. Skipping taplo installation."
      TAPLO_ARCH=""
      ;;
  esac
  
  if [ -n "$TAPLO_ARCH" ]; then
    curl -sSfL "https://github.com/tamasfe/taplo/releases/latest/download/taplo-full-${TAPLO_ARCH}.gz" \
      | gunzip > /usr/local/bin/taplo && chmod +x /usr/local/bin/taplo || true
  fi
fi

# Install dev tools from Cargo.toml dev-dependencies
echo "Installing development tools from Cargo.toml..."
if command -v make >/dev/null 2>&1; then
  make install-dev-tools || true
fi

echo "Post-create setup completed!"