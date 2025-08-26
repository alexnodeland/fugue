FROM mcr.microsoft.com/devcontainers/rust:1-1-bullseye

# Install OS dependencies here
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    pkg-config libssl-dev cmake clang && \
    apt-get clean && rm -rf /var/lib/apt/lists/*