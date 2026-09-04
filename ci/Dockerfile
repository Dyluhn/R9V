# R9V Base CI & Runner Container (Spec 14 §3)
# Base: pinned reference-stack ROCm 7.14.0 image; Rust 1.96.0

FROM rocm/dev-ubuntu-24.04:7.14.0-full@sha256:439edaa8f0c4be4a3728e528f87b8a2ea1f051f34cf10b27caa4bd94f562eda7

ENV DEBIAN_FRONTEND=noninteractive \
    RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:/opt/rocm/llvm/bin:/opt/rocm/bin:$PATH \
    RUST_VERSION=1.96.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    build-essential \
    pkg-config \
    python3 \
    python3-pip \
    shellcheck \
    && rm -rf /var/lib/apt/lists/*

# Keep the image self-contained: source is deliberately mounted only at run
# time, so its pinned Python gate tools are mirrored here explicitly.
RUN python3 -m pip install --break-system-packages --no-cache-dir \
    pytest==9.1.1 \
    ruff==0.16.5

# Install pinned Rust toolchain (Spec 14 §3, rust-toolchain.toml)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    --default-toolchain ${RUST_VERSION} \
    --profile minimal \
    && rustup component add rustfmt clippy \
    && chmod -R a+w $RUSTUP_HOME $CARGO_HOME

# Install cargo-deny (0.20.2)
RUN curl -sL "https://github.com/EmbarkStudios/cargo-deny/releases/download/0.20.2/cargo-deny-0.20.2-x86_64-unknown-linux-musl.tar.gz" \
    | tar -xz -C /usr/local/cargo/bin --strip-components=1 "cargo-deny-0.20.2-x86_64-unknown-linux-musl/cargo-deny"

WORKDIR /workspace
ENV R9V_CI_REQUIRE_SHELLCHECK=1
