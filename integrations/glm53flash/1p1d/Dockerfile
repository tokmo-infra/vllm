ARG BASE_IMAGE=local/glm53flash-nvfp4kv:compat-api-v7-20261124

FROM docker.io/library/rust:1.95-bookworm@sha256:6258907abe69656e41cd992e0b705cdcfabcbbe3db374f92ed2d47121282d4a1 AS rust-build
WORKDIR /src/rust
COPY rust/ ./
ENV CARGO_NET_GIT_FETCH_WITH_CLI=true
RUN cargo build --release --locked --package vllm-cmd --bin vllm-rs --features native-tls-vendored

FROM ${BASE_IMAGE}
COPY --from=rust-build /src/rust/target/release/vllm-rs /usr/local/bin/vllm-rs
# build-dflash.sh stages only the runtime allowlist here.
COPY runtime/vllm/ /usr/local/lib/python3.12/dist-packages/vllm/
