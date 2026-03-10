# Protocol-algorithm v2.0 - Docker Multi-stage Build

FROM rust:1.75 as rust-builder

WORKDIR /app

# Copy workspace
COPY Cargo.toml ./
COPY core ./core

# Build Rust core
RUN cd core && cargo build --release

# Build web backend
COPY web ./web
RUN cd web && cargo build --release


FROM python:3.11-slim as python-builder

WORKDIR /app

# Install dependencies
COPY viz/requirements.txt ./viz/
RUN pip install --no-cache-dir -r viz/requirements.txt

# Copy Python package
COPY python ./python
COPY --from=rust-builder /app/core ./core
WORKDIR /app/python
RUN pip install --no-cache-dir maturin && \
    maturin build --release


FROM node:18-slim as frontend-builder

WORKDIR /app/web/frontend

# Copy package files
COPY web/frontend/package*.json ./
COPY web/frontend/*.config.* ./
COPY web/frontend/*.json ./

# Install dependencies
RUN npm ci

# Copy source and build
COPY web/frontend/src ./src
COPY web/frontend/index.html ./
RUN npm run build


FROM debian:bookworm-slim as runtime

LABEL maintainer="Kk <xfengyin@gmail.com>"
LABEL description="Protocol-algorithm v2.0 - WSN Simulation Platform"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Rust binaries
COPY --from=rust-builder /app/core/target/release/libprotocol_algo_core.so ./lib/
COPY --from=rust-builder /app/web/target/release/protocol-algo-web ./bin/

# Copy Python package
COPY --from=python-builder /app/python/target/wheels/*.whl ./wheels/
RUN pip3 install --no-cache-dir ./wheels/*.whl

# Copy frontend build
COPY --from=frontend-builder /app/web/dist ./web/dist

# Copy visualization scripts
COPY --from=python-builder /usr/local/lib/python3.11/site-packages/ ./lib/python3.11/site-packages/
COPY viz/ ./viz/

# Copy examples
COPY python/examples/ ./examples/

# Expose web port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3000/api/health || exit 1

# Default command
CMD ["./bin/protocol-algo-web"]
