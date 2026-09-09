# Ling-3.0-flash (MXFP4) on one DGX Spark -- SGLang from inclusionAI's Humming branch.
# arm64 / CUDA 13. Weights are NOT in the image: mount them at /models.
FROM nvidia/cuda:13.0.2-devel-ubuntu24.04
ARG SGLANG_REF=079d40460
ARG FLASHINFER_CUBIN=0.6.17
ENV DEBIAN_FRONTEND=noninteractive UV_LINK_MODE=copy PATH=/opt/venv/bin:/root/.local/bin:$PATH \
    VIRTUAL_ENV=/opt/venv SGLANG_BUILD_RUST_EXTS=none
RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates build-essential ninja-build cmake pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && git config --global http.version HTTP/1.1
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN git clone --filter=blob:none https://github.com/inclusionAI/sglang.git /opt/sglang \
    && git -C /opt/sglang checkout -q ${SGLANG_REF}
RUN uv venv --python 3.11 /opt/venv \
    && uv pip install --python /opt/venv/bin/python "openai>=1.52.0,<2.0.0" \
    && MAX_JOBS=4 uv pip install --python /opt/venv/bin/python -e "/opt/sglang/python[all]" \
    && uv pip install --python /opt/venv/bin/python "flashinfer-cubin==${FLASHINFER_CUBIN}" --index-url https://flashinfer.ai/whl
WORKDIR /app
COPY start.sh stop.sh bench.py compile-kernel.py download.py env.example run.sh ./
RUN chmod +x start.sh stop.sh compile-kernel.py download.py run.sh \
    && ln -s /opt/venv /app/.venv && ln -s /opt/sglang /app/sglang \
    && mkdir -p /models /app/logs
ENV MODEL_DIR=/models/Ling-3.0-flash-fp4 DRAFT_DIR=/models/Ling-3.0-flash-dspark HOST=0.0.0.0 PORT=30000
EXPOSE 30000
ENTRYPOINT ["/app/run.sh"]
CMD ["serve"]
