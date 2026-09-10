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
# Shallow, and with the version pinned. Both matter: setuptools_scm derives the
# dev counter from how much history is present, so a full clone reports
# 0.0.0.dev17258+g079d40460 while a --depth 1 clone reports 0.0.0.dev1+g079d40460
# for the SAME commit. Anything keyed on the engine version string -- an
# inference-atlas cell, a cached kernel dir -- would then see two engines where
# there is one. Pin it so container and native builds agree.
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0.dev1+g079d40460
RUN git clone --depth 1 -b ling_v3_support_mxfp4_humming https://github.com/inclusionAI/sglang.git /opt/sglang \
    && git -C /opt/sglang rev-parse --short=9 HEAD | grep -q "^${SGLANG_REF}$" \
    || (echo "branch head is not ${SGLANG_REF}; update SGLANG_REF" >&2; exit 1)
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
