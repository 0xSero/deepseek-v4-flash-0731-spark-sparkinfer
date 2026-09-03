# syntax=docker/dockerfile:1.7
FROM nvcr.io/nvidia/vllm:26.02-py3

ARG VLLM_COMMIT=30038602b71395f481ef4a6edfe4fcf8551d9c15
ARG SPARKINFER_COMMIT=272a84bd97ce791a1e92d1f3a0da3dd5f3c6565f

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    CUTE_DSL_ARCH=sm_121a \
    PYTHONPATH=/opt/vllm:/opt/sparkinfer \
    PATH=/opt/runtime-venv/bin:/usr/local/cuda/bin:/usr/bin:/bin \
    KV_FP8_ROPE=0 \
    VLLM_DSV4_PADDED_NVFP4=1

RUN test "$(uname -m)" = aarch64 || \
    (echo "This image targets one DGX Spark (Linux aarch64 + GB10/SM121)." >&2; exit 2)

RUN rm -rf /opt/vllm /opt/sparkinfer && \
    git init /opt/vllm && \
    git -C /opt/vllm remote add origin \
      https://github.com/local-inference-lab/vllm.git && \
    git -C /opt/vllm fetch --depth=1 origin "${VLLM_COMMIT}" && \
    git -C /opt/vllm checkout --detach FETCH_HEAD && \
    git init /opt/sparkinfer && \
    git -C /opt/sparkinfer remote add origin \
      https://github.com/local-inference-lab/sparkinfer.git && \
    git -C /opt/sparkinfer fetch --depth=1 origin "${SPARKINFER_COMMIT}" && \
    git -C /opt/sparkinfer checkout --detach FETCH_HEAD

COPY patches/vllm.patch /tmp/vllm.patch
COPY patches/vllm-padded-fp8-compat.patch /tmp/vllm-padded-fp8-compat.patch
COPY patches/vllm-dspark-compact-draft.patch /tmp/vllm-dspark-compact-draft.patch
COPY patches/sparkinfer.patch /tmp/sparkinfer.patch
RUN git -C /opt/vllm apply --check /tmp/vllm.patch && \
    git -C /opt/vllm apply /tmp/vllm.patch && \
    git -C /opt/vllm apply --check /tmp/vllm-padded-fp8-compat.patch && \
    git -C /opt/vllm apply /tmp/vllm-padded-fp8-compat.patch && \
    git -C /opt/vllm apply --check /tmp/vllm-dspark-compact-draft.patch && \
    git -C /opt/vllm apply /tmp/vllm-dspark-compact-draft.patch && \
    git -C /opt/sparkinfer apply --check /tmp/sparkinfer.patch && \
    git -C /opt/sparkinfer apply /tmp/sparkinfer.patch

# The NVIDIA 26.02 image supplies CUDA 13.1 and the remaining vLLM runtime.
# Pin the ABI used by the validated GB10 build in a separate environment.
RUN python3 -m venv --system-site-packages /opt/runtime-venv && \
    /opt/runtime-venv/bin/python -m pip install --upgrade pip

RUN /opt/runtime-venv/bin/python -m pip install \
      --index-url https://download.pytorch.org/whl/cu130 \
      'torch==2.12.0+cu130' 'torchvision==0.27.0+cu130'

# NVIDIA's base image constrains CUTLASS DSL to 4.3.5 globally. The validated
# SparkInfer build needs 4.6.0, so override that constraint only in this venv.
RUN env -u PIP_CONSTRAINT /opt/runtime-venv/bin/python -m pip install \
      'nvidia-cutlass-dsl==4.6.0' 'safetensors==0.8.0' \
      'huggingface-hub>=0.34,<2' 'pytest==9.1.1'

RUN /opt/runtime-venv/bin/python -m pip install --no-deps -e /opt/sparkinfer

COPY patches/vllm-build-fetch.patch /tmp/vllm-build-fetch.patch
RUN git -C /opt/vllm apply --check /tmp/vllm-build-fetch.patch && \
    git -C /opt/vllm apply /tmp/vllm-build-fetch.patch

# Build only the stable vLLM extension needed by EXL3 and the SM12x NVFP4
# MLA cache writer. This deliberately avoids unrelated heavyweight extensions.
RUN export PATH=/usr/local/bin:${PATH} && \
    export TORCH_CUDA_ARCH_LIST=12.0 && \
    cmake -S /opt/vllm -B /opt/vllm-build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=120 \
      -DVLLM_TARGET_DEVICE=cuda \
      -DVLLM_PYTHON_EXECUTABLE=/opt/runtime-venv/bin/python \
      -DVLLM_ENABLE_SCALED_MM_C2X=OFF \
      -DVLLM_ENABLE_SM12X_NVFP4=OFF \
      -DVLLM_ENABLE_SM12X_NVFP4_KV=ON \
      -DVLLM_BUILD_RUNTIME_EXTERNALS=OFF \
      -DVLLM_BUILD_QUTLASS=OFF \
      -DVLLM_BUILD_VLLM_FLASH_ATTN=OFF && \
    cmake --build /opt/vllm-build --target _C_stable_libtorch --parallel 4 && \
    cp /opt/vllm-build/_C_stable_libtorch.abi3.so \
      /opt/vllm/vllm/_C_stable_libtorch.abi3.so

# The CuTeDSL indexer imports FA4's Python utilities at runtime. Fetch only the
# exact upstream Python source and perform the same package rewrite as vLLM's
# _vllm_fa4_cutedsl_C install component; no extra attention extension is built.
RUN git init /tmp/vllm-flash-attn && \
    git -C /tmp/vllm-flash-attn remote add origin \
      https://github.com/vllm-project/flash-attention.git && \
    git -C /tmp/vllm-flash-attn fetch --depth=1 origin \
      caaa4eb59845388a20b1f435ecaafb4bd9517ad8 && \
    git -C /tmp/vllm-flash-attn checkout --detach FETCH_HEAD && \
    mkdir -p /opt/vllm/vllm/vllm_flash_attn/cute && \
    cp -a /tmp/vllm-flash-attn/flash_attn/cute/. \
      /opt/vllm/vllm/vllm_flash_attn/cute/ && \
    find /opt/vllm/vllm/vllm_flash_attn/cute -type f -name '*.py' \
      -exec sed -i 's/flash_attn\.cute/vllm.vllm_flash_attn.cute/g' {} + && \
    rm -rf /tmp/vllm-flash-attn

RUN env -u PIP_CONSTRAINT /opt/runtime-venv/bin/python -m pip install \
      'nvidia-cutlass-dsl-libs-cu13==4.6.0' 'transformers==5.13.1' \
      'mistral-common==1.11.5' 'instanttensor==0.1.5' 'openai==2.44.0' \
      'compressed-tensors==0.17.0' 'apache-tvm-ffi==0.1.10' \
      'torch-c-dlpack-ext==0.1.5' 'z3-solver==4.15.4' 'tilelang==0.1.9' && \
    env -u PIP_CONSTRAINT /opt/runtime-venv/bin/python -m pip install \
      --no-deps 'quack-kernels==0.6.2' 'xgrammar==0.2.3' && \
    /opt/runtime-venv/bin/python -c \
      'from xgrammar import normalize_tool_choice'

ENV PATH=/opt/runtime-venv/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
RUN case "$(ninja --version)" in 1.13.0*) ;; *) exit 1 ;; esac

COPY patches/instanttensor.patch /tmp/instanttensor.patch
RUN patch --dry-run -p1 -d /opt/runtime-venv/lib/python3.12/site-packages \
      < /tmp/instanttensor.patch && \
    patch -p1 -d /opt/runtime-venv/lib/python3.12/site-packages \
      < /tmp/instanttensor.patch

COPY scripts /opt/recipe/scripts
COPY config /opt/recipe/config
RUN chmod +x /opt/recipe/scripts/*.sh

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20m --retries=3 \
  CMD /opt/recipe/scripts/healthcheck.sh

ENTRYPOINT ["/opt/recipe/scripts/entrypoint.sh"]
