#!/usr/bin/env bash
set -Eeuo pipefail

model_repo=${MODEL_REPO:-0xSero/deepseek-v4-flash-0731-spark}
model_revision=${MODEL_REVISION:-22f28d32b9b29b4352eaa380ff8c2c170b2847ab}
source_dir=${MODEL_SOURCE_DIR:-/models/source}
model_dir=${MODEL_PATH:-/models/tp1}

if [[ ! -f "${model_dir}/rank-sliced-tp1-manifest.json" ]]; then
  mkdir -p "${source_dir}" "${model_dir}"
  MODEL_REPO=${model_repo} MODEL_REVISION=${model_revision} \
    MODEL_SOURCE_DIR=${source_dir} /opt/runtime-venv/bin/python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["MODEL_REPO"],
    revision=os.environ["MODEL_REVISION"],
    local_dir=os.environ["MODEL_SOURCE_DIR"],
    token=os.environ.get("HF_TOKEN") or None,
)
PY
  /opt/runtime-venv/bin/python /opt/recipe/scripts/coalesce_rank_sliced_exl3.py \
    --input-dir "${source_dir}" \
    --output-dir "${model_dir}" \
    --link-carried \
    --reuse-complete \
    --workers "${COALESCE_WORKERS:-1}"
fi

verify_args=()
if [[ "${VERIFY_MODEL_CHECKSUMS:-1}" != "1" ]]; then
  verify_args+=(--skip-checksums)
fi
/opt/runtime-venv/bin/python /opt/recipe/scripts/verify_tp1_manifest.py \
  "${model_dir}" "${verify_args[@]}"

/opt/runtime-venv/bin/python /opt/recipe/scripts/selftest.py

export VLLM_PYTHON=/opt/runtime-venv/bin/python
export MODE=mtp0
export BACKEND=b12x-a8
export INDEXER_BACKEND=b12x
export ALLREDUCE_MODE=nccl
export TP_SIZE=1
export DCP_SIZE=1
export MODEL_PATH=${model_dir}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-deepseek-v4-flash-0731-spark}
export KV_CACHE_DTYPE=nvfp4_ds_mla
export KV_FP8_ROPE=0
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}
export MAX_NUM_SEQS=${MAX_NUM_SEQS:-4}
export MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
export MAX_CUDAGRAPH_CAPTURE_SIZE=${MAX_CUDAGRAPH_CAPTURE_SIZE:-4}
export CUDAGRAPH_CAPTURE_SIZES=${CUDAGRAPH_CAPTURE_SIZES:-1,2,4}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}
export LOAD_FORMAT=instanttensor
export PREFIX_CACHE=1
export ENABLE_FLASHINFER_AUTOTUNE=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export XDG_CACHE_HOME=/cache

exec /opt/vllm/serve-ds4-flash.sh
