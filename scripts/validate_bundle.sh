#!/usr/bin/env bash
set -Eeuo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
vllm_source=${1:-}
sparkinfer_source=${2:-}

bash -n "${repo_root}/scripts/entrypoint.sh" \
  "${repo_root}/scripts/healthcheck.sh"
python3 -m py_compile \
  "${repo_root}/scripts/benchmark.py" \
  "${repo_root}/scripts/coalesce_rank_sliced_exl3.py" \
  "${repo_root}/scripts/selftest.py" \
  "${repo_root}/scripts/verify_tp1_manifest.py"
python3 -m json.tool "${repo_root}/config/recipe.json" >/dev/null

if rg -n --glob '!README.md' --glob '!patches/*.patch' \
  --glob '!scripts/validate_bundle.sh' \
  'enforce.eager|disable.cuda.graph|max_tokens' "${repo_root}"; then
  echo "Forbidden serving fallback or token parameter found." >&2
  exit 1
fi

if [[ -n "${vllm_source}" ]]; then
  validation_tree=$(mktemp -d)
  trap 'rm -rf "${validation_tree}"' EXIT
  git clone --shared --quiet "${vllm_source}" "${validation_tree}/vllm"
  git -C "${validation_tree}/vllm" checkout --quiet --detach \
    30038602b71395f481ef4a6edfe4fcf8551d9c15
  git -C "${validation_tree}/vllm" apply --check \
    "${repo_root}/patches/vllm.patch"
  git -C "${validation_tree}/vllm" apply "${repo_root}/patches/vllm.patch"
  git -C "${validation_tree}/vllm" apply --check \
    "${repo_root}/patches/vllm-padded-fp8-compat.patch"
  git -C "${validation_tree}/vllm" apply \
    "${repo_root}/patches/vllm-padded-fp8-compat.patch"
  git -C "${validation_tree}/vllm" apply --check \
    "${repo_root}/patches/vllm-build-fetch.patch"
fi
if [[ -n "${sparkinfer_source}" ]]; then
  git -C "${sparkinfer_source}" apply --check \
    "${repo_root}/patches/sparkinfer.patch"
fi

echo "bundle validation passed"
