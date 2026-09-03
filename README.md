# DeepSeek V4 Flash on one DGX Spark

A pinned Docker recipe for serving [`0xSero/deepseek-v4-flash-0731-spark`](https://huggingface.co/0xSero/deepseek-v4-flash-0731-spark) on **one NVIDIA DGX Spark** with [Local Inference Lab's SparkInfer](https://github.com/local-inference-lab/sparkinfer).

The validated configuration exposes a 262,144-token model limit and uses a compact K64 DSpark speculative draft with fixed K5 verification. The target's EXL3/Trellis weights and all 216 target experts remain untouched. Real code generation and strict JSON-schema structured output both passed on a physical GB10/SM121 Spark.

> **Important KV-format disclosure:** the reliable single-Spark route uses a 584-byte padded FP8 sparse-MLA record under the `nvfp4_ds_mla` control path. It is **not** a true 432-byte NVFP4 KV record. The true 432-byte implementation boots and passes its isolated numerical oracle, but produced corrupted full-model text and is therefore disabled in the serving recipe. This follows the conservative compatibility envelope documented by the [MiaAI-Lab DSpark work](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark).

This image is intentionally specific to Linux aarch64 and GB10/SM121. It exits on other architectures. It does not silently use eager execution, disable CUDA graphs, shorten context, or substitute another model.

## What the pieces do

- **EXL3** is TurboDerp's low-bit weight format. It compresses the model weights so they fit on the Spark.
- **Trellis** allocates quantization precision non-uniformly where it matters, instead of forcing every weight group to use the same bit width. That is why an EXL3 model can preserve more useful quality at the same average weight budget.
- **SparkInfer** supplies the GB10-optimized sparse-MLA attention path used for prefill and decode.
- **REAP** prunes low-value MoE experts using router-weighted activation observations. This artifact is already a REAP-derived model; the runtime does not prune it again.

## Pinned components

- public runtime image `ghcr.io/0xsero/deepseek-v4-flash-0731-spark-sparkinfer@sha256:2e077489a83a0360952828051fe7f7a32c1801e5ce8436d85f7267583d614ff4`
- `nvcr.io/nvidia/vllm:26.02-py3`
- Local Inference Lab vLLM commit `30038602b71395f481ef4a6edfe4fcf8551d9c15`
- SparkInfer commit `272a84bd97ce791a1e92d1f3a0da3dd5f3c6565f`
- vLLM Flash Attention FA4 Python source `caaa4eb59845388a20b1f435ecaafb4bd9517ad8`
- model revision `22f28d32b9b29b4352eaa380ff8c2c170b2847ab`
- PyTorch `2.12.0+cu130`; CUTLASS DSL `4.6.0`; Transformers `5.13.1`; Mistral Common `1.11.5`; InstantTensor `0.1.5`; OpenAI `2.44.0`; compressed-tensors `0.17.0`; XGrammar `0.2.3`; TileLang `0.1.9`; quack-kernels `0.6.2`; Ninja `1.13.0`
- TP1; 262,144-token limit; four scheduled sequences
- fixed K5 DSpark with a 64-expert draft selected from preserved REAP rankings
- `FULL_AND_PIECEWISE` CUDA graph for the six-row C1 verifier shape
- B12X fused attention output projection
- non-reasoning mode by default for the measured speed profile; callers can opt into thinking per request
- 584-byte padded FP8 sparse-MLA compatibility record

The patch files are readable source patches. Docker verifies that each patch applies to its pinned upstream commit before compiling the required vLLM extension. The build-only patch makes pinned fetches shallow and skips optional external packages when compiling the single required extension; it does not change runtime kernels. The InstantTensor patch preserves scalar tensor shapes and clones each view before its reusable loader buffer advances; this is the exact correction exercised by the validated development runtime.

## Run

Install current NVIDIA drivers, Docker, the NVIDIA Container Toolkit, and Docker Compose on the Spark. The first run pulls the pinned GB10 image, downloads about 107 GB of source weights, losslessly coalesces the rank-sliced EXL3 archive to TP1, verifies checksums, constructs and validates the 3.0 GB K64 DSpark draft, runs a low-level SparkInfer CUDA-graph self-test, and starts the OpenAI-compatible server. The source target is never modified.

One command performs the complete install and launch:

```bash
git clone https://github.com/0xSero/deepseek-v4-flash-0731-spark-sparkinfer.git && cd deepseek-v4-flash-0731-spark-sparkinfer && docker compose up -d
```

Compose pulls the public [GHCR package](https://github.com/users/0xSero/packages/container/package/deepseek-v4-flash-0731-spark-sparkinfer) by immutable manifest digest, so no registry login is required and the command cannot silently drift to a different image.

Follow startup with `docker compose logs -f`. Model data and compiled caches persist under `./data` and `./cache`, so restarts do not redownload or rebuild valid artifacts. To build the pinned source image locally instead of pulling GHCR, run `docker build -t deepseek-v4-flash-0731-spark-sparkinfer:local .`.

For a gated model, place `HF_TOKEN=...` in a local `.env`; it is ignored by Git. Model data and build caches remain in `./data` and `./cache`.

Wait for health:

```bash
until curl -fsS http://127.0.0.1:8000/health; do sleep 10; done
```

Generate:

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4-flash-0731-spark",
    "messages": [{"role": "user", "content": "Write a Python hello world."}],
    "temperature": 0,
    "max_completion_tokens": 128
  }'
```

## Measured performance

The clean repository-built image passed checksum verification, real generation, strict JSON schema, 262,144-token capacity, and FULL/PIECEWISE CUDA-graph capture. Five independent 512-token code generations ran at C1 with `thinking=false`; decode excludes TTFT and the first generated token.

| Surface | Result | Gate |
|---|---:|---:|
| Clean-image C1 code decode minimum | 34.30 tok/s | >=35 tok/s: **pending** |
| Clean-image C1 code decode median | 38.12 tok/s | informational |
| Clean-image C1 code decode mean | 39.49 tok/s | informational |
| Same-path tuning run, cold 252,047-token prefill | 1,055.45 tok/s | clean-image rerun pending |

The clean-image trials were `38.12`, `34.30`, `48.88`, `37.77`, and `38.40` tok/s. The image is publishable as a correct reproducible runtime, but the requested steady 35 tok/s floor is still an open optimization gate and is not claimed as complete. A prior same-path cold-prefill run returned the expected `PREFILL OK.` response at 1,055.45 tok/s with zero cached prompt tokens; it remains tuning evidence until repeated on the final published image.

Dynamic K1–K5 depth was also tested and was faster on some prompts, but a strict JSON-schema request exposed a grammar-mask assertion in the pinned vLLM runtime. It is disabled in the published profile. Fixed K5 passes correctness and structured output; further decode tuning continues separately.

Run the same benchmark:

```bash
python3 scripts/acceptance_c1.py --skip-prefill
```

See [`VALIDATION.md`](VALIDATION.md) and the committed files under `results/` for full evidence. Performance depends on prompt shape, prefix-cache state, thermals, and clocks.

## Context coherence

The final-image default profile was tested at exact message-content lengths from 500 through 20,000 characters. Every request had distinct facts at the beginning, middle, and end plus an arithmetic checksum; the server had to return all four through a strict JSON schema.

| Context characters | Prompt tokens | Result | End-to-end time |
|---:|---:|---:|---:|
| 500 | 116 | Pass | 9.80 s |
| 2,000 | 384 | Pass | 3.79 s |
| 5,000 | 910 | Pass | 4.33 s |
| 10,000 | 1,782 | Pass | 3.80 s |
| 20,000 | 3,532 | Pass | 6.28 s |

All five responses preserved the exact beginning/middle/end values, checksum, schema, and `stop` finish reason. This demonstrates coherent fact retention and structured generation across the requested character range; it is not a claim that every possible reasoning workload at those lengths is error-free. Reproduce it with:

```bash
python3 scripts/context_coherence.py
```

## Runtime implementation

The serving route:

1. keeps the EXL3/Trellis target payload and all 216 target experts intact while coalescing rank-sliced files to TP1;
2. derives a separate 64-expert DSpark draft from the saved REAP rankings, with both structured-output calibration categories represented;
3. stores each sparse-MLA token in the proven 584-byte padded FP8 record;
4. routes compressed decode and multi-group prefill through SparkInfer, with B12X fused output projection;
5. uses a real FULL CUDA graph for fixed K5 C1 verification and PIECEWISE graphs elsewhere;
6. verifies the model manifest and draft tensor shapes before launch.

`scripts/selftest.py` separately exercises the experimental 432-byte writer against a dequantized PyTorch oracle and through a CUDA graph. That is a useful kernel regression test, but it is not a claim that the 432-byte format passes full-model generation. The server-level semantic and structured-output tests are the authoritative gates.

## Credits

Thanks to [Local Inference Lab](https://github.com/local-inference-lab) for SparkInfer and the DGX Spark vLLM work, [TurboDerp](https://github.com/turboderp-org/exllamav3) for EXL3/Trellis, [MiaAI-Lab](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark) for the DSpark reference work, NVIDIA for the CUDA stack and DGX Spark, DeepSeek for DeepSeek V4 Flash, and the authors and contributors of vLLM and REAP.

Apache-2.0. Model use remains subject to the model repository's license and terms.
