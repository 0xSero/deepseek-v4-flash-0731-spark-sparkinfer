# DeepSeek V4 Flash on one DGX Spark

A pinned Docker recipe for serving [`0xSero/deepseek-v4-flash-0731-spark`](https://huggingface.co/0xSero/deepseek-v4-flash-0731-spark) on **one NVIDIA DGX Spark** with [Local Inference Lab's SparkInfer](https://github.com/local-inference-lab/sparkinfer).

The validated configuration exposes a 262,144-token model limit, captures CUDA graphs for C1/C2/C4, and preserves the model's EXL3/Trellis weights. Real text generation and JSON-schema structured output both passed on a physical GB10/SM121 Spark.

> **Important KV-format disclosure:** the reliable single-Spark route uses a 584-byte padded FP8 sparse-MLA record under the `nvfp4_ds_mla` control path. It is **not** a true 432-byte NVFP4 KV record. The true 432-byte implementation boots and passes its isolated numerical oracle, but produced corrupted full-model text and is therefore disabled in the serving recipe. This follows the conservative compatibility envelope documented by the [MiaAI-Lab DSpark work](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark).

This image is intentionally specific to Linux aarch64 and GB10/SM121. It exits on other architectures. It does not silently use eager execution, disable CUDA graphs, shorten context, or substitute another model.

## What the pieces do

- **EXL3** is TurboDerp's low-bit weight format. It compresses the model weights so they fit on the Spark.
- **Trellis** allocates quantization precision non-uniformly where it matters, instead of forcing every weight group to use the same bit width. That is why an EXL3 model can preserve more useful quality at the same average weight budget.
- **SparkInfer** supplies the GB10-optimized sparse-MLA attention path used for prefill and decode.
- **REAP** prunes low-value MoE experts using router-weighted activation observations. This artifact is already a REAP-derived model; the runtime does not prune it again.

## Pinned components

- `nvcr.io/nvidia/vllm:26.02-py3`
- Local Inference Lab vLLM commit `30038602b71395f481ef4a6edfe4fcf8551d9c15`
- SparkInfer commit `272a84bd97ce791a1e92d1f3a0da3dd5f3c6565f`
- model revision `22f28d32b9b29b4352eaa380ff8c2c170b2847ab`
- PyTorch `2.12.0+cu130`; CUTLASS DSL `4.6.0`
- TP1; 262,144-token limit; four scheduled sequences
- `FULL_AND_PIECEWISE` CUDA graphs at batch sizes 1, 2, and 4
- 584-byte padded FP8 sparse-MLA compatibility record

The patch files are readable source patches. Docker verifies that each patch applies to its pinned upstream commit before compiling the required vLLM extension. The build-only patch makes pinned fetches shallow and skips optional external packages when compiling the single required extension; it does not change runtime kernels.

## Run

Install current NVIDIA drivers, Docker, the NVIDIA Container Toolkit, and Docker Compose on the Spark. The first run downloads about 107 GB of source weights, losslessly coalesces the rank-sliced EXL3 archive to TP1, verifies checksums, runs a low-level SparkInfer CUDA-graph self-test, and starts the OpenAI-compatible server.

```bash
git clone https://github.com/0xSero/deepseek-v4-flash-0731-spark-sparkinfer.git
cd deepseek-v4-flash-0731-spark-sparkinfer
docker compose build
docker compose up -d
docker compose logs -f
```

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

The acceptance run used a 3,589-token prompt and up to 128 completion tokens. C2 and C4 reused the same prompt, so prefix caching warmed those runs; their TTFT/prefill figures are not cold-prefill results.

| Concurrency | Aggregate decode | Mean per-request decode | TTFT p50 | Mean prefill |
|---:|---:|---:|---:|---:|
| C1 | 20.53 tok/s | 20.53 tok/s | 11.262 s | 318.67 tok/s |
| C2 | 33.50 tok/s | 17.21 tok/s | 0.913 s | 3,929.26 tok/s |
| C4 | 61.38 tok/s | 16.11 tok/s | 3.129 s | 1,161.09 tok/s |

The requested 35–50 tok/s single-stream target was **not** reached. The measured C1 decode rate was 20.53 tok/s. MTP speculative decoding was tested but cannot load because this quantized artifact omits the model's layer-43 MTP tensors, so the recipe fails closed without MTP rather than fabricating that speed claim.

Run the same benchmark:

```bash
python3 scripts/benchmark.py --prompt-repetitions 256 --output-tokens 128 \
  | tee results/local-c1-c2-c4.json
```

See [`VALIDATION.md`](VALIDATION.md) and the committed files under `results/` for full evidence. Performance depends on prompt shape, prefix-cache state, thermals, and clocks.

## Runtime implementation

The serving route:

1. keeps the EXL3/Trellis weight payload intact while coalescing rank-sliced files to TP1;
2. stores each sparse-MLA token in the proven 584-byte padded FP8 record;
3. routes compressed decode and multi-group prefill through SparkInfer;
4. uses real CUDA graphs for C1, C2, and C4;
5. verifies the model manifest before launch.

`scripts/selftest.py` separately exercises the experimental 432-byte writer against a dequantized PyTorch oracle and through a CUDA graph. That is a useful kernel regression test, but it is not a claim that the 432-byte format passes full-model generation. The server-level semantic and structured-output tests are the authoritative gates.

## Credits

Thanks to [Local Inference Lab](https://github.com/local-inference-lab) for SparkInfer and the DGX Spark vLLM work, [TurboDerp](https://github.com/turboderp-org/exllamav3) for EXL3/Trellis, [MiaAI-Lab](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark) for the DSpark reference work, NVIDIA for the CUDA stack and DGX Spark, DeepSeek for DeepSeek V4 Flash, and the authors and contributors of vLLM and REAP.

Apache-2.0. Model use remains subject to the model repository's license and terms.
