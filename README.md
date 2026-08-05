# DeepSeek V4 Flash on one DGX Spark

Pinned Docker recipe for serving [`0xSero/deepseek-v4-flash-0731-spark`](https://huggingface.co/0xSero/deepseek-v4-flash-0731-spark) on **one NVIDIA DGX Spark** with Local Inference Lab's SparkInfer path.

The important part is memory: the EXL3 weights are kept rank-sliced on disk and losslessly coalesced to TP1, while the 262,144-token MLA cache uses a 432-byte NVFP4 record. The runtime keeps CUDA graphs enabled and captures C1, C2, and C4 shapes.

> This image is intentionally specific to Linux aarch64 and GB10/SM121. It exits on other architectures. It does not silently fall back to eager execution, a different KV format, or a shorter context.

## What is pinned

- NVIDIA vLLM `26.02-py3` base image
- Local Inference Lab vLLM at `30038602b71395f481ef4a6edfe4fcf8551d9c15`
- SparkInfer at `272a84bd97ce791a1e92d1f3a0da3dd5f3c6565f`
- model revision `22f28d32b9b29b4352eaa380ff8c2c170b2847ab`
- PyTorch `2.12.0+cu130` and CUTLASS DSL `4.6.0`
- TP1, 262,144-token limit, four concurrent sequences
- `FULL_AND_PIECEWISE` CUDA graphs at batch sizes 1, 2, and 4

The two patch files are source, not opaque binaries. Docker checks that each patch applies to its pinned upstream commit before compiling the one vLLM extension required by the port.

## Run

Install current NVIDIA drivers, Docker, the NVIDIA Container Toolkit, and Docker Compose on the Spark. The first run downloads about 107 GB of source weights, coalesces the rank-sliced EXL3 archive without requantizing it, verifies checksums, runs the NVFP4/SparkInfer CUDA-graph self-test, and starts the OpenAI-compatible server.

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

## Benchmark C1/C2/C4

The streaming benchmark reports TTFT, prompt-token/TTFT prefill rate, per-request decode rate, and aggregate decode throughput. It uses the server's returned token counts.

```bash
python3 scripts/benchmark.py --prompt-repetitions 2048 --output-tokens 128 \
  | tee results/c1-c2-c4.json
```

Performance depends on prompt length, thermal state, clocks, and whether prefix and kernel caches are warm. Treat numbers from another Spark as a reference, not a guarantee.

## What the port changes

- Implements the DeepSeek V4 432-byte MLA record: 256 bytes packed E2M1 latent, 32 bytes E4M3 group scales, 16 bytes padding, and 128 bytes BF16 RoPE.
- Adds an SM121-safe software E2M1 packer; it does not call an SM100-only conversion op.
- Routes SparkInfer compressed decode and multi-group prefill through the DeepSeek V4 NVFP4 layout.
- Pads the 27,648-byte logical C4 MLA page to the 32 KiB compressor-state page required by DeepSeek V4's shared physical KV tensor.
- Adds TP1 loading for the rank-sliced EXL3/Trellis tensors without decoding or requantizing their payloads.

`scripts/selftest.py` compares SparkInfer output to a PyTorch dequantization oracle and repeats it through a real CUDA graph. Container startup is fail-closed: the API server is not launched if this check fails.

## Validation status

The isolated NVFP4 write + SparkInfer attention path was validated on a GB10/SM121 with C1 and a 16-row batch. Both CUDA-graph replays were bit-stable; cosine similarity against the dequantized PyTorch oracle was greater than `0.999997`.

Full-service startup, post-capture KV capacity, generation, structured output, and C1/C2/C4 benchmark evidence are recorded only after completing the live acceptance run; see `VALIDATION.md`.

## Credits

Thanks to [Local Inference Lab](https://github.com/local-inference-lab) for SparkInfer and the DGX Spark vLLM work, [TurboDerp](https://github.com/turboderp-org/exllamav3) for EXL3/Trellis, NVIDIA for the CUDA stack and DGX Spark, DeepSeek for DeepSeek V4 Flash, and the authors and contributors of vLLM.

Apache-2.0. Model use remains subject to the model repository's license and terms.
