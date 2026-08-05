# Validation evidence

Acceptance is fail-closed and each surface is recorded separately. The evidence below was observed on one physical DGX Spark with a GB10 GPU (SM121) on 2026-08-05. Image `sha256:08b1c7bbbbd3d395808810b933e05dc1997f4dbec61a858eca3467f0d5f299c5` was built from public commit `bcdc291794fccbb4b46125172ef4ba025bf484c2` without runtime edits.

| Surface | Status | Evidence |
|---|---:|---|
| Repository-built image | Pass | Image started from only target/draft/cache mounts; zero restarts; healthy |
| Full manifest verification | Pass | 48 tensor files; 106,096,777,512 tensor bytes; checksums enabled |
| Model identity | Pass | HF revision `22f28d32b9b29b4352eaa380ff8c2c170b2847ab`; served name `deepseek-v4-flash-0731-spark` |
| Target preservation | Pass | All 216 target experts and the target EXL3/Trellis tensors remain unchanged; the K64 draft is a separate directory |
| Compact draft integrity | Pass | Three complete DSpark stages; 64 routed experts per stage; saved gate/expert shapes reloaded and checked; draft SHA-256 `d72dd9d92abe2cfd2d90931072ae3b920a8f0be09465a88c839072a16d7e5cd5` |
| Weight load | Pass | 95.39 GiB reported after target plus compact draft load |
| Exact model limit | Pass | `max_model_len=262144`, `max_num_seqs=4` |
| Post-capture KV capacity | Pass | 8.01 GiB; 264,867 GPU KV tokens; 1.01x maximum concurrency at 262,144 tokens |
| CUDA graphs | Pass | `FULL_AND_PIECEWISE`; fixed K5 verifier capture size 6; eager execution disabled |
| Semantic generation | Pass | Deterministic arithmetic returned the correct result `17 × 19 = 323` |
| JSON-schema output | Pass | Exact parsed object `{"language":"Python","answer":323,"valid":true}` |
| C1 code decode | **Optimization pending** | Five 512-token clean-image trials; minimum 34.30, median 38.12, mean 39.49 tok/s |
| Cold C1 prefill | Candidate pass; release rerun pending | Same runtime path measured 252,047 uncached tokens at 1,055.45 tok/s before clean packaging |
| Thermals | Candidate pass | 89 C peak, 76.67 W peak, 2,151 MHz minimum SM clock during the prior 252k cold-prefill window |
| Dynamic draft depth | **Rejected** | A strict JSON-schema request hit a grammar-mask cardinality assertion; the published profile uses fixed K5 |
| Experimental 432-byte full generation | **Rejected** | The server booted but generated corrupted text; this route is disabled |

The machine-readable clean-image record is [`results/clean-image-acceptance.json`](results/clean-image-acceptance.json). [`results/dspark-c1-acceptance.json`](results/dspark-c1-acceptance.json) is the earlier tuning candidate. The older `acceptance.json`, `c1-c2-c4.json`, and thermal CSV record the original non-speculative baseline and are retained as provenance, not current release claims.

## Performance gate

The decode gate deliberately uses five different coding workloads—Python, Rust, TypeScript, CUDA C++, and Go—at concurrency one. Decode throughput is `(completion_tokens - 1) / (last token time - first token time)`, so TTFT is excluded. All trials generated 512 completion tokens with `thinking=false`:

| Trial | Decode tok/s |
|---:|---:|
| Python | 38.12 |
| Rust | 34.30 |
| TypeScript | 48.88 |
| CUDA C++ | 37.77 |
| Go | 38.40 |

The minimum, not aggregate throughput, is the gate. The clean-image Rust trial misses 35 tok/s, so steady decode optimization remains open even though the median and mean exceed the target.

The tuning prefill gate used a 252,047-token prompt with zero cached prompt tokens and one short completion. It returned the expected text and measured 1,055.45 prompt tok/s. Prefix-cache hits are excluded, and the value is not promoted to final-image acceptance until rerun after publication.

## Why fixed K5

The GLM-5.2 recipe established the useful pattern: a real speculative draft, a fixed verification shape captured in a FULL CUDA graph, probabilistic sampling, and optimized B12X output projection. DeepSeek V4 Flash already carries three native DSpark stages, but retaining all 216 draft experts consumed too much unified memory for a 262k KV cache. This recipe therefore derives a separate K64 draft from the preserved REAP rankings while leaving the target untouched.

Dynamic K1–K5 depth was faster for some prompts, but it made scheduler token cardinality incompatible with the pinned structured-output grammar mask. Because structured output is an acceptance requirement, the release profile fixes the draft at K5 and captures the six-row verifier shape.

## KV-format interpretation

The production recipe exports `KV_CACHE_DTYPE=nvfp4_ds_mla` to select the SparkInfer sparse-MLA control path, plus `VLLM_DSV4_PADDED_NVFP4=1` to select the proven 584-byte FP8 physical record. The control-path name is not a claim that the physical cache is true NVFP4.

The 432-byte E2M1 implementation remains available for low-level research and passes its isolated numerical test, but a kernel oracle is not full-model acceptance. It is excluded from the default Docker entrypoint because semantic generation failed.

## Release gate

Docker packaging acceptance requires the repository-built image to start without runtime edits, verify source weights, report the documented model/KV/graph configuration, pass semantic generation and strict JSON schema, remain healthy with zero restarts, and be remotely pullable by digest. Performance acceptance remains a separate gate and is still open where explicitly marked above.
