# Validation evidence

Acceptance is fail-closed and each surface is recorded separately. The evidence below was observed on one physical DGX Spark with a GB10 GPU (SM121) on 2026-08-05.

| Surface | Status | Evidence |
|---|---:|---|
| Published Docker image | Pass | `sha256:ff63b9b54d08df16cdc705436fdb59b3c3db4ac57e9591819e8de44ab4f36589`; healthy; zero restarts after acceptance |
| Model identity | Pass | HF revision `22f28d32b9b29b4352eaa380ff8c2c170b2847ab`; served name `deepseek-v4-flash-0731-spark` |
| Weight load | Pass | 92.13 GiB reported after load; TP1 EXL3 manifest checks passed |
| Exact model limit | Pass | `max_model_len=262144`, `max_num_seqs=4` |
| Post-capture KV capacity | Pass | 12.49 GiB available; 433,174 GPU KV tokens; 1.65x maximum concurrency at 262,144 tokens |
| CUDA graphs | Pass | `FULL_AND_PIECEWISE`; FULL and PIECEWISE captures for C1/C2/C4 |
| Real generation | Pass | Deterministic prompt returned `The capital of France is Paris.` with finish reason `stop` |
| JSON-schema output | Pass | Exact parsed object `{"city":"Paris","country":"France"}` with finish reason `stop` |
| C1/C2/C4 benchmark | Pass | 18.94 / 31.52 / 63.70 aggregate decode tok/s; full JSON committed |
| Thermals | Pass | 72 C peak, 96% peak utilization during the captured benchmark window |
| Experimental 432-byte writer oracle | Pass, diagnostic only | C1 cosine 0.999997735; C16 cosine 0.999997914; CUDA-graph replay delta 0.0 |
| Experimental 432-byte full generation | **Fail** | Server booted but generated corrupted text; this route is disabled |
| MTP speculative decode | **Unavailable** | Artifact lacks `model.layers.43.mtp_block.main_norm.weight` and related MTP payload |

## KV-format interpretation

The production recipe exports `KV_CACHE_DTYPE=nvfp4_ds_mla` to select the SparkInfer sparse-MLA control path, plus `VLLM_DSV4_PADDED_NVFP4=1` to select the proven 584-byte FP8 physical record. The name of the control path must not be read as a claim that the physical cache is true NVFP4.

The 432-byte E2M1 implementation remains available for low-level research and passes its isolated numerical test, but a kernel oracle is not full-model acceptance. It is excluded from the default Docker entrypoint because semantic generation failed.

## Performance notes

The committed C1/C2/C4 run used the same 3,589-token prompt at every concurrency. C1 was the cold run. C2 and C4 benefited from prefix caching, which explains their much lower TTFT and higher calculated prefill rate. Decode throughput is the useful comparison surface:

- C1: 18.94 aggregate tok/s; 18.94 tok/s per request.
- C2: 31.52 aggregate tok/s; 17.90 tok/s mean per request.
- C4: 63.70 aggregate tok/s; 16.17 tok/s mean per request.

This does not meet a 35–50 tok/s single-stream expectation. The repo reports the measured result rather than translating aggregate C2/C4 throughput into a C1 claim.
