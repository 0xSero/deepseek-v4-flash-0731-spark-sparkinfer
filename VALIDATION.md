# Validation evidence

Acceptance is fail-closed. A check is marked complete only when observed on a physical one-GPU DGX Spark (GB10, SM121).

| Surface | Status | Evidence |
|---|---:|---|
| NVFP4 record writer | Pass | 432-byte record; reserved bytes zero |
| SparkInfer numerical oracle | Pass | C1 cosine 0.999997735; C16 cosine 0.999997914 |
| CUDA-graph replay | Pass | C1 and C16 maximum replay delta 0.0 |
| Exact 262,144-token server configuration | Pending | Awaiting final full-service run |
| Post-capture KV capacity | Pending | Awaiting final full-service run |
| Real text generation | Pending | Awaiting final full-service run |
| JSON-schema structured output | Pending | Awaiting final full-service run |
| C1/C2/C4 benchmark | Pending | Awaiting final full-service run |

The pending rows must not be interpreted as runtime support claims. They will be updated with the captured server log and benchmark JSON after the live run passes.
