# Inference optimization results

CPU-only benchmark. F1 over the full 1,607-example test set (batched,
multi-threaded). Latency is single-threaded (intra_op_num_threads=1), batch
size 1, 100 runs after 10 warmup calls — this matches one request per
worker process in a real deployment. The multi-threaded default made both
ONNX variants look slower than plain PyTorch here, since per-call
thread-pool spawn overhead dominated a batch-1 workload; single-threading
removes that artifact.

| Variant | F1 (micro) | F1 (macro) | Size (MB) | p50 latency (ms) | p95 latency (ms) |
|---|---|---|---|---|---|
| PyTorch (fp32) | 0.6240 | 0.6017 | 256.1 | 925.44 | 1031.13 |
| ONNX Runtime (fp32) | 0.6240 | 0.6017 | 255.5 | 872.71 | 1035.99 |
| ONNX Runtime (int8, dynamic quantized) | 0.6269 | 0.6046 | 64.2 | 420.90 | 451.37 |

**Takeaways:**
- Dynamic INT8 quantization shrinks the model **~4x** (256 MB → 64 MB) with
  **no measurable F1 cost** this time — quantized F1 is marginally higher
  (0.6046 vs. 0.6017 macro), which is noise on a 1,607-example test set and
  a coarse 0.5 per-category threshold, not a real improvement from
  quantizing. Unlike the two earlier sentiment-classification versions of
  this project, quantization was essentially free here.
- Quantization delivers a real **~2.2x** single-threaded latency
  improvement (p50 925ms → 421ms), consistent with the ~1.9-2x speedups
  measured on two previous fine-tuned models in this project's earlier
  iterations.
- Plain ONNX Runtime (fp32, no quantization) is close to PyTorch on CPU
  here (873ms vs. 925ms) — as in every prior run of this benchmark, the win
  comes specifically from quantization, not from switching runtimes.
