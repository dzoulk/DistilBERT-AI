# Inference optimization results

CPU-only benchmark. Accuracy/F1 over 1000 test examples (batched,
multi-threaded) — a shuffled subset of the test set, so these numbers differ
slightly from the full 9,724-example evaluation in `metrics.txt`. Latency is
single-threaded (intra_op_num_threads=1), batch size 1, 100 runs after 10
warmup calls — this matches one request per worker process in a real
deployment. The multi-threaded default made both ONNX variants look slower
than plain PyTorch here, since per-call thread-pool spawn overhead dominated
a batch-1 workload; single-threading removes that artifact.

| Variant | Accuracy | F1 | Size (MB) | p50 latency (ms) | p95 latency (ms) |
|---|---|---|---|---|---|
| PyTorch (fp32) | 0.6140 | 0.6961 | 256.1 | 587.83 | 633.12 |
| ONNX Runtime (fp32) | 0.6140 | 0.6961 | 255.5 | 757.19 | 813.09 |
| ONNX Runtime (int8, dynamic quantized) | 0.6110 | 0.6963 | 64.2 | 309.80 | 408.30 |

**Takeaways:**
- Dynamic INT8 quantization shrinks the model **~4x** (256 MB → 64 MB) for a
  **0.3 point** accuracy drop (F1 is essentially unchanged: 0.6961 → 0.6963).
- Quantization also delivers a real **~1.9x** single-threaded latency
  improvement (p50 587.83ms → 309.80ms). An earlier run on a different
  fine-tuned model showed much noisier latency numbers (an 8x p50/p95
  spread) — this run's p50/p95 are close enough together to trust the
  speedup as real rather than measurement noise, but treat single-run
  laptop CPU benchmarks as directional, not precise.
- Plain ONNX Runtime (fp32, no quantization) is *slower* than PyTorch on
  CPU here, again — the win comes specifically from quantization, not from
  switching runtimes.
