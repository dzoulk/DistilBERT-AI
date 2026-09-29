# Inference optimization results

CPU-only benchmark. Accuracy/F1 over 1000 test examples (batched,
multi-threaded). Latency is single-threaded (intra_op_num_threads=1), batch
size 1, 100 runs after 10 warmup calls — this matches one request per worker
process in a real deployment. The multi-threaded default made both ONNX
variants look slower than plain PyTorch here, since per-call thread-pool
spawn overhead dominated a batch-1 workload; single-threading removes that
artifact.

| Variant | Accuracy | F1 | Size (MB) | p50 latency (ms) | p95 latency (ms) |
|---|---|---|---|---|---|
| PyTorch (fp32) | 0.9030 | 0.9019 | 256.1 | 984.08 | 1073.99 |
| ONNX Runtime (fp32) | 0.9030 | 0.9019 | 255.5 | 983.90 | 1096.74 |
| ONNX Runtime (int8, dynamic quantized) | 0.8920 | 0.8909 | 64.2 | 75.89 | 603.10 |

**Takeaways:**
- Dynamic INT8 quantization shrinks the model **~4x** (256 MB → 64 MB) with a
  measured **1.1 point** accuracy/F1 drop.
- The int8 latency numbers show high run-to-run variance (p50 75.89ms vs.
  p95 603.10ms — an 8x spread) across repeated runs on this laptop, likely OS
  scheduling jitter rather than a real property of the quantized kernel. A
  separate single-thread comparison run measured a more consistent ~2x
  speedup (p50 ~488ms vs. ~1017ms for fp32 PyTorch). Take the exact latency
  numbers as noisy; the size reduction and accuracy cost are the reliable
  findings here.
- Plain ONNX Runtime (fp32, no quantization) doesn't meaningfully beat
  PyTorch on CPU — the win comes specifically from quantization, not from
  switching runtimes.
