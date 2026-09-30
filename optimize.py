"""
Export the fine-tuned DistilBERT model to ONNX, apply dynamic INT8
quantization, and benchmark all three variants (PyTorch fp32, ONNX fp32,
ONNX int8) on accuracy, on-disk size, and single-example inference latency.

WHY THIS MATTERS:
The PyTorch model in ./sentiment-model is fine for training and prototyping,
but it's a heavy way to serve a single classification head. ONNX Runtime
gives a leaner, framework-independent inference graph, and dynamic
quantization (compressing the linear-layer weights from float32 to int8)
shrinks that further and speeds up CPU inference — at some (measured, not
assumed) accuracy cost. This is a CPU-only comparison: dynamic quantization
targets CPU inference specifically and isn't where GPU serving spends its
time.

Run with:
    python optimize.py
"""

import os
import time

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data import load_unfair_tos_dataset

MODEL_DIR = "./sentiment-model"
ONNX_PATH = "./sentiment-model.onnx"
ONNX_QUANTIZED_PATH = "./sentiment-model-quantized.onnx"
MAX_LENGTH = 128  # must match train.py
DECISION_THRESHOLD = 0.5
LATENCY_RUNS = 100


def dir_size_mb(path):
    return sum(os.path.getsize(os.path.join(root, f)) for root, _, files in os.walk(path) for f in files) / (1024 * 1024)


def file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def export_to_onnx(model, tokenizer):
    print(f"Exporting to ONNX at {ONNX_PATH}...")
    model.eval()
    dummy = tokenizer("This is a dummy input for tracing.", return_tensors="pt", padding="max_length", max_length=MAX_LENGTH)
    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        ONNX_PATH,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=17,
    )


def quantize(onnx_path, output_path):
    print(f"Applying dynamic INT8 quantization -> {output_path}...")
    quantize_dynamic(onnx_path, output_path, weight_type=QuantType.QInt8)


def pytorch_predict_batch(model, tokenizer, texts):
    inputs = tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LENGTH)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = 1 / (1 + np.exp(-logits.numpy()))
    return (probs >= DECISION_THRESHOLD).astype(int)


def onnx_predict_batch(session, tokenizer, texts):
    inputs = tokenizer(texts, return_tensors="np", padding="max_length", truncation=True, max_length=MAX_LENGTH)
    logits = session.run(
        ["logits"],
        {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]},
    )[0]
    probs = 1 / (1 + np.exp(-logits))
    return (probs >= DECISION_THRESHOLD).astype(int)


def measure_latency(predict_one_fn, warmup=10, runs=LATENCY_RUNS):
    for _ in range(warmup):
        predict_one_fn()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        predict_one_fn()
        times.append((time.perf_counter() - start) * 1000)  # ms
    times = np.array(times)
    return {"p50_ms": float(np.percentile(times, 50)), "p95_ms": float(np.percentile(times, 95))}


def evaluate_multilabel(predict_batch_fn, texts, labels, batch_size=32):
    predictions = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        predictions.extend(predict_batch_fn(batch))
    predictions = np.array(predictions)
    return {
        "f1_micro": f1_score(labels, predictions, average="micro", zero_division=0),
        "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
    }


def main():
    print("Loading PyTorch model + tokenizer (CPU)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    pt_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    pt_model.eval()

    export_to_onnx(pt_model, tokenizer)
    quantize(ONNX_PATH, ONNX_QUANTIZED_PATH)

    print("Loading ONNX Runtime sessions (multi-threaded, for batched accuracy eval)...")
    onnx_session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    onnx_int8_session = ort.InferenceSession(ONNX_QUANTIZED_PATH, providers=["CPUExecutionProvider"])

    print("Loading unfair ToS test set for evaluation...")
    _, test_dataset = load_unfair_tos_dataset()
    texts = test_dataset["text"]
    labels = np.array(test_dataset["labels"])
    sample_text = "We may terminate your account at any time, for any reason, without notice."

    results = {}

    print("Evaluating PyTorch fp32...")
    results["pytorch_fp32"] = evaluate_multilabel(lambda b: pytorch_predict_batch(pt_model, tokenizer, b), texts, labels)
    results["pytorch_fp32"]["size_mb"] = dir_size_mb(MODEL_DIR)

    print("Evaluating ONNX fp32...")
    results["onnx_fp32"] = evaluate_multilabel(lambda b: onnx_predict_batch(onnx_session, tokenizer, b), texts, labels)
    results["onnx_fp32"]["size_mb"] = file_size_mb(ONNX_PATH)

    print("Evaluating ONNX int8...")
    results["onnx_int8"] = evaluate_multilabel(lambda b: onnx_predict_batch(onnx_int8_session, tokenizer, b), texts, labels)
    results["onnx_int8"]["size_mb"] = file_size_mb(ONNX_QUANTIZED_PATH)

    # Latency is measured separately, single-threaded (intra_op_num_threads=1),
    # matching one request per worker process in a real deployment. Letting
    # ONNX Runtime spawn a thread pool across every core for each batch-1 call
    # (the default used above for accuracy) makes per-call thread-spawn
    # overhead dominate and produces misleading numbers — it initially made
    # both ONNX variants look *slower* than plain PyTorch here.
    print("Measuring single-threaded batch=1 latency...")
    torch.set_num_threads(1)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    onnx_session_1t = ort.InferenceSession(ONNX_PATH, sess_options=so, providers=["CPUExecutionProvider"])
    onnx_int8_session_1t = ort.InferenceSession(ONNX_QUANTIZED_PATH, sess_options=so, providers=["CPUExecutionProvider"])

    results["pytorch_fp32"]["latency"] = measure_latency(lambda: pytorch_predict_batch(pt_model, tokenizer, [sample_text]))
    results["onnx_fp32"]["latency"] = measure_latency(lambda: onnx_predict_batch(onnx_session_1t, tokenizer, [sample_text]))
    results["onnx_int8"]["latency"] = measure_latency(lambda: onnx_predict_batch(onnx_int8_session_1t, tokenizer, [sample_text]))

    print(f"\n=== Results (CPU, batch=1 latency, n={len(texts)} eval examples) ===")
    header = f"{'Variant':<15}{'F1 (micro)':<12}{'F1 (macro)':<12}{'Size (MB)':<12}{'p50 (ms)':<10}{'p95 (ms)':<10}"
    print(header)
    for name, r in results.items():
        print(f"{name:<15}{r['f1_micro']:<12.4f}{r['f1_macro']:<12.4f}{r['size_mb']:<12.1f}{r['latency']['p50_ms']:<10.2f}{r['latency']['p95_ms']:<10.2f}")

    with open("optimization_results.md", "w", encoding="utf-8") as f:
        f.write("# Inference optimization results\n\n")
        f.write(
            f"CPU-only benchmark. F1 over the full {len(texts)}-example test set "
            f"(batched, multi-threaded). Latency is single-threaded "
            f"(intra_op_num_threads=1), batch size 1, {LATENCY_RUNS} runs after "
            "10 warmup calls — this matches one request per worker process in "
            "a real deployment. The multi-threaded default made both ONNX "
            "variants look slower than plain PyTorch here, since per-call "
            "thread-pool spawn overhead dominated a batch-1 workload; "
            "single-threading removes that artifact.\n\n"
        )
        f.write("| Variant | F1 (micro) | F1 (macro) | Size (MB) | p50 latency (ms) | p95 latency (ms) |\n")
        f.write("|---|---|---|---|---|---|\n")
        labels_map = {"pytorch_fp32": "PyTorch (fp32)", "onnx_fp32": "ONNX Runtime (fp32)", "onnx_int8": "ONNX Runtime (int8, dynamic quantized)"}
        for name, r in results.items():
            f.write(f"| {labels_map[name]} | {r['f1_micro']:.4f} | {r['f1_macro']:.4f} | {r['size_mb']:.1f} | {r['latency']['p50_ms']:.2f} | {r['latency']['p95_ms']:.2f} |\n")

    print("\nResults written to optimization_results.md")


if __name__ == "__main__":
    main()
