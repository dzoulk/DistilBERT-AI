# Unfair ToS Clause Detection with DistilBERT

Fine-tunes DistilBERT to flag unfair clauses in Terms-of-Service documents
(arbitration requirements, unilateral changes, content removal rights, and
more), compares it against a classical TF-IDF + Logistic Regression
baseline, and benchmarks ONNX export + INT8 quantization for CPU inference.

## Dataset

[`coastalcph/lex_glue`](https://huggingface.co/datasets/coastalcph/lex_glue)
(`unfair_tos` config) — part of the [LexGLUE benchmark](https://arxiv.org/abs/2110.00976),
built from real ToS documents (Spotify, Facebook, Tinder, and others),
annotated by legal researchers. 5,532 train / 1,607 test clauses. Unlike the
sentiment datasets this project started with, these labels are
**human-annotated by legal experts**, not scraped or auto-generated.

This is a **multi-label** problem: a clause can match zero, one, or several
of 8 unfair categories at once (a clause with no matches is "fair"):

`Limitation of liability`, `Unilateral termination`, `Unilateral change`,
`Content removal`, `Contract by using`, `Choice of law`, `Jurisdiction`,
`Arbitration`

It's also **heavily imbalanced** — most clauses are fair, and in the test
set, category support ranges from 38 examples (Limitation of liability) down
to just 7 (Arbitration).

## Results

| Model | F1 (micro) | F1 (macro) |
|---|---|---|
| TF-IDF + One-vs-Rest Logistic Regression (`class_weight="balanced"`) | 0.593 | 0.600 |
| DistilBERT (unweighted loss) | 0.620 | **0.454** |
| DistilBERT (raw inverse-frequency `pos_weight`) | 0.571 | 0.555 |
| DistilBERT (sqrt-dampened `pos_weight`) | **0.624** | **0.602** |

**Micro F1** aggregates true/false positives across all 8 categories before
computing F1 — it's the metric LexGLUE itself reports, and common categories
dominate it. **Macro F1** averages each category's F1 equally, so it's much
more sensitive to how the model does on the rarest ones.

**What actually happened, in order:**

1. The baseline used `class_weight="balanced"` in `LogisticRegression`
   from the start, since ignoring class imbalance in a linear model is an
   obvious mistake. DistilBERT's default multi-label loss
   (`BCEWithLogitsLoss`, no weighting) got no such correction — and it
   showed: DistilBERT beat the baseline on micro F1 but macro F1 collapsed
   to 0.454, well below the baseline's 0.600. It had essentially given up on
   the rarest categories, since they contribute almost nothing to
   unweighted loss.
2. The standard fix is `pos_weight` per category
   (`# negative / # positive`) in `BCEWithLogitsLoss` — the multi-label
   equivalent of `class_weight="balanced"`. Applying it raw overcorrected
   badly: for Arbitration (~24 positive examples in training), the weight
   worked out to 200+, and the model started flagging clauses as Arbitration
   about 5x too often (precision 0.18 against 7 true positives in test).
   Both micro and macro F1 got *worse* (0.571 / 0.555).
3. Dampening the weight with a square root
   (`sqrt(# negative / # positive)`) fixed it: DistilBERT finally beat the
   baseline on both metrics (0.624 / 0.602), though only narrowly on macro
   F1. Per-category, it clearly wins on "Limitation of liability" and
   "Contract by using", is competitive elsewhere, and is still notably weak
   on Arbitration (precision 0.16 — better than the raw-weight version, but
   still overcalling this rarest category more than 5x).

The takeaway isn't "the transformer wins" — it's that a naive multi-label
setup actively loses to a properly-weighted linear baseline, and getting a
fair comparison took two rounds of the same fix and a real look at where it
was still failing.

## Project structure

- `data.py` — loads the dataset and builds the multi-hot label vectors
- `baseline.py` — TF-IDF + One-vs-Rest Logistic Regression baseline
- `train.py` — fine-tunes `distilbert-base-uncased` with a class-weighted multi-label loss, saves to `./sentiment-model`
- `serve.py` — FastAPI endpoint serving the fine-tuned model
- `optimize.py` — ONNX export, INT8 quantization, and inference benchmarking
- `requirements.txt` — dependencies

## Setup

```bash
python -m venv venv
source venv/bin/activate    # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

If you have an NVIDIA GPU, install a CUDA-enabled torch build instead of the
default CPU wheel, e.g.:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## Usage

```bash
python baseline.py   # writes baseline_metrics.txt
python train.py      # writes metrics.txt, saves model to ./sentiment-model
uvicorn serve:app --reload
```

Test the API:

```bash
curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "You agree to resolve any dispute through binding arbitration and waive your right to a jury trial."}'
```

```json
{
  "is_unfair": true,
  "categories": [
    {"category": "Limitation of liability", "probability": 0.02, "flagged": false},
    {"category": "Unilateral termination", "probability": 0.01, "flagged": false},
    {"category": "Unilateral change", "probability": 0.03, "flagged": false},
    {"category": "Content removal", "probability": 0.01, "flagged": false},
    {"category": "Contract by using", "probability": 0.05, "flagged": false},
    {"category": "Choice of law", "probability": 0.04, "flagged": false},
    {"category": "Jurisdiction", "probability": 0.06, "flagged": false},
    {"category": "Arbitration", "probability": 0.91, "flagged": true}
  ]
}
```

The per-category flag threshold is a runtime environment variable
(`FLAG_THRESHOLD`, default 0.5) rather than hardcoded — how aggressively to
flag depends on whether missed unfair clauses or false alarms cost more
downstream, which this repo doesn't have a real use case to optimize
against.

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Requires a trained model at `./sentiment-model` (run `train.py` first), since
`serve.py` loads it at import time.

## Docker

The image doesn't bake in the model weights — mount your trained
`./sentiment-model` directory at runtime instead:

```bash
docker build -t unfair-tos-classifier .
docker run -p 8000:8000 -v "$(pwd)/sentiment-model:/app/sentiment-model" unfair-tos-classifier
```

## Inference optimization

```bash
pip install -r requirements-dev.txt
python optimize.py
```

Exports the fine-tuned model to ONNX and applies dynamic INT8 quantization,
then benchmarks PyTorch fp32 vs. ONNX fp32 vs. ONNX int8 on F1, on-disk
size, and CPU latency. Full results and methodology notes (including a
benchmarking pitfall around ONNX Runtime's default multi-threading) are in
[optimization_results.md](optimization_results.md).
