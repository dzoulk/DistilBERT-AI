# Unfair ToS Clause Detection with DistilBERT

This project fine-tunes DistilBERT to flag unfair clauses in Terms of Service documents, things like forced arbitration, unilateral changes to the contract, or a company reserving the right to remove your content without notice. It compares the model against a classical TF-IDF plus Logistic Regression baseline, and it benchmarks ONNX export with INT8 quantization for CPU inference.

## Dataset

The data comes from [`coastalcph/lex_glue`](https://huggingface.co/datasets/coastalcph/lex_glue), config `unfair_tos`, which is part of the [LexGLUE benchmark](https://arxiv.org/abs/2110.00976). It's built from real Terms of Service documents (Spotify, Facebook, Tinder, and others) and annotated by legal researchers. There are 5,532 training clauses and 1,607 test clauses. Unlike the sentiment datasets this project started out with, these labels come from actual legal experts instead of being scraped or auto-generated.

It's a multi-label problem. A single clause can match zero, one, or several of 8 unfair categories at once, and a clause matching none of them is just considered fair:

Limitation of liability, Unilateral termination, Unilateral change, Content removal, Contract by using, Choice of law, Jurisdiction, Arbitration

It's also heavily imbalanced. Most clauses in the dataset are fair, and within the test set, the number of examples per category ranges from 38 (Limitation of liability) down to only 7 (Arbitration).

## Results

| Model | F1 (micro) | F1 (macro) |
|---|---|---|
| TF-IDF + One-vs-Rest Logistic Regression (class_weight="balanced") | 0.593 | 0.600 |
| DistilBERT (unweighted loss) | 0.620 | 0.454 |
| DistilBERT (raw inverse-frequency pos_weight) | 0.571 | 0.555 |
| DistilBERT (sqrt-dampened pos_weight) | 0.624 | 0.602 |

Micro F1 adds up true and false positives across all 8 categories before computing a single F1 score. It's the metric LexGLUE itself reports, and it's dominated by whichever categories are most common. Macro F1 instead averages each category's F1 equally, so it's much more sensitive to how the model handles the rarest categories.

Here's what actually happened while building this, in order:

The baseline used `class_weight="balanced"` in scikit-learn's LogisticRegression from the start, because ignoring class imbalance in a linear model is a pretty basic mistake to avoid. DistilBERT's default multi-label loss, BCEWithLogitsLoss with no weighting, got no such correction, and it showed. DistilBERT beat the baseline on micro F1, but its macro F1 collapsed to 0.454, well below the baseline's 0.600. It had essentially given up on the rarest categories, since they barely move the needle in an unweighted loss.

The standard fix for this is a `pos_weight` per category, calculated as the number of negative examples divided by the number of positive examples, inside BCEWithLogitsLoss. That's the multi-label version of `class_weight="balanced"`. Applying it directly overcorrected badly. For Arbitration, which only has about 24 positive examples in training, the weight worked out to over 200, and the model started flagging clauses as Arbitration roughly five times too often, with a precision of just 0.18 against 7 true positives in the test set. Both micro and macro F1 actually got worse, dropping to 0.571 and 0.555.

Dampening that weight with a square root fixed it. DistilBERT finally beat the baseline on both metrics, 0.624 and 0.602, though only by a narrow margin on macro F1. Looking at individual categories, it clearly wins on Limitation of liability and Contract by using, holds its own elsewhere, and is still noticeably weak on Arbitration, where precision sits at 0.16. That's better than the raw-weighted version, but the model is still flagging that category more than five times too often.

The real takeaway here isn't that the transformer wins. It's that a naive multi-label setup actively loses to a properly weighted linear baseline, and getting a fair comparison took two attempts at the same fix plus an honest look at where it was still falling short.

## Project structure

- `data.py` loads the dataset and builds the multi-hot label vectors
- `baseline.py` is the TF-IDF + One-vs-Rest Logistic Regression baseline
- `train.py` fine-tunes `distilbert-base-uncased` with a class-weighted multi-label loss and saves the result to `./sentiment-model`
- `serve.py` is a FastAPI endpoint that serves the fine-tuned model
- `optimize.py` handles ONNX export, INT8 quantization, and inference benchmarking
- `requirements.txt` lists the dependencies

## Setup

```bash
python -m venv venv
source venv/bin/activate    # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

If you have an NVIDIA GPU, install a CUDA-enabled torch build instead of the default CPU wheel:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## Usage

```bash
python baseline.py   # writes baseline_metrics.txt
python train.py      # writes metrics.txt, saves model to ./sentiment-model
uvicorn serve:app --reload
```

To test the API:

```bash
curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "You agree to resolve any dispute through binding arbitration and waive your right to a jury trial."}'
```

Which returns something like this:

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

The per-category flag threshold is a runtime environment variable (`FLAG_THRESHOLD`, default 0.5) instead of a hardcoded value. How aggressively you want to flag clauses depends on whether missed unfair clauses or false alarms matter more for your use case, and this repo doesn't have a real downstream use case to optimize that against, so it's left configurable.

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

You'll need a trained model at `./sentiment-model` first (run `train.py`), since `serve.py` loads it as soon as it's imported.

## Docker

The image doesn't bake in the model weights. Instead, mount your trained `./sentiment-model` directory at runtime:

```bash
docker build -t unfair-tos-classifier .
docker run -p 8000:8000 -v "$(pwd)/sentiment-model:/app/sentiment-model" unfair-tos-classifier
```

## Inference optimization

```bash
pip install -r requirements-dev.txt
python optimize.py
```

This exports the fine-tuned model to ONNX and applies dynamic INT8 quantization, then benchmarks PyTorch fp32, ONNX fp32, and ONNX int8 against each other on F1, on-disk size, and CPU latency. The full results and methodology notes, including a benchmarking pitfall around ONNX Runtime's default multi-threading, are in [optimization_results.md](optimization_results.md).
