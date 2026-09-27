# IMDB Sentiment Classification: DistilBERT vs. TF-IDF Baseline

Fine-tunes DistilBERT for binary sentiment classification (positive/negative)
on the IMDB movie review dataset, and compares it against a classical
TF-IDF + Logistic Regression baseline to show what the transformer actually
buys you.

## Results

| Model | Accuracy | F1 |
|---|---|---|
| TF-IDF + Logistic Regression (baseline) | 83.9% | 0.839 |
| DistilBERT (fine-tuned, 2 epochs) | **88.7%** | **0.888** |

Both models were trained/evaluated on the same 4,000-example train /
1,000-example test subset (seed 42) for a fair comparison. Fine-tuning
DistilBERT improves accuracy by ~5 points over the baseline — a
transformer's contextual understanding (word order, negation, sarcasm cues)
outperforms a bag-of-words approach, though the baseline is a strong,
cheap-to-run comparison point.

## Project structure

- `baseline.py` — TF-IDF + Logistic Regression baseline
- `train.py` — fine-tunes `distilbert-base-uncased` on IMDB, saves to `./sentiment-model`
- `serve.py` — FastAPI endpoint serving the fine-tuned model
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
     -d '{"text": "This movie was absolutely wonderful, I loved it!"}'
# {"label":"positive","confidence":0.99}
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Requires a trained model at `./sentiment-model` (run `train.py` first), since
`serve.py` loads it at import time.

## Resume bullet

> Fine-tuned DistilBERT for binary sentiment classification on IMDB movie
> reviews, improving accuracy from 83.9% (TF-IDF + Logistic Regression
> baseline) to 88.7% (F1: 0.888); served the model via a FastAPI REST endpoint.
