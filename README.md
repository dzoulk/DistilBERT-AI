# IMDB Sentiment Classification: DistilBERT vs. TF-IDF Baseline

Fine-tunes DistilBERT for binary sentiment classification (positive/negative)
on the IMDB movie review dataset, and compares it against a classical
TF-IDF + Logistic Regression baseline to show what the transformer actually
buys you.

## Results

| Model | Accuracy | F1 |
|---|---|---|
| TF-IDF + Logistic Regression (baseline) | 88.0% | 0.880 |
| DistilBERT (fine-tuned, 2 epochs) | **91.3%** | **0.913** |

Both models were trained/evaluated on the full 25,000-example train /
25,000-example test IMDB split for a fair comparison. Fine-tuning
DistilBERT improves accuracy by ~3.3 points over the baseline. That gap is
smaller than you'd see on a small subset — with enough data, a linear
bag-of-words model gets surprisingly competitive — but DistilBERT's
contextual understanding (word order, negation, sarcasm cues) still wins
out, at the cost of a much heavier model to train and serve.

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

## Docker

The image doesn't bake in the model weights — mount your trained
`./sentiment-model` directory at runtime instead:

```bash
docker build -t imdb-sentiment .
docker run -p 8000:8000 -v "$(pwd)/sentiment-model:/app/sentiment-model" imdb-sentiment
```
