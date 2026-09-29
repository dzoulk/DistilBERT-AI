# Crypto Tweet Sentiment Classification: DistilBERT vs. TF-IDF Baseline

Fine-tunes DistilBERT for binary sentiment classification (positive/negative)
on real Bitcoin-related tweets, compares it against a classical TF-IDF +
Logistic Regression baseline, and benchmarks ONNX export + INT8 quantization
for CPU inference.

## Dataset

[`cvnberk/bitcoin_tweets_sentiment_kaggle`](https://huggingface.co/datasets/cvnberk/bitcoin_tweets_sentiment_kaggle)
— a Kaggle-sourced re-upload of ~87k real Bitcoin tweets, labeled
Positive/Negative (77,788 train / 9,724 test after dropping a handful of
stray "Neutral" rows). Real social media text: spam, promotional posts,
non-English tweets, hashtags, links, and all — much noisier than a curated
movie-review dataset like IMDB.

**Caveat worth being upfront about:** the dataset's own documentation
doesn't say how the labels were generated, and these kinds of scraped
Twitter/crypto sentiment datasets are frequently auto-labeled with a
lexicon-based tool (e.g. VADER/TextBlob polarity) rather than human-annotated.
Treat the numbers below as measuring agreement with those labels, not
"true" sentiment accuracy.

## Results

| Model | Accuracy | F1 |
|---|---|---|
| TF-IDF + Logistic Regression (baseline) | 64.4% | 0.696 |
| DistilBERT (fine-tuned, 2 epochs) | **65.2%** | **0.729** |

The transformer's edge here is much smaller than on IMDB (where it beat the
baseline by ~3.3 points) — only 0.8 points of accuracy. Digging into why:
both models are biased toward predicting "positive". DistilBERT's confusion
matrix shows it catches 90% of actual positive tweets but only 38% of actual
negative ones (precision/recall of 0.78/0.38 on the negative class). Likely
causes: noisy/inconsistent auto-generated labels cap how much any model can
learn, and short, jargon-heavy tweets (hype language like "to the moon"
appearing in both genuinely positive posts and sarcastic or spam ones) give
a transformer less contextual signal to work with than a full-length movie
review. This is a more honest, more interesting result than a clean win —
on messy real-world social data, the theoretical advantage of a transformer
doesn't automatically show up.

## Project structure

- `data.py` — shared dataset loading/cleaning for the crypto tweets data
- `baseline.py` — TF-IDF + Logistic Regression baseline
- `train.py` — fine-tunes `distilbert-base-uncased`, saves to `./sentiment-model`
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
     -d '{"text": "Bitcoin just broke $100k, this bull run is incredible!"}'
# {"label":"positive","confidence":0.98}
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
docker build -t crypto-sentiment .
docker run -p 8000:8000 -v "$(pwd)/sentiment-model:/app/sentiment-model" crypto-sentiment
```

## Inference optimization

```bash
pip install -r requirements-dev.txt
python optimize.py
```

Exports the fine-tuned model to ONNX and applies dynamic INT8 quantization,
then benchmarks PyTorch fp32 vs. ONNX fp32 vs. ONNX int8 on accuracy, on-disk
size, and CPU latency. Full results and methodology notes (including a
benchmarking pitfall around ONNX Runtime's default multi-threading) are in
[optimization_results.md](optimization_results.md). Headline result:
quantization shrinks the model ~4x (256 MB → 64 MB) with a ~1.9x latency
improvement, for only a 0.3 point accuracy cost.
