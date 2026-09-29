"""
Fine-tune DistilBERT for binary sentiment classification on crypto tweets.

WHAT THIS SCRIPT DOES, CONCEPTUALLY:
1. Loads a Bitcoin-tweets sentiment dataset (real, noisy social media text).
2. Loads a pretrained DistilBERT model + its tokenizer.
   - The tokenizer converts raw text into numeric IDs the model understands.
   - The model already "knows" English from pretraining on huge amounts of text;
     we are NOT training it from scratch.
3. Adds a small classification head on top of DistilBERT (this part IS
   trained from scratch) and fine-tunes the whole thing on our labeled data.
4. Evaluates on a held-out test set and reports accuracy + F1.
5. Saves the fine-tuned model to disk so we can serve it later via an API.
"""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from data import load_crypto_sentiment_dataset

MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "./sentiment-model"
MAX_LENGTH = 128  # tweets are short; no need for IMDB's 256-token budget


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # 1. Load the dataset
    # ------------------------------------------------------------------
    print("Loading crypto tweets sentiment dataset...")
    train_dataset, test_dataset = load_crypto_sentiment_dataset()

    # ------------------------------------------------------------------
    # 2. Tokenize the text
    # ------------------------------------------------------------------
    # Transformers don't read raw text — they read sequences of integer
    # token IDs. The tokenizer also truncates/pads tweets to a fixed
    # length so they can be batched together.
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )

    print("Tokenizing...")
    train_dataset = train_dataset.map(tokenize_fn, batched=True)
    test_dataset = test_dataset.map(tokenize_fn, batched=True)

    # ------------------------------------------------------------------
    # 3. Load the pretrained model with a classification head
    # ------------------------------------------------------------------
    # num_labels=2 tells transformers to attach a fresh linear layer
    # (768 -> 2) on top of DistilBERT's output. That layer starts with
    # random weights; the rest of the model starts with pretrained weights.
    # Fine-tuning updates BOTH — the new head learns from scratch, and the
    # pretrained layers get nudged slightly to specialize on sentiment.
    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )

    # ------------------------------------------------------------------
    # 4. Define how we measure success
    # ------------------------------------------------------------------
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions),
        }

    # ------------------------------------------------------------------
    # 5. Set training hyperparameters
    # ------------------------------------------------------------------
    # batch_size: how many examples processed at once. Tweets are short, so
    # we can afford a bigger batch than IMDB's 256-token reviews. Lower this
    # if you get an out-of-memory error on CPU/small GPU.
    # num_train_epochs: how many full passes over the training data.
    # learning_rate: how big a step the optimizer takes each update —
    # 2e-5 is a standard, safe default for fine-tuning transformers.
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # ------------------------------------------------------------------
    # 6. Train
    # ------------------------------------------------------------------
    print("Starting training...")
    trainer.train()

    # ------------------------------------------------------------------
    # 7. Final evaluation on the held-out test set
    # ------------------------------------------------------------------
    print("Evaluating on test set...")
    metrics = trainer.evaluate()
    print(f"Final test metrics: {metrics}")

    # ------------------------------------------------------------------
    # 8. Save the model + tokenizer so we can load them later for serving
    # ------------------------------------------------------------------
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")

    # Save metrics to a file too, so you have a permanent record for your
    # README.
    with open("metrics.txt", "w") as f:
        f.write(str(metrics))


if __name__ == "__main__":
    main()
