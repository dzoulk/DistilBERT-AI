"""
Fine-tune DistilBERT for binary sentiment classification on the IMDB dataset.

WHAT THIS SCRIPT DOES, CONCEPTUALLY:
1. Loads the IMDB movie review dataset (25k train / 25k test, labeled pos/neg).
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
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "./sentiment-model"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # 1. Load the dataset
    # ------------------------------------------------------------------
    # The Hugging Face `datasets` library downloads and caches IMDB for us.
    # Each example is a dict: {"text": "<review text>", "label": 0 or 1}
    # label 0 = negative, label 1 = positive
    print("Loading IMDB dataset...")
    raw_datasets = load_dataset("stanfordnlp/imdb")

    # To keep training time reasonable on a laptop, we'll use a subset.
    # Feel free to increase these numbers if your machine can handle it —
    # more data generally means better accuracy, up to a point.
    train_dataset = raw_datasets["train"].shuffle(seed=42).select(range(4000))
    test_dataset = raw_datasets["test"].shuffle(seed=42).select(range(1000))

    # ------------------------------------------------------------------
    # 2. Tokenize the text
    # ------------------------------------------------------------------
    # Transformers don't read raw text — they read sequences of integer
    # token IDs. The tokenizer also truncates/pads reviews to a fixed
    # length (256 tokens here) so they can be batched together.
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=256,
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
    # batch_size: how many examples processed at once. Lower this (e.g. to 4)
    # if you get an out-of-memory error on CPU/small GPU.
    # num_train_epochs: how many full passes over the training data.
    # learning_rate: how big a step the optimizer takes each update —
    # 2e-5 is a standard, safe default for fine-tuning transformers.
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
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
    # README / resume bullet.
    with open("metrics.txt", "w") as f:
        f.write(str(metrics))


if __name__ == "__main__":
    main()
