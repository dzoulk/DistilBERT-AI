"""
Fine-tune DistilBERT to flag unfair clauses in Terms-of-Service documents.

WHAT THIS SCRIPT DOES, CONCEPTUALLY:
1. Loads real ToS clauses, each annotated with zero or more of 8 "unfair"
   categories (arbitration, unilateral change, content removal, etc.) - a
   MULTI-LABEL problem, not a single positive/negative choice.
2. Loads a pretrained DistilBERT model + its tokenizer.
   - The tokenizer converts raw text into numeric IDs the model understands.
   - The model already "knows" English from pretraining; we are NOT
     training it from scratch.
3. Adds a classification head with one output per category (this part IS
   trained from scratch), using a sigmoid + binary cross-entropy loss per
   category instead of a single softmax - each category is an independent
   yes/no decision, so a clause can trigger several at once.
4. Evaluates on a held-out test set and reports micro/macro F1.
5. Saves the fine-tuned model to disk so we can serve it later via an API.
"""

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from data import NUM_LABELS, load_unfair_tos_dataset

MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "./sentiment-model"
MAX_LENGTH = 128
DECISION_THRESHOLD = 0.5


class WeightedTrainer(Trainer):
    """
    Plain BCEWithLogitsLoss (what problem_type="multi_label_classification"
    uses by default) treats every category equally, so with categories this
    imbalanced the model can shrug off rare ones with near-zero loss impact.
    pos_weight upweights the loss contribution of positive examples for
    rare categories - the multi-label equivalent of the baseline's
    class_weight="balanced". Without this, DistilBERT actually loses to the
    TF-IDF baseline on macro F1, purely because the baseline got
    imbalance-correction and DistilBERT didn't.
    """

    def __init__(self, *args, pos_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(outputs.logits.device))
        loss = loss_fct(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # 1. Load the dataset
    # ------------------------------------------------------------------
    print("Loading unfair ToS dataset...")
    train_dataset, test_dataset = load_unfair_tos_dataset()

    # ------------------------------------------------------------------
    # 2. Tokenize the text
    # ------------------------------------------------------------------
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
    # 3. Load the pretrained model with a multi-label classification head
    # ------------------------------------------------------------------
    # problem_type="multi_label_classification" makes the model use
    # sigmoid + BCEWithLogitsLoss (independent per-category probabilities)
    # instead of the softmax + cross-entropy used for single-label tasks.
    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
    )

    # ------------------------------------------------------------------
    # 4. Define how we measure success
    # ------------------------------------------------------------------
    # Micro F1 aggregates true/false positives across all categories before
    # computing F1 - it's the metric LexGLUE itself reports and isn't
    # dominated by rare categories. Macro F1 (unweighted per-category
    # average) is reported too since it exposes how badly the model does on
    # the rarest categories, which micro F1 can hide.
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = 1 / (1 + np.exp(-logits))  # sigmoid
        predictions = (probs >= DECISION_THRESHOLD).astype(int)
        return {
            "f1_micro": f1_score(labels, predictions, average="micro", zero_division=0),
            "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
        }

    # ------------------------------------------------------------------
    # 5. Set training hyperparameters
    # ------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
    )

    # pos_weight[c] = sqrt((# negative examples) / (# positive examples)) for
    # each category - the standard BCEWithLogitsLoss imbalance correction,
    # mirroring the baseline's class_weight="balanced". The raw (undampened)
    # ratio badly overcorrected the rarest categories in an earlier run
    # (e.g. Arbitration, ~7 positive examples, got a weight of ~200+, and
    # the model started flagging it on ~5x too many clauses, tanking
    # precision) - the sqrt softens that without giving up on rare
    # categories entirely.
    train_labels = np.array(train_dataset["labels"])
    num_positive = train_labels.sum(axis=0)
    num_negative = len(train_labels) - num_positive
    pos_weight = torch.tensor(np.sqrt(num_negative / np.maximum(num_positive, 1)), dtype=torch.float32)
    print(f"Per-category pos_weight (imbalance correction): {pos_weight.tolist()}")

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        pos_weight=pos_weight,
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

    with open("metrics.txt", "w") as f:
        f.write(str(metrics))


if __name__ == "__main__":
    main()
