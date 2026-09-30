"""
Baseline model: TF-IDF + One-vs-Rest Logistic Regression on the same
unfair ToS clause data.

WHY WE DO THIS:
Anyone can fine-tune a transformer and report "90% accuracy" — but that
number is meaningless without a comparison point. This baseline answers
the interview question: "why did you need a transformer instead of
something simpler?" If the transformer doesn't meaningfully beat this,
that's actually an interesting finding worth discussing too.

TF-IDF turns each clause into a vector of word-importance scores — no
understanding of word order or context, just "which words appear, and how
distinctively." This is a MULTI-LABEL problem (a clause can match zero, one,
or several unfair categories at once), so instead of one Logistic
Regression, we train one-per-category (One-vs-Rest) and combine their
independent yes/no decisions.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.multiclass import OneVsRestClassifier

from data import CATEGORIES, load_unfair_tos_dataset


def main():
    print("Loading unfair ToS dataset...")
    train_dataset, test_dataset = load_unfair_tos_dataset()

    X_train_text = train_dataset["text"]
    y_train = np.array(train_dataset["labels"])
    X_test_text = test_dataset["text"]
    y_test = np.array(test_dataset["labels"])

    print("Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    print("Training one-vs-rest logistic regression (one classifier per category)...")
    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced"))
    clf.fit(X_train, y_train)

    print("Evaluating...")
    predictions = clf.predict(X_test)
    f1_micro = f1_score(y_test, predictions, average="micro", zero_division=0)
    f1_macro = f1_score(y_test, predictions, average="macro", zero_division=0)

    print(f"Baseline F1 (micro): {f1_micro:.4f}")
    print(f"Baseline F1 (macro): {f1_macro:.4f}")
    print("\nPer-category report:")
    print(classification_report(y_test, predictions, target_names=CATEGORIES, zero_division=0))

    with open("baseline_metrics.txt", "w") as f:
        f.write(f"f1_micro: {f1_micro}\nf1_macro: {f1_macro}\n")


if __name__ == "__main__":
    main()
