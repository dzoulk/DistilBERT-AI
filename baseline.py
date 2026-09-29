"""
Baseline model: TF-IDF + Logistic Regression on the same crypto tweets data.

WHY WE DO THIS:
Anyone can fine-tune a transformer and report "90% accuracy" — but that
number is meaningless without a comparison point. This baseline answers
the interview question: "why did you need a transformer instead of
something simpler?" If the transformer doesn't meaningfully beat this,
that's actually an interesting finding worth discussing too.

TF-IDF (Term Frequency - Inverse Document Frequency) turns each tweet
into a vector of word-importance scores — no understanding of word order
or context, just "which words appear, and how distinctively." Logistic
Regression then draws a linear boundary between positive/negative based
on those word scores.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

from data import load_crypto_sentiment_dataset


def main():
    print("Loading crypto tweets sentiment dataset...")
    train_dataset, test_dataset = load_crypto_sentiment_dataset()

    X_train_text = train_dataset["text"]
    y_train = train_dataset["label"]
    X_test_text = test_dataset["text"]
    y_test = test_dataset["label"]

    print("Vectorizing text with TF-IDF...")
    # max_features caps vocabulary size to keep this fast and memory-light.
    vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    print("Training logistic regression...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    print("Evaluating...")
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print(f"Baseline accuracy: {accuracy:.4f}")
    print(f"Baseline F1: {f1:.4f}")
    print("\nFull report:")
    print(classification_report(y_test, predictions, target_names=["negative", "positive"]))

    with open("baseline_metrics.txt", "w") as f:
        f.write(f"accuracy: {accuracy}\nf1: {f1}\n")


if __name__ == "__main__":
    main()
