"""
Shared dataset loading for the unfair Terms-of-Service clause detection task.

Source: coastalcph/lex_glue, config "unfair_tos" - part of the LexGLUE
benchmark (Chalkidis et al.). Real clauses pulled from actual ToS documents
(Spotify, Facebook, Tinder, and others), annotated by legal researchers as
fair or unfair, with unfair clauses further tagged into one or more of 8
categories. This is a MULTI-LABEL task: a clause can belong to zero
categories (fair), one, or several at once.

Heavily imbalanced: most clauses are fair (empty label list), and some
categories (e.g. "Content removal", "Choice of law") have very few positive
examples relative to others (e.g. "Arbitration", "Unilateral change").
"""

CATEGORIES = [
    "Limitation of liability",
    "Unilateral termination",
    "Unilateral change",
    "Content removal",
    "Contract by using",
    "Choice of law",
    "Jurisdiction",
    "Arbitration",
]
NUM_LABELS = len(CATEGORIES)


def _multi_hot(example):
    vector = [0.0] * NUM_LABELS
    for label_idx in example["labels"]:
        vector[label_idx] = 1.0
    return {"labels": vector}


def load_unfair_tos_dataset():
    from datasets import Sequence, Value, load_dataset

    raw = load_dataset("coastalcph/lex_glue", "unfair_tos")
    # Overwrites the raw "labels" column (a list of category indices) with a
    # multi-hot float vector - the format both the sklearn baseline and the
    # HF Trainer's multi-label loss expect. map() alone keeps the original
    # column's List(ClassLabel) (integer) schema even though we return
    # floats, which silently casts them back to int and breaks
    # BCEWithLogitsLoss ("Float can't be cast to Long") - cast_column forces
    # the schema to match what we actually produced.
    train = raw["train"].map(_multi_hot).cast_column("labels", Sequence(Value("float32")))
    test = raw["test"].map(_multi_hot).cast_column("labels", Sequence(Value("float32")))
    return train, test
