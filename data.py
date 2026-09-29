"""
Shared dataset loading for the crypto Twitter sentiment task.

Source: cvnberk/bitcoin_tweets_sentiment_kaggle (a re-upload of a Kaggle
Bitcoin tweets dataset). Real, noisy social media text — spam, promotional
posts, non-English tweets, and all. Labels are Positive/Negative; the
dataset's own docs don't specify how they were generated originally, and
these kinds of scraped Twitter datasets are frequently auto-labeled with a
lexicon-based tool (e.g. VADER/TextBlob polarity) rather than human-annotated.
Treat these as a reasonable proxy for sentiment, not ground truth.
"""

DATASET_NAME = "cvnberk/bitcoin_tweets_sentiment_kaggle"
LABEL_MAP = {"Negative": 0, "Positive": 1}


def load_crypto_sentiment_dataset():
    from datasets import load_dataset

    raw = load_dataset(DATASET_NAME)

    def clean_split(split):
        split = split.filter(lambda ex: ex["Sentiment"] in LABEL_MAP)
        return split.map(lambda ex: {"label": LABEL_MAP[ex["Sentiment"]]})

    train = clean_split(raw["train"])
    test = clean_split(raw["test"])
    return train, test
