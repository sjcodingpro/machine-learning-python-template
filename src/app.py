"""Score the WeLoveReviews reviews with a pinned pretrained sentiment model.

Run from the repository root:
    python src/app.py
"""
from pathlib import Path

import pandas as pd
from transformers import pipeline

MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
MODEL_REVISION = "8f6f4e3a8f70be4b65d3a4a8762b6d781cda240d"  # pinned Hugging Face commit, so every clone gets the same model

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "reviews.csv"
OUT_PATH = ROOT / "data" / "processed" / "reviews_with_sentiment.csv"


def stars_to_band(stars: int) -> str:
    """Map a 1-5 star value to a sentiment band: 1-2 Negative, 3 Neutral, 4-5 Positive."""
    if stars <= 2:
        return "Negative"
    if stars == 3:
        return "Neutral"
    return "Positive"


def load_classifier():
    """Load the model once. Weights download to the Hugging Face cache, not the repo."""
    return pipeline("sentiment-analysis", model=MODEL_NAME, revision=MODEL_REVISION)


def score_reviews(df: pd.DataFrame, classifier, batch_size: int = 32) -> pd.DataFrame:
    """Add the customer's band and the model's stars, band and confidence to each review."""
    predictions = classifier(df["review_text"].tolist(), batch_size=batch_size, truncation=True)
    scored = df[["review_id", "rating", "review_text"]].copy()
    scored["rating_band"] = scored["rating"].map(stars_to_band)
    scored["model_stars"] = [int(p["label"].split()[0]) for p in predictions]  # "4 stars" -> 4
    scored["model_band"] = scored["model_stars"].map(stars_to_band)
    scored["model_confidence"] = [round(p["score"], 3) for p in predictions]
    return scored


def main() -> None:
    reviews = pd.read_csv(RAW_PATH)
    classifier = load_classifier()
    scored = score_reviews(reviews, classifier)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(OUT_PATH, index=False)

    agree = (scored["model_band"] == scored["rating_band"]).mean() * 100
    print(f"Scored {len(scored)} reviews with {MODEL_NAME} @ {MODEL_REVISION}")
    print(f"Model band matches the star band in {agree:.1f}% of reviews")
    print(f"Saved to {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
