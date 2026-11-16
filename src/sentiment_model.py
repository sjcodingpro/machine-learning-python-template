"""Run a pretrained sentiment model on the WeLoveReviews data and compare it with the star ratings."""
from pathlib import Path

import pandas as pd
from transformers import pipeline

ROOT = Path(__file__).resolve().parents[1]
MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"  # 3 labels: negative, neutral, positive
ORDER = ["Positive", "Neutral", "Negative"]

df = pd.read_csv(ROOT / "data" / "processed" / "reviews_clean.csv")

# 1. Score every review with the pretrained model
classifier = pipeline("sentiment-analysis", model=MODEL)
predictions = classifier(df["review_text"].tolist(), batch_size=32, truncation=True)
df["model_label"] = [p["label"].capitalize() for p in predictions]
df["model_confidence"] = [round(p["score"], 3) for p in predictions]

# 2. Stars vs words: share of each sentiment
comparison = pd.DataFrame({
    "from_stars_%": df["rating_band"].value_counts(normalize=True).mul(100).round(1),
    "from_text_%": df["model_label"].value_counts(normalize=True).mul(100).round(1),
}).reindex(ORDER).fillna(0)
print("Sentiment share: stars vs written text")
print(comparison.to_string(), "\n")

# 3. How often do they agree?
agree = (df["model_label"] == df["rating_band"]).mean() * 100
print(f"Model label matches the star band in {agree:.1f}% of reviews\n")

print("Star rating (rows) vs model label (columns):")
print(pd.crosstab(df["rating"], df["model_label"]).reindex(columns=ORDER, fill_value=0).to_string(), "\n")

# 4. The mismatches the client cares about: high stars, not-positive words
mismatch = df[(df["rating"] >= 4) & (df["model_label"] != "Positive")]
print(f"4-5 star reviews the model did NOT read as positive: {len(mismatch)} of {(df['rating'] >= 4).sum()}")
print(mismatch[["review_id", "rating", "model_label", "review_text"]].head(5).to_string(index=False), "\n")

# 5. Save results for the report
out = ROOT / "data" / "processed" / "reviews_with_sentiment.csv"
df.to_csv(out, index=False)
print(f"Saved {len(df)} rows to {out.relative_to(ROOT)}")
