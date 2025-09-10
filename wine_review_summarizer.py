"""
Wine Review Summarization Script
--------------------------------
This script loads raw wine reviews, cleans and processes the data,
groups reviews by wine variant and rating band, and generates summaries 
using a transformer summarization model (facebook/bart-large-cnn).
"""

import pandas as pd
import numpy as np
import logging
import html
from transformers import pipeline, AutoTokenizer

# ---------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Data Loading and Cleaning
# ---------------------------------------------------------------------
def load_wine_data(file_path: str) -> pd.DataFrame:
    """
    Load and process wine review data from a text file.

    Args:
        file_path (str): Path to the wine reviews text file.

    Returns:
        pd.DataFrame: Cleaned DataFrame containing wine review data.
    """
    logger.info(f"Loading data from {file_path}")
    records, current_record = [], {}

    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            line = line.strip()

            if not line:  # Blank line = end of record
                if current_record:
                    records.append(current_record)
                    current_record = {}
                continue

            if ":" in line:
                key, value = line.split(":", 1)
                current_record[key.strip()] = html.unescape(value.strip())

    if current_record:  # Catch last record
        records.append(current_record)

    logger.info(f"Total records collected: {len(records):,}")

    df = pd.DataFrame(records)
    df.columns = df.columns.str.replace("/", "_")
    df = df.rename(columns={"wine_wineId": "wine_id"})

    if "review_time" in df.columns:
        df["review_time"] = pd.to_datetime(df["review_time"].astype(int), unit="s")
        logger.info(f"Converted review_time to datetime. "
                    f"Range: {df['review_time'].min()} → {df['review_time'].max()}")

    # Remove duplicates and unnecessary columns
    df = df.drop_duplicates(subset=["wine_id", "review_userId", "review_time"], keep="first")
    df = df.drop(columns=["review_time", "review_userId", "review_userName", "wine_id"], errors="ignore")

    # Clean missing values
    df.replace("N/A", np.nan, inplace=True)
    df = df.dropna()

    # Sample down to 50,000 records
    if len(df) > 50000:
        df = df.sample(n=50000, random_state=42)

    logger.info(f"Final DataFrame shape: {df.shape}")
    return df


# ---------------------------------------------------------------------
# Rating Band Assignment
# ---------------------------------------------------------------------
def assign_rating_band(points: int) -> str:
    """
    Assign a rating band based on review points.

    Args:
        points (int): Review score.

    Returns:
        str: Rating band description.
    """
    if points < 60:
        return "50-59 (Very Poor)"
    elif points < 70:
        return "60-69 (Poor)"
    elif points < 80:
        return "70-79 (Average)"
    elif points < 85:
        return "80-84 (Good)"
    elif points < 90:
        return "85-89 (Very Good)"
    elif points < 95:
        return "90-94 (Excellent)"
    return "95-100 (Perfect)"


# ---------------------------------------------------------------------
# Summarization Utilities
# ---------------------------------------------------------------------
MODEL_NAME = "facebook/bart-large-cnn"
summarizer = pipeline("summarization", model=MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def chunk_text(texts, max_tokens: int = 900) -> list[str]:
    """
    Split long text into manageable chunks for the summarizer.

    Args:
        texts (list[str]): List of review texts.
        max_tokens (int): Maximum tokens per chunk.

    Returns:
        list[str]: Chunked review text.
    """
    combined = " ".join(texts)
    tokens = tokenizer.encode(combined, truncation=False)

    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
    return chunks


def summarize_group(row, max_len: int = 80, min_len: int = 30) -> str:
    """
    Generate a summary for a group of reviews.

    Args:
        row (pd.Series): Row containing grouped reviews and metadata.
        max_len (int): Maximum summary length.
        min_len (int): Minimum summary length.

    Returns:
        str: Generated summary.
    """
    reviews = row["review_texts"]
    intro = (
        f"Summarize the following reviews for a group of wines.\n"
    )

    chunks = chunk_text(reviews)
    summaries = []

    for chunk in chunks:
        input_len = len(tokenizer.encode(chunk, add_special_tokens=False))
        dynamic_max = min(max_len, max(min_len + 5, int(input_len * 0.8)))

        result = summarizer(
            intro + chunk,
            max_length=dynamic_max,
            min_length=min_len,
            do_sample=False,
            truncation=True,
        )
        summaries.append(result[0]["summary_text"])

    if len(summaries) > 1:  # Re-summarize combined chunks if necessary
        combined = " ".join(summaries)
        combined_chunks = chunk_text([combined])
        final_summaries = []

        for chunk in combined_chunks:
            input_len = len(tokenizer.encode(chunk, add_special_tokens=False))
            dynamic_max = min(max_len, max(min_len + 5, int(input_len * 0.8)))

            result = summarizer(
                intro + chunk,
                max_length=dynamic_max,
                min_length=min_len,
                do_sample=False,
                truncation=True,
            )
            final_summaries.append(result[0]["summary_text"])

        summary_text = " ".join(final_summaries)
    else:
        summary_text = summaries[0]

    logger.info(
        f"Successfully generated summary"
    )

    return summary_text


# ---------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------
def main():
    """Main pipeline execution."""
    file_path = "data/cellartracker.txt"

    try:
        df = load_wine_data(file_path)
        df.to_csv("data/cleaned_wine_reviews.csv", index=False)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

    # Reload cleaned data
    df = pd.read_csv("data/cleaned_wine_reviews.csv")

    # Assign rating bands
    df["rating_band"] = df["review_points"].apply(assign_rating_band)

    # Group reviews
    grouped = (
        df.groupby(["wine_variant", "rating_band"])
        .agg(
            review_texts=("review_text", list),
            avg_points=("review_points", "mean"),
            wine_years=("wine_year", lambda x: list(set(x))),
            review_count=("review_text", "size"),
        )
        .reset_index()
    )

    # Filter groups with at least 5 reviews
    filtered = grouped[grouped["review_count"] >= 5]

    # Generate sample summaries
    logger.info("Generating summaries for sample groups...")
    sample_groups = filtered.sample(n=5, random_state=42).copy()
    sample_groups["summary"] = sample_groups.apply(summarize_group, axis=1)

    # Save results
    sample_groups.to_csv("data/wine_group_summaries.csv", index=False)
    logger.info("Summaries saved to data/wine_group_summaries.csv")


if __name__ == "__main__":
    main()
