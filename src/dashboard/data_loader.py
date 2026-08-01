"""
Module 2, data layer — turns the raw notice store into a clean DataFrame
ready for charting.

Reads from the same data/notices_metadata.jsonl file that Module 1's
predict_category() hook already writes to via main.py. No changes to the
pipeline are needed for this to work.
"""
import re
from collections import Counter
from typing import List

import pandas as pd

from src.notice_store import load_all_notices

# A small stopword list covering common English + transliterated Hindi
# filler words, so keyword frequency charts show meaningful terms.
_STOPWORDS = {
    "the", "a", "an", "to", "of", "in", "on", "for", "and", "or", "is",
    "are", "be", "will", "with", "at", "by", "from", "this", "that",
    "as", "it", "de", "ke", "ka", "ki", "hai", "ko", "se", "ki", "और",
    "के", "की", "का", "को", "से", "है", "में", "हेतु", "तथा", "एवं",
    "सभी", "द्वारा", "संबंध", "दिनांक", "किया", "जाता", "जाती",
}

_DATE_PATTERN = re.compile(r"(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})")


def _parse_date(raw: str):
    """Best-effort date parsing across the mixed dd.mm.yyyy / dd/mm/yyyy
    formats seen in these notices. Returns None if nothing matches."""
    if not raw:
        return None
    match = _DATE_PATTERN.search(str(raw))
    if not match:
        return None
    day, month, year = match.groups()
    if len(year) == 2:
        year = "20" + year
    try:
        return pd.Timestamp(year=int(year), month=int(month), day=int(day))
    except (ValueError, TypeError):
        return None


def load_dashboard_data() -> pd.DataFrame:
    """Loads every stored notice into a DataFrame with a parsed date column."""
    records = load_all_notices()
    df = pd.DataFrame(records)

    if df.empty:
        return df

    for col in ["issuing_authority", "subject_line", "main_body_content", "category"]:
        if col not in df.columns:
            df[col] = None

    df["parsed_date"] = df["date_issued"].apply(_parse_date)
    df["month"] = df["parsed_date"].dt.to_period("M").astype(str)
    df["category"] = df["category"].fillna("Uncategorized")
    df["issuing_authority"] = df["issuing_authority"].fillna("Unknown")

    return df


def monthly_counts(df: pd.DataFrame) -> pd.Series:
    dated = df.dropna(subset=["parsed_date"])
    if dated.empty:
        return pd.Series(dtype=int)
    return dated.groupby("month").size().sort_index()


def category_counts(df: pd.DataFrame) -> pd.Series:
    return df["category"].value_counts()


def authority_counts(df: pd.DataFrame, top_n: int = 8) -> pd.Series:
    return df["issuing_authority"].value_counts().head(top_n)


def top_keywords(df: pd.DataFrame, top_n: int = 10) -> List[tuple]:
    text = " ".join(
        (df["subject_line"].fillna("") + " " + df["main_body_content"].fillna(""))
        .tolist()
    )
    words = re.findall(r"[A-Za-z\u0900-\u097F]{3,}", text.lower())
    words = [w for w in words if w not in _STOPWORDS]
    return Counter(words).most_common(top_n)