"""
Module 1, step 1 — generate a labeling sheet.

Run this after you've processed at least ~30-50 notices through the main
pipeline (so notice_store.py has data). It exports a CSV with an empty
`category` column for you to fill in by hand — the one manual step in the
whole classifier workflow.

Usage:
    python -m src.classifier.label_notices
"""
import csv

from config import LABEL_TEMPLATE_PATH, CATEGORIES
from src.notice_store import load_all_notices


def generate_labeling_template() -> str:
    notices = load_all_notices()
    if not notices:
        raise RuntimeError(
            "No notices found in the store yet. Run the main pipeline on some "
            "notice images first so there's data to label."
        )

    with open(LABEL_TEMPLATE_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "subject_line", "main_body_content", "category"])
        for n in notices:
            body_preview = (n.get("main_body_content") or "")[:300]
            writer.writerow([n["filename"], n.get("subject_line") or "", body_preview, ""])

    print(f"Wrote {len(notices)} notices to {LABEL_TEMPLATE_PATH}")
    print(f"Open it, fill the 'category' column using one of: {', '.join(CATEGORIES)}")
    print("Then save your filled copy as data/labels/labeled_notices.csv")
    return LABEL_TEMPLATE_PATH


if __name__ == "__main__":
    generate_labeling_template()
