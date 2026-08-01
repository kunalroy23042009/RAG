"""
Central configuration for the academic notice RAG system.

Replaces the notebook's hardcoded Google Drive paths and inline API key.
All paths are now local to the repo (or override via .env for a server/VM).
"""
import os
from dotenv import load_dotenv

load_dotenv()  # reads .env in the project root

# --- API key -----------------------------------------------------------
# Never hardcode this. Set it in a local .env file (see .env.example),
# which is git-ignored, or as a real environment variable in production.
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key."
    )

# --- Paths ---------------------------------------------------------------
BASE_PATH = os.environ.get("RAG_BASE_PATH", os.path.join(os.path.dirname(__file__), "data"))

INPUT_PATH = os.path.join(BASE_PATH, "input_docs")
PROCESSED_PATH = os.path.join(BASE_PATH, "processed_docs")
CHROMA_PATH = os.path.join(BASE_PATH, "chroma_db")
LOG_PATH = os.path.join(BASE_PATH, "processed_log.json")
BM25_DOCS_PATH = os.path.join(BASE_PATH, "bm25_docs.json")

# New: single source of truth for structured notice data, used by both
# the classifier and the dashboard.
NOTICES_STORE_PATH = os.path.join(BASE_PATH, "notices_metadata.jsonl")

# New: classifier artifacts
LABELS_DIR = os.path.join(BASE_PATH, "labels")
LABEL_TEMPLATE_PATH = os.path.join(LABELS_DIR, "notices_to_label.csv")
LABELED_DATA_PATH = os.path.join(LABELS_DIR, "labeled_notices.csv")
MODELS_DIR = os.path.join(BASE_PATH, "models")
CLASSIFIER_MODEL_PATH = os.path.join(MODELS_DIR, "notice_classifier.joblib")
CONFUSION_MATRIX_PATH = os.path.join(MODELS_DIR, "confusion_matrix.png")

CATEGORIES = ["Exam", "Holiday", "Event", "Circular", "Admission", "Other"]

for _path in (INPUT_PATH, PROCESSED_PATH, CHROMA_PATH, LABELS_DIR, MODELS_DIR):
    os.makedirs(_path, exist_ok=True)
