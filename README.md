# Academic Notice RAG System

Extracts structured data from scanned academic notices (Gemini), classifies
them by category (Module 1), and answers natural-language questions about
them using hybrid retrieval. A dashboard (Module 2) is planned next.

## Setup

```bash
git clone <your-repo-url>
cd academic-notice-rag
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env           # then paste your Gemini API key into .env
```

## Project layout

```
academic-notice-rag/
├── config.py                  # paths, env vars, categories
├── main.py                    # run the pipeline + ask questions
├── src/
│   ├── extraction.py          # Gemini structured extraction
│   ├── chunking.py            # context-aware chunking
│   ├── embeddings.py          # BGE-M3 multilingual embeddings
│   ├── retrieval.py           # hybrid (semantic + BM25) retrieval
│   ├── notice_store.py        # persistent metadata store (new)
│   └── classifier/            # Module 1
│       ├── label_notices.py   # step 1: export a CSV to label
│       ├── train_classifier.py# step 2: train + evaluate
│       └── predict.py         # step 3: used live by main.py
└── data/                      # created at runtime, git-ignored
```

## Day-to-day usage

1. Drop notice images into `data/input_docs/`.
2. Run the pipeline:
   ```bash
   python main.py
   ```
   This extracts each notice, predicts its category (once trained — see
   below), stores it, embeds it, and opens a query prompt.

## Module 1 — training the classifier (one-time + occasional retrain)

1. Process at least 30–50 notices through `main.py` first, so there's data
   to label.
2. Export a labeling sheet:
   ```bash
   python -m src.classifier.label_notices
   ```
   This writes `data/labels/notices_to_label.csv`.
3. Open that CSV and fill in the `category` column for each row, using one
   of: `Exam, Holiday, Event, Circular, Admission, Other`.
4. Save your filled copy as `data/labels/labeled_notices.csv`.
5. Train:
   ```bash
   python -m src.classifier.train_classifier
   ```
   This prints an accuracy/precision/recall report, saves a confusion
   matrix image to `data/models/confusion_matrix.png`, and saves the
   trained model to `data/models/notice_classifier.joblib`.

From then on, `main.py` automatically tags every new notice using this
model — no manual step required unless you want to retrain on more data
later.

## Module 2 — dashboard

Not built yet — coming next, on top of the same `notice_store.py` data.
