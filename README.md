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

## REST API Server

A FastAPI-based REST API is available for programmatic access to the RAG system. The API provides OpenAI/Groq-style endpoints with streaming support.

### Starting the API Server

```bash
# Install dependencies (if not already done)
pip install -r requirements.txt

# Start the API server
python api_server.py
```

The server will start on `http://localhost:8000` by default. You can change the host and port via environment variables:

```bash
export API_HOST=0.0.0.0
export API_PORT=8000
python api_server.py
```

### API Endpoints

#### Health Check
```bash
GET /health
```
Returns API status and model information.

#### Query Notices (Non-streaming)
```bash
POST /v1/query
Content-Type: application/json

{
  "query": "What are the upcoming exam dates?",
  "stream": false
}
```
Returns a complete response with sources.

#### Query Notices (Streaming)
```bash
POST /v1/query/stream
Content-Type: application/json

{
  "query": "What events are scheduled this month?",
  "stream": true
}
```
Streams the response word-by-word as it's generated.

#### Upload Document
```bash
POST /v1/documents/upload
Content-Type: multipart/form-data

file: <document file>
```
Uploads a document (image or PDF) for processing and adds it to the RAG system.
Supported formats: `.jpg`, `.jpeg`, `.png`, `.webp`, `.pdf`

#### Get Categories
```bash
GET /v1/categories
```
Returns available notice categories.

### Example Usage

See `api_client_example.py` for complete examples of how to interact with the API:

```bash
python api_client_example.py
```

Or use curl:

```bash
# Health check
curl http://localhost:8000/health

# Query (non-streaming)
curl -X POST http://localhost:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the upcoming exam dates?"}'

# Upload document
curl -X POST http://localhost:8000/v1/documents/upload \
  -F "file=@notice.pdf"
```

### Authentication

Currently, the API is open access (no authentication). API key authentication will be added in a future update.

### Deployment

The API runs locally by default. For cloud deployment:
- The API can be deployed to any cloud provider (AWS, GCP, Render, etc.)
- Ensure environment variables (GOOGLE_API_KEY, API_HOST, API_PORT) are configured
- The data directory (`data/`) should be persisted or mounted as a volume
