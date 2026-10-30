# RNSIT AI Assistant

A React chat interface for the existing RNSIT retrieval-augmented generation project. The API uses hybrid FAISS and BM25 retrieval with weighted reciprocal rank fusion, then Groq answer generation from bounded retrieved context. Source cards are answer-level references to the deduplicated official pages and include the retrieved excerpts and ranking metadata. They do not claim claim-level attribution.

## Requirements

- Python 3.10 or newer
- Node.js 18 or newer and npm
- A Groq API key
- Existing `chunks.json` and `rnsit.index` files in the project root

## Setup

1. Create and activate a Python virtual environment, then install backend packages:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and set `GROQ_API_KEY`. Set `CORS_ORIGINS` to the exact frontend origins you use, separated by commas. The default allows Vite on localhost only. `RNSIT_MAX_PAGES` and `RNSIT_CRAWL_DELAY` configure ingestion.

3. Install frontend packages:

   ```powershell
   cd frontend
   npm install
   Copy-Item .env.example .env
   ```

   `VITE_API_URL` defaults to `http://localhost:8000` and can be changed in `frontend/.env`.

## Run locally

Open two terminals from the project root.

Backend:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

Frontend:

```powershell
cd frontend
npm run dev
```

Open the URL printed by Vite (normally `http://localhost:5173`). FastAPI loads the FAISS index, chunks, sentence-transformer model, BM25 index, and Groq client once at startup. `POST /api/chat` accepts `{"question":"...","conversation":[]}` and returns an answer plus sources containing `id`, `title`, `url`, `excerpt`, `retrieval_rank`, and `retrieval_score`. Empty questions are rejected with HTTP 422. Groq calls use a bounded timeout; upstream failures return HTTP 502 or 504.

The API returns deduplicated official pages whose excerpts were included in that answer's context. Before a source is used, its chunk ID, title, URL, and text are checked against the loaded corpus; page URLs are restricted to the official RNSIT hosts. The generator does not verify claim-to-source mappings, so citations are answer-level references rather than proof for individual sentences. `/health` reports `ok` only after all five RAG resources have loaded; it returns HTTP 503 if they are unavailable.

## Refresh the knowledge base

The checked-in corpus and FAISS index are the previous crawl snapshot: 20 pages and 467 chunks. A local audit found no empty chunks, duplicate IDs, or chunk/index positional gaps; maximum chunk length is 1,000 characters. It found nine exact duplicate-text occurrences, including one duplicate within the same page; future chunking runs skip same-page repeats while retaining repeated official content across different pages to preserve provenance. The updated crawler is bounded to the official `rnsit.ac.in` and `www.rnsit.ac.in` hosts, skips linked PDFs and non-content pages, and writes a crawl summary to `crawl_report.json`. PDF text extraction is not implemented; linked PDFs are excluded by design.

After changing the crawler or chunking code, regenerate artifacts in this order from the project root:

```powershell
python crawler.py
python chunk_pages.py
python embed_chunks.py
python build_index.py
```

`chunk_pages.py` prints page and chunk quality counts. New embedding metadata fingerprints `chunks.json`; the index builder verifies that fingerprint and writes `index_info.json` with the index type and inner-product metric. At startup, the API checks counts, dimensions, sequential chunk IDs, vector order, normalization, model name, index type, and available fingerprints. The current snapshot predates those fingerprints: `embedding_info.json` has no chunk hash and `index_info.json` is absent. Startup warns that content-to-vector alignment cannot be proven; the existing artifacts have been preserved. Rerun the commands to create fingerprinted artifacts. FAISS indexes other than `IndexFlatIP` are rejected because the configured vectors are normalized and retrieval expects inner-product cosine similarity.

Before intentionally refreshing existing artifacts, preserve `pages.json`, `chunks.json`, `embeddings.npy`, `embedding_info.json`, `rnsit.index`, and `index_info.json` (if present) in a separate dated backup directory. The commands above replace their corresponding outputs.

## Verification

Run tests that do not call the LLM:

```powershell
python -m unittest discover -s tests -v
python evaluate_retrieval.py
```

`evaluate_retrieval.py` checks page and fact recall in the top five for CSE intake (720), ISE intake (240), AI&ML intake, MCA duration, a CSE overview, a contextualized follow-up, and an unsupported query. It loads the Sentence Transformers model and may need its initial model download.

Run answer-generation checks separately when `GROQ_API_KEY` and network access are available:

```powershell
python evaluate_answers.py
```

This checks supported facts, an unsupported question, answer-level source URL/title mapping, deduplication, and API response-model compatibility. The frontend production build is `cd frontend; npm run build` after `npm install`.
