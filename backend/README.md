# SHL AI Assessment Recommender (Single Deployable FastAPI App)

Production-ready, full-stack SHL conversational recommender where FastAPI serves both API and frontend in one service.

## Architecture

- **Single service**: FastAPI backend + frontend pages served from `app/templates` and `app/static`.
- **Stateless conversation**: `/chat` only uses `messages` provided in each request.
- **RAG retrieval**: `sentence-transformers/all-MiniLM-L6-v2` embeddings + FAISS index + keyword boost reranking.
- **Guardrails**: off-topic/injection refusal and catalog-grounded output constraints.
- **Deployment**: one Dockerized Render Web Service.

## UI Routes

- `GET /` - landing page
- `GET /chat-ui` - modern chat interface (dark glassmorphism style)

## API Contract (Strict)

### `GET /health`

```json
{
  "status": "ok"
}
```

### `POST /chat`

Request:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Hiring a Java developer"
    }
  ]
}
```

Response:

```json
{
  "reply": "Here are assessments for a Java developer.",
  "recommendations": [
    {
      "name": "Java 8 (New)",
      "url": "https://www.shl.com/...",
      "test_type": "K"
    }
  ],
  "end_of_conversation": false
}
```

Rules enforced:

- `recommendations` is `[]` when clarification is needed.
- recommendation count is `1..10` when present.
- recommendations only come from local SHL catalog data.
- response schema is stable.

## Local Setup

1. Create virtual env and install deps:
   - `pip install -r requirements.txt`
2. Configure environment:
   - `cp .env.example .env`
   - set `GEMINI_API_KEY` or `OPENROUTER_API_KEY`
3. Ingest catalog and build index:
   - `python scripts/scrape_catalog.py`
   - `python scripts/build_index.py`
4. Run app:
   - `uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`
5. Open:
   - `http://localhost:8000/`
   - `http://localhost:8000/chat-ui`

## Render One-Click Deployment (Single Service)

This repo is deployment-ready with:

- `Dockerfile`
- `render.yaml`

Steps:

1. Push this project to GitHub.
2. In Render, create a **Blueprint** from repo (uses `render.yaml`).
3. Set secrets:
   - `GEMINI_API_KEY` or `OPENROUTER_API_KEY`
4. Deploy.

The container starts with:

- `uvicorn app.main:app --host 0.0.0.0 --port ${PORT}`

So Render-assigned `PORT` is used automatically.

## SHL Catalog Ingestion

Source:

- `https://www.shl.com/solutions/products/product-catalog/`

Script:

- `scripts/scrape_catalog.py`

Stored artifacts:

- `data/catalog.json`
- `data/catalog.index`
- `data/catalog_meta.pkl`

## Key Components

- `app/routes/chat.py` - strict `/chat` endpoint
- `app/routes/ui.py` - landing and chat pages
- `app/services/agent.py` - clarification/recommend/refine/compare flow
- `app/services/guardrails.py` - refusal logic
- `app/rag/catalog_store.py` - embeddings + FAISS
- `app/rag/retriever.py` - hybrid retrieval/ranking

## Evaluation and Testing

- `pytest`
- `eval/schema_validator.py`
- `eval/hallucination_checker.py`
- `eval/recall_at_10.py`
- `eval/sample_conversation_tester.py`
- `eval/conversation_suite.py` (covers clarification, refinement, refusal, and end-of-conversation checks)
- `eval/run_local_eval.py` (multi-turn API replay for schema, count, hallucination, refinement, comparison, refusal, and latency checks)

Run full local eval:

- `python eval/run_local_eval.py`

Optional env vars:

- `EVAL_BASE_URL=http://127.0.0.1:8000`
- `EVAL_LATENCY_SLO_SECONDS=30`

## Screenshot Placeholders

- `docs/screenshots/landing-page.png`
- `docs/screenshots/chat-interface.png`
- `docs/screenshots/recommendation-cards.png`
