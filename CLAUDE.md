# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Embeddable RAG (Retrieval-Augmented Generation) chat widget for any documentation. Uses ChromaDB for vector storage with local embeddings (all-MiniLM-L6-v2) and Groq as primary LLM with automatic fallback to Cerebras on rate limits. Documents are processed from Markdown files. The chat widget can be embedded in any website via `<iframe>`.

## Running the Services

```bash
# Start services
docker compose up -d --build

# View logs
docker compose logs -f api

# Stop services
docker compose down
```

**Endpoints:**
- Chat Widget: http://localhost:9099
- Chat API: `POST http://localhost:9099/chat` and `POST http://localhost:9099/chat/stream` (SSE)
- Suggestions API: `GET http://localhost:9099/suggestions`
- ChromaDB Server: http://localhost:8000
- ChromaDB Admin UI: http://localhost:3001 (connect to `http://chromadb:8000`)

**Embedding in any website:**
```html
<iframe src="http://your-server:9099/" width="400" height="600" frameborder="0"></iframe>
```

## Architecture

```
docker-compose.yml
├── api (port 9099) - FastAPI server + static chat widget
├── chromadb (port 8000) - Vector database server, data persisted in Docker volume
└── chromadb-admin (port 3001) - Web UI for browsing vectors and running test queries

pipelines/
├── Dockerfile                  # Container build (CPU-only PyTorch for local embeddings)
├── server.py                   # FastAPI app - /chat, /chat/stream, /suggestions + static files
├── requirements.txt            # Python dependencies
├── widget/                     # Embeddable chat UI (vanilla HTML/JS/CSS)
│   ├── index.html
│   ├── style.css
│   └── app.js
├── data/                       # Markdown source files for ingestion
└── bot_tecnico/
    ├── .env                    # API keys (GROQ_API_KEY, LLM_MODEL, etc.)
    ├── .env.example            # Template with all config options
    ├── pipeline.py             # Thin orchestrator - coordinates agents and modules
    ├── config.py               # Configuration, paths, Valves class, env loading
    ├── models.py               # Pydantic models (TriageResult, Language, ProductInfo, etc.)
    ├── llm.py                  # LLM fallback logic (Groq → Cerebras on 429)
    ├── database.py             # VectorStore class - ChromaDB operations
    ├── context.py              # Conversation context extraction (clarification detection)
    ├── agents/
    │   ├── triage.py           # Triage Agent - document ID + language detection
    │   └── query.py            # Query Agent - RAG response generation
    ├── ingest.py               # Reads Markdown files, creates embeddings, stores in ChromaDB
    └── test_qa.py              # Test script for validating pipeline
```

### Module Responsibilities (Single Responsibility Principle)

- **server.py**: FastAPI app. Exposes `/chat` (JSON), `/chat/stream` (SSE line-by-line), and `/suggestions` endpoints. Serves widget static files. Runs `pipeline.pipe()` in a thread pool.
- **pipeline.py**: Thin orchestrator only. No domain logic - delegates to specialized modules. Uses `run_agent_with_fallback()` for automatic LLM fallback.
- **config.py**: `Valves` class for settings, `LLM_MODEL`/`LLM_FALLBACK_MODEL` config, `CHROMA_HOST` for server connection, Cerebras API key mapping.
- **models.py**: All Pydantic models - `TriageResult`, `Language`, `ConversationContext`, `ProductInfo`.
- **llm.py**: `run_agent_with_fallback()` - catches rate limit errors (429) and retries with fallback model via `agent.override(model=...)`.
- **database.py**: `VectorStore` class with `search()` and `discover_products()` methods. Supports both structured metadata (product_key) and simple source-based discovery. Uses `$or` filter for flexible document matching.
- **context.py**: `check_clarification_context()` and `extract_conversation_context()` functions. Supports both "product" and "document" clarification markers.
- **agents/**: Agent factory functions (`create_triage_agent`, `create_query_agent`) and generic prompts (work with any document type).
- **widget/**: Vanilla HTML/JS/CSS chat interface. SSE streaming with markdown rendering (marked.js). Clickable suggestion chips on startup. Tables wrapped in scrollable containers.

## Pipeline Development

### Setup
```bash
cp pipelines/bot_tecnico/.env.example pipelines/bot_tecnico/.env
# Edit .env with your API keys
```

### Data Ingestion
Place Markdown files in `pipelines/data/`. Any filename works, but the structured format `[CODE]_[YEAR]_[STANDARD].md` enables richer metadata. Then run:
```bash
docker compose exec api python /app/pipelines/bot_tecnico/ingest.py
```

### Testing
```bash
docker compose exec api python /app/pipelines/bot_tecnico/test_qa.py              # Default tests
docker compose exec api python /app/pipelines/bot_tecnico/test_qa.py "Question"   # Single question
docker compose exec api python /app/pipelines/bot_tecnico/test_qa.py -i           # Interactive mode
docker compose exec api python /app/pipelines/bot_tecnico/test_qa.py --products   # List documents
docker compose exec api python /app/pipelines/bot_tecnico/test_qa.py --clarification  # Test clarification flow
```

### API Usage

```bash
# Full response
curl -X POST http://localhost:9099/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Que documentos hay disponibles?", "history": []}'

# Streaming (SSE)
curl -N -X POST http://localhost:9099/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Que documentos hay disponibles?", "history": []}'

# Available documents for suggestions
curl http://localhost:9099/suggestions
```

### Verification & Full Pipeline Test

```bash
# 1. Rebuild containers
docker compose up -d --build

# 2. Run ingestion (connects to ChromaDB server)
docker compose exec api python /app/pipelines/bot_tecnico/ingest.py

# 3. Verify documents were created and indexed
docker compose exec api python /app/pipelines/bot_tecnico/test_qa.py --products

# 4. Open the chat widget
# http://localhost:9099/

# 5. Browse vectors in ChromaDB Admin UI
# Open http://localhost:3001 and connect to http://chromadb:8000
```

## CRITICAL: pipe() debe ser sincrono

**REGLA ABSOLUTA**: `pipe()` es sincrono. El servidor FastAPI lo ejecuta en un `ThreadPoolExecutor`.

```python
# CORRECTO - pipe() sincrono con run_sync()
def pipe(self, user_message: str, ...) -> str:
    result = run_agent_with_fallback(agent, prompt, fallback_model=LLM_FALLBACK_MODEL)
    return result.output
```

**PROHIBIDO:**
- `async def pipe()` - No es necesario, el server lo maneja en un thread
- `nest_asyncio` - Incompatible con uvloop de Uvicorn
- `asyncio.run()` dentro de pipe - Falla con "event loop already running"

**CORRECTO:**
- `def pipe()` sincrono
- `agent.run_sync()` de Pydantic AI (maneja async internamente)
- `run_agent_with_fallback()` para llamadas con fallback automatico

## Multi-Agent Architecture

The pipeline uses a two-agent pattern with Pydantic AI:

1. **Triage Agent** (`agents/triage.py`): Outputs `TriageResult` - identifies document from query + detects language. If ambiguous, returns a clarification question. Prompts are generic (work with any document type).
2. **Query Agent** (`agents/query.py`): Generates structured markdown response (headers, bold, tables, bullets) in detected language using retrieved context. No source citations or disclaimers.

Flow: `pipe()` → Triage → (clarification if needed) → Context search with document filter → Query Agent → Response

## LLM Fallback (Groq → Cerebras)

Automatic fallback when primary model returns rate limit errors (429):

```python
# In llm.py - run_agent_with_fallback()
try:
    return agent.run_sync(prompt)
except Exception as e:
    if is_rate_limit_error(e) and fallback_model:
        return agent.override(model=fallback_model).run_sync(prompt)
    raise
```

Config in `.env`:
```
LLM_MODEL=groq:llama-3.1-8b-instant
LLM_FALLBACK_MODEL=openai:llama-3.3-70b
CEREBRAS_API_KEY=your_key  # Mapped to OPENAI_API_KEY automatically
OPENAI_BASE_URL=https://api.cerebras.ai/v1
```

## SSE Streaming

The `/chat/stream` endpoint streams responses **line by line** (not word by word) to preserve markdown structure. The client strips exactly one SSE-standard space after `data:` and preserves all content whitespace.

## Observability (Logfire)

The pipeline uses Logfire for structured logging. Configure with `LOGFIRE_TOKEN` in `.env`.

| Level | Usage |
|-------|-------|
| `logfire.span()` | Trace execution flow (Pipeline.pipe, Triage Agent, Context Retrieval, Query Agent) |
| `logfire.info()` | Key events: queries received, triage results, documents found, response generated |
| `logfire.warn()` | Non-critical issues: empty collection, rate limit fallback triggered |
| `logfire.error()` | Failures: missing API key, file read errors, exceptions |
