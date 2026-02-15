# docs-assistant-frame

Widget de chat RAG (Retrieval-Augmented Generation) embebible via iframe. Sube documentos Markdown, haz ingestion, y obtén un chatbot inteligente sobre tu documentacion.

## Stack

- **LLM**: Groq (llama-3.1-8b-instant) con fallback automatico a Cerebras (llama-3.3-70b)
- **Embeddings**: all-MiniLM-L6-v2 (local, sin API)
- **Vector Store**: ChromaDB (servidor dedicado)
- **Agent Framework**: Pydantic AI (multi-agente: Triage + Query)
- **Frontend**: Widget HTML/JS/CSS vanilla (embebible via iframe)
- **Servidor**: FastAPI con SSE streaming
- **Observabilidad**: Logfire (opcional)

## Arquitectura

```
┌──────────────────────────────────┐
│         Chat Widget (iframe)     │  ← Embebible en cualquier web
│         http://localhost:9099    │
└──────────────┬───────────────────┘
               │ SSE / JSON
┌──────────────▼───────────────────┐
│        FastAPI (port 9099)       │
│  /chat  /chat/stream  /suggestions│
│                                  │
│  ┌────────────────────────────┐  │
│  │    Bot Tecnico Pipeline    │  │
│  │  ┌──────────┐ ┌─────────┐ │  │
│  │  │ Triage   │→│ Query   │ │  │
│  │  │ Agent    │ │ Agent   │ │  │
│  │  └──────────┘ └─────────┘ │  │
│  │         │           │      │  │
│  │         ▼           ▼      │  │
│  │  ┌────────────────────────┐│  │
│  │  │  ChromaDB (vectores)   ││  │
│  │  │  all-MiniLM-L6-v2     ││  │
│  │  └────────────────────────┘│  │
│  └────────────────────────────┘  │
└──────────────────────────────────┘
```

## Inicio Rapido

### 1. Configuracion

```bash
git clone <repo-url>
cd docs-assistant-frame

cp pipelines/bot_tecnico/.env.example pipelines/bot_tecnico/.env
# Editar .env con tus API keys (ver .env.example para detalles)
```

### 2. Agregar Documentos

Coloca archivos Markdown en `pipelines/data/`. Cualquier nombre funciona, pero el formato recomendado es:

```
[CODIGO]_[NOMBRE]_[AÑO].md
```

Ejemplo: `GC_Oposiciones_2026.md`

### 3. Iniciar Servicios

```bash
docker compose up -d --build

# Ejecutar ingestion de documentos
docker compose exec api python /app/pipelines/bot_tecnico/ingest.py

# Verificar documentos indexados
docker compose exec api python /app/pipelines/bot_tecnico/test_qa.py --products
```

### 4. Usar

- **Chat Widget**: http://localhost:9099
- **ChromaDB Admin**: http://localhost:3001 (conectar a `http://chromadb:8000`)

### 5. Embeber en cualquier web

```html
<iframe src="http://tu-servidor:9099/" width="400" height="600" frameborder="0"></iframe>
```

## Servicios Docker

| Servicio | Puerto | Descripcion |
|----------|--------|-------------|
| api | 9099 | FastAPI + chat widget |
| chromadb | 8000 | Base de datos vectorial |
| chromadb-admin | 3001 | UI para explorar vectores |

## API

```bash
# Respuesta completa
curl -X POST http://localhost:9099/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Que documentos hay disponibles?", "history": []}'

# Streaming (SSE)
curl -N -X POST http://localhost:9099/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Que documentos hay disponibles?", "history": []}'

# Sugerencias (para el widget)
curl http://localhost:9099/suggestions
```

## Estructura del Proyecto

```
docs-assistant-frame/
├── docker-compose.yml
├── CLAUDE.md                       # Instrucciones para Claude Code
├── README.md
└── pipelines/
    ├── Dockerfile
    ├── server.py                   # FastAPI: /chat, /chat/stream, /suggestions
    ├── requirements.txt
    ├── widget/                     # Chat UI embebible
    │   ├── index.html
    │   ├── style.css
    │   └── app.js
    ├── data/                       # Documentos Markdown fuente
    └── bot_tecnico/
        ├── .env                    # API keys (no versionado)
        ├── .env.example            # Plantilla de configuracion
        ├── pipeline.py             # Orquestador (thin orchestrator)
        ├── config.py               # Configuracion y Valves
        ├── models.py               # Modelos Pydantic
        ├── llm.py                  # Fallback automatico Groq → Cerebras
        ├── database.py             # VectorStore (ChromaDB)
        ├── context.py              # Contexto de conversacion
        ├── agents/
        │   ├── triage.py           # Triage Agent (identificacion + idioma)
        │   └── query.py            # Query Agent (respuesta RAG)
        ├── ingest.py               # Script de ingestion
        └── test_qa.py              # Script de pruebas
```

## Flujo del Pipeline

1. **Triage Agent**: Identifica el documento relevante y detecta el idioma del usuario
2. **Clarificacion** (si hay ambiguedad): Pregunta al usuario que documento quiere consultar
3. **Busqueda en ChromaDB**: Recupera contexto relevante filtrado por documento
4. **Query Agent**: Genera respuesta estructurada (markdown con headers, tablas, bold)

## Fallback Automatico

Si Groq devuelve un error 429 (rate limit), el sistema cambia automaticamente a Cerebras para esa peticion. Configurable via `.env`:

```
LLM_MODEL=groq:llama-3.1-8b-instant          # Primario
LLM_FALLBACK_MODEL=openai:llama-3.3-70b      # Fallback (Cerebras via OpenAI API)
```

## Comandos Utiles

```bash
# Reiniciar tras cambios
docker compose up -d --build api

# Ver logs
docker compose logs -f api

# Tests
docker compose exec api python /app/pipelines/bot_tecnico/test_qa.py              # Default
docker compose exec api python /app/pipelines/bot_tecnico/test_qa.py "Pregunta"   # Una pregunta
docker compose exec api python /app/pipelines/bot_tecnico/test_qa.py -i           # Interactivo

# Re-ingestar documentos
docker compose exec api python /app/pipelines/bot_tecnico/ingest.py
```

## Troubleshooting

### Error "No documents available"
```bash
docker compose exec api python /app/pipelines/bot_tecnico/ingest.py
```

### Error de API key
Verifica que `pipelines/bot_tecnico/.env` tiene `GROQ_API_KEY` valida.

### Verificar ChromaDB
```bash
docker compose exec api python -c "
import chromadb
client = chromadb.HttpClient(host='chromadb', port=8000)
col = client.get_or_create_collection('technical_manuals')
print(f'Documents: {col.count()}')
"
```
