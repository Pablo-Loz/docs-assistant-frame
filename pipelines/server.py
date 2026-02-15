"""
FastAPI server that wraps the existing RAG pipeline.

Exposes:
- POST /chat       → Full response (JSON)
- POST /chat/stream → Streaming response (SSE)
- GET  /           → Chat widget (static files)
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from bot_tecnico.pipeline import Pipeline

app = FastAPI(title="Bot Tecnico API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = Pipeline()
executor = ThreadPoolExecutor(max_workers=4)


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


class ChatResponse(BaseModel):
    response: str


def _run_pipe(request: ChatRequest) -> str:
    """Run pipeline.pipe() synchronously in a thread."""
    messages = request.history + [{"role": "user", "content": request.message}]
    return pipeline.pipe(
        user_message=request.message,
        messages=messages,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Full response endpoint."""
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(executor, _run_pipe, request)
    return ChatResponse(response=response)


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming response via SSE."""

    async def event_generator():
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(executor, _run_pipe, request)

        # Stream line by line to preserve markdown structure
        lines = response.split("\n")
        for i, line in enumerate(lines):
            chunk = line if i == 0 else "\n" + line
            yield {"data": chunk}
            await asyncio.sleep(0.04)

        yield {"event": "done", "data": ""}

    return EventSourceResponse(event_generator())


@app.get("/suggestions")
async def suggestions():
    """Return suggested questions based on available documents."""
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(executor, pipeline._ensure_initialized)

        items = []
        for key in pipeline._available_products:
            product = pipeline._products.get(key)
            if product:
                items.append({"key": key, "description": product.description})

        return {"documents": items}
    except Exception:
        return {"documents": []}


@app.on_event("startup")
async def startup():
    """Initialize pipeline on server start."""
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(executor, pipeline._ensure_initialized)
    except Exception as e:
        print(f"Warning: Pipeline pre-initialization failed ({e}). Will retry on first request.")


# Serve static widget files (must be mounted last to not override API routes)
widget_dir = Path(__file__).parent / "widget"
if widget_dir.exists():
    app.mount("/", StaticFiles(directory=widget_dir, html=True), name="widget")
