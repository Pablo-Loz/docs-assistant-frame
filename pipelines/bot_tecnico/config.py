"""
Configuration for the Bot Tecnico pipeline.

Single Responsibility: Configuration and paths only.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

# Resolve paths relative to this script's location
SCRIPT_DIR = Path(__file__).parent.resolve()
CHROMA_DIR = SCRIPT_DIR / "chroma_db"  # Legacy, kept for backwards compatibility

# Load environment variables
load_dotenv(SCRIPT_DIR / ".env")

# ChromaDB server configuration
# Use CHROMA_HOST env var for server mode, or None for local PersistentClient
CHROMA_HOST = os.getenv("CHROMA_HOST", None)


class Valves(BaseModel):
    """Pipeline configuration options."""
    GROQ_API_KEY: str = ""
    LOGFIRE_TOKEN: str = ""
    TOP_K_RESULTS: int = 5
    SIMILARITY_THRESHOLD: float = 0.3
    LLM_MODEL: str = "groq:llama-3.1-8b-instant"
    LLM_FALLBACK_MODEL: str = ""


# LLM model configuration
LLM_MODEL = os.getenv("LLM_MODEL", "groq:llama-3.1-8b-instant")
LLM_FALLBACK_MODEL = os.getenv("LLM_FALLBACK_MODEL", "")

# If using Cerebras (OpenAI-compatible), map CEREBRAS_API_KEY → OPENAI_API_KEY
if os.getenv("CEREBRAS_API_KEY"):
    os.environ.setdefault("OPENAI_API_KEY", os.getenv("CEREBRAS_API_KEY"))


def get_default_valves() -> Valves:
    """Create Valves with environment defaults."""
    return Valves(
        GROQ_API_KEY=os.getenv("GROQ_API_KEY", ""),
        LOGFIRE_TOKEN=os.getenv("LOGFIRE_TOKEN", ""),
        LLM_MODEL=LLM_MODEL,
        LLM_FALLBACK_MODEL=LLM_FALLBACK_MODEL,
    )
