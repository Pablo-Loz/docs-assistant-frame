"""
Pydantic models for the Bot Tecnico pipeline.

Single Responsibility: Data structures only.
"""

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class Language(str, Enum):
    """Supported languages for responses."""
    SPANISH = "es"
    ENGLISH = "en"


class TriageResult(BaseModel):
    """Structured output from the Triage Agent."""

    detected_language: Language = Field(
        description="The language the user is writing in"
    )

    identified_product: Optional[str] = Field(
        default=None,
        description="The product_key if confidently identified (e.g., 'PCGH_2025_Eurovent')"
    )

    confidence: Literal["high", "low", "ambiguous"] = Field(
        description="Confidence level in product identification"
    )

    is_product_listing_request: bool = Field(
        default=False,
        description="True if user is asking what products are available"
    )

    clarification_question: Optional[str] = Field(
        default=None,
        description="Question to ask user if product is ambiguous. Must be in detected_language."
    )

    reformulated_query: str = Field(
        description="The user's query reformulated for semantic search"
    )


class ConversationContext(BaseModel):
    """Context extracted from conversation history."""
    previously_identified_product: Optional[str] = Field(
        default=None,
        description="Product identified in previous messages"
    )


class ProductInfo(BaseModel):
    """Product information discovered from ChromaDB."""
    key: str
    code: str
    year: str
    standard: str

    @property
    def description(self) -> str:
        if self.year and self.standard != self.key.replace("_", " "):
            return f"{self.code} ({self.year} - {self.standard})"
        # Fallback: just use the readable name from standard field
        return self.standard if self.standard else self.key
