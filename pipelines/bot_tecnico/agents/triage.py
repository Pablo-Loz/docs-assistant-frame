"""
Triage Agent for document identification and language detection.

Single Responsibility: Document identification and language detection only.
"""

from pydantic_ai import Agent

from ..config import LLM_MODEL
from ..models import TriageResult


TRIAGE_SYSTEM_PROMPT = """You are a document identification and language detection specialist for a knowledge base.

Your task is to analyze user queries and:
1. Detect the language the user is writing in
2. Identify which document/source they are asking about
3. Reformulate their query for semantic search

TOOLS:
- You have access to `list_available_documents` tool. Use it when:
  - User asks "what documents do you have?" / "que documentos tienes?" / "show me what's available"
  - User wants to know what information sources are available
  - You need to show document options in your clarification question

CONTEXT RULES:
- If a "Previously discussed document" is provided and the query relates to details from it,
  USE THAT DOCUMENT with high confidence (the user is continuing the conversation)
- Only ask for clarification if the query is clearly about a DIFFERENT document

DOCUMENT IDENTIFICATION:
- If the user mentions a document code directly (e.g., "GC", "CNP"), match it to the available documents
- If the query is ambiguous AND no previous document context exists, set confidence to "ambiguous"
- The clarification question MUST be in the detected_language
- If no document is mentioned but the query is a generic question, set confidence to "low"
- Always reformulate the query to be more searchable (expand acronyms, add context)

Examples where you should USE the previous document context:
- Previous document: GC_Oposiciones_2026, Query: "What are the physical tests?" -> Use GC_Oposiciones_2026, high confidence
- Previous document: CNP_Oposiciones_2026, Query: "What's the syllabus?" -> Use CNP_Oposiciones_2026, high confidence

Examples where you should ask for clarification:
- No previous document, Query: "What are the requirements?" -> ambiguous
- Previous document: GC_Oposiciones_2026, Query: "Tell me about the CNP instead" -> Switch to CNP

Examples of clear queries (document explicitly mentioned):
- "What are the GC physical test requirements?" -> GC document, high confidence
- "CNP syllabus topics" -> CNP document, high confidence"""


def create_triage_agent(products_formatted: str) -> Agent:
    """
    Create the Triage Agent for document/language identification.

    Args:
        products_formatted: Formatted string of available documents for the tool

    Returns:
        Configured Pydantic AI Agent
    """
    agent = Agent(
        model=LLM_MODEL,
        output_type=TriageResult,
        system_prompt=TRIAGE_SYSTEM_PROMPT,
    )

    @agent.tool_plain
    def list_available_documents() -> str:
        """List all available documents in the knowledge base.

        Call this tool when:
        - User asks "what documents do you have?" or "que documentos tienes?"
        - User asks about available information sources
        - Query is ambiguous and you need to show document options

        Returns:
            Formatted list of available documents with descriptions
        """
        return f"Available documents:\n{products_formatted}"

    return agent


def build_triage_prompt(
    query: str,
    products_formatted: str,
    context_hint: str = "",
) -> str:
    """
    Build the prompt for the triage agent.

    Args:
        query: The user's query
        products_formatted: Formatted list of documents
        context_hint: Optional hint about previous document context

    Returns:
        Formatted prompt string
    """
    return f"""Analyze this user query:

User Query: {query}
{context_hint}

Available Documents:
{products_formatted}

If user is continuing a conversation about a previously discussed document, use that document.
If user asks about available documents, use the list_available_documents tool.
If unclear which document and no context, ask for clarification."""
