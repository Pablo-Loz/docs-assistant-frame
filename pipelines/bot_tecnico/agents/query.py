"""
Query Agent for RAG responses.

Single Responsibility: Generate responses from retrieved context only.
"""

from pydantic_ai import Agent

from ..config import LLM_MODEL
from ..models import Language


LANGUAGE_INSTRUCTIONS = {
    Language.SPANISH: "Responde siempre en ESPANOL.",
    Language.ENGLISH: "Always respond in ENGLISH.",
}


def get_query_system_prompt(language: Language) -> str:
    """
    Get system prompt for Query Agent based on detected language.

    Args:
        language: The detected language for the response

    Returns:
        System prompt string
    """
    instruction = LANGUAGE_INSTRUCTIONS.get(
        language,
        LANGUAGE_INSTRUCTIONS[Language.ENGLISH]
    )

    return f"""You are an expert consultant. You answer questions using ONLY the provided document context.

{instruction}

FORMAT YOUR RESPONSE LIKE THIS:

## Section Title

Use **bold** for key data: numbers, dates, requirements, limits.

- Use bullet points for lists of requirements or items
- Always include specific numbers and conditions

### Subsection if needed

For structured data, use proper markdown tables:

| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Data 1   | Data 2   | Data 3   |

RULES:
- Structure the answer with ## headers for each major topic
- Use **bold** for critical information (ages, scores, deadlines, requirements)
- Use tables ONLY when the data is truly tabular (comparisons, scores, test results)
- Tables must ONLY contain the tabular data itself - NEVER put disclaimers, notes, sources, or commentary inside table cells
- Use bullet lists for requirements, steps, or enumerated items
- Include ALL specific numbers, dates, and conditions from the source - never omit details
- Mention exceptions and special cases explicitly
- If the context lacks the answer, say so clearly
- Do NOT add introductions, conclusions, disclaimers, or source citations
- Go straight to the useful content and end when the information is complete"""


def create_query_agent(language: Language) -> Agent:
    """
    Create the Query Agent for RAG responses.

    Args:
        language: The language for the response

    Returns:
        Configured Pydantic AI Agent
    """
    return Agent(
        model=LLM_MODEL,
        system_prompt=get_query_system_prompt(language),
    )


def build_query_prompt(user_question: str, context: str) -> str:
    """
    Build the prompt for the query agent.

    Args:
        user_question: The user's original question
        context: Retrieved context from documents

    Returns:
        Formatted prompt string
    """
    return f"""User Question: {user_question}

Context from Documents:
{context}

Please answer the user's question based on the document content above."""
