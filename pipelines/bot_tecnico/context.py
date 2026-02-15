"""
Conversation context extraction for the Bot Tecnico pipeline.

Single Responsibility: Extract context from conversation history only.
"""

from typing import Optional

import logfire

from .models import ConversationContext


# Markers that indicate a clarification question was asked
CLARIFICATION_MARKERS = [
    "producto te refieres",
    "cual de estos productos",
    "which product",
    "productos disponibles",
    "available products",
    "documento te refieres",
    "cual de estos documentos",
    "which document",
    "documentos disponibles",
    "available documents",
]


def is_clarification_question(content: str) -> bool:
    """Detect if a message is a clarification question we asked."""
    content_lower = content.lower()
    return any(marker in content_lower for marker in CLARIFICATION_MARKERS)


def check_clarification_context(messages: Optional[list]) -> Optional[dict]:
    """
    Check if the previous assistant message was a clarification question.

    Returns:
        Context dict with original_query and our_question if clarification was asked,
        None otherwise.
    """
    if not messages or len(messages) < 2:
        return None

    # Find the most recent assistant message
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") == "assistant":
            content = msg.get("content", "")

            if is_clarification_question(content):
                # Find the original user query before the clarification
                for j in range(i - 1, -1, -1):
                    if messages[j].get("role") == "user":
                        return {
                            "original_query": messages[j].get("content", ""),
                            "our_question": content,
                        }
            break

    return None


def extract_conversation_context(
    messages: Optional[list],
    available_products: list[str],
) -> ConversationContext:
    """
    Extract context from conversation history.

    Looks for previously identified products in assistant messages
    to maintain context across the conversation.

    Args:
        messages: List of conversation messages
        available_products: List of valid product keys

    Returns:
        ConversationContext with previously identified product if found
    """
    if not messages:
        return ConversationContext()

    # Search through assistant messages for product mentions
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")

            # Check if we mentioned a specific product in our response
            for product_key in available_products:
                product_code = product_key.split("_")[0]  # e.g., PCGH, PDWA

                # Look for product code in our response (indicates we answered about this product)
                if product_code in content and not is_clarification_question(content):
                    logfire.info(f"Found previously discussed product: {product_key}")
                    return ConversationContext(
                        previously_identified_product=product_key
                    )

    return ConversationContext()
