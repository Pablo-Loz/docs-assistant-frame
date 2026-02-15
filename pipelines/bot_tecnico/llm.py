"""
LLM execution with automatic fallback on rate limit errors.

Single Responsibility: Run agents with retry/fallback logic.
"""

import logfire


def _is_rate_limit_error(error: Exception) -> bool:
    """Check if an exception is a rate limit (429) error."""
    error_str = str(error).lower()
    if "429" in error_str or "rate" in error_str:
        return True
    # Check nested cause
    if error.__cause__:
        cause_str = str(error.__cause__).lower()
        if "429" in cause_str or "rate" in cause_str:
            return True
    return False


def run_agent_with_fallback(agent, prompt: str, fallback_model: str = ""):
    """
    Execute agent.run_sync(prompt) with automatic fallback on rate limit.

    If the primary model returns a 429/rate-limit error and a fallback_model
    is configured, retries the same prompt with the fallback model.

    Args:
        agent: Pydantic AI Agent instance
        prompt: The prompt to send
        fallback_model: Model ID for fallback (e.g. "openai:llama-3.3-70b")

    Returns:
        The agent run result
    """
    try:
        return agent.run_sync(prompt)
    except Exception as e:
        if _is_rate_limit_error(e) and fallback_model:
            logfire.warn(
                f"Rate limited on primary model, falling back to {fallback_model}",
                error=str(e),
            )
            return agent.override(model=fallback_model).run_sync(prompt)
        raise
