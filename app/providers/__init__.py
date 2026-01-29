"""AI Provider abstraction for different LLM backends."""

import logging
from typing import Optional

from app.config import settings, AIProvider
from app.providers.base import BaseProvider

logger = logging.getLogger(__name__)


def get_provider() -> BaseProvider:
    """Get the configured AI provider instance."""
    if settings.ai_provider == AIProvider.OPENAI:
        from app.providers.openai_provider import OpenAIProvider
        return OpenAIProvider()
    elif settings.ai_provider == AIProvider.CLAUDE:
        from app.providers.claude_provider import ClaudeProvider
        return ClaudeProvider()
    elif settings.ai_provider == AIProvider.GOOGLE:
        from app.providers.google_provider import GoogleProvider
        return GoogleProvider()
    else:
        raise ValueError(f"Unknown AI provider: {settings.ai_provider}")


def _create_provider_instance(provider: AIProvider, model: Optional[str] = None) -> BaseProvider:
    """Create a provider instance with optional model override."""
    if provider == AIProvider.OPENAI:
        from app.providers.openai_provider import OpenAIProvider
        instance = OpenAIProvider()
        if model:
            instance.model = model
        return instance
    elif provider == AIProvider.CLAUDE:
        from app.providers.claude_provider import ClaudeProvider
        instance = ClaudeProvider()
        if model:
            instance.model = model
        return instance
    elif provider == AIProvider.GOOGLE:
        from app.providers.google_provider import GoogleProvider
        instance = GoogleProvider()
        if model:
            # For Google provider, we need to update both model_name and recreate the model object
            instance.model_name = model
            import google.generativeai as genai
            instance.model = genai.GenerativeModel(model, system_instruction=None)
        return instance
    else:
        raise ValueError(f"Unknown AI provider: {provider}")


def get_provider_for_task(task: str) -> BaseProvider:
    """Get the appropriate provider instance for a specific task.

    Args:
        task: One of 'structure_extractor' or 'content_summarizer'

    Returns:
        BaseProvider instance configured for the task
    """
    provider, model = settings.get_provider_for_task(task)
    logger.info(f"Creating provider for task '{task}': provider={provider.value}, model={model}")
    return _create_provider_instance(provider, model)


__all__ = ["get_provider", "get_provider_for_task", "BaseProvider"]
