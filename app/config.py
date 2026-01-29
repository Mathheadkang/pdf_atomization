"""Configuration and settings for the PDF Atomization application."""

import os
from pathlib import Path
from enum import Enum
from typing import Optional, Tuple

from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class AIProvider(str, Enum):
    OPENAI = "openai"
    CLAUDE = "claude"
    GOOGLE = "google"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # AI Provider settings
    ai_provider: AIProvider = AIProvider.OPENAI
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None

    # OpenAI specific settings
    openai_model: str = "gpt-4o"
    openai_vision_model: str = "gpt-4o"

    # Claude specific settings
    claude_model: str = "claude-sonnet-4-20250514"

    # Google specific settings
    google_model: str = "gemini-1.5-pro"

    # Directory settings
    output_dir: Path = Path("./output")
    uploads_dir: Path = Path("./uploads")

    # Processing settings
    max_pages_per_chunk: int = 10
    max_concurrent_ocr: int = 5
    max_structure_text_chars: int = 300000  # ~75K tokens for structure extraction
    max_content_text_chars: int = 100000  # For per-section content extraction

    # Chunked processing settings - Token limits for large documents
    max_toc_chars: int = 60000           # ~15K tokens for TOC extraction
    max_chapter_chars: int = 80000       # ~20K tokens per chapter
    chars_per_token_estimate: float = 4  # Conservative estimate

    # Task-specific AI Configuration (optional - falls back to default ai_provider)
    structure_extractor_provider: Optional[AIProvider] = None
    structure_extractor_model: Optional[str] = None
    content_summarizer_provider: Optional[AIProvider] = None
    content_summarizer_model: Optional[str] = None

    # Atomization settings
    max_recursion_depth: int = 10
    min_content_length_for_split: int = 500

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def get_active_api_key(self) -> str:
        """Get the API key for the currently configured provider."""
        if self.ai_provider == AIProvider.OPENAI:
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY not set")
            return self.openai_api_key
        elif self.ai_provider == AIProvider.CLAUDE:
            if not self.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")
            return self.anthropic_api_key
        elif self.ai_provider == AIProvider.GOOGLE:
            if not self.google_api_key:
                raise ValueError("GOOGLE_API_KEY not set")
            return self.google_api_key
        raise ValueError(f"Unknown provider: {self.ai_provider}")

    def get_provider_for_task(self, task: str) -> Tuple[AIProvider, str]:
        """Get the provider and model for a specific task.

        Args:
            task: One of 'structure_extractor' or 'content_summarizer'

        Returns:
            Tuple of (AIProvider, model_name)
        """
        if task == "structure_extractor":
            provider = self.structure_extractor_provider or self.ai_provider
            if self.structure_extractor_model:
                model = self.structure_extractor_model
            else:
                model = self._get_default_model(provider)
        elif task == "content_summarizer":
            provider = self.content_summarizer_provider or self.ai_provider
            if self.content_summarizer_model:
                model = self.content_summarizer_model
            else:
                model = self._get_default_model(provider)
        else:
            # Default to main provider
            provider = self.ai_provider
            model = self._get_default_model(provider)

        return provider, model

    def _get_default_model(self, provider: AIProvider) -> str:
        """Get the default model for a provider."""
        if provider == AIProvider.OPENAI:
            return self.openai_model
        elif provider == AIProvider.CLAUDE:
            return self.claude_model
        elif provider == AIProvider.GOOGLE:
            return self.google_model
        raise ValueError(f"Unknown provider: {provider}")

    def get_api_key_for_provider(self, provider: AIProvider) -> str:
        """Get the API key for a specific provider."""
        if provider == AIProvider.OPENAI:
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY not set")
            return self.openai_api_key
        elif provider == AIProvider.CLAUDE:
            if not self.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")
            return self.anthropic_api_key
        elif provider == AIProvider.GOOGLE:
            if not self.google_api_key:
                raise ValueError("GOOGLE_API_KEY not set")
            return self.google_api_key
        raise ValueError(f"Unknown provider: {provider}")


# Global settings instance
settings = Settings()

# Ensure directories exist
settings.output_dir.mkdir(parents=True, exist_ok=True)
settings.uploads_dir.mkdir(parents=True, exist_ok=True)
