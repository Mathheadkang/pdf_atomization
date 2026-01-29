"""Abstract base class for AI providers."""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any


class BaseProvider(ABC):
    """Abstract base class for AI/LLM providers."""

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096
    ) -> str:
        """Generate a text completion.

        Args:
            prompt: The user prompt
            system_prompt: Optional system instruction
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate

        Returns:
            The generated text response
        """
        pass

    @abstractmethod
    async def analyze_image(
        self,
        image_base64: str,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """Analyze an image and generate a response.

        Args:
            image_base64: Base64 encoded image data
            prompt: The prompt describing what to extract/analyze
            system_prompt: Optional system instruction

        Returns:
            The analysis result as text
        """
        pass

    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        """Generate embeddings for text.

        Args:
            text: The text to embed

        Returns:
            List of embedding values
        """
        pass

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Default implementation calls embed_text for each text.
        Providers can override for batch optimization.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in texts:
            embedding = await self.embed_text(text)
            embeddings.append(embedding)
        return embeddings

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration."""
        return {
            "provider": self.__class__.__name__,
        }
