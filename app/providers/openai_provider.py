"""OpenAI provider implementation."""

from typing import Optional, List, Dict, Any

from openai import AsyncOpenAI

from app.config import settings
from app.providers.base import BaseProvider


class OpenAIProvider(BaseProvider):
    """OpenAI API provider for completions and vision."""

    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
        self.vision_model = settings.openai_vision_model

    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096
    ) -> str:
        """Generate a text completion using OpenAI."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return response.choices[0].message.content

    async def analyze_image(
        self,
        image_base64: str,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """Analyze an image using OpenAI Vision."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}",
                        "detail": "high"
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        })

        response = await self.client.chat.completions.create(
            model=self.vision_model,
            messages=messages,
            max_tokens=4096
        )

        return response.choices[0].message.content

    async def embed_text(self, text: str) -> List[float]:
        """Generate embeddings using OpenAI."""
        response = await self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts (batch optimized)."""
        response = await self.client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return [item.embedding for item in response.data]

    def get_model_info(self) -> Dict[str, Any]:
        """Get OpenAI model information."""
        return {
            "provider": "OpenAI",
            "model": self.model,
            "vision_model": self.vision_model,
            "embedding_model": "text-embedding-3-small"
        }
