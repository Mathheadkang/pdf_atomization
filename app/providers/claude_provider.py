"""Anthropic Claude provider implementation."""

from typing import Optional, List, Dict, Any

import anthropic

from app.config import settings
from app.providers.base import BaseProvider


class ClaudeProvider(BaseProvider):
    """Anthropic Claude API provider for completions and vision."""

    def __init__(self):
        self.client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        self.model = settings.claude_model

    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096
    ) -> str:
        """Generate a text completion using Claude."""
        message = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt if system_prompt else "",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )

        # Extract text from response
        return message.content[0].text

    async def analyze_image(
        self,
        image_base64: str,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """Analyze an image using Claude Vision."""
        message = await self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system_prompt if system_prompt else "",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )

        return message.content[0].text

    async def embed_text(self, text: str) -> List[float]:
        """Generate embeddings - Claude doesn't have native embeddings.

        Falls back to using OpenAI embeddings if available,
        otherwise raises NotImplementedError.
        """
        # Try to use OpenAI for embeddings as fallback
        if settings.openai_api_key:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=settings.openai_api_key)
            response = await client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding

        raise NotImplementedError(
            "Claude doesn't support embeddings natively. "
            "Please set OPENAI_API_KEY for embedding support."
        )

    def get_model_info(self) -> Dict[str, Any]:
        """Get Claude model information."""
        return {
            "provider": "Anthropic",
            "model": self.model,
            "vision_supported": True,
            "embeddings": "fallback_to_openai"
        }
