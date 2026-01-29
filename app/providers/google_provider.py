"""Google AI (Gemini) provider implementation."""

from typing import Optional, List, Dict, Any
import base64
import logging

import google.generativeai as genai
from PIL import Image
from io import BytesIO

from app.config import settings
from app.providers.base import BaseProvider

logger = logging.getLogger(__name__)


class GoogleProvider(BaseProvider):
    """Google Generative AI (Gemini) provider for completions and vision."""

    def __init__(self):
        genai.configure(api_key=settings.google_api_key)
        self.model_name = settings.google_model
        self.model = genai.GenerativeModel(
            self.model_name,
            system_instruction=None  # Will be set per-request if needed
        )

    def _extract_response_text(self, response) -> str:
        """Safely extract text from Gemini response, handling blocked/empty responses."""
        logger.info(f"Gemini response type: {type(response)}")

        # Try using the .text property first (simplest way for newer SDK)
        try:
            if hasattr(response, 'text') and response.text:
                logger.info(f"Using response.text, length: {len(response.text)}")
                return response.text
        except Exception as e:
            logger.debug(f"response.text access failed: {e}")

        # Check if prompt was blocked
        if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
            logger.debug(f"Prompt feedback: {response.prompt_feedback}")
            if hasattr(response.prompt_feedback, 'block_reason') and response.prompt_feedback.block_reason:
                logger.warning(f"Gemini prompt blocked: {response.prompt_feedback.block_reason}")
                raise ValueError(f"Prompt was blocked by Gemini: {response.prompt_feedback.block_reason}")

        # Check if there are candidates
        if not hasattr(response, 'candidates') or not response.candidates:
            logger.warning(f"Gemini returned no candidates. Full response: {response}")
            raise ValueError("Gemini returned no response candidates")

        candidate = response.candidates[0]
        logger.debug(f"First candidate: {candidate}")
        logger.debug(f"Candidate finish_reason: {getattr(candidate, 'finish_reason', 'N/A')}")

        # Check finish reason - some reasons indicate issues
        if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
            finish_reason_name = getattr(candidate.finish_reason, 'name', str(candidate.finish_reason))
            if finish_reason_name not in ("STOP", "MAX_TOKENS", "1", "2"):  # 1=STOP, 2=MAX_TOKENS in enum
                logger.warning(f"Gemini response finish reason: {finish_reason_name}")
                if finish_reason_name in ("SAFETY", "3"):
                    raise ValueError("Gemini response was blocked due to safety filters")

        # Extract text from parts
        if not hasattr(candidate, 'content') or not candidate.content:
            # Check if this was due to MAX_TOKENS with no output
            finish_reason_name = getattr(getattr(candidate, 'finish_reason', None), 'name', str(getattr(candidate, 'finish_reason', '')))
            if finish_reason_name in ("MAX_TOKENS", "2"):
                logger.error("Gemini hit MAX_TOKENS with no output - the model used all tokens for thinking. Try increasing max_tokens or using a simpler prompt.")
                raise ValueError("Gemini hit token limit during thinking phase - no output generated. Consider using a non-thinking model or increasing token limit.")
            logger.warning(f"Gemini response has no content. Candidate: {candidate}")
            raise ValueError("Gemini returned empty content")

        if not hasattr(candidate.content, 'parts') or not candidate.content.parts:
            # Check if this was due to MAX_TOKENS with no output
            finish_reason_name = getattr(getattr(candidate, 'finish_reason', None), 'name', str(getattr(candidate, 'finish_reason', '')))
            if finish_reason_name in ("MAX_TOKENS", "2"):
                logger.error("Gemini hit MAX_TOKENS with no output - the model used all tokens for thinking. Try increasing max_tokens or using a simpler prompt.")
                raise ValueError("Gemini hit token limit during thinking phase - no output generated. Consider using a non-thinking model or increasing token limit.")
            logger.warning(f"Gemini response has no parts. Content: {candidate.content}")
            raise ValueError("Gemini returned empty content parts")

        text_parts = []
        for part in candidate.content.parts:
            logger.debug(f"Part type: {type(part)}, Part: {str(part)[:200]}")
            # Handle different part types
            if hasattr(part, 'text') and part.text:
                text_parts.append(part.text)
            # For thinking models, there might be a 'thought' attribute we should skip
            # or the text might be in a different attribute

        result = "".join(text_parts)
        logger.info(f"Extracted text length: {len(result)}")
        logger.debug(f"Extracted text (first 500 chars): {result[:500] if result else 'EMPTY'}")

        if not result.strip():
            logger.warning("Gemini returned empty text")
            raise ValueError("Gemini returned empty text response")

        return result

    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 65536
    ) -> str:
        """Generate a text completion using Gemini.
        
        Note: For Gemini 2.5 Pro (thinking models), a high max_tokens is needed
        because the model uses tokens for internal reasoning before generating output.
        """
        generation_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens
        )

        # Use system instruction properly for Gemini
        if system_prompt:
            model = genai.GenerativeModel(
                self.model_name,
                system_instruction=system_prompt
            )
        else:
            model = self.model

        response = await model.generate_content_async(
            prompt,
            generation_config=generation_config
        )

        return self._extract_response_text(response)

    async def analyze_image(
        self,
        image_base64: str,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """Analyze an image using Gemini Vision."""
        # Decode base64 to PIL Image
        image_data = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_data))

        # Use system instruction properly for Gemini
        if system_prompt:
            model = genai.GenerativeModel(
                self.model_name,
                system_instruction=system_prompt
            )
        else:
            model = self.model

        response = await model.generate_content_async(
            [prompt, image]
        )

        return self._extract_response_text(response)

    async def embed_text(self, text: str) -> List[float]:
        """Generate embeddings using Google's embedding model."""
        result = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        result = genai.embed_content(
            model="models/embedding-001",
            content=texts,
            task_type="retrieval_document"
        )
        return result['embedding']

    def get_model_info(self) -> Dict[str, Any]:
        """Get Gemini model information."""
        return {
            "provider": "Google",
            "model": self.model_name,
            "vision_supported": True,
            "embedding_model": "embedding-001"
        }
