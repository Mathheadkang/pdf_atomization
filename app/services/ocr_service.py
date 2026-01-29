"""OCR service for extracting text from PDF pages."""

import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path

from app.models import OCRResult
from app.config import settings, AIProvider
from app.providers import get_provider


class OCRService:
    """Service for OCR processing using AI vision APIs."""

    def __init__(self):
        self.provider = get_provider()
        self.max_concurrent = settings.max_concurrent_ocr

    async def process_page(
        self,
        image_base64: str,
        page_number: int,
        extract_structure: bool = True
    ) -> OCRResult:
        """Process a single page image with OCR."""

        prompt = """Analyze this book/document page and extract:
1. All text content, preserving the original formatting as much as possible
2. If this appears to be a table of contents, chapter heading, or section title, note that

Format your response as:
TEXT:
[extracted text here]

STRUCTURE_HINTS:
[any structural information like "Chapter heading", "Table of contents", "Section 1.2", etc.]
"""

        result = await self.provider.analyze_image(
            image_base64=image_base64,
            prompt=prompt
        )

        # Parse the response
        text = ""
        structure_hints = {}

        if "TEXT:" in result:
            parts = result.split("STRUCTURE_HINTS:")
            text = parts[0].replace("TEXT:", "").strip()
            if len(parts) > 1:
                hints_text = parts[1].strip()
                structure_hints = {"raw_hints": hints_text}
        else:
            text = result.strip()

        return OCRResult(
            page_number=page_number,
            text=text,
            structure_hints=structure_hints if structure_hints else None
        )

    async def process_pages_batch(
        self,
        pages: List[tuple[int, str]],  # List of (page_number, base64_image)
        progress_callback: Optional[callable] = None
    ) -> List[OCRResult]:
        """Process multiple pages with concurrency control."""

        semaphore = asyncio.Semaphore(self.max_concurrent)
        results = []

        async def process_with_semaphore(page_num: int, image: str) -> OCRResult:
            async with semaphore:
                result = await self.process_page(image, page_num)
                if progress_callback:
                    await progress_callback(page_num)
                return result

        tasks = [
            process_with_semaphore(page_num, image)
            for page_num, image in pages
        ]

        results = await asyncio.gather(*tasks)
        return sorted(results, key=lambda r: r.page_number)

    def combine_ocr_results(self, results: List[OCRResult]) -> str:
        """Combine OCR results into a single text document."""
        combined = []
        for result in sorted(results, key=lambda r: r.page_number):
            combined.append(f"=== PAGE {result.page_number + 1} ===\n{result.text}")
        return "\n\n".join(combined)

    async def process_pdf_text(
        self,
        text: str,
        analyze_structure: bool = True
    ) -> Dict[str, Any]:
        """Process already-extracted PDF text (for PDFs with text layer)."""

        # For PDFs with text layers, we just need structure analysis
        # This is handled by the StructureExtractor
        return {
            "text": text,
            "needs_ocr": False
        }
