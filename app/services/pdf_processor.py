"""PDF processing service for page extraction."""

import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Tuple, Optional, AsyncIterator
import asyncio
import base64
from io import BytesIO

from PIL import Image

from app.models import PageChunk


class PDFProcessor:
    """Service for extracting pages from PDF files."""

    def __init__(self, max_pages_per_chunk: int = 10, max_concurrent: int = 5):
        self.max_pages_per_chunk = max_pages_per_chunk
        self.max_concurrent = max_concurrent

    def get_page_count(self, pdf_path: Path) -> int:
        """Get the total number of pages in a PDF."""
        with fitz.open(pdf_path) as doc:
            return len(doc)

    def get_chunks(self, pdf_path: Path) -> List[PageChunk]:
        """Split PDF into chunks for processing."""
        total_pages = self.get_page_count(pdf_path)
        chunks = []

        for start in range(0, total_pages, self.max_pages_per_chunk):
            end = min(start + self.max_pages_per_chunk, total_pages)
            chunks.append(PageChunk(
                start_page=start,
                end_page=end,
                total_pages=total_pages
            ))

        return chunks

    def extract_page_as_image(
        self,
        pdf_path: Path,
        page_number: int,
        dpi: int = 150
    ) -> Image.Image:
        """Extract a single page as a PIL Image."""
        with fitz.open(pdf_path) as doc:
            page = doc[page_number]
            # Render page to image
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)

            # Convert to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            return img

    def extract_page_as_base64(
        self,
        pdf_path: Path,
        page_number: int,
        dpi: int = 150,
        format: str = "PNG",
        for_ai: bool = False
    ) -> str:
        """Extract a page and return as base64 encoded string.
        
        Args:
            pdf_path: Path to the PDF file
            page_number: Page number to extract
            dpi: Resolution (default 150, use 100 for AI to reduce payload)
            format: Image format (PNG or JPEG)
            for_ai: If True, use lower resolution and JPEG for faster AI processing
        """
        # Use lower resolution for AI analysis (doesn't need full quality)
        actual_dpi = 100 if for_ai else dpi
        actual_format = "JPEG" if for_ai else format
        
        img = self.extract_page_as_image(pdf_path, page_number, actual_dpi)

        buffer = BytesIO()
        if actual_format == "JPEG":
            # Use quality 85 for good balance of size/quality
            img.save(buffer, format=actual_format, quality=85)
        else:
            img.save(buffer, format=actual_format)
        buffer.seek(0)

        return base64.standard_b64encode(buffer.read()).decode("utf-8")

    def extract_text_layer(self, pdf_path: Path, page_number: int) -> Optional[str]:
        """Extract text layer from a page if it exists."""
        with fitz.open(pdf_path) as doc:
            page = doc[page_number]
            text = page.get_text()

            # Return None if text is mostly whitespace (likely scanned)
            if text and len(text.strip()) > 50:
                return text.strip()
            return None

    def has_text_layer(self, pdf_path: Path) -> bool:
        """Check if PDF has a usable text layer."""
        with fitz.open(pdf_path) as doc:
            # Check first few pages
            for i in range(min(3, len(doc))):
                text = doc[i].get_text()
                if text and len(text.strip()) > 100:
                    return True
        return False

    async def extract_pages_async(
        self,
        pdf_path: Path,
        start_page: int,
        end_page: int,
        as_base64: bool = True
    ) -> AsyncIterator[Tuple[int, str]]:
        """Asynchronously extract pages from a range."""
        loop = asyncio.get_event_loop()

        for page_num in range(start_page, end_page):
            if as_base64:
                # Run in executor to avoid blocking
                result = await loop.run_in_executor(
                    None,
                    self.extract_page_as_base64,
                    pdf_path,
                    page_num
                )
            else:
                result = await loop.run_in_executor(
                    None,
                    self.extract_text_layer,
                    pdf_path,
                    page_num
                )
            yield page_num, result

    def extract_all_text(self, pdf_path: Path) -> str:
        """Extract all text from PDF (if it has a text layer)."""
        texts = []
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text = page.get_text()
                if text:
                    texts.append(f"--- Page {page.number + 1} ---\n{text}")
        return "\n\n".join(texts)

    def get_metadata(self, pdf_path: Path) -> dict:
        """Extract PDF metadata."""
        with fitz.open(pdf_path) as doc:
            return {
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "creator": doc.metadata.get("creator", ""),
                "page_count": len(doc),
            }

    async def extract_pages_parallel(
        self,
        pdf_path: Path,
        start_page: int = 0,
        end_page: Optional[int] = None,
        for_ai: bool = True,
        max_concurrent: int = 5
    ) -> List[Tuple[int, str]]:
        """Extract multiple pages in parallel for faster processing.
        
        Args:
            pdf_path: Path to the PDF file
            start_page: Starting page number (0-indexed)
            end_page: Ending page number (exclusive), None for all pages
            for_ai: If True, use optimized settings for AI (lower res, JPEG)
            max_concurrent: Maximum number of concurrent extractions
            
        Returns:
            List of (page_number, base64_image) tuples
        """
        if end_page is None:
            end_page = self.get_page_count(pdf_path)
        
        loop = asyncio.get_event_loop()
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def extract_single_page(page_num: int) -> Tuple[int, str]:
            async with semaphore:
                result = await loop.run_in_executor(
                    None,
                    lambda: self.extract_page_as_base64(
                        pdf_path, page_num, for_ai=for_ai
                    )
                )
                return (page_num, result)
        
        tasks = [
            extract_single_page(page_num)
            for page_num in range(start_page, end_page)
        ]
        
        results = await asyncio.gather(*tasks)
        return sorted(results, key=lambda x: x[0])
