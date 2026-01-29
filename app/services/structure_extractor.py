"""Structure extraction service using LLM analysis."""

import json
import logging
import re
from typing import Optional, Callable, Awaitable, List
import uuid

from app.models import (
    DocumentStructure,
    StructureNode,
    SectionType,
    ContentCategory
)
from app.providers import get_provider_for_task
from app.config import settings

logger = logging.getLogger(__name__)


class StructureExtractor:
    """Service for extracting document structure using LLM."""

    def __init__(self):
        self.provider = get_provider_for_task("structure_extractor")

    async def extract_structure(
        self,
        text: str,
        title_hint: Optional[str] = None,
        author_hint: Optional[str] = None,
        progress_callback: Optional[Callable[[str, float], Awaitable[None]]] = None
    ) -> DocumentStructure:
        """Extract hierarchical structure using chunked processing for large documents."""

        # Check if document is small enough for single-pass
        if len(text) <= settings.max_toc_chars:
            logger.info(f"Document size ({len(text)} chars) within limit, using single-pass extraction")
            return await self._extract_structure_single_pass(text, title_hint, author_hint)

        logger.info(f"Document size ({len(text)} chars) exceeds limit, using two-pass chunked extraction")

        # Large document: use two-pass approach
        # Pass 1: Extract TOC
        if progress_callback:
            await progress_callback("Extracting document outline...", 0.0)

        toc = await self.extract_toc(text, title_hint)
        logger.info(f"TOC extracted: {len(toc.get('chapters', []))} chapters found")

        # Pass 2: Process each chapter
        chapter_nodes = []
        chapters = toc.get("chapters", [])
        total_chapters = len(chapters)

        for i, chapter in enumerate(chapters):
            chapter_title = chapter.get("title", f"Chapter {i+1}")
            section_titles = chapter.get("sections", [])

            if progress_callback:
                progress = (i + 1) / (total_chapters + 1)
                await progress_callback(f"Processing: {chapter_title[:50]}...", progress)

            logger.info(f"Processing chapter {i+1}/{total_chapters}: {chapter_title}")

            node = await self.extract_chapter_structure(
                text,
                chapter_title,
                section_titles
            )
            chapter_nodes.append(node)

        # Build complete structure
        if progress_callback:
            await progress_callback("Building final structure...", 0.95)

        return self._build_structure_from_chapters(toc, chapter_nodes, text)

    async def extract_toc(self, text: str, title_hint: Optional[str] = None) -> dict:
        """Extract only the table of contents / high-level structure."""

        prompt = f"""Analyze this document and extract ONLY the hierarchical outline.

Return chapter and section TITLES only (no content). Include all levels of hierarchy you can identify.

{"The document title might be: " + title_hint if title_hint else "Detect the document title from content."}

Return JSON in this exact format:
{{
    "title": "Book Title",
    "author": "Author Name or null",
    "chapters": [
        {{
            "title": "Chapter 1: Introduction",
            "category": "knowledge",
            "sections": ["1.1 Background", "1.2 Overview", "1.3 Summary"]
        }},
        {{
            "title": "Preface",
            "category": "meta",
            "sections": []
        }}
    ]
}}

Valid categories:
- "knowledge": Contains substantive educational/informational content
- "meta": Preface, foreword, acknowledgements, table of contents, index, copyright, about author, etc.

IMPORTANT: Include ALL chapters/major sections you can identify, even from the beginning and end of the document.

DOCUMENT TEXT (beginning portion):
{text[:settings.max_toc_chars]}
"""

        response = await self.provider.complete(prompt)
        return self._parse_json_response(response)

    async def extract_chapter_structure(
        self,
        full_text: str,
        chapter_title: str,
        section_titles: list[str]
    ) -> StructureNode:
        """Extract detailed structure for a single chapter."""

        # Find chapter boundaries in text
        chapter_text = self._extract_chapter_text(full_text, chapter_title)

        if len(chapter_text) > settings.max_chapter_chars:
            logger.warning(f"Chapter '{chapter_title}' exceeds limit ({len(chapter_text)} chars), truncating")
            chapter_text = chapter_text[:settings.max_chapter_chars]

        sections_hint = ""
        if section_titles:
            sections_hint = f"\nKnown sections in this chapter: {', '.join(section_titles[:10])}"

        prompt = f"""Extract the complete hierarchical structure for this chapter: "{chapter_title}"
{sections_hint}

Return JSON with full hierarchy including subsections and content summaries.

Return format:
{{
    "title": "{chapter_title}",
    "type": "chapter",
    "level": 1,
    "category": "knowledge",
    "content_summary": "Brief summary of chapter content",
    "children": [
        {{
            "title": "Section Title",
            "type": "section",
            "level": 2,
            "category": "knowledge",
            "content_summary": "Section summary",
            "children": [
                {{
                    "title": "Subsection Title",
                    "type": "subsection",
                    "level": 3,
                    "category": "knowledge",
                    "content_summary": "Subsection summary",
                    "children": []
                }}
            ]
        }}
    ]
}}

Valid types: "chapter", "section", "subsection", "content"
Valid categories: "knowledge", "meta"

IMPORTANT: Capture ALL levels of hierarchy present. Do not flatten subsections.

CHAPTER TEXT:
{chapter_text}
"""

        response = await self.provider.complete(prompt)
        chapter_data = self._parse_json_response(response)

        # Convert to StructureNode
        return self._build_chapter_node(chapter_data, chapter_title)

    def _extract_chapter_text(self, full_text: str, chapter_title: str) -> str:
        """Extract text belonging to a specific chapter."""

        # Try to find chapter start with exact match first
        pattern = re.escape(chapter_title)
        match = re.search(pattern, full_text, re.IGNORECASE)

        if not match:
            # Try a more flexible match (just the chapter number/name part)
            # Handle "Chapter 1: Intro" -> try "Chapter 1"
            simplified = re.sub(r':.*$', '', chapter_title).strip()
            if simplified != chapter_title:
                match = re.search(re.escape(simplified), full_text, re.IGNORECASE)

        if not match:
            logger.warning(f"Could not find chapter '{chapter_title}' in text, using beginning")
            return full_text[:settings.max_chapter_chars]

        start = match.start()

        # Find next chapter marker (various patterns)
        chapter_patterns = [
            r'\n\s*(Chapter\s+\d+)',
            r'\n\s*(CHAPTER\s+\d+)',
            r'\n\s*(Part\s+\d+)',
            r'\n\s*(PART\s+\d+)',
            r'\n\s*(\d+\.\s+[A-Z][A-Za-z]+)',  # "1. Introduction"
        ]

        # Search for next chapter starting after current chapter title
        search_start = start + len(chapter_title) + 100  # Skip past current chapter title
        end = len(full_text)

        for pattern in chapter_patterns:
            next_match = re.search(pattern, full_text[search_start:])
            if next_match:
                potential_end = search_start + next_match.start()
                if potential_end < end:
                    end = potential_end

        return full_text[start:end]

    def _build_chapter_node(self, chapter_data: dict, fallback_title: str) -> StructureNode:
        """Build a StructureNode from chapter extraction data."""

        def build_node(section_data: dict, parent_level: int = 0) -> StructureNode:
            node_id = str(uuid.uuid4())[:8]

            type_str = section_data.get("type", "section")
            try:
                section_type = SectionType(type_str)
            except ValueError:
                section_type = SectionType.SECTION

            level = section_data.get("level", parent_level + 1)

            category_str = section_data.get("category", "knowledge")
            try:
                category = ContentCategory(category_str)
            except ValueError:
                category = ContentCategory.KNOWLEDGE

            content = section_data.get("content_summary", "")

            children = [
                build_node(child, level)
                for child in section_data.get("children", [])
            ]

            return StructureNode(
                id=node_id,
                title=section_data.get("title", "Untitled"),
                type=section_type,
                level=level,
                content=content,
                children=children,
                category=category,
                included=category == ContentCategory.KNOWLEDGE
            )

        # Handle case where chapter_data might be malformed
        if not chapter_data or not isinstance(chapter_data, dict):
            chapter_data = {
                "title": fallback_title,
                "type": "chapter",
                "level": 1,
                "category": "knowledge",
                "content_summary": "",
                "children": []
            }

        # Ensure we have the title
        if "title" not in chapter_data:
            chapter_data["title"] = fallback_title

        return build_node(chapter_data, 0)

    def _build_structure_from_chapters(
        self,
        toc: dict,
        chapter_nodes: list[StructureNode],
        full_text: str
    ) -> DocumentStructure:
        """Build complete DocumentStructure from TOC and processed chapters."""

        root = StructureNode(
            id="root",
            title=toc.get("title", "Untitled Document"),
            type=SectionType.BOOK,
            level=0,
            children=chapter_nodes,
            category=ContentCategory.KNOWLEDGE,
            included=True
        )

        # Count pages (estimate from text)
        page_markers = re.findall(r'=== PAGE (\d+) ===', full_text)
        total_pages = max([int(p) for p in page_markers]) if page_markers else 1

        return DocumentStructure(
            title=toc.get("title", "Untitled Document"),
            author=toc.get("author"),
            root=root,
            total_pages=total_pages
        )

    async def _extract_structure_single_pass(
        self,
        text: str,
        title_hint: Optional[str] = None,
        author_hint: Optional[str] = None
    ) -> DocumentStructure:
        """Original single-pass extraction for smaller documents."""

        prompt = f"""Analyze this document text and extract its hierarchical structure.

The document appears to be a book or textbook. Identify:
1. The document title (if not provided: {title_hint or 'detect from content'})
2. Author (if not provided: {author_hint or 'detect from content'})
3. All chapters, sections, and subsections with their hierarchy
4. For each section, classify it as either:
   - "knowledge": Contains substantive educational/informational content
   - "meta": Preface, foreword, acknowledgements, table of contents, index, copyright, about author, etc.

Extract the COMPLETE hierarchical structure to any depth. Subsections can contain sub-subsections, and those can contain further nested content.

Return a JSON structure like this:
{{
    "title": "Book Title",
    "author": "Author Name or null",
    "sections": [
        {{
            "title": "Chapter 1: Foundations",
            "type": "chapter",
            "level": 1,
            "category": "knowledge",
            "content_summary": "Brief summary of what this chapter contains",
            "children": [
                {{
                    "title": "1.1 Basic Concepts",
                    "type": "section",
                    "level": 2,
                    "category": "knowledge",
                    "content_summary": "...",
                    "children": [
                        {{
                            "title": "1.1.1 Definitions",
                            "type": "subsection",
                            "level": 3,
                            "category": "knowledge",
                            "content_summary": "...",
                            "children": [
                                {{
                                    "title": "1.1.1.1 Primary Terms",
                                    "type": "content",
                                    "level": 4,
                                    "category": "knowledge",
                                    "content_summary": "...",
                                    "children": []
                                }}
                            ]
                        }}
                    ]
                }}
            ]
        }}
    ]
}}

Valid types: "book", "chapter", "section", "subsection", "content"
Valid categories: "knowledge", "meta"

IMPORTANT: Capture ALL levels of hierarchy present in the document. Do not flatten subsections - preserve the full depth.

DOCUMENT TEXT:
{text[:settings.max_structure_text_chars]}
"""

        response = await self.provider.complete(prompt)

        # Parse JSON from response
        structure_data = self._parse_json_response(response)

        # Convert to DocumentStructure
        return self._build_structure(structure_data, text)

    def _try_repair_json(self, json_str: str) -> dict:
        """Attempt to repair truncated or malformed JSON."""
        # Try to fix common truncation issues
        # Count open brackets and braces
        open_braces = json_str.count('{') - json_str.count('}')
        open_brackets = json_str.count('[') - json_str.count(']')
        
        repaired = json_str.rstrip()
        
        # Remove trailing comma if present
        if repaired.endswith(','):
            repaired = repaired[:-1]
        
        # Close any unclosed strings (look for odd number of unescaped quotes)
        # This is a simplistic check
        in_string = False
        last_char = ''
        for char in repaired:
            if char == '"' and last_char != '\\':
                in_string = not in_string
            last_char = char
        
        if in_string:
            repaired += '"'
        
        # Close brackets and braces
        repaired += ']' * open_brackets
        repaired += '}' * open_braces
        
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            # Try a more aggressive repair - truncate to last complete element
            # Find the last valid closing structure
            for i in range(len(json_str) - 1, 0, -1):
                if json_str[i] in '}]':
                    try:
                        test_str = json_str[:i+1]
                        # Balance the brackets
                        open_braces = test_str.count('{') - test_str.count('}')
                        open_brackets = test_str.count('[') - test_str.count(']')
                        test_str += ']' * open_brackets + '}' * open_braces
                        return json.loads(test_str)
                    except json.JSONDecodeError:
                        continue
            raise

    def _parse_json_response(self, response: str) -> dict:
        """Extract JSON from LLM response."""
        logger.debug(f"Raw LLM response (first 500 chars): {response[:500] if response else 'EMPTY'}")

        if not response:
            logger.error("LLM returned empty response")
            return {
                "title": "Untitled Document",
                "author": None,
                "sections": [],
                "chapters": []
            }

        json_content = None
        
        # First, try to extract JSON from markdown code blocks (common with Gemini)
        # Handle ```json ... ``` or ``` ... ``` blocks
        code_block_match = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', response)
        if code_block_match:
            json_content = code_block_match.group(1).strip()
        else:
            # Try to find JSON in the response (greedy match for outermost braces)
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                json_content = json_match.group()

        if json_content:
            # First try direct parsing
            try:
                parsed = json.loads(json_content)
                logger.debug(f"Successfully parsed JSON with keys: {list(parsed.keys())}")
                return parsed
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error: {e}, attempting repair")
                # Try to repair the JSON
                try:
                    parsed = self._try_repair_json(json_content)
                    logger.info(f"Successfully repaired and parsed JSON with keys: {list(parsed.keys())}")
                    return parsed
                except json.JSONDecodeError as e2:
                    logger.error(f"JSON repair also failed: {e2}")
                    logger.error(f"Failed JSON content (first 500 chars): {json_content[:500]}")

        # Fallback: return minimal structure
        logger.warning(f"Could not find valid JSON in response, using fallback. Response preview: {response[:200]}")
        return {
            "title": "Untitled Document",
            "author": None,
            "sections": []
        }

    def _build_structure(self, data: dict, full_text: str) -> DocumentStructure:
        """Build DocumentStructure from parsed data."""

        def build_node(section_data: dict, parent_level: int = 0) -> StructureNode:
            node_id = str(uuid.uuid4())[:8]
            section_type = SectionType(section_data.get("type", "section"))
            level = section_data.get("level", parent_level + 1)
            category = ContentCategory(section_data.get("category", "knowledge"))

            # Extract content for this section from full text if possible
            content = section_data.get("content_summary", "")

            children = [
                build_node(child, level)
                for child in section_data.get("children", [])
            ]

            return StructureNode(
                id=node_id,
                title=section_data.get("title", "Untitled"),
                type=section_type,
                level=level,
                content=content,
                children=children,
                category=category,
                included=category == ContentCategory.KNOWLEDGE
            )

        # Build root node
        root_children = [
            build_node(section)
            for section in data.get("sections", [])
        ]

        root = StructureNode(
            id="root",
            title=data.get("title", "Untitled Document"),
            type=SectionType.BOOK,
            level=0,
            children=root_children,
            category=ContentCategory.KNOWLEDGE,
            included=True
        )

        # Count pages (estimate from text)
        page_markers = re.findall(r'=== PAGE (\d+) ===', full_text)
        total_pages = max([int(p) for p in page_markers]) if page_markers else 1

        return DocumentStructure(
            title=data.get("title", "Untitled Document"),
            author=data.get("author"),
            root=root,
            total_pages=total_pages
        )

    def populate_source_text_from_full_text(
        self,
        root: StructureNode,
        full_text: str
    ) -> None:
        """Populate source_text on all nodes that don't have it, using raw text extraction.

        This finds each node's text in full_text by matching section title boundaries.
        Should be called after structure extraction and before atomization.
        """
        self._populate_node_source_text(root, full_text, root.children)

    def _populate_node_source_text(
        self,
        node: StructureNode,
        full_text: str,
        siblings: list
    ) -> None:
        """Recursively populate source_text for a node and its descendants."""
        if not node.included:
            return

        # Skip if source_text is already populated with substantial content
        if node.source_text and len(node.source_text) > 200:
            # Still process children
            for child in node.children:
                self._populate_node_source_text(child, full_text, node.children)
            return

        # Skip the root book node
        if node.type == SectionType.BOOK:
            for child in node.children:
                self._populate_node_source_text(child, full_text, node.children)
            return

        # Extract raw text for this node from full_text
        raw_text = self._extract_section_raw_text(full_text, node.title, siblings, node)

        if raw_text and len(raw_text) > len(node.source_text or ''):
            node.source_text = raw_text
            logger.debug(f"Populated source_text for '{node.title}': {len(raw_text)} chars")

        # Process children
        for child in node.children:
            self._populate_node_source_text(child, full_text, node.children)

    def _extract_section_raw_text(
        self,
        full_text: str,
        title: str,
        siblings: list,
        current_node: StructureNode
    ) -> str:
        """Extract raw text for a section by finding its boundaries in full_text."""
        # Find section start
        pattern = re.escape(title)
        match = re.search(pattern, full_text, re.IGNORECASE)

        if not match:
            # Try simplified title (remove leading numbers like "1.1 " or "Chapter 1: ")
            simplified = re.sub(r'^(\d+\.?\d*\.?\s*|Chapter\s+\d+[.:]\s*)', '', title).strip()
            if simplified and simplified != title:
                match = re.search(re.escape(simplified), full_text, re.IGNORECASE)

        if not match:
            return ""

        start = match.start()

        # Find end: look for the next sibling's title
        end = len(full_text)
        found_current = False
        for sibling in siblings:
            if found_current:
                next_pattern = re.escape(sibling.title)
                next_match = re.search(next_pattern, full_text[start + len(title):], re.IGNORECASE)
                if next_match:
                    end = start + len(title) + next_match.start()
                break
            if sibling.id == current_node.id:
                found_current = True

        # Also check for next chapter/section markers as boundaries
        section_start = start + len(title) + 50  # skip past current title
        if section_start < len(full_text):
            chapter_patterns = [
                r'\n\s*(Chapter\s+\d+)',
                r'\n\s*(CHAPTER\s+\d+)',
            ]
            # Only use chapter patterns for non-chapter nodes
            if current_node.type != SectionType.CHAPTER:
                for cp in chapter_patterns:
                    next_match = re.search(cp, full_text[section_start:])
                    if next_match:
                        potential_end = section_start + next_match.start()
                        if potential_end < end:
                            end = potential_end

        # Cap at a reasonable size
        max_section_size = 50000
        if end - start > max_section_size:
            end = start + max_section_size

        return full_text[start:end].strip()

    async def extract_sub_structure(
        self,
        content: str,
        parent_title: str,
        parent_level: int
    ) -> List[StructureNode]:
        """Extract hierarchical sub-structure from content.

        Args:
            content: The content to analyze
            parent_title: Title of the parent section
            parent_level: Level of the parent section

        Returns:
            List of child StructureNode objects
        """
        prompt = f"""Analyze this mathematical content and identify logical sub-sections.

Parent section: "{parent_title}"

Content to analyze:
---
{content[:15000]}
---

Identify the logical divisions in this content. Each division should contain a coherent unit
(such as a theorem, definition, lemma, proof, example, or remark).

Return JSON in this format:
{{
    "sections": [
        {{
            "title": "Definition 1.1: Continuous Function",
            "type": "content",
            "content_summary": "Defines what it means for a function to be continuous",
            "start_char": 0,
            "end_char": 500
        }},
        {{
            "title": "Theorem 1.2: Intermediate Value Theorem",
            "type": "content",
            "content_summary": "States and proves the IVT",
            "start_char": 501,
            "end_char": 1200
        }}
    ]
}}

If the content cannot be meaningfully divided, return:
{{"sections": []}}
"""

        try:
            response = await self.provider.complete(prompt)
            result = self._parse_json_response(response)
            sections = result.get("sections", [])

            if not sections:
                return []

            children = []
            for section in sections:
                start = section.get("start_char", 0)
                end = section.get("end_char", len(content))
                section_content = content[start:end] if start < end else content

                child = StructureNode(
                    id=str(uuid.uuid4())[:8],
                    title=section.get("title", f"{parent_title} - Part"),
                    type=SectionType.CONTENT,
                    level=parent_level + 1,
                    content=section.get("content_summary", ""),
                    source_text=section_content,
                    category=ContentCategory.KNOWLEDGE,
                    included=True
                )
                children.append(child)

            return children

        except Exception as e:
            logger.error(f"Failed to extract sub-structure: {e}")
            return []

    async def extract_content_for_section(
        self,
        full_text: str,
        section_title: str,
        next_section_title: Optional[str] = None
    ) -> str:
        """Extract the content for a specific section from the full text."""

        prompt = f"""From the following document text, extract ONLY the content that belongs to the section titled "{section_title}".

If there's a next section titled "{next_section_title or 'END OF DOCUMENT'}", stop before that section.

Return only the extracted content, no additional commentary.

DOCUMENT TEXT:
{full_text[:settings.max_content_text_chars]}
"""

        content = await self.provider.complete(prompt)
        return content.strip()

    async def refine_structure(
        self,
        structure: DocumentStructure,
        full_text: str
    ) -> DocumentStructure:
        """Refine structure by extracting actual content for each section."""

        async def fill_content(node: StructureNode, siblings: list) -> StructureNode:
            # Skip non-included nodes to save API calls
            if not node.included:
                # Still process children in case any are included
                for i, child in enumerate(node.children):
                    node.children[i] = await fill_content(child, node.children)
                return node

            # Find next sibling for boundary
            next_title = None
            found_current = False
            for sibling in siblings:
                if found_current:
                    next_title = sibling.title
                    break
                if sibling.id == node.id:
                    found_current = True

            # Extract content if this is a leaf node or has minimal content
            if not node.children or len(node.content) < 100:
                content = await self.extract_content_for_section(
                    full_text,
                    node.title,
                    next_title
                )
                node.content = content
                # Also preserve as source_text for atomization/splitting
                if not node.source_text:
                    node.source_text = content

            # Recursively process children
            for i, child in enumerate(node.children):
                node.children[i] = await fill_content(child, node.children)

            return node

        # Process root's children
        for i, child in enumerate(structure.root.children):
            structure.root.children[i] = await fill_content(
                child,
                structure.root.children
            )

        return structure
