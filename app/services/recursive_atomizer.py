"""Recursive atomization service for splitting content into atomic units."""

import json
import logging
import re
import traceback
import uuid
from typing import Optional, Callable, Tuple, List, Awaitable

from app.config import settings
from app.models import (
    DocumentStructure,
    StructureNode,
    SectionType,
    AtomizationStatus,
    AtomType,
)
from app.providers import get_provider_for_task

logger = logging.getLogger(__name__)


ATOMICITY_CHECK_PROMPT = """Analyze this mathematical content and determine:
1. Does it contain exactly ONE atomic concept (theorem/definition/lemma/corollary/proposition/example/remark)?
2. If it contains multiple concepts or is a container section (like a chapter overview), it is NOT atomic.
3. If yes, what type is it?

Content to analyze:
---
{content}
---

Respond ONLY with valid JSON (no markdown code blocks):
{{"is_atomic": true/false, "atom_type": "theorem|definition|lemma|corollary|proposition|example|remark|other|null", "reason": "brief explanation"}}
"""


SPLIT_NODE_PROMPT = """You are analyzing a mathematical document section that needs to be split into smaller atomic units.

IMPORTANT: The user has explicitly requested to split this content, so please try to find logical divisions.

Look for ANY of these as potential split points:
- Theorems, Propositions, Lemmas, Corollaries (often start with "Theorem", "Proposition", etc.)
- Definitions (often start with "Definition" or "Let X be...")
- Examples (often start with "Example" or "Consider...")
- Remarks, Notes, or Observations
- Proofs (can be separated from the theorem statement)
- Different numbered items (1.1, 1.2, etc.)
- Paragraphs discussing distinctly different concepts

Current section title: {title}

Content to split:
---
{content}
---

Find logical divisions in this content. For each division, provide:
- A descriptive title
- The character positions [start, end] where this unit appears

Respond ONLY with valid JSON (no markdown code blocks):
{{
    "splits": [
        {{"title": "Definition 1.1: Continuous Function", "start": 0, "end": 500}},
        {{"title": "Theorem 1.2: Intermediate Value Theorem", "start": 501, "end": 1200}}
    ]
}}

If there is truly only ONE concept with no logical divisions possible, return:
{{"splits": []}}

But please try hard to find at least 2 splits if possible - the user wants this content divided.
"""


def _parse_json_from_response(response: str) -> dict:
    """Parse JSON from an AI response, handling markdown code blocks."""
    text = response.strip()
    # Strip markdown code block wrapper if present
    code_block_match = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', text)
    if code_block_match:
        text = code_block_match.group(1).strip()
    else:
        # Try finding raw JSON object
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            text = json_match.group()
    return json.loads(text)


class RecursiveAtomizer:
    """Service for recursively splitting document structure into atomic units."""

    def __init__(self):
        self.provider = get_provider_for_task("content_summarizer")
        self.max_depth = settings.max_recursion_depth
        self.min_content_length = settings.min_content_length_for_split

    async def atomize(
        self,
        structure: DocumentStructure,
        full_text: str,
        progress_callback: Optional[Callable[[str, float], Awaitable[None]]] = None
    ) -> DocumentStructure:
        """Main entry point for atomization.

        Args:
            structure: The document structure to atomize
            full_text: The full document text for context
            progress_callback: Optional callback for progress updates

        Returns:
            DocumentStructure with atomized nodes
        """
        total_nodes = self._count_included_nodes(structure.root)
        processed = [0]  # Use list to allow modification in nested function

        async def update_progress(node_title: str):
            processed[0] += 1
            if progress_callback:
                pct = processed[0] / max(total_nodes, 1)
                await progress_callback(f"Atomizing: {node_title[:50]}...", pct)

        # Recursively atomize starting from root
        await self._atomize_node(structure.root, full_text, depth=0, progress_callback=update_progress)

        return structure

    def _count_included_nodes(self, node: StructureNode) -> int:
        """Count total number of included nodes."""
        if not node.included:
            return 0
        count = 1
        for child in node.children:
            count += self._count_included_nodes(child)
        return count

    async def _atomize_node(
        self,
        node: StructureNode,
        full_text: str,
        depth: int,
        progress_callback: Optional[Callable[[str], Awaitable[None]]] = None
    ) -> None:
        """Recursively process a node and its children.

        Args:
            node: The node to process
            full_text: Full document text
            depth: Current recursion depth
            progress_callback: Optional progress callback
        """
        if not node.included:
            return

        if progress_callback:
            await progress_callback(node.title)

        # Don't atomize container types (book, chapter)
        if node.type in [SectionType.BOOK, SectionType.CHAPTER]:
            # Process children only
            for child in node.children:
                await self._atomize_node(child, full_text, depth, progress_callback)
            return

        # Check if we've exceeded max depth
        if depth >= self.max_depth:
            node.atomization_status = AtomizationStatus.ATOMIC
            node.atom_type = AtomType.OTHER
            return

        # Get content to analyze
        content = node.source_text or node.content
        if not content or len(content) < self.min_content_length:
            # Too short to split, mark as atomic
            node.atomization_status = AtomizationStatus.ATOMIC
            is_atomic, atom_type = await self._check_atomicity(node, content or "")
            node.atom_type = atom_type
            return

        # Check if this node is atomic
        is_atomic, atom_type = await self._check_atomicity(node, content)

        if is_atomic:
            node.atomization_status = AtomizationStatus.ATOMIC
            node.atom_type = atom_type
            # Process any existing children
            for child in node.children:
                await self._atomize_node(child, full_text, depth + 1, progress_callback)
        else:
            # Need to split this node
            node.atomization_status = AtomizationStatus.NEEDS_SPLITTING

            # If node already has children, just process them
            if node.children:
                for child in node.children:
                    await self._atomize_node(child, full_text, depth + 1, progress_callback)
            else:
                # Split the node into new children
                new_children = await self._split_node(node, content)
                if new_children:
                    node.children = new_children
                    # Recursively process new children
                    for child in node.children:
                        await self._atomize_node(child, full_text, depth + 1, progress_callback)
                else:
                    # Could not split, mark as atomic anyway
                    node.atomization_status = AtomizationStatus.ATOMIC
                    node.atom_type = atom_type or AtomType.OTHER

    async def check_single_node_atomicity(self, node: StructureNode) -> dict:
        """Check atomicity for a single node, returning full decision.

        Args:
            node: The node to check

        Returns:
            Dict with keys: is_atomic, atom_type, reason
        """
        content = node.source_text or node.content or ""
        return await self._check_atomicity_with_reason(content)

    async def _check_atomicity_with_reason(self, content: str) -> dict:
        """Check atomicity and return full decision including reason.

        Args:
            content: The content to analyze

        Returns:
            Dict with keys: is_atomic, atom_type, reason
        """
        # Build prompt with content
        prompt = ATOMICITY_CHECK_PROMPT.format(content=content[:10000])  # Limit content length

        try:
            response = await self.provider.complete(
                prompt=prompt,
                system_prompt="You are a mathematical document analyzer. Respond only with valid JSON.",
                temperature=0.1,
                max_tokens=16000
            )

            # Parse JSON response (handle markdown code blocks)
            result = _parse_json_from_response(response)
            is_atomic = result.get("is_atomic", False)
            atom_type_str = result.get("atom_type")
            reason = result.get("reason", "No reason provided.")

            atom_type = None
            if atom_type_str and atom_type_str != "null":
                try:
                    atom_type = AtomType(atom_type_str.lower())
                except ValueError:
                    atom_type = AtomType.OTHER

            return {
                "is_atomic": is_atomic,
                "atom_type": atom_type,
                "reason": reason
            }

        except Exception as e:
            # On error, assume not atomic if content is long enough
            logger.error(f"_check_atomicity: Failed: {type(e).__name__}: {e}")
            return {
                "is_atomic": len(content) < self.min_content_length,
                "atom_type": AtomType.OTHER,
                "reason": f"Analysis failed: {str(e)}"
            }

    async def _check_atomicity(
        self,
        node: StructureNode,
        content: str
    ) -> Tuple[bool, Optional[AtomType]]:
        """Use AI to determine if a node is atomic.

        Args:
            node: The node to check
            content: The content to analyze

        Returns:
            Tuple of (is_atomic, atom_type)
        """
        result = await self._check_atomicity_with_reason(content)
        return result["is_atomic"], result["atom_type"]

    async def split_single_node(self, node: StructureNode) -> List[StructureNode]:
        """Split a single node into children using structure extraction.

        Args:
            node: The node to split

        Returns:
            List of new child StructureNode objects
        """

        content = node.source_text or node.content or ""
        logger.info(f"split_single_node called for '{node.title}': source_text={len(node.source_text or '')} chars, content={len(node.content or '')} chars, using {len(content)} chars")

        if not content:
            logger.warning(f"No content available for node '{node.title}'")
            return []

        if len(content) < 100:
            logger.warning(f"Content too short for node '{node.title}': {content[:100]}")

        return await self._split_node(node, content)

    async def _split_node(
        self,
        node: StructureNode,
        content: str
    ) -> List[StructureNode]:
        """Use AI to split a non-atomic node into atomic parts.

        Args:
            node: The node to split
            content: The content to split

        Returns:
            List of new child StructureNode objects
        """
        logger.info(f"_split_node: Attempting to split '{node.title}' with {len(content)} chars of content")

        prompt = SPLIT_NODE_PROMPT.format(
            title=node.title,
            content=content[:15000]  # Limit content length
        )

        try:
            response = await self.provider.complete(
                prompt=prompt,
                system_prompt="You are a mathematical document analyzer. Respond only with valid JSON.",
                temperature=0.1,
                max_tokens=65536
            )

            logger.info(f"_split_node: AI response for '{node.title}': {response[:200]}...")
            logger.debug(f"_split_node: Full AI response ({len(response)} chars): {response}")

            # Parse JSON response (handle markdown code blocks)
            try:
                result = _parse_json_from_response(response)
            except json.JSONDecodeError as je:
                logger.error(f"_split_node: JSON parse failed for '{node.title}': {je}")
                logger.error(f"_split_node: Raw response was: {response!r}")
                return []

            splits = result.get("splits", [])
            logger.info(f"_split_node: Found {len(splits)} splits for '{node.title}'")

            if not splits:
                return []

            # Create new child nodes
            new_children = []
            for i, split in enumerate(splits):
                title = split.get("title", f"{node.title} Part {i+1}")
                start = split.get("start", 0)
                end = split.get("end", len(content))

                # Validate positions
                start = max(0, min(start, len(content)))
                end = max(start, min(end, len(content)))

                # Extract content for this split
                split_content = content[start:end] if start < end else content

                child = StructureNode(
                    id=str(uuid.uuid4())[:8],
                    title=title,
                    type=SectionType.CONTENT,
                    level=node.level + 1,
                    content="",  # Will be filled by content summarizer
                    source_text=split_content,
                    page_start=node.page_start,
                    page_end=node.page_end,
                    atomization_status=AtomizationStatus.PENDING,
                    included=True
                )
                new_children.append(child)
                logger.info(f"_split_node: Created child '{title}' [{start}:{end}] = {len(split_content)} chars")

            return new_children

        except Exception as e:
            logger.error(f"_split_node: Failed to split '{node.title}': {type(e).__name__}: {e}")
            logger.error(f"_split_node: Traceback: {traceback.format_exc()}")
            return []
