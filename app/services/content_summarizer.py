"""Content summarization service for filling atomic nodes with structured content."""

import json
import logging
from typing import Optional, Callable

from app.models import (
    DocumentStructure,
    StructureNode,
    SectionType,
    AtomizationStatus,
    AtomType,
    AtomContent,
)
from app.providers import get_provider_for_task

logger = logging.getLogger(__name__)


CONTENT_SUMMARIZATION_PROMPT = """You are a mathematician responsible for summarizing mathematical content into a structured format.

Atom Type: {atom_type}
Title: {title}

Content to summarize:
---
{content}
---

IMPORTANT:
- Preserve ALL LaTeX notation exactly as written (e.g., $x^2$, \\frac{{a}}{{b}}, \\int, etc.)
- Description and Statement are REQUIRED
- Proof is OPTIONAL (only include if a proof is present in the content)
- Lemmas are OPTIONAL (only include if supporting lemmas are mentioned)
- Related Content is OPTIONAL (only include if related concepts are discussed)

Return ONLY valid JSON (no markdown code blocks):
{{
    "description": "A 1-2 sentence AI-generated summary explaining what this {atom_type} represents and why it matters",
    "statement": "The exact mathematical statement with all LaTeX preserved",
    "proof": "The complete proof if present, null otherwise",
    "lemmas": ["Supporting lemma 1 with LaTeX", "Supporting lemma 2"] or [],
    "related_content": "Brief summary of related concepts mentioned, or null"
}}
"""


class ContentSummarizer:
    """Service for filling atomic nodes with AI-summarized structured content."""

    def __init__(self):
        self.provider = get_provider_for_task("content_summarizer")
        # Log which model is being used
        model_info = self.provider.get_model_info()
        logger.info(f"ContentSummarizer initialized with provider: {model_info}")

    async def fill_content(
        self,
        structure: DocumentStructure,
        full_text: str,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> DocumentStructure:
        """Fill all atomic nodes with structured content.

        Args:
            structure: The document structure with atomic nodes
            full_text: The full document text for context
            progress_callback: Optional callback for progress updates

        Returns:
            DocumentStructure with filled atom_content fields
        """
        # Count atomic nodes that need filling
        atomic_nodes = self._collect_atomic_nodes(structure.root)
        total = len(atomic_nodes)
        processed = [0]

        async def update_progress(node_title: str):
            processed[0] += 1
            if progress_callback:
                pct = processed[0] / max(total, 1)
                await progress_callback(f"Summarizing: {node_title[:50]}...", pct)

        # Fill each atomic node
        for node in atomic_nodes:
            await self._fill_node_content(node)
            await update_progress(node.title)

        return structure

    async def summarize_single_node(self, node: StructureNode) -> AtomContent:
        """Generate content summary for a single node.

        Args:
            node: The atomic node to summarize

        Returns:
            AtomContent with structured summary
        """
        content = node.source_text or node.content
        if not content:
            return AtomContent(
                description=f"A {node.atom_type.value if node.atom_type else 'mathematical'} concept.",
                statement=node.title
            )

        atom_type = node.atom_type.value if node.atom_type else "mathematical concept"

        prompt = CONTENT_SUMMARIZATION_PROMPT.format(
            atom_type=atom_type,
            title=node.title,
            content=content[:12000]
        )

        try:
            response = await self.provider.complete(
                prompt=prompt,
                system_prompt="You are a mathematical content summarizer. Preserve all LaTeX notation. Respond only with valid JSON.",
                temperature=0.2,
                max_tokens=4000
            )

            result = json.loads(response.strip())

            return AtomContent(
                description=result.get("description", f"A {atom_type}."),
                statement=result.get("statement", node.title),
                proof=result.get("proof"),
                lemmas=result.get("lemmas", []),
                related_content=result.get("related_content")
            )

        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Content summarization failed for {node.title}: {e}")
            return AtomContent(
                description=f"A {atom_type}.",
                statement=content[:500] if content else node.title
            )

    def _collect_atomic_nodes(self, node: StructureNode) -> list[StructureNode]:
        """Collect all atomic nodes that need content filling."""
        nodes = []

        if not node.included:
            return nodes

        # Check if this node is atomic and needs filling
        if (node.atomization_status == AtomizationStatus.ATOMIC and
            node.atom_content is None and
            node.type not in [SectionType.BOOK, SectionType.CHAPTER]):
            nodes.append(node)

        # Recursively collect from children
        for child in node.children:
            nodes.extend(self._collect_atomic_nodes(child))

        return nodes

    async def _fill_node_content(self, node: StructureNode) -> None:
        """Fill a single atomic node with AI-summarized content.

        Args:
            node: The atomic node to fill
        """
        # Get content to summarize
        content = node.source_text or node.content
        if not content:
            # No content to summarize, create minimal atom content
            node.atom_content = AtomContent(
                description=f"A {node.atom_type.value if node.atom_type else 'mathematical'} concept.",
                statement=node.title
            )
            node.atomization_status = AtomizationStatus.FILLED
            return

        # Get atom type string
        atom_type = node.atom_type.value if node.atom_type else "mathematical concept"

        prompt = CONTENT_SUMMARIZATION_PROMPT.format(
            atom_type=atom_type,
            title=node.title,
            content=content[:12000]  # Limit content length
        )

        try:
            response = await self.provider.complete(
                prompt=prompt,
                system_prompt="You are a mathematical content summarizer. Preserve all LaTeX notation. Respond only with valid JSON.",
                temperature=0.2,
                max_tokens=4000
            )

            # Parse JSON response
            result = json.loads(response.strip())

            # Create AtomContent from response
            node.atom_content = AtomContent(
                description=result.get("description", f"A {atom_type}."),
                statement=result.get("statement", node.title),
                proof=result.get("proof"),
                lemmas=result.get("lemmas", []),
                related_content=result.get("related_content")
            )

            node.atomization_status = AtomizationStatus.FILLED

        except (json.JSONDecodeError, Exception) as e:
            # On error, create minimal content
            node.atom_content = AtomContent(
                description=f"A {atom_type}.",
                statement=content[:500] if content else node.title
            )
            node.atomization_status = AtomizationStatus.FILLED
