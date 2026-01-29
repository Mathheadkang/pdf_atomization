"""Content filtering service to identify knowledge vs meta sections."""

from typing import List, Set
import re

from app.models import StructureNode, ContentCategory, DocumentStructure
from app.providers import get_provider


# Keywords that typically indicate meta/non-knowledge content
META_KEYWORDS = {
    "preface", "foreword", "acknowledgement", "acknowledgment",
    "table of contents", "contents", "toc",
    "index", "glossary", "bibliography", "references",
    "copyright", "about the author", "about author",
    "dedication", "epigraph", "frontmatter", "back matter",
    "appendix",  # Often meta, but can contain useful content
    "colophon", "endnotes", "footnotes",
    "list of figures", "list of tables", "list of illustrations",
    "permissions", "credits", "contributor", "contributors"
}

# Keywords that strongly indicate knowledge content
KNOWLEDGE_KEYWORDS = {
    "chapter", "part", "unit", "module", "lesson",
    "introduction", "conclusion", "summary",
    "theory", "method", "methodology", "analysis",
    "results", "discussion", "findings",
    "case study", "example", "exercise", "problem"
}


class ContentFilter:
    """Service for filtering non-knowledge sections from documents."""

    def __init__(self):
        self.provider = get_provider()

    def classify_by_title(self, title: str) -> ContentCategory:
        """Quick classification based on section title."""
        title_lower = title.lower()

        # Check for meta keywords
        for keyword in META_KEYWORDS:
            if keyword in title_lower:
                return ContentCategory.META

        # Check for knowledge keywords
        for keyword in KNOWLEDGE_KEYWORDS:
            if keyword in title_lower:
                return ContentCategory.KNOWLEDGE

        # Default to knowledge (conservative approach)
        return ContentCategory.KNOWLEDGE

    async def classify_with_llm(
        self,
        title: str,
        content_preview: str
    ) -> ContentCategory:
        """Use LLM to classify ambiguous sections."""

        prompt = f"""Classify this document section as either "knowledge" or "meta".

- "knowledge": Contains substantive educational, informational, or instructional content that readers would want to study or reference
- "meta": Administrative content like preface, acknowledgements, table of contents, index, copyright notices, author bios, etc.

Section Title: {title}
Content Preview: {content_preview[:500]}

Respond with just one word: "knowledge" or "meta"
"""

        response = await self.provider.complete(prompt)
        response_lower = response.strip().lower()

        if "meta" in response_lower:
            return ContentCategory.META
        return ContentCategory.KNOWLEDGE

    def filter_structure(
        self,
        structure: DocumentStructure,
        include_appendices: bool = False
    ) -> DocumentStructure:
        """Apply filtering to document structure based on title keywords."""

        def filter_node(node: StructureNode) -> StructureNode:
            # Classify this node
            category = self.classify_by_title(node.title)

            # Special handling for appendices
            if "appendix" in node.title.lower() and include_appendices:
                category = ContentCategory.KNOWLEDGE

            node.category = category
            node.included = category == ContentCategory.KNOWLEDGE

            # Recursively filter children
            for child in node.children:
                filter_node(child)

            return node

        # Filter root's children
        for child in structure.root.children:
            filter_node(child)

        return structure

    async def filter_structure_with_llm(
        self,
        structure: DocumentStructure
    ) -> DocumentStructure:
        """Use LLM for more accurate filtering of ambiguous sections."""

        async def classify_node(node: StructureNode) -> StructureNode:
            # First try title-based classification
            category = self.classify_by_title(node.title)

            # If uncertain (defaulted to knowledge), use LLM
            title_lower = node.title.lower()
            is_ambiguous = not any(
                kw in title_lower
                for kw in META_KEYWORDS | KNOWLEDGE_KEYWORDS
            )

            if is_ambiguous and node.content:
                category = await self.classify_with_llm(
                    node.title,
                    node.content
                )

            node.category = category
            node.included = category == ContentCategory.KNOWLEDGE

            # Recursively process children
            for child in node.children:
                await classify_node(child)

            return node

        for child in structure.root.children:
            await classify_node(child)

        return structure

    def get_filtered_sections(
        self,
        structure: DocumentStructure
    ) -> List[StructureNode]:
        """Get list of sections that were filtered out."""
        filtered = []

        def collect_filtered(node: StructureNode):
            if not node.included:
                filtered.append(node)
            for child in node.children:
                collect_filtered(child)

        for child in structure.root.children:
            collect_filtered(child)

        return filtered

    def get_included_sections(
        self,
        structure: DocumentStructure
    ) -> List[StructureNode]:
        """Get list of sections that will be included."""
        included = []

        def collect_included(node: StructureNode):
            if node.included:
                included.append(node)
            for child in node.children:
                collect_included(child)

        for child in structure.root.children:
            collect_included(child)

        return included

    def update_inclusion(
        self,
        structure: DocumentStructure,
        section_id: str,
        included: bool
    ) -> DocumentStructure:
        """Update inclusion status for a specific section."""

        def update_node(node: StructureNode) -> bool:
            if node.id == section_id:
                node.included = included
                return True
            for child in node.children:
                if update_node(child):
                    return True
            return False

        update_node(structure.root)
        return structure
