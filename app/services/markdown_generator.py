"""Markdown file generator service."""

import re
from pathlib import Path
from typing import List, Dict, Optional
import aiofiles

from app.models import DocumentStructure, StructureNode, SectionType, AtomizationStatus
from app.services.link_manager import LinkManager


class MarkdownGenerator:
    """Service for generating markdown files from document structure."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.link_manager = LinkManager()
        self.generated_files: Dict[str, Path] = {}  # section_id -> file_path
        self._current_file_path: Optional[Path] = None  # Track current file for link generation

    def _sanitize_filename(self, name: str) -> str:
        """Convert a title to a valid filename."""
        # Remove or replace invalid characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '', name)
        # Replace spaces with underscores
        sanitized = re.sub(r'\s+', '_', sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        # Limit length
        if len(sanitized) > 50:
            sanitized = sanitized[:50]
        return sanitized or "untitled"

    def _get_file_path(
        self,
        node: StructureNode,
        parent_path: Path
    ) -> Path:
        """Determine the file path for a node."""
        name = self._sanitize_filename(node.title)

        # Create folder if node has children OR is book/chapter
        if node.children or node.type in [SectionType.BOOK, SectionType.CHAPTER]:
            folder = parent_path / name
            return folder / "index.md"
        else:
            # Create a single file for leaf nodes
            return parent_path / f"{name}.md"

    def _generate_header(
        self,
        node: StructureNode,
        parent: Optional[StructureNode] = None,
        file_path: Optional[Path] = None
    ) -> str:
        """Generate the markdown header with navigation links."""
        lines = [f"# {node.title}", ""]

        # Add parent link using standard markdown format
        if parent:
            parent_link = self.link_manager.create_link(parent.title, file_path)
            lines.append(f"> Parent: {parent_link}")

        # Add children links
        if node.children:
            included_children = [c for c in node.children if c.included]
            if included_children:
                child_links = [
                    self.link_manager.create_link(c.title, file_path)
                    for c in included_children
                ]
                lines.append(f"> Children: {', '.join(child_links)}")

        lines.append("")
        return "\n".join(lines)

    def _generate_content(self, node: StructureNode) -> str:
        """Generate the main content section."""
        # Check if node has atom_content (filled atomic node)
        if node.atomization_status == AtomizationStatus.FILLED and node.atom_content:
            return self._generate_atom_content(node)

        # Fallback to regular content
        if not node.content:
            return ""
        return node.content + "\n"

    def _generate_atom_content(self, node: StructureNode) -> str:
        """Generate structured content for an atomic node.

        Format:
        ## Description
        {ai_summarized_description}

        ## {Theorem|Definition|Lemma|...}
        {statement}

        ## Proof (optional)
        {proof}

        ## Supporting Lemmas (optional)
        - {lemma 1}
        - {lemma 2}

        ## Related Content (optional)
        {related_content}
        """
        if not node.atom_content:
            return ""

        atom = node.atom_content
        atom_type = node.atom_type.value.title() if node.atom_type else "Statement"

        lines = []

        # Description (required)
        lines.append("## Description")
        lines.append("")
        lines.append(atom.description)
        lines.append("")

        # Statement (required)
        lines.append(f"## {atom_type}")
        lines.append("")
        lines.append(atom.statement)
        lines.append("")

        # Proof (optional)
        if atom.proof:
            lines.append("## Proof")
            lines.append("")
            lines.append(atom.proof)
            lines.append("")

        # Supporting Lemmas (optional)
        if atom.lemmas:
            lines.append("## Supporting Lemmas")
            lines.append("")
            for lemma in atom.lemmas:
                lines.append(f"- {lemma}")
            lines.append("")

        # Related Content (optional)
        if atom.related_content:
            lines.append("## Related Content")
            lines.append("")
            lines.append(atom.related_content)
            lines.append("")

        return "\n".join(lines)

    def _generate_footer(
        self,
        node: StructureNode,
        related_sections: List[StructureNode] = None,
        file_path: Optional[Path] = None
    ) -> str:
        """Generate the footer with related links."""
        lines = ["", "---", "## Related"]

        if related_sections:
            for section in related_sections:
                link = self.link_manager.create_link(section.title, file_path)
                lines.append(f"- {link}")
        else:
            lines.append("- *No related sections identified*")

        return "\n".join(lines)

    def generate_markdown(
        self,
        node: StructureNode,
        parent: Optional[StructureNode] = None,
        related_sections: List[StructureNode] = None,
        file_path: Optional[Path] = None
    ) -> str:
        """Generate complete markdown content for a node."""
        parts = [
            self._generate_header(node, parent, file_path),
            self._generate_content(node),
            self._generate_footer(node, related_sections, file_path)
        ]
        return "\n".join(parts)

    async def generate_files(
        self,
        structure: DocumentStructure,
        include_filtered: bool = False
    ) -> Dict[str, Path]:
        """Generate all markdown files for a document structure.

        Uses a two-pass approach:
        1. First pass: Register all file paths with link_manager
        2. Second pass: Generate content with proper link resolution
        """

        # Create base directory
        base_name = self._sanitize_filename(structure.title)
        base_dir = self.output_dir / base_name
        base_dir.mkdir(parents=True, exist_ok=True)

        # Track all generated files
        self.generated_files = {}
        self.link_manager.clear()

        # Pass 1: Register all files first (for link resolution)
        index_path = base_dir / "index.md"
        self.link_manager.register_file(structure.root.title, index_path)
        self.generated_files["root"] = index_path
        await self._register_all_files(structure.root, base_dir, include_filtered)

        # Pass 2: Generate index file for the book
        await self._generate_index(structure, base_dir)

        # Pass 2 continued: Generate files for each section
        await self._generate_node_files(
            structure.root,
            base_dir,
            parent=None,
            include_filtered=include_filtered
        )

        return self.generated_files

    async def _register_all_files(
        self,
        node: StructureNode,
        current_dir: Path,
        include_filtered: bool = False
    ) -> None:
        """First pass: Register all file paths with link manager."""
        if not node.included and not include_filtered:
            return

        if node.type == SectionType.BOOK:
            for child in node.children:
                await self._register_all_files(child, current_dir, include_filtered)
            return

        file_path = self._get_file_path(node, current_dir)
        self.link_manager.register_file(node.title, file_path)
        self.generated_files[node.id] = file_path

        child_dir = file_path.parent if (node.children or node.type in [SectionType.CHAPTER]) else current_dir
        for child in node.children:
            await self._register_all_files(child, child_dir, include_filtered)

    async def _generate_index(
        self,
        structure: DocumentStructure,
        base_dir: Path
    ) -> None:
        """Generate the main index file for the document."""
        index_path = base_dir / "index.md"

        lines = [
            f"# {structure.title}",
            ""
        ]

        if structure.author:
            lines.append(f"**Author:** {structure.author}")
            lines.append("")

        lines.extend([
            f"**Pages:** {structure.total_pages}",
            f"**Extracted:** {structure.extracted_at.strftime('%Y-%m-%d %H:%M')}",
            "",
            "## Contents",
            ""
        ])

        # Add links to top-level sections
        for child in structure.root.children:
            if child.included:
                link = self.link_manager.create_link(child.title, index_path)
                lines.append(f"- {link}")

        content = "\n".join(lines)

        async with aiofiles.open(index_path, 'w', encoding='utf-8') as f:
            await f.write(content)

        self.generated_files["root"] = index_path
        self.link_manager.register_file(structure.root.title, index_path)

    async def _generate_node_files(
        self,
        node: StructureNode,
        current_dir: Path,
        parent: Optional[StructureNode] = None,
        include_filtered: bool = False
    ) -> None:
        """Recursively generate files for a node and its children."""

        # Skip if not included (unless include_filtered is True)
        if not node.included and not include_filtered:
            return

        # Skip root node (handled by index)
        if node.type == SectionType.BOOK:
            for child in node.children:
                await self._generate_node_files(
                    child, current_dir, node, include_filtered
                )
            return

        # Determine file path
        file_path = self._get_file_path(node, current_dir)

        # Create directory if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Find related sections (for now, just siblings)
        related = []
        if parent:
            related = [
                c for c in parent.children
                if c.id != node.id and c.included
            ][:3]  # Limit to 3 related sections

        # Generate content with file_path for link resolution
        content = self.generate_markdown(node, parent, related, file_path)

        # Write file
        async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
            await f.write(content)

        # Process children - use node's folder as child directory if it has children
        child_dir = file_path.parent if (node.children or node.type in [SectionType.CHAPTER]) else current_dir
        for child in node.children:
            await self._generate_node_files(
                child, child_dir, node, include_filtered
            )

    def get_output_path(self, structure: DocumentStructure) -> Path:
        """Get the output directory path for a document."""
        base_name = self._sanitize_filename(structure.title)
        return self.output_dir / base_name
