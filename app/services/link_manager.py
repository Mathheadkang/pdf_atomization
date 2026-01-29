"""Link management service for creating and tracking interlinks."""

import re
from pathlib import Path
from typing import Dict, List, Set, Optional
from dataclasses import dataclass, field

import aiofiles


@dataclass
class LinkTarget:
    """A target for markdown links."""
    title: str
    file_path: Path
    aliases: Set[str] = field(default_factory=set)


class LinkManager:
    """Service for managing standard markdown links between files."""

    def __init__(self):
        # Map from normalized title to LinkTarget
        self.targets: Dict[str, LinkTarget] = {}
        # Map from file path to title
        self.path_to_title: Dict[Path, str] = {}
        # Track all links created
        self.links_created: List[tuple[str, str]] = []  # (from_title, to_title)
        # Base directory for calculating relative paths
        self.base_dir: Optional[Path] = None

    def _normalize_title(self, title: str) -> str:
        """Normalize a title for matching."""
        # Lowercase, remove extra whitespace
        normalized = title.lower().strip()
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized

    def register_file(
        self,
        title: str,
        file_path: Path,
        aliases: Optional[List[str]] = None
    ) -> None:
        """Register a file as a link target."""
        normalized = self._normalize_title(title)

        target = LinkTarget(
            title=title,
            file_path=file_path,
            aliases=set(aliases) if aliases else set()
        )

        self.targets[normalized] = target
        self.path_to_title[file_path] = title

        # Also register aliases
        if aliases:
            for alias in aliases:
                alias_normalized = self._normalize_title(alias)
                if alias_normalized not in self.targets:
                    self.targets[alias_normalized] = target

    def create_link(
        self,
        title: str,
        from_path: Optional[Path] = None,
        display_text: Optional[str] = None
    ) -> str:
        """Create a standard markdown link to a title.

        Args:
            title: The title to link to
            from_path: The file path this link is being created in (for relative path calculation)
            display_text: Optional display text (defaults to title)

        Returns:
            Markdown link in format [Text](./path.md) or [Text](#) if target not found
        """
        target = self.find_target(title)
        text = display_text or title

        if target and from_path:
            rel_path = self._calculate_relative_path(from_path, target.file_path)
            return f"[{text}]({rel_path})"
        elif target:
            # No from_path, use absolute-ish path from base
            return f"[{text}]({target.file_path.name})"
        else:
            # Target not found, create unresolved link placeholder
            return f"[{text}](#)"

    def _calculate_relative_path(self, from_path: Path, to_path: Path) -> str:
        """Calculate relative path from one file to another.

        Args:
            from_path: The source file path
            to_path: The target file path

        Returns:
            Relative path string (e.g., "../sibling/file.md", "./child/file.md")
        """
        try:
            # Get directories containing the files
            from_dir = from_path.parent
            to_dir = to_path.parent

            # Calculate relative path from from_dir to to_path
            rel_path = Path(to_path).relative_to(from_dir)
            return "./" + str(rel_path)
        except ValueError:
            # Files are in different directory trees, need to go up
            try:
                # Find common ancestor
                from_parts = from_path.parent.parts
                to_parts = to_path.parts

                # Find where paths diverge
                common_length = 0
                for i, (f, t) in enumerate(zip(from_parts, to_parts)):
                    if f == t:
                        common_length = i + 1
                    else:
                        break

                # Calculate number of "../" needed
                ups = len(from_parts) - common_length
                downs = to_parts[common_length:]

                rel_parts = [".."] * ups + list(downs)
                return "/".join(rel_parts)
            except Exception:
                # Fallback to just the filename
                return to_path.name

    def find_target(self, title: str) -> Optional[LinkTarget]:
        """Find a link target by title or alias."""
        normalized = self._normalize_title(title)
        return self.targets.get(normalized)

    def resolve_link(self, link_text: str) -> Optional[Path]:
        """Resolve a markdown link to a file path."""
        # Extract title from standard markdown link [text](#) placeholder
        match = re.match(r'\[([^\]]+)\]\(#\)', link_text)
        if match:
            title = match.group(1)
            target = self.find_target(title)
            return target.file_path if target else None

        # Also support wiki-style links for backwards compatibility
        match = re.match(r'\[\[([^|\]]+)(?:\|[^\]]+)?\]\]', link_text)
        if not match:
            return None

        title = match.group(1)
        target = self.find_target(title)
        return target.file_path if target else None

    def get_relative_path(self, from_path: Path, to_path: Path) -> str:
        """Get a relative path from one file to another."""
        return self._calculate_relative_path(from_path, to_path)

    def extract_links_from_content(self, content: str) -> List[str]:
        """Extract all links from markdown content.

        Supports both standard markdown links and wiki-style links.
        """
        links = []

        # Standard markdown links [text](path)
        md_pattern = r'\[([^\]]+)\]\([^)]+\)'
        links.extend(re.findall(md_pattern, content))

        # Wiki-style links for backwards compatibility
        wiki_pattern = r'\[\[([^|\]]+)(?:\|[^\]]+)?\]\]'
        links.extend(re.findall(wiki_pattern, content))

        return links

    def find_cross_references(
        self,
        content: str,
        exclude_titles: Optional[Set[str]] = None
    ) -> List[str]:
        """Find potential cross-references in content based on registered titles."""
        exclude = exclude_titles or set()
        found_refs = []

        content_lower = content.lower()

        for normalized, target in self.targets.items():
            if target.title in exclude:
                continue

            # Check if title appears in content (case-insensitive)
            if normalized in content_lower:
                found_refs.append(target.title)

        return found_refs

    def add_backlinks_section(
        self,
        content: str,
        backlinks: List[str],
        from_path: Optional[Path] = None
    ) -> str:
        """Add a backlinks section to markdown content."""
        if not backlinks:
            return content

        backlinks_section = "\n\n## Backlinks\n"
        for title in backlinks:
            backlinks_section += f"- {self.create_link(title, from_path)}\n"

        return content + backlinks_section

    def build_link_graph(self) -> Dict[str, List[str]]:
        """Build a graph of all links between files."""
        graph = {title: [] for title in self.targets}

        for from_title, to_title in self.links_created:
            if from_title in graph:
                graph[from_title].append(to_title)

        return graph

    def find_orphans(self) -> List[str]:
        """Find files with no incoming links."""
        graph = self.build_link_graph()

        # Files that are linked to
        linked = set()
        for targets in graph.values():
            linked.update(targets)

        # Files that have no incoming links
        orphans = [
            title for title in self.targets
            if title not in linked
        ]

        return orphans

    def suggest_links(
        self,
        title: str,
        content: str,
        max_suggestions: int = 5
    ) -> List[str]:
        """Suggest potential links based on content similarity."""
        # Find titles mentioned in content
        refs = self.find_cross_references(content, exclude_titles={title})

        # Limit suggestions
        return refs[:max_suggestions]

    def clear(self) -> None:
        """Clear all registered targets and links."""
        self.targets.clear()
        self.path_to_title.clear()
        self.links_created.clear()

    async def resolve_all_links(self, output_dir: Path) -> Dict[str, int]:
        """Post-generation pass to resolve all placeholder links after all files exist.

        This method scans all generated markdown files and replaces placeholder
        links [Text](#) with actual relative paths [Text](./path.md).

        Args:
            output_dir: The output directory containing generated files

        Returns:
            Dict with statistics: {'files_processed': int, 'links_resolved': int, 'links_unresolved': int}
        """
        stats = {
            'files_processed': 0,
            'links_resolved': 0,
            'links_unresolved': 0
        }

        # Find all markdown files
        md_files = list(output_dir.rglob("*.md"))

        for file_path in md_files:
            try:
                async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                    content = await f.read()

                # Find all placeholder links [Text](#)
                pattern = r'\[([^\]]+)\]\(#\)'
                matches = list(re.finditer(pattern, content))

                if not matches:
                    continue

                new_content = content
                for match in reversed(matches):  # Reverse to maintain positions
                    display_text = match.group(1)
                    target = self.find_target(display_text)

                    if target and target.file_path.exists():
                        # Calculate relative path and replace
                        rel_path = self._calculate_relative_path(file_path, target.file_path)
                        replacement = f"[{display_text}]({rel_path})"
                        new_content = (
                            new_content[:match.start()] +
                            replacement +
                            new_content[match.end():]
                        )
                        stats['links_resolved'] += 1
                    else:
                        stats['links_unresolved'] += 1

                # Write updated content if changed
                if new_content != content:
                    async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                        await f.write(new_content)

                stats['files_processed'] += 1

            except Exception as e:
                # Log error but continue processing other files
                pass

        return stats
