# PDF Atomization Project

A FastAPI application that converts PDF documents (especially mathematical textbooks) into structured, interlinked markdown files organized as an Obsidian-compatible vault.

## Project Overview

This application processes PDFs through a multi-stage pipeline:
1. **Page Extraction** - Extract pages from PDF
2. **OCR Processing** - Extract text (via AI vision if no text layer)
3. **Structure Analysis** - Identify document hierarchy (chapters, sections, etc.)
4. **Content Filtering** - Separate knowledge content from meta content
5. **Content Refinement** - Extract actual text for each section
6. **Recursive Atomization** - Split content until each segment is an "atom" (single theorem/definition/lemma)
7. **Content Summarization** - AI-generated structured summaries for atomic units
8. **Markdown Generation** - Output interlinked markdown files

## Key Concepts

### Atomic Content
An "atom" is the smallest meaningful unit of mathematical content:
- Theorem, Definition, Lemma, Corollary, Proposition, Example, Remark

Each atom has structured content:
- **Description** (required) - AI summary of what it represents
- **Statement** (required) - The mathematical statement with LaTeX preserved
- **Proof** (optional) - If present in source
- **Lemmas** (optional) - Supporting lemmas
- **Related Content** (optional) - Related concepts

### Task-Specific AI Providers
The system supports using different AI models for different tasks:
- `structure_extractor` - For analyzing document structure and atomicity
- `content_summarizer` - For generating structured summaries

Configure via environment variables or use the default `AI_PROVIDER`.

## Directory Structure

```
app/
├── config.py              # Settings and AI provider configuration
├── models.py              # Pydantic models (StructureNode, AtomContent, etc.)
├── providers/
│   ├── __init__.py        # get_provider(), get_provider_for_task()
│   ├── base.py            # BaseProvider abstract class
│   ├── openai_provider.py
│   ├── claude_provider.py
│   └── google_provider.py
├── services/
│   ├── pdf_processor.py      # PDF page extraction
│   ├── ocr_service.py        # OCR via AI vision
│   ├── structure_extractor.py # Document structure analysis
│   ├── content_filter.py     # Knowledge vs meta filtering
│   ├── recursive_atomizer.py # Split content into atomic units
│   ├── content_summarizer.py # AI content summarization
│   ├── link_manager.py       # Markdown link management
│   └── markdown_generator.py # File generation
└── routers/
    ├── upload.py          # POST /upload, GET /status/{job_id}
    ├── preview.py         # GET /preview/{job_id}, PUT /preview/{job_id}
    └── export.py          # POST /export/{job_id}, GET /export/{job_id}/download
```

## Key Models

### StructureNode
Represents a node in the document tree with atomization fields:
```python
class StructureNode(BaseModel):
    id: str
    title: str
    type: SectionType  # BOOK, CHAPTER, SECTION, SUBSECTION, CONTENT
    level: int
    content: str
    children: List[StructureNode]

    # Atomization fields
    atomization_status: AtomizationStatus  # PENDING, NEEDS_SPLITTING, ATOMIC, FILLED
    atom_type: Optional[AtomType]  # THEOREM, DEFINITION, LEMMA, etc.
    atom_content: Optional[AtomContent]  # Structured summary
    source_text: Optional[str]  # Original text for summarization
```

### AtomContent
Structured content for atomic units:
```python
class AtomContent(BaseModel):
    description: str      # Required - AI summary
    statement: str        # Required - Mathematical statement
    proof: Optional[str]  # Optional
    lemmas: List[str]     # Optional
    related_content: Optional[str]  # Optional
```

## Link Format

Links use standard markdown format for Obsidian compatibility:
- `[Title](./relative/path.md)` - Resolved links
- `[Title](#)` - Placeholder for unresolved (resolved in post-processing)

## API Endpoints

- `POST /upload` - Upload PDF, returns job_id
- `GET /status/{job_id}` - Check processing status
- `GET /preview/{job_id}` - Get document structure for editing
- `PUT /preview/{job_id}` - Update structure (toggle sections)
- `POST /export/{job_id}` - Generate markdown files
- `GET /export/{job_id}/download` - Download as ZIP

## Configuration

Key environment variables:
```bash
AI_PROVIDER=openai|claude|google
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Task-specific (optional)
STRUCTURE_EXTRACTOR_PROVIDER=openai
STRUCTURE_EXTRACTOR_MODEL=gpt-4o
CONTENT_SUMMARIZER_PROVIDER=claude
CONTENT_SUMMARIZER_MODEL=claude-sonnet-4-20250514

# Atomization
MAX_RECURSION_DEPTH=10
MIN_CONTENT_LENGTH_FOR_SPLIT=500
```

## Development Notes

- All AI calls use the provider abstraction (`get_provider()` or `get_provider_for_task()`)
- LaTeX notation must be preserved exactly in all AI prompts and outputs
- The atomization process is recursive but depth-limited
- Link resolution happens in two passes: registration first, then content generation
- Progress callbacks are used throughout for UI updates
