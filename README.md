# PDF Atomization

A FastAPI application that converts PDF documents (especially mathematical textbooks) into structured, interlinked markdown files organized as an Obsidian-compatible vault.

## 🎯 Overview

PDF Atomization processes PDFs through a multi-stage AI-powered pipeline to:

1. **Extract text** from PDFs (with OCR support for scanned documents)
2. **Analyze structure** to identify chapters, sections, and subsections
3. **Filter content** to separate knowledge from meta content (prefaces, indexes, etc.)
4. **Atomize recursively** until each segment contains a single concept (theorem, definition, lemma, etc.)
5. **Generate summaries** with structured AI-generated content preserving LaTeX notation
6. **Output markdown** files with proper hierarchy and interlinks

## ✨ Features

- **Multi-provider AI support**: OpenAI (GPT-4o), Anthropic (Claude), and Google (Gemini)
- **Task-specific models**: Use different AI models for structure extraction vs content summarization
- **Interactive workflow**: Web UI for reviewing and approving each processing stage
- **OCR support**: Process scanned PDFs using AI vision APIs
- **LaTeX preservation**: All mathematical notation is preserved throughout the pipeline
- **Obsidian-compatible**: Output uses standard markdown links compatible with Obsidian
- **Docker ready**: Easy deployment with Docker Compose

## 🏗️ Architecture

```
PDF Input
    │
    ▼
┌─────────────────┐
│ Page Extraction │  PyMuPDF
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ OCR Processing  │  AI Vision API (if needed)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Structure       │  AI analyzes document hierarchy
│ Extraction      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Content Filter  │  Separate knowledge vs meta
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Recursive       │  Split until atomic
│ Atomization     │  (single theorem/definition/lemma)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Content         │  AI-generated structured summaries
│ Summarization   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Markdown        │  Interlinked files with hierarchy
│ Generation      │
└────────┬────────┘
         │
         ▼
   Obsidian Vault
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- API key for at least one AI provider (OpenAI, Anthropic, or Google)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/pdf_atomization.git
   cd pdf_atomization
   ```

2. **Create virtual environment and install dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Run the application**
   ```bash
   uvicorn app.main:app --reload
   ```

5. **Open the web UI**
   
   Navigate to http://localhost:8000

### Docker Deployment

```bash
# Set your API keys in environment or .env file
docker-compose up -d
```

## ⚙️ Configuration

Create a `.env` file based on `.env.example`:

```bash
# AI Provider: openai, claude, or google
AI_PROVIDER=openai

# API Keys (set the one for your chosen provider)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...

# Optional: Task-specific models
STRUCTURE_EXTRACTOR_PROVIDER=openai
STRUCTURE_EXTRACTOR_MODEL=gpt-4o
CONTENT_SUMMARIZER_PROVIDER=claude
CONTENT_SUMMARIZER_MODEL=claude-sonnet-4-20250514

# Atomization settings
MAX_RECURSION_DEPTH=10
MIN_CONTENT_LENGTH_FOR_SPLIT=500

# Directories
OUTPUT_DIR=./output
UPLOADS_DIR=./uploads
```

## 📖 Usage

### Web Interface

1. **Upload PDF**: Drag and drop or click to upload your PDF
2. **Review OCR** (if applicable): Verify extracted text quality
3. **Approve Structure**: Review and modify the detected document hierarchy
4. **Approve Atomization**: Review AI decisions on what constitutes an "atom"
5. **Approve Content**: Review AI-generated summaries for each atomic unit
6. **Export**: Download as a ZIP or export to a folder

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/upload` | POST | Upload a PDF file |
| `/api/status/{job_id}` | GET | Check processing status |
| `/api/preview/{job_id}` | GET | Get document structure |
| `/api/preview/{job_id}` | PUT | Update structure (toggle sections) |
| `/api/export/{job_id}` | POST | Generate markdown files |
| `/api/export/{job_id}/download` | GET | Download as ZIP |
| `/api/workflow/{job_id}` | GET | Get current workflow stage |
| `/health` | GET | Health check |

## 📁 Project Structure

```
pdf_atomization/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application entry
│   ├── config.py            # Settings and configuration
│   ├── models.py            # Pydantic models
│   ├── providers/           # AI provider implementations
│   │   ├── base.py          # Abstract base provider
│   │   ├── openai_provider.py
│   │   ├── claude_provider.py
│   │   └── google_provider.py
│   ├── services/            # Core processing services
│   │   ├── pdf_processor.py      # PDF page extraction
│   │   ├── ocr_service.py        # OCR via AI vision
│   │   ├── structure_extractor.py # Document structure analysis
│   │   ├── content_filter.py     # Knowledge vs meta filtering
│   │   ├── recursive_atomizer.py # Recursive content splitting
│   │   ├── content_summarizer.py # AI content summarization
│   │   ├── markdown_generator.py # File generation
│   │   └── link_manager.py       # Cross-reference handling
│   └── routers/             # API route handlers
│       ├── upload.py
│       ├── preview.py
│       ├── export.py
│       └── workflow.py
├── static/
│   └── index.html           # Web UI (single-page application)
├── output/                  # Generated markdown files
├── uploads/                 # Uploaded PDF files
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

## 🧠 Key Concepts

### Atomic Content

An "atom" is the smallest meaningful unit of mathematical content:
- **Theorem** - A proven mathematical statement
- **Definition** - A formal definition of a concept
- **Lemma** - A helper theorem used to prove larger theorems
- **Corollary** - A direct consequence of a theorem
- **Proposition** - A statement to be proven
- **Example** - An illustrative example
- **Remark** - An observation or note

### Structured Atom Content

Each atom is summarized with:
- **Description** (required) - AI-generated summary explaining the concept
- **Statement** (required) - The exact mathematical statement with LaTeX preserved
- **Proof** (optional) - The proof if present in the source
- **Lemmas** (optional) - Supporting lemmas referenced
- **Related Content** (optional) - Related concepts mentioned

### Output Format

Generated markdown uses standard links for Obsidian compatibility:
```markdown
# Theorem 1.1: Intermediate Value Theorem

> Parent: [Chapter 1: Continuity](../index.md)
> Children: [Corollary 1.1.1](./Corollary_1.1.1.md)

## Description
The Intermediate Value Theorem states that...

## Theorem
Let $f: [a,b] \to \mathbb{R}$ be continuous...

## Proof
Suppose $f(a) < y < f(b)$...

---
## Related
- [Definition 1.0: Continuous Function](./Definition_1.0.md)
```

## 🛠️ Tech Stack

- **FastAPI** - Modern Python web framework
- **Uvicorn** - ASGI server
- **PyMuPDF** - PDF processing and rendering
- **Pydantic** - Data validation
- **aiofiles** - Async file operations
- **OpenAI/Anthropic/Google APIs** - AI providers
- **Tailwind CSS** - Frontend styling (via CDN)

## 📄 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
