"""Pydantic models for requests and responses."""

from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Status of a processing job."""
    PENDING = "pending"
    EXTRACTING_PAGES = "extracting_pages"
    PROCESSING_OCR = "processing_ocr"
    ANALYZING_STRUCTURE = "analyzing_structure"
    FILTERING_CONTENT = "filtering_content"
    REFINING_CONTENT = "refining_content"
    ATOMIZING = "atomizing"
    FILLING_CONTENT = "filling_content"
    GENERATING_LINKS = "generating_links"
    COMPLETED = "completed"
    FAILED = "failed"


class SectionType(str, Enum):
    """Type of document section."""
    BOOK = "book"
    CHAPTER = "chapter"
    SECTION = "section"
    SUBSECTION = "subsection"
    CONTENT = "content"


class ContentCategory(str, Enum):
    """Category of content for filtering."""
    KNOWLEDGE = "knowledge"
    META = "meta"


class AtomizationStatus(str, Enum):
    """Status of atomization for a node."""
    PENDING = "pending"
    NEEDS_SPLITTING = "needs_splitting"
    ATOMIC = "atomic"
    FILLED = "filled"


class WorkflowStage(str, Enum):
    """Stage of the interactive workflow."""
    UPLOADING = "uploading"
    EXTRACTING = "extracting"
    AWAITING_STRUCTURE_APPROVAL = "awaiting_structure_approval"
    ATOMIZING = "atomizing"
    AWAITING_ATOMIZATION_APPROVAL = "awaiting_atomization_approval"
    FILLING_CONTENT = "filling_content"
    AWAITING_CONTENT_APPROVAL = "awaiting_content_approval"
    COMPLETED = "completed"
    FAILED = "failed"


class NodeApprovalStatus(str, Enum):
    """Approval status for a node in the workflow."""
    PENDING = "pending"
    APPROVED = "approved"
    REGENERATING = "regenerating"


class AtomType(str, Enum):
    """Type of atomic content unit."""
    THEOREM = "theorem"
    DEFINITION = "definition"
    LEMMA = "lemma"
    COROLLARY = "corollary"
    PROPOSITION = "proposition"
    EXAMPLE = "example"
    REMARK = "remark"
    OTHER = "other"


class AtomContent(BaseModel):
    """Structured content for an atomic unit."""
    description: str              # AI-summarized description (required)
    statement: str                # The theorem/definition statement (required)
    proof: Optional[str] = None   # The proof if present
    lemmas: List[str] = Field(default_factory=list)  # Supporting lemmas
    related_content: Optional[str] = None  # Related concepts summary


class StructureNode(BaseModel):
    """A node in the document structure tree."""
    id: str
    title: str
    type: SectionType
    level: int
    content: str = ""
    children: List["StructureNode"] = Field(default_factory=list)
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    category: ContentCategory = ContentCategory.KNOWLEDGE
    included: bool = True

    # Atomization fields
    atomization_status: AtomizationStatus = AtomizationStatus.PENDING
    atom_type: Optional[AtomType] = None
    atom_content: Optional[AtomContent] = None
    source_text: Optional[str] = None  # Preserved source for content filling

    # Workflow approval fields
    approval_status: NodeApprovalStatus = NodeApprovalStatus.PENDING
    ai_atomicity_reason: Optional[str] = None  # Store AI's reasoning
    user_edited: bool = False

    class Config:
        from_attributes = True


# Required for self-referential model
StructureNode.model_rebuild()


class DocumentStructure(BaseModel):
    """Complete document structure."""
    title: str
    author: Optional[str] = None
    root: StructureNode
    total_pages: int
    extracted_at: datetime = Field(default_factory=datetime.now)


class ProcessingJob(BaseModel):
    """A PDF processing job."""
    job_id: str
    filename: str
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    message: str = ""
    structure: Optional[DocumentStructure] = None
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    full_text: Optional[str] = None  # Store for content refinement

    # Workflow fields
    workflow_stage: WorkflowStage = WorkflowStage.UPLOADING
    pending_atomization_nodes: List[str] = Field(default_factory=list)  # Node IDs awaiting decision
    pending_content_nodes: List[str] = Field(default_factory=list)  # Node IDs awaiting content approval


class UploadResponse(BaseModel):
    """Response from PDF upload."""
    job_id: str
    filename: str
    message: str


class StatusResponse(BaseModel):
    """Response for job status check."""
    job_id: str
    status: JobStatus
    progress: float
    message: str
    error: Optional[str] = None


class PreviewResponse(BaseModel):
    """Response containing structure preview."""
    job_id: str
    structure: DocumentStructure
    editable: bool = True


class StructureUpdateRequest(BaseModel):
    """Request to update document structure."""
    structure: DocumentStructure


class ExportRequest(BaseModel):
    """Request to export processed document."""
    job_id: str
    output_format: str = "folder"  # "folder" or "zip"
    include_filtered: bool = False


class ExportResponse(BaseModel):
    """Response from export operation."""
    job_id: str
    output_path: Optional[str] = None
    download_url: Optional[str] = None
    message: str


class OCRResult(BaseModel):
    """Result from OCR processing of a page."""
    page_number: int
    text: str
    confidence: Optional[float] = None
    structure_hints: Optional[Dict[str, Any]] = None


class PageChunk(BaseModel):
    """A chunk of pages for processing."""
    start_page: int
    end_page: int
    total_pages: int


# Models for merge workflow (Phase 2)

class VaultInfo(BaseModel):
    """Information about an existing vault."""
    name: str
    path: str
    created_at: datetime
    document_count: int
    total_sections: int


class SimilarSection(BaseModel):
    """A section similar to new content."""
    section_id: str
    title: str
    similarity_score: float
    file_path: str


class MergeAction(str, Enum):
    """Action to take when merging."""
    CREATE_NEW = "create_new"
    MERGE_INTO = "merge_into"
    LINK_ONLY = "link_only"
    SKIP = "skip"


class MergeDecision(BaseModel):
    """User's decision for merging a section."""
    new_section_id: str
    action: MergeAction
    target_section_id: Optional[str] = None


class MergePreview(BaseModel):
    """Preview of merge operation."""
    new_sections: List[StructureNode]
    similar_sections: Dict[str, List[SimilarSection]]


class MergeRequest(BaseModel):
    """Request to execute merge."""
    decisions: List[MergeDecision]
