"""Workflow control endpoints for interactive approval process."""

import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.models import (
    WorkflowStage,
    NodeApprovalStatus,
    AtomizationStatus,
    AtomContent,
    StructureNode,
)
from app.routers.upload import jobs, get_job
from app.services.recursive_atomizer import RecursiveAtomizer
from app.services.structure_extractor import StructureExtractor
from app.services.content_summarizer import ContentSummarizer

logger = logging.getLogger(__name__)

router = APIRouter()


# Response models
class WorkflowStatusResponse(BaseModel):
    """Response for workflow status."""
    job_id: str
    workflow_stage: WorkflowStage
    message: str
    pending_atomization_count: int = 0
    pending_content_count: int = 0


class AtomizationQueueItem(BaseModel):
    """Item in the atomization approval queue."""
    node_id: str
    title: str
    path: List[str]  # Breadcrumb path
    is_atomic: bool
    atom_type: Optional[str]
    ai_reason: Optional[str]
    content_preview: str
    full_content: str  # Full source_text or content for detailed review
    approval_status: NodeApprovalStatus


class ContentQueueItem(BaseModel):
    """Item in the content approval queue."""
    node_id: str
    title: str
    path: List[str]  # Breadcrumb path
    atom_type: Optional[str]
    source_text_preview: str
    atom_content: Optional[AtomContent]
    approval_status: NodeApprovalStatus


class AtomizationDecisionResponse(BaseModel):
    """Response after an atomization decision."""
    success: bool
    message: str
    new_children: Optional[List[str]] = None  # Node IDs if split


class ContentEditRequest(BaseModel):
    """Request to manually edit content."""
    description: str
    statement: str
    proof: Optional[str] = None
    lemmas: List[str] = []
    related_content: Optional[str] = None


# Helper functions
def find_node_by_id(root: StructureNode, node_id: str) -> Optional[StructureNode]:
    """Find a node by ID in the structure tree."""
    if root.id == node_id:
        return root
    for child in root.children:
        result = find_node_by_id(child, node_id)
        if result:
            return result
    return None


def get_node_path(root: StructureNode, node_id: str, path: List[str] = None) -> List[str]:
    """Get the breadcrumb path to a node."""
    if path is None:
        path = []

    if root.id == node_id:
        return path + [root.title]

    for child in root.children:
        result = get_node_path(child, node_id, path + [root.title])
        if result:
            return result

    return []


def collect_pending_atomization_nodes(node: StructureNode, nodes: List[StructureNode] = None) -> List[StructureNode]:
    """Collect all nodes pending atomization approval."""
    if nodes is None:
        nodes = []

    if not node.included:
        return nodes

    from app.models import SectionType

    # Skip container types - they don't need atomization decisions
    if node.type not in [SectionType.BOOK, SectionType.CHAPTER]:
        # Node needs approval if user hasn't approved it yet
        if node.approval_status == NodeApprovalStatus.PENDING:
            nodes.append(node)

    for child in node.children:
        collect_pending_atomization_nodes(child, nodes)

    return nodes


def collect_pending_content_nodes(node: StructureNode, nodes: List[StructureNode] = None) -> List[StructureNode]:
    """Collect all atomic nodes pending content approval."""
    if nodes is None:
        nodes = []

    if not node.included:
        return nodes

    from app.models import SectionType

    # Skip container types
    if node.type in [SectionType.BOOK, SectionType.CHAPTER]:
        for child in node.children:
            collect_pending_content_nodes(child, nodes)
        return nodes

    # Check if this node has generated content pending approval
    if (node.atom_content is not None and
        node.approval_status == NodeApprovalStatus.PENDING):
        nodes.append(node)

    for child in node.children:
        collect_pending_content_nodes(child, nodes)

    return nodes


# Endpoints
@router.get("/workflow/{job_id}/ocr-text")
async def get_ocr_text(job_id: str):
    """Get the full OCR/extracted text for review."""
    job = get_job(job_id)

    if not job.full_text:
        raise HTTPException(status_code=400, detail="No extracted text available yet")

    return {
        "job_id": job_id,
        "text": job.full_text,
        "length": len(job.full_text),
    }


@router.get("/workflow/{job_id}/atomization/{node_id}/content")
async def get_node_full_content(job_id: str, node_id: str):
    """Get the full content of a node for review during atomization."""
    job = get_job(job_id)

    if not job.structure:
        raise HTTPException(status_code=400, detail="No structure available")

    node = find_node_by_id(job.structure.root, node_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")

    source_text = node.source_text or ""
    content = node.content or ""

    return {
        "node_id": node_id,
        "title": node.title,
        "source_text": source_text,
        "content": content,
        "source_text_length": len(source_text),
        "content_length": len(content),
    }


@router.get("/workflow/{job_id}", response_model=WorkflowStatusResponse)
async def get_workflow_status(job_id: str) -> WorkflowStatusResponse:
    """Get current workflow stage and status."""
    job = get_job(job_id)

    logger.info(f"Workflow status request for job {job_id}: stage={job.workflow_stage}, status={job.status}")

    pending_atomization = len(job.pending_atomization_nodes)
    pending_content = len(job.pending_content_nodes)

    messages = {
        WorkflowStage.UPLOADING: "Uploading PDF...",
        WorkflowStage.EXTRACTING: "Extracting content from PDF...",
        WorkflowStage.AWAITING_STRUCTURE_APPROVAL: "Structure extracted. Please review and approve.",
        WorkflowStage.ATOMIZING: "Analyzing content for atomicity...",
        WorkflowStage.AWAITING_ATOMIZATION_APPROVAL: f"Please review {pending_atomization} nodes for atomicity decisions.",
        WorkflowStage.FILLING_CONTENT: "Generating content summaries...",
        WorkflowStage.AWAITING_CONTENT_APPROVAL: f"Please review {pending_content} content summaries.",
        WorkflowStage.COMPLETED: "Workflow complete. Ready for export.",
        WorkflowStage.FAILED: f"Processing failed: {job.error or 'Unknown error'}",
    }

    return WorkflowStatusResponse(
        job_id=job_id,
        workflow_stage=job.workflow_stage,
        message=messages.get(job.workflow_stage, "Processing..."),
        pending_atomization_count=pending_atomization,
        pending_content_count=pending_content,
    )


@router.post("/workflow/{job_id}/approve-structure")
async def approve_structure(job_id: str):
    """Approve structure and start atomization analysis."""
    job = get_job(job_id)

    if job.workflow_stage != WorkflowStage.AWAITING_STRUCTURE_APPROVAL:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot approve structure in stage: {job.workflow_stage}"
        )

    if not job.structure:
        raise HTTPException(status_code=400, detail="No structure available")

    # Transition to atomizing stage
    job.workflow_stage = WorkflowStage.ATOMIZING
    job.message = "Populating section content from full text..."

    # Populate source_text on all nodes from full_text before atomization
    if job.full_text:
        extractor = StructureExtractor()
        extractor.populate_source_text_from_full_text(job.structure.root, job.full_text)
        logger.info("Populated source_text on all nodes from full_text")

    job.message = "Analyzing content for atomicity..."

    # Run atomization analysis on all included nodes
    atomizer = RecursiveAtomizer()

    async def analyze_node_atomicity(node: StructureNode):
        """Analyze atomicity for a single node without recursing."""
        if not node.included:
            return

        # Skip container types
        from app.models import SectionType
        if node.type in [SectionType.BOOK, SectionType.CHAPTER]:
            for child in node.children:
                await analyze_node_atomicity(child)
            return

        # Get content
        content = node.source_text or node.content
        if not content or len(content) < atomizer.min_content_length:
            node.atomization_status = AtomizationStatus.ATOMIC
            node.ai_atomicity_reason = "Content too short to split further."
            node.approval_status = NodeApprovalStatus.PENDING
        else:
            # Check atomicity
            result = await atomizer.check_single_node_atomicity(node)
            node.atomization_status = AtomizationStatus.ATOMIC if result["is_atomic"] else AtomizationStatus.NEEDS_SPLITTING
            node.atom_type = result.get("atom_type")
            node.ai_atomicity_reason = result.get("reason", "No reason provided.")
            node.approval_status = NodeApprovalStatus.PENDING

        # Process children
        for child in node.children:
            await analyze_node_atomicity(child)

    await analyze_node_atomicity(job.structure.root)

    # Collect pending nodes
    pending_nodes = collect_pending_atomization_nodes(job.structure.root)
    job.pending_atomization_nodes = [n.id for n in pending_nodes]

    # Transition to approval stage
    job.workflow_stage = WorkflowStage.AWAITING_ATOMIZATION_APPROVAL
    job.message = f"Please review {len(pending_nodes)} nodes for atomicity decisions."

    return {
        "success": True,
        "message": f"Structure approved. {len(pending_nodes)} nodes pending atomicity review.",
        "pending_count": len(pending_nodes)
    }


@router.get("/workflow/{job_id}/atomization-queue", response_model=List[AtomizationQueueItem])
async def get_atomization_queue(job_id: str) -> List[AtomizationQueueItem]:
    """Get nodes pending atomicity decisions."""
    job = get_job(job_id)

    if not job.structure:
        return []

    pending_nodes = collect_pending_atomization_nodes(job.structure.root)

    items = []
    for node in pending_nodes:
        content = node.source_text or node.content or ""
        items.append(AtomizationQueueItem(
            node_id=node.id,
            title=node.title,
            path=get_node_path(job.structure.root, node.id),
            is_atomic=node.atomization_status == AtomizationStatus.ATOMIC,
            atom_type=node.atom_type.value if node.atom_type else None,
            ai_reason=node.ai_atomicity_reason,
            content_preview=content[:500] + ("..." if len(content) > 500 else ""),
            full_content=content,
            approval_status=node.approval_status,
        ))

    return items


@router.post("/workflow/{job_id}/atomization/{node_id}/approve")
async def approve_atomization(job_id: str, node_id: str):
    """Approve AI's atomicity decision for a node."""
    job = get_job(job_id)

    if not job.structure:
        raise HTTPException(status_code=400, detail="No structure available")

    node = find_node_by_id(job.structure.root, node_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")

    node.approval_status = NodeApprovalStatus.APPROVED

    # Sync pending list from live data
    remaining = collect_pending_atomization_nodes(job.structure.root)
    job.pending_atomization_nodes = [n.id for n in remaining]

    return {
        "success": True,
        "message": f"Approved atomicity decision for: {node.title}",
        "remaining_count": len(remaining)
    }


@router.post("/workflow/{job_id}/atomization/{node_id}/regenerate")
async def regenerate_atomization(job_id: str, node_id: str):
    """Re-run atomicity check for a node."""
    job = get_job(job_id)

    if not job.structure:
        raise HTTPException(status_code=400, detail="No structure available")

    node = find_node_by_id(job.structure.root, node_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")

    node.approval_status = NodeApprovalStatus.REGENERATING

    # Re-run atomicity check
    atomizer = RecursiveAtomizer()
    content = node.source_text or node.content or ""

    result = await atomizer.check_single_node_atomicity(node)
    node.atomization_status = AtomizationStatus.ATOMIC if result["is_atomic"] else AtomizationStatus.NEEDS_SPLITTING
    node.atom_type = result.get("atom_type")
    node.ai_atomicity_reason = result.get("reason", "No reason provided.")
    node.approval_status = NodeApprovalStatus.PENDING

    return {
        "success": True,
        "message": f"Regenerated atomicity analysis for: {node.title}",
        "is_atomic": result["is_atomic"],
        "atom_type": result.get("atom_type"),
        "reason": result.get("reason")
    }


@router.post("/workflow/{job_id}/atomization/{node_id}/split", response_model=AtomizationDecisionResponse)
async def split_node(job_id: str, node_id: str) -> AtomizationDecisionResponse:
    """Split a node into sub-structure (disagree with atomic decision)."""
    job = get_job(job_id)

    if not job.structure:
        raise HTTPException(status_code=400, detail="No structure available")

    node = find_node_by_id(job.structure.root, node_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")

    # Split the node
    atomizer = RecursiveAtomizer()
    new_children = await atomizer.split_single_node(node)

    if not new_children:
        return AtomizationDecisionResponse(
            success=False,
            message="Could not split node further. The content may already be atomic.",
        )

    # Add new children to the node
    node.children.extend(new_children)
    node.atomization_status = AtomizationStatus.NEEDS_SPLITTING
    node.approval_status = NodeApprovalStatus.APPROVED  # User made the decision

    # Run atomicity analysis on new children
    for child in new_children:
        content = child.source_text or child.content or ""
        if len(content) < atomizer.min_content_length:
            child.atomization_status = AtomizationStatus.ATOMIC
            child.ai_atomicity_reason = "Content too short to split further."
        else:
            result = await atomizer.check_single_node_atomicity(child)
            child.atomization_status = AtomizationStatus.ATOMIC if result["is_atomic"] else AtomizationStatus.NEEDS_SPLITTING
            child.atom_type = result.get("atom_type")
            child.ai_atomicity_reason = result.get("reason", "No reason provided.")
        child.approval_status = NodeApprovalStatus.PENDING

    # Sync pending list from live data
    new_child_ids = [child.id for child in new_children]
    remaining = collect_pending_atomization_nodes(job.structure.root)
    job.pending_atomization_nodes = [n.id for n in remaining]

    return AtomizationDecisionResponse(
        success=True,
        message=f"Split into {len(new_children)} children. Please review the new nodes.",
        new_children=new_child_ids,
    )


@router.post("/workflow/{job_id}/approve-all-atomization")
async def approve_all_atomization(job_id: str):
    """Approve all remaining atomization nodes.

    Nodes marked NEEDS_SPLITTING are auto-split first, then all results approved.
    """
    job = get_job(job_id)

    if not job.structure:
        raise HTTPException(status_code=400, detail="No structure available")

    atomizer = RecursiveAtomizer()

    # Iteratively process: split NEEDS_SPLITTING nodes, then approve all
    max_iterations = 5  # Prevent infinite loops
    for iteration in range(max_iterations):
        pending = collect_pending_atomization_nodes(job.structure.root)
        if not pending:
            break

        needs_split = [n for n in pending if n.atomization_status == AtomizationStatus.NEEDS_SPLITTING and not n.children]
        atomic = [n for n in pending if n not in needs_split]

        # Approve atomic nodes
        for node in atomic:
            node.approval_status = NodeApprovalStatus.APPROVED

        # Split nodes that need splitting
        if not needs_split:
            break

        for node in needs_split:
            new_children = await atomizer.split_single_node(node)
            if new_children:
                node.children.extend(new_children)
                node.atomization_status = AtomizationStatus.NEEDS_SPLITTING
                node.approval_status = NodeApprovalStatus.APPROVED

                # Analyze new children
                for child in new_children:
                    content = child.source_text or child.content or ""
                    if len(content) < atomizer.min_content_length:
                        child.atomization_status = AtomizationStatus.ATOMIC
                        child.ai_atomicity_reason = "Content too short to split further."
                    else:
                        result = await atomizer.check_single_node_atomicity(child)
                        child.atomization_status = AtomizationStatus.ATOMIC if result["is_atomic"] else AtomizationStatus.NEEDS_SPLITTING
                        child.atom_type = result.get("atom_type")
                        child.ai_atomicity_reason = result.get("reason", "No reason provided.")
                    child.approval_status = NodeApprovalStatus.PENDING
            else:
                # Can't split further, approve as-is
                node.approval_status = NodeApprovalStatus.APPROVED

    # Final pass: approve anything still pending
    remaining = collect_pending_atomization_nodes(job.structure.root)
    for node in remaining:
        node.approval_status = NodeApprovalStatus.APPROVED

    job.pending_atomization_nodes = []

    return {
        "success": True,
        "message": "All atomization nodes processed and approved.",
        "remaining_count": 0
    }


@router.post("/workflow/{job_id}/proceed-to-content")
async def proceed_to_content(job_id: str):
    """Move from atomization to content summarization stage."""
    job = get_job(job_id)

    if job.workflow_stage != WorkflowStage.AWAITING_ATOMIZATION_APPROVAL:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot proceed in stage: {job.workflow_stage}"
        )

    if not job.structure:
        raise HTTPException(status_code=400, detail="No structure available")

    # Check if all atomization decisions are approved (use live collection, not stale list)
    pending_nodes = collect_pending_atomization_nodes(job.structure.root)
    if pending_nodes:
        # Sync the job's pending list
        job.pending_atomization_nodes = [n.id for n in pending_nodes]
        raise HTTPException(
            status_code=400,
            detail=f"Cannot proceed: {len(pending_nodes)} nodes still pending atomization review"
        )

    # Transition to content filling stage
    job.workflow_stage = WorkflowStage.FILLING_CONTENT
    job.message = "Generating content summaries..."

    # Collect all leaf nodes that need content (any approved node without children,
    # regardless of atomization status - user approved them as final units)
    summarizer = ContentSummarizer()

    def collect_content_nodes(node: StructureNode, nodes: List[StructureNode] = None) -> List[StructureNode]:
        if nodes is None:
            nodes = []
        if not node.included:
            return nodes
        from app.models import SectionType
        if node.type in [SectionType.BOOK, SectionType.CHAPTER]:
            for child in node.children:
                collect_content_nodes(child, nodes)
            return nodes

        # Collect included children
        included_children = [c for c in node.children if c.included]

        if included_children:
            # Has included children - recurse into them
            for child in node.children:
                collect_content_nodes(child, nodes)
        else:
            # Leaf node (no included children) - needs content
            nodes.append(node)

        return nodes

    atomic_nodes = collect_content_nodes(job.structure.root)
    logger.info(f"Collected {len(atomic_nodes)} leaf nodes for content generation")

    # Generate summaries
    for node in atomic_nodes:
        atom_content = await summarizer.summarize_single_node(node)
        node.atom_content = atom_content
        node.atomization_status = AtomizationStatus.FILLED
        node.approval_status = NodeApprovalStatus.PENDING

    # Collect pending content nodes
    pending_nodes = collect_pending_content_nodes(job.structure.root)
    job.pending_content_nodes = [n.id for n in pending_nodes]

    # Transition to content approval
    job.workflow_stage = WorkflowStage.AWAITING_CONTENT_APPROVAL
    job.message = f"Please review {len(pending_nodes)} content summaries."

    return {
        "success": True,
        "message": f"Content generated. {len(pending_nodes)} summaries pending review.",
        "pending_count": len(pending_nodes)
    }


@router.get("/workflow/{job_id}/content-queue", response_model=List[ContentQueueItem])
async def get_content_queue(job_id: str) -> List[ContentQueueItem]:
    """Get atomic nodes pending content approval."""
    job = get_job(job_id)

    if not job.structure:
        return []

    pending_nodes = collect_pending_content_nodes(job.structure.root)

    items = []
    for node in pending_nodes:
        source = node.source_text or node.content or ""
        items.append(ContentQueueItem(
            node_id=node.id,
            title=node.title,
            path=get_node_path(job.structure.root, node.id),
            atom_type=node.atom_type.value if node.atom_type else None,
            source_text_preview=source[:500] + ("..." if len(source) > 500 else ""),
            atom_content=node.atom_content,
            approval_status=node.approval_status,
        ))

    return items


@router.post("/workflow/{job_id}/content/{node_id}/approve")
async def approve_content(job_id: str, node_id: str):
    """Approve AI-generated content for a node."""
    job = get_job(job_id)

    if not job.structure:
        raise HTTPException(status_code=400, detail="No structure available")

    node = find_node_by_id(job.structure.root, node_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")

    node.approval_status = NodeApprovalStatus.APPROVED

    # Remove from pending list
    if node_id in job.pending_content_nodes:
        job.pending_content_nodes.remove(node_id)

    return {
        "success": True,
        "message": f"Approved content for: {node.title}",
        "remaining_count": len(job.pending_content_nodes)
    }


@router.post("/workflow/{job_id}/content/{node_id}/regenerate")
async def regenerate_content(job_id: str, node_id: str):
    """Re-generate content summary for a node."""
    job = get_job(job_id)

    if not job.structure:
        raise HTTPException(status_code=400, detail="No structure available")

    node = find_node_by_id(job.structure.root, node_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")

    node.approval_status = NodeApprovalStatus.REGENERATING

    # Re-generate content
    summarizer = ContentSummarizer()
    atom_content = await summarizer.summarize_single_node(node)
    node.atom_content = atom_content
    node.approval_status = NodeApprovalStatus.PENDING

    return {
        "success": True,
        "message": f"Regenerated content for: {node.title}",
        "atom_content": atom_content.model_dump() if atom_content else None
    }


@router.post("/workflow/{job_id}/content/{node_id}/edit")
async def edit_content(job_id: str, node_id: str, edit_request: ContentEditRequest):
    """Save manually edited content for a node."""
    job = get_job(job_id)

    if not job.structure:
        raise HTTPException(status_code=400, detail="No structure available")

    node = find_node_by_id(job.structure.root, node_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")

    # Update content with user edits
    node.atom_content = AtomContent(
        description=edit_request.description,
        statement=edit_request.statement,
        proof=edit_request.proof,
        lemmas=edit_request.lemmas,
        related_content=edit_request.related_content,
    )
    node.user_edited = True
    node.approval_status = NodeApprovalStatus.APPROVED

    # Remove from pending list
    if node_id in job.pending_content_nodes:
        job.pending_content_nodes.remove(node_id)

    return {
        "success": True,
        "message": f"Saved manual edits for: {node.title}",
        "remaining_count": len(job.pending_content_nodes)
    }


@router.post("/workflow/{job_id}/approve-all-content")
async def approve_all_content(job_id: str):
    """Approve all remaining content nodes."""
    job = get_job(job_id)

    if not job.structure:
        raise HTTPException(status_code=400, detail="No structure available")

    pending = collect_pending_content_nodes(job.structure.root)
    for node in pending:
        node.approval_status = NodeApprovalStatus.APPROVED

    job.pending_content_nodes = []

    return {
        "success": True,
        "message": f"Approved {len(pending)} content summaries.",
        "approved_count": len(pending)
    }


@router.post("/workflow/{job_id}/complete")
async def complete_workflow(job_id: str):
    """Finalize workflow and prepare for export."""
    job = get_job(job_id)

    if job.workflow_stage != WorkflowStage.AWAITING_CONTENT_APPROVAL:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot complete in stage: {job.workflow_stage}"
        )

    # Check if all content is approved
    if job.pending_content_nodes:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot complete: {len(job.pending_content_nodes)} nodes still pending content review"
        )

    # Transition to completed
    job.workflow_stage = WorkflowStage.COMPLETED
    job.status = "completed"
    job.message = "Workflow complete. Ready for export."
    job.progress = 1.0

    return {
        "success": True,
        "message": "Workflow complete! You can now export the document."
    }
