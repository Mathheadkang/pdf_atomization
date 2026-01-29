"""Preview and structure editing endpoints."""

from fastapi import APIRouter, HTTPException

from app.models import (
    PreviewResponse,
    StructureUpdateRequest,
    JobStatus,
    DocumentStructure
)
from app.routers.upload import get_job, update_job_structure

router = APIRouter()


@router.get("/preview/{job_id}", response_model=PreviewResponse)
async def get_preview(job_id: str) -> PreviewResponse:
    """Get the proposed structure for a processed PDF."""

    job = get_job(job_id)

    # Check if processing is complete
    if job.status == JobStatus.FAILED:
        raise HTTPException(
            status_code=400,
            detail=f"Job failed: {job.error}"
        )

    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job not ready. Current status: {job.status.value}"
        )

    if not job.structure:
        raise HTTPException(
            status_code=500,
            detail="Structure not available"
        )

    return PreviewResponse(
        job_id=job_id,
        structure=job.structure,
        editable=True
    )


@router.post("/preview/{job_id}", response_model=PreviewResponse)
async def update_preview(
    job_id: str,
    update: StructureUpdateRequest
) -> PreviewResponse:
    """Update the document structure (user edits)."""

    job = get_job(job_id)

    # Check if processing is complete
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot edit. Job status: {job.status.value}"
        )

    # Update the structure
    update_job_structure(job_id, update.structure)

    return PreviewResponse(
        job_id=job_id,
        structure=update.structure,
        editable=True
    )


@router.post("/preview/{job_id}/toggle/{section_id}")
async def toggle_section_inclusion(
    job_id: str,
    section_id: str,
    included: bool = True
) -> dict:
    """Toggle whether a section is included in export."""

    job = get_job(job_id)

    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot edit. Job status: {job.status.value}"
        )

    if not job.structure:
        raise HTTPException(
            status_code=500,
            detail="Structure not available"
        )

    # Find and update the section
    def update_node(node):
        if node.id == section_id:
            node.included = included
            return True
        for child in node.children:
            if update_node(child):
                return True
        return False

    found = update_node(job.structure.root)

    if not found:
        raise HTTPException(
            status_code=404,
            detail=f"Section {section_id} not found"
        )

    return {
        "success": True,
        "section_id": section_id,
        "included": included
    }


@router.get("/preview/{job_id}/stats")
async def get_structure_stats(job_id: str) -> dict:
    """Get statistics about the document structure."""

    job = get_job(job_id)

    if job.status != JobStatus.COMPLETED or not job.structure:
        raise HTTPException(
            status_code=400,
            detail="Structure not available"
        )

    structure = job.structure

    # Count sections
    total_sections = 0
    included_sections = 0
    filtered_sections = 0

    def count_nodes(node):
        nonlocal total_sections, included_sections, filtered_sections
        total_sections += 1
        if node.included:
            included_sections += 1
        else:
            filtered_sections += 1
        for child in node.children:
            count_nodes(child)

    for child in structure.root.children:
        count_nodes(child)

    return {
        "title": structure.title,
        "author": structure.author,
        "total_pages": structure.total_pages,
        "total_sections": total_sections,
        "included_sections": included_sections,
        "filtered_sections": filtered_sections
    }
