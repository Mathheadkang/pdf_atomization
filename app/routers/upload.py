"""PDF upload endpoint."""

import uuid
import asyncio
import logging
from pathlib import Path
from typing import Dict

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
import aiofiles

logger = logging.getLogger(__name__)

from app.config import settings
from app.models import (
    UploadResponse,
    StatusResponse,
    ProcessingJob,
    JobStatus,
    DocumentStructure,
    WorkflowStage,
)
from app.services.pdf_processor import PDFProcessor
from app.services.ocr_service import OCRService
from app.services.structure_extractor import StructureExtractor
from app.services.content_filter import ContentFilter

router = APIRouter()

# In-memory job storage (in production, use Redis or database)
jobs: Dict[str, ProcessingJob] = {}


async def process_pdf(job_id: str, pdf_path: Path) -> None:
    """Background task to process a PDF."""
    job = jobs[job_id]

    try:
        # Initialize services
        pdf_processor = PDFProcessor(max_pages_per_chunk=settings.max_pages_per_chunk)
        ocr_service = OCRService()
        structure_extractor = StructureExtractor()
        content_filter = ContentFilter()

        # Step 1: Extract pages
        job.status = JobStatus.EXTRACTING_PAGES
        job.workflow_stage = WorkflowStage.EXTRACTING
        job.message = "Extracting pages from PDF..."
        job.progress = 0.1

        total_pages = pdf_processor.get_page_count(pdf_path)
        has_text = pdf_processor.has_text_layer(pdf_path)
        metadata = pdf_processor.get_metadata(pdf_path)

        # Step 2: OCR or extract text
        job.status = JobStatus.PROCESSING_OCR
        job.message = "Processing pages..."
        job.progress = 0.2

        if has_text:
            # Use existing text layer
            full_text = pdf_processor.extract_all_text(pdf_path)
        else:
            # Need to OCR each page - use parallel extraction for speed
            job.message = f"Extracting {total_pages} pages in parallel..."
            
            # Extract all pages in parallel with AI-optimized settings (lower res, JPEG)
            pages_to_process = await pdf_processor.extract_pages_parallel(
                pdf_path,
                start_page=0,
                end_page=total_pages,
                for_ai=True,  # Use lower resolution and JPEG for faster processing
                max_concurrent=settings.max_concurrent_ocr
            )
            
            job.progress = 0.5
            job.message = f"Running OCR on {total_pages} pages..."

            # Process OCR in batches
            async def update_progress(page_num: int):
                job.progress = 0.5 + (0.2 * (page_num + 1) / total_pages)
                job.message = f"OCR processing page {page_num + 1}/{total_pages}..."

            ocr_results = await ocr_service.process_pages_batch(
                pages_to_process,
                progress_callback=update_progress
            )

            full_text = ocr_service.combine_ocr_results(ocr_results)

        # Step 3: Extract structure (with chunked processing for large documents)
        job.status = JobStatus.ANALYZING_STRUCTURE
        job.message = "Analyzing document structure..."
        job.progress = 0.70

        async def structure_progress_callback(msg: str, pct: float):
            """Update job progress during structure extraction."""
            job.message = msg
            job.progress = 0.70 + (pct * 0.15)  # Scale to 0.70-0.85 range

        structure = await structure_extractor.extract_structure(
            full_text,
            title_hint=metadata.get("title"),
            author_hint=metadata.get("author"),
            progress_callback=structure_progress_callback
        )

        # Step 4: Filter content
        job.status = JobStatus.FILTERING_CONTENT
        job.message = "Filtering content sections..."
        job.progress = 0.85

        structure = content_filter.filter_structure(structure)

        # Step 5: Refine content - extract actual text for each section
        job.status = JobStatus.REFINING_CONTENT
        job.message = "Extracting content for each section..."
        job.progress = 0.85

        structure = await structure_extractor.refine_structure(structure, full_text)

        # Store full_text for potential later use
        job.full_text = full_text
        job.structure = structure

        # PAUSE: Set workflow stage and stop - wait for user approval
        job.workflow_stage = WorkflowStage.AWAITING_STRUCTURE_APPROVAL
        job.status = JobStatus.COMPLETED  # Ready for user interaction
        job.progress = 0.35
        job.message = "Structure extracted. Please review and approve."

        logger.info(f"Job {job_id} completed initial processing. workflow_stage={job.workflow_stage}, status={job.status}")

        # Don't continue - wait for user to call /approve-structure endpoint

        # The following steps are now handled by the workflow router:
        # - Step 6: Recursive atomization (triggered by /approve-structure)
        # - Step 7: AI content filling (triggered by /proceed-to-content)
        # - Final completion (triggered by /complete)

    except Exception as e:
        job.status = JobStatus.FAILED
        job.error = str(e)
        job.message = f"Processing failed: {str(e)}"
        raise


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
) -> UploadResponse:
    """Upload a PDF for processing."""

    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted"
        )

    # Generate job ID
    job_id = str(uuid.uuid4())

    # Save uploaded file
    pdf_path = settings.uploads_dir / f"{job_id}.pdf"

    async with aiofiles.open(pdf_path, 'wb') as f:
        content = await file.read()
        await f.write(content)

    # Create job record
    job = ProcessingJob(
        job_id=job_id,
        filename=file.filename,
        status=JobStatus.PENDING,
        message="Upload received, processing will begin shortly..."
    )
    jobs[job_id] = job

    # Start background processing
    background_tasks.add_task(process_pdf, job_id, pdf_path)

    return UploadResponse(
        job_id=job_id,
        filename=file.filename,
        message="PDF uploaded successfully. Processing started."
    )


@router.get("/status/{job_id}", response_model=StatusResponse)
async def get_status(job_id: str) -> StatusResponse:
    """Get the status of a processing job."""

    if job_id not in jobs:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found"
        )

    job = jobs[job_id]

    logger.debug(f"Status check for job {job_id}: status={job.status}, workflow_stage={job.workflow_stage}, progress={job.progress}")

    return StatusResponse(
        job_id=job_id,
        status=job.status,
        progress=job.progress,
        message=job.message,
        error=job.error
    )


def get_job(job_id: str) -> ProcessingJob:
    """Get a job by ID (used by other routers)."""
    if job_id not in jobs:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found"
        )
    return jobs[job_id]


def update_job_structure(job_id: str, structure: DocumentStructure) -> None:
    """Update the structure for a job (used by preview router)."""
    if job_id not in jobs:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found"
        )
    jobs[job_id].structure = structure
